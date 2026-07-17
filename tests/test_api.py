"""Offline unit tests for the Phase 6 FastAPI app (Steps 6.1 + 6.3).

Hermetic by the same convention as test_predict.py / test_pipeline.py: fakes are injected
via FastAPI's dependency overrides, and TestClient(app) is built WITHOUT the `with` context
manager so lifespan never runs -- the real ModernBERT checkpoint is never downloaded and no
Groq key is needed. The fakes duck-type the only surfaces the API uses, Detector.predict()
and RAGPipeline.answer(), returning preset results and recording every call for assertion.

/presets is tested against an in-memory cache rather than the checked-in demo_cache.json, so
these tests don't depend on that file existing (mirroring tests/test_app.py's convention).

One real end-to-end test (test_real_api_*) is gated behind RUN_INTEGRATION=1 and skipped by
default, matching test_predict.py / test_pipeline.py.
"""

import os

import pytest
from fastapi.testclient import TestClient

from api.main import app, get_cache, get_detector, get_pipeline, get_presets
from app.presets import Preset
from src.rag.pipeline import GenerationError

GREEN, RED = "🟢", "🔴"


# --- Hermetic fakes -------------------------------------------------------------------


class _FakeDetector:
    """Stub detector: returns a preset verdict dict and records (context, response)."""

    def __init__(self, verdict):
        self.verdict = verdict
        self.calls = []

    def predict(self, context, response):
        self.calls.append({"context": context, "response": response})
        return self.verdict


class _FakePipeline:
    """Stub pipeline: returns a preset result (or raises), and records every answer() call.

    `result` may be an Exception instance, in which case it is raised -- that's how the
    GenerationError paths are exercised without touching Groq.
    """

    def __init__(self, result):
        self.result = result
        self.calls = []

    def answer(self, question, no_context=False):
        self.calls.append({"question": question, "no_context": no_context})
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


@pytest.fixture(autouse=True)
def _clear_overrides():
    """Ensure dependency overrides never leak between tests."""
    yield
    app.dependency_overrides.clear()


def _client(detector=None, pipeline=None, cache=None, presets=None):
    # No `with` -> lifespan (and thus Detector.from_pretrained) never runs.
    app.dependency_overrides[get_detector] = lambda: detector
    app.dependency_overrides[get_pipeline] = lambda: pipeline
    app.dependency_overrides[get_cache] = lambda: cache if cache is not None else {}
    if presets is not None:
        app.dependency_overrides[get_presets] = lambda: presets
    return TestClient(app)


_RED_VERDICT = {
    "score": 0.9975,
    "color": RED,
    "spans": [{"start": 21, "end": 29, "text": "of Spain"}],
}
_GREEN_VERDICT = {"score": 0.0, "color": GREEN, "spans": []}

_RAG_RESULT = {
    "question": "What did Gauss contribute to number theory?",
    "contexts": [{"source": "gauss.txt", "text": "Gauss worked on number theory.", "score": 0.51}],
    "answer": "Gauss contributed to number theory.",
    "verdict": _GREEN_VERDICT,
}


# --- 1. /detect happy path ------------------------------------------------------------


def test_detect_returns_verdict_and_passes_context_response_to_detector():
    detector = _FakeDetector(_RED_VERDICT)
    client = _client(detector=detector)

    resp = client.post(
        "/detect",
        json={"context": "France info.", "response": "Paris is the capital of Spain."},
    )

    assert resp.status_code == 200
    assert resp.json() == _RED_VERDICT  # emoji color + span shape round-trip exactly
    # No per-span "label" key leaks in (the token-level model doesn't produce one).
    assert set(resp.json()["spans"][0]) == {"start", "end", "text"}
    assert detector.calls == [{"context": "France info.", "response": "Paris is the capital of Spain."}]


def test_detect_green_no_span_response():
    client = _client(detector=_FakeDetector(_GREEN_VERDICT))

    resp = client.post("/detect", json={"context": "ctx", "response": "faithful answer"})

    assert resp.status_code == 200
    assert resp.json() == _GREEN_VERDICT


# --- 2. /detect request validation ----------------------------------------------------


@pytest.mark.parametrize("payload", [{"response": "r"}, {"context": "c"}, {}])
def test_detect_missing_fields_returns_422_and_detector_never_runs(payload):
    detector = _FakeDetector(_GREEN_VERDICT)
    client = _client(detector=detector)

    resp = client.post("/detect", json=payload)

    assert resp.status_code == 422
    assert detector.calls == []


# --- 3. /detect when the model failed to load -----------------------------------------


def test_detect_returns_503_when_model_not_loaded():
    client = _client(detector=None)

    resp = client.post("/detect", json={"context": "c", "response": "r"})

    assert resp.status_code == 503


# --- 4. /ask happy paths --------------------------------------------------------------


def test_ask_returns_rag_result_and_defaults_ablation_false():
    pipeline = _FakePipeline(_RAG_RESULT)
    client = _client(pipeline=pipeline)

    resp = client.post("/ask", json={"question": "What did Gauss contribute to number theory?"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "Gauss contributed to number theory."
    assert body["contexts"] == [{"source": "gauss.txt", "text": "Gauss worked on number theory.", "score": 0.51}]
    assert body["verdict"] == _GREEN_VERDICT
    # RAGPipeline.answer() omits "ablation" entirely unless no_context=True; the response
    # model must still carry it so the frontend never has to guess.
    assert body["ablation"] is False
    assert pipeline.calls == [{"question": "What did Gauss contribute to number theory?", "no_context": False}]


def test_ask_no_context_reaches_pipeline_and_reports_ablation():
    ablated = {**_RAG_RESULT, "verdict": _RED_VERDICT, "ablation": True}
    pipeline = _FakePipeline(ablated)
    client = _client(pipeline=pipeline)

    resp = client.post("/ask", json={"question": "When did von Neumann marry?", "no_context": True})

    assert resp.status_code == 200
    assert resp.json()["ablation"] is True
    # The flag must actually reach the pipeline -- that's what makes it a real ablation.
    assert pipeline.calls == [{"question": "When did von Neumann marry?", "no_context": True}]


# --- 5. /ask validation ---------------------------------------------------------------


@pytest.mark.parametrize("payload", [{}, {"question": ""}, {"no_context": True}])
def test_ask_invalid_question_returns_422_and_pipeline_never_runs(payload):
    pipeline = _FakePipeline(_RAG_RESULT)
    client = _client(pipeline=pipeline)

    resp = client.post("/ask", json=payload)

    assert resp.status_code == 422
    assert pipeline.calls == []


# --- 6. /ask failure modes ------------------------------------------------------------


def test_ask_returns_503_with_no_key_kind_when_pipeline_unavailable():
    client = _client(pipeline=None)

    resp = client.post("/ask", json={"question": "anything"})

    assert resp.status_code == 503
    detail = resp.json()["detail"]
    assert detail["kind"] == "no_key"
    assert "GROQ_API_KEY" in detail["message"]


def test_ask_rate_limit_returns_429_and_surfaces_message():
    message = "Groq rate limit hit (free tier is ~30 requests/min); wait a minute and retry."
    client = _client(pipeline=_FakePipeline(GenerationError(message, kind="rate_limit")))

    resp = client.post("/ask", json={"question": "anything"})

    assert resp.status_code == 429
    assert resp.json()["detail"] == {"kind": "rate_limit", "message": message}


@pytest.mark.parametrize("kind", ["connection", "api_error", "empty_completion"])
def test_ask_other_generation_errors_return_502_with_their_kind(kind):
    client = _client(pipeline=_FakePipeline(GenerationError("it broke", kind=kind)))

    resp = client.post("/ask", json={"question": "anything"})

    assert resp.status_code == 502
    assert resp.json()["detail"] == {"kind": kind, "message": "it broke"}


def test_ask_never_leaks_a_traceback():
    client = _client(pipeline=_FakePipeline(GenerationError("Groq API call failed: 401", kind="api_error")))

    resp = client.post("/ask", json={"question": "anything"})

    assert "Traceback" not in resp.text
    assert "src/rag/pipeline.py" not in resp.text


# --- 7. /presets ----------------------------------------------------------------------

_PRESET_DEFS = [
    Preset(id="a", label="Grounded", question="Q one?", no_context=False),
    Preset(id="b", label="Ablation", question="Q two?", no_context=True),
]


def test_presets_returns_cached_results_without_detector_or_pipeline():
    cache = {"Q one?": _RAG_RESULT, "Q two?": {**_RAG_RESULT, "ablation": True}}
    # Explicitly no detector and no pipeline: the cached path must stand on its own.
    client = _client(detector=None, pipeline=None, cache=cache, presets=_PRESET_DEFS)

    resp = client.get("/presets")

    assert resp.status_code == 200
    presets = resp.json()["presets"]
    assert [p["id"] for p in presets] == ["a", "b"]
    assert [p["label"] for p in presets] == ["Grounded", "Ablation"]
    assert presets[0]["result"]["ablation"] is False
    assert presets[1]["result"]["ablation"] is True
    assert presets[0]["result"]["verdict"] == _GREEN_VERDICT


def test_presets_returns_empty_list_when_cache_is_empty():
    client = _client(cache={}, presets=_PRESET_DEFS)

    resp = client.get("/presets")

    assert resp.status_code == 200
    assert resp.json() == {"presets": []}


def test_presets_omits_questions_missing_from_the_cache():
    client = _client(cache={"Q two?": _RAG_RESULT}, presets=_PRESET_DEFS)

    resp = client.get("/presets")

    assert [p["id"] for p in resp.json()["presets"]] == ["b"]


def test_presets_uses_the_real_preset_definitions_by_default():
    from app.presets import PRESETS

    # No presets override -> the endpoint reads app/presets.py, the shared source of truth.
    client = _client(cache={p.question: _RAG_RESULT for p in PRESETS})

    resp = client.get("/presets")

    assert [p["id"] for p in resp.json()["presets"]] == [p.id for p in PRESETS]


# --- 8. /health -----------------------------------------------------------------------


def test_health_reports_both_model_and_pipeline_loaded():
    client = _client(detector=_FakeDetector(_GREEN_VERDICT), pipeline=_FakePipeline(_RAG_RESULT))

    resp = client.get("/health")

    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "model_loaded": True, "pipeline_loaded": True}


def test_health_reports_model_loaded_but_pipeline_missing_without_a_key():
    client = _client(detector=_FakeDetector(_GREEN_VERDICT), pipeline=None)

    resp = client.get("/health")

    assert resp.json() == {"status": "ok", "model_loaded": True, "pipeline_loaded": False}


def test_health_reports_nothing_loaded():
    client = _client(detector=None, pipeline=None)

    resp = client.get("/health")

    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "model_loaded": False, "pipeline_loaded": False}


# --- 9. frontend mount ordering -------------------------------------------------------


def test_frontend_is_served_at_root():
    resp = _client().get("/")

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    assert "RAG Hallucination Detector" in resp.text


@pytest.mark.parametrize("path", ["/health", "/presets", "/docs", "/openapi.json"])
def test_static_mount_does_not_shadow_api_routes(path):
    """The mount is a catch-all at "/", so route order is load-bearing (ADR-019).

    Starlette matches in registration order; if the mount ever moves above the API routes
    in api/main.py, these GETs start returning 404s from StaticFiles instead.
    """
    resp = _client().get(path)

    assert resp.status_code == 200


def test_static_mount_still_404s_unknown_paths():
    assert _client().get("/no-such-page").status_code == 404


def test_frontend_assets_are_served():
    css = _client().get("/styles.css")

    assert css.status_code == 200
    assert "--signal-red" in css.text


# --- 10. real end-to-end (gated) ------------------------------------------------------


@pytest.mark.skipif(
    os.environ.get("RUN_INTEGRATION") != "1",
    reason="downloads the real ModernBERT checkpoint; set RUN_INTEGRATION=1 to run",
)
def test_real_api_flags_obvious_hallucination():
    from src.models.predict import Detector

    with TestClient(app) as client:  # `with` -> lifespan loads the real Detector
        assert isinstance(app.state.detector, Detector)

        health = client.get("/health").json()
        assert health["status"] == "ok"
        assert health["model_loaded"] is True

        context = (
            "Marie Curie was a physicist and chemist who conducted pioneering research on "
            "radioactivity. She was born in Warsaw in 1867 and won two Nobel Prizes."
        )
        response = "Marie Curie was born in Paris in 1901 and personally invented the telephone."
        body = client.post("/detect", json={"context": context, "response": response}).json()

    assert body["color"] == RED
    assert body["spans"]
    for span in body["spans"]:
        assert response[span["start"] : span["end"]] == span["text"]
