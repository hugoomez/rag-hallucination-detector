"""Offline unit tests for the Phase 6 FastAPI wrapper (Step 6.1).

Hermetic by the same convention as test_predict.py / test_pipeline.py: a fake detector
is injected via FastAPI's dependency override, and TestClient(app) is built WITHOUT the
`with` context manager so lifespan never runs -- the real ModernBERT checkpoint is never
downloaded. The fake duck-types the only Detector surface the API uses, .predict(context,
response), returning a preset verdict dict and recording every call for assertion.

One real end-to-end test (test_real_api_*) is gated behind RUN_INTEGRATION=1 and skipped
by default, matching test_predict.py / test_pipeline.py.
"""

import os

import pytest
from fastapi.testclient import TestClient

from api.main import app, get_detector

GREEN, RED = "🟢", "🔴"


# --- Hermetic fake --------------------------------------------------------------------


class _FakeDetector:
    """Stub detector: returns a preset verdict dict and records (context, response)."""

    def __init__(self, verdict):
        self.verdict = verdict
        self.calls = []

    def predict(self, context, response):
        self.calls.append({"context": context, "response": response})
        return self.verdict


@pytest.fixture(autouse=True)
def _clear_overrides():
    """Ensure dependency overrides never leak between tests."""
    yield
    app.dependency_overrides.clear()


def _client(detector):
    # No `with` -> lifespan (and thus Detector.from_pretrained) never runs.
    app.dependency_overrides[get_detector] = lambda: detector
    return TestClient(app)


_RED_VERDICT = {
    "score": 0.9975,
    "color": RED,
    "spans": [{"start": 21, "end": 29, "text": "of Spain"}],
}
_GREEN_VERDICT = {"score": 0.0, "color": GREEN, "spans": []}


# --- 1. /detect happy path ------------------------------------------------------------


def test_detect_returns_verdict_and_passes_context_response_to_detector():
    detector = _FakeDetector(_RED_VERDICT)
    client = _client(detector)

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
    client = _client(_FakeDetector(_GREEN_VERDICT))

    resp = client.post("/detect", json={"context": "ctx", "response": "faithful answer"})

    assert resp.status_code == 200
    assert resp.json() == _GREEN_VERDICT


# --- 2. /detect request validation ----------------------------------------------------


@pytest.mark.parametrize("payload", [{"response": "r"}, {"context": "c"}, {}])
def test_detect_missing_fields_returns_422_and_detector_never_runs(payload):
    detector = _FakeDetector(_GREEN_VERDICT)
    client = _client(detector)

    resp = client.post("/detect", json=payload)

    assert resp.status_code == 422
    assert detector.calls == []


# --- 3. /detect when the model failed to load -----------------------------------------


def test_detect_returns_503_when_model_not_loaded():
    client = _client(None)

    resp = client.post("/detect", json={"context": "c", "response": "r"})

    assert resp.status_code == 503


# --- 4. /health -----------------------------------------------------------------------


def test_health_reports_model_loaded_true_when_detector_present():
    client = _client(_FakeDetector(_GREEN_VERDICT))

    resp = client.get("/health")

    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "model_loaded": True}


def test_health_reports_model_loaded_false_when_detector_missing():
    client = _client(None)

    resp = client.get("/health")

    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "model_loaded": False}


# --- 5. real end-to-end (gated) -------------------------------------------------------


@pytest.mark.skipif(
    os.environ.get("RUN_INTEGRATION") != "1",
    reason="downloads the real ModernBERT checkpoint; set RUN_INTEGRATION=1 to run",
)
def test_real_api_flags_obvious_hallucination():
    from src.models.predict import Detector

    with TestClient(app) as client:  # `with` -> lifespan loads the real Detector
        assert isinstance(app.state.detector, Detector)

        health = client.get("/health").json()
        assert health == {"status": "ok", "model_loaded": True}

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
