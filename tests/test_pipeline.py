"""Offline unit tests for the Phase 5 RAG pipeline (retrieve -> generate -> detect).

Hermetic by the same convention as test_predict.py / test_retriever.py: a fake retriever,
fake detector, and fake Groq client are injected into RAGPipeline, so no test ever touches
the network, reads .env, or downloads a model. The fake Groq client duck-types just the
`client.chat.completions.create(...)` call path the pipeline uses, records every call's
kwargs for assertion, and can be armed with an exception to exercise the error-translation
paths.

One real end-to-end test (test_real_pipeline_*) is gated behind RUN_INTEGRATION=1 plus a
GROQ_API_KEY and skipped by default.
"""

import os
from types import SimpleNamespace

import groq
import httpx
import pytest

from src.rag.pipeline import (
    DEFAULT_GROQ_MODEL,
    RAG_PROMPT,
    GenerationError,
    RAGPipeline,
    build_context,
)
from src.rag.retriever import Chunk, RetrievalResult

# --- Hermetic fakes -------------------------------------------------------------------


class _FakeRetriever:
    """Stub retriever: returns preset RetrievalResults and records (query, k)."""

    def __init__(self, results):
        self.results = results
        self.calls = []

    def retrieve(self, query, k):
        self.calls.append({"query": query, "k": k})
        return self.results


class _FakeDetector:
    """Stub detector: returns a preset verdict dict and records (context, response)."""

    def __init__(self, verdict):
        self.verdict = verdict
        self.calls = []

    def predict(self, context, response):
        self.calls.append({"context": context, "response": response})
        return self.verdict


class _FakeCompletions:
    """Stub of client.chat.completions: records create() kwargs, then returns a canned
    completion whose message.content is `outcome` -- or raises it if it's an exception."""

    def __init__(self, outcome):
        self.outcome = outcome
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self.outcome))])


class _FakeGroqClient:
    """Duck-types the only Groq surface the pipeline touches: .chat.completions.create."""

    def __init__(self, outcome):
        self.chat = SimpleNamespace(completions=_FakeCompletions(outcome))


_RESULTS = [
    RetrievalResult(
        chunk=Chunk(text="Gauss proved quadratic reciprocity.", source="gauss.txt", chunk_id=0), score=0.91
    ),
    RetrievalResult(chunk=Chunk(text="Riemann studied under Gauss.", source="riemann.txt", chunk_id=3), score=0.72),
]
_VERDICT = {"score": 0.02, "color": "\U0001f7e2", "spans": []}


def _build_pipeline(outcome="Gauss proved it in 1796.", results=_RESULTS, verdict=_VERDICT, **kwargs):
    retriever = _FakeRetriever(results)
    detector = _FakeDetector(verdict)
    client = _FakeGroqClient(outcome)
    return RAGPipeline(retriever, detector, client, **kwargs), retriever, detector, client


# --- 1. build_context (pure) ----------------------------------------------------------


def test_build_context_labels_sources_and_separates_chunks():
    context = build_context(_RESULTS)

    assert context == (
        "[source: gauss.txt]\nGauss proved quadratic reciprocity."
        "\n\n---\n\n"
        "[source: riemann.txt]\nRiemann studied under Gauss."
    )


# --- 2. answer: the full retrieve -> generate -> detect chain --------------------------


def test_answer_chains_retrieve_generate_detect():
    pipeline, retriever, detector, _client = _build_pipeline()

    result = pipeline.answer("Who proved quadratic reciprocity?")

    # Exact output contract for the Phase 6 demo UI.
    assert set(result) == {"question", "contexts", "answer", "verdict"}
    assert result["question"] == "Who proved quadratic reciprocity?"
    assert result["answer"] == "Gauss proved it in 1796."
    assert result["verdict"] is _VERDICT  # detector's dict passed through untouched
    assert result["contexts"] == [
        {"source": "gauss.txt", "text": "Gauss proved quadratic reciprocity.", "score": 0.91},
        {"source": "riemann.txt", "text": "Riemann studied under Gauss.", "score": 0.72},
    ]

    # The detector judged the generated answer against the SAME context string the
    # generator saw (both chunk texts present), not some other rendering of the chunks.
    detect_call = detector.calls[-1]
    assert detect_call["response"] == "Gauss proved it in 1796."
    assert detect_call["context"] == build_context(_RESULTS)

    assert retriever.calls == [{"query": "Who proved quadratic reciprocity?", "k": 3}]


def test_answer_passes_constructor_k_to_retriever():
    pipeline, retriever, _detector, _client = _build_pipeline(k=2)

    pipeline.answer("any question")

    assert retriever.calls[-1]["k"] == 2


# --- 3. generate: prompt construction and Groq call parameters -------------------------


def test_generate_builds_prompt_and_params():
    pipeline, _retriever, _detector, client = _build_pipeline()

    answer = pipeline.generate("Who was Gauss?", "some context text")

    assert answer == "Gauss proved it in 1796."
    call = client.chat.completions.calls[-1]
    assert call["model"] == DEFAULT_GROQ_MODEL
    assert call["messages"] == [
        {"role": "user", "content": RAG_PROMPT.format(context="some context text", question="Who was Gauss?")}
    ]
    assert "some context text" in call["messages"][0]["content"]
    assert "Who was Gauss?" in call["messages"][0]["content"]
    # gpt-oss reasoning knobs: cheap/fast reasoning, and no CoT in the payload at all.
    assert call["reasoning_effort"] == "low"
    assert call["include_reasoning"] is False


def test_generate_uses_model_name_override():
    pipeline, _retriever, _detector, client = _build_pipeline(model_name="openai/gpt-oss-120b")

    pipeline.generate("q", "ctx")

    call = client.chat.completions.calls[-1]
    assert call["model"] == "openai/gpt-oss-120b"
    assert call["reasoning_effort"] == "low"  # 120b is still gpt-oss: knobs apply


def test_generate_uses_reasoning_format_hidden_for_qwen_family():
    # Groq rejects reasoning_effort="low"/include_reasoning on non-gpt-oss models
    # (verified live: qwen returns 400 "`reasoning_effort` must be one of `none` or
    # `default`"), and without reasoning_format="hidden" qwen leaks its <think> CoT
    # into message.content (also verified live) -- which would burn the token budget
    # and feed CoT to the detector. The swap-by-model_name contract needs both handled.
    pipeline, _retriever, _detector, client = _build_pipeline(model_name="qwen/qwen3.6-27b")

    pipeline.generate("q", "ctx")

    call = client.chat.completions.calls[-1]
    assert call["model"] == "qwen/qwen3.6-27b"
    assert call["reasoning_format"] == "hidden"
    assert "reasoning_effort" not in call
    assert "include_reasoning" not in call


def test_generate_sends_no_reasoning_params_for_unknown_model_families():
    pipeline, _retriever, _detector, client = _build_pipeline(model_name="meta-llama/llama-4-scout-17b-16e-instruct")

    pipeline.generate("q", "ctx")

    call = client.chat.completions.calls[-1]
    assert "reasoning_effort" not in call
    assert "include_reasoning" not in call
    assert "reasoning_format" not in call


def test_generate_strips_answer_whitespace():
    pipeline, _retriever, _detector, _client = _build_pipeline(outcome="  padded answer \n")

    assert pipeline.generate("q", "ctx") == "padded answer"


# --- 4. generate: error translation ----------------------------------------------------


def _rate_limit_error():
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    return groq.RateLimitError("rate limited", response=httpx.Response(429, request=request), body=None)


def test_generate_translates_rate_limit_into_actionable_message():
    pipeline, _retriever, _detector, client = _build_pipeline(outcome=_rate_limit_error())
    client.api_key = "gsk_FAKE_KEY_SHOULD_NEVER_LEAK"  # sentinel: must not surface in errors

    with pytest.raises(GenerationError) as exc_info:
        pipeline.generate("q", "ctx")

    assert "rate limit" in str(exc_info.value).lower()
    assert isinstance(exc_info.value.__cause__, groq.RateLimitError)
    assert "gsk_FAKE_KEY_SHOULD_NEVER_LEAK" not in str(exc_info.value)


def test_generate_translates_connection_error():
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    error = groq.APIConnectionError(request=request)
    pipeline, _retriever, _detector, _client = _build_pipeline(outcome=error)

    with pytest.raises(GenerationError) as exc_info:
        pipeline.generate("q", "ctx")

    assert "network" in str(exc_info.value).lower() or "reach" in str(exc_info.value).lower()
    assert exc_info.value.__cause__ is error


def test_generate_wraps_any_other_groq_error():
    error = groq.GroqError("model decommissioned")
    pipeline, _retriever, _detector, _client = _build_pipeline(outcome=error)

    with pytest.raises(GenerationError) as exc_info:
        pipeline.generate("q", "ctx")

    assert "model decommissioned" in str(exc_info.value)
    assert exc_info.value.__cause__ is error


def test_generate_rejects_none_content_and_detector_never_runs():
    pipeline, _retriever, detector, _client = _build_pipeline(outcome=None)

    with pytest.raises(GenerationError):
        pipeline.answer("q")

    assert detector.calls == []  # failure happened before detection


def test_generate_rejects_blank_content():
    # Reasoning models can exhaust max_completion_tokens on hidden CoT and return ""
    # with finish_reason="length" (verified live with qwen at a 512 cap): that must be
    # a loud GenerationError, not a silently-detected empty answer.
    pipeline, _retriever, detector, _client = _build_pipeline(outcome="  \n")

    with pytest.raises(GenerationError) as exc_info:
        pipeline.answer("q")

    assert "empty" in str(exc_info.value).lower()
    assert detector.calls == []


# --- 5. real end-to-end (gated) ---------------------------------------------------------


@pytest.mark.skipif(
    os.environ.get("RUN_INTEGRATION") != "1",
    reason="calls the Groq API and downloads models; set RUN_INTEGRATION=1 to run",
)
def test_real_pipeline_answers_in_corpus_question():
    from src.models.predict import Detector
    from src.rag.pipeline import create_groq_client
    from src.rag.retriever import CORPUS_DIR, Retriever

    retriever = Retriever()
    retriever.build(CORPUS_DIR)
    pipeline = RAGPipeline(retriever, Detector.from_pretrained(device="cpu"), create_groq_client())

    result = pipeline.answer("What did Gauss contribute to number theory?")

    print(f"\nanswer: {result['answer']}\nverdict: {result['verdict']}")
    assert set(result) == {"question", "contexts", "answer", "verdict"}
    assert result["answer"].strip()
    assert len(result["contexts"]) == 3
    assert {"score", "color", "spans"} <= set(result["verdict"])
