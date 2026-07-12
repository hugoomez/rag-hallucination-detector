"""Offline unit tests for the Track B span-level Detector.

The hermetic tests never download the ModernBERT token-classification checkpoint: a fake
tokenizer + fake model are injected into the Detector (mirroring the constructor-injection
pattern in src/models/nli_baseline.py and src/rag/retriever.py). The fake tokenizer returns
a preset paired encoding (input_ids/attention_mask/offset_mapping + sequence_ids()), and the
fake model returns preset per-token logits, so span reconstruction and scoring can be
asserted exactly.

One real-model integration test (test_real_model_*) is gated behind RUN_INTEGRATION=1 and
skipped by default, matching the download-free convention in test_retriever.py /
test_nli_baseline.py.
"""

import os

import pytest
import torch

from src.models.predict import Detector, color_from_score

GREEN, YELLOW, RED = "🟢", "🟡", "🔴"


# --- Hermetic fakes -------------------------------------------------------------------


class _FakeEncoding(dict):
    """dict-like encoding that also exposes sequence_ids(), like a real BatchEncoding."""

    def __init__(self, data, seq_ids):
        super().__init__(data)
        self._seq_ids = seq_ids

    def sequence_ids(self):
        return self._seq_ids


class _FakeTokenizer:
    """Stub tokenizer: records call kwargs and returns a preset paired encoding."""

    def __init__(self, encoding):
        self.encoding = encoding
        self.calls = []

    def __call__(self, context, response, **kwargs):
        self.calls.append({"context": context, "response": response, **kwargs})
        return self.encoding


class _FakeOutput:
    def __init__(self, logits):
        self.logits = logits


class _FakeModel:
    """Stub token-classification model returning preset (1, seq, 2) logits."""

    def __init__(self, token_logits):
        self.token_logits = token_logits

    def to(self, _device):
        return self

    def eval(self):
        return self

    def __call__(self, **_inputs):
        return _FakeOutput(torch.tensor([self.token_logits], dtype=torch.float))


def _build_detector(tokens, red_threshold=0.5, yellow_threshold=0.45):
    """Build a Detector over a token spec.

    tokens: list of (seq_id, (char_start, char_end), is_hallucinated). A positive token
    gets logits [0.0, 6.0] (P(hallucinated) ~= 0.998); a supported token gets [6.0, 0.0]
    (~0.002). Context/special tokens have seq_id != 1 and are ignored by span/score logic.
    """
    input_ids = list(range(len(tokens)))
    attention_mask = [1] * len(tokens)
    offsets = [offset for _seq_id, offset, _pos in tokens]
    seq_ids = [seq_id for seq_id, _offset, _pos in tokens]
    logits = [[0.0, 6.0] if pos else [6.0, 0.0] for _seq_id, _offset, pos in tokens]

    encoding = _FakeEncoding(
        {"input_ids": input_ids, "attention_mask": attention_mask, "offset_mapping": offsets},
        seq_ids,
    )
    return Detector(
        _FakeModel(logits),
        _FakeTokenizer(encoding),
        red_threshold=red_threshold,
        yellow_threshold=yellow_threshold,
    )


# --- 1. color_from_score (pure, no model) ---------------------------------------------


@pytest.mark.parametrize(
    "score, expected",
    [
        (0.85, RED),
        (0.50, RED),  # >= red_threshold, the model's own decision boundary
        (0.49, YELLOW),
        (0.45, YELLOW),  # >= yellow_threshold
        (0.44, GREEN),
        (0.00, GREEN),
    ],
)
def test_color_from_score_default_thresholds(score, expected):
    assert color_from_score(score) == expected


def test_color_from_score_thresholds_are_overridable():
    # Move both thresholds up: 0.6 is now below red and below yellow -> green.
    assert color_from_score(0.6, red_threshold=0.8, yellow_threshold=0.7) == GREEN
    assert color_from_score(0.75, red_threshold=0.8, yellow_threshold=0.7) == YELLOW
    assert color_from_score(0.85, red_threshold=0.8, yellow_threshold=0.7) == RED


def test_color_yellow_collapses_when_yellow_equals_red():
    # yellow_threshold == red_threshold -> pure green/red, no yellow band.
    assert color_from_score(0.49, red_threshold=0.5, yellow_threshold=0.5) == GREEN
    assert color_from_score(0.5, red_threshold=0.5, yellow_threshold=0.5) == RED


# --- 2. predict: hallucinated example (merged multi-token span) ------------------------


def test_predict_flags_hallucinated_span_and_slices_exact_substring():
    # response = "Paris is the capital of Spain." with "of Spain" (chars 21..29) hallucinated
    # across two consecutive positive tokens ("of" 21-23, "Spain" 24-29): merge_predicted_spans
    # runs first-token-start -> last-token-end, absorbing the space at index 23.
    response = "Paris is the capital of Spain."
    tokens = [
        (None, (0, 0), False),  # [CLS]
        (0, (0, 6), False),  # context token
        (None, (0, 0), False),  # [SEP]
        (1, (0, 5), False),  # "Paris"
        (1, (5, 8), False),  # " is"
        (1, (8, 12), False),  # " the"
        (1, (12, 20), False),  # " capital"
        (1, (21, 23), True),  # "of"    (hallucinated)
        (1, (24, 29), True),  # " Spain"(hallucinated)
        (1, (29, 30), False),  # "."
        (None, (0, 0), False),  # [SEP]
    ]
    detector = _build_detector(tokens)

    result = detector.predict("France info.", response)

    assert result["color"] == RED
    assert result["score"] == pytest.approx(0.9975, abs=1e-3)
    assert result["spans"] == [{"start": 21, "end": 29, "text": "of Spain"}]
    assert response[21:29] == "of Spain"


def test_predict_returns_multiple_spans_split_by_supported_tokens():
    # response = "Cats fly and dogs swim." with "fly" (5..8) and "swim" (18..22)
    # hallucinated but separated by supported tokens -> two distinct spans.
    response = "Cats fly and dogs swim."
    tokens = [
        (None, (0, 0), False),  # [CLS]
        (0, (0, 3), False),  # context
        (None, (0, 0), False),  # [SEP]
        (1, (0, 4), False),  # "Cats"
        (1, (5, 8), True),  # "fly"  (hallucinated)
        (1, (9, 12), False),  # "and"
        (1, (13, 17), False),  # "dogs"
        (1, (18, 22), True),  # "swim" (hallucinated)
        (1, (22, 23), False),  # "."
        (None, (0, 0), False),  # [SEP]
    ]
    detector = _build_detector(tokens)

    result = detector.predict("ctx", response)

    assert result["color"] == RED
    assert [span["text"] for span in result["spans"]] == ["fly", "swim"]
    assert result["spans"] == [
        {"start": 5, "end": 8, "text": "fly"},
        {"start": 18, "end": 22, "text": "swim"},
    ]


# --- 3. predict: faithful example ------------------------------------------------------


def test_predict_faithful_response_has_no_spans_and_is_green():
    response = "Paris is the capital of France."
    tokens = [
        (None, (0, 0), False),  # [CLS]
        (0, (0, 6), False),  # context
        (None, (0, 0), False),  # [SEP]
        (1, (0, 5), False),  # "Paris"
        (1, (5, 8), False),  # " is"
        (1, (8, 12), False),  # " the"
        (1, (12, 20), False),  # " capital"
        (1, (21, 23), False),  # " of"
        (1, (24, 30), False),  # " France"
        (1, (30, 31), False),  # "."
        (None, (0, 0), False),  # [SEP]
    ]
    detector = _build_detector(tokens)

    result = detector.predict("France info.", response)

    assert result["spans"] == []
    assert result["color"] == GREEN
    assert result["score"] == pytest.approx(0.0025, abs=1e-3)


# --- 4. predict: empty response --------------------------------------------------------


def test_predict_empty_response_scores_zero_and_is_green():
    # No response tokens (seq_id == 1); only context + specials.
    tokens = [
        (None, (0, 0), False),  # [CLS]
        (0, (0, 6), False),  # context
        (None, (0, 0), False),  # [SEP]
        (None, (0, 0), False),  # [SEP]
    ]
    detector = _build_detector(tokens)

    result = detector.predict("France info.", "")

    assert result["spans"] == []
    assert result["color"] == GREEN
    assert result["score"] == 0.0


# --- 5. tokenizer contract ------------------------------------------------------------


def test_predict_tokenizes_context_response_pair_like_preprocess_token_level():
    tokens = [(None, (0, 0), False), (1, (0, 3), False), (None, (0, 0), False)]
    detector = _build_detector(tokens)

    detector.predict("some context", "abc")

    call = detector.tokenizer.calls[-1]
    assert call["context"] == "some context"
    assert call["response"] == "abc"
    assert call["truncation"] == "only_first"
    assert call["return_offsets_mapping"] is True
    assert call["max_length"] == 4096


# --- 6. real-model integration (gated) ------------------------------------------------


@pytest.mark.skipif(
    os.environ.get("RUN_INTEGRATION") != "1",
    reason="downloads the real ModernBERT checkpoint; set RUN_INTEGRATION=1 to run",
)
def test_real_model_flags_obvious_hallucination_and_slices_real_substrings():
    detector = Detector.from_pretrained(device="cpu")

    context = (
        "Marie Curie was a physicist and chemist who conducted pioneering research on "
        "radioactivity. She was born in Warsaw in 1867 and won two Nobel Prizes."
    )
    faithful = "Marie Curie was a physicist and chemist who researched radioactivity."
    hallucinated = "Marie Curie was born in Paris in 1901 and personally invented the telephone."

    faithful_result = detector.predict(context, faithful)
    hallucinated_result = detector.predict(context, hallucinated)

    print(
        f"\n[faithful]     score={faithful_result['score']:.4f} color={faithful_result['color']} "
        f"spans={faithful_result['spans']}"
    )
    print(f"[hallucinated] score={hallucinated_result['score']:.4f} color={hallucinated_result['color']}")
    for span in hallucinated_result["spans"]:
        print(f"    span [{span['start']:>3},{span['end']:>3}) -> {span['text']!r}")

    # The obvious hallucination must produce at least one span, flagged red.
    assert hallucinated_result["spans"], hallucinated_result
    assert hallucinated_result["color"] == RED

    # Every span's text must be exactly the response substring at its offsets (no stripping).
    for span in hallucinated_result["spans"]:
        assert hallucinated[span["start"] : span["end"]] == span["text"]

    # The hallucinated response should score strictly higher than the faithful one.
    assert hallucinated_result["score"] > faithful_result["score"]
