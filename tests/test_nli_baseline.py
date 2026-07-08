"""Offline unit tests for the zero-shot NLI hallucination detector.

These tests never download the ~750MB NLI model and never require nltk's punkt data:
a fake model + fake tokenizer are injected into the detector, and ``split_sentences`` is
monkeypatched to a trivial splitter.
"""

import pytest
import torch

from src.models import nli_baseline
from src.models.nli_baseline import (
    DetectionResult,
    NLIHallucinationDetector,
    SentenceVerdict,
    apply_thresholds,
    flag_from_scores,
    response_color,
)

# Standard checkpoint label order, kept separate so tests can deliberately shuffle it.
STANDARD_ID2LABEL = {0: "entailment", 1: "neutral", 2: "contradiction"}


class _FakeConfig:
    def __init__(self, id2label):
        self.id2label = id2label


class _FakeOutput:
    def __init__(self, logits):
        self.logits = logits


class _FakeTokenizer:
    """Stub tokenizer: records call kwargs and returns an all-tensor encoding.

    Like a real HF tokenizer, every value in the returned encoding is a tensor (so the
    detector's ``.to(device)`` loop works unchanged). The premises are recorded on the
    tokenizer so the paired fake model can look up per-premise logits.
    """

    def __init__(self):
        self.calls = []

    def __call__(self, premises, hypotheses, **kwargs):
        self.calls.append({"premises": premises, "hypotheses": hypotheses, **kwargs})
        batch = len(premises)
        return {
            "input_ids": torch.zeros((batch, 1), dtype=torch.long),
            "attention_mask": torch.ones((batch, 1), dtype=torch.long),
        }


class _FakeModel:
    """Stub NLI model: returns preset logits keyed by premise text.

    ``logits_by_premise`` maps a premise string to a length-3 logit list laid out in the
    same column order as ``id2label``. The premises for the current batch are read from
    the paired tokenizer's most recent recorded call.
    """

    def __init__(self, id2label, tokenizer, logits_by_premise=None, default_logits=None):
        self.config = _FakeConfig(id2label)
        self.tokenizer = tokenizer
        self.logits_by_premise = logits_by_premise or {}
        self.default_logits = default_logits or [0.0, 0.0, 0.0]

    def to(self, _device):
        return self

    def eval(self):
        return self

    def __call__(self, **_encoded):
        premises = self.tokenizer.calls[-1]["premises"]
        rows = [self.logits_by_premise.get(p, self.default_logits) for p in premises]
        return _FakeOutput(torch.tensor(rows, dtype=torch.float))


def _build_detector(id2label=None, logits_by_premise=None, default_logits=None):
    id2label = id2label or STANDARD_ID2LABEL
    tokenizer = _FakeTokenizer()
    model = _FakeModel(id2label, tokenizer, logits_by_premise=logits_by_premise, default_logits=default_logits)
    return NLIHallucinationDetector(model, tokenizer)


# --- 1. flag_from_scores (pure, no model) ---------------------------------------------


def test_flag_supported_when_only_entailment_meets_threshold():
    assert flag_from_scores(0.9, 0.1, ent_thr=0.5, con_thr=0.5) == "supported"


def test_flag_contradicted_when_only_contradiction_meets_threshold():
    assert flag_from_scores(0.1, 0.8, ent_thr=0.5, con_thr=0.5) == "contradicted"


def test_flag_contradiction_wins_when_both_thresholds_met():
    # Per ADR-007: contradicted takes priority over supported.
    assert flag_from_scores(0.9, 0.9, ent_thr=0.5, con_thr=0.5) == "contradicted"


def test_flag_unverifiable_when_neither_threshold_met():
    assert flag_from_scores(0.2, 0.3, ent_thr=0.5, con_thr=0.5) == "unverifiable"


# --- 2. Dynamic id2label mapping ------------------------------------------------------


def test_entailment_read_by_label_name_not_position():
    # Shuffled config: entailment is the LAST column, not the first.
    shuffled = {0: "contradiction", 1: "neutral", 2: "entailment"}
    # Large logit in column 2 -> high entailment only if the label name is honored.
    detector = _build_detector(
        id2label=shuffled,
        logits_by_premise={"ctx": [0.0, 0.0, 10.0]},
    )

    scores = detector._score_pairs(["ctx"], ["hyp"])
    entailment, _neutral, contradiction = scores[0]

    assert entailment > 0.99  # softmax of the dominant column 2 (entailment)
    assert contradiction < 0.01  # column 0 is contradiction here, and it's tiny


def test_init_rejects_config_missing_nli_labels():
    tokenizer = _FakeTokenizer()
    bad_model = _FakeModel({0: "positive", 1: "negative"}, tokenizer)
    with pytest.raises(ValueError, match="missing required NLI labels"):
        NLIHallucinationDetector(bad_model, tokenizer)


# --- 3. Tokenizer contract ------------------------------------------------------------


def test_tokenizer_called_with_max_length_512_and_truncation():
    detector = _build_detector()
    detector._score_pairs(["ctx"], ["hyp"])

    call = detector.tokenizer.calls[-1]
    assert call["max_length"] == 512
    assert call["truncation"] is True


# --- 4. Max aggregation across context sentences --------------------------------------


def test_aggregate_sentence_takes_max_entailment_and_max_contradiction_independently():
    # Two context sentences: one strongly entails, a different one strongly contradicts.
    detector = _build_detector(
        logits_by_premise={
            "ent_ctx": [10.0, 0.0, 0.0],  # ~1.0 entailment, ~0 contradiction
            "con_ctx": [0.0, 0.0, 10.0],  # ~0 entailment, ~1.0 contradiction
        }
    )

    entailment, _neutral, contradiction = detector._aggregate_sentence("resp", ["ent_ctx", "con_ctx"])

    assert entailment > 0.99  # max entailment came from ent_ctx
    assert contradiction > 0.99  # max contradiction came from con_ctx (a different sentence)


def test_detect_flags_contradicted_when_support_and_contradiction_coexist(monkeypatch):
    # Exercised through public detect(): one context chunk entails, another contradicts;
    # contradiction wins (ADR-007). Context chunks are passed in directly (ADR-008).
    monkeypatch.setattr(nli_baseline, "split_sentences", lambda text: [text] if text and text.strip() else [])
    detector = _build_detector(
        logits_by_premise={
            "ent_ctx": [10.0, 0.0, 0.0],  # ~1.0 entailment
            "con_ctx": [0.0, 0.0, 10.0],  # ~1.0 contradiction
        }
    )
    result = detector.detect(["ent_ctx", "con_ctx"], "resp")
    assert result.verdicts[0].entailment > 0.99
    assert result.verdicts[0].contradiction > 0.99
    assert result.verdicts[0].flag == "contradicted"
    assert result.response_hallucinated is True


def test_aggregate_sentence_with_no_context_returns_zeros():
    detector = _build_detector()
    assert detector._aggregate_sentence("resp", []) == (0.0, 0.0, 0.0)


# --- 5. Response-level aggregation ----------------------------------------------------


@pytest.fixture
def one_sentence_per_string(monkeypatch):
    """Treat each input string as exactly one sentence (no punkt dependency)."""
    monkeypatch.setattr(
        nli_baseline,
        "split_sentences",
        lambda text: [text] if text and text.strip() else [],
    )


def test_response_hallucinated_if_any_sentence_unsupported(monkeypatch, one_sentence_per_string):
    detector = _build_detector()
    # Canned (entailment, neutral, contradiction) per response sentence:
    # "good" -> supported, "bad" -> unverifiable (neither threshold met).
    scores = {
        "good": (0.9, 0.05, 0.05),
        "bad": (0.1, 0.8, 0.1),
    }
    monkeypatch.setattr(detector, "_aggregate_sentence", lambda sentence, ctx: scores[sentence])

    result = detector.detect(["ctx"], "good")
    assert result.response_hallucinated is False

    # detect() splits the response on sentences; feed both as separate one-sentence calls.
    monkeypatch.setattr(nli_baseline, "split_sentences", lambda text: text.split("|") if text else [])
    result = detector.detect(["ctx"], "good|bad")
    assert result.response_hallucinated is True
    assert [v.flag for v in result.verdicts] == ["supported", "unverifiable"]


def test_empty_response_is_not_hallucinated(one_sentence_per_string):
    detector = _build_detector()
    result = detector.detect(["some context"], "")
    assert result.response_hallucinated is False
    assert result.verdicts == []


# --- 6. response_color traffic-light aggregation (pure, no model) ----------------------


def _verdict(flag):
    return SentenceVerdict("s", 0.0, 0.0, 0.0, flag)


def test_response_color_red_when_any_contradicted():
    result = DetectionResult(True, [_verdict("supported"), _verdict("contradicted"), _verdict("unverifiable")])
    assert response_color(result) == "🔴"


def test_response_color_yellow_when_only_unverifiable():
    result = DetectionResult(True, [_verdict("supported"), _verdict("unverifiable")])
    assert response_color(result) == "🟡"


def test_response_color_green_when_all_supported():
    result = DetectionResult(False, [_verdict("supported"), _verdict("supported")])
    assert response_color(result) == "🟢"


def test_response_color_green_when_no_verdicts():
    result = DetectionResult(False, [])
    assert response_color(result) == "🟢"


# --- 7. score_response / apply_thresholds (reusable raw scores) ------------------------


def test_score_response_returns_raw_ent_con_per_sentence_dropping_neutral(monkeypatch):
    # Split on "|"; canned aggregation per response sentence to isolate the projection.
    monkeypatch.setattr(nli_baseline, "split_sentences", lambda text: text.split("|") if text else [])
    detector = _build_detector()
    triples = {"a": (0.8, 0.1, 0.2), "b": (0.3, 0.5, 0.7)}
    monkeypatch.setattr(detector, "_aggregate_sentence", lambda sentence, ctx: triples[sentence])

    scores = detector.score_response(["some context chunk"], "a|b")

    # (max_entailment, max_contradiction) per sentence, in order, neutral dropped.
    assert scores == [(0.8, 0.2), (0.3, 0.7)]


def test_apply_thresholds_flags_rows_model_free():
    all_scores = [
        [(0.9, 0.1)],  # supported -> not hallucinated
        [(0.1, 0.8)],  # contradicted -> hallucinated
        [(0.9, 0.1), (0.2, 0.2)],  # one supported, one unverifiable -> hallucinated
        [],  # no sentences -> not hallucinated
    ]

    result = apply_thresholds(all_scores, ent_thr=0.5, con_thr=0.5)

    assert result == [False, True, True, False]


def test_apply_thresholds_is_threshold_sensitive():
    all_scores = [[(0.6, 0.1)]]  # entailment 0.6
    assert apply_thresholds(all_scores, ent_thr=0.5, con_thr=0.5) == [False]  # supported
    assert apply_thresholds(all_scores, ent_thr=0.7, con_thr=0.5) == [True]  # now unverifiable
