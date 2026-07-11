import pytest
from transformers import AutoTokenizer

from src.data.preprocess_token_level import (
    HALLUCINATED_LABEL,
    IGNORE_LABEL,
    SUPPORTED_LABEL,
    normalize_spans,
    tokenize_and_align_labels,
)

MODEL_NAME = "answerdotai/ModernBERT-base"

# Empirically verified tokenization (ModernBERT BPE) for the shared test response:
# tokens:  ['It', 'Ġra', 'ined', 'Ġyesterday', 'Ġin', 'ĠParis', '.']
# offsets: (0,2) (2,5) (5,9) (9,19) (19,22) (22,28) (28,29)
CONTEXT = "Filler context, irrelevant to the span."
RESPONSE = "It rained yesterday in Paris."


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_NAME)


def response_labels_for(spans: list[dict], tokenizer) -> list[int]:
    """Run alignment on the shared CONTEXT/RESPONSE pair and return response-token labels only."""
    result = tokenize_and_align_labels(CONTEXT, RESPONSE, spans, tokenizer, max_length=128)
    sequence_ids = tokenizer(CONTEXT, RESPONSE, max_length=128, truncation="only_first").sequence_ids()
    return [label for label, seq_id in zip(result["labels"], sequence_ids) if seq_id == 1]


def test_known_span_produces_exact_expected_binary_sequence(tokenizer):
    labels = [{"start": 9, "end": 28, "text": " yesterday in Paris"}]  # spans 3 tokens exactly
    assert response_labels_for(labels, tokenizer) == [0, 0, 0, 1, 1, 1, 0]


def test_context_and_special_tokens_get_ignore_label(tokenizer):
    result = tokenize_and_align_labels(CONTEXT, RESPONSE, [], tokenizer, max_length=128)
    sequence_ids = tokenizer(CONTEXT, RESPONSE, max_length=128, truncation="only_first").sequence_ids()

    for label, seq_id in zip(result["labels"], sequence_ids):
        if seq_id != 1:
            assert label == IGNORE_LABEL


def test_no_spans_means_all_response_tokens_are_supported(tokenizer):
    response_labels = response_labels_for([], tokenizer)
    assert all(label == SUPPORTED_LABEL for label in response_labels)
    assert HALLUCINATED_LABEL not in response_labels


def test_span_starting_and_ending_mid_subword_labels_whole_tokens(tokenizer):
    # (12, 15) sits strictly inside the 'Ġyesterday' token (9, 19): partial overlap
    # in either direction must label the whole token hallucinated.
    labels = [{"start": 12, "end": 15, "text": "ter"}]
    assert response_labels_for(labels, tokenizer) == [0, 0, 0, 1, 0, 0, 0]

    # Starts mid-'Ġra' (2,5) and ends mid-'Ġyesterday' (9,19): all touched tokens flip.
    labels = [{"start": 3, "end": 12, "text": "ained yes"}]
    assert response_labels_for(labels, tokenizer) == [0, 1, 1, 1, 0, 0, 0]


def test_span_at_response_start(tokenizer):
    labels = [{"start": 0, "end": 2, "text": "It"}]
    assert response_labels_for(labels, tokenizer) == [1, 0, 0, 0, 0, 0, 0]


def test_span_at_response_end(tokenizer):
    # Runs to the final character (len(RESPONSE) == 29), covering 'ĠParis' and '.'.
    labels = [{"start": 22, "end": 29, "text": " Paris."}]
    assert response_labels_for(labels, tokenizer) == [0, 0, 0, 0, 0, 1, 1]


def test_two_spans_separated_by_one_supported_token(tokenizer):
    # 'Ġyesterday' and 'ĠParis' hallucinated, 'Ġin' (19,22) supported between them.
    labels = [
        {"start": 9, "end": 19, "text": " yesterday"},
        {"start": 22, "end": 28, "text": " Paris"},
    ]
    assert response_labels_for(labels, tokenizer) == [0, 0, 0, 1, 0, 1, 0]


def test_overlapping_gold_spans_are_unioned(tokenizer):
    labels = [
        {"start": 9, "end": 24, "text": " yesterday in Pa"},
        {"start": 19, "end": 28, "text": " in Paris"},
    ]
    result = tokenize_and_align_labels(CONTEXT, RESPONSE, labels, tokenizer, max_length=128)
    assert result["gold_starts"] == [9]
    assert result["gold_ends"] == [28]
    assert response_labels_for(labels, tokenizer) == [0, 0, 0, 1, 1, 1, 0]


def test_token_offsets_align_with_input_ids(tokenizer):
    result = tokenize_and_align_labels(CONTEXT, RESPONSE, [], tokenizer, max_length=128)
    assert len(result["token_starts"]) == len(result["input_ids"])
    assert len(result["token_ends"]) == len(result["input_ids"])


class TestNormalizeSpans:
    def test_disjoint_spans_pass_through_sorted(self):
        labels = [{"start": 20, "end": 30}, {"start": 0, "end": 10}]
        assert normalize_spans(labels) == [(0, 10), (20, 30)]

    def test_overlapping_spans_are_unioned(self):
        labels = [{"start": 0, "end": 15}, {"start": 10, "end": 30}]
        assert normalize_spans(labels) == [(0, 30)]

    def test_adjacent_spans_are_unioned(self):
        labels = [{"start": 0, "end": 10}, {"start": 10, "end": 20}]
        assert normalize_spans(labels) == [(0, 20)]

    def test_contained_span_is_absorbed(self):
        labels = [{"start": 0, "end": 30}, {"start": 5, "end": 10}]
        assert normalize_spans(labels) == [(0, 30)]

    def test_empty_input(self):
        assert normalize_spans([]) == []

    def test_empty_span_raises(self):
        with pytest.raises(AssertionError):
            normalize_spans([{"start": 5, "end": 5}])
