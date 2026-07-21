import pytest
from transformers import AutoTokenizer

from src.data.preprocess_token_level import (
    HALLUCINATED_LABEL,
    IGNORE_LABEL,
    SUPPORTED_LABEL,
    is_noisy_span,
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


def response_flags_for(spans: list[dict], tokenizer) -> list[bool]:
    """is_implicit_true flags for the response tokens only (parallel to response_labels_for)."""
    result = tokenize_and_align_labels(CONTEXT, RESPONSE, spans, tokenizer, max_length=128)
    sequence_ids = tokenizer(CONTEXT, RESPONSE, max_length=128, truncation="only_first").sequence_ids()
    return [flag for flag, seq_id in zip(result["is_implicit_true"], sequence_ids) if seq_id == 1]


class TestIsNoisySpan:
    def test_implicit_true_not_due_to_null_is_noisy(self):
        assert is_noisy_span({"implicit_true": True, "due_to_null": False}) is True

    def test_implicit_true_but_due_to_null_is_not_noisy(self):
        # due_to_null spans are genuine hallucinations over null fields -- full weight.
        assert is_noisy_span({"implicit_true": True, "due_to_null": True}) is False

    def test_not_implicit_true_is_not_noisy(self):
        assert is_noisy_span({"implicit_true": False, "due_to_null": False}) is False

    def test_missing_keys_default_to_not_noisy(self):
        assert is_noisy_span({}) is False


class TestImplicitTrueFlagging:
    def test_flag_length_matches_input_ids_and_is_false_off_response(self, tokenizer):
        result = tokenize_and_align_labels(
            CONTEXT, RESPONSE, [{"start": 9, "end": 19, "implicit_true": True}], tokenizer, max_length=128
        )
        assert len(result["is_implicit_true"]) == len(result["input_ids"])
        sequence_ids = tokenizer(CONTEXT, RESPONSE, max_length=128, truncation="only_first").sequence_ids()
        for flag, seq_id in zip(result["is_implicit_true"], sequence_ids):
            if seq_id != 1:
                assert flag is False  # context/special positions never flagged

    def test_no_spans_means_no_flags(self, tokenizer):
        assert response_flags_for([], tokenizer) == [False] * 7

    def test_noisy_span_flags_its_tokens(self, tokenizer):
        # ' yesterday in Paris' (9,28) flagged noisy -> its 3 tokens are hallucinated AND flagged.
        spans = [{"start": 9, "end": 28, "implicit_true": True, "due_to_null": False}]
        assert response_labels_for(spans, tokenizer) == [0, 0, 0, 1, 1, 1, 0]
        assert response_flags_for(spans, tokenizer) == [False, False, False, True, True, True, False]

    def test_due_to_null_span_is_labeled_but_not_flagged(self, tokenizer):
        # implicit_true AND due_to_null: a genuine hallucination -> hallucinated label, NO flag.
        spans = [{"start": 9, "end": 28, "implicit_true": True, "due_to_null": True}]
        assert response_labels_for(spans, tokenizer) == [0, 0, 0, 1, 1, 1, 0]
        assert response_flags_for(spans, tokenizer) == [False] * 7

    def test_plain_hallucination_span_not_flagged(self, tokenizer):
        spans = [{"start": 9, "end": 28, "implicit_true": False}]
        assert response_labels_for(spans, tokenizer) == [0, 0, 0, 1, 1, 1, 0]
        assert response_flags_for(spans, tokenizer) == [False] * 7

    def test_merged_noisy_and_genuine_span_token_keeps_full_weight(self, tokenizer):
        # Edge case: a noisy span (5,19) and a genuine span (15,22) overlap. normalize_spans
        # would union them, but the flag is computed from RAW spans:
        #   'ined'(5,9)       -> noisy only            -> flagged
        #   'Ġyesterday'(9,19)-> noisy AND genuine     -> NOT flagged (genuine backing wins)
        #   'Ġin'(19,22)      -> genuine only          -> not flagged
        spans = [
            {"start": 5, "end": 19, "implicit_true": True, "due_to_null": False},  # noisy
            {"start": 15, "end": 22, "implicit_true": False},  # genuine
        ]
        assert response_labels_for(spans, tokenizer) == [0, 0, 1, 1, 1, 0, 0]
        assert response_flags_for(spans, tokenizer) == [False, False, True, False, False, False, False]


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
