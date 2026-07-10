import pytest
from transformers import AutoTokenizer

from src.data.preprocess_token_level import B_LABEL, I_LABEL, IGNORE_LABEL, O_LABEL, tokenize_and_align_labels

MODEL_NAME = "answerdotai/ModernBERT-base"


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_NAME)


def test_known_span_produces_exact_expected_bio_sequence(tokenizer):
    # Empirically verified tokenization (ModernBERT BPE) for this exact response:
    # tokens: ['It', 'Ġra', 'ined', 'Ġyesterday', 'Ġin', 'ĠParis', '.']
    # offsets: (0,2) (2,5) (5,9) (9,19) (19,22) (22,28) (28,29)
    context = "Filler context, irrelevant to the span."
    response = "It rained yesterday in Paris."
    labels = [{"start": 9, "end": 28, "text": " yesterday in Paris"}]  # spans 3 tokens exactly

    result = tokenize_and_align_labels(context, response, labels, tokenizer, max_length=128)
    sequence_ids = tokenizer(context, response, max_length=128, truncation="only_first").sequence_ids()

    response_labels = [label for label, seq_id in zip(result["labels"], sequence_ids) if seq_id == 1]
    assert response_labels == [O_LABEL, O_LABEL, O_LABEL, B_LABEL, I_LABEL, I_LABEL, O_LABEL]


def test_context_and_special_tokens_get_ignore_label(tokenizer):
    context = "Some context sentence here."
    response = "A faithful response with no hallucination."
    labels = []

    result = tokenize_and_align_labels(context, response, labels, tokenizer, max_length=128)
    sequence_ids = tokenizer(context, response, max_length=128, truncation="only_first").sequence_ids()

    for label, seq_id in zip(result["labels"], sequence_ids):
        if seq_id != 1:
            assert label == IGNORE_LABEL


def test_no_spans_means_all_response_tokens_are_o(tokenizer):
    context = "Some context sentence here."
    response = "A completely faithful response with no hallucinated content at all."
    labels = []

    result = tokenize_and_align_labels(context, response, labels, tokenizer, max_length=128)
    sequence_ids = tokenizer(context, response, max_length=128, truncation="only_first").sequence_ids()

    response_labels = [label for label, seq_id in zip(result["labels"], sequence_ids) if seq_id == 1]
    assert all(label == O_LABEL for label in response_labels)
    assert B_LABEL not in response_labels
    assert I_LABEL not in response_labels
