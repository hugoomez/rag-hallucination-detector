import pandas as pd
import pytest
from transformers import AutoTokenizer

from src.data.preprocess_modernbert import report_combined_length_exceedance, tokenize_modernbert

MODEL_NAME = "answerdotai/ModernBERT-base"


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_NAME)


def test_tokenize_modernbert_preserves_short_response_with_long_context(tokenizer):
    # Small injected max_length just to exercise the truncation branch quickly; the
    # real-4096 behavior is covered by test_realistic_lengths_not_truncated below.
    long_context = "This is a filler sentence used to pad the context. " * 50
    short_response = "The answer is yes."
    max_length = 128

    result = tokenize_modernbert(long_context, short_response, tokenizer, max_length=max_length)

    assert result["was_truncated"] is True
    assert len(result["input_ids"]) <= max_length

    response_ids = tokenizer.encode(short_response, add_special_tokens=False)
    input_ids = result["input_ids"]
    n = len(response_ids)

    response_preserved_intact = any(input_ids[i : i + n] == response_ids for i in range(len(input_ids) - n + 1))
    assert response_preserved_intact, "full response tokens were not preserved intact in the output"


def test_realistic_lengths_not_truncated(tokenizer):
    # Moderately sized context + response, comfortably under 4096 tokens — the core
    # hypothesis that truncation becomes a non-issue at ModernBERT's max_length.
    context = "The mitochondrion is the powerhouse of the cell. " * 40
    response = "Mitochondria generate ATP, the cell's main energy currency."
    max_length = 4096

    result = tokenize_modernbert(context, response, tokenizer, max_length=max_length)

    assert result["was_truncated"] is False
    assert len(result["input_ids"]) <= max_length


def test_report_combined_length_exceedance_counts_correctly(tokenizer):
    # One short pair (fits) and one deliberately long pair (exceeds) at a small injected
    # max_length; the returned per-row Series must flag exactly the oversized row.
    merged_df = pd.DataFrame(
        {
            "task_type": ["QA", "Summary"],
            "context": ["Short context.", "Padding sentence. " * 100],
            "response": ["Short response.", "Padding sentence. " * 100],
        }
    )
    max_length = 64

    combined_len = report_combined_length_exceedance(merged_df, tokenizer, max_length=max_length)

    exceeds = (combined_len > max_length).tolist()
    assert exceeds == [False, True]
