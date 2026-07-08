from transformers import AutoTokenizer

from src.data.preprocess import truncate_and_tokenize

MODEL_NAME = "microsoft/deberta-v3-base"


def test_truncate_and_tokenize_preserves_short_response_with_long_context():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    long_context = "This is a filler sentence used to pad the context. " * 500
    short_response = "The answer is yes."
    max_length = 512

    result = truncate_and_tokenize(long_context, short_response, tokenizer, max_length=max_length)

    assert result["was_truncated"] is True
    assert len(result["input_ids"]) <= max_length

    response_ids = tokenizer.encode(short_response, add_special_tokens=False)
    input_ids = result["input_ids"]
    n = len(response_ids)

    response_preserved_intact = any(
        input_ids[i : i + n] == response_ids for i in range(len(input_ids) - n + 1)
    )
    assert response_preserved_intact, "full response tokens were not preserved intact in the output"
