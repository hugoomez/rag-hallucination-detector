"""Build token-level BIO-labeled train/val/test parquet datasets from RAGTruth for Track B.

Reuses source-loading, context normalization, and splitting utilities from
src/data/preprocess.py. Tokenizes (context, response) pairs with ModernBERT's fast
tokenizer using return_offsets_mapping=True, then assigns each response token an
O/B-HALL/I-HALL label based on character-offset overlap with RAGTruth's hallucination
spans (which are relative to the response text alone -- verified empirically against
the tokenizer's per-sequence offsets). Context and special tokens get -100 (ignored by
the loss). Per ADR-011, 0% of rows exceed the 4096-token budget, so no sliding-window
logic is needed; a truncation guard is kept as a sanity check only.
"""

import warnings
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

from src.data.preprocess import (
    RANDOM_STATE,
    filter_oversized_responses,
    load_merged_dataframe,
    make_group_stratified_val_split,
    print_truncation_report,
)

PROCESSED_DIR = Path("data/processed")
MODEL_NAME = "answerdotai/ModernBERT-base"
MAX_LENGTH = 4096
OUTPUT_TEMPLATE = "token_level_modernbert_{split}.parquet"

O_LABEL = 0
B_LABEL = 1
I_LABEL = 2
IGNORE_LABEL = -100
LABEL_NAMES = {O_LABEL: "O", B_LABEL: "B-HALL", I_LABEL: "I-HALL", IGNORE_LABEL: "IGN"}


def tokenize_and_align_labels(
    context: str, response: str, labels: list[dict], tokenizer, max_length: int = MAX_LENGTH
) -> dict:
    """Tokenize (context, response) as a pair and assign a BIO label to each token.

    Context tokens (sequence_id 0) and special tokens (sequence_id None) get -100
    (ignored by the loss). Response tokens (sequence_id 1) get O/B-HALL/I-HALL based
    on character-offset overlap with `labels`' start/end spans. Those spans are
    relative to the response text alone, and the tokenizer's offset_mapping for a
    paired encoding resets to (0, N) per sequence -- i.e. also relative to each
    sequence's own original string -- so no offset shifting is needed.
    """
    encoding = tokenizer(
        context,
        response,
        max_length=max_length,
        truncation="only_first",
        return_offsets_mapping=True,
        return_token_type_ids=False,
    )
    input_ids = encoding["input_ids"]
    offsets = encoding["offset_mapping"]
    sequence_ids = encoding.sequence_ids()

    assert len(input_ids) <= max_length, f"Token budget violated: got {len(input_ids)} tokens (max {max_length})"

    # Sanity guard: ADR-011 found 0% of rows exceed 4096 combined tokens. Compute
    # was_truncated independently (same technique as preprocess_modernbert.py's
    # tokenize_modernbert) rather than trust the tokenizer's own truncation behavior.
    num_special_tokens = tokenizer.num_special_tokens_to_add(pair=True)
    context_len = len(tokenizer(context, add_special_tokens=False)["input_ids"])
    response_len = len(tokenizer(response, add_special_tokens=False)["input_ids"])
    was_truncated = (context_len + response_len + num_special_tokens) > max_length

    sorted_spans = sorted(((label["start"], label["end"]) for label in labels), key=lambda s: s[0])

    def find_span_id(token_start: int, token_end: int) -> int | None:
        for span_id, (span_start, span_end) in enumerate(sorted_spans):
            if token_start < span_end and token_end > span_start:
                return span_id
        return None

    token_labels = []
    prev_span_id = None
    for seq_id, (char_start, char_end) in zip(sequence_ids, offsets):
        if seq_id != 1:
            # Context tokens and special tokens ([CLS]/[SEP]) are always ignored.
            token_labels.append(IGNORE_LABEL)
            continue
        if char_start == char_end:
            # Degenerate offset within the response segment (not expected in practice
            # for ModernBERT's BPE tokenizer, guarded defensively).
            token_labels.append(O_LABEL)
            prev_span_id = None
            continue

        span_id = find_span_id(char_start, char_end)
        if span_id is None:
            token_labels.append(O_LABEL)
            prev_span_id = None
        elif span_id == prev_span_id:
            token_labels.append(I_LABEL)
        else:
            token_labels.append(B_LABEL)
            prev_span_id = span_id

    assert len(token_labels) == len(input_ids)
    return {
        "input_ids": input_ids,
        "attention_mask": encoding["attention_mask"],
        "labels": token_labels,
        "was_truncated": was_truncated,
    }


def build_token_level_dataset(merged_df: pd.DataFrame, tokenizer, max_length: int = MAX_LENGTH) -> pd.DataFrame:
    """Apply tokenize_and_align_labels row-wise and assemble the token-level dataset."""
    encodings = merged_df.apply(
        lambda row: tokenize_and_align_labels(row["context"], row["response"], row["labels"], tokenizer, max_length),
        axis=1,
        result_type="expand",
    )

    return pd.DataFrame(
        {
            "source_id": merged_df["source_id"],
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "labels": encodings["labels"],
            "was_truncated": encodings["was_truncated"],
            "task_type": merged_df["task_type"],
            "split": merged_df["split"],
        }
    )


def print_alignment_sample(
    merged_df: pd.DataFrame, tokenizer, max_length: int = MAX_LENGTH, n: int = 5, random_state: int = RANDOM_STATE
) -> None:
    """Visually verify span/token alignment on a few sample rows before trusting it at scale.

    Prints the response with raw-offset spans re-highlighted, plus the resulting
    per-token BIO labels, for `n` hallucinated rows plus one row with zero spans
    (to visually confirm the "no spans -> all O" case too).
    """
    has_spans = merged_df[merged_df["labels"].apply(len) > 0]
    no_spans = merged_df[merged_df["labels"].apply(len) == 0]
    sample_rows = pd.concat(
        [
            has_spans.sample(n=min(n, len(has_spans)), random_state=random_state),
            no_spans.head(1),
        ]
    )

    for _, row in sample_rows.iterrows():
        response = row["response"]
        spans = sorted(((label["start"], label["end"]) for label in row["labels"]), key=lambda s: s[0])

        highlighted = response
        for start, end in reversed(spans):  # reversed so earlier insertions don't shift later offsets
            highlighted = highlighted[:end] + "<<<" + highlighted[end:]
            highlighted = highlighted[:start] + ">>>" + highlighted[start:]

        print(f"\n=== source_id={row['source_id']} task_type={row['task_type']} n_spans={len(spans)} ===")
        print("Response (spans marked >>>...<<<):")
        print(f"  {highlighted}")

        result = tokenize_and_align_labels(row["context"], response, row["labels"], tokenizer, max_length)
        tokens = tokenizer.convert_ids_to_tokens(result["input_ids"])
        sequence_ids = tokenizer(
            row["context"], response, max_length=max_length, truncation="only_first"
        ).sequence_ids()

        print("Token-level alignment (response tokens only):")
        for token, seq_id, label in zip(tokens, sequence_ids, result["labels"]):
            if seq_id == 1:
                print(f"  {token!r:20s} label={LABEL_NAMES[label]}")

        n_response_tokens = sum(1 for seq_id in sequence_ids if seq_id == 1)
        n_o = sum(1 for label in result["labels"] if label == O_LABEL)
        n_b = sum(1 for label in result["labels"] if label == B_LABEL)
        n_i = sum(1 for label in result["labels"] if label == I_LABEL)
        print(
            f"Summary: {len(result['labels']) - n_response_tokens} context/special tokens (-100), "
            f"response: {n_o} O, {n_b} B-HALL, {n_i} I-HALL"
        )


def main() -> None:
    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Loading and merging RAGTruth ...")
    merged_df = load_merged_dataframe()
    merged_df = filter_oversized_responses(merged_df, tokenizer, max_length=MAX_LENGTH)
    print(f"Dataset shape after filtering oversized responses: {merged_df.shape}")

    print("\nSanity-checking span/token alignment on sample rows ...")
    print_alignment_sample(merged_df, tokenizer, max_length=MAX_LENGTH)

    print("\nTokenizing (context, response) pairs and assigning BIO labels ...")
    processed_df = build_token_level_dataset(merged_df, tokenizer, max_length=MAX_LENGTH)

    train_full_df = processed_df[processed_df["split"] == "train"].reset_index(drop=True)
    test_df = processed_df[processed_df["split"] == "test"].reset_index(drop=True)
    test_df["split"] = "test"

    # make_group_stratified_val_split stratifies on a scalar "label_response" column
    # (preprocess.py's convention); derive it as a temporary proxy from the per-token
    # BIO sequence and drop it afterward -- it isn't part of Track B's saved schema.
    train_full_df["label_response"] = train_full_df["labels"].apply(
        lambda seq: int(any(label in (B_LABEL, I_LABEL) for label in seq))
    )
    train_df, val_df = make_group_stratified_val_split(train_full_df)
    train_df = train_df.drop(columns=["label_response"])
    val_df = val_df.drop(columns=["label_response"])

    train_ids = set(train_df["source_id"])
    val_ids = set(val_df["source_id"])
    leaked_ids = train_ids & val_ids
    assert not leaked_ids, f"Data leakage detected: {len(leaked_ids)} source_id(s) in both train and val"
    print(
        f"\nLeakage check passed: 0 source_id overlap between train ({len(train_ids)} unique) and val ({len(val_ids)} unique)."
    )

    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        print_truncation_report(df, name)
        if df["was_truncated"].sum() > 0:
            warnings.warn(
                f"[{name}] {int(df['was_truncated'].sum())} row(s) unexpectedly truncated at "
                f"max_length={MAX_LENGTH} -- ADR-011 found this should never happen."
            )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    saved = []
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        path = PROCESSED_DIR / OUTPUT_TEMPLATE.format(split=name)
        df.drop(columns=["was_truncated"]).to_parquet(path, index=False)
        saved.append((path, len(df)))

    print("\nSaved parquet files:")
    for path, n_rows in saved:
        print(f"  {path} — {n_rows} rows")


if __name__ == "__main__":
    main()
