"""Build binary token-labeled train/val/test parquet datasets from RAGTruth for Track B.

Per ADR-013 (LettuceDetect parity, arXiv:2502.17125), each response token gets a BINARY
label: 0 = supported, 1 = hallucinated (any character overlap with a gold span). The
earlier 3-class BIO scheme was retired: it created an ultra-rare B-HALL class (0.35% of
tokens) whose inverse-frequency weight actively rewarded span fragmentation. Span
boundaries are now reconstructed at inference time by merging consecutive positive
tokens (train_token_level.py), not learned as separate classes.

Reuses source-loading, context normalization, and splitting utilities from
src/data/preprocess.py. Tokenizes (context, response) pairs with ModernBERT's fast
tokenizer using return_offsets_mapping=True; RAGTruth's hallucination spans are relative
to the response text alone, matching the tokenizer's per-sequence offsets (verified
empirically). Context and special tokens get -100 (ignored by the loss). Gold spans are
normalized (sorted, overlapping/adjacent spans unioned -- RAGTruth has 115 responses
with overlapping annotations) and saved per row, alongside per-token character offsets,
so the trainer can compute LettuceDetect's character-overlap span metrics. Per ADR-011,
0% of rows exceed the 4096-token budget, so no sliding-window logic is needed; a
truncation guard is kept as a sanity check only.
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
OUTPUT_TEMPLATE = "token_level_binary_{split}.parquet"

SUPPORTED_LABEL = 0
HALLUCINATED_LABEL = 1
IGNORE_LABEL = -100
LABEL_NAMES = {SUPPORTED_LABEL: "supported", HALLUCINATED_LABEL: "hallucinated", IGNORE_LABEL: "IGN"}


def is_noisy_span(span: dict) -> bool:
    """True for gold spans the RAGTruth annotators themselves flagged as contextually true.

    `implicit_true=True` marks a span annotated as hallucinated that the annotator
    acknowledged is actually true given the context (13.5% of all gold spans; 73.6% of
    "Subtle Baseless Info"). `due_to_null=True` spans are excluded from the noisy set:
    they are genuine hallucinations over null JSON fields (98% Evident Baseless Info in
    Data2txt) and must keep full training weight. Used only to build the auxiliary
    `is_implicit_true` column consumed by --implicit_true_weight at training time --
    it never affects the binary labels themselves.
    """
    return bool(span.get("implicit_true", False)) and not bool(span.get("due_to_null", False))


def normalize_spans(labels: list[dict]) -> list[tuple[int, int]]:
    """Sort gold spans and union any that overlap or touch, returning disjoint spans.

    RAGTruth contains overlapping annotations (115 of 17,790 responses), so a plain
    non-overlap assert would crash on real data. For binary labels the union is
    semantically lossless (a token in ANY span is hallucinated), and disjoint gold
    spans keep the character-overlap metric free of double counting.
    """
    spans = sorted((label["start"], label["end"]) for label in labels)
    merged: list[tuple[int, int]] = []
    for start, end in spans:
        assert end > start, f"Empty or inverted gold span ({start}, {end})"
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    for (_, prev_end), (next_start, _) in zip(merged, merged[1:]):
        assert next_start > prev_end, "normalize_spans produced non-disjoint spans"
    return merged


def tokenize_and_align_labels(
    context: str, response: str, labels: list[dict], tokenizer, max_length: int = MAX_LENGTH
) -> dict:
    """Tokenize (context, response) as a pair and assign a binary label to each token.

    Context tokens (sequence_id 0) and special tokens (sequence_id None) get -100
    (ignored by the loss). Response tokens (sequence_id 1) get 1 (hallucinated) iff
    they overlap any normalized gold span by at least one character, else 0
    (supported). Those spans are relative to the response text alone, and the
    tokenizer's offset_mapping for a paired encoding resets to (0, N) per sequence --
    i.e. also relative to each sequence's own original string -- so no offset shifting
    is needed. Per-token offsets and the normalized gold spans are returned too, so
    the trainer can rebuild character-level spans from token predictions.
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

    gold_spans = normalize_spans(labels)

    # ACWS auxiliary flag (never touches the labels): a token is is_implicit_true iff
    # every raw gold span covering it is annotator-flagged noise (is_noisy_span). The
    # noisy/clean regions come from the RAW spans, not the normalized union, because
    # normalize_spans can merge a noisy span into a genuine one -- a token backed by ANY
    # genuine annotation must keep full training weight.
    noisy_regions = normalize_spans([span for span in labels if is_noisy_span(span)])
    clean_regions = normalize_spans([span for span in labels if not is_noisy_span(span)])

    token_labels = []
    token_implicit_flags = []
    for seq_id, (char_start, char_end) in zip(sequence_ids, offsets):
        if seq_id != 1:
            # Context tokens and special tokens ([CLS]/[SEP]) are always ignored.
            token_labels.append(IGNORE_LABEL)
            token_implicit_flags.append(False)
            continue
        # A zero-width response-token offset would silently corrupt labels and the
        # trainer's span reconstruction; ModernBERT's BPE tokenizer never emits one.
        assert char_start < char_end, f"Zero-width offset ({char_start}, {char_end}) on a response token"
        overlaps = any(char_start < span_end and char_end > span_start for span_start, span_end in gold_spans)
        token_labels.append(HALLUCINATED_LABEL if overlaps else SUPPORTED_LABEL)

        overlaps_noisy = any(char_start < span_end and char_end > span_start for span_start, span_end in noisy_regions)
        overlaps_clean = any(char_start < span_end and char_end > span_start for span_start, span_end in clean_regions)
        flagged = overlaps_noisy and not overlaps_clean
        assert not flagged or token_labels[-1] == HALLUCINATED_LABEL, "flag on a non-hallucinated token"
        token_implicit_flags.append(flagged)

    assert len(token_labels) == len(input_ids)
    assert len(token_implicit_flags) == len(input_ids)
    return {
        "input_ids": input_ids,
        "attention_mask": encoding["attention_mask"],
        "labels": token_labels,
        "is_implicit_true": token_implicit_flags,
        "token_starts": [start for start, _ in offsets],
        "token_ends": [end for _, end in offsets],
        "gold_starts": [start for start, _ in gold_spans],
        "gold_ends": [end for _, end in gold_spans],
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
            # Raw response.jsonl `id`: a unique per-response key (source_id repeats 6x),
            # enabling key-based join-backs to raw span metadata instead of positional-only.
            "response_id": merged_df["id"].astype(str),
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "labels": encodings["labels"],
            "is_implicit_true": encodings["is_implicit_true"],
            "token_starts": encodings["token_starts"],
            "token_ends": encodings["token_ends"],
            "gold_starts": encodings["gold_starts"],
            "gold_ends": encodings["gold_ends"],
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
    per-token binary labels, for `n` hallucinated rows plus one row with zero spans
    (to visually confirm the "no spans -> all supported" case too).
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
        spans = normalize_spans(row["labels"])

        highlighted = response
        for start, end in reversed(spans):  # reversed so earlier insertions don't shift later offsets
            highlighted = highlighted[:end] + "<<<" + highlighted[end:]
            highlighted = highlighted[:start] + ">>>" + highlighted[start:]

        print(f"\n=== source_id={row['source_id']} task_type={row['task_type']} n_spans={len(spans)} ===")
        print("Response (normalized spans marked >>>...<<<):")
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
        n_supported = sum(1 for label in result["labels"] if label == SUPPORTED_LABEL)
        n_hallucinated = sum(1 for label in result["labels"] if label == HALLUCINATED_LABEL)
        print(
            f"Summary: {len(result['labels']) - n_response_tokens} context/special tokens (-100), "
            f"response: {n_supported} supported, {n_hallucinated} hallucinated"
        )


def print_implicit_true_report(df: pd.DataFrame, split_name: str) -> None:
    """Live sanity check of the is_implicit_true column against the audit numbers.

    Expected (verified 2026-07-12 against response.jsonl): ~13.5% of gold-span char mass
    is annotator-flagged implicit_true overall, and ~605 hallucinated responses across
    the official train split (our train+val) consist ONLY of flagged spans.
    """
    n_pos = int(df["labels"].apply(lambda seq: sum(1 for v in seq if v == HALLUCINATED_LABEL)).sum())
    n_flagged = int(df["is_implicit_true"].apply(lambda seq: sum(bool(v) for v in seq)).sum())

    def all_flagged(row) -> bool:
        pos = [flag for label, flag in zip(row["labels"], row["is_implicit_true"]) if label == HALLUCINATED_LABEL]
        return len(pos) > 0 and all(pos)

    n_all_flagged = int(df.apply(all_flagged, axis=1).sum())
    share = n_flagged / n_pos if n_pos else 0.0
    print(
        f"[{split_name}] implicit_true: {n_flagged}/{n_pos} positive tokens flagged ({share:.2%}) | "
        f"all-flagged hallucinated responses: {n_all_flagged}"
    )


def main(processed_dir: Path = PROCESSED_DIR) -> None:
    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Loading and merging RAGTruth ...")
    merged_df = load_merged_dataframe()
    merged_df = filter_oversized_responses(merged_df, tokenizer, max_length=MAX_LENGTH)
    print(f"Dataset shape after filtering oversized responses: {merged_df.shape}")

    n_merged_rows = int(
        merged_df["labels"].apply(lambda labels: len(labels) > 0 and len(normalize_spans(labels)) < len(labels)).sum()
    )
    print(f"Rows with overlapping/adjacent gold spans unioned by normalize_spans: {n_merged_rows}")

    print("\nSanity-checking span/token alignment on sample rows ...")
    print_alignment_sample(merged_df, tokenizer, max_length=MAX_LENGTH)

    print("\nTokenizing (context, response) pairs and assigning binary labels ...")
    processed_df = build_token_level_dataset(merged_df, tokenizer, max_length=MAX_LENGTH)

    train_full_df = processed_df[processed_df["split"] == "train"].reset_index(drop=True)
    test_df = processed_df[processed_df["split"] == "test"].reset_index(drop=True)

    # make_group_stratified_val_split stratifies on a scalar "label_response" column
    # (preprocess.py's convention); derive it as a temporary proxy from the per-token
    # binary sequence and drop it afterward -- it isn't part of Track B's saved schema.
    train_full_df["label_response"] = train_full_df["labels"].apply(
        lambda seq: int(any(label == HALLUCINATED_LABEL for label in seq))
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

    print("\nimplicit_true flag diagnostics (auxiliary column, labels untouched):")
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        print_implicit_true_report(df, name)

    processed_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        path = processed_dir / OUTPUT_TEMPLATE.format(split=name)
        df.drop(columns=["was_truncated"]).to_parquet(path, index=False)
        saved.append((path, len(df)))

    print("\nSaved parquet files:")
    for path, n_rows in saved:
        print(f"  {path} — {n_rows} rows")


if __name__ == "__main__":
    main()
