"""Build response-level train/val/test parquet datasets from RAGTruth for ModernBERT.

Parallel to src/data/preprocess.py (which targets DeBERTa-v3-base at max_length=512),
but tokenizes with answerdotai/ModernBERT-base at max_length=4096. The heavy lifting is
imported from preprocess.py rather than duplicated: the load/merge, context
normalization, ADR-004 context-head truncation, oversized-response filtering, dataset
assembly, and group-stratified val split are all tokenizer- and max_length-agnostic and
reused as-is. This guarantees the ModernBERT pipeline's truncation behavior is byte
identical to the DeBERTa one instead of a re-derivation that could drift.

The one new piece is report_combined_length_exceedance: given ModernBERT's much larger
context window, truncation should touch very few or zero rows. That function prints how
many rows' full (context + response + special tokens) length exceeds 4096 BEFORE any
truncation, computed directly from raw token counts, as an independent check of the
"truncation becomes a non-issue at 4096" hypothesis.

Writes response_level_modernbert_{train,val,test}.parquet to data/processed/ (new files;
the DeBERTa response_level_{train,val,test}.parquet are left untouched).
"""

from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

from src.data.preprocess import (
    RANDOM_STATE,
    VAL_SIZE,
    filter_oversized_responses,
    load_merged_dataframe,
    make_group_stratified_val_split,
    print_truncation_report,
)

PROCESSED_DIR = Path("data/processed")
MODEL_NAME = "answerdotai/ModernBERT-base"
MAX_LENGTH = 4096
OUTPUT_TEMPLATE = "response_level_modernbert_{split}.parquet"


def report_combined_length_exceedance(merged_df: pd.DataFrame, tokenizer, max_length: int = MAX_LENGTH) -> pd.Series:
    """Print how many rows' full (context + response + special) length exceeds max_length.

    Measured BEFORE any truncation, straight from raw token counts, so it stands as an
    independent check of the "truncation becomes a non-issue at 4096" hypothesis rather
    than being inferred from the was_truncated flag. Reports overall and per task_type,
    with the max observed combined length per group (the strongest single signal for
    whether max_length has real headroom). Returns the per-row combined-length Series so
    callers/tests can reuse it.
    """
    num_special_tokens = tokenizer.num_special_tokens_to_add(pair=True)

    context_len = merged_df["context"].apply(lambda text: len(tokenizer(text, add_special_tokens=False)["input_ids"]))
    response_len = merged_df["response"].apply(lambda text: len(tokenizer(text, add_special_tokens=False)["input_ids"]))
    combined_len = context_len + response_len + num_special_tokens

    report = pd.DataFrame(
        {
            "task_type": merged_df["task_type"].to_numpy(),
            "combined_len": combined_len.to_numpy(),
            "exceeds": (combined_len > max_length).to_numpy(),
        }
    )

    print(f"\n[combined-length check @ max_length={max_length}]")
    grouped = report.groupby("task_type")["exceeds"].agg(["sum", "count", "mean"])
    max_by_task = report.groupby("task_type")["combined_len"].max()
    for task_type, row in grouped.iterrows():
        print(
            f"  {task_type:10s} n={int(row['count']):5d}  exceeding={int(row['sum']):5d}  "
            f"({row['mean']:.2%})  max_combined_len={int(max_by_task[task_type]):5d}"
        )
    total_exceeding = int(report["exceeds"].sum())
    print(
        f"  {'ALL':10s} n={len(report):5d}  exceeding={total_exceeding:5d}  "
        f"({report['exceeds'].mean():.2%})  max_combined_len={int(report['combined_len'].max()):5d}"
    )
    print("  (response-alone drops reported separately below; expected 0 at this max_length)")

    return combined_len


def tokenize_modernbert(context: str, response: str, tokenizer, max_length: int = MAX_LENGTH) -> dict:
    """Tokenize (context, response) as a pair using ModernBERT's fast tokenizer.

    ModernBERT's fast-tokenizer backend does not implement prepare_for_model (which the
    reused preprocess.truncate_and_tokenize relies on), so this uses the standard fast
    pair-encoding call. truncation="only_first" truncates the context (first sequence)
    if the pair ever exceeds max_length, always preserving the full response — the same
    guarantee ADR-004 gives. Given the combined-length diagnostic showed 0% of rows
    exceed 4096 (max observed 2618), this truncation branch is not expected to trigger,
    but the safety-net assertion and the properly computed was_truncated flag stay in.
    """
    encoding = tokenizer(
        context,
        response,
        max_length=max_length,
        truncation="only_first",
        return_token_type_ids=False,
    )

    # Permanent safety net (same pattern as the DeBERTa pipeline's own assertion).
    assert (
        len(encoding["input_ids"]) <= max_length
    ), f"Token budget violated: got {len(encoding['input_ids'])} tokens (max {max_length})"

    # was_truncated: did the untruncated pair (context + response + special tokens) exceed
    # the budget? Computed from raw token counts rather than hardcoded False, even though
    # the diagnostic says this is False for every RAGTruth row at max_length=4096.
    num_special_tokens = tokenizer.num_special_tokens_to_add(pair=True)
    context_len = len(tokenizer(context, add_special_tokens=False)["input_ids"])
    response_len = len(tokenizer(response, add_special_tokens=False)["input_ids"])
    was_truncated = (context_len + response_len + num_special_tokens) > max_length

    return {
        "input_ids": encoding["input_ids"],
        "attention_mask": encoding["attention_mask"],
        "was_truncated": was_truncated,
    }


def build_response_level_dataset_modernbert(
    merged_df: pd.DataFrame, tokenizer, max_length: int = MAX_LENGTH
) -> pd.DataFrame:
    """Apply tokenize_modernbert row-wise and assemble the response-level dataset.

    Local equivalent of preprocess.build_response_level_dataset, differing only in the
    per-row tokenizer call (tokenize_modernbert instead of the prepare_for_model-based
    truncate_and_tokenize). Output schema is identical to the DeBERTa pipeline.
    """
    encodings = merged_df.apply(
        lambda row: tokenize_modernbert(row["context"], row["response"], tokenizer, max_length),
        axis=1,
        result_type="expand",
    )

    return pd.DataFrame(
        {
            "source_id": merged_df["source_id"],
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "label_response": merged_df["labels"].apply(lambda labels: int(len(labels) > 0)),
            "was_truncated": encodings["was_truncated"],
            "task_type": merged_df["task_type"],
            "split": merged_df["split"],
        }
    )


def main() -> None:
    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Loading and merging RAGTruth ...")
    merged_df = load_merged_dataframe()
    print(f"Merged dataset shape: {merged_df.shape}")

    report_combined_length_exceedance(merged_df, tokenizer, MAX_LENGTH)

    merged_df = filter_oversized_responses(merged_df, tokenizer, MAX_LENGTH)
    print(f"Dataset shape after filtering oversized responses: {merged_df.shape}")

    print(
        "Tokenizing (context, response) pairs — context head-truncated, response always kept whole "
        f"(max_length={MAX_LENGTH}) ..."
    )
    processed_df = build_response_level_dataset_modernbert(merged_df, tokenizer, MAX_LENGTH)

    train_full_df = processed_df[processed_df["split"] == "train"].reset_index(drop=True)
    test_df = processed_df[processed_df["split"] == "test"].reset_index(drop=True)
    test_df["split"] = "test"

    train_df, val_df = make_group_stratified_val_split(train_full_df, VAL_SIZE, RANDOM_STATE)

    # Sanity check: no source_id (data leakage) shared between train and val.
    train_ids = set(train_df["source_id"])
    val_ids = set(val_df["source_id"])
    leaked_ids = train_ids & val_ids
    assert not leaked_ids, f"Data leakage detected: {len(leaked_ids)} source_id(s) in both train and val"
    print(
        f"\nLeakage check passed: 0 source_id overlap between train ({len(train_ids)} unique) "
        f"and val ({len(val_ids)} unique)."
    )

    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        print_truncation_report(df, name)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    saved = []
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        path = PROCESSED_DIR / OUTPUT_TEMPLATE.format(split=name)
        df.to_parquet(path, index=False)
        saved.append((path, len(df)))

    print("\nSaved parquet files:")
    for path, n_rows in saved:
        print(f"  {path} — {n_rows} rows")


if __name__ == "__main__":
    main()
