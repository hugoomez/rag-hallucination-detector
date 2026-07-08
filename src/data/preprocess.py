"""Build response-level train/val/test parquet datasets from RAGTruth for DeBERTa-v3.

Loads source_info.jsonl + response.jsonl, normalizes context by task_type (see
notebooks/01_eda_ragtruth.ipynb for the original EDA), tokenizes (context, response)
pairs while always preserving the full response (context-only head truncation, per
ADR-004), and writes response_level_{train,val,test}.parquet to data/processed/.
"""

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

DATASET_DIR = Path("data/raw/ragtruth/dataset")
PROCESSED_DIR = Path("data/processed")
MODEL_NAME = "microsoft/deberta-v3-base"
MAX_LENGTH = 512
VAL_SIZE = 0.10
RANDOM_STATE = 42


def normalize_context(row: pd.Series) -> str:
    """Flatten source_info into a single text string, format depends on task_type."""
    task_type = row["task_type"]
    source_info = row["source_info"]

    if task_type == "Summary":
        return source_info
    elif task_type == "QA":
        return f"Question: {source_info['question']}\n\nPassages: {source_info['passages']}"
    elif task_type == "Data2txt":
        return json.dumps(source_info, indent=2, ensure_ascii=False)
    else:
        raise ValueError(f"Unknown task_type: {task_type}")


def load_merged_dataframe(dataset_dir: Path = DATASET_DIR) -> pd.DataFrame:
    """Load source_info + response, normalize context, and merge on source_id."""
    source_info_df = pd.read_json(dataset_dir / "source_info.jsonl", lines=True)
    response_df = pd.read_json(dataset_dir / "response.jsonl", lines=True)

    source_info_df["context"] = source_info_df.apply(normalize_context, axis=1)

    merged_df = response_df.merge(source_info_df, on="source_id", how="left")
    assert merged_df.shape[0] == response_df.shape[0], "merge dropped or duplicated rows"

    return merged_df


def filter_oversized_responses(merged_df: pd.DataFrame, tokenizer, max_length: int = MAX_LENGTH) -> pd.DataFrame:
    """Drop rows whose response alone (zero context tokens) wouldn't fit the token budget.

    Per ADR-004, the response is never truncated; rows where it alone exceeds the token
    budget are excluded rather than breaking that guarantee.
    """
    num_special_tokens = tokenizer.num_special_tokens_to_add(pair=True)
    max_response_len = max_length - num_special_tokens

    response_token_count = merged_df["response"].apply(
        lambda text: len(tokenizer(text, add_special_tokens=False)["input_ids"])
    )
    oversized = response_token_count > max_response_len

    if oversized.any():
        dropped = merged_df.loc[oversized, ["source_id", "split"]]
        print(
            f"Dropped {len(dropped)} row(s) where response alone exceeds the token budget: "
            f"{dropped['source_id'].tolist()}"
        )
        for split_name, group in dropped.groupby("split"):
            print(f"  from split={split_name}: {group['source_id'].tolist()}")

    return merged_df.loc[~oversized].reset_index(drop=True)


def truncate_and_tokenize(context: str, response: str, tokenizer, max_length: int = MAX_LENGTH) -> dict:
    """Tokenize (context, response) as a pair, always keeping the full response.

    Only the context is truncated, and only from the end (head truncation: the
    beginning of the context is kept, since that's what fits the leftover budget
    after reserving room for the full response + special tokens).
    """
    context_ids = tokenizer.encode(context, add_special_tokens=False)
    response_ids = tokenizer.encode(response, add_special_tokens=False)
    num_special_tokens = tokenizer.num_special_tokens_to_add(pair=True)

    max_context_len = max(max_length - num_special_tokens - len(response_ids), 0)
    was_truncated = len(context_ids) > max_context_len
    context_ids = context_ids[:max_context_len]

    encoding = tokenizer.prepare_for_model(
        context_ids,
        response_ids,
        add_special_tokens=True,
        truncation=False,
        padding=False,
        return_attention_mask=True,
        return_token_type_ids=False,
    )

    # Safety net per ADR-006: filter_oversized_responses should already guarantee this,
    # but assert the actual invariant we care about rather than trust the intermediate budget math.
    assert (
        len(encoding["input_ids"]) <= max_length
    ), f"Token budget violated: got {len(encoding['input_ids'])} tokens (max {max_length})"

    return {
        "input_ids": encoding["input_ids"],
        "attention_mask": encoding["attention_mask"],
        "was_truncated": was_truncated,
    }


def build_response_level_dataset(merged_df: pd.DataFrame, tokenizer, max_length: int = MAX_LENGTH) -> pd.DataFrame:
    """Apply truncate_and_tokenize row-wise and assemble the response-level dataset."""
    encodings = merged_df.apply(
        lambda row: truncate_and_tokenize(row["context"], row["response"], tokenizer, max_length),
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


def make_group_stratified_val_split(
    train_full_df: pd.DataFrame, val_size: float = VAL_SIZE, random_state: int = RANDOM_STATE
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split off a validation set at the source_id level, so no source_id leaks across sets.

    Each source_id has multiple responses (one per model); a row-level split could put
    siblings of the same source_id in both train and val. We split whole source_id
    groups instead, stratifying on each group's majority label_response as a proxy for
    the requested "stratify by label_response".
    """
    group_majority_label = train_full_df.groupby("source_id")["label_response"].mean().round().astype(int)
    source_ids = group_majority_label.index.to_numpy()

    train_ids, val_ids = train_test_split(
        source_ids,
        test_size=val_size,
        stratify=group_majority_label.to_numpy(),
        random_state=random_state,
    )
    train_ids, val_ids = set(train_ids), set(val_ids)

    train_df = train_full_df[train_full_df["source_id"].isin(train_ids)].reset_index(drop=True)
    val_df = train_full_df[train_full_df["source_id"].isin(val_ids)].reset_index(drop=True)

    train_df["split"] = "train"
    val_df["split"] = "val"

    return train_df, val_df


def print_truncation_report(df: pd.DataFrame, split_name: str) -> None:
    print(f"\n[{split_name}] truncation rate by task_type:")
    grouped = df.groupby("task_type")["was_truncated"].agg(["sum", "count", "mean"])
    for task_type, row in grouped.iterrows():
        print(f"  {task_type:10s} n={int(row['count']):5d}  truncated={int(row['sum']):5d}  ({row['mean']:.2%})")
    total_truncated = int(df["was_truncated"].sum())
    print(f"  {'ALL':10s} n={len(df):5d}  truncated={total_truncated:5d}  ({df['was_truncated'].mean():.2%})")


def main() -> None:
    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print(f"Loading and merging RAGTruth from {DATASET_DIR} ...")
    merged_df = load_merged_dataframe()
    print(f"Merged dataset shape: {merged_df.shape}")

    merged_df = filter_oversized_responses(merged_df, tokenizer)
    print(f"Dataset shape after filtering oversized responses: {merged_df.shape}")

    print("Tokenizing (context, response) pairs — context head-truncated, response always kept whole ...")
    processed_df = build_response_level_dataset(merged_df, tokenizer)

    train_full_df = processed_df[processed_df["split"] == "train"].reset_index(drop=True)
    test_df = processed_df[processed_df["split"] == "test"].reset_index(drop=True)
    test_df["split"] = "test"

    train_df, val_df = make_group_stratified_val_split(train_full_df)

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
        path = PROCESSED_DIR / f"response_level_{name}.parquet"
        df.to_parquet(path, index=False)
        saved.append((path, len(df)))

    print("\nSaved parquet files:")
    for path, n_rows in saved:
        print(f"  {path} — {n_rows} rows")


if __name__ == "__main__":
    main()
