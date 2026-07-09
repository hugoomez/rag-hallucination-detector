"""Inference-only diagnostic: does context truncation correlate with Track A prediction errors?

Loads a fine-tuned Track A checkpoint from the Hub and runs a single forward pass (no training,
no gradients) over the untouched test parquet, then breaks accuracy/recall down by `was_truncated`
and `task_type`. Targets the low Summary recall (0.245) finding from
results/finetuned_track_a_metrics.json: is it concentrated in rows whose context was truncated
under ADR-004's context-only truncation, or spread evenly?

Meant to run on Kaggle (needs the fine-tuned model on the Hub), but since it's inference-only
over 2700 rows it should take a few minutes, not hours -- no GPU-scale training loop here.

Example:
    python scripts/analyze_track_a_predictions.py --hub_model_id <user>/deberta-v3-ragtruth-track-a
"""

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding  # noqa: E402

RESULTS_DIR = Path("results")
PREDICTIONS_PATH = RESULTS_DIR / "track_a_test_predictions.json"
TASK_TYPES = ["Summary", "QA", "Data2txt"]
BATCH_SIZE = 32


def parse_args() -> argparse.Namespace:
    """Only what's needed for inference: which model to load and which split to score."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hub_model_id", required=True, help="Fine-tuned Track A repo id on the Hub.")
    parser.add_argument("--test_path", default="data/processed/response_level_test.parquet")
    return parser.parse_args()


def load_test_df(path: str) -> pd.DataFrame:
    """Read the Phase 1 test parquet as-is; input_ids/attention_mask are already tokenized."""
    df = pd.read_parquet(path)
    df["input_ids"] = df["input_ids"].apply(lambda a: np.asarray(a).tolist())
    df["attention_mask"] = df["attention_mask"].apply(lambda a: np.asarray(a).tolist())
    return df


def run_inference(df: pd.DataFrame, model, collator, device: torch.device) -> list[int]:
    """Batched forward pass over the full test set; returns argmax predictions in row order."""
    examples = [{"input_ids": row.input_ids, "attention_mask": row.attention_mask} for row in df.itertuples()]
    loader = DataLoader(examples, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collator)

    model.eval()
    preds: list[int] = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            preds.extend(logits.argmax(dim=-1).cpu().tolist())
    return preds


def build_row_results(df: pd.DataFrame, preds: list[int]) -> list[dict]:
    """Per-row diagnostic record: id, task/truncation context, true vs predicted label."""
    rows = []
    for row, pred in zip(df.itertuples(), preds):
        true_label = int(row.label_response)
        rows.append(
            {
                "source_id": row.source_id,
                "task_type": row.task_type,
                "was_truncated": bool(row.was_truncated),
                "label_response": true_label,
                "predicted_label": int(pred),
                "correct": true_label == int(pred),
            }
        )
    return rows


def _accuracy(rows: list[dict]) -> float:
    return sum(r["correct"] for r in rows) / len(rows) if rows else float("nan")


def _recall_on_positives(rows: list[dict]) -> tuple[float, int]:
    """Recall on label_response=1 (hallucinated) rows only; also returns the row count."""
    positives = [r for r in rows if r["label_response"] == 1]
    if not positives:
        return float("nan"), 0
    hits = sum(1 for r in positives if r["predicted_label"] == 1)
    return hits / len(positives), len(positives)


def print_diagnostics(rows: list[dict]) -> None:
    """Overall + per-task_type accuracy by was_truncated, plus truncation-split recall on positives."""
    truncated = [r for r in rows if r["was_truncated"]]
    not_truncated = [r for r in rows if not r["was_truncated"]]

    print("\n=== Overall accuracy by was_truncated ===")
    print(f"  was_truncated=True  (n={len(truncated):4d}): accuracy={_accuracy(truncated):.4f}")
    print(f"  was_truncated=False (n={len(not_truncated):4d}): accuracy={_accuracy(not_truncated):.4f}")

    print("\n=== Accuracy by task_type x was_truncated ===")
    for task_type in TASK_TYPES:
        task_rows = [r for r in rows if r["task_type"] == task_type]
        task_truncated = [r for r in task_rows if r["was_truncated"]]
        task_not_truncated = [r for r in task_rows if not r["was_truncated"]]
        print(f"  {task_type}:")
        print(f"    was_truncated=True  (n={len(task_truncated):4d}): accuracy={_accuracy(task_truncated):.4f}")
        print(f"    was_truncated=False (n={len(task_not_truncated):4d}): accuracy={_accuracy(task_not_truncated):.4f}")

    print("\n=== Recall on label_response=1 (hallucinated) rows, by task_type x was_truncated ===")
    for task_type in TASK_TYPES:
        task_rows = [r for r in rows if r["task_type"] == task_type]
        task_truncated = [r for r in task_rows if r["was_truncated"]]
        task_not_truncated = [r for r in task_rows if not r["was_truncated"]]
        recall_t, n_t = _recall_on_positives(task_truncated)
        recall_nt, n_nt = _recall_on_positives(task_not_truncated)
        print(f"  {task_type}:")
        print(f"    was_truncated=True  (n_hallucinated={n_t:4d}): recall={recall_t:.4f}")
        print(f"    was_truncated=False (n_hallucinated={n_nt:4d}): recall={recall_nt:.4f}")


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(exist_ok=True)

    df = load_test_df(args.test_path)
    print(f"Loaded {len(df)} test rows from {args.test_path}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.hub_model_id)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(args.hub_model_id)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Loaded model {args.hub_model_id} on {device}", flush=True)

    preds = run_inference(df, model, collator, device)
    rows = build_row_results(df, preds)

    PREDICTIONS_PATH.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Saved per-row predictions -> {PREDICTIONS_PATH}", flush=True)

    print_diagnostics(rows)


if __name__ == "__main__":
    main()
