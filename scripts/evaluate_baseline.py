"""Full Step 2.4 evaluation of the zero-shot NLI baseline on RAGTruth val/test.

Pipeline:
  1. Reconstruct full-text val/test rows (source_info, response, label_response) from the
     raw merged dataset, using the processed parquets only for split membership.
  2. Score every row once with chunk_context + score_response (the GPU-bound step) and
     cache the raw per-sentence (max_entailment, max_contradiction) scores to JSON.
  3. Grid-search thresholds on the CACHED val scores (no model calls) and pick best F1.
  4. Apply the best thresholds to the CACHED test scores; report P/R/F1 plus two trivial
     baselines (always-hallucinated, random) and a per-task_type breakdown.

Run on a GPU (see scripts/KAGGLE_SETUP.md); on CPU it is impractically slow. Data must be
regenerated first (src/data/download.py then src/data/preprocess.py).

Note on reconstruction: the processed parquet has one row per response but no response-level
key (no id/model), and source_id maps to up to 6 responses, so a naive parquet->merged join
on source_id would 6x-explode. Instead we take the parquet's source_id SET per split and pull
the matching response-level rows straight from merged (which carries labels/response/
source_info). The val count may be +1 vs the parquet: the ADR-006 oversized-response row was
dropped from the parquet but is fine for the NLI baseline (it sentence-splits the response
rather than truncating it), so we keep it.
"""

import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import precision_recall_fscore_support  # noqa: E402

from src.data.context_chunking import chunk_context  # noqa: E402
from src.data.preprocess import load_merged_dataframe  # noqa: E402
from src.models.nli_baseline import DEFAULT_MODEL, NLIHallucinationDetector, apply_thresholds  # noqa: E402

RESULTS_DIR = Path("results")
VAL_PARQUET = "data/processed/response_level_val.parquet"
TEST_PARQUET = "data/processed/response_level_test.parquet"
TASK_TYPES = ["Summary", "QA", "Data2txt"]
ENT_THRS = [0.4, 0.5, 0.6, 0.7]
CON_THRS = [0.3, 0.4, 0.5]
RANDOM_SEED = 42


def reconstruct_split(merged: pd.DataFrame, parquet_path: str) -> tuple[pd.DataFrame, int]:
    """Pull response-level rows for a split from merged, keyed by the parquet's source_ids."""
    parquet_rows = pd.read_parquet(parquet_path, columns=["source_id"])
    source_ids = set(parquet_rows["source_id"].unique())
    rows = merged[merged["source_id"].isin(source_ids)][
        ["source_id", "task_type", "source_info", "response", "labels"]
    ].copy()
    rows["label_response"] = rows["labels"].apply(lambda labels: int(len(labels) > 0))
    return rows.reset_index(drop=True), len(parquet_rows)


def build_scores_cache(detector: NLIHallucinationDetector, rows: pd.DataFrame, split_name: str) -> list[dict]:
    """Score every row once (the expensive, GPU-bound pass) into a reusable cache."""
    cache: list[dict] = []
    total = len(rows)
    start = time.perf_counter()
    for i, row in enumerate(rows.itertuples(), start=1):
        context_chunks = chunk_context(row.task_type, row.source_info)
        sentence_scores = detector.score_response(context_chunks, row.response)
        cache.append(
            {
                "source_id": int(row.source_id),
                "task_type": row.task_type,
                "label_response": int(row.label_response),
                "sentence_scores": [[float(ent), float(con)] for ent, con in sentence_scores],
            }
        )
        if i % 100 == 0 or i == total:
            elapsed = time.perf_counter() - start
            print(f"[{split_name}] {i}/{total} rows scored | elapsed {elapsed:.1f}s", flush=True)
    return cache


def predict_from_cache(cache: list[dict], ent_thr: float, con_thr: float) -> tuple[list[int], list[int]]:
    """Turn cached scores into (y_true, y_pred) at the given thresholds — no model calls."""
    all_scores = [row["sentence_scores"] for row in cache]
    y_pred = [int(flag) for flag in apply_thresholds(all_scores, ent_thr, con_thr)]
    y_true = [row["label_response"] for row in cache]
    return y_true, y_pred


def prf(y_true: list[int], y_pred: list[int]) -> dict:
    """Binary precision/recall/F1 with hallucinated (label 1) as the positive class."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0
    )
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}


def grid_search(val_cache: list[dict]) -> tuple[list[dict], dict]:
    """Search ent_thr x con_thr on cached val scores; return the full grid and the best cell."""
    grid: list[dict] = []
    best: dict | None = None
    print("\n=== val threshold grid search (P / R / F1) ===", flush=True)
    for ent_thr in ENT_THRS:
        for con_thr in CON_THRS:
            y_true, y_pred = predict_from_cache(val_cache, ent_thr, con_thr)
            metrics = prf(y_true, y_pred)
            cell = {"ent_thr": ent_thr, "con_thr": con_thr, **metrics}
            grid.append(cell)
            print(
                f"  ent={ent_thr:.1f} con={con_thr:.1f}  "
                f"P={metrics['precision']:.3f} R={metrics['recall']:.3f} F1={metrics['f1']:.3f}",
                flush=True,
            )
            if best is None or cell["f1"] > best["f1"]:
                best = cell
    print(f"\nbest: ent_thr={best['ent_thr']} con_thr={best['con_thr']} F1={best['f1']:.3f}", flush=True)
    return grid, best


def per_task_breakdown(cache: list[dict], y_true: list[int], y_pred: list[int]) -> dict:
    """Test P/R/F1 split by task_type (chunking quality differs a lot across tasks)."""
    breakdown: dict = {}
    for task_type in TASK_TYPES:
        idx = [i for i, row in enumerate(cache) if row["task_type"] == task_type]
        if not idx:
            continue
        breakdown[task_type] = {
            "n": len(idx),
            **prf([y_true[i] for i in idx], [y_pred[i] for i in idx]),
        }
    return breakdown


def trivial_baselines(y_true: list[int]) -> dict:
    """Always-hallucinated (all 1s) and random (seeded) baselines for comparison."""
    n = len(y_true)
    always = [1] * n
    rng = np.random.default_rng(RANDOM_SEED)
    random_pred = rng.integers(0, 2, size=n).tolist()
    return {
        "always_hallucinated": prf(y_true, always),
        "random": prf(y_true, random_pred),
    }


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    merged = load_merged_dataframe()

    val_rows, val_parquet_n = reconstruct_split(merged, VAL_PARQUET)
    test_rows, test_parquet_n = reconstruct_split(merged, TEST_PARQUET)
    print(
        f"val: {len(val_rows)} rows (parquet {val_parquet_n}) | test: {len(test_rows)} rows (parquet {test_parquet_n})",
        flush=True,
    )

    detector = NLIHallucinationDetector.from_pretrained()
    print(f"model: {DEFAULT_MODEL} | device: {detector.device}", flush=True)

    # Expensive pass: score each split once, cache raw scores for reuse.
    val_cache = build_scores_cache(detector, val_rows, "val")
    (RESULTS_DIR / "nli_scores_val.json").write_text(json.dumps(val_cache), encoding="utf-8")
    test_cache = build_scores_cache(detector, test_rows, "test")
    (RESULTS_DIR / "nli_scores_test.json").write_text(json.dumps(test_cache), encoding="utf-8")

    # Threshold tuning on cached val scores only.
    grid, best = grid_search(val_cache)

    # Final test evaluation with the tuned thresholds (cached scores, no model calls).
    y_true_test, y_pred_test = predict_from_cache(test_cache, best["ent_thr"], best["con_thr"])
    test_metrics = prf(y_true_test, y_pred_test)
    baselines = trivial_baselines(y_true_test)
    per_task = per_task_breakdown(test_cache, y_true_test, y_pred_test)

    metrics = {
        "model_name": DEFAULT_MODEL,
        "best_thresholds": {"ent_thr": best["ent_thr"], "con_thr": best["con_thr"]},
        "counts": {"val_rows": len(val_cache), "test_rows": len(test_cache)},
        "val_grid_search": grid,
        "test": {
            "nli_baseline": test_metrics,
            "always_hallucinated": baselines["always_hallucinated"],
            "random": baselines["random"],
            "per_task_type": per_task,
        },
    }
    (RESULTS_DIR / "baseline_nli_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("\n=== TEST results (best thresholds) ===", flush=True)
    print(f"  NLI baseline        : {test_metrics}", flush=True)
    print(f"  always-hallucinated : {baselines['always_hallucinated']}", flush=True)
    print(f"  random              : {baselines['random']}", flush=True)
    print("  per task_type:", flush=True)
    for task_type, task_metrics in per_task.items():
        print(f"    {task_type:8s}: {task_metrics}", flush=True)
    print(
        "\nSaved: results/nli_scores_val.json, results/nli_scores_test.json, results/baseline_nli_metrics.json",
        flush=True,
    )


if __name__ == "__main__":
    main()
