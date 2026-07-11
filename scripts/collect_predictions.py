"""Backfill per-row test predictions (labels + probability scores) into the unified table.

Phase 4 compares every trained system on identical footing: one long-format table
(results/unified_predictions.parquet by default) with one row per (system, test row) and
columns system/row_index/source_id/task_type/split/y_true/y_pred/y_score, where y_score
is always P(hallucinated)-like (higher = more likely hallucinated). The systems live on
different Kaggle sessions and Hub repos, so this script runs ONE system at a time and
merges into the accumulating file — re-running a system replaces only that system's rows.

Modes:
    baseline    Model-free. Aggregates results/nli_scores_test.json to response level:
                y_score = max over sentences of max(contradiction, 1 - entailment) (the
                single-threshold reduction of ADR-007's disjunctive decision rule);
                y_pred = apply_thresholds at the tuned thresholds from
                results/baseline_nli_metrics.json. Note y_pred is the operational
                decision at (ent_thr, con_thr) and is not exactly a threshold on
                y_score — y_score exists for threshold-free PR curves.
    track_a     Batched Hub inference over data/processed/response_level_test.parquet
                (same pattern as scripts/analyze_track_a_predictions.py) capturing
                softmax P(hallucinated) per row.
    approach_1  Same, over data/processed/response_level_modernbert_test.parquet
                (regenerate via `python -m src.data.preprocess_modernbert` if absent).
    merge       Fold a unified-schema parquet produced elsewhere (e.g. downloaded from a
                Kaggle session) into the local accumulating file.

Examples:
    python scripts/collect_predictions.py baseline
    python scripts/collect_predictions.py track_a
    python scripts/collect_predictions.py approach_1
    python scripts/collect_predictions.py merge --input kaggle_output/unified_predictions.parquet
"""

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding  # noqa: E402

from src.evaluation.metrics import UNIFIED_COLUMNS, UNIFIED_PREDICTIONS_PATH  # noqa: E402
from src.models.nli_baseline import apply_thresholds  # noqa: E402

SYSTEM_BASELINE = "baseline_nli"
SYSTEM_TRACK_A = "track_a_deberta"
SYSTEM_APPROACH_1 = "approach_1_modernbert"

NLI_SCORES_PATH = Path("results/nli_scores_test.json")
BASELINE_METRICS_PATH = Path("results/baseline_nli_metrics.json")
HUB_DEFAULTS = {
    SYSTEM_TRACK_A: "hugoomezz/deberta-v3-ragtruth-hallucination",
    SYSTEM_APPROACH_1: "hugoomezz/deberta-v3-modernbert-ragtruth-hallucination",
}
TEST_PATH_DEFAULTS = {
    SYSTEM_TRACK_A: "data/processed/response_level_test.parquet",
    SYSTEM_APPROACH_1: "data/processed/response_level_modernbert_test.parquet",
}
BATCH_SIZE = 32


def baseline_y_score(sentence_scores: list) -> float:
    """Response-level hallucination score from per-sentence (entailment, contradiction) pairs.

    Per sentence: max(contradiction, 1 - entailment) — the one-dimensional reduction of
    ADR-007's disjunctive flag rule (not-supported iff contradiction >= con_thr OR
    entailment < ent_thr), so a single threshold sweep on this score walks the coupled
    rule family con_thr = t, ent_thr = 1 - t. Contradiction-priority is honored: a high
    contradiction dominates the max even when some context sentence strongly entails the
    claim. Response score is the max over sentences; empty responses score 0.0
    (vacuously-not-hallucinated, matching apply_thresholds).
    """
    if not sentence_scores:
        return 0.0
    return max(max(float(con), 1.0 - float(ent)) for ent, con in sentence_scores)


def build_prediction_rows(system, source_ids, task_types, y_true, y_pred, y_score) -> pd.DataFrame:
    """Assemble one system's rows in the unified schema, with positional row_index.

    row_index (0..n-1, input order) is the per-row key: source_id is NOT unique in
    RAGTruth (6 model responses per source). All systems iterate the same deterministic
    2700-row test set, so row_index also serves as the cross-system join key.
    """
    lengths = {len(source_ids), len(task_types), len(y_true), len(y_pred), len(y_score)}
    if len(lengths) != 1:
        raise ValueError(f"Input columns have mismatched lengths: {lengths}")
    n_rows = lengths.pop()
    return pd.DataFrame(
        {
            "system": [system] * n_rows,
            "row_index": range(n_rows),
            "source_id": source_ids,
            "task_type": task_types,
            "split": ["test"] * n_rows,
            "y_true": pd.array(y_true, dtype="int64"),
            "y_pred": pd.array(y_pred, dtype="int64"),
            "y_score": pd.array(y_score, dtype="float64"),
        }
    )[UNIFIED_COLUMNS]


def merge_predictions(existing: pd.DataFrame | None, new: pd.DataFrame) -> pd.DataFrame:
    """Replace rows for systems present in `new`; keep every other system's rows.

    This is what makes stage-wise collection safe: re-running one system (or folding in
    a file from another Kaggle session) never drops previously collected systems.
    """
    if existing is None:
        return new.reset_index(drop=True)
    replaced = set(new["system"].unique())
    kept = existing[~existing["system"].isin(replaced)]
    return pd.concat([kept, new], ignore_index=True)
