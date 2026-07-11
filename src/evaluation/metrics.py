"""Response-level evaluation metrics over the unified predictions table.

Every trained system (zero-shot NLI baseline, Track A DeBERTa, Approach 1 ModernBERT,
and eventually Track B) backfills per-row test predictions into one long-format table
(results/unified_predictions.parquet, written by scripts/collect_predictions.py) with a
shared schema. This module is the read side: filter that table to one system and compute
the Phase 4 response-level metrics, so building the full comparison becomes a loop over
system names (see comparison_table).

Schema note: source_id is NOT unique per row (RAGTruth has 6 model responses per source),
so row_index — the positional index within a system's test set — is the per-row key. All
current systems share the same 2700-row test composition and order, making row_index a
valid cross-system join key for paired analyses.
"""

from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
)

UNIFIED_PREDICTIONS_PATH = Path("results/unified_predictions.parquet")

UNIFIED_COLUMNS = ["system", "row_index", "source_id", "task_type", "split", "y_true", "y_pred", "y_score"]

TARGET_NAMES = ["not_hallucinated", "hallucinated"]


def response_level_metrics(y_true, y_pred) -> dict:
    """Binary response-level metrics with hallucinated (1) as the positive class.

    Returns a json-serializable dict: n, precision, recall, f1, confusion_matrix
    (2x2 nested list, rows = true 0/1, cols = predicted 0/1), classification_report
    (nested dict). zero_division=0 so degenerate prediction vectors (e.g. a system
    that never predicts positive) report 0.0 rather than raising or returning NaN.
    """
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0
    )
    return {
        "n": int(len(y_true)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, labels=[0, 1], target_names=TARGET_NAMES, output_dict=True, zero_division=0
        ),
    }


def load_predictions(path: Path | str = UNIFIED_PREDICTIONS_PATH) -> pd.DataFrame:
    """Read the unified predictions parquet, validating the required columns exist."""
    df = pd.read_parquet(path)
    missing = [column for column in UNIFIED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Unified predictions file {path} is missing columns {missing}")
    return df


def system_predictions(df: pd.DataFrame, system: str) -> pd.DataFrame:
    """Filter the unified table to one system's rows; error if the system is absent."""
    subset = df[df["system"] == system]
    if subset.empty:
        available = sorted(df["system"].unique())
        raise ValueError(f"No rows for system {system!r}; available systems: {available}")
    return subset


def metrics_for_system(df: pd.DataFrame, system: str) -> dict:
    """Response-level metrics for one system out of the unified table."""
    subset = system_predictions(df, system)
    return response_level_metrics(subset["y_true"].to_numpy(), subset["y_pred"].to_numpy())


def pr_curve_for_system(df: pd.DataFrame, system: str) -> dict:
    """Precision-recall curve data (threshold-free) for one system, from y_score.

    Returns sklearn's arrays as-is: precision and recall have len(thresholds) + 1
    entries (the final (1, 0) point has no threshold). y_score semantics: higher =
    more likely hallucinated.
    """
    subset = system_predictions(df, system)
    precision, recall, thresholds = precision_recall_curve(
        subset["y_true"].to_numpy(), subset["y_score"].to_numpy(), pos_label=1
    )
    return {"precision": precision, "recall": recall, "thresholds": thresholds}


def comparison_table(df: pd.DataFrame, systems: list[str] | None = None) -> pd.DataFrame:
    """One summary row (n/precision/recall/f1) per system — the Phase 4 comparison loop.

    systems=None uses every system present, in first-appearance order (so the table
    reads in collection order); pass an explicit list to control ordering.
    """
    if systems is None:
        systems = list(dict.fromkeys(df["system"]))
    rows = []
    for system in systems:
        system_metrics = metrics_for_system(df, system)
        rows.append(
            {
                "system": system,
                "n": system_metrics["n"],
                "precision": system_metrics["precision"],
                "recall": system_metrics["recall"],
                "f1": system_metrics["f1"],
            }
        )
    return pd.DataFrame(rows)
