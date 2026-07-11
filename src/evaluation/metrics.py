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
