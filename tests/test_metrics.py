"""Offline unit tests for the unified-predictions metrics module.

All data is synthetic; no result files are read and nothing touches the network.
"""

import numpy as np
import pandas as pd
import pytest

from src.evaluation import metrics
from src.evaluation.metrics import (
    UNIFIED_COLUMNS,
    comparison_table,
    load_predictions,
    metrics_for_system,
    pr_curve_for_system,
    response_level_metrics,
    system_predictions,
)


class TestResponseLevelMetrics:
    def test_perfect_predictions(self):
        result = response_level_metrics([0, 1, 0, 1], [0, 1, 0, 1])
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0
        assert result["f1"] == 1.0
        assert result["n"] == 4
        assert result["confusion_matrix"] == [[2, 0], [0, 2]]

    def test_known_mixed_case(self):
        # true: 1 1 1 0 0 0 ; pred: 1 1 0 1 0 0
        # TP=2 FN=1 FP=1 TN=2 -> precision=2/3 recall=2/3 f1=2/3
        result = response_level_metrics([1, 1, 1, 0, 0, 0], [1, 1, 0, 1, 0, 0])
        assert result["precision"] == pytest.approx(2 / 3)
        assert result["recall"] == pytest.approx(2 / 3)
        assert result["f1"] == pytest.approx(2 / 3)
        assert result["confusion_matrix"] == [[2, 1], [1, 2]]

    def test_classification_report_is_dict_with_both_classes(self):
        result = response_level_metrics([0, 1], [0, 1])
        report = result["classification_report"]
        assert isinstance(report, dict)
        assert "hallucinated" in report
        assert "not_hallucinated" in report

    def test_no_predicted_positives_yields_zero_not_nan(self):
        result = response_level_metrics([1, 0], [0, 0])
        assert result["precision"] == 0.0
        assert result["f1"] == 0.0

    def test_accepts_numpy_arrays(self):
        result = response_level_metrics(np.array([0, 1]), np.array([1, 1]))
        assert result["recall"] == 1.0
        # json-serializable output: plain python floats/ints/lists
        assert isinstance(result["precision"], float)
        assert isinstance(result["confusion_matrix"], list)


def make_unified_df() -> pd.DataFrame:
    """Two synthetic systems over the same 4 test rows.

    sys_perfect predicts everything right; sys_never never predicts positive.
    """
    base = {
        "row_index": [0, 1, 2, 3],
        "source_id": [10, 10, 11, 11],
        "task_type": ["QA", "QA", "Summary", "Summary"],
        "split": ["test"] * 4,
        "y_true": [0, 1, 0, 1],
    }
    perfect = pd.DataFrame({"system": ["sys_perfect"] * 4, **base, "y_pred": [0, 1, 0, 1], "y_score": [0.1, 0.9, 0.2, 0.8]})
    never = pd.DataFrame({"system": ["sys_never"] * 4, **base, "y_pred": [0, 0, 0, 0], "y_score": [0.4, 0.3, 0.2, 0.1]})
    return pd.concat([perfect, never], ignore_index=True)[UNIFIED_COLUMNS]


class TestSystemSlicing:
    def test_system_predictions_filters(self):
        df = make_unified_df()
        subset = system_predictions(df, "sys_perfect")
        assert len(subset) == 4
        assert set(subset["system"]) == {"sys_perfect"}

    def test_unknown_system_raises_with_available_names(self):
        with pytest.raises(ValueError, match="sys_never"):
            system_predictions(make_unified_df(), "nonexistent")

    def test_metrics_for_system(self):
        df = make_unified_df()
        assert metrics_for_system(df, "sys_perfect")["f1"] == 1.0
        assert metrics_for_system(df, "sys_never")["recall"] == 0.0


class TestPrCurve:
    def test_perfectly_separable_scores_reach_precision_1(self):
        df = make_unified_df()
        curve = pr_curve_for_system(df, "sys_perfect")
        assert set(curve) == {"precision", "recall", "thresholds"}
        # sklearn invariant: len(thresholds) == len(precision) - 1
        assert len(curve["thresholds"]) == len(curve["precision"]) - 1
        # sys_perfect scores separate the classes, so some threshold hits P=1, R=1
        found = any(p == 1.0 and r == 1.0 for p, r in zip(curve["precision"], curve["recall"]))
        assert found

    def test_uses_y_score_not_y_pred(self):
        # sys_never has all y_pred=0 but ANTI-correlated scores; the curve must
        # come from y_score (its top-scored row is a true negative -> precision
        # at the highest threshold is 0), proving y_pred is not involved.
        curve = pr_curve_for_system(make_unified_df(), "sys_never")
        assert curve["precision"][-2] == 0.0


class TestComparisonTable:
    def test_one_row_per_system(self):
        table = comparison_table(make_unified_df())
        assert list(table.columns) == ["system", "n", "precision", "recall", "f1"]
        assert sorted(table["system"]) == ["sys_never", "sys_perfect"]
        assert (table["n"] == 4).all()

    def test_explicit_system_order_is_respected(self):
        table = comparison_table(make_unified_df(), systems=["sys_never", "sys_perfect"])
        assert list(table["system"]) == ["sys_never", "sys_perfect"]


class TestLoadPredictions:
    def test_round_trip_and_validation(self, tmp_path):
        path = tmp_path / "unified.parquet"
        make_unified_df().to_parquet(path, index=False)
        df = load_predictions(path)
        assert len(df) == 8

    def test_missing_columns_raise(self, tmp_path):
        path = tmp_path / "bad.parquet"
        make_unified_df().drop(columns=["y_score"]).to_parquet(path, index=False)
        with pytest.raises(ValueError, match="y_score"):
            load_predictions(path)
