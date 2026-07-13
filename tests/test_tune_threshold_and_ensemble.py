"""Offline unit tests for the pure logic of scripts/tune_threshold_and_ensemble.py.

The script lives outside the src package, so it is loaded by file path via importlib.
Only pure helpers and the alignment guard are tested; the full val/test sweeps run
operationally against results/unified_predictions.parquet.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "tune_threshold_and_ensemble.py"
_spec = importlib.util.spec_from_file_location("tune_threshold_and_ensemble", _SCRIPT)
te = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(te)


class TestMetricsAtThreshold:
    def test_recovers_expected_predictions(self):
        y_true = [0, 0, 1, 1]
        y_score = [0.1, 0.6, 0.4, 0.9]
        # thr=0.5 -> preds [0,1,0,1]: one FP (row1), one FN (row2), two correct
        m = te.metrics_at_threshold(y_true, y_score, 0.5)
        assert m["precision"] == pytest.approx(0.5)  # 1 TP / (1 TP + 1 FP)
        assert m["recall"] == pytest.approx(0.5)  # 1 TP / (1 TP + 1 FN)

    def test_threshold_boundary_is_inclusive(self):
        # score == threshold predicts positive (>=)
        m = te.metrics_at_threshold([1], [0.3], 0.3)
        assert m["recall"] == pytest.approx(1.0)


class TestWeightGrid:
    def test_all_vectors_sum_to_one(self):
        grid = te.weight_grid()
        assert all(abs(sum(w) - 1.0) < 1e-9 for w in grid)

    def test_count_and_contains_references(self):
        grid = te.weight_grid()
        assert len(grid) == 231  # simplex with denom=20: C(22, 2)
        # Track-B-heavy and Track-B-only are exactly on the 0.05 grid.
        assert any(np.allclose(w, (0.25, 0.25, 0.50)) for w in grid)
        assert (0.0, 0.0, 1.0) in grid


class TestEnsembleAlignmentGuard:
    def _row(self, system, row_index, y_true, y_score, split="val", task="QA"):
        return {
            "system": system,
            "row_index": row_index,
            "source_id": 1,
            "task_type": task,
            "split": split,
            "y_true": y_true,
            "y_pred": 0,
            "y_score": y_score,
        }

    def _frame(self, y_true_by_system):
        rows = []
        for system, labels in y_true_by_system.items():
            for idx, yt in enumerate(labels):
                rows.append(self._row(system, idx, yt, 0.5))
        return pd.DataFrame(rows)

    def test_aligned_systems_build_ok(self):
        df = self._frame({s: [0, 1] for s in te.ENSEMBLE_SYSTEMS})
        scores, y_true, tasks = te.build_ensemble_matrix(df, "val")
        assert scores.shape == (2, 3)
        assert list(y_true) == [0, 1]

    def test_mismatched_y_true_raises(self):
        labels = {te.ENSEMBLE_SYSTEMS[0]: [0, 1], te.ENSEMBLE_SYSTEMS[1]: [0, 1], te.ENSEMBLE_SYSTEMS[2]: [1, 1]}
        df = self._frame(labels)
        with pytest.raises(ValueError, match="misaligned"):
            te.build_ensemble_matrix(df, "val")

    def test_missing_rows_raise(self):
        labels = {te.ENSEMBLE_SYSTEMS[0]: [0, 1], te.ENSEMBLE_SYSTEMS[1]: [0, 1], te.ENSEMBLE_SYSTEMS[2]: [0]}
        df = self._frame(labels)
        with pytest.raises(ValueError, match="gaps"):
            te.build_ensemble_matrix(df, "val")


class TestPerTaskMetrics:
    def test_splits_by_task(self):
        y_true = [1, 1, 0]
        y_pred = [1, 0, 0]
        tasks = ["Summary", "Summary", "QA"]
        out = te.per_task_metrics(y_true, y_pred, tasks)
        assert out["Summary"]["n"] == 2
        assert out["Summary"]["recall"] == pytest.approx(0.5)
        assert out["QA"]["n"] == 1
