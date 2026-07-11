"""Offline unit tests for the pure (model-free) parts of scripts/collect_predictions.py.

The script lives outside the src package, so it is loaded by file path via importlib.
Only the pure helpers are tested; the inference paths require Hub models and are
verified operationally against already-published metrics (see the plan doc).
"""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from src.evaluation.metrics import UNIFIED_COLUMNS

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "collect_predictions.py"
_spec = importlib.util.spec_from_file_location("collect_predictions", _SCRIPT)
cp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cp)


class TestBaselineYScore:
    def test_high_contradiction_dominates(self):
        # ADR-007 case: strongly entailed by one context sentence, contradicted by another.
        assert cp.baseline_y_score([[0.95, 0.9]]) == pytest.approx(0.9)

    def test_unverifiable_sentence_scores_high(self):
        # low entailment, low contradiction -> 1 - ent drives the score
        assert cp.baseline_y_score([[0.1, 0.05]]) == pytest.approx(0.9)

    def test_supported_sentence_scores_low(self):
        assert cp.baseline_y_score([[0.98, 0.01]]) == pytest.approx(0.02)

    def test_max_over_sentences(self):
        scores = [[0.98, 0.01], [0.2, 0.1], [0.9, 0.7]]
        assert cp.baseline_y_score(scores) == pytest.approx(0.8)  # 1 - 0.2

    def test_empty_sentences_is_zero(self):
        assert cp.baseline_y_score([]) == 0.0


class TestBuildPredictionRows:
    def test_schema_and_row_index(self):
        df = cp.build_prediction_rows(
            system="sys_x",
            source_ids=[10, 10, 11],
            task_types=["QA", "QA", "Summary"],
            y_true=[0, 1, 1],
            y_pred=[0, 1, 0],
            y_score=[0.1, 0.9, 0.4],
        )
        assert list(df.columns) == UNIFIED_COLUMNS
        assert list(df["row_index"]) == [0, 1, 2]
        assert set(df["split"]) == {"test"}
        assert set(df["system"]) == {"sys_x"}
        assert df["y_true"].dtype.kind == "i"
        assert df["y_pred"].dtype.kind == "i"
        assert df["y_score"].dtype.kind == "f"

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            cp.build_prediction_rows("sys_x", [1, 2], ["QA"], [0], [0], [0.5])


class TestMergePredictions:
    def _rows(self, system, y_pred):
        return cp.build_prediction_rows(system, [10], ["QA"], [1], [y_pred], [0.5])

    def test_merge_into_empty(self):
        merged = cp.merge_predictions(None, self._rows("sys_a", 1))
        assert len(merged) == 1

    def test_rerun_replaces_same_system_only(self):
        existing = pd.concat([self._rows("sys_a", 0), self._rows("sys_b", 1)], ignore_index=True)
        merged = cp.merge_predictions(existing, self._rows("sys_a", 1))
        assert len(merged) == 2
        assert merged.loc[merged["system"] == "sys_a", "y_pred"].item() == 1  # replaced
        assert merged.loc[merged["system"] == "sys_b", "y_pred"].item() == 1  # untouched

    def test_new_system_appends(self):
        merged = cp.merge_predictions(self._rows("sys_a", 0), self._rows("sys_c", 1))
        assert sorted(merged["system"].unique()) == ["sys_a", "sys_c"]
