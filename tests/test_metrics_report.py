"""Unit tests for build_metrics_report / build_prediction_records's gold_token_spans.

build_metrics_report was extracted out of build_token_test_report so it can be shared by
scripts/dump_token_predictions.py (arm a / any already-trained checkpoint, no trainer or
training hyperparameters available) without duplicating the test-block assembly logic.
These tests exercise it directly, with a small synthetic 3-row split (no model, no data
files), and lock in that its output is a drop-in match for scripts/aggregate_seeds.py's
expected metrics-JSON schema.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.aggregate_seeds import extract_metrics, run_seed  # noqa: E402
from src.models.train_token_level import build_metrics_report, build_prediction_records  # noqa: E402

# 3 rows, one per TASK_TYPES entry, each 5 real response tokens + 1 trailing ignore (-100)
# position, same offset layout as tests/test_span_metrics.py.
STARTS = [0, 5, 11, 16, 20]
ENDS = [5, 10, 16, 20, 25]

# Row "Summary": no hallucination anywhere, predicted correctly.
LABELS_SUMMARY = np.array([0, 0, 0, 0, 0, -100])
PREDS_SUMMARY = np.array([0, 0, 0, 0, 0, -100])

# Row "QA": tokens 2-3 hallucinated (merges to char span (11, 20)), predicted correctly.
LABELS_QA = np.array([0, 0, 1, 1, 0, -100])
PREDS_QA = np.array([0, 0, 1, 1, 0, -100])

# Row "Data2txt": tokens 1-2 hallucinated (merges to char span (5, 16)); model only catches
# token 2 -> a partial (non-exact) prediction, so this row also exercises a genuine miss.
LABELS_D2T = np.array([0, 1, 1, 0, 0, -100])
PREDS_D2T = np.array([0, 0, 1, 0, 0, -100])


def _make_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source_id": [1, 2, 3],
            "response_id": ["r1", "r2", "r3"],
            "task_type": ["Summary", "QA", "Data2txt"],
            "token_starts": [STARTS, STARTS, STARTS],
            "token_ends": [ENDS, ENDS, ENDS],
            # Normalized gold char spans mirror the token-aligned merge exactly here.
            "gold_starts": [[], [11], [5]],
            "gold_ends": [[], [20], [16]],
        }
    )


LABELS = [LABELS_SUMMARY, LABELS_QA, LABELS_D2T]
PREDS = [PREDS_SUMMARY, PREDS_QA, PREDS_D2T]


class TestBuildMetricsReport:
    def test_output_matches_aggregate_seeds_schema(self):
        df = _make_df()
        report = {
            "hyperparameters": {"seed": 123},
            "test": build_metrics_report(LABELS, PREDS, df, seed=123),
        }
        # run_seed / extract_metrics are the exact functions aggregate_seeds.py calls on
        # every --base/--large input; if these don't raise, the schema is a real match.
        assert run_seed(report) == 123
        metrics = extract_metrics(report)
        assert set(metrics) == {
            "response.precision",
            "response.recall",
            "response.f1",
            "response.accuracy",
            "span_char.precision",
            "span_char.recall",
            "span_char.f1",
            "span_exact.f1",
            "per_task.Summary.f1",
            "per_task.QA.f1",
            "per_task.Data2txt.f1",
        }
        assert all(0.0 <= v <= 1.0 for v in metrics.values())

    def test_response_level_derived_reflects_any_positive_token_rule(self):
        df = _make_df()
        test_block = build_metrics_report(LABELS, PREDS, df, seed=0)
        # Summary: true+pred both all-supported -> both rows correct except D2T is TP too
        # (label has a hallucination, pred does too) => 3/3 response predictions correct.
        assert test_block["response_level_derived"]["accuracy"] == pytest.approx(1.0)

    def test_span_exact_match_penalizes_the_partial_prediction(self):
        df = _make_df()
        test_block = build_metrics_report(LABELS, PREDS, df, seed=0)
        # QA is an exact match ((11,20) predicted and gold-token-aligned); Data2txt's
        # predicted (11,16) != gold-token-aligned (5,16) -- not an exact match.
        assert test_block["span_exact_match"]["precision"] == pytest.approx(0.5)

    def test_seed_changes_only_the_random_baseline(self):
        df = _make_df()
        block_a = build_metrics_report(LABELS, PREDS, df, seed=1)
        block_b = build_metrics_report(LABELS, PREDS, df, seed=2)
        # Non-random blocks are seed-independent.
        assert block_a["span_char_level"] == block_b["span_char_level"]
        assert block_a["response_level_derived"] == block_b["response_level_derived"]
        assert block_a["always_hallucinated"] == block_b["always_hallucinated"]


class TestBuildPredictionRecordsGoldTokenSpans:
    def test_gold_token_spans_present_and_token_aligned(self):
        df = _make_df()
        records = build_prediction_records(LABELS, PREDS, df)
        by_task = {r["task_type"]: r for r in records}
        assert by_task["Summary"]["gold_token_spans"] == []
        assert by_task["QA"]["gold_token_spans"] == [[11, 20]]
        assert by_task["Data2txt"]["gold_token_spans"] == [[5, 16]]

    def test_gold_token_spans_can_differ_from_char_level_gold_spans(self):
        # gold_spans (normalized/annotated) and gold_token_spans (token-merge-aligned) are
        # independent fields; this fixture happens to make them equal, but the record must
        # carry both under distinct keys rather than collapsing them.
        df = _make_df()
        records = build_prediction_records(LABELS, PREDS, df)
        for r in records:
            assert "gold_spans" in r
            assert "gold_token_spans" in r
