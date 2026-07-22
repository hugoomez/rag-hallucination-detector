"""Unit tests for scripts/aggregate_seeds.py pure helpers: descriptive per-seed summary and
the matched (paired) large-minus-base deltas. No model, no data files -- all fast CPU logic.

Reporting is deliberately restricted to raw values / mean / min-max range / per-seed deltas;
these tests also lock in that no p-value or significance field is produced."""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.aggregate_seeds import (  # noqa: E402
    build_report,
    extract_metrics,
    paired_deltas,
    summarize_model,
    summary_stats,
)


def _report(seed: int, model_name: str = "m", **metric_overrides) -> dict:
    """Minimal metrics-JSON dict: every METRIC_PATHS leaf defaults to 0.0, override by dotted key.

    Keys accepted mirror the flat metric names, e.g. _report(42, **{"response.f1": 0.76}).
    """
    o = metric_overrides
    return {
        "model_name": model_name,
        "hyperparameters": {"seed": seed},
        "test": {
            "response_level_derived": {
                "precision": o.get("response.precision", 0.0),
                "recall": o.get("response.recall", 0.0),
                "f1": o.get("response.f1", 0.0),
                "accuracy": o.get("response.accuracy", 0.0),
            },
            "span_char_level": {
                "precision": o.get("span_char.precision", 0.0),
                "recall": o.get("span_char.recall", 0.0),
                "f1": o.get("span_char.f1", 0.0),
            },
            "span_exact_match": {"f1": o.get("span_exact.f1", 0.0)},
            "per_task_type": {
                "Summary": {"f1": o.get("per_task.Summary.f1", 0.0)},
                "QA": {"f1": o.get("per_task.QA.f1", 0.0)},
                "Data2txt": {"f1": o.get("per_task.Data2txt.f1", 0.0)},
            },
        },
    }


class TestSummaryStats:
    def test_mean_min_max_range(self):
        s = summary_stats([1.0, 2.0, 3.0])
        assert s["n"] == 3
        assert s["values"] == [1.0, 2.0, 3.0]
        assert abs(s["mean"] - 2.0) < 1e-9
        assert abs(s["min"] - 1.0) < 1e-9
        assert abs(s["max"] - 3.0) < 1e-9
        assert abs(s["range"] - 2.0) < 1e-9

    def test_unordered_input(self):
        s = summary_stats([0.74, 0.70, 0.72])
        assert abs(s["mean"] - 0.72) < 1e-9
        assert abs(s["min"] - 0.70) < 1e-9
        assert abs(s["max"] - 0.74) < 1e-9
        assert abs(s["range"] - 0.04) < 1e-9

    def test_single_value_zero_range(self):
        s = summary_stats([5.0])
        assert s["n"] == 1
        assert abs(s["mean"] - 5.0) < 1e-9
        assert abs(s["range"] - 0.0) < 1e-9

    def test_no_std_or_significance_fields(self):
        # Restricted reporting: only descriptive spread, never std / ci / p-value.
        s = summary_stats([1.0, 2.0, 3.0])
        assert set(s) == {"n", "values", "mean", "min", "max", "range"}

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            summary_stats([])


class TestExtractMetrics:
    def test_flattens_nested_test_block(self):
        rep = _report(42, **{"response.f1": 0.76, "span_char.f1": 0.53, "per_task.QA.f1": 0.67})
        m = extract_metrics(rep)
        assert abs(m["response.f1"] - 0.76) < 1e-9
        assert abs(m["span_char.f1"] - 0.53) < 1e-9
        assert abs(m["per_task.QA.f1"] - 0.67) < 1e-9
        assert abs(m["span_exact.f1"] - 0.0) < 1e-9

    def test_missing_metric_path_raises(self):
        rep = _report(42)
        del rep["test"]["span_char_level"]
        with pytest.raises(KeyError):
            extract_metrics(rep)


class TestSummarizeModel:
    def test_orders_by_seed_and_summarizes(self):
        reports = [
            _report(456, **{"response.f1": 0.74}),
            _report(42, **{"response.f1": 0.70}),
            _report(123, **{"response.f1": 0.72}),
        ]
        summ = summarize_model(reports)
        assert summ["seeds"] == [42, 123, 456]  # sorted, not input order
        f1 = summ["per_metric"]["response.f1"]
        assert f1["values"] == [0.70, 0.72, 0.74]  # seed order
        assert abs(f1["mean"] - 0.72) < 1e-9
        assert abs(f1["range"] - 0.04) < 1e-9

    def test_duplicate_seed_raises(self):
        with pytest.raises(ValueError):
            summarize_model([_report(42), _report(42)])


class TestPairedDeltas:
    def test_per_seed_deltas_and_mean(self):
        base = [
            _report(42, **{"response.f1": 0.70}),
            _report(123, **{"response.f1": 0.72}),
            _report(456, **{"response.f1": 0.74}),
        ]
        large = [
            _report(42, **{"response.f1": 0.79}),
            _report(123, **{"response.f1": 0.80}),
            _report(456, **{"response.f1": 0.78}),
        ]
        paired = paired_deltas(base, large)
        assert paired["seeds"] == [42, 123, 456]
        d = paired["per_metric"]["response.f1"]
        # deltas: 0.09, 0.08, 0.04
        assert all(abs(a - b) < 1e-9 for a, b in zip(d["per_seed_delta"], [0.09, 0.08, 0.04]))
        assert abs(d["mean_delta"] - 0.07) < 1e-9
        assert abs(d["min_delta"] - 0.04) < 1e-9
        assert abs(d["max_delta"] - 0.09) < 1e-9
        assert d["n_large_higher"] == 3
        assert d["n"] == 3

    def test_pairs_by_seed_not_position(self):
        # Same seeds, different list orders -> deltas still matched per seed.
        base = [_report(42, **{"response.f1": 0.70}), _report(123, **{"response.f1": 0.72})]
        large = [_report(123, **{"response.f1": 0.80}), _report(42, **{"response.f1": 0.79})]
        d = paired_deltas(base, large)["per_metric"]["response.f1"]
        # seed 42: 0.79-0.70=0.09 ; seed 123: 0.80-0.72=0.08 (seed-sorted order)
        assert all(abs(a - b) < 1e-9 for a, b in zip(d["per_seed_delta"], [0.09, 0.08]))

    def test_mixed_direction_counts_wins_not_significance(self):
        base = [
            _report(42, **{"span_char.f1": 0.50}),
            _report(123, **{"span_char.f1": 0.50}),
            _report(456, **{"span_char.f1": 0.50}),
        ]
        large = [
            _report(42, **{"span_char.f1": 0.55}),
            _report(123, **{"span_char.f1": 0.49}),
            _report(456, **{"span_char.f1": 0.52}),
        ]
        d = paired_deltas(base, large)["per_metric"]["span_char.f1"]
        assert d["n_large_higher"] == 2  # +0.05, -0.01, +0.02
        assert set(d) == {"n", "per_seed_delta", "mean_delta", "min_delta", "max_delta", "n_large_higher"}

    def test_mismatched_seed_sets_raise(self):
        base = [_report(42), _report(123), _report(456)]
        large = [_report(42), _report(123), _report(999)]
        with pytest.raises(ValueError):
            paired_deltas(base, large)


class TestBuildReport:
    def test_assembles_and_carries_caveat(self):
        base = [_report(42, model_name="base"), _report(123, model_name="base")]
        large = [_report(42, model_name="large"), _report(123, model_name="large")]
        report = build_report(base, large)
        assert report["seeds"] == [42, 123]
        assert report["n_seeds"] == 2
        assert report["base"]["model_name"] == "base"
        assert report["large"]["model_name"] == "large"
        # No p-value / significance leakage anywhere in the top-level shape.
        assert "n=2 seeds" in report["caveat"]
        assert "significance" in report["caveat"]
