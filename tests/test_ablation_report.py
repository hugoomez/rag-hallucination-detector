"""Unit tests for scripts/ablation_report.py pure helpers: interval subtraction and the
pre-registered decision rule. No model, no data files -- all fast CPU logic."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.ablation_report import (  # noqa: E402
    char_mass,
    char_recall,
    decision_rule,
    subtract_intervals,
    total_overlap,
)


class TestSubtractIntervals:
    def test_no_holes_returns_input(self):
        assert subtract_intervals([(0, 10)], []) == [(0, 10)]

    def test_hole_fully_covers_interval(self):
        assert subtract_intervals([(2, 8)], [(0, 10)]) == []

    def test_hole_in_middle_splits(self):
        assert subtract_intervals([(0, 10)], [(4, 6)]) == [(0, 4), (6, 10)]

    def test_hole_at_start_trims(self):
        assert subtract_intervals([(0, 10)], [(0, 4)]) == [(4, 10)]

    def test_hole_at_end_trims(self):
        assert subtract_intervals([(0, 10)], [(6, 10)]) == [(0, 6)]

    def test_non_overlapping_hole_leaves_interval(self):
        assert subtract_intervals([(0, 5)], [(6, 10)]) == [(0, 5)]

    def test_touching_hole_is_no_op(self):
        # holes are half-open; a hole starting exactly at the interval end removes nothing.
        assert subtract_intervals([(0, 5)], [(5, 10)]) == [(0, 5)]

    def test_multiple_holes_punch_multiple_gaps(self):
        assert subtract_intervals([(0, 20)], [(2, 4), (10, 12)]) == [(0, 2), (4, 10), (12, 20)]

    def test_zero_width_remnants_dropped(self):
        # Hole (0,5) over (0,5) leaves nothing; over-punching never yields (x,x).
        assert subtract_intervals([(0, 5)], [(0, 5)]) == []

    def test_multiple_intervals_processed_independently(self):
        assert subtract_intervals([(0, 10), (20, 30)], [(5, 25)]) == [(0, 5), (25, 30)]

    def test_overlapping_holes_handled(self):
        assert subtract_intervals([(0, 10)], [(2, 6), (4, 8)]) == [(0, 2), (8, 10)]


class TestOverlapHelpers:
    def test_total_overlap(self):
        assert total_overlap([(0, 10)], [(5, 15)]) == 5
        assert total_overlap([(0, 5)], [(10, 15)]) == 0

    def test_char_mass(self):
        assert char_mass([(0, 10), (20, 25)]) == 15

    def test_char_recall_micro_aggregates(self):
        # gold mass = 10 + 10 = 20; overlap = 5 (row0) + 10 (row1) = 15 -> recall 0.75.
        pred = [[(0, 5)], [(0, 10)]]
        gold = [[(0, 10)], [(0, 10)]]
        result = char_recall(pred, gold)
        assert result["gold_char_mass"] == 20
        assert result["overlap_char_mass"] == 15
        assert abs(result["recall"] - 0.75) < 1e-9

    def test_char_recall_empty_gold_is_zero(self):
        assert char_recall([[(0, 5)]], [[]])["recall"] == 0.0


def _block(clean_f1: float, response_f1: float, span_recall: float) -> dict:
    return {
        "official": {
            "span_char_level": {"precision": 0.0, "recall": span_recall, "f1": 0.0},
            "response_level": {"f1": response_f1},
        },
        "clean_span": {"f1": clean_f1},
    }


class TestDecisionRule:
    def test_all_conditions_pass_adopts_c(self):
        b = _block(clean_f1=0.50, response_f1=0.76, span_recall=0.50)
        c = _block(clean_f1=0.55, response_f1=0.77, span_recall=0.48)  # drop 0.02 <= share 0.135
        d = decision_rule(b, c, noisy_char_mass_share=0.135)
        assert d["adopt_c_over_b"] is True
        assert d["clean_span_f1_improved"] is True
        assert d["response_f1_improved"] is True
        assert d["recall_drop_within_noisy_share"] is True

    def test_clean_f1_not_improved_fails(self):
        b = _block(0.55, 0.76, 0.50)
        c = _block(0.55, 0.77, 0.49)  # equal clean F1 -> not strictly greater
        assert decision_rule(b, c, 0.135)["adopt_c_over_b"] is False

    def test_response_f1_not_improved_fails(self):
        b = _block(0.50, 0.77, 0.50)
        c = _block(0.55, 0.77, 0.49)  # equal response F1
        assert decision_rule(b, c, 0.135)["adopt_c_over_b"] is False

    def test_recall_drop_exceeds_share_fails(self):
        b = _block(0.50, 0.76, 0.50)
        c = _block(0.55, 0.77, 0.30)  # drop 0.20 > share 0.135
        d = decision_rule(b, c, 0.135)
        assert d["recall_drop_within_noisy_share"] is False
        assert d["adopt_c_over_b"] is False

    def test_recall_increase_is_always_within_share(self):
        b = _block(0.50, 0.76, 0.50)
        c = _block(0.55, 0.77, 0.55)  # negative drop
        assert decision_rule(b, c, 0.135)["adopt_c_over_b"] is True
