"""Unit tests for Track B's span reconstruction and span-level metrics (pure functions).

These exercise train_token_level.py's post-processing/evaluation logic directly, with no
model or GPU: merging binary token predictions into character spans (LettuceDetect-style)
and the character-overlap / exact-match span metrics.
"""

import pytest

from src.models.train import compute_class_weights
from src.models.train_token_level import char_span_prf, exact_span_prf, merge_predicted_spans

# A tiny synthetic layout used across merge tests. Real labels on 5 "response" tokens,
# with character offsets including a whitespace gap between tokens 1 and 2:
#   token:   0        1        2        3        4
#   chars:   (0,5)    (5,10)   (11,16)  (16,20)  (20,25)
# Position 5 is trailing padding/special (-100).
LABELS = [0, 0, 0, 0, 0, -100]
STARTS = [0, 5, 11, 16, 20]
ENDS = [5, 10, 16, 20, 25]


class TestMergePredictedSpans:
    def test_single_positive_run_merges_into_one_span(self):
        preds = [0, 1, 1, 0, 0, 1]  # the final 1 sits on a -100 position: ignored
        assert merge_predicted_spans(preds, LABELS, STARTS, ENDS) == [(5, 16)]

    def test_runs_split_by_a_negative_token_become_separate_spans(self):
        preds = [1, 0, 1, 1, 0, 0]
        assert merge_predicted_spans(preds, LABELS, STARTS, ENDS) == [(0, 5), (11, 20)]

    def test_whitespace_gap_between_consecutive_positive_tokens_is_absorbed(self):
        # Tokens 1 (5,10) and 2 (11,16) are consecutive in token order but leave a
        # character gap (10,11); the merged span must cover it, LettuceDetect-style.
        preds = [0, 1, 1, 0, 0, 0]
        assert merge_predicted_spans(preds, LABELS, STARTS, ENDS) == [(5, 16)]

    def test_positive_run_ending_at_last_response_token_is_flushed(self):
        preds = [0, 0, 0, 1, 1, 0]
        assert merge_predicted_spans(preds, LABELS, STARTS, ENDS) == [(16, 25)]

    def test_all_negative_predictions_yield_no_spans(self):
        preds = [0, 0, 0, 0, 0, 0]
        assert merge_predicted_spans(preds, LABELS, STARTS, ENDS) == []

    def test_ignore_positions_are_skipped(self):
        # Leading -100 (context/special) positions must not open spans even if the
        # (meaningless) prediction there is positive.
        labels = [-100, -100, 0, 0, 0, -100]
        starts = [0, 0, 0, 5, 10]
        ends = [0, 0, 5, 10, 15]
        preds = [1, 1, 0, 1, 0, 0]
        assert merge_predicted_spans(preds, labels, starts, ends) == [(5, 10)]

    def test_padded_rows_longer_than_offsets_are_handled(self):
        # Trainer pads predictions/labels across batches; offsets keep the true length.
        preds = [0, 1, 1, 0, 0, 0, 1, 1]
        labels = LABELS + [-100, -100]
        assert merge_predicted_spans(preds, labels, STARTS, ENDS) == [(5, 16)]


class TestCharSpanPrf:
    def test_exact_match_is_perfect(self):
        result = char_span_prf([[(0, 10)]], [[(0, 10)]])
        assert result == {"precision": 1.0, "recall": 1.0, "f1": 1.0}

    def test_half_overlap_arithmetic(self):
        # pred (0,10) vs gold (5,15): overlap 5, pred chars 10, gold chars 10.
        result = char_span_prf([[(0, 10)]], [[(5, 15)]])
        assert result["precision"] == pytest.approx(0.5)
        assert result["recall"] == pytest.approx(0.5)
        assert result["f1"] == pytest.approx(0.5)

    def test_disjoint_spans_score_zero(self):
        result = char_span_prf([[(0, 5)]], [[(10, 15)]])
        assert result == {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    def test_no_predicted_spans(self):
        result = char_span_prf([[]], [[(0, 10)]])
        assert result == {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    def test_no_gold_spans_with_predictions(self):
        result = char_span_prf([[(0, 10)]], [[]])
        assert result == {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    def test_micro_aggregation_across_examples(self):
        # Example 1: pred (0,10) vs gold (0,10) -> overlap 10 / pred 10 / gold 10.
        # Example 2: pred (0,10) vs gold (0,40) -> overlap 10 / pred 10 / gold 40.
        # Micro totals: overlap 20, pred 20, gold 50 -> P=1.0, R=0.4 (NOT the 0.625
        # mean of per-example recalls, proving totals are aggregated, not averaged).
        result = char_span_prf([[(0, 10)], [(0, 10)]], [[(0, 10)], [(0, 40)]])
        assert result["precision"] == pytest.approx(1.0)
        assert result["recall"] == pytest.approx(0.4)
        assert result["f1"] == pytest.approx(2 * 1.0 * 0.4 / 1.4)


class TestExactSpanPrf:
    def test_exact_match_counts(self):
        result = exact_span_prf([[(0, 10), (20, 30)]], [[(0, 10), (40, 50)]])
        assert result["precision"] == pytest.approx(0.5)  # 1 of 2 predicted spans exact
        assert result["recall"] == pytest.approx(0.5)  # 1 of 2 gold spans found
        assert result["f1"] == pytest.approx(0.5)

    def test_partial_overlap_is_not_a_match(self):
        result = exact_span_prf([[(0, 9)]], [[(0, 10)]])
        assert result == {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    def test_empty_both_sides(self):
        assert exact_span_prf([[]], [[]]) == {"precision": 0.0, "recall": 0.0, "f1": 0.0}


class TestComputeClassWeightsGuard:
    def test_zero_count_class_raises(self):
        with pytest.raises(ValueError, match="zero"):
            compute_class_weights([0, 0, 0], num_labels=2)

    def test_valid_counts_still_work(self):
        weights = compute_class_weights([0, 0, 0, 1], num_labels=2)
        # w_c = N / (num_classes * count_c) = [4/(2*3), 4/(2*1)]
        assert weights.tolist() == pytest.approx([2 / 3, 2.0])
