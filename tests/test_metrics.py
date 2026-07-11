"""Offline unit tests for the unified-predictions metrics module.

All data is synthetic; no result files are read and nothing touches the network.
"""

import numpy as np
import pandas as pd
import pytest

from src.evaluation import metrics
from src.evaluation.metrics import response_level_metrics


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
