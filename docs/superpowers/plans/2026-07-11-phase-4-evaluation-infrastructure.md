# Phase 4 Evaluation Infrastructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Backfill per-row test predictions (labels + probability scores) for all trained systems into one unified long-format table, plus reusable metrics functions that turn that table into the Phase 4 comparison.

**Architecture:** `scripts/collect_predictions.py` runs one system at a time (`baseline` | `track_a` | `approach_1`, plus a `merge` mode for folding in files produced on Kaggle) and merges results into `results/unified_predictions.parquet` without touching other systems' rows. `src/evaluation/metrics.py` is the read side: filter the table to one system, compute precision/recall/F1/confusion-matrix/classification-report and PR-curve data, so the full comparison table is a loop over system names.

**Tech Stack:** pandas, scikit-learn (`precision_recall_fscore_support`, `confusion_matrix`, `classification_report`, `precision_recall_curve`), torch + transformers (inference stages only), pytest.

## Global Constraints

- Run everything with the project venv: `.venv\Scripts\python.exe` (Windows; see memory: windows-env-quirks).
- Positive class is `1` = hallucinated, everywhere.
- `results/unified_predictions.parquet` is the single accumulating artifact; re-running one system must never drop another system's rows.
- System name strings (exact): `baseline_nli`, `track_a_deberta`, `approach_1_modernbert`. Track B will later use `track_b_modernbert_token` (not implemented in this plan).
- `scripts/collect_predictions.py` must be runnable standalone on Kaggle (`python scripts/collect_predictions.py <system> ...`), like `scripts/analyze_track_a_predictions.py`. No `os.chdir` at import time (only inside `main()`), so tests can import it safely.
- Tests are offline: no model downloads, no reading of real result files. Synthetic data only.

## Design Decisions (locked in)

### Unified schema (long format, one row per (system, test row))

| column | dtype | meaning |
|---|---|---|
| `system` | str | `baseline_nli` \| `track_a_deberta` \| `approach_1_modernbert` (later `track_b_modernbert_token`) |
| `row_index` | int | positional index (0..2699) in that system's test set. **Required because `source_id` is NOT unique** — the test parquet has 2700 rows but only 450 unique source_ids (6 model responses per source). All current systems share the same 2700-row composition and order (verified: same deterministic pipeline `load_merged_dataframe -> filter -> split=="test" -> reset_index`; nli json and track A json both have 2700 rows starting source_id 15596 x6), so `row_index` is a valid cross-system join key for paired analyses. |
| `source_id` | int | RAGTruth source id (repeats across the 6 responses per source) |
| `task_type` | str | Summary \| QA \| Data2txt |
| `split` | str | always `"test"` here |
| `y_true` | int | 0/1 ground-truth response label |
| `y_pred` | int | 0/1 system's operational decision |
| `y_score` | float | score in [0,1], higher = more likely hallucinated (positive class) |

File format: **parquet** (typed columns survive round-trips; pandas-native merge; it is a data artifact like `data/processed/*.parquet`, not a human-readable report like the `results/*.json` summaries).

### Baseline y_score derivation

Per sentence with aggregated `(max_entailment, max_contradiction)` (ADR-007), the hallucination score is
`max(contradiction, 1 - entailment)`; the response `y_score` is the **max over its sentences**. Empty sentence list -> `0.0` (vacuously-not-hallucinated convention; test json has 0 such rows, but the convention must match `apply_thresholds`).

Justification (vs. the two single-signal options):

- `max_contradiction` alone misses "unverifiable" hallucinations (low entailment, low contradiction). Empirically the baseline's 0.997 recall is driven by unverifiable flags, so this signal alone would misrank most detected positives.
- `1 - min_entailment` alone misses ADR-007's core case: a sentence with high entailment from one context sentence but high contradiction from a *different* one. ADR-007 tracks the two maxima independently precisely because a contradicted-but-partially-supported claim is still a hallucination; a pure-entailment score would rank it as safe.
- The decision rule in `flag_from_scores` is a **disjunction**: not-supported iff `contradiction >= con_thr` OR `entailment < ent_thr`. The elementwise `max(con, 1 - ent)` is the score whose single threshold `s >= t` reproduces exactly the coupled rule family `con_thr = t, ent_thr = 1 - t` — the faithful one-dimensional reduction of the two-signal rule, honoring ADR-007's contradiction-priority (a high contradiction dominates the max regardless of entailment).

Caveat to document in the code: the tuned operating point `(ent_thr=0.4, con_thr=0.4)` is *not* on that coupled family, so baseline `y_pred` (computed with `apply_thresholds` at the tuned thresholds, read from `results/baseline_nli_metrics.json` `best_thresholds`) is not exactly `y_score >= t` for any single `t`. That is fine: `y_pred` records the system's operational decision; `y_score` supports threshold-free PR curves.

Comparability note: baseline `y_score` is a max of independent maxima, not a calibrated probability, while Track A / Approach 1 scores are softmax probabilities. All are monotone "higher = more hallucinated" scores in [0,1]; PR curves and ranking metrics are valid per system, but raw score values should not be compared across systems.

### Track A / Approach 1 y_score

`softmax(logits)[:, 1]` (index 1 = hallucinated, matching `label_response = int(len(labels) > 0)` used in training). `y_pred = argmax(logits)` (equivalent to `y_score >= 0.5` for binary softmax). Same batched-inference pattern as `scripts/analyze_track_a_predictions.py`.

### Track B (future, not blocking)

When the Kaggle run finishes: add a `track_b` mode producing the same schema with `system="track_b_modernbert_token"`. Response-level aggregation from token predictions: `y_score = max` per-token P(hallucinated) over the response's tokens, `y_pred = 1` if any token is predicted positive. Nothing in this plan hard-codes a closed set of systems — `merge_predictions`, `metrics_for_system`, and `comparison_table` all key off the `system` column dynamically.

### File structure

- Create: `src/evaluation/metrics.py` — read side (schema constants, loading/validation, per-system metrics, PR curve, comparison table). Pure pandas/sklearn, no torch.
- Create: `scripts/collect_predictions.py` — write side (CLI with modes `baseline` / `track_a` / `approach_1` / `merge`; pure helpers `baseline_y_score`, `build_prediction_rows`, `merge_predictions` at module level so tests can import them; torch inference only inside the transformer path).
- Create: `tests/test_metrics.py`, `tests/test_collect_predictions.py`.

---

### Task 1: `response_level_metrics` in `src/evaluation/metrics.py`

**Files:**
- Create: `src/evaluation/metrics.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Produces: `response_level_metrics(y_true, y_pred) -> dict` with keys `n`, `precision`, `recall`, `f1`, `confusion_matrix` (2x2 nested list, rows=true 0/1, cols=pred 0/1), `classification_report` (dict). Also module constants `UNIFIED_PREDICTIONS_PATH = Path("results/unified_predictions.parquet")`, `UNIFIED_COLUMNS`, `TARGET_NAMES = ["not_hallucinated", "hallucinated"]` used by Tasks 2–4.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_metrics.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_metrics.py -v`
Expected: FAIL with `ImportError` / `ModuleNotFoundError` (metrics module has no content yet).

- [ ] **Step 3: Write the implementation**

```python
# src/evaluation/metrics.py
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_metrics.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/evaluation/metrics.py tests/test_metrics.py
git commit -m "feat: response_level_metrics for Phase 4 unified evaluation"
```

---

### Task 2: Per-system slicing, PR curve, and comparison table in `metrics.py`

**Files:**
- Modify: `src/evaluation/metrics.py` (append functions)
- Test: `tests/test_metrics.py` (append tests)

**Interfaces:**
- Consumes: `UNIFIED_COLUMNS`, `response_level_metrics` from Task 1.
- Produces:
  - `load_predictions(path=UNIFIED_PREDICTIONS_PATH) -> pd.DataFrame` (reads parquet, validates required columns)
  - `system_predictions(df: pd.DataFrame, system: str) -> pd.DataFrame` (filter; raises ValueError listing available systems if empty)
  - `metrics_for_system(df: pd.DataFrame, system: str) -> dict` (same dict shape as `response_level_metrics`)
  - `pr_curve_for_system(df: pd.DataFrame, system: str) -> dict` with keys `precision`, `recall`, `thresholds` (numpy arrays from sklearn)
  - `comparison_table(df: pd.DataFrame, systems: list[str] | None = None) -> pd.DataFrame` with columns `system`, `n`, `precision`, `recall`, `f1`

- [ ] **Step 1: Write the failing tests (append to tests/test_metrics.py)**

```python
from src.evaluation.metrics import (
    UNIFIED_COLUMNS,
    comparison_table,
    load_predictions,
    metrics_for_system,
    pr_curve_for_system,
    system_predictions,
)


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
```

- [ ] **Step 2: Run tests to verify the new ones fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_metrics.py -v`
Expected: Task 1 tests PASS, new tests FAIL with ImportError.

- [ ] **Step 3: Write the implementation (append to src/evaluation/metrics.py)**

```python
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
```

- [ ] **Step 4: Run the full test file**

Run: `.venv\Scripts\python.exe -m pytest tests/test_metrics.py -v`
Expected: all tests pass (Task 1's 5 + Task 2's 9).

- [ ] **Step 5: Commit**

```bash
git add src/evaluation/metrics.py tests/test_metrics.py
git commit -m "feat: per-system metrics, PR curve, and comparison table over unified predictions"
```

---

### Task 3: Pure helpers in `scripts/collect_predictions.py` (score derivation, row building, merge)

**Files:**
- Create: `scripts/collect_predictions.py` (module docstring, imports, pure helpers — CLI comes in Task 4)
- Test: `tests/test_collect_predictions.py`

**Interfaces:**
- Consumes: `UNIFIED_COLUMNS` from `src.evaluation.metrics`; `apply_thresholds` from `src.models.nli_baseline` (existing: takes `list[list[tuple[ent, con]]]`, `ent_thr`, `con_thr`, returns `list[bool]`).
- Produces (used by Task 4/5):
  - `baseline_y_score(sentence_scores: list) -> float` — sentence pairs are `[entailment, contradiction]`
  - `build_prediction_rows(system, source_ids, task_types, y_true, y_pred, y_score) -> pd.DataFrame` — adds `row_index` = 0..n-1 and `split="test"`, returns columns in `UNIFIED_COLUMNS` order
  - `merge_predictions(existing: pd.DataFrame | None, new: pd.DataFrame) -> pd.DataFrame` — replaces rows for systems present in `new`, keeps everything else
  - Constants: `SYSTEM_BASELINE = "baseline_nli"`, `SYSTEM_TRACK_A = "track_a_deberta"`, `SYSTEM_APPROACH_1 = "approach_1_modernbert"`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_collect_predictions.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_collect_predictions.py -v`
Expected: FAIL — the script file does not exist yet (FileNotFoundError from importlib).

- [ ] **Step 3: Write the script skeleton with the pure helpers**

```python
# scripts/collect_predictions.py
"""Backfill per-row test predictions (labels + probability scores) into the unified table.

Phase 4 compares every trained system on identical footing: one long-format table
(results/unified_predictions.parquet by default) with one row per (system, test row) and
columns system/row_index/source_id/task_type/split/y_true/y_pred/y_score, where y_score
is always P(hallucinated)-like (higher = more likely hallucinated). The systems live on
different Kaggle sessions and Hub repos, so this script runs ONE system at a time and
merges into the accumulating file — re-running a system replaces only that system's rows.

Modes:
    baseline    Model-free. Aggregates results/nli_scores_test.json to response level:
                y_score = max over sentences of max(contradiction, 1 - entailment) (the
                single-threshold reduction of ADR-007's disjunctive decision rule);
                y_pred = apply_thresholds at the tuned thresholds from
                results/baseline_nli_metrics.json. Note y_pred is the operational
                decision at (ent_thr, con_thr) and is not exactly a threshold on
                y_score — y_score exists for threshold-free PR curves.
    track_a     Batched Hub inference over data/processed/response_level_test.parquet
                (same pattern as scripts/analyze_track_a_predictions.py) capturing
                softmax P(hallucinated) per row.
    approach_1  Same, over data/processed/response_level_modernbert_test.parquet
                (regenerate via `python -m src.data.preprocess_modernbert` if absent).
    merge       Fold a unified-schema parquet produced elsewhere (e.g. downloaded from a
                Kaggle session) into the local accumulating file.

Examples:
    python scripts/collect_predictions.py baseline
    python scripts/collect_predictions.py track_a
    python scripts/collect_predictions.py approach_1
    python scripts/collect_predictions.py merge --input kaggle_output/unified_predictions.parquet
"""

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding  # noqa: E402

from src.evaluation.metrics import UNIFIED_COLUMNS, UNIFIED_PREDICTIONS_PATH  # noqa: E402
from src.models.nli_baseline import apply_thresholds  # noqa: E402

SYSTEM_BASELINE = "baseline_nli"
SYSTEM_TRACK_A = "track_a_deberta"
SYSTEM_APPROACH_1 = "approach_1_modernbert"

NLI_SCORES_PATH = Path("results/nli_scores_test.json")
BASELINE_METRICS_PATH = Path("results/baseline_nli_metrics.json")
HUB_DEFAULTS = {
    SYSTEM_TRACK_A: "hugoomezz/deberta-v3-ragtruth-hallucination",
    SYSTEM_APPROACH_1: "hugoomezz/deberta-v3-modernbert-ragtruth-hallucination",
}
TEST_PATH_DEFAULTS = {
    SYSTEM_TRACK_A: "data/processed/response_level_test.parquet",
    SYSTEM_APPROACH_1: "data/processed/response_level_modernbert_test.parquet",
}
BATCH_SIZE = 32


def baseline_y_score(sentence_scores: list) -> float:
    """Response-level hallucination score from per-sentence (entailment, contradiction) pairs.

    Per sentence: max(contradiction, 1 - entailment) — the one-dimensional reduction of
    ADR-007's disjunctive flag rule (not-supported iff contradiction >= con_thr OR
    entailment < ent_thr), so a single threshold sweep on this score walks the coupled
    rule family con_thr = t, ent_thr = 1 - t. Contradiction-priority is honored: a high
    contradiction dominates the max even when some context sentence strongly entails the
    claim. Response score is the max over sentences; empty responses score 0.0
    (vacuously-not-hallucinated, matching apply_thresholds).
    """
    if not sentence_scores:
        return 0.0
    return max(max(float(con), 1.0 - float(ent)) for ent, con in sentence_scores)


def build_prediction_rows(system, source_ids, task_types, y_true, y_pred, y_score) -> pd.DataFrame:
    """Assemble one system's rows in the unified schema, with positional row_index.

    row_index (0..n-1, input order) is the per-row key: source_id is NOT unique in
    RAGTruth (6 model responses per source). All systems iterate the same deterministic
    2700-row test set, so row_index also serves as the cross-system join key.
    """
    lengths = {len(source_ids), len(task_types), len(y_true), len(y_pred), len(y_score)}
    if len(lengths) != 1:
        raise ValueError(f"Input columns have mismatched lengths: {lengths}")
    n_rows = lengths.pop()
    return pd.DataFrame(
        {
            "system": [system] * n_rows,
            "row_index": range(n_rows),
            "source_id": source_ids,
            "task_type": task_types,
            "split": ["test"] * n_rows,
            "y_true": pd.array(y_true, dtype="int64"),
            "y_pred": pd.array(y_pred, dtype="int64"),
            "y_score": pd.array(y_score, dtype="float64"),
        }
    )[UNIFIED_COLUMNS]


def merge_predictions(existing: pd.DataFrame | None, new: pd.DataFrame) -> pd.DataFrame:
    """Replace rows for systems present in `new`; keep every other system's rows.

    This is what makes stage-wise collection safe: re-running one system (or folding in
    a file from another Kaggle session) never drops previously collected systems.
    """
    if existing is None:
        return new.reset_index(drop=True)
    replaced = set(new["system"].unique())
    kept = existing[~existing["system"].isin(replaced)]
    return pd.concat([kept, new], ignore_index=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_collect_predictions.py -v`
Expected: 10 passed.

- [ ] **Step 5: Run the whole suite to catch regressions**

Run: `.venv\Scripts\python.exe -m pytest -q`
Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add scripts/collect_predictions.py tests/test_collect_predictions.py
git commit -m "feat: unified-prediction helpers (baseline y_score, row schema, merge semantics)"
```

---

### Task 4: Baseline stage + merge mode + CLI; run baseline for real

**Files:**
- Modify: `scripts/collect_predictions.py` (append collection functions, `main`, CLI)
- Modify: `docs/decisions.md` (append ADR documenting the baseline y_score derivation)

**Interfaces:**
- Consumes: Task 3 helpers; `apply_thresholds` from `src.models.nli_baseline`.
- Produces:
  - `collect_baseline(scores_path: Path, metrics_path: Path) -> pd.DataFrame`
  - `save_merged(new_df: pd.DataFrame, unified_path: Path) -> pd.DataFrame` (reads existing file if present, merges, writes, prints per-system counts)
  - `main()` with subcommand-style `mode` positional: `baseline | track_a | approach_1 | merge` (track_a/approach_1 wired in Task 5; in this task they exit with a clear "implemented in Task 5" error is NOT acceptable — instead define the CLI fully in Task 5 and register only `baseline` and `merge` here, extending choices in Task 5).

- [ ] **Step 1: Append the baseline collection, save, and CLI code**

```python
def collect_baseline(scores_path: Path = NLI_SCORES_PATH, metrics_path: Path = BASELINE_METRICS_PATH) -> pd.DataFrame:
    """Aggregate the per-sentence NLI scores to response level in the unified schema.

    Model-free: reuses results/nli_scores_test.json (sentence_scores are
    [max_entailment, max_contradiction] pairs, ADR-007) and the tuned thresholds
    already selected on the validation split (results/baseline_nli_metrics.json).
    """
    rows = json.loads(Path(scores_path).read_text(encoding="utf-8"))
    thresholds = json.loads(Path(metrics_path).read_text(encoding="utf-8"))["best_thresholds"]

    all_scores = [[(pair[0], pair[1]) for pair in row["sentence_scores"]] for row in rows]
    y_pred = [int(flag) for flag in apply_thresholds(all_scores, thresholds["ent_thr"], thresholds["con_thr"])]
    y_score = [baseline_y_score(scores) for scores in all_scores]

    print(
        f"Baseline: {len(rows)} rows from {scores_path}, "
        f"thresholds ent={thresholds['ent_thr']} con={thresholds['con_thr']}"
    )
    return build_prediction_rows(
        system=SYSTEM_BASELINE,
        source_ids=[row["source_id"] for row in rows],
        task_types=[row["task_type"] for row in rows],
        y_true=[int(row["label_response"]) for row in rows],
        y_pred=y_pred,
        y_score=y_score,
    )


def save_merged(new_df: pd.DataFrame, unified_path: Path) -> pd.DataFrame:
    """Merge one stage's rows into the accumulating file and report what it now holds."""
    unified_path = Path(unified_path)
    existing = pd.read_parquet(unified_path) if unified_path.exists() else None
    merged = merge_predictions(existing, new_df)
    unified_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(unified_path, index=False)
    print(f"Saved {unified_path} — rows per system:")
    for system, count in merged["system"].value_counts().sort_index().items():
        print(f"  {system}: {count}")
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["baseline", "merge"])
    parser.add_argument("--unified_path", default=str(UNIFIED_PREDICTIONS_PATH))
    parser.add_argument("--input", help="merge mode: unified-schema parquet to fold in (e.g. from Kaggle).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.chdir(REPO_ROOT)

    if args.mode == "baseline":
        new_df = collect_baseline()
    elif args.mode == "merge":
        if not args.input:
            raise SystemExit("merge mode requires --input")
        new_df = pd.read_parquet(args.input)
        missing = [column for column in UNIFIED_COLUMNS if column not in new_df.columns]
        if missing:
            raise SystemExit(f"--input file is missing unified-schema columns: {missing}")

    save_merged(new_df, Path(args.unified_path))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Re-run the offline suite (import side effects, no regressions)**

Run: `.venv\Scripts\python.exe -m pytest -q`
Expected: all pass (importing the script must not execute `main`/`os.chdir`).

- [ ] **Step 3: Run the baseline stage for real**

Run: `.venv\Scripts\python.exe scripts\collect_predictions.py baseline`
Expected output: `Baseline: 2700 rows ... thresholds ent=0.4 con=0.4` then `Saved results\unified_predictions.parquet — rows per system:` with `baseline_nli: 2700`.

- [ ] **Step 4: Verify end-to-end against the published baseline metrics**

Run:
```powershell
.venv\Scripts\python.exe -c "
from src.evaluation.metrics import load_predictions, metrics_for_system
m = metrics_for_system(load_predictions(), 'baseline_nli')
print({k: m[k] for k in ('n', 'precision', 'recall', 'f1')})
"
```
Expected: matches `results/baseline_nli_metrics.json` -> `test.nli_baseline` exactly: precision 0.35469280060309083, recall 0.9978791092258749, f1 0.5233592880978866, n 2700. **If these do not match, stop and debug — do not commit.**

- [ ] **Step 5: Verify idempotent re-run**

Run: `.venv\Scripts\python.exe scripts\collect_predictions.py baseline` again.
Expected: still exactly 2700 `baseline_nli` rows (replaced, not duplicated).

- [ ] **Step 6: Append the ADR to docs/decisions.md**

Append (adjusting the ADR number to be the next free one at execution time — check the current max in docs/decisions.md and in any unmerged branch docs; memory indicates ADR-013 may exist on the Track B branch, so this is likely ADR-014):

```markdown
## ADR-014: Baseline y_score = max over sentences of max(contradiction, 1 - entailment)

**Context:** Phase 4's unified comparison needs one probability-like score per response
from every system. Fine-tuned classifiers provide softmax P(hallucinated) directly, but
the zero-shot NLI baseline produces per-sentence (max_entailment, max_contradiction)
pairs (ADR-007) plus a two-threshold decision rule — there is no single native score.

**Decision:** y_score = max over response sentences of max(contradiction, 1 - entailment).
The flag rule is a disjunction (not-supported iff contradiction >= con_thr OR
entailment < ent_thr), and this is its faithful one-dimensional reduction: sweeping a
single threshold t over the score walks the coupled rule family con_thr = t,
ent_thr = 1 - t, while preserving ADR-007's contradiction-priority (high contradiction
dominates the max even under high entailment). Empty responses score 0.0, matching the
vacuously-not-hallucinated convention.

**Alternatives considered:** max_contradiction alone (rejected: blind to "unverifiable"
hallucinations, which drive the baseline's 0.997 recall); 1 - min_entailment alone
(rejected: blind to ADR-007's core case of a claim entailed by one context sentence but
contradicted by another). Note the tuned operating point (ent_thr=0.4, con_thr=0.4) is
not on the coupled family, so the recorded y_pred (operational decision) is not exactly
a threshold on y_score; y_pred and y_score serve different purposes (point metrics vs
PR curves).

**Status:** Implemented in scripts/collect_predictions.py (baseline mode); verified by
reproducing results/baseline_nli_metrics.json test metrics from the unified table.
```

- [ ] **Step 7: Commit (including the generated parquet)**

```bash
git add scripts/collect_predictions.py docs/decisions.md results/unified_predictions.parquet
git commit -m "feat: baseline stage of unified prediction collection + ADR for y_score derivation"
```

---

### Task 5: Transformer inference stages (track_a, approach_1)

**Files:**
- Modify: `scripts/collect_predictions.py` (append inference functions, extend CLI)

**Interfaces:**
- Consumes: Task 3/4 helpers; `HUB_DEFAULTS`, `TEST_PATH_DEFAULTS`, `BATCH_SIZE` constants.
- Produces:
  - `run_inference_with_probs(df: pd.DataFrame, model, collator, device) -> tuple[list[int], list[float]]` — (argmax preds, softmax P(class 1)) in row order
  - `collect_transformer(system: str, hub_model_id: str, test_path: str, limit: int | None) -> pd.DataFrame`
  - CLI extension: `mode` choices become `baseline | track_a | approach_1 | merge`; new args `--hub_model_id` (default from `HUB_DEFAULTS[system]`), `--test_path` (default from `TEST_PATH_DEFAULTS[system]`), `--limit N` (smoke-test: run inference on the first N rows, print metrics, and **skip the merge-write** so a partial run can never poison the unified file).

- [ ] **Step 1: Append the inference code**

```python
def run_inference_with_probs(df: pd.DataFrame, model, collator, device: torch.device) -> tuple[list[int], list[float]]:
    """Batched forward pass in row order, returning argmax labels and softmax P(hallucinated).

    Same batching pattern as scripts/analyze_track_a_predictions.py, extended to keep the
    positive-class probability (index 1 = hallucinated, matching training's
    label_response encoding) for threshold-free PR curves.
    """
    examples = [{"input_ids": row.input_ids, "attention_mask": row.attention_mask} for row in df.itertuples()]
    loader = DataLoader(examples, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collator)

    model.eval()
    preds: list[int] = []
    scores: list[float] = []
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            logits = model(**batch).logits
            probs = torch.softmax(logits, dim=-1)
            preds.extend(probs.argmax(dim=-1).cpu().tolist())
            scores.extend(probs[:, 1].cpu().tolist())
    return preds, scores


def load_test_df(path: str) -> pd.DataFrame:
    """Read a response-level test parquet; input_ids/attention_mask are already tokenized."""
    df = pd.read_parquet(path)
    df["input_ids"] = df["input_ids"].apply(lambda a: np.asarray(a).tolist())
    df["attention_mask"] = df["attention_mask"].apply(lambda a: np.asarray(a).tolist())
    return df


def collect_transformer(system: str, hub_model_id: str, test_path: str, limit: int | None = None) -> pd.DataFrame:
    """Hub-checkpoint inference over a test parquet, in the unified schema."""
    df = load_test_df(test_path)
    if limit is not None:
        df = df.head(limit)
    print(f"{system}: {len(df)} rows from {test_path}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(hub_model_id)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(hub_model_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Loaded {hub_model_id} on {device}", flush=True)

    y_pred, y_score = run_inference_with_probs(df, model, collator, device)
    return build_prediction_rows(
        system=system,
        source_ids=df["source_id"].tolist(),
        task_types=df["task_type"].tolist(),
        y_true=df["label_response"].astype(int).tolist(),
        y_pred=y_pred,
        y_score=y_score,
    )
```

And rewrite `parse_args`/`main` to:

```python
MODE_TO_SYSTEM = {"track_a": SYSTEM_TRACK_A, "approach_1": SYSTEM_APPROACH_1}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["baseline", "track_a", "approach_1", "merge"])
    parser.add_argument("--unified_path", default=str(UNIFIED_PREDICTIONS_PATH))
    parser.add_argument("--hub_model_id", help="Override the default Hub repo for track_a/approach_1.")
    parser.add_argument("--test_path", help="Override the default test parquet for track_a/approach_1.")
    parser.add_argument("--input", help="merge mode: unified-schema parquet to fold in (e.g. from Kaggle).")
    parser.add_argument(
        "--limit",
        type=int,
        help="Smoke test: run inference on only the first N rows, print metrics, and skip writing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.chdir(REPO_ROOT)

    if args.mode == "baseline":
        new_df = collect_baseline()
    elif args.mode in MODE_TO_SYSTEM:
        system = MODE_TO_SYSTEM[args.mode]
        new_df = collect_transformer(
            system=system,
            hub_model_id=args.hub_model_id or HUB_DEFAULTS[system],
            test_path=args.test_path or TEST_PATH_DEFAULTS[system],
            limit=args.limit,
        )
    else:  # merge
        if not args.input:
            raise SystemExit("merge mode requires --input")
        new_df = pd.read_parquet(args.input)
        missing = [column for column in UNIFIED_COLUMNS if column not in new_df.columns]
        if missing:
            raise SystemExit(f"--input file is missing unified-schema columns: {missing}")

    if args.limit is not None:
        from src.evaluation.metrics import response_level_metrics

        preview = response_level_metrics(new_df["y_true"].to_numpy(), new_df["y_pred"].to_numpy())
        print(f"[--limit smoke test] not writing. Metrics on {preview['n']} rows:")
        print({key: preview[key] for key in ("precision", "recall", "f1")})
        return

    save_merged(new_df, Path(args.unified_path))
```

(Move the `response_level_metrics` import to the top-level import block from `src.evaluation.metrics` alongside `UNIFIED_COLUMNS` — shown inline above only for diff clarity.)

- [ ] **Step 2: Run the offline suite**

Run: `.venv\Scripts\python.exe -m pytest -q`
Expected: all pass.

- [ ] **Step 3: Smoke-test track_a locally with --limit (downloads the Hub model, CPU)**

Run: `.venv\Scripts\python.exe scripts\collect_predictions.py track_a --limit 16`
Expected: loads `hugoomezz/deberta-v3-ragtruth-hallucination`, prints `[--limit smoke test] not writing.` with metrics over 16 rows. Then confirm the unified file still has only `baseline_nli` rows:
```powershell
.venv\Scripts\python.exe -c "import pandas as pd; print(pd.read_parquet('results/unified_predictions.parquet')['system'].value_counts())"
```

- [ ] **Step 4: Commit**

```bash
git add scripts/collect_predictions.py
git commit -m "feat: track_a/approach_1 inference stages with probability capture"
```

---

### Task 6: Operational runs (execute per stage as resources allow)

These are runbook steps, not code. Full inference over 2700 rows: Track A (DeBERTa-base, 512 tokens) is feasible on local CPU (~tens of minutes) or fast on Kaggle GPU; Approach 1 (ModernBERT-base, 4096 tokens) should run on Kaggle GPU, same as `scripts/analyze_track_a_predictions.py` did.

- [ ] **Step 1: Full Track A run**

Locally: `.venv\Scripts\python.exe scripts\collect_predictions.py track_a`
Or on Kaggle: `python scripts/collect_predictions.py track_a --unified_path /kaggle/working/unified_predictions.parquet`, download the output, then locally:
`.venv\Scripts\python.exe scripts\collect_predictions.py merge --input <downloaded>.parquet`

- [ ] **Step 2: Verify Track A against the existing predictions file**

```powershell
.venv\Scripts\python.exe -c "
import json, pandas as pd
old = json.load(open('results/track_a_test_predictions.json', encoding='utf-8'))
df = pd.read_parquet('results/unified_predictions.parquet')
ta = df[df['system'] == 'track_a_deberta'].sort_values('row_index')
assert len(ta) == len(old) == 2700
mismatches = sum(int(r['predicted_label']) != p for r, p in zip(old, ta['y_pred']))
consistent = ((ta['y_score'] >= 0.5) == (ta['y_pred'] == 1)).all()
print(f'label mismatches vs analyze-script run: {mismatches} (expect 0)')
print(f'y_pred == (y_score >= 0.5): {consistent} (expect True)')
"
```
Same model + argmax must reproduce `results/track_a_test_predictions.json` exactly. Nonzero mismatches mean a bug (row order, tokenization path) — stop and debug.

- [ ] **Step 3: Full Approach 1 run**

The ModernBERT test parquet is not on local disk. Either regenerate locally first
(`.venv\Scripts\python.exe -m src.data.preprocess_modernbert`, needs `data/raw`) or run on
Kaggle where it already exists:
`python scripts/collect_predictions.py approach_1 --unified_path /kaggle/working/unified_predictions.parquet`
then download + `merge` locally as in Step 1.

- [ ] **Step 4: Verify Approach 1 against published metrics**

```powershell
.venv\Scripts\python.exe -c "
from src.evaluation.metrics import load_predictions, metrics_for_system
m = metrics_for_system(load_predictions(), 'approach_1_modernbert')
print({k: m[k] for k in ('n', 'precision', 'recall', 'f1')})
"
```
Expected: matches `results/finetuned_approach1_modernbert_metrics.json` -> `test.finetuned`: precision 0.6838649155722326, recall 0.7730646871686108, f1 0.7257341961174714, n 2700.

- [ ] **Step 5: Sanity-check the comparison loop end-to-end**

```powershell
.venv\Scripts\python.exe -c "
from src.evaluation.metrics import comparison_table, load_predictions
print(comparison_table(load_predictions()).to_string(index=False))
"
```
Expected: three rows (baseline_nli, track_a_deberta, approach_1_modernbert), n=2700 each, F1s ≈ 0.523 / 0.786 (Track A test F1 from finetuned_track_a_metrics.json) / 0.726.

- [ ] **Step 6: Commit the updated parquet**

```bash
git add results/unified_predictions.parquet
git commit -m "data: unified test predictions for baseline, track A, and approach 1"
```

---

## Self-review notes

- Spec coverage: unified schema + accumulating file (Tasks 3–4), baseline aggregation with justified score choice (Task 4 + ADR), Track A/Approach 1 probability capture (Task 5), non-destructive per-system re-runs (Task 3 merge semantics + Task 4 Step 5 verification), `response_level_metrics` + per-system filter + comparison loop (Tasks 1–2), PR-curve function (Task 2), Track B accommodation (dynamic `system` column; documented future mode). All requirements mapped.
- Type consistency: `build_prediction_rows` output column order == `UNIFIED_COLUMNS`; `merge_predictions(existing: DataFrame | None, new)` used identically in tests and `save_merged`; `sentence_scores` pair order is `[entailment, contradiction]` everywhere (matches `score_response` and the json).
- The exact Track A test F1 (0.786) cited in Task 6 Step 5 should be read from `results/finetuned_track_a_metrics.json` at execution time rather than trusted from this plan.
