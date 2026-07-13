"""Research-report recommendations #1 (threshold tuning) and #4 (simple ensemble).

Both analyses tune strictly on the VAL split and apply the chosen parameters to TEST
exactly once — no iterating or re-selecting on test. The script is model-free: it reads
only results/unified_predictions.parquet (produced by scripts/collect_predictions.py,
now with both val and test rows) and re-thresholds the stored y_score.

Rec #1 — Track B threshold tuning:
    Track B's y_score is the max per-token P(hallucinated) over a response's real tokens;
    thresholding it at 0.5 recovers the any-positive decision rule (ADR-013/-015). We
    sweep the decision threshold on val for (a) one global response-level threshold and
    (b) a separate threshold per task_type (Summary/QA/Data2txt), targeting the diagnosed
    Summary recall weakness. Best val thresholds are then applied to test once.

Rec #4 — Simple 3-system ensemble:
    A weighted average of the raw [0,1] y_scores of baseline_nli + approach_1_modernbert +
    track_b_modernbert, plus a decision threshold, tuned on val. track_a_deberta is
    EXCLUDED: its val parquet was preprocessed with a different within-source response
    ordering, so it cannot be row_index-joined against the other three on val (test-only
    alignment can't be tuned on val without breaking the discipline). The winning
    weights+threshold are applied to test once and compared against Track B alone.

Usage:
    python scripts/tune_threshold_and_ensemble.py
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.evaluation.metrics import (  # noqa: E402
    UNIFIED_PREDICTIONS_PATH,
    load_predictions,
    response_level_metrics,
)

SYSTEM_BASELINE = "baseline_nli"
SYSTEM_TRACK_A = "track_a_deberta"
SYSTEM_APPROACH_1 = "approach_1_modernbert"
SYSTEM_TRACK_B = "track_b_modernbert"

# Column order fixes the weight-vector order: (w_baseline, w_approach1, w_track_b).
ENSEMBLE_SYSTEMS = [SYSTEM_BASELINE, SYSTEM_APPROACH_1, SYSTEM_TRACK_B]
TASK_TYPES = ["Summary", "QA", "Data2txt"]

# Shared decision-threshold grid for both the Track-B sweep and the ensemble threshold.
THRESHOLD_GRID = np.round(np.arange(0.10, 0.9001, 0.05), 2)
# Simplex weight grid step (0.05 -> 231 vectors); contains (0.25,0.25,0.50) exactly.
WEIGHT_DENOM = 20

DEFAULT_OUTPUT = Path("results/threshold_ensemble_tuning.json")


# --------------------------------------------------------------------------------------
# Small pure helpers
# --------------------------------------------------------------------------------------
def metrics_at_threshold(y_true, y_score, thr: float) -> dict:
    """Response-level metrics from thresholding a score at `thr` (>= thr -> hallucinated)."""
    y_pred = (np.asarray(y_score, dtype=float) >= thr).astype(int)
    return response_level_metrics(np.asarray(y_true), y_pred)


def sweep_thresholds(y_true, y_score, grid=THRESHOLD_GRID) -> pd.DataFrame:
    """One row per threshold: precision/recall/f1 on the given split."""
    rows = []
    for thr in grid:
        m = metrics_at_threshold(y_true, y_score, thr)
        rows.append({"threshold": float(thr), "precision": m["precision"], "recall": m["recall"], "f1": m["f1"]})
    return pd.DataFrame(rows)


def best_by_f1(sweep: pd.DataFrame) -> pd.Series:
    """The max-F1 row of a sweep (ties: lowest threshold, i.e. first index)."""
    return sweep.loc[sweep["f1"].idxmax()]


def weight_grid(denom: int = WEIGHT_DENOM) -> list[tuple[float, float, float]]:
    """All 3-weight vectors on the simplex with the given denominator, summing to 1.0."""
    combos = []
    for i in range(denom + 1):
        for j in range(denom - i + 1):
            k = denom - i - j
            combos.append((i / denom, j / denom, k / denom))
    return combos


def per_task_metrics(y_true, y_pred, tasks) -> dict:
    """Response-level F1/precision/recall for each task_type subset."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tasks = np.asarray(tasks)
    out = {}
    for task in TASK_TYPES:
        mask = tasks == task
        if not mask.any():  # empty subset — sklearn raises on empty input, so short-circuit
            out[task] = {"n": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
            continue
        m = response_level_metrics(y_true[mask], y_pred[mask])
        out[task] = {"n": m["n"], "precision": m["precision"], "recall": m["recall"], "f1": m["f1"]}
    return out


# --------------------------------------------------------------------------------------
# Printing
# --------------------------------------------------------------------------------------
def _fmt(x) -> str:
    return f"{x:.4f}" if isinstance(x, float) else str(x)


def print_table(title: str, df: pd.DataFrame, columns: list[str]) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    widths = {c: max(len(c), *(len(_fmt(v)) for v in df[c])) for c in columns}
    print("  ".join(c.rjust(widths[c]) for c in columns))
    for _, row in df.iterrows():
        print("  ".join(_fmt(row[c]).rjust(widths[c]) for c in columns))


def print_metrics_line(label: str, m: dict) -> None:
    print(f"  {label:<28} P={m['precision']:.4f}  R={m['recall']:.4f}  F1={m['f1']:.4f}  (n={m['n']})")


# --------------------------------------------------------------------------------------
# Rec #1 — Track B threshold tuning
# --------------------------------------------------------------------------------------
def track_b_threshold_tuning(df: pd.DataFrame) -> dict:
    val = df[(df["system"] == SYSTEM_TRACK_B) & (df["split"] == "val")]
    test = df[(df["system"] == SYSTEM_TRACK_B) & (df["split"] == "test")]
    if val.empty:
        raise SystemExit("No Track B val rows — run: collect_predictions.py track_b_modernbert --split val")

    print("\n" + "=" * 78)
    print("REC #1 — TRACK B THRESHOLD TUNING  (tuned on VAL, applied to TEST once)")
    print("=" * 78)

    # (a) Global sweep on val.
    global_sweep = sweep_thresholds(val["y_true"], val["y_score"])
    best_global = best_by_f1(global_sweep)
    t_global = float(best_global["threshold"])
    print_table("VAL — global threshold sweep", global_sweep, ["threshold", "precision", "recall", "f1"])
    print(f"  -> best global val threshold = {t_global:.2f}  (F1={best_global['f1']:.4f})")

    # Reference: Track B's stored operating point (~0.5 any-positive rule).
    val_stored = response_level_metrics(val["y_true"].to_numpy(), val["y_pred"].to_numpy())
    print_metrics_line("val @ stored y_pred (~0.5)", val_stored)

    # (b) Per-task_type sweep on val.
    per_task_thr = {}
    per_task_best = {}
    for task in TASK_TYPES:
        sub = val[val["task_type"] == task]
        sweep = sweep_thresholds(sub["y_true"], sub["y_score"])
        best = best_by_f1(sweep)
        per_task_thr[task] = float(best["threshold"])
        per_task_best[task] = best
        stored = response_level_metrics(sub["y_true"].to_numpy(), sub["y_pred"].to_numpy())
        print_table(f"VAL — {task} threshold sweep (n={len(sub)})", sweep, ["threshold", "precision", "recall", "f1"])
        print(
            f"  -> best {task} val threshold = {per_task_thr[task]:.2f}  "
            f"F1={best['f1']:.4f} (P={best['precision']:.4f} R={best['recall']:.4f})  |  "
            f"stored ~0.5: P={stored['precision']:.4f} R={stored['recall']:.4f} F1={stored['f1']:.4f}"
        )

    # ---- Apply ONCE to test ----
    print("\n" + "-" * 78)
    print("FINAL TEST (applied once) — Track B threshold configurations")
    print("-" * 78)
    test_stored = response_level_metrics(test["y_true"].to_numpy(), test["y_pred"].to_numpy())
    test_global = metrics_at_threshold(test["y_true"], test["y_score"], t_global)
    pertask_pred = np.array([int(s >= per_task_thr[t]) for s, t in zip(test["y_score"], test["task_type"])])
    test_pertask = response_level_metrics(test["y_true"].to_numpy(), pertask_pred)

    print_metrics_line("stored y_pred (~0.5)", test_stored)
    print_metrics_line(f"global thr = {t_global:.2f}", test_global)
    print_metrics_line("per-task thresholds", test_pertask)

    # Per-task breakdown on test for each config.
    breakdown = pd.DataFrame(
        [
            {
                "task": task,
                "thr(stored~0.5)": 0.50,
                "F1_stored": per_task_metrics(test["y_true"], test["y_pred"], test["task_type"])[task]["f1"],
                "thr(global)": t_global,
                "F1_global": per_task_metrics(
                    test["y_true"], (test["y_score"].to_numpy() >= t_global).astype(int), test["task_type"]
                )[task]["f1"],
                "thr(per-task)": per_task_thr[task],
                "F1_pertask": per_task_metrics(test["y_true"], pertask_pred, test["task_type"])[task]["f1"],
                "R_pertask": per_task_metrics(test["y_true"], pertask_pred, test["task_type"])[task]["recall"],
            }
            for task in TASK_TYPES
        ]
    )
    print_table(
        "FINAL TEST — per-task F1 by threshold config",
        breakdown,
        [
            "task",
            "thr(stored~0.5)",
            "F1_stored",
            "thr(global)",
            "F1_global",
            "thr(per-task)",
            "F1_pertask",
            "R_pertask",
        ],
    )

    return {
        "global_threshold": t_global,
        "per_task_thresholds": per_task_thr,
        "val_global_best": {k: float(best_global[k]) for k in ["threshold", "precision", "recall", "f1"]},
        "val_per_task_best": {
            t: {k: float(per_task_best[t][k]) for k in ["threshold", "precision", "recall", "f1"]} for t in TASK_TYPES
        },
        "test_stored": {k: test_stored[k] for k in ["precision", "recall", "f1", "n"]},
        "test_global": {k: test_global[k] for k in ["precision", "recall", "f1", "n"]},
        "test_per_task_applied": {k: test_pertask[k] for k in ["precision", "recall", "f1", "n"]},
    }


# --------------------------------------------------------------------------------------
# Rec #4 — Simple 3-system ensemble
# --------------------------------------------------------------------------------------
def build_ensemble_matrix(df: pd.DataFrame, split: str):
    """Aligned (scores, y_true, task_type) for the 3 ensemble systems, joined by row_index.

    Raises if the systems disagree on y_true/task_type at any row_index (misalignment) or
    if any system is missing rows (NaN after pivot) — this is the correctness guard that
    makes the row_index join safe.
    """
    sub = df[(df["system"].isin(ENSEMBLE_SYSTEMS)) & (df["split"] == split)]
    missing = [s for s in ENSEMBLE_SYSTEMS if s not in set(sub["system"])]
    if missing:
        raise SystemExit(f"Missing {split} rows for ensemble systems {missing} — collect them first.")

    scores = sub.pivot(index="row_index", columns="system", values="y_score")
    y_true = sub.pivot(index="row_index", columns="system", values="y_true")
    tasks = sub.pivot(index="row_index", columns="system", values="task_type")
    if scores.isna().any().any():
        raise ValueError(f"Ensemble {split} matrix has gaps: a system is missing some row_index values.")
    ref = ENSEMBLE_SYSTEMS[0]
    for s in ENSEMBLE_SYSTEMS[1:]:
        if not (y_true[ref].to_numpy() == y_true[s].to_numpy()).all():
            raise ValueError(f"y_true disagrees between {ref} and {s} on {split}: systems are misaligned.")
        if not (tasks[ref].to_numpy() == tasks[s].to_numpy()).all():
            raise ValueError(f"task_type disagrees between {ref} and {s} on {split}: systems are misaligned.")
    return scores[ENSEMBLE_SYSTEMS].to_numpy(), y_true[ref].to_numpy(), tasks[ref].to_numpy()


def ensemble_search(scores: np.ndarray, y_true: np.ndarray) -> tuple[pd.DataFrame, dict]:
    """Grid-search weights x threshold on val; return all records and the best config."""
    candidates = weight_grid() + [(1 / 3, 1 / 3, 1 / 3)]  # uniform isn't on the 0.05 grid
    records = []
    for w in candidates:
        ens = scores @ np.asarray(w)
        best = best_by_f1(sweep_thresholds(y_true, ens))
        records.append(
            {
                "w_baseline": round(w[0], 4),
                "w_approach1": round(w[1], 4),
                "w_track_b": round(w[2], 4),
                "threshold": float(best["threshold"]),
                "precision": float(best["precision"]),
                "recall": float(best["recall"]),
                "f1": float(best["f1"]),
            }
        )
    df = pd.DataFrame(records)
    best = df.loc[df["f1"].idxmax()].to_dict()
    return df, best


def _reference_row(records: pd.DataFrame, w: tuple[float, float, float]) -> pd.DataFrame:
    w = tuple(round(x, 4) for x in w)
    mask = (records["w_baseline"] == w[0]) & (records["w_approach1"] == w[1]) & (records["w_track_b"] == w[2])
    return records[mask]


def ensemble_analysis(df: pd.DataFrame, track_b_global_thr: float) -> dict:
    print("\n" + "=" * 78)
    print("REC #4 — SIMPLE 3-SYSTEM ENSEMBLE  (tuned on VAL, applied to TEST once)")
    print(f"systems: {ENSEMBLE_SYSTEMS}  (track_a_deberta excluded — val alignment)")
    print("=" * 78)

    val_scores, val_y, _ = build_ensemble_matrix(df, "val")
    records, best = ensemble_search(val_scores, val_y)

    # Top val configs + named reference rows.
    top = records.sort_values("f1", ascending=False).head(10)
    cols = ["w_baseline", "w_approach1", "w_track_b", "threshold", "precision", "recall", "f1"]
    print_table("VAL — top 10 weight/threshold configs", top, cols)

    refs = pd.concat(
        [
            _reference_row(records, (1 / 3, 1 / 3, 1 / 3)).assign(label="uniform"),
            _reference_row(records, (0.25, 0.25, 0.50)).assign(label="track_b_heavy"),
            _reference_row(records, (0.0, 0.0, 1.0)).assign(label="track_b_only"),
        ],
        ignore_index=True,
    )
    print_table("VAL — reference weightings", refs, ["label"] + cols)

    w_best = (best["w_baseline"], best["w_approach1"], best["w_track_b"])
    thr_best = best["threshold"]
    print(
        f"\n  -> best val ensemble: weights={tuple(round(x,3) for x in w_best)} "
        f"threshold={thr_best:.2f}  val F1={best['f1']:.4f}"
    )

    # ---- Apply ONCE to test ----
    test_scores, test_y, test_tasks = build_ensemble_matrix(df, "test")
    ens_test = test_scores @ np.asarray(w_best)
    ens_pred = (ens_test >= thr_best).astype(int)
    ens_metrics = response_level_metrics(test_y, ens_pred)

    # Track B alone on test: stored y_pred and Track B at its tuned global threshold.
    tb = df[(df["system"] == SYSTEM_TRACK_B) & (df["split"] == "test")]
    tb_stored = response_level_metrics(tb["y_true"].to_numpy(), tb["y_pred"].to_numpy())
    tb_tuned = metrics_at_threshold(tb["y_true"], tb["y_score"], track_b_global_thr)

    print("\n" + "-" * 78)
    print("FINAL TEST (applied once) — ensemble vs Track B alone")
    print("-" * 78)
    print_metrics_line(f"ensemble w={tuple(round(x,2) for x in w_best)} thr={thr_best:.2f}", ens_metrics)
    print_metrics_line("Track B alone (stored ~0.5)", tb_stored)
    print_metrics_line(f"Track B alone (thr={track_b_global_thr:.2f})", tb_tuned)

    ens_per_task = per_task_metrics(test_y, ens_pred, test_tasks)
    per_task_df = pd.DataFrame(
        [
            {
                "task": t,
                "n": ens_per_task[t]["n"],
                "precision": ens_per_task[t]["precision"],
                "recall": ens_per_task[t]["recall"],
                "f1": ens_per_task[t]["f1"],
            }
            for t in TASK_TYPES
        ]
    )
    print_table("FINAL TEST — ensemble per-task breakdown", per_task_df, ["task", "n", "precision", "recall", "f1"])

    return {
        "systems": ENSEMBLE_SYSTEMS,
        "excluded": [SYSTEM_TRACK_A],
        "best_val": {"weights": [round(x, 4) for x in w_best], "threshold": thr_best, "f1": float(best["f1"])},
        "test_ensemble": {k: ens_metrics[k] for k in ["precision", "recall", "f1", "n"]},
        "test_ensemble_per_task": ens_per_task,
        "track_b_alone_stored": {k: tb_stored[k] for k in ["precision", "recall", "f1", "n"]},
        "track_b_alone_tuned": {
            "threshold": track_b_global_thr,
            **{k: tb_tuned[k] for k in ["precision", "recall", "f1", "n"]},
        },
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--unified_path", default=str(UNIFIED_PREDICTIONS_PATH))
    p.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = load_predictions(args.unified_path)
    if "val" not in set(df["split"]):
        raise SystemExit("Unified table has no val rows — run collect_predictions.py --split val first.")

    threshold_result = track_b_threshold_tuning(df)
    ensemble_result = ensemble_analysis(df, track_b_global_thr=threshold_result["global_threshold"])

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"threshold_tuning": threshold_result, "ensemble": ensemble_result}, indent=2),
        encoding="utf-8",
    )
    print(f"\nWrote {output}")


if __name__ == "__main__":
    main()
