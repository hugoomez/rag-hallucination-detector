"""Diagnose WHY the zero-shot NLI baseline over-flags, using only cached val scores.

Runs entirely on results/nli_scores_val.json + results/baseline_nli_metrics.json — no model,
no GPU. Recomputes per-sentence flags at the tuned thresholds and inspects, separately for
faithful (label_response==0) and hallucinated (label_response==1) rows, how often sentences
are flagged and how entailment is distributed. Also tests whether a proportion-of-unsupported
decision rule beats the current "any unsupported sentence -> hallucinated" rule.
"""

import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

import numpy as np  # noqa: E402
from sklearn.metrics import precision_recall_fscore_support  # noqa: E402

from src.models.nli_baseline import flag_from_scores  # noqa: E402

METRICS_PATH = "results/baseline_nli_metrics.json"
VAL_SCORES_PATH = "results/nli_scores_val.json"
PROPORTION_THRESHOLDS = [0.1, 0.25, 0.5, 0.75]


def row_flags(row: dict, ent_thr: float, con_thr: float) -> list[str]:
    return [flag_from_scores(ent, con, ent_thr, con_thr) for ent, con in row["sentence_scores"]]


def prf(y_true: list[int], y_pred: list[int]) -> tuple[float, float, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0
    )
    return float(precision), float(recall), float(f1)


def main() -> None:
    best = json.load(open(METRICS_PATH, encoding="utf-8"))["best_thresholds"]
    ent_thr, con_thr = best["ent_thr"], best["con_thr"]
    rows = json.load(open(VAL_SCORES_PATH, encoding="utf-8"))
    print(f"Loaded {len(rows)} val rows | best thresholds: ent_thr={ent_thr} con_thr={con_thr}\n")

    # Precompute flags once per row.
    for row in rows:
        row["_flags"] = row_flags(row, ent_thr, con_thr)

    groups = {
        "faithful (label=0)": [r for r in rows if r["label_response"] == 0],
        "hallucinated (label=1)": [r for r in rows if r["label_response"] == 1],
    }

    for name, group in groups.items():
        n = len(group)
        with_unverifiable = sum(1 for r in group if "unverifiable" in r["_flags"])
        with_contradicted = sum(1 for r in group if "contradicted" in r["_flags"])
        sent_counts = [len(r["_flags"]) for r in group]
        all_flags = [flag for r in group for flag in r["_flags"]]
        total_sentences = len(all_flags)

        print(f"=== {name}: {n} rows ===")
        print(f"  rows with >=1 unverifiable sentence : {with_unverifiable / n:6.1%}  ({with_unverifiable}/{n})")
        print(f"  rows with >=1 contradicted sentence : {with_contradicted / n:6.1%}  ({with_contradicted}/{n})")
        print(f"  avg sentences per response          : {np.mean(sent_counts):.2f}")
        print(f"  sentence-level flag breakdown ({total_sentences} sentences):")
        for flag in ("supported", "unverifiable", "contradicted"):
            count = all_flags.count(flag)
            share = count / total_sentences if total_sentences else 0.0
            print(f"      {flag:12s}: {share:6.1%}  ({count})")
        print()

    # max_entailment distribution across ALL sentences of faithful rows.
    faithful_ents = [ent for r in groups["faithful (label=0)"] for ent, _con in r["sentence_scores"]]
    pct = np.percentile(faithful_ents, [0, 25, 50, 75, 100])
    print(f"=== max_entailment over ALL faithful-row sentences ({len(faithful_ents)} sentences) ===")
    print(f"  min={pct[0]:.3f}  25%={pct[1]:.3f}  median={pct[2]:.3f}  75%={pct[3]:.3f}  max={pct[4]:.3f}\n")

    # Decision-rule experiment: hallucinated if fraction-not-supported exceeds a threshold.
    y_true = [r["label_response"] for r in rows]

    def frac_not_supported(row: dict) -> float:
        flags = row["_flags"]
        if not flags:
            return 0.0
        return sum(1 for f in flags if f != "supported") / len(flags)

    fractions = [frac_not_supported(r) for r in rows]

    print("=== decision rule comparison on val (P / R / F1) ===")
    any_pred = [int(frac > 0.0) for frac in fractions]  # "any not-supported sentence"
    p, r, f1 = prf(y_true, any_pred)
    print(f"  rule=ANY not-supported        P={p:.3f} R={r:.3f} F1={f1:.3f}")
    for thr in PROPORTION_THRESHOLDS:
        pred = [int(frac > thr) for frac in fractions]
        p, r, f1 = prf(y_true, pred)
        print(f"  rule=fraction > {thr:<4}          P={p:.3f} R={r:.3f} F1={f1:.3f}")


if __name__ == "__main__":
    main()
