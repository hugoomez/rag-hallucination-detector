"""Subtle-only response miss rate under arm-b, with/without implicit_true-only responses.

Follow-on from the implicit_true audit (docs/research/02-implicit-true-audit.md) and its
dilution question, answered in docs/research/04-subtle-only-reconciliation.md: does
excluding implicit_true-flagged responses from the reported Subtle-only miss rate reduce
it? It does not -- excluding them RAISES it (48.1% -> 53.3%). See that doc for the result
and its reading.

implicit_true marks ungrounded-but-TRUE content ("correct while the info is not mentioned
in the context", per RAGTruth's README) -- a correctly-labelled, low-severity subclass of
hallucination, NOT label noise. See docs/research/02 Sec 2.1.

Reuses the exact join-back pattern from scripts/ablation_report.py (build_test_meta /
load_arm): load_merged_dataframe(), filter split=='test', reset_index -- positional index
== row_index in results/token_preds_arm_b.json.

Cohort definition (matches docs/research/02-implicit-true-audit.md Sec 2 and
docs/research/03-candidate-methods.md's 48.1%/30.6% table): a test response is
"Subtle-only" if it has >=1 gold span and EVERY gold span's label_type starts with
"Subtle" (i.e. no Evident spans present). All such responses are gold-hallucinated
(resp_true == 1) by construction (labels non-empty => y_true=1, per
src/data/preprocess.py:compute_label_response).

Rate (a): FN / N over the Subtle-only cohort, exactly as currently reported.
Rate (b): same, but responses whose ENTIRE gold span set is implicit_true==True are
removed from the cohort entirely (both numerator and denominator) -- these are the ones
RAGTruth marks as ungrounded-but-true, so a "miss" on them is arguably correct behaviour
for a detector aimed at consequential hallucinations, not a real detection failure.

Usage:
    python scripts/subtle_only_miss_rate.py
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.preprocess import load_merged_dataframe  # noqa: E402


def is_subtle(label_type: str) -> bool:
    return str(label_type).startswith("Subtle")


def main() -> None:
    merged = load_merged_dataframe()
    test = merged[merged["split"] == "test"].reset_index(drop=True)

    preds = json.loads((REPO_ROOT / "results/token_preds_arm_b.json").read_text(encoding="utf-8"))
    preds = sorted(preds, key=lambda r: r["row_index"])
    assert len(preds) == len(test), f"{len(preds)} preds != {len(test)} test rows"

    rows = []
    for i, (rec, (_, row)) in enumerate(zip(preds, test.iterrows())):
        assert rec["row_index"] == i
        assert rec["source_id"] == int(row["source_id"]), f"source_id mismatch at row {i}"
        spans = row["labels"] if isinstance(row["labels"], list) else []
        rows.append(
            {
                "row_index": i,
                "resp_true": int(rec["resp_true"]),
                "resp_pred": int(rec["resp_pred"]),
                "spans": spans,
            }
        )

    subtle_only = [
        r for r in rows
        if r["spans"] and all(is_subtle(s["label_type"]) for s in r["spans"])
    ]
    # Sanity: Subtle-only responses are gold-hallucinated by construction.
    assert all(r["resp_true"] == 1 for r in subtle_only)

    n_a = len(subtle_only)
    fn_a = sum(1 for r in subtle_only if r["resp_pred"] == 0)
    rate_a = fn_a / n_a if n_a else float("nan")

    all_implicit_true = [
        r for r in subtle_only
        if all(bool(s.get("implicit_true")) for s in r["spans"])
    ]
    n_all_it = len(all_implicit_true)
    fn_all_it = sum(1 for r in all_implicit_true if r["resp_pred"] == 0)

    all_it_indices = {r["row_index"] for r in all_implicit_true}
    filtered = [r for r in subtle_only if r["row_index"] not in all_it_indices]
    n_b = len(filtered)
    fn_b = sum(1 for r in filtered if r["resp_pred"] == 0)
    rate_b = fn_b / n_b if n_b else float("nan")

    print("=== Subtle-only cohort (test, arm-b) ===")
    print(f"Total Subtle-only responses (gold-hallucinated, no Evident spans): {n_a}")
    print(f"  ...of which ALL gold spans implicit_true-flagged: {n_all_it} "
          f"({n_all_it / n_a:.1%})  [FN among these: {fn_all_it}/{n_all_it}]")
    print()
    print("(a) As currently reported -- full Subtle-only cohort:")
    print(f"    miss rate = {fn_a}/{n_a} = {rate_a:.4f} ({rate_a:.1%})")
    print()
    print("(b) Excluding all-implicit_true responses from the cohort:")
    print(f"    miss rate = {fn_b}/{n_b} = {rate_b:.4f} ({rate_b:.1%})")
    print()
    print("(c) All-implicit_true-only subset (for reference):")
    print(f"    miss rate = {fn_all_it}/{n_all_it} = {fn_all_it / n_all_it:.4f} ({fn_all_it / n_all_it:.1%})")
    print()
    print(f"Delta: {rate_a:.1%} -> {rate_b:.1%}  ({(rate_b - rate_a) * 100:+.1f} pp), "
          f"N shrank by {n_a - n_b} responses ({(n_a - n_b) / n_a:.1%} of cohort)")


if __name__ == "__main__":
    main()
