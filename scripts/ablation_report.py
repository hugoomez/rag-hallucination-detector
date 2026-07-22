"""Stratified ACWS ablation report: compare arms (a)/(b)/(c) on the Track B test set.

Consumes the per-example prediction dumps written by src/models/train_token_level.py (arms
b/c) and scripts/dump_token_predictions.py (arm a), joins each back to the RAW RAGTruth test
slice (implicit_true / due_to_null / label_type live only there), and computes five
stratified comparison blocks per arm:

  (i)   official   -- char-overlap span P/R/F1 + response-level P/R/F1 (the headline numbers,
                      identical scoring to train_token_level so arm a reproduces its published
                      0.5113 span-F1 / 0.7619 response-F1).
  (ii)  clean_span -- same char-span metric with the annotator-flagged noisy (implicit_true,
                      non-due_to_null) intervals subtracted from BOTH gold and predictions.
  (iii) noisy_recall_only -- char-overlap RECALL of predictions against the implicit_true-only
                      gold mass (noisy minus genuine). Precision is meaningless here, so recall
                      only: lower = the arm predicts less over annotator-acknowledged-true text.
  (iv)  by_task_severity -- char-overlap recall per (task_type x Evident/Subtle) gold cell;
                      recall only for the same reason (a predicted span carries no severity).
  (v)   response_precision -- response-level precision overall + false-positive rate among
                      FAITHFUL (resp_true==0) responses -- the paraphrase-FP failure mode.

Then the PRE-REGISTERED decision rule (printed PASS/FAIL, no eyeballing): adopt (c) over (b) iff
  clean_span_f1(c)  > clean_span_f1(b)   AND
  response_f1(c)    > response_f1(b)     AND
  official_span_recall(b) - official_span_recall(c) <= noisy_char_mass_share
where noisy_char_mass_share is the noisy fraction of official gold char mass, computed from the
data at runtime (never hardcoded).

Usage:
    # Gate-4 dry run: arm a alone must reproduce the published headline numbers.
    python scripts/ablation_report.py --arm a=results/token_preds_arm_a.json
    # Full three-arm comparison + decision.
    python scripts/ablation_report.py \
        --arm a=results/token_preds_arm_a.json \
        --arm b=results/token_preds_arm_b.json \
        --arm c=results/token_preds_arm_c.json \
        --out results/ablation_report.json
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


from src.data.preprocess import load_merged_dataframe  # noqa: E402
from src.models.train import acc_prf  # noqa: E402
from src.models.train_token_level import char_span_prf  # noqa: E402

# Published Track B headline numbers (ADR-013 / model card) -- arm a must reproduce these
# through the new pipeline within inference noise, else the eval/join path itself is broken.
PUBLISHED_SPAN_F1 = 0.5113
PUBLISHED_RESPONSE_F1 = 0.7619
REPRODUCTION_TOL = 0.01


# ---------------------------------------------------------------------------
# Pure interval helpers (unit-tested in tests/test_ablation_report.py).
# ---------------------------------------------------------------------------
def subtract_intervals(intervals: list[tuple[int, int]], holes: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Return `intervals` with every `holes` region removed (half-open [start, end)).

    Splits an interval into the pieces left after punching out each overlapping hole;
    zero-width remnants are dropped. Inputs need not be sorted or disjoint. Used to strip
    the noisy (implicit_true) char mass out of both gold and predicted spans before scoring.
    """
    result: list[tuple[int, int]] = []
    for start, end in intervals:
        pieces = [(start, end)]
        for hole_start, hole_end in holes:
            next_pieces: list[tuple[int, int]] = []
            for piece_start, piece_end in pieces:
                if hole_end <= piece_start or hole_start >= piece_end:
                    next_pieces.append((piece_start, piece_end))  # no overlap
                    continue
                if piece_start < hole_start:
                    next_pieces.append((piece_start, hole_start))
                if hole_end < piece_end:
                    next_pieces.append((hole_end, piece_end))
            pieces = next_pieces
        result.extend((s, e) for s, e in pieces if e > s)
    return result


def total_overlap(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> int:
    """Total overlapping char count between two interval lists (each internally disjoint)."""
    overlap = 0
    for a_start, a_end in a:
        for b_start, b_end in b:
            overlap += max(0, min(a_end, b_end) - max(a_start, b_start))
    return overlap


def char_mass(intervals: list[tuple[int, int]]) -> int:
    return sum(end - start for start, end in intervals)


def char_recall(pred: list[list[tuple[int, int]]], gold: list[list[tuple[int, int]]]) -> dict:
    """Micro char-overlap recall of pred against gold, aggregated over all rows."""
    overlap = sum(total_overlap(p, g) for p, g in zip(pred, gold))
    gold_total = sum(char_mass(g) for g in gold)
    recall = overlap / gold_total if gold_total > 0 else 0.0
    return {"recall": float(recall), "gold_char_mass": int(gold_total), "overlap_char_mass": int(overlap)}


def decision_rule(block_b: dict, block_c: dict, noisy_char_mass_share: float) -> dict:
    """Pre-registered adopt-(c)-over-(b) rule; returns each clause + the overall verdict.

    Clauses (all must hold): clean-span F1 up, response F1 up, and official span-recall drop
    within the noisy char-mass share (removing noisy predictions can legitimately shed at most
    that much official recall).
    """
    clean_f1_up = block_c["clean_span"]["f1"] > block_b["clean_span"]["f1"]
    response_f1_up = block_c["official"]["response_level"]["f1"] > block_b["official"]["response_level"]["f1"]
    recall_drop = block_b["official"]["span_char_level"]["recall"] - block_c["official"]["span_char_level"]["recall"]
    recall_drop_ok = recall_drop <= noisy_char_mass_share
    return {
        "clean_span_f1_improved": bool(clean_f1_up),
        "response_f1_improved": bool(response_f1_up),
        "official_span_recall_drop": float(recall_drop),
        "noisy_char_mass_share": float(noisy_char_mass_share),
        "recall_drop_within_noisy_share": bool(recall_drop_ok),
        "adopt_c_over_b": bool(clean_f1_up and response_f1_up and recall_drop_ok),
    }


# ---------------------------------------------------------------------------
# Raw-data join + per-row metadata.
# ---------------------------------------------------------------------------
def is_noisy(span: dict) -> bool:
    """Matches preprocess_token_level.is_noisy_span: implicit_true and NOT due_to_null."""
    return bool(span.get("implicit_true", False)) and not bool(span.get("due_to_null", False))


def severity(span: dict) -> str:
    """Evident vs Subtle from the label_type prefix (e.g. 'Subtle Baseless Info')."""
    label_type = str(span.get("label_type", ""))
    if label_type.startswith("Evident"):
        return "Evident"
    if label_type.startswith("Subtle"):
        return "Subtle"
    return "Other"


def build_test_meta() -> list[dict]:
    """Per test row (in the canonical parquet/row_index order): raw span-derived metadata.

    Mirrors notebooks/03_error_analysis.ipynb: load_merged_dataframe(), take split=='test',
    reset_index -- positional index == every arm's row_index. Returns, per row: source_id,
    response_id (raw `id`), task_type, and the noisy / clean / noisy-only / per-severity
    interval lists derived from the RAW (pre-normalization) span annotations.
    """
    merged = load_merged_dataframe()
    test = merged[merged["split"] == "test"].reset_index(drop=True)
    meta = []
    for _, row in test.iterrows():
        spans = row["labels"] if isinstance(row["labels"], list) else []
        noisy = [(int(s["start"]), int(s["end"])) for s in spans if is_noisy(s)]
        clean = [(int(s["start"]), int(s["end"])) for s in spans if not is_noisy(s)]
        # implicit_true-only mass: noisy regions NOT backed by any genuine span (a token
        # backed by a real annotation keeps full weight -- same rule as the training flag).
        noisy_only = subtract_intervals(noisy, clean)
        by_sev: dict[str, list[tuple[int, int]]] = {"Evident": [], "Subtle": [], "Other": []}
        for s in spans:
            by_sev[severity(s)].append((int(s["start"]), int(s["end"])))
        meta.append(
            {
                "source_id": int(row["source_id"]),
                "response_id": str(row["id"]),
                "task_type": str(row["task_type"]),
                "noisy": noisy,
                "clean": clean,
                "noisy_only": noisy_only,
                "by_severity": by_sev,
            }
        )
    return meta


def load_arm(path: str, meta: list[dict]) -> list[dict]:
    """Load one arm's prediction dump, sort by row_index, and assert it aligns to `meta`."""
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    records = sorted(records, key=lambda r: r["row_index"])
    assert len(records) == len(meta), f"{path}: {len(records)} rows != {len(meta)} test rows"
    for i, (rec, m) in enumerate(zip(records, meta)):
        assert rec["row_index"] == i, f"{path}: row_index not contiguous at {i}"
        assert rec["source_id"] == m["source_id"], f"{path}: source_id mismatch at row {i}"
        if "response_id" in rec:
            assert str(rec["response_id"]) == m["response_id"], f"{path}: response_id mismatch at row {i}"
    return records


def as_spans(record_spans) -> list[tuple[int, int]]:
    return [(int(s), int(e)) for s, e in record_spans]


# ---------------------------------------------------------------------------
# The five stratified blocks.
# ---------------------------------------------------------------------------
def compute_blocks(records: list[dict], meta: list[dict]) -> dict:
    pred = [as_spans(r["pred_spans"]) for r in records]
    gold = [as_spans(r["gold_spans"]) for r in records]
    resp_true = [int(r["resp_true"]) for r in records]
    resp_pred = [int(r["resp_pred"]) for r in records]
    noisy = [m["noisy"] for m in meta]
    noisy_only = [m["noisy_only"] for m in meta]

    # (i) official
    official = {
        "span_char_level": char_span_prf(pred, gold),
        "response_level": acc_prf(resp_true, resp_pred),
    }

    # (ii) clean-span: strip noisy mass from both gold and predictions.
    gold_clean = [subtract_intervals(g, n) for g, n in zip(gold, noisy)]
    pred_clean = [subtract_intervals(p, n) for p, n in zip(pred, noisy)]
    clean_span = char_span_prf(pred_clean, gold_clean)

    # (iii) recall over implicit_true-only gold.
    noisy_recall_only = char_recall(pred, noisy_only)

    # (iv) per task_type x severity char-overlap recall.
    by_task_severity: dict = {}
    task_types = sorted({m["task_type"] for m in meta})
    for task in task_types:
        by_task_severity[task] = {}
        for sev in ("Evident", "Subtle"):
            cell_gold = [m["by_severity"][sev] if m["task_type"] == task else [] for m in meta]
            if sum(char_mass(g) for g in cell_gold) == 0:
                continue
            by_task_severity[task][sev] = char_recall(pred, cell_gold)

    # (v) response-level precision + FP rate among faithful.
    tp = sum(1 for t, p in zip(resp_true, resp_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(resp_true, resp_pred) if t == 0 and p == 1)
    n_faithful = sum(1 for t in resp_true if t == 0)
    response_precision = {
        "precision": float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0,
        "false_positive_rate_faithful": float(fp / n_faithful) if n_faithful > 0 else 0.0,
        "n_faithful": int(n_faithful),
        "n_false_positive": int(fp),
    }

    return {
        "official": official,
        "clean_span": clean_span,
        "noisy_recall_only": noisy_recall_only,
        "by_task_severity": by_task_severity,
        "response_precision": response_precision,
    }


def noisy_char_mass_share(records: list[dict], meta: list[dict]) -> float:
    """Noisy fraction of official gold char mass: intersect(gold, noisy) / gold, over all rows."""
    gold = [as_spans(r["gold_spans"]) for r in records]
    noisy = [m["noisy"] for m in meta]
    gold_total = sum(char_mass(g) for g in gold)
    noisy_in_gold = sum(total_overlap(g, n) for g, n in zip(gold, noisy))
    return noisy_in_gold / gold_total if gold_total > 0 else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--arm",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="Arm predictions dump, e.g. a=results/token_preds_arm_a.json. Repeatable.",
    )
    parser.add_argument("--out", default="results/ablation_report.json", help="Output report JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    arms = {}
    for spec in args.arm:
        name, _, path = spec.partition("=")
        if not path:
            raise ValueError(f"--arm expects NAME=PATH, got {spec!r}")
        arms[name] = path

    print("Building raw test-slice metadata (implicit_true / due_to_null / label_type) ...", flush=True)
    meta = build_test_meta()
    print(f"Test rows: {len(meta)}", flush=True)

    report: dict = {"arms": {}, "noisy_char_mass_share": None, "decision": None}
    share = None
    for name, path in arms.items():
        records = load_arm(path, meta)
        blocks = compute_blocks(records, meta)
        report["arms"][name] = blocks
        if share is None:
            share = noisy_char_mass_share(records, meta)
        _print_arm(name, blocks)

    report["noisy_char_mass_share"] = float(share) if share is not None else None
    print(f"\nnoisy char-mass share of official gold: {share:.4f}", flush=True)

    # Gate-4 reproduction check whenever arm a is present.
    if "a" in arms:
        _check_reproduction(report["arms"]["a"])

    # Decision rule requires both b and c.
    if "b" in arms and "c" in arms:
        decision = decision_rule(report["arms"]["b"], report["arms"]["c"], share)
        report["decision"] = decision
        _print_decision(decision)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote report -> {out_path}", flush=True)


def _print_arm(name: str, blocks: dict) -> None:
    official = blocks["official"]
    print(f"\n=== arm {name} ===", flush=True)
    print(f"  official span (char)   : {official['span_char_level']}", flush=True)
    print(
        f"  official response      : P/R/F1 "
        f"{official['response_level']['precision']:.4f}/"
        f"{official['response_level']['recall']:.4f}/{official['response_level']['f1']:.4f}",
        flush=True,
    )
    print(f"  clean-span (char)      : {blocks['clean_span']}", flush=True)
    print(f"  noisy recall-only      : {blocks['noisy_recall_only']}", flush=True)
    print(f"  response precision     : {blocks['response_precision']}", flush=True)
    print("  by task_type x severity (char recall):", flush=True)
    for task, sevs in blocks["by_task_severity"].items():
        for sev, m in sevs.items():
            print(f"    {task:10s} {sev:8s}: recall={m['recall']:.4f} (gold_mass={m['gold_char_mass']})", flush=True)


def _check_reproduction(blocks: dict) -> None:
    span_f1 = blocks["official"]["span_char_level"]["f1"]
    resp_f1 = blocks["official"]["response_level"]["f1"]
    span_ok = abs(span_f1 - PUBLISHED_SPAN_F1) <= REPRODUCTION_TOL
    resp_ok = abs(resp_f1 - PUBLISHED_RESPONSE_F1) <= REPRODUCTION_TOL
    print("\n=== ARM-A REPRODUCTION GATE ===", flush=True)
    print(
        f"  span-F1     : {span_f1:.4f} vs published {PUBLISHED_SPAN_F1} -> {'PASS' if span_ok else 'FAIL'}", flush=True
    )
    print(
        f"  response-F1 : {resp_f1:.4f} vs published {PUBLISHED_RESPONSE_F1} -> {'PASS' if resp_ok else 'FAIL'}",
        flush=True,
    )
    if not (span_ok and resp_ok):
        print("  *** REPRODUCTION FAILED -- do NOT train arms b/c; the eval/join pipeline is suspect. ***", flush=True)


def _print_decision(decision: dict) -> None:
    print("\n=== PRE-REGISTERED DECISION RULE (adopt c over b) ===", flush=True)
    for key in ("clean_span_f1_improved", "response_f1_improved", "recall_drop_within_noisy_share"):
        print(f"  {key}: {decision[key]}", flush=True)
    print(
        f"  (official span-recall drop {decision['official_span_recall_drop']:.4f} "
        f"vs noisy share {decision['noisy_char_mass_share']:.4f})",
        flush=True,
    )
    print(f"  ==> adopt_c_over_b: {decision['adopt_c_over_b']}", flush=True)


if __name__ == "__main__":
    main()
