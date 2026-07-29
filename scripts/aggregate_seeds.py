"""Aggregate multi-seed Track B runs and report a matched (paired) base-vs-large comparison.

Consumes the per-run metrics JSONs written by src/models/train_token_level.py (the dict with
top-level model_name / hyperparameters / counts / best_checkpoint / val / test). Given N seed
runs for each backbone -- ModernBERT-base (arm-b recipe) and ModernBERT-large -- it reports,
for every headline test metric:

  * per model: the raw per-seed values, their mean, and their min/max range;
  * paired (large - base), matched BY SEED (same seed => same weight init draw, same data
    shuffle, only the backbone differs): the per-seed deltas, their mean, their min/max range,
    and on how many seeds large scored higher.

Deliberately NO p-values and NO significance test. With only ~3 seeds a significance claim is
not defensible; this script reports a descriptive variance summary (raw values, mean, range,
per-seed deltas) and nothing more. Seeds are read from each JSON's hyperparameters.seed, so
filenames are free-form and pairing is content-driven (base<->large matched on seed value).

Usage:
    python scripts/aggregate_seeds.py \
        --base  results/arm_b_metrics.json \
                results/arm_b_seed123_metrics.json \
                results/arm_b_seed456_metrics.json \
        --large results/modernbert_large_metrics.json \
                results/modernbert_large_seed123_metrics.json \
                results/modernbert_large_seed456_metrics.json \
        --out   results/seed_aggregate.json
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

DEFAULT_OUT = "results/seed_aggregate.json"

# Headline test metrics pulled from each run's "test" block, as flat dotted keys. Order here
# is the order printed and stored; every run is expected to carry all of them.
METRIC_PATHS: dict[str, tuple[str, ...]] = {
    "response.precision": ("response_level_derived", "precision"),
    "response.recall": ("response_level_derived", "recall"),
    "response.f1": ("response_level_derived", "f1"),
    "response.accuracy": ("response_level_derived", "accuracy"),
    "span_char.precision": ("span_char_level", "precision"),
    "span_char.recall": ("span_char_level", "recall"),
    "span_char.f1": ("span_char_level", "f1"),
    "span_exact.f1": ("span_exact_match", "f1"),
    "per_task.Summary.f1": ("per_task_type", "Summary", "f1"),
    "per_task.QA.f1": ("per_task_type", "QA", "f1"),
    "per_task.Data2txt.f1": ("per_task_type", "Data2txt", "f1"),
}

CAVEAT = (
    "n={n} seeds: descriptive variance summary only (raw per-seed values, mean, min/max "
    "range, and per-seed paired deltas). No p-values or significance tests are reported -- "
    "this many seeds cannot support a significance claim."
)


# ---------------------------------------------------------------------------
# Pure aggregation helpers (unit-tested in tests/test_aggregate_seeds.py).
# ---------------------------------------------------------------------------
def summary_stats(values: list[float]) -> dict:
    """Descriptive summary of one metric across seeds: raw values, mean, min, max, range.

    No standard deviation and no interval estimate -- with a handful of seeds only the raw
    spread (min/max range) is honestly interpretable. `range` is max - min (0.0 for n == 1).
    """
    if not values:
        raise ValueError("summary_stats requires at least one value")
    n = len(values)
    lo = min(values)
    hi = max(values)
    return {
        "n": n,
        "values": [float(v) for v in values],
        "mean": float(sum(values) / n),
        "min": float(lo),
        "max": float(hi),
        "range": float(hi - lo),
    }


def _dig(mapping: dict, keys: tuple[str, ...]) -> float:
    """Follow a tuple of nested keys into a dict; raise a clear error if any is missing."""
    node = mapping
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            raise KeyError(f"metric path {'/'.join(keys)} missing at {key!r}")
        node = node[key]
    return float(node)


def extract_metrics(report: dict) -> dict[str, float]:
    """Flatten one run's test block to the flat dotted METRIC_PATHS keys."""
    test = report["test"]
    return {name: _dig(test, path) for name, path in METRIC_PATHS.items()}


def run_seed(report: dict) -> int:
    """The seed a run used -- read from the saved hyperparameters (not the filename)."""
    return int(report["hyperparameters"]["seed"])


def summarize_model(reports: list[dict]) -> dict:
    """Per-metric descriptive summary across a model's seed runs, ordered by seed."""
    ordered = sorted(reports, key=run_seed)
    seeds = [run_seed(r) for r in ordered]
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"duplicate seeds among runs: {seeds}")
    per_run = [extract_metrics(r) for r in ordered]
    per_metric = {name: summary_stats([m[name] for m in per_run]) for name in METRIC_PATHS}
    return {
        "model_name": ordered[0].get("model_name"),
        "seeds": seeds,
        "per_metric": per_metric,
    }


def paired_deltas(base_reports: list[dict], large_reports: list[dict]) -> dict:
    """Matched (large - base) deltas per metric, paired BY SEED value (not list order).

    Requires the two models to have run the exact same set of seeds; raises otherwise, since
    an unmatched seed would break the paired interpretation. Per metric it reports the
    per-seed deltas (ordered by seed), their mean, their min/max range, and n_large_higher =
    how many seeds large scored strictly above base. No p-value / significance test.
    """
    base_by_seed = {run_seed(r): extract_metrics(r) for r in base_reports}
    large_by_seed = {run_seed(r): extract_metrics(r) for r in large_reports}
    if len(base_by_seed) != len(base_reports) or len(large_by_seed) != len(large_reports):
        raise ValueError("duplicate seeds within a model's runs -- cannot pair")
    if set(base_by_seed) != set(large_by_seed):
        raise ValueError(
            f"seed sets differ: base={sorted(base_by_seed)} vs large={sorted(large_by_seed)} "
            "-- paired comparison needs the same seeds on both backbones"
        )
    seeds = sorted(base_by_seed)
    per_metric = {}
    for name in METRIC_PATHS:
        deltas = [large_by_seed[s][name] - base_by_seed[s][name] for s in seeds]
        per_metric[name] = {
            "n": len(seeds),
            "per_seed_delta": [float(d) for d in deltas],
            "mean_delta": float(sum(deltas) / len(deltas)),
            "min_delta": float(min(deltas)),
            "max_delta": float(max(deltas)),
            "n_large_higher": int(sum(1 for d in deltas if d > 0)),
        }
    return {"seeds": seeds, "per_metric": per_metric}


# ---------------------------------------------------------------------------
# I/O + CLI.
# ---------------------------------------------------------------------------
def load_reports(paths: list[str]) -> list[dict]:
    return [json.loads(Path(p).read_text(encoding="utf-8")) for p in paths]


def build_report(base_reports: list[dict], large_reports: list[dict]) -> dict:
    base = summarize_model(base_reports)
    large = summarize_model(large_reports)
    paired = paired_deltas(base_reports, large_reports)
    n = len(paired["seeds"])
    return {
        "seeds": paired["seeds"],
        "n_seeds": n,
        "base": base,
        "large": large,
        "paired_large_minus_base": paired,
        "caveat": CAVEAT.format(n=n),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--base", nargs="+", required=True, metavar="METRICS_JSON", help="ModernBERT-base seed run metrics JSONs."
    )
    parser.add_argument(
        "--large", nargs="+", required=True, metavar="METRICS_JSON", help="ModernBERT-large seed run metrics JSONs."
    )
    parser.add_argument("--out", default=DEFAULT_OUT, help=f"Output aggregate JSON (default: {DEFAULT_OUT}).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_reports = load_reports(args.base)
    large_reports = load_reports(args.large)
    report = build_report(base_reports, large_reports)

    _print_model("base ", report["base"])
    _print_model("large", report["large"])
    _print_paired(report["paired_large_minus_base"])
    print(f"\n{report['caveat']}", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote aggregate -> {out_path}", flush=True)


def _print_model(label: str, summary: dict) -> None:
    print(f"\n=== {label} ({summary['model_name']}) | seeds {summary['seeds']} ===", flush=True)
    for name, stat in summary["per_metric"].items():
        vals = ", ".join(f"{v:.4f}" for v in stat["values"])
        print(
            f"  {name:22s}: mean {stat['mean']:.4f}  range [{stat['min']:.4f}, {stat['max']:.4f}]  ({vals})", flush=True
        )


def _print_paired(paired: dict) -> None:
    print(f"\n=== paired large - base | seeds {paired['seeds']} ===", flush=True)
    for name, stat in paired["per_metric"].items():
        deltas = ", ".join(f"{d:+.4f}" for d in stat["per_seed_delta"])
        print(
            f"  {name:22s}: mean {stat['mean_delta']:+.4f}  "
            f"range [{stat['min_delta']:+.4f}, {stat['max_delta']:+.4f}]  "
            f"large higher on {stat['n_large_higher']}/{stat['n']}  ({deltas})",
            flush=True,
        )


if __name__ == "__main__":
    main()
