"""Dump arm (a)'s per-example test predictions without retraining.

Arm (a) of the ACWS ablation (docs/adr/ADR-020) is the ALREADY-published Track B model
(hugoomezz/modernbert-ragtruth-token-level-binary). Arms (b) and (c) are freshly trained
and write their prediction dumps directly from src/models/train_token_level.py's main().
This script gives arm (a) an equivalent dump by running the published Hub model through
the exact same inference + record-building path (build_prediction_records), so all three
arms feed scripts/ablation_report.py in one identical schema. It doubles as the way to
score ANY already-trained checkpoint (e.g. a local seed-run checkpoint recovered after a
crash, with no --metrics_out from its own training run) on equal footing with runs that
came straight out of train_token_level.py's --metrics_out.

The prediction records are byte-for-byte the same shape train_token_level writes:
row_index, source_id, response_id, task_type, pred_spans, gold_spans, gold_token_spans,
resp_true, resp_pred.

When --metrics_out is given (requires --seed), also writes a metrics report in the exact
schema scripts/aggregate_seeds.py expects (hyperparameters.seed + the test.* metric paths),
built via the same build_metrics_report() train_token_level.py's own --metrics_out uses --
so a checkpoint scored here is directly comparable to seeds trained end-to-end.

Usage (CPU is fine; ~2700 test rows):
    python scripts/dump_token_predictions.py
    python scripts/dump_token_predictions.py --model <hub_or_local> --out results/token_preds_arm_a.json
    python scripts/dump_token_predictions.py --model models/large_seed123/checkpoint-6792 \\
        --seed 123 --metrics_out results/large_seed123_metrics.json \\
        --out results/token_preds_large_seed123.json
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from transformers import (  # noqa: E402
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    TrainingArguments,
)

from src.models.train_token_level import (  # noqa: E402
    ImplicitMaskCollator,
    WeightedTokenTrainer,
    _predict_token_labels,
    build_metrics_report,
    build_prediction_records,
    load_token_split,
)

DEFAULT_MODEL = "hugoomezz/modernbert-ragtruth-token-level-binary"
DEFAULT_TEST_PATH = "data/processed/token_level_binary_test.parquet"
DEFAULT_OUT = "results/token_preds_arm_a.json"

# config.json's model_type -> a known-good tokenizer repo, keyed for a checkpoint whose OWN
# tokenizer_config.json a locally-pinned transformers can't parse (raises "Tokenizer class
# TokenizersBackend does not exist" -- seen on both the published Hub repo and every local
# checkpoint train_token_level.py saves). answerdotai/ModernBERT-base's tokenizer.json is
# byte-for-byte identical to answerdotai/ModernBERT-large's (verified: matching MD5 across
# tokenizer.json, tokenizer_config.json, special_tokens_map.json), so one entry covers both
# ModernBERT sizes; it's a model_type -> tokenizer map (not per-size) for exactly that reason.
MODEL_FAMILY_TOKENIZER_FALLBACK: dict[str, str] = {
    "modernbert": "answerdotai/ModernBERT-base",
}


def load_tokenizer(model_path: str, tokenizer_id: str | None):
    """Load the tokenizer for `model_path`, working around the TokenizersBackend bug.

    --tokenizer_id (if given) always wins -- an explicit caller override, matching
    scripts/collect_predictions.py's --tokenizer_id convention. Otherwise try model_path's
    own tokenizer first; only on the specific "tokenizer_class does not exist" failure do we
    consult MODEL_FAMILY_TOKENIZER_FALLBACK (keyed by AutoConfig's model_type, so it works
    for local paths and Hub ids alike). Any other error, or a model_type with no mapped
    fallback, is re-raised rather than guessed at.
    """
    if tokenizer_id:
        return AutoTokenizer.from_pretrained(tokenizer_id)
    try:
        return AutoTokenizer.from_pretrained(model_path)
    except ValueError as e:
        if "does not exist or is not currently imported" not in str(e):
            raise
        model_type = AutoConfig.from_pretrained(model_path).model_type
        fallback = MODEL_FAMILY_TOKENIZER_FALLBACK.get(model_type)
        if fallback is None:
            raise
        print(
            f"WARNING: {model_path}'s own tokenizer_config.json is incompatible with the "
            f"installed transformers ({e}); falling back to {fallback} ({model_type} family).",
            flush=True,
        )
        return AutoTokenizer.from_pretrained(fallback)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hub repo id or local path (arm a's published model).")
    parser.add_argument(
        "--tokenizer_id",
        default=None,
        help="Explicit tokenizer override, taking priority over --model and the automatic "
        "MODEL_FAMILY_TOKENIZER_FALLBACK lookup.",
    )
    parser.add_argument("--test_path", default=DEFAULT_TEST_PATH, help="Token-level test parquet.")
    parser.add_argument("--out", default=DEFAULT_OUT, help="Output predictions JSON.")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Labels hyperparameters.seed in --metrics_out's report. Purely descriptive -- "
        "this model is already trained, so it has zero effect on inference. Required "
        "whenever --metrics_out is set.",
    )
    parser.add_argument(
        "--metrics_out",
        default=None,
        help="Optional metrics report JSON, in the exact schema scripts/aggregate_seeds.py "
        "expects (hyperparameters.seed + test.{span_char_level,response_level_derived,"
        "span_exact_match,per_task_type,...}). Not written unless set. Requires --seed.",
    )
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    args = parser.parse_args(argv)
    if args.metrics_out is not None and args.seed is None:
        parser.error("--metrics_out requires --seed (aggregate_seeds.py needs a real hyperparameters.seed)")
    return args


def main() -> None:
    args = parse_args()

    # No implicit mask needed: inference never touches the loss. load_token_split without
    # with_implicit_mask keeps the eval path identical to arms b/c's test evaluation.
    test_ds, test_df = load_token_split(args.test_path)
    print(f"Loaded {len(test_df)} test rows from {args.test_path}", flush=True)

    tokenizer = load_tokenizer(args.model, args.tokenizer_id)
    model = AutoModelForTokenClassification.from_pretrained(args.model)

    with tempfile.TemporaryDirectory() as tmp_dir:
        training_args = TrainingArguments(
            output_dir=tmp_dir,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            report_to="none",
        )
        trainer = WeightedTokenTrainer(
            model=model,
            args=training_args,
            data_collator=ImplicitMaskCollator(tokenizer=tokenizer),
            processing_class=tokenizer,
            # Same GPU-side argmax as training so predict() returns (n, seq) ids, not logits.
            # For 2 classes argmax == LettuceDetect's p >= 0.5 threshold.
            preprocess_logits_for_metrics=lambda logits, labels: logits.argmax(dim=-1),
        )

        test_labels, test_preds = _predict_token_labels(trainer, test_ds)

    records = build_prediction_records(test_labels, test_preds, test_df)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(records, indent=2), encoding="utf-8")

    n_pred_pos = sum(1 for r in records if r["resp_pred"] == 1)
    n_true_pos = sum(1 for r in records if r["resp_true"] == 1)
    print(
        f"Wrote {len(records)} prediction records -> {out_path} "
        f"(resp_pred positives={n_pred_pos}, resp_true positives={n_true_pos})",
        flush=True,
    )

    if args.metrics_out is not None:
        report = {
            "model_name": args.model,
            "hyperparameters": {"seed": args.seed},
            "test": build_metrics_report(test_labels, test_preds, test_df, args.seed),
        }
        metrics_path = Path(args.metrics_out)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Wrote metrics report -> {metrics_path}", flush=True)
        print(f"  span_char_level        : {report['test']['span_char_level']}", flush=True)
        print(f"  response_level_derived : {report['test']['response_level_derived']}", flush=True)
        print(f"  span_exact_match       : {report['test']['span_exact_match']}", flush=True)


if __name__ == "__main__":
    main()
