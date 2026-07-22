"""Dump arm (a)'s per-example test predictions without retraining.

Arm (a) of the ACWS ablation (docs/adr/ADR-020) is the ALREADY-published Track B model
(hugoomezz/modernbert-ragtruth-token-level-binary). Arms (b) and (c) are freshly trained
and write their prediction dumps directly from src/models/train_token_level.py's main().
This script gives arm (a) an equivalent dump by running the published Hub model through
the exact same inference + record-building path (build_prediction_records), so all three
arms feed scripts/ablation_report.py in one identical schema.

The prediction records are byte-for-byte the same shape train_token_level writes:
row_index, source_id, response_id, task_type, pred_spans, gold_spans, resp_true, resp_pred.

Usage (CPU is fine; ~2700 test rows):
    python scripts/dump_token_predictions.py
    python scripts/dump_token_predictions.py --model <hub_or_local> --out results/token_preds_arm_a.json
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from transformers import (  # noqa: E402
    AutoModelForTokenClassification,
    AutoTokenizer,
    TrainingArguments,
)

from src.models.train_token_level import (  # noqa: E402
    ImplicitMaskCollator,
    WeightedTokenTrainer,
    _predict_token_labels,
    build_prediction_records,
    load_token_split,
)

DEFAULT_MODEL = "hugoomezz/modernbert-ragtruth-token-level-binary"
DEFAULT_TEST_PATH = "data/processed/token_level_binary_test.parquet"
DEFAULT_OUT = "results/token_preds_arm_a.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hub repo id or local path (arm a's published model).")
    parser.add_argument("--test_path", default=DEFAULT_TEST_PATH, help="Token-level test parquet.")
    parser.add_argument("--out", default=DEFAULT_OUT, help="Output predictions JSON.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # No implicit mask needed: inference never touches the loss. load_token_split without
    # with_implicit_mask keeps the eval path identical to arms b/c's test evaluation.
    test_ds, test_df = load_token_split(args.test_path)
    print(f"Loaded {len(test_df)} test rows from {args.test_path}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
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


if __name__ == "__main__":
    main()
