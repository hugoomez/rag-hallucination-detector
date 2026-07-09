"""Fine-tune ModernBERT-base for response-level hallucination classification (Approach 1).

Parallel to src/models/train.py (DeBERTa-v3-base at max_length=512), but consumes the
truncation-free parquets from src/data/preprocess_modernbert.py
(data/processed/response_level_modernbert_{train,val,test}.parquet, ADR-011: 0% of RAGTruth
rows exceed ModernBERT's 4096-token budget, max combined length 2618). The shared training
logic -- WeightedTrainer, split loading, class weights from actual train counts, per-epoch
metrics, the test report with per-task_type breakdown and trivial baselines, and the guarded
Hub push -- is imported from train.py rather than duplicated, mirroring how
preprocess_modernbert.py reuses preprocess.py.

Differences from train.py are driven by 4096-token sequences on a Kaggle T4 (16 GB, Turing):
attn_implementation="sdpa" (FlashAttention 2 unsupported on Turing), fp16 (no bf16), small
per-device batches with gradient accumulation (effective batch 16, matching Track A), and
gradient checkpointing on by default. Memory behavior at 4096 tokens is unproven on the T4,
so run a smoke test (--max_train_samples/--max_eval_samples) before any full run.

Writes results/finetuned_approach1_modernbert_metrics.json in the same schema as
results/baseline_nli_metrics.json and results/finetuned_track_a_metrics.json for a direct
3-way comparison.

Example:
    python src/models/train_modernbert.py --max_train_samples 64 --max_eval_samples 64 --num_train_epochs 1
    python src/models/train_modernbert.py --push_to_hub --hub_model_id <user>/modernbert-ragtruth-approach1
"""

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

import numpy as np  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    TrainingArguments,
    set_seed,
)

from src.models.train import (  # noqa: E402
    DEFAULT_SEED,
    ID2LABEL,
    LABEL2ID,
    NUM_LABELS,
    RESULTS_DIR,
    WeightedTrainer,
    build_test_report,
    compute_class_weights,
    compute_metrics,
    load_split,
    maybe_push_to_hub,
)

MODEL_NAME = "answerdotai/ModernBERT-base"
ATTN_IMPLEMENTATION = "sdpa"
METRICS_PATH = RESULTS_DIR / "finetuned_approach1_modernbert_metrics.json"


def parse_args() -> argparse.Namespace:
    """Expose all hyperparameters, paths, and flags so runs are reproducible and rerunnable."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train_path", default="data/processed/response_level_modernbert_train.parquet")
    parser.add_argument("--val_path", default="data/processed/response_level_modernbert_val.parquet")
    parser.add_argument("--test_path", default="data/processed/response_level_modernbert_test.parquet")
    parser.add_argument("--model_name", default=MODEL_NAME, help="Base encoder checkpoint.")
    parser.add_argument(
        "--output_dir", default="models/finetuned_approach1_modernbert", help="Where the best model + tokenizer go."
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Kept equal to Track A for comparability; ModernBERT literature often uses higher (5e-5 to 8e-5).",
    )
    parser.add_argument("--per_device_train_batch_size", type=int, default=4, help="4096-token sequences on a T4.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument(
        "--gradient_accumulation_steps", type=int, default=4, help="Effective train batch = 16, matching Track A."
    )
    parser.add_argument(
        "--num_train_epochs", type=float, default=5.0, help="Upper bound; early stopping may cut short."
    )
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--early_stopping_patience", type=int, default=2, help="Epochs without val-F1 gain.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--fp16",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Mixed precision (default on; the T4 has no bf16); use --no-fp16 for a CPU smoke test.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Trade compute for memory (default on for the T4); --no-gradient_checkpointing on larger GPUs.",
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Off by default; Kaggle-only, needs an HF token.")
    parser.add_argument("--hub_model_id", default=None, help="Target repo id; required when --push_to_hub is set.")
    parser.add_argument(
        "--max_train_samples", type=int, default=None, help="Cap train rows for a quick smoke test (default: all)."
    )
    parser.add_argument(
        "--max_eval_samples", type=int, default=None, help="Cap val rows for a quick smoke test (default: all)."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    RESULTS_DIR.mkdir(exist_ok=True)

    train_ds, train_df = load_split(args.train_path)
    val_ds, _ = load_split(args.val_path)
    test_ds, test_df = load_split(args.test_path)

    # Optional smoke-test caps (train/val only -- test is never artificially shrunk).
    if args.max_train_samples is not None:
        n = min(args.max_train_samples, len(train_ds))
        train_ds = train_ds.select(range(n))
        train_df = train_df.iloc[:n].reset_index(drop=True)
        print(f"WARNING: using only {n} training rows (smoke test mode)", flush=True)
    if args.max_eval_samples is not None:
        n = min(args.max_eval_samples, len(val_ds))
        val_ds = val_ds.select(range(n))
        print(f"WARNING: using only {n} eval rows (smoke test mode)", flush=True)

    test_task_types = test_df["task_type"].tolist()
    counts = {"train_rows": len(train_ds), "val_rows": len(val_ds), "test_rows": len(test_ds)}
    print(f"rows -> train {counts['train_rows']} | val {counts['val_rows']} | test {counts['test_rows']}", flush=True)

    class_weights = compute_class_weights(train_df["label_response"].tolist())
    label_counts = np.bincount(train_df["label_response"].tolist(), minlength=NUM_LABELS)
    print(
        f"train labels -> faithful(0)={label_counts[0]} hallucinated(1)={label_counts[1]} | "
        f"class_weights={class_weights.tolist()}",
        flush=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        attn_implementation=ATTN_IMPLEMENTATION,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        fp16=args.fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=1,
        seed=args.seed,
        logging_strategy="epoch",
        report_to="none",
        # push_to_hub deliberately NOT set (defaults False): no auto-push of intermediate
        # checkpoints. All Hub pushing happens once via maybe_push_to_hub() at the end.
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
    )

    trainer.train()

    # First and only touch of the test split.
    report = build_test_report(args, trainer, class_weights, counts, val_ds, test_ds, test_task_types)
    report["hyperparameters"]["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    report["hyperparameters"]["gradient_checkpointing"] = args.gradient_checkpointing
    report["hyperparameters"]["attn_implementation"] = ATTN_IMPLEMENTATION

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    METRICS_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n=== TEST results (best checkpoint by val F1) ===", flush=True)
    print(f"  finetuned           : {report['test']['finetuned']}", flush=True)
    print(f"  always-hallucinated : {report['test']['always_hallucinated']}", flush=True)
    print(f"  random              : {report['test']['random']}", flush=True)
    print("  per task_type:", flush=True)
    for task_type, task_metrics in report["test"]["per_task_type"].items():
        print(f"    {task_type:8s}: {task_metrics}", flush=True)
    print(f"\nSaved: {METRICS_PATH} | model + tokenizer -> {args.output_dir}", flush=True)

    maybe_push_to_hub(trainer, tokenizer, args)


if __name__ == "__main__":
    main()
