"""Fine-tune ModernBERT-base for token-level BIO hallucination span detection (Track B).

Consumes the truncation-free BIO-labeled parquets from src/data/preprocess_token_level.py
(data/processed/token_level_modernbert_{train,val,test}.parquet: per-token O/B-HALL/I-HALL
labels on response tokens, -100 on context/special tokens, ADR-011: 0% of rows exceed
ModernBERT's 4096-token budget). Trains AutoModelForTokenClassification with class-weighted
cross-entropy -- weights computed from the ACTUAL per-token label distribution, which is far
more imbalanced (~95% O per docs/notes.md) than the response-level 55/45 -- selects the best
checkpoint by validation span-level F1 (seqeval), then evaluates ONCE on the untouched test
split.

Shared infrastructure (seed, results dir, response-level P/R/F1 helpers, trivial baselines,
per-task breakdown, WeightedTrainer base, guarded Hub push) is imported from train.py; the
exact BIO label scheme is imported from preprocess_token_level.py rather than redefined.
T4 constraints mirror train_modernbert.py: attn_implementation="sdpa" (no FlashAttention 2
on Turing), fp16 (no bf16), small per-device batches with gradient accumulation (effective
batch 16), gradient checkpointing on by default.

Writes results/finetuned_track_b_token_level_metrics.json with seqeval SPAN-level metrics
plus a DERIVED response-level block (a response is "predicted hallucinated" iff any of its
response tokens got a non-O prediction) directly comparable to baseline_nli_metrics.json,
finetuned_track_a_metrics.json, and finetuned_approach1_modernbert_metrics.json.

Example:
    python src/models/train_token_level.py --max_train_samples 64 --max_eval_samples 64 --num_train_epochs 1
    python src/models/train_token_level.py --push_to_hub --hub_model_id <user>/modernbert-ragtruth-track-b
"""

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

import datasets  # noqa: E402
import evaluate  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    TrainingArguments,
    set_seed,
)

from src.data.preprocess_token_level import IGNORE_LABEL, LABEL_NAMES, O_LABEL  # noqa: E402
from src.models.train import (  # noqa: E402
    DEFAULT_SEED,
    RESULTS_DIR,
    WeightedTrainer,
    acc_prf,
    compute_class_weights,
    maybe_push_to_hub,
    per_task_breakdown,
    trivial_baselines,
)

MODEL_NAME = "answerdotai/ModernBERT-base"
ATTN_IMPLEMENTATION = "sdpa"
METRICS_PATH = RESULTS_DIR / "finetuned_track_b_token_level_metrics.json"

# The BIO scheme comes from preprocess_token_level.py; only IGNORE_LABEL (-100) is
# excluded -- it marks context/special tokens, not a class the model predicts.
ID2LABEL = {label_id: name for label_id, name in LABEL_NAMES.items() if label_id != IGNORE_LABEL}
LABEL2ID = {name: label_id for label_id, name in ID2LABEL.items()}
NUM_LABELS = len(ID2LABEL)

seqeval_metric = evaluate.load("seqeval")


def parse_args() -> argparse.Namespace:
    """Expose all hyperparameters, paths, and flags so runs are reproducible and rerunnable."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train_path", default="data/processed/token_level_modernbert_train.parquet")
    parser.add_argument("--val_path", default="data/processed/token_level_modernbert_val.parquet")
    parser.add_argument("--test_path", default="data/processed/token_level_modernbert_test.parquet")
    parser.add_argument("--model_name", default=MODEL_NAME, help="Base encoder checkpoint.")
    parser.add_argument(
        "--output_dir", default="models/finetuned_track_b_token_level", help="Where the best model + tokenizer go."
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
    parser.add_argument("--early_stopping_patience", type=int, default=2, help="Epochs without val span-F1 gain.")
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


def load_token_split(path: str) -> tuple[datasets.Dataset, pd.DataFrame]:
    """Read a token-level parquet into a Dataset plus the raw df (for task_type order).

    Unlike train.load_split, "labels" here is a per-token sequence (not a scalar), already
    aligned to input_ids by preprocess_token_level.py. Everything stays as plain int lists
    so DataCollatorForTokenClassification can pad input_ids/attention_mask AND labels
    (with -100) dynamically per batch.
    """
    df = pd.read_parquet(path)
    for column in ("input_ids", "attention_mask", "labels"):
        df[column] = df[column].apply(lambda a: np.asarray(a).tolist())
    ds = datasets.Dataset.from_pandas(df[["input_ids", "attention_mask", "labels"]], preserve_index=False)
    return ds, df


def flatten_token_labels(label_sequences: list[list[int]]) -> np.ndarray:
    """All real token labels across a split, with IGNORE_LABEL (context/special) dropped."""
    flat = np.concatenate([np.asarray(seq) for seq in label_sequences])
    return flat[flat != IGNORE_LABEL]


def compute_metrics(eval_pred) -> dict:
    """Span-level P/R/F1 via seqeval for Trainer's per-epoch eval.

    Predictions arrive already argmaxed to label ids by preprocess_logits_for_metrics.
    Per example, -100 positions (context/special/padding) are filtered out and ids are
    mapped back to O/B-HALL/I-HALL strings before seqeval reconstructs and scores spans.
    """
    predictions, labels = eval_pred
    predictions = np.asarray(predictions)
    labels = np.asarray(labels)

    true_sequences = [[ID2LABEL[label] for label in label_row if label != IGNORE_LABEL] for label_row in labels]
    pred_sequences = [
        [ID2LABEL[pred] for pred, label in zip(pred_row, label_row) if label != IGNORE_LABEL]
        for pred_row, label_row in zip(predictions, labels)
    ]

    results = seqeval_metric.compute(predictions=pred_sequences, references=true_sequences)
    return {
        "precision": float(results["overall_precision"]),
        "recall": float(results["overall_recall"]),
        "f1": float(results["overall_f1"]),
        "accuracy": float(results["overall_accuracy"]),
    }


def _predict_token_labels(trainer: "WeightedTokenTrainer", ds: datasets.Dataset) -> tuple[np.ndarray, np.ndarray]:
    """Run inference over a split; returns (label_ids, predicted_ids), both (n, seq)."""
    output = trainer.predict(ds)
    return np.asarray(output.label_ids), np.asarray(output.predictions)


def derive_response_labels(labels: np.ndarray, predictions: np.ndarray) -> tuple[list[int], list[int]]:
    """Collapse token-level output to response-level: hallucinated iff ANY response token is non-O.

    Only positions with a real label (!= -100) count -- predictions on context/special/
    padding tokens are meaningless. y_true uses the same any-non-O rule on the gold labels,
    matching how preprocess_token_level.py derived its stratification proxy.
    """
    y_true, y_pred = [], []
    for label_row, pred_row in zip(labels, predictions):
        mask = label_row != IGNORE_LABEL
        y_true.append(int(np.any(label_row[mask] != O_LABEL)))
        y_pred.append(int(np.any(pred_row[mask] != O_LABEL)))
    return y_true, y_pred


def span_level_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict:
    """Seqeval span-level P/R/F1 for a fully predicted split (same filtering as compute_metrics)."""
    metrics = compute_metrics((predictions, labels))
    return {"precision": metrics["precision"], "recall": metrics["recall"], "f1": metrics["f1"]}


def build_token_test_report(
    args: argparse.Namespace,
    trainer: "WeightedTokenTrainer",
    class_weights: torch.Tensor,
    counts: dict,
    val_ds: datasets.Dataset,
    test_ds: datasets.Dataset,
    test_task_types: list[str],
) -> dict:
    """Evaluate the best model on val and (for the first time) test; assemble the metrics dict.

    Same top-level shape as the response-level reports (model_name, hyperparameters, counts,
    val, test) but each split carries BOTH the seqeval span-level numbers and a derived
    response-level block comparable to the other three systems.
    """
    val_labels, val_preds = _predict_token_labels(trainer, val_ds)
    test_labels, test_preds = _predict_token_labels(trainer, test_ds)

    val_resp_true, val_resp_pred = derive_response_labels(val_labels, val_preds)
    test_resp_true, test_resp_pred = derive_response_labels(test_labels, test_preds)
    baselines = trivial_baselines(test_resp_true, args.seed)

    return {
        "model_name": args.model_name,
        "hyperparameters": {
            "learning_rate": args.learning_rate,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "num_train_epochs": args.num_train_epochs,
            "warmup_ratio": args.warmup_ratio,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "class_weights": [round(float(w), 4) for w in class_weights.tolist()],
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "gradient_checkpointing": args.gradient_checkpointing,
            "attn_implementation": ATTN_IMPLEMENTATION,
        },
        "counts": counts,
        "val": {
            "span_level": span_level_metrics(val_labels, val_preds),
            "response_level_derived": acc_prf(val_resp_true, val_resp_pred),
        },
        "test": {
            "span_level": span_level_metrics(test_labels, test_preds),
            "response_level_derived": acc_prf(test_resp_true, test_resp_pred),
            "always_hallucinated": baselines["always_hallucinated"],
            "random": baselines["random"],
            "per_task_type": per_task_breakdown(test_task_types, test_resp_true, test_resp_pred),
        },
    }


class WeightedTokenTrainer(WeightedTrainer):
    """WeightedTrainer variant for token classification: flatten (batch, seq) before the loss."""

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        weight = None if self.class_weights is None else self.class_weights.to(logits.device)
        # ignore_index is CrossEntropyLoss's default, stated explicitly: -100 marks
        # context/special/padding tokens that must never contribute to the loss.
        loss = nn.CrossEntropyLoss(weight=weight, ignore_index=IGNORE_LABEL)(
            logits.view(-1, NUM_LABELS), labels.view(-1)
        )
        return (loss, outputs) if return_outputs else loss


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    RESULTS_DIR.mkdir(exist_ok=True)

    train_ds, train_df = load_token_split(args.train_path)
    val_ds, _ = load_token_split(args.val_path)
    test_ds, test_df = load_token_split(args.test_path)

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

    # Class weights from the ACTUAL per-token distribution (not the ~94.7%/5.3%
    # character-based estimate in docs/notes.md): flatten every real token label
    # in the train split and apply the same inverse-frequency formula as Track A.
    flat_train_labels = flatten_token_labels(train_df["labels"].tolist())
    class_weights = compute_class_weights(flat_train_labels.tolist(), num_labels=NUM_LABELS)
    token_counts = np.bincount(flat_train_labels, minlength=NUM_LABELS)
    total_tokens = int(token_counts.sum())
    counts = {
        "train_rows": len(train_ds),
        "val_rows": len(val_ds),
        "test_rows": len(test_ds),
        "train_token_labels": {ID2LABEL[i]: int(token_counts[i]) for i in range(NUM_LABELS)},
    }
    print(f"rows -> train {counts['train_rows']} | val {counts['val_rows']} | test {counts['test_rows']}", flush=True)
    print(
        "train token labels -> "
        + " ".join(
            f"{ID2LABEL[i]}={token_counts[i]} ({100 * token_counts[i] / total_tokens:.2f}%)" for i in range(NUM_LABELS)
        )
        + f" | class_weights={class_weights.tolist()}",
        flush=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    model = AutoModelForTokenClassification.from_pretrained(
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

    trainer = WeightedTokenTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        # Argmax on GPU so eval accumulates (n, seq) int ids, not (n, 4096, 3) logits.
        preprocess_logits_for_metrics=lambda logits, labels: logits.argmax(dim=-1),
        class_weights=class_weights,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
    )

    trainer.train()

    # First and only touch of the test split.
    report = build_token_test_report(args, trainer, class_weights, counts, val_ds, test_ds, test_task_types)

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    METRICS_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n=== TEST results (best checkpoint by val span-F1) ===", flush=True)
    print(f"  span-level (seqeval)        : {report['test']['span_level']}", flush=True)
    print(f"  response-level (derived)    : {report['test']['response_level_derived']}", flush=True)
    print(f"  always-hallucinated         : {report['test']['always_hallucinated']}", flush=True)
    print(f"  random                      : {report['test']['random']}", flush=True)
    print("  per task_type (response-level derived):", flush=True)
    for task_type, task_metrics in report["test"]["per_task_type"].items():
        print(f"    {task_type:8s}: {task_metrics}", flush=True)
    print(f"\nSaved: {METRICS_PATH} | model + tokenizer -> {args.output_dir}", flush=True)

    maybe_push_to_hub(trainer, tokenizer, args)


if __name__ == "__main__":
    main()
