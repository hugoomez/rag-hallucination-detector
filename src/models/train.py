"""Fine-tune DeBERTa-v3-base for response-level hallucination classification (Phase 3, Track A).

Consumes the already-tokenized parquets from Phase 1 (data/processed/response_level_{train,val,test}
.parquet) directly -- each row carries `input_ids` / `attention_mask` built with ADR-004's
context-only truncation (the full response is always preserved), so there is NO re-tokenization
here. Trains a binary classifier (0 = faithful, 1 = hallucinated) with class-weighted cross-entropy
to counter the ~55/45 label imbalance, selects the best checkpoint by validation F1, then evaluates
ONCE on the untouched test split and writes results/finetuned_track_a_metrics.json in a schema
comparable to results/baseline_nli_metrics.json (Phase 2), with a per-task_type breakdown.

Meant to run on a GPU (Kaggle, see scripts/KAGGLE_SETUP.md); on CPU it is impractically slow.
Data must be generated first (src/data/download.py then src/data/preprocess.py). Hub pushing is
off by default and only happens via --push_to_hub (run on Kaggle with an HF token).

Example:
    python src/models/train.py --learning_rate 2e-5 --num_train_epochs 5
    python src/models/train.py --push_to_hub --hub_model_id <user>/deberta-v3-ragtruth-track-a
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
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from sklearn.metrics import accuracy_score, precision_recall_fscore_support  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

MODEL_NAME = "microsoft/deberta-v3-base"
NUM_LABELS = 2
ID2LABEL = {0: "faithful", 1: "hallucinated"}
LABEL2ID = {"faithful": 0, "hallucinated": 1}
TASK_TYPES = ["Summary", "QA", "Data2txt"]
DEFAULT_SEED = 42
RESULTS_DIR = Path("results")
METRICS_PATH = RESULTS_DIR / "finetuned_track_a_metrics.json"


def parse_args() -> argparse.Namespace:
    """Expose all hyperparameters, paths, and flags so runs are reproducible and rerunnable."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train_path", default="data/processed/response_level_train.parquet")
    parser.add_argument("--val_path", default="data/processed/response_level_val.parquet")
    parser.add_argument("--test_path", default="data/processed/response_level_test.parquet")
    parser.add_argument("--model_name", default=MODEL_NAME, help="Base encoder checkpoint.")
    parser.add_argument("--output_dir", default="models/finetuned_track_a", help="Where the best model + tokenizer go.")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="DeBERTa paper range: 1.5e-5 to 4e-5.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=32)
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
        help="Mixed precision (default on for GPU); use --no-fp16 for a CPU smoke test.",
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Off by default; Kaggle-only, needs an HF token.")
    parser.add_argument("--hub_model_id", default=None, help="Target repo id; required when --push_to_hub is set.")
    return parser.parse_args()


def load_split(path: str) -> tuple[datasets.Dataset, pd.DataFrame]:
    """Read a Phase 1 parquet into a Dataset (labels renamed) plus the raw df (for task_type order).

    Leaves input_ids / attention_mask as plain int lists so DataCollatorWithPadding can pad each
    batch dynamically; no fixed-length torch format is imposed here.
    """
    df = pd.read_parquet(path)
    df["input_ids"] = df["input_ids"].apply(lambda a: np.asarray(a).tolist())
    df["attention_mask"] = df["attention_mask"].apply(lambda a: np.asarray(a).tolist())
    ds = datasets.Dataset.from_pandas(df[["input_ids", "attention_mask", "label_response"]], preserve_index=False)
    ds = ds.rename_column("label_response", "labels")
    return ds, df


def compute_class_weights(labels: list[int]) -> torch.Tensor:
    """Inverse-frequency weights from the ACTUAL train labels: w_c = N / (num_classes * count_c)."""
    counts = np.bincount(labels, minlength=NUM_LABELS)
    total = counts.sum()
    weights = total / (NUM_LABELS * counts)
    return torch.tensor(weights, dtype=torch.float32)


def compute_metrics(eval_pred) -> dict:
    """Accuracy + binary P/R/F1 (positive class = hallucinated) for Trainer's per-epoch eval."""
    logits, labels = eval_pred
    preds = np.asarray(logits).argmax(axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", pos_label=1, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def prf(y_true: list[int], y_pred: list[int]) -> dict:
    """Binary precision/recall/F1 with hallucinated (label 1) as the positive class."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0
    )
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}


def acc_prf(y_true: list[int], y_pred: list[int]) -> dict:
    """Model-block metrics: accuracy plus the shared P/R/F1 shape."""
    return {"accuracy": float(accuracy_score(y_true, y_pred)), **prf(y_true, y_pred)}


def per_task_breakdown(task_types: list[str], y_true: list[int], y_pred: list[int]) -> dict:
    """Test P/R/F1 split by task_type (difficulty differs a lot across tasks)."""
    breakdown: dict = {}
    for task_type in TASK_TYPES:
        idx = [i for i, t in enumerate(task_types) if t == task_type]
        if not idx:
            continue
        breakdown[task_type] = {
            "n": len(idx),
            **prf([y_true[i] for i in idx], [y_pred[i] for i in idx]),
        }
    return breakdown


def trivial_baselines(y_true: list[int], seed: int) -> dict:
    """Always-hallucinated (all 1s) and random (seeded) baselines, matching Phase 2 for comparison."""
    n = len(y_true)
    always = [1] * n
    rng = np.random.default_rng(seed)
    random_pred = rng.integers(0, 2, size=n).tolist()
    return {
        "always_hallucinated": prf(y_true, always),
        "random": prf(y_true, random_pred),
    }


def _predict_labels(trainer: "WeightedTrainer", ds: datasets.Dataset) -> tuple[list[int], list[int]]:
    """Run inference over a split and return (y_true, y_pred) in dataset order."""
    output = trainer.predict(ds)
    y_pred = np.asarray(output.predictions).argmax(axis=-1).tolist()
    y_true = np.asarray(output.label_ids).tolist()
    return y_true, y_pred


def build_test_report(
    args: argparse.Namespace,
    trainer: "WeightedTrainer",
    class_weights: torch.Tensor,
    counts: dict,
    val_ds: datasets.Dataset,
    test_ds: datasets.Dataset,
    test_task_types: list[str],
) -> dict:
    """Evaluate the best model on val and (for the first time) test; assemble the metrics dict."""
    val_true, val_pred = _predict_labels(trainer, val_ds)
    test_true, test_pred = _predict_labels(trainer, test_ds)
    baselines = trivial_baselines(test_true, args.seed)
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
        },
        "counts": counts,
        "val": {"finetuned": acc_prf(val_true, val_pred)},
        "test": {
            "finetuned": acc_prf(test_true, test_pred),
            "always_hallucinated": baselines["always_hallucinated"],
            "random": baselines["random"],
            "per_task_type": per_task_breakdown(test_task_types, test_true, test_pred),
        },
    }


def maybe_push_to_hub(trainer: "WeightedTrainer", tokenizer, args: argparse.Namespace) -> None:
    """The ONLY Hub push path: explicit, guarded, and called last (after train + test are done).

    Trainer's automatic push is left fully disabled, so no intermediate checkpoints leak to the Hub.
    """
    if not args.push_to_hub:
        print("push_to_hub disabled (default) -- model kept local only.", flush=True)
        return
    if not args.hub_model_id:
        raise ValueError("--push_to_hub requires --hub_model_id")
    print(f"Pushing model + tokenizer to Hub: {args.hub_model_id}", flush=True)
    trainer.model.push_to_hub(args.hub_model_id)
    tokenizer.push_to_hub(args.hub_model_id)
    print("Hub push complete.", flush=True)


class WeightedTrainer(Trainer):
    """Trainer variant that applies class-weighted cross-entropy loss."""

    def __init__(self, *args, class_weights: torch.Tensor | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        weight = None if self.class_weights is None else self.class_weights.to(logits.device)
        loss = nn.CrossEntropyLoss(weight=weight)(logits, labels)
        return (loss, outputs) if return_outputs else loss


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    RESULTS_DIR.mkdir(exist_ok=True)

    train_ds, train_df = load_split(args.train_path)
    val_ds, _ = load_split(args.val_path)
    test_ds, test_df = load_split(args.test_path)
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
        args.model_name, num_labels=NUM_LABELS, id2label=ID2LABEL, label2id=LABEL2ID
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        fp16=args.fp16,
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
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
    )

    trainer.train()

    # First and only touch of the test split.
    report = build_test_report(args, trainer, class_weights, counts, val_ds, test_ds, test_task_types)

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
