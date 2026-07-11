"""Fine-tune ModernBERT-base for binary token-level hallucination span detection (Track B).

Consumes the truncation-free binary-labeled parquets from src/data/preprocess_token_level.py
(data/processed/token_level_binary_{train,val,test}.parquet: per-token supported(0)/
hallucinated(1) labels on response tokens, -100 on context/special tokens, plus per-token
character offsets and normalized gold char spans; ADR-011: 0% of rows exceed ModernBERT's
4096-token budget). Per ADR-013 (LettuceDetect parity, arXiv:2502.17125) this replaces the
retired 3-class BIO scheme: plain cross-entropy by DEFAULT (no class weighting; a capped
inverse-frequency fallback is available via --class_weight_cap), character-level spans are
reconstructed at inference by merging consecutive positive tokens, and checkpoint selection
uses the derived response-level F1 instead of the noisy exact-match span metric.

Metrics reported per split (val and, once, the untouched test split):
- span_char_level: LettuceDetect's character-overlap micro P/R/F1 (headline span metric,
  comparable to their published 55.44 base / 58.93 large span-level F1);
- response_level_derived: a response is "predicted hallucinated" iff any response token is
  predicted positive (comparable to their 76.07/79.22 example-level F1 and to
  baseline_nli_metrics.json / finetuned_track_a_metrics.json /
  finetuned_approach1_modernbert_metrics.json);
- span_exact_match: strict secondary measure -- predicted spans matching token-aligned gold
  spans exactly (replaces the retired seqeval exact-entity scoring).

Shared infrastructure (seed, results dir, response-level P/R/F1 helpers, trivial baselines,
per-task breakdown, WeightedTrainer base, guarded Hub push) is imported from train.py; the
binary label scheme is imported from preprocess_token_level.py rather than redefined.
T4 constraints mirror train_modernbert.py: attn_implementation="sdpa" (no FlashAttention 2
on Turing), fp16 (no bf16), small per-device batches with gradient accumulation (effective
batch 16), gradient checkpointing on by default.

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

from src.data.preprocess_token_level import HALLUCINATED_LABEL, IGNORE_LABEL, LABEL_NAMES  # noqa: E402
from src.models.train import (  # noqa: E402
    DEFAULT_SEED,
    RESULTS_DIR,
    WeightedTrainer,
    acc_prf,
    compute_class_weights,
    maybe_push_to_hub,
    per_task_breakdown,
    prf,
    trivial_baselines,
)

MODEL_NAME = "answerdotai/ModernBERT-base"
ATTN_IMPLEMENTATION = "sdpa"
METRICS_PATH = RESULTS_DIR / "finetuned_track_b_token_level_metrics.json"

# The binary scheme comes from preprocess_token_level.py; only IGNORE_LABEL (-100) is
# excluded -- it marks context/special tokens, not a class the model predicts.
ID2LABEL = {label_id: name for label_id, name in LABEL_NAMES.items() if label_id != IGNORE_LABEL}
LABEL2ID = {name: label_id for label_id, name in ID2LABEL.items()}
NUM_LABELS = len(ID2LABEL)


def parse_args() -> argparse.Namespace:
    """Expose all hyperparameters, paths, and flags so runs are reproducible and rerunnable."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train_path", default="data/processed/token_level_binary_train.parquet")
    parser.add_argument("--val_path", default="data/processed/token_level_binary_val.parquet")
    parser.add_argument("--test_path", default="data/processed/token_level_binary_test.parquet")
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
        "--num_train_epochs",
        type=float,
        default=8.0,
        help="Upper bound; early stopping may cut short. ADR-013: the 5-epoch BIO run was still improving when capped.",
    )
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument(
        "--early_stopping_patience", type=int, default=2, help="Epochs without val response-level F1 gain."
    )
    parser.add_argument(
        "--class_weight_cap",
        type=float,
        default=None,
        help="Default None = plain cross-entropy (LettuceDetect parity, per ADR-013). If set, "
        "inverse-frequency class weights are computed and clamped to this maximum.",
    )
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
    """Read a token-level parquet into a Dataset plus the raw df (offsets, gold spans, task_type).

    Unlike train.load_split, "labels" here is a per-token sequence (not a scalar), already
    aligned to input_ids by preprocess_token_level.py. Model inputs stay as plain int lists
    so DataCollatorForTokenClassification can pad input_ids/attention_mask AND labels
    (with -100) dynamically per batch. token_starts/token_ends/gold_starts/gold_ends stay
    in the df only -- they feed span reconstruction and the char-overlap metric, never the
    model.
    """
    df = pd.read_parquet(path)
    for column in ("input_ids", "attention_mask", "labels", "token_starts", "token_ends", "gold_starts", "gold_ends"):
        df[column] = df[column].apply(lambda a: np.asarray(a).tolist())
    ds = datasets.Dataset.from_pandas(df[["input_ids", "attention_mask", "labels"]], preserve_index=False)
    return ds, df


def flatten_token_labels(label_sequences: list[list[int]]) -> np.ndarray:
    """All real token labels across a split, with IGNORE_LABEL (context/special) dropped."""
    flat = np.concatenate([np.asarray(seq) for seq in label_sequences])
    return flat[flat != IGNORE_LABEL]


def merge_predicted_spans(
    pred_row, label_row, token_starts: list[int], token_ends: list[int]
) -> list[tuple[int, int]]:
    """Merge consecutive positive-token predictions into character-level spans.

    Mirrors LettuceDetect's span builder: walking response positions (label != -100) in
    token order, a positive prediction opens a span at that token's char start (or extends
    the current span's end to that token's char end); a non-positive prediction closes it.
    Whitespace between consecutive positive tokens is absorbed because the span runs from
    the first token's start to the last token's end. pred_row/label_row may be padded
    beyond len(token_starts) by the Trainer; zip() stops at the real tokens.
    """
    spans: list[tuple[int, int]] = []
    current: list[int] | None = None
    for pred, label, char_start, char_end in zip(pred_row, label_row, token_starts, token_ends):
        if label == IGNORE_LABEL:
            continue
        if pred == HALLUCINATED_LABEL:
            if current is None:
                current = [char_start, char_end]
            else:
                current[1] = char_end
        elif current is not None:
            spans.append((current[0], current[1]))
            current = None
    if current is not None:
        spans.append((current[0], current[1]))
    return spans


def char_span_prf(
    pred_spans_per_example: list[list[tuple[int, int]]], gold_spans_per_example: list[list[tuple[int, int]]]
) -> dict:
    """Character-overlap span micro P/R/F1, verbatim LettuceDetect's evaluator semantics.

    Overlapping characters are summed over every (predicted span x gold span) pair and
    micro-aggregated over the whole split: precision = overlap / total predicted chars,
    recall = overlap / total gold chars. No double counting is possible because predicted
    spans are disjoint by construction (merge_predicted_spans) and gold spans are
    normalized-disjoint in preprocessing (normalize_spans).
    """
    total_overlap = total_pred = total_gold = 0
    for pred_spans, gold_spans in zip(pred_spans_per_example, gold_spans_per_example):
        total_pred += sum(end - start for start, end in pred_spans)
        total_gold += sum(end - start for start, end in gold_spans)
        for pred_start, pred_end in pred_spans:
            for gold_start, gold_end in gold_spans:
                total_overlap += max(0, min(pred_end, gold_end) - max(pred_start, gold_start))
    precision = total_overlap / total_pred if total_pred > 0 else 0.0
    recall = total_overlap / total_gold if total_gold > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}


def exact_span_prf(
    pred_spans_per_example: list[list[tuple[int, int]]], gold_spans_per_example: list[list[tuple[int, int]]]
) -> dict:
    """Strict secondary metric: a predicted span counts only on exact (start, end) match.

    Gold spans here must be TOKEN-aligned (built by running merge_predicted_spans on the
    gold labels), since predicted spans always land on token boundaries -- comparing them
    against raw character-level gold spans would make exact match nearly impossible by
    construction. Replaces the retired seqeval exact-entity scoring from the BIO era.
    """
    true_positives = n_pred = n_gold = 0
    for pred_spans, gold_spans in zip(pred_spans_per_example, gold_spans_per_example):
        gold_set = set(gold_spans)
        n_pred += len(pred_spans)
        n_gold += len(gold_spans)
        true_positives += sum(1 for span in pred_spans if span in gold_set)
    precision = true_positives / n_pred if n_pred > 0 else 0.0
    recall = true_positives / n_gold if n_gold > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}


def derive_response_labels(labels: np.ndarray, predictions: np.ndarray) -> tuple[list[int], list[int]]:
    """Collapse token-level output to response-level: hallucinated iff ANY response token is positive.

    Only positions with a real label (!= -100) count -- predictions on context/special/
    padding tokens are meaningless. y_true uses the same any-positive rule on the gold
    labels, matching how preprocess_token_level.py derived its stratification proxy.
    """
    y_true, y_pred = [], []
    for label_row, pred_row in zip(labels, predictions):
        mask = label_row != IGNORE_LABEL
        y_true.append(int(np.any(label_row[mask] == HALLUCINATED_LABEL)))
        y_pred.append(int(np.any(pred_row[mask] == HALLUCINATED_LABEL)))
    return y_true, y_pred


def split_spans(
    predictions: np.ndarray, labels: np.ndarray, df: pd.DataFrame
) -> tuple[list[list[tuple[int, int]]], list[list[tuple[int, int]]], list[list[tuple[int, int]]]]:
    """Per example: (predicted char spans, normalized gold char spans, token-aligned gold spans).

    df rows must be in the same order as predictions/labels (Trainer preserves dataset
    order). Token-aligned gold spans reuse merge_predicted_spans with the gold labels
    standing in as predictions.
    """
    pred_spans, gold_char_spans, gold_token_spans = [], [], []
    for pred_row, label_row, starts, ends, gold_starts, gold_ends in zip(
        predictions, labels, df["token_starts"], df["token_ends"], df["gold_starts"], df["gold_ends"]
    ):
        pred_spans.append(merge_predicted_spans(pred_row, label_row, starts, ends))
        gold_char_spans.append(list(zip(gold_starts, gold_ends)))
        gold_token_spans.append(merge_predicted_spans(label_row, label_row, starts, ends))
    return pred_spans, gold_char_spans, gold_token_spans


def make_compute_metrics(eval_df: pd.DataFrame):
    """Build Trainer's per-epoch compute_metrics closed over the eval split's offsets/gold spans.

    The HF Dataset only carries model inputs; char-level span metrics additionally need
    each row's token offsets and gold spans, which live in the DataFrame (same row order,
    which Trainer preserves). eval_df MUST be sliced identically to the eval Dataset when
    --max_eval_samples is set.
    """

    def compute_metrics(eval_pred) -> dict:
        predictions, labels = eval_pred
        predictions = np.asarray(predictions)
        labels = np.asarray(labels)

        resp_true, resp_pred = derive_response_labels(labels, predictions)
        response = acc_prf(resp_true, resp_pred)

        pred_spans, gold_char_spans, _ = split_spans(predictions, labels, eval_df)
        char_span = char_span_prf(pred_spans, gold_char_spans)

        mask = labels != IGNORE_LABEL
        token = prf(labels[mask].tolist(), predictions[mask].tolist())

        return {
            "response_accuracy": response["accuracy"],
            "response_precision": response["precision"],
            "response_recall": response["recall"],
            "response_f1": response["f1"],
            "char_span_precision": char_span["precision"],
            "char_span_recall": char_span["recall"],
            "char_span_f1": char_span["f1"],
            "token_precision": token["precision"],
            "token_recall": token["recall"],
            "token_f1": token["f1"],
        }

    return compute_metrics


def _predict_token_labels(trainer: "WeightedTokenTrainer", ds: datasets.Dataset) -> tuple[np.ndarray, np.ndarray]:
    """Run inference over a split; returns (label_ids, predicted_ids), both (n, seq)."""
    output = trainer.predict(ds)
    return np.asarray(output.label_ids), np.asarray(output.predictions)


def evaluate_split(labels: np.ndarray, predictions: np.ndarray, df: pd.DataFrame) -> tuple[dict, list[int], list[int]]:
    """All three metric blocks for a fully predicted split, plus the derived response labels."""
    pred_spans, gold_char_spans, gold_token_spans = split_spans(predictions, labels, df)
    resp_true, resp_pred = derive_response_labels(labels, predictions)
    metrics = {
        "span_char_level": char_span_prf(pred_spans, gold_char_spans),
        "response_level_derived": acc_prf(resp_true, resp_pred),
        "span_exact_match": exact_span_prf(pred_spans, gold_token_spans),
    }
    return metrics, resp_true, resp_pred


def build_token_test_report(
    args: argparse.Namespace,
    trainer: "WeightedTokenTrainer",
    class_weights: torch.Tensor | None,
    counts: dict,
    val_ds: datasets.Dataset,
    val_df: pd.DataFrame,
    test_ds: datasets.Dataset,
    test_df: pd.DataFrame,
) -> dict:
    """Evaluate the best model on val and (for the first time) test; assemble the metrics dict.

    Same top-level shape as the response-level reports (model_name, hyperparameters, counts,
    val, test) but each split carries the char-overlap span metrics (headline), a derived
    response-level block comparable to the other three systems, and the strict exact-match
    span block.
    """
    val_labels, val_preds = _predict_token_labels(trainer, val_ds)
    test_labels, test_preds = _predict_token_labels(trainer, test_ds)

    val_metrics, _, _ = evaluate_split(val_labels, val_preds, val_df)
    test_metrics, test_resp_true, test_resp_pred = evaluate_split(test_labels, test_preds, test_df)
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
            "class_weight_cap": args.class_weight_cap,
            "class_weights": None if class_weights is None else [round(float(w), 4) for w in class_weights.tolist()],
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "gradient_checkpointing": args.gradient_checkpointing,
            "attn_implementation": ATTN_IMPLEMENTATION,
        },
        "counts": counts,
        "val": val_metrics,
        "test": {
            **test_metrics,
            "always_hallucinated": baselines["always_hallucinated"],
            "random": baselines["random"],
            "per_task_type": per_task_breakdown(test_df["task_type"].tolist(), test_resp_true, test_resp_pred),
        },
    }


class WeightedTokenTrainer(WeightedTrainer):
    """WeightedTrainer variant for token classification: flatten (batch, seq) before the loss.

    With class_weights=None (the ADR-013 default) this is plain cross-entropy, exactly
    matching LettuceDetect's recipe; --class_weight_cap re-enables (clamped) weighting.
    """

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
    val_ds, val_df = load_token_split(args.val_path)
    test_ds, test_df = load_token_split(args.test_path)

    # Optional smoke-test caps (train/val only -- test is never artificially shrunk).
    # The df must stay aligned with the Dataset: compute_metrics pairs them by row order.
    if args.max_train_samples is not None:
        n = min(args.max_train_samples, len(train_ds))
        train_ds = train_ds.select(range(n))
        train_df = train_df.iloc[:n].reset_index(drop=True)
        print(f"WARNING: using only {n} training rows (smoke test mode)", flush=True)
    if args.max_eval_samples is not None:
        n = min(args.max_eval_samples, len(val_ds))
        val_ds = val_ds.select(range(n))
        val_df = val_df.iloc[:n].reset_index(drop=True)
        print(f"WARNING: using only {n} eval rows (smoke test mode)", flush=True)

    # Per ADR-013 the default is NO class weighting (plain CE, LettuceDetect parity);
    # --class_weight_cap opts back into inverse-frequency weights clamped to the cap.
    flat_train_labels = flatten_token_labels(train_df["labels"].tolist())
    token_counts = np.bincount(flat_train_labels, minlength=NUM_LABELS)
    total_tokens = int(token_counts.sum())
    if args.class_weight_cap is not None:
        class_weights = compute_class_weights(flat_train_labels.tolist(), num_labels=NUM_LABELS)
        class_weights = torch.clamp(class_weights, max=args.class_weight_cap)
    else:
        class_weights = None
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
        + f" | class_weights={'None (plain CE)' if class_weights is None else class_weights.tolist()}",
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
        # ADR-013: select on the derived response-level F1 -- stable from epoch 1, unlike
        # the near-zero exact-match span F1 that drove the BIO run's selection.
        metric_for_best_model="response_f1",
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
        compute_metrics=make_compute_metrics(val_df),
        # Argmax on GPU so eval accumulates (n, seq) int ids, not (n, 4096, 2) logits.
        # For 2 classes argmax is exactly LettuceDetect's p >= 0.5 threshold.
        preprocess_logits_for_metrics=lambda logits, labels: logits.argmax(dim=-1),
        class_weights=class_weights,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
    )

    trainer.train()

    # First and only touch of the test split.
    report = build_token_test_report(args, trainer, class_weights, counts, val_ds, val_df, test_ds, test_df)

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    METRICS_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n=== TEST results (best checkpoint by val response-level F1) ===", flush=True)
    print(f"  span (char-overlap)         : {report['test']['span_char_level']}", flush=True)
    print(f"  response-level (derived)    : {report['test']['response_level_derived']}", flush=True)
    print(f"  span (exact match, strict)  : {report['test']['span_exact_match']}", flush=True)
    print(f"  always-hallucinated         : {report['test']['always_hallucinated']}", flush=True)
    print(f"  random                      : {report['test']['random']}", flush=True)
    print("  per task_type (response-level derived):", flush=True)
    for task_type, task_metrics in report["test"]["per_task_type"].items():
        print(f"    {task_type:8s}: {task_metrics}", flush=True)
    print(f"\nSaved: {METRICS_PATH} | model + tokenizer -> {args.output_dir}", flush=True)

    maybe_push_to_hub(trainer, tokenizer, args)


if __name__ == "__main__":
    main()
