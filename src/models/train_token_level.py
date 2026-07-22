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

# --checkpoint_metric choice -> compute_metrics key (Trainer prefixes "eval_" itself).
CHECKPOINT_METRICS = {"response_f1": "response_f1", "span_f1": "char_span_f1", "token_f1": "token_f1"}

# The binary scheme comes from preprocess_token_level.py; only IGNORE_LABEL (-100) is
# excluded -- it marks context/special tokens, not a class the model predicts.
ID2LABEL = {label_id: name for label_id, name in LABEL_NAMES.items() if label_id != IGNORE_LABEL}
LABEL2ID = {name: label_id for label_id, name in ID2LABEL.items()}
NUM_LABELS = len(ID2LABEL)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
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
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Gradient-clipping max norm. 1.0 (default) == the HF Trainer default, so leaving it "
        "unset is a no-op; lower it (e.g. 0.5) as an fp16-stability fallback for ModernBERT-large.",
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=2,
        help="Epochs without improvement on --checkpoint_metric. Set >= num_train_epochs to disable "
        "(fixed-epoch recipes like the LettuceDetect-parity ablation arms).",
    )
    parser.add_argument(
        "--checkpoint_metric",
        choices=sorted(CHECKPOINT_METRICS),
        default="response_f1",
        help="Val metric for best-checkpoint selection (and early stopping). response_f1 is the "
        "historical Track B default (ADR-013); token_f1 matches LettuceDetect's documented recipe "
        "(ablation arms b/c); span_f1 = char-overlap span F1.",
    )
    parser.add_argument(
        "--implicit_true_weight",
        type=float,
        default=1.0,
        help="ACWS: per-token loss weight for positions inside annotator-flagged implicit_true "
        "spans (is_implicit_true column). 1.0 (default) = current behavior, identical loss code "
        "path; 0.0 = exclude flagged tokens from the loss entirely. Training-time only -- labels "
        "and metrics never change.",
    )
    parser.add_argument(
        "--metrics_out",
        default=None,
        help=f"Metrics JSON path (default: {METRICS_PATH}). Ablation arms must set this to avoid "
        "overwriting the published Track B report.",
    )
    parser.add_argument(
        "--predictions_out",
        default=None,
        help="Per-example test predictions JSON for stratified evaluation "
        "(default: results/token_preds_<output_dir basename>.json).",
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
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=None,
        help="If set, log every N optimizer steps (logging_strategy='steps'); default (None) keeps "
        "the historical per-epoch logging. Use a small value (e.g. 10) so a smoke test surfaces "
        "NaN/inf loss within a few steps instead of once per epoch.",
    )
    return parser.parse_args(argv)


def load_token_split(path: str, with_implicit_mask: bool = False) -> tuple[datasets.Dataset, pd.DataFrame]:
    """Read a token-level parquet into a Dataset plus the raw df (offsets, gold spans, task_type).

    Unlike train.load_split, "labels" here is a per-token sequence (not a scalar), already
    aligned to input_ids by preprocess_token_level.py. Model inputs stay as plain int lists
    so DataCollatorForTokenClassification can pad input_ids/attention_mask AND labels
    (with -100) dynamically per batch. token_starts/token_ends/gold_starts/gold_ends stay
    in the df only -- they feed span reconstruction and the char-overlap metric, never the
    model.

    with_implicit_mask=True (ACWS training split only) additionally exposes the parquet's
    is_implicit_true column to the Dataset as an int "implicit_true_mask" sequence, so the
    collator can pad it and compute_loss can down-weight flagged tokens.
    """
    df = pd.read_parquet(path)
    columns = ["input_ids", "attention_mask", "labels", "token_starts", "token_ends", "gold_starts", "gold_ends"]
    if "is_implicit_true" in df.columns:
        columns.append("is_implicit_true")
    for column in columns:
        df[column] = df[column].apply(lambda a: np.asarray(a).tolist())
    ds_columns = ["input_ids", "attention_mask", "labels"]
    if with_implicit_mask:
        if "is_implicit_true" not in df.columns:
            raise ValueError(
                f"{path} has no is_implicit_true column -- regenerate it with "
                "src/data/preprocess_token_level.py before using --implicit_true_weight != 1.0."
            )
        df["implicit_true_mask"] = df["is_implicit_true"].apply(lambda seq: [int(bool(v)) for v in seq])
        ds_columns.append("implicit_true_mask")
    ds = datasets.Dataset.from_pandas(df[ds_columns], preserve_index=False)
    return ds, df


def flatten_token_labels(label_sequences: list[list[int]]) -> np.ndarray:
    """All real token labels across a split, with IGNORE_LABEL (context/special) dropped."""
    flat = np.concatenate([np.asarray(seq) for seq in label_sequences])
    return flat[flat != IGNORE_LABEL]


def merge_predicted_spans(pred_row, label_row, token_starts: list[int], token_ends: list[int]) -> list[tuple[int, int]]:
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


def build_prediction_records(labels, predictions, df: pd.DataFrame) -> list[dict]:
    """Per-example test predictions in the ablation dump format (scripts/ablation_report.py).

    labels/predictions are per-row sequences (rows may have different lengths -- lists of
    1-D arrays are fine, everything downstream zips row-wise). Spans are char-level;
    gold_spans mirror the parquet's normalized gold so the report script can cross-check
    them against raw metadata.
    """
    pred_spans, gold_char_spans, _ = split_spans(predictions, labels, df)
    resp_true, resp_pred = derive_response_labels(labels, predictions)
    records = []
    for i in range(len(df)):
        record = {
            "row_index": i,
            "source_id": int(df["source_id"].iloc[i]),
            "task_type": str(df["task_type"].iloc[i]),
            "pred_spans": [[int(start), int(end)] for start, end in pred_spans[i]],
            "gold_spans": [[int(start), int(end)] for start, end in gold_char_spans[i]],
            "resp_true": int(resp_true[i]),
            "resp_pred": int(resp_pred[i]),
        }
        if "response_id" in df.columns:
            record["response_id"] = str(df["response_id"].iloc[i])
        records.append(record)
    return records


def build_token_test_report(
    args: argparse.Namespace,
    trainer: "WeightedTokenTrainer",
    class_weights: torch.Tensor | None,
    counts: dict,
    val_ds: datasets.Dataset,
    val_df: pd.DataFrame,
    test_ds: datasets.Dataset,
    test_df: pd.DataFrame,
    best_checkpoint: dict | None = None,
) -> tuple[dict, np.ndarray, np.ndarray]:
    """Evaluate the best model on val and (for the first time) test; assemble the metrics dict.

    Same top-level shape as the response-level reports (model_name, hyperparameters, counts,
    val, test) but each split carries the char-overlap span metrics (headline), a derived
    response-level block comparable to the other three systems, and the strict exact-match
    span block. Also returns the raw test (labels, predictions) so main() can dump
    per-example records without a second inference pass.
    """
    val_labels, val_preds = _predict_token_labels(trainer, val_ds)
    test_labels, test_preds = _predict_token_labels(trainer, test_ds)

    val_metrics, _, _ = evaluate_split(val_labels, val_preds, val_df)
    test_metrics, test_resp_true, test_resp_pred = evaluate_split(test_labels, test_preds, test_df)
    baselines = trivial_baselines(test_resp_true, args.seed)

    report = {
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
            "checkpoint_metric": args.checkpoint_metric,
            "implicit_true_weight": args.implicit_true_weight,
            "early_stopping_patience": args.early_stopping_patience,
        },
        "counts": counts,
        "best_checkpoint": best_checkpoint,
        "val": val_metrics,
        "test": {
            **test_metrics,
            "always_hallucinated": baselines["always_hallucinated"],
            "random": baselines["random"],
            "per_task_type": per_task_breakdown(test_df["task_type"].tolist(), test_resp_true, test_resp_pred),
        },
    }
    return report, test_labels, test_preds


class ImplicitMaskCollator(DataCollatorForTokenClassification):
    """DataCollatorForTokenClassification that also pads implicit_true_mask (with 0).

    The mask must be popped before super().torch_call -- tokenizer.pad would reject the
    unknown key. Padding positions get 0, which is a no-op in the weighted loss because
    their labels are already -100 (weight 0 regardless). When no feature carries the key
    (eval splits, or implicit_true_weight == 1.0), behavior is byte-identical to the base
    collator.
    """

    def torch_call(self, features):
        has_mask = "implicit_true_mask" in features[0]
        masks = [feature.pop("implicit_true_mask") for feature in features] if has_mask else None
        batch = super().torch_call(features)
        if masks is not None:
            assert self.tokenizer.padding_side == "right", "implicit_true_mask padding assumes right padding"
            seq_len = batch["input_ids"].shape[1]
            padded = [list(mask) + [0] * (seq_len - len(mask)) for mask in masks]
            batch["implicit_true_mask"] = torch.tensor(padded, dtype=torch.long)
        return batch


def weighted_token_ce(
    logits: torch.Tensor,
    labels: torch.Tensor,
    implicit_mask: torch.Tensor,
    implicit_true_weight: float,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """ACWS objective: L = sum_t(w_t * l_t) / clamp(sum_t(w_t), 1e-8).

    l_t is the usual per-token CE (0 at ignore positions via ignore_index + reduction
    "none"); w_t = 0 where the label is -100, implicit_true_weight where the token is
    annotator-flagged (implicit_mask), 1 elsewhere. Properties (unit-tested):
    - implicit_true_weight == 1.0 reduces to the plain mean CE over valid tokens
      (main() never routes here in that case -- the legacy path stays bit-identical);
    - implicit_true_weight == 0.0 is exactly loss-masking flagged tokens (they leave
      numerator AND denominator);
    - flagged tokens are ~0.7-0.8% of supervised tokens, so the denominator shift is
      <1%: effectively per-token gradient scaling.
    Note: with class_weights set, the denominator uses only w_t (not PyTorch's
    class-weighted-mean convention); irrelevant for the ablation arms (plain CE).
    """
    per_token = nn.CrossEntropyLoss(weight=class_weights, ignore_index=IGNORE_LABEL, reduction="none")(
        logits.view(-1, NUM_LABELS), labels.view(-1)
    )
    flat_labels = labels.view(-1)
    weights = torch.ones_like(per_token)
    weights = torch.where(implicit_mask.view(-1).bool(), weights * implicit_true_weight, weights)
    weights = torch.where(flat_labels == IGNORE_LABEL, torch.zeros_like(weights), weights)
    return (per_token * weights).sum() / weights.sum().clamp(min=1e-8)


class WeightedTokenTrainer(WeightedTrainer):
    """WeightedTrainer variant for token classification: flatten (batch, seq) before the loss.

    With class_weights=None (the ADR-013 default) this is plain cross-entropy, exactly
    matching LettuceDetect's recipe; --class_weight_cap re-enables (clamped) weighting.
    When the batch carries implicit_true_mask (ACWS: --implicit_true_weight != 1.0, train
    split only), flagged tokens' loss is scaled by implicit_true_weight instead.
    """

    def __init__(self, *args, implicit_true_weight: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.implicit_true_weight = implicit_true_weight

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        implicit_mask = inputs.pop("implicit_true_mask", None)
        outputs = model(**inputs)
        logits = outputs.logits
        weight = None if self.class_weights is None else self.class_weights.to(logits.device)
        if implicit_mask is None:
            # ignore_index is CrossEntropyLoss's default, stated explicitly: -100 marks
            # context/special/padding tokens that must never contribute to the loss.
            loss = nn.CrossEntropyLoss(weight=weight, ignore_index=IGNORE_LABEL)(
                logits.view(-1, NUM_LABELS), labels.view(-1)
            )
        else:
            loss = weighted_token_ce(logits, labels, implicit_mask, self.implicit_true_weight, weight)
        return (loss, outputs) if return_outputs else loss


def find_best_checkpoint(trainer: "WeightedTokenTrainer", metric_name: str) -> dict | None:
    """Recover the epoch/step that produced the best checkpoint (Trainer records only the value).

    trainer.state.best_metric holds the winning eval_<metric_name> value but not which epoch
    it came from. Scan log_history for the eval entry whose eval_<metric_name> matches (exact,
    since Trainer copies the logged float verbatim; falls back to the closest match if float
    round-tripping ever perturbs it). Returns None when no checkpoint was selected (e.g. a
    zero-epoch smoke run).
    """
    best = trainer.state.best_metric
    if best is None:
        return None
    key = f"eval_{metric_name}"
    evals = [entry for entry in trainer.state.log_history if key in entry]
    if not evals:
        return None
    exact = [entry for entry in evals if entry[key] == best]
    match = exact[-1] if exact else min(evals, key=lambda e: abs(e[key] - best))
    return {
        "metric": metric_name,
        "value": float(best),
        "epoch": match.get("epoch"),
        "step": match.get("step"),
    }


def build_training_args(args: argparse.Namespace) -> TrainingArguments:
    """Assemble TrainingArguments from parsed CLI args.

    Extracted from main() so the argument wiring is unit-testable without loading a model
    or data. Leaving --logging_steps and --max_grad_norm at their defaults yields exactly
    the historical configuration (per-epoch logging, clip norm 1.0 == the HF Trainer
    default), so published base-model runs remain bit-identical.
    """
    # Logging cadence: default keeps per-epoch logging; --logging_steps switches to
    # step-based logging so a smoke test surfaces NaN/inf within a few optimizer steps.
    if args.logging_steps is not None:
        logging_kwargs = {"logging_strategy": "steps", "logging_steps": args.logging_steps}
    else:
        logging_kwargs = {"logging_strategy": "epoch"}
    return TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        # 1.0 == the HF Trainer default (a no-op); lower it as an fp16-stability fallback.
        max_grad_norm=args.max_grad_norm,
        fp16=args.fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        # ADR-013 default: select on the derived response-level F1 -- stable from epoch 1,
        # unlike the near-zero exact-match span F1 that drove the BIO run's selection. The
        # ablation arms (b/c) switch to token_f1 via --checkpoint_metric to match
        # LettuceDetect's documented recipe. Trainer prefixes eval_ itself.
        metric_for_best_model=f"eval_{CHECKPOINT_METRICS[args.checkpoint_metric]}",
        greater_is_better=True,
        save_total_limit=1,
        seed=args.seed,
        report_to="none",
        # push_to_hub deliberately NOT set (defaults False): no auto-push of intermediate
        # checkpoints. All Hub pushing happens once via maybe_push_to_hub() at the end.
        **logging_kwargs,
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    RESULTS_DIR.mkdir(exist_ok=True)

    # ACWS down-weighting needs the flagged-token mask on the TRAIN split only; when the
    # weight is 1.0 (arms a/b) we never load it, keeping the loss path bit-identical to the
    # published Track B run.
    use_implicit_weighting = args.implicit_true_weight != 1.0
    train_ds, train_df = load_token_split(args.train_path, with_implicit_mask=use_implicit_weighting)
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
    # ImplicitMaskCollator is a drop-in for DataCollatorForTokenClassification: it only does
    # extra work when a feature carries implicit_true_mask (train split under ACWS), and is
    # byte-identical to the base collator otherwise.
    collator = ImplicitMaskCollator(tokenizer=tokenizer)
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        attn_implementation=ATTN_IMPLEMENTATION,
    )

    training_args = build_training_args(args)

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
        implicit_true_weight=args.implicit_true_weight,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
    )

    trainer.train()

    best_checkpoint = find_best_checkpoint(trainer, CHECKPOINT_METRICS[args.checkpoint_metric])
    if best_checkpoint is not None:
        print(
            f"best checkpoint: epoch {best_checkpoint['epoch']} "
            f"({best_checkpoint['metric']}={best_checkpoint['value']:.4f})",
            flush=True,
        )

    # First and only touch of the test split.
    report, test_labels, test_preds = build_token_test_report(
        args, trainer, class_weights, counts, val_ds, val_df, test_ds, test_df, best_checkpoint=best_checkpoint
    )

    metrics_path = Path(args.metrics_out) if args.metrics_out else METRICS_PATH
    run_name = os.path.basename(os.path.normpath(args.output_dir))
    predictions_path = (
        Path(args.predictions_out) if args.predictions_out else RESULTS_DIR / f"token_preds_{run_name}.json"
    )

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Persist per-example test predictions (spans, response labels) for stratified analysis
    # in scripts/ablation_report.py -- reuses the labels/preds already produced above, no
    # second inference pass.
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    predictions = build_prediction_records(test_labels, test_preds, test_df)
    predictions_path.write_text(json.dumps(predictions, indent=2), encoding="utf-8")

    print(f"\n=== TEST results (best checkpoint by val {args.checkpoint_metric}) ===", flush=True)
    print(f"  span (char-overlap)         : {report['test']['span_char_level']}", flush=True)
    print(f"  response-level (derived)    : {report['test']['response_level_derived']}", flush=True)
    print(f"  span (exact match, strict)  : {report['test']['span_exact_match']}", flush=True)
    print(f"  always-hallucinated         : {report['test']['always_hallucinated']}", flush=True)
    print(f"  random                      : {report['test']['random']}", flush=True)
    print("  per task_type (response-level derived):", flush=True)
    for task_type, task_metrics in report["test"]["per_task_type"].items():
        print(f"    {task_type:8s}: {task_metrics}", flush=True)
    print(
        f"\nSaved: {metrics_path} | predictions -> {predictions_path} | model + tokenizer -> {args.output_dir}",
        flush=True,
    )

    maybe_push_to_hub(trainer, tokenizer, args)


if __name__ == "__main__":
    main()
