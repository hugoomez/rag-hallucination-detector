"""Backfill per-row test predictions (labels + probability scores) into the unified table.

Phase 4 compares every trained system on identical footing: one long-format table
(results/unified_predictions.parquet by default) with one row per (system, test row) and
columns system/row_index/source_id/task_type/split/y_true/y_pred/y_score, where y_score
is always P(hallucinated)-like (higher = more likely hallucinated). The systems live on
different Kaggle sessions and Hub repos, so this script runs ONE system at a time and
merges into the accumulating file — re-running a system replaces only that system's rows.

Modes:
    baseline    Model-free. Aggregates results/nli_scores_test.json to response level:
                y_score = max over sentences of max(contradiction, 1 - entailment) (the
                single-threshold reduction of ADR-007's disjunctive decision rule);
                y_pred = apply_thresholds at the tuned thresholds from
                results/baseline_nli_metrics.json. Note y_pred is the operational
                decision at (ent_thr, con_thr) and is not exactly a threshold on
                y_score — y_score exists for threshold-free PR curves.
    track_a     Batched Hub inference over data/processed/response_level_test.parquet
                (same pattern as scripts/analyze_track_a_predictions.py) capturing
                softmax P(hallucinated) per row.
    approach_1  Same, over data/processed/response_level_modernbert_test.parquet
                (regenerate via `python -m src.data.preprocess_modernbert` if absent).
    track_b_modernbert
                Token-classification inference (AutoModelForTokenClassification) over
                data/processed/token_level_binary_test.parquet. y_true/y_pred collapse
                token predictions to response level via train_token_level.derive_response_labels
                (any positive token -> hallucinated); y_score is the max per-token
                P(hallucinated) over the response's real tokens (see collect_track_b_modernbert
                for the derivation rationale).
    merge       Fold a unified-schema parquet produced elsewhere (e.g. downloaded from a
                Kaggle session) into the local accumulating file.

Every mode takes --split {val,test} (default test): it selects the split's parquet/scores
file and writes that value into the 'split' column. Re-running a (system, split) pair
replaces only those rows, so val and test accumulate side by side in the one file.

Examples:
    python scripts/collect_predictions.py baseline
    python scripts/collect_predictions.py baseline --split val
    python scripts/collect_predictions.py track_a --split val
    python scripts/collect_predictions.py approach_1
    python scripts/collect_predictions.py track_b_modernbert --split val
    python scripts/collect_predictions.py merge --input kaggle_output/unified_predictions.parquet
"""

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    DataCollatorWithPadding,
)

from src.data.preprocess_token_level import HALLUCINATED_LABEL, IGNORE_LABEL  # noqa: E402
from src.evaluation.metrics import UNIFIED_COLUMNS, UNIFIED_PREDICTIONS_PATH, response_level_metrics  # noqa: E402
from src.models.nli_baseline import apply_thresholds  # noqa: E402
from src.models.train_token_level import derive_response_labels  # noqa: E402

SYSTEM_BASELINE = "baseline_nli"
SYSTEM_TRACK_A = "track_a_deberta"
SYSTEM_APPROACH_1 = "approach_1_modernbert"
SYSTEM_TRACK_B = "track_b_modernbert"

NLI_SCORES_PATHS = {
    "val": Path("results/nli_scores_val.json"),
    "test": Path("results/nli_scores_test.json"),
}
BASELINE_METRICS_PATH = Path("results/baseline_nli_metrics.json")
HUB_DEFAULTS = {
    SYSTEM_TRACK_A: "hugoomezz/deberta-v3-ragtruth-hallucination",
    SYSTEM_APPROACH_1: "hugoomezz/modernbert-ragtruth-response-level",
    SYSTEM_TRACK_B: "hugoomezz/modernbert-ragtruth-token-level-binary",
}
# Per-system parquet for each split. row_index (input order) is a valid cross-system
# join key on TEST (all systems iterate the identical 2700-row set in the same order),
# but on VAL only the three modernbert-era systems (baseline/approach_1/track_b) share
# one ordering; track_a's val parquet was preprocessed with a different within-source
# response order (1 missing response + 1 swapped pair), so it must not be row_index-joined
# against the others on val. See docs/decisions.md and tune_threshold_and_ensemble.py.
SPLIT_PATH_DEFAULTS = {
    SYSTEM_TRACK_A: {
        "val": "data/processed/response_level_val.parquet",
        "test": "data/processed/response_level_test.parquet",
    },
    SYSTEM_APPROACH_1: {
        "val": "data/processed/response_level_modernbert_val.parquet",
        "test": "data/processed/response_level_modernbert_test.parquet",
    },
    SYSTEM_TRACK_B: {
        "val": "data/processed/token_level_binary_val.parquet",
        "test": "data/processed/token_level_binary_test.parquet",
    },
}
BATCH_SIZE = 32


def baseline_y_score(sentence_scores: list) -> float:
    """Response-level hallucination score from per-sentence (entailment, contradiction) pairs.

    Per sentence: max(contradiction, 1 - entailment) — the one-dimensional reduction of
    ADR-007's disjunctive flag rule (not-supported iff contradiction >= con_thr OR
    entailment < ent_thr), so a single threshold sweep on this score walks the coupled
    rule family con_thr = t, ent_thr = 1 - t. Contradiction-priority is honored: a high
    contradiction dominates the max even when some context sentence strongly entails the
    claim. Response score is the max over sentences; empty responses score 0.0
    (vacuously-not-hallucinated, matching apply_thresholds).
    """
    if not sentence_scores:
        return 0.0
    return max(max(float(con), 1.0 - float(ent)) for ent, con in sentence_scores)


def build_prediction_rows(system, source_ids, task_types, y_true, y_pred, y_score, split="test") -> pd.DataFrame:
    """Assemble one system's rows in the unified schema, with positional row_index.

    row_index (0..n-1, input order) is the per-row key: source_id is NOT unique in
    RAGTruth (6 model responses per source). On the test split all systems iterate the
    same deterministic 2700-row set, so row_index also serves as the cross-system join
    key; on val that only holds within the modernbert-era systems (see SPLIT_PATH_DEFAULTS).
    """
    lengths = {len(source_ids), len(task_types), len(y_true), len(y_pred), len(y_score)}
    if len(lengths) != 1:
        raise ValueError(f"Input columns have mismatched lengths: {lengths}")
    n_rows = lengths.pop()
    return pd.DataFrame(
        {
            "system": [system] * n_rows,
            "row_index": range(n_rows),
            "source_id": source_ids,
            "task_type": task_types,
            "split": [split] * n_rows,
            "y_true": pd.array(y_true, dtype="int64"),
            "y_pred": pd.array(y_pred, dtype="int64"),
            "y_score": pd.array(y_score, dtype="float64"),
        }
    )[UNIFIED_COLUMNS]


def merge_predictions(existing: pd.DataFrame | None, new: pd.DataFrame) -> pd.DataFrame:
    """Replace rows for the (system, split) pairs present in `new`; keep everything else.

    This is what makes stage-wise collection safe: re-running one system on one split (or
    folding in a file from another Kaggle session) never drops previously collected
    (system, split) combinations. Keying on the pair — not the system alone — is what lets
    `--split val` accumulate alongside a system's existing test rows instead of wiping them.
    """
    if existing is None:
        return new.reset_index(drop=True)
    replaced = set(zip(new["system"], new["split"]))
    keep_mask = [(system, split) not in replaced for system, split in zip(existing["system"], existing["split"])]
    kept = existing[pd.Series(keep_mask, index=existing.index)]
    return pd.concat([kept, new], ignore_index=True)


def collect_baseline(
    split: str = "test", scores_path: Path | None = None, metrics_path: Path = BASELINE_METRICS_PATH
) -> pd.DataFrame:
    """Aggregate the per-sentence NLI scores to response level in the unified schema.

    Model-free: reuses results/nli_scores_{split}.json (sentence_scores are
    [max_entailment, max_contradiction] pairs, ADR-007) and the tuned thresholds
    already selected on the validation split (results/baseline_nli_metrics.json). The
    tuned thresholds are split-independent, so the same operating point is applied to
    whichever split's scores are aggregated.
    """
    scores_path = scores_path or NLI_SCORES_PATHS[split]
    rows = json.loads(Path(scores_path).read_text(encoding="utf-8"))
    thresholds = json.loads(Path(metrics_path).read_text(encoding="utf-8"))["best_thresholds"]

    all_scores = [[(pair[0], pair[1]) for pair in row["sentence_scores"]] for row in rows]
    y_pred = [int(flag) for flag in apply_thresholds(all_scores, thresholds["ent_thr"], thresholds["con_thr"])]
    y_score = [baseline_y_score(scores) for scores in all_scores]

    print(
        f"Baseline: {len(rows)} rows from {scores_path}, "
        f"thresholds ent={thresholds['ent_thr']} con={thresholds['con_thr']}"
    )
    return build_prediction_rows(
        system=SYSTEM_BASELINE,
        source_ids=[row["source_id"] for row in rows],
        task_types=[row["task_type"] for row in rows],
        y_true=[int(row["label_response"]) for row in rows],
        y_pred=y_pred,
        y_score=y_score,
        split=split,
    )


def save_merged(new_df: pd.DataFrame, unified_path: Path) -> pd.DataFrame:
    """Merge one stage's rows into the accumulating file and report what it now holds."""
    unified_path = Path(unified_path)
    existing = pd.read_parquet(unified_path) if unified_path.exists() else None
    merged = merge_predictions(existing, new_df)
    unified_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(unified_path, index=False)
    print(f"Saved {unified_path} — rows per (system, split):")
    for (system, split), count in merged.groupby(["system", "split"]).size().sort_index().items():
        print(f"  {system} [{split}]: {count}")
    return merged


def run_inference_with_probs(df: pd.DataFrame, model, collator, device: torch.device) -> tuple[list[int], list[float]]:
    """Batched forward pass in row order, returning argmax labels and softmax P(hallucinated).

    Same batching pattern as scripts/analyze_track_a_predictions.py, extended to keep the
    positive-class probability (index 1 = hallucinated, matching training's
    label_response encoding) for threshold-free PR curves.
    """
    examples = [{"input_ids": row.input_ids, "attention_mask": row.attention_mask} for row in df.itertuples()]
    loader = DataLoader(examples, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collator)

    model.eval()
    preds: list[int] = []
    scores: list[float] = []
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            logits = model(**batch).logits
            probs = torch.softmax(logits, dim=-1)
            preds.extend(probs.argmax(dim=-1).cpu().tolist())
            scores.extend(probs[:, 1].cpu().tolist())
    return preds, scores


def load_test_df(path: str) -> pd.DataFrame:
    """Read a response-level test parquet; input_ids/attention_mask are already tokenized."""
    df = pd.read_parquet(path)
    df["input_ids"] = df["input_ids"].apply(lambda a: np.asarray(a).tolist())
    df["attention_mask"] = df["attention_mask"].apply(lambda a: np.asarray(a).tolist())
    return df


def collect_transformer(
    system: str,
    hub_model_id: str,
    test_path: str,
    split: str = "test",
    limit: int | None = None,
    tokenizer_id: str | None = None,
) -> pd.DataFrame:
    """Hub-checkpoint inference over a test parquet, in the unified schema.

    tokenizer_id overrides where the tokenizer is loaded from: the collator only needs
    padding metadata (rows are pre-tokenized), so the base model's tokenizer is an exact
    substitute when a fine-tuned repo's tokenizer_config was written by a newer
    transformers than the local install can parse.
    """
    df = load_test_df(test_path)
    if limit is not None:
        df = df.head(limit)
    print(f"{system}: {len(df)} rows from {test_path}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id or hub_model_id)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(hub_model_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Loaded {hub_model_id} on {device}", flush=True)

    y_pred, y_score = run_inference_with_probs(df, model, collator, device)
    return build_prediction_rows(
        system=system,
        source_ids=df["source_id"].tolist(),
        task_types=df["task_type"].tolist(),
        y_true=df["label_response"].astype(int).tolist(),
        y_pred=y_pred,
        y_score=y_score,
        split=split,
    )


def load_token_test_df(path: str) -> pd.DataFrame:
    """Read a token-level binary test parquet; input_ids/attention_mask/labels are per-token."""
    df = pd.read_parquet(path)
    for column in ("input_ids", "attention_mask", "labels"):
        df[column] = df[column].apply(lambda a: np.asarray(a).tolist())
    return df


def run_token_inference_with_probs(
    df: pd.DataFrame, model, collator, device: torch.device
) -> tuple[list[int], list[int], list[float]]:
    """Batched token-classification forward pass, collapsed to one row per response.

    DataCollatorForTokenClassification pads labels with IGNORE_LABEL (-100, its default
    label_pad_token_id), so padding positions are excluded by the same mask that already
    excludes context/special tokens -- one mask does both jobs. y_true/y_pred reuse
    derive_response_labels verbatim (row-independent, so calling it per batch and
    concatenating is identical to calling it once over the full split). y_score is the
    max per-token P(hallucinated) over each response's real tokens.
    """
    examples = [
        {"input_ids": row.input_ids, "attention_mask": row.attention_mask, "labels": row.labels}
        for row in df.itertuples()
    ]
    loader = DataLoader(examples, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collator)

    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    y_score: list[float] = []
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels").numpy()
            batch = {key: value.to(device) for key, value in batch.items()}
            logits = model(**batch).logits  # (batch, seq, 2)
            probs = torch.softmax(logits, dim=-1)[..., HALLUCINATED_LABEL].cpu().numpy()
            preds = logits.argmax(dim=-1).cpu().numpy()

            batch_true, batch_pred = derive_response_labels(labels, preds)
            y_true.extend(batch_true)
            y_pred.extend(batch_pred)

            mask = labels != IGNORE_LABEL
            for row_probs, row_mask in zip(probs, mask):
                masked = row_probs[row_mask]
                # Vacuously not-hallucinated if a row somehow has zero real tokens,
                # matching ADR-015's empty-response convention for the baseline's y_score.
                y_score.append(float(masked.max()) if masked.size else 0.0)
    return y_true, y_pred, y_score


def collect_track_b_modernbert(
    hub_model_id: str, test_path: str, split: str = "test", limit: int | None = None, tokenizer_id: str | None = None
) -> pd.DataFrame:
    """Hub-checkpoint token-classification inference over the binary token-level test parquet.

    Unlike track_a/approach_1 (AutoModelForSequenceClassification, one label per response),
    Track B is AutoModelForTokenClassification: one supported(0)/hallucinated(1) label per
    real response token, -100 elsewhere (ADR-013). y_true/y_pred collapse to response level
    via train_token_level.derive_response_labels (any positive token -> hallucinated),
    reused verbatim rather than reimplemented so this always matches the training-time
    definition exactly.

    y_score = max per-token P(hallucinated) over the response's real tokens. This is the
    direct continuous relaxation of the same any-positive decision rule: thresholding the
    max at 0.5 recovers y_pred exactly (argmax-positive at some token iff that token's
    P(hallucinated) >= 0.5), the same way ADR-015 derived the baseline's y_score as a
    one-dimensional reduction of its own disjunctive flag rule. A mean-over-tokens score
    was considered and rejected: it would decouple the score from the rule it's meant to
    relax, since a response with one high-confidence hallucinated token among many
    confidently-supported tokens should score high, not get diluted by the majority.
    """
    df = load_token_test_df(test_path)
    if limit is not None:
        df = df.head(limit)
    print(f"{SYSTEM_TRACK_B}: {len(df)} rows from {test_path}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id or hub_model_id)
    collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    model = AutoModelForTokenClassification.from_pretrained(hub_model_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Loaded {hub_model_id} on {device}", flush=True)

    y_true, y_pred, y_score = run_token_inference_with_probs(df, model, collator, device)
    return build_prediction_rows(
        system=SYSTEM_TRACK_B,
        source_ids=df["source_id"].tolist(),
        task_types=df["task_type"].tolist(),
        y_true=y_true,
        y_pred=y_pred,
        y_score=y_score,
        split=split,
    )


MODE_TO_SYSTEM = {"track_a": SYSTEM_TRACK_A, "approach_1": SYSTEM_APPROACH_1, "track_b_modernbert": SYSTEM_TRACK_B}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["baseline", "track_a", "approach_1", "track_b_modernbert", "merge"])
    parser.add_argument(
        "--split",
        choices=["val", "test"],
        default="test",
        help="Which split to collect (default test). Writes the value into the 'split' column; "
        "picks the split's parquet/scores file for each mode. Ignored by merge mode.",
    )
    parser.add_argument("--unified_path", default=str(UNIFIED_PREDICTIONS_PATH))
    parser.add_argument(
        "--hub_model_id", help="Override the default Hub repo for track_a/approach_1/track_b_modernbert."
    )
    parser.add_argument(
        "--test_path",
        help="Override the default parquet for track_a/approach_1/track_b_modernbert "
        "(otherwise resolved from the mode and --split).",
    )
    parser.add_argument(
        "--tokenizer_id",
        help="Override tokenizer repo for track_a/approach_1 (e.g. the base model, when the "
        "fine-tuned repo's tokenizer_config was written by an incompatible transformers version).",
    )
    parser.add_argument("--input", help="merge mode: unified-schema parquet to fold in (e.g. from Kaggle).")
    parser.add_argument(
        "--limit",
        type=int,
        help="Smoke test: run inference on only the first N rows, print metrics, and skip writing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.chdir(REPO_ROOT)

    if args.mode == "baseline":
        new_df = collect_baseline(split=args.split)
    elif args.mode == "track_b_modernbert":
        new_df = collect_track_b_modernbert(
            hub_model_id=args.hub_model_id or HUB_DEFAULTS[SYSTEM_TRACK_B],
            test_path=args.test_path or SPLIT_PATH_DEFAULTS[SYSTEM_TRACK_B][args.split],
            split=args.split,
            limit=args.limit,
            tokenizer_id=args.tokenizer_id,
        )
    elif args.mode in MODE_TO_SYSTEM:
        system = MODE_TO_SYSTEM[args.mode]
        new_df = collect_transformer(
            system=system,
            hub_model_id=args.hub_model_id or HUB_DEFAULTS[system],
            test_path=args.test_path or SPLIT_PATH_DEFAULTS[system][args.split],
            split=args.split,
            limit=args.limit,
            tokenizer_id=args.tokenizer_id,
        )
    else:  # merge
        if not args.input:
            raise SystemExit("merge mode requires --input")
        new_df = pd.read_parquet(args.input)
        missing = [column for column in UNIFIED_COLUMNS if column not in new_df.columns]
        if missing:
            raise SystemExit(f"--input file is missing unified-schema columns: {missing}")

    if args.limit is not None:
        preview = response_level_metrics(new_df["y_true"].to_numpy(), new_df["y_pred"].to_numpy())
        print(f"[--limit smoke test] not writing. Metrics on {preview['n']} rows:")
        print({key: preview[key] for key in ("precision", "recall", "f1")})
        return

    save_merged(new_df, Path(args.unified_path))


if __name__ == "__main__":
    main()
