"""Time the NLI baseline (task-type-aware chunking + scoring) on a small sample.

Samples 10 rows per task_type (30 total, fixed seed) from the raw merged RAGTruth dataset,
times chunk_context + score_response per row while streaming progress, then reports per-task
timing and extrapolates full val/test wall-clock using each task_type's own mean weighted by
its actual proportion in the val/test splits.

Run from the repo root and redirect output so progress can be tailed:
    ./.venv/Scripts/python.exe scripts/time_baseline_sample.py > results/timing_log.txt 2>&1
"""

import os
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

import pandas as pd  # noqa: E402

from src.data.context_chunking import chunk_context  # noqa: E402
from src.data.preprocess import load_merged_dataframe  # noqa: E402
from src.models.nli_baseline import NLIHallucinationDetector  # noqa: E402

RANDOM_STATE = 42
N_PER_TASK = 10
TASK_TYPES = ["Summary", "QA", "Data2txt"]
VAL_PARQUET = "data/processed/response_level_val.parquet"
TEST_PARQUET = "data/processed/response_level_test.parquet"


def build_sample(merged: pd.DataFrame) -> pd.DataFrame:
    """10 rows per task_type, fixed seed, concatenated in a stable task order."""
    frames = [
        merged[merged["task_type"] == task_type].sample(N_PER_TASK, random_state=RANDOM_STATE)
        for task_type in TASK_TYPES
    ]
    return pd.concat(frames).reset_index(drop=True)


def main() -> None:
    merged = load_merged_dataframe()
    sample = build_sample(merged)

    detector = NLIHallucinationDetector.from_pretrained()
    print(f"device: {detector.device} | sample: {len(sample)} rows " f"({N_PER_TASK} per task_type)", flush=True)

    times_by_task: dict[str, list[float]] = {task_type: [] for task_type in TASK_TYPES}

    for i, row in enumerate(sample.itertuples(), start=1):
        start = time.perf_counter()
        context_chunks = chunk_context(row.task_type, row.source_info)
        detector.score_response(context_chunks, row.response)
        elapsed = time.perf_counter() - start

        times_by_task[row.task_type].append(elapsed)
        print(
            f"[{i:2d}/{len(sample)}] {row.task_type:8s} " f"chunks={len(context_chunks):3d} elapsed={elapsed:6.2f}s",
            flush=True,
        )

    print("\n=== per-task timing (seconds per row) ===", flush=True)
    for task_type in TASK_TYPES:
        times = times_by_task[task_type]
        print(
            f"  {task_type:8s} n={len(times):2d}  "
            f"mean={statistics.mean(times):6.2f}  "
            f"median={statistics.median(times):6.2f}  "
            f"max={max(times):6.2f}",
            flush=True,
        )

    # Extrapolate each split with each task_type's OWN mean, weighted by that task_type's
    # actual row proportion in the split (task_type counts come from the processed parquets).
    print("\n=== extrapolated full-set wall-clock (per-task mean x actual proportion) ===", flush=True)
    for name, parquet in [("val", VAL_PARQUET), ("test", TEST_PARQUET)]:
        counts = pd.read_parquet(parquet, columns=["task_type"])["task_type"].value_counts()
        total_seconds = sum(statistics.mean(times_by_task[task_type]) * n for task_type, n in counts.items())
        breakdown = ", ".join(f"{task_type}={counts.get(task_type, 0)}" for task_type in TASK_TYPES)
        print(
            f"  {name:4s} ({counts.sum()} rows: {breakdown}): "
            f"{total_seconds / 60:.1f} min ({total_seconds / 3600:.2f} h)",
            flush=True,
        )


if __name__ == "__main__":
    main()
