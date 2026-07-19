# Running the NLI baseline on Kaggle (GPU)

> **Historical Phase-2 artifact.** This walkthrough documents the exact steps used to
> run the zero-shot NLI baseline on Kaggle early in the project, kept for reference. It
> predates `main` and some details are stale: it clones a since-merged feature branch,
> and Section 4b's "no full-set scoring runner exists yet" note is no longer true — that
> runner was built (`scripts/evaluate_baseline.py` + `scripts/collect_predictions.py`)
> and is wired into the `evaluate` target in the [Makefile](../Makefile). For current
> reproduction instructions, run `make evaluate` (see the README) rather than following
> Section 4b below.

The zero-shot NLI baseline is impractical on CPU: a single Summary row takes ~15–110s
(it scores every response sentence against every context chunk, and Summary contexts have
20–48 chunks). On a Kaggle GPU the same work runs ~10–20× faster, making a full val/test
sweep feasible. This is the exact setup to run it.

## 1. Create the notebook and enable the GPU

1. On Kaggle: **Create → New Notebook**.
2. Right sidebar → **Settings → Accelerator → GPU T4 x2** (or **P100**). Either is fine;
   the model is DeBERTa-v3-base (~440MB), so one GPU is plenty — T4 x2 just gives you two.
3. Right sidebar → **Settings → Internet → On**. Required: cloning the repo, `pip install`,
   downloading the HF model, the nltk `punkt` data, and cloning the RAGTruth dataset all
   need network access.

## 2. Clone the repo and install dependencies

Cell 0 — upgrade pip first. Kaggle's base image pip can be old enough to fail on
Python 3.12 (old pip imports the removed `distutils` module internally), which
breaks installing `seqeval` and other packages in `requirements.txt` — this is
unrelated to any package itself:

```bash
!pip install --upgrade pip
```

Then, in the next cell:

```bash
!git clone --branch feature/phase-2-nli-baseline-impl \
    https://github.com/hugoomez/rag-hallucination-detector.git
%cd rag-hallucination-detector
!pip install -q -r requirements.txt
```

(Once this branch is merged, clone `main` instead.)

## 3. Regenerate the data

`data/raw/` and `data/processed/` are gitignored, so a fresh clone has **no data**. Rebuild
both from source (RAGTruth is cloned by `download.py`, then the parquets are built by
`preprocess.py`):

```bash
!python src/data/download.py        # clones RAGTruth into data/raw/ragtruth/
!python src/data/preprocess.py      # builds data/processed/response_level_{train,val,test}.parquet
```

`preprocess.py` also loads the DeBERTa-v3 tokenizer from the HF hub on first run (cached
afterwards). The val/test splits it writes are what the evaluation samples from.

## 4. Run the baseline

**GPU is automatic — no code change needed.** `NLIHallucinationDetector.from_pretrained()`
already selects the device itself:

```python
# src/models/nli_baseline.py
if device is None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
```

`__init__` then does `model.to(device).eval()`, and every batch in `_score_pairs` is moved
to `self.device`. So on a GPU-enabled Kaggle session `from_pretrained()` returns a
CUDA-backed detector with no arguments and no edits. Confirm it picked the GPU:

```python
from src.models.nli_baseline import NLIHallucinationDetector
detector = NLIHallucinationDetector.from_pretrained()
print(detector.device)   # -> "cuda"
```

### 4a. Sanity check + timing probe

```bash
!python scripts/time_baseline_sample.py
```

This prints `device: cuda`, per-row progress, per-task mean/median/max seconds, and an
extrapolated full val/test wall-clock (each task_type's own mean weighted by its real
proportion). Use it to confirm the GPU speedup and to project the full-run cost before
committing to it. The demo notebook `notebooks/02_baseline_nli.ipynb` (one worked example
per task_type via `chunk_context` → `detect`) is also a good smoke test — open it and
Run All.

### 4b. Full val/test evaluation

The building blocks are in place — `chunk_context(task_type, source_info)` →
`detector.score_response(context_chunks, response)` for each row (the expensive step, run
once), then `apply_thresholds(all_scores, ent_thr, con_thr)` to sweep thresholds
model-free and compare `response_hallucinated` against the `label_response` ground truth
(`int(len(labels) > 0)`) to compute precision/recall/F1.

> NOTE: a dedicated full-set scoring + metrics runner (extending
> `scripts/time_baseline_sample.py` from the 30-row probe to the full val/test sets, with
> threshold selection on val and final metrics on test) is the next Step 2.4 deliverable
> and does not exist yet. On Kaggle you can either add it as a new script/cell or scale the
> timing script up. Reconstruct full-text rows the same way the demo notebook does: take
> the val/test `source_id`s from the processed parquets and pull `source_info`/`response`
> from the raw merged dataset (`load_merged_dataframe`).

## Notes

- Keep Internet **On** for the whole session — the nltk `punkt` download happens lazily on
  the first `chunk_context`/`split_sentences` call, not at import.
- No `requirements.txt` change is needed for GPU: the pinned `torch>=2.2` resolves to a
  CUDA build in Kaggle's GPU image.
