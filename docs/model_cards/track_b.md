---
license: mit
datasets:
- ragtruth
language:
- en
base_model: answerdotai/ModernBERT-base
pipeline_tag: token-classification
tags:
- hallucination-detection
- rag
- token-classification
- modernbert
- lettucedetect
---

# modernbert-ragtruth-token-level-binary (Track B)

Fine-tuned `answerdotai/ModernBERT-base`, binary **token**-classification model for
**span-level** RAG hallucination detection: given a `(context, response)` pair, labels
every response token supported (0) or hallucinated (1), so it recovers character-level
spans, not just a single per-response score.

**This is the best-performing system in this project (F1 0.7631 at the response level)
and the model deployed in the live demo** (`src/models/predict.py`), superseding both
Track A (`hugoomezz/deberta-v3-ragtruth-hallucination`) and Approach 1
(`hugoomezz/modernbert-ragtruth-response-level`). Its recipe matches
[LettuceDetect](https://arxiv.org/abs/2502.17125)'s approach: binary token labels,
unweighted loss, and character-overlap span evaluation, with checkpoint selection on
span-level F1 rather than response-level F1 (see [ADR-020](https://github.com/hugoomez/rag-hallucination-detector/blob/main/docs/decisions.md)
below).

## Intended use

Research / portfolio demonstration of span-level RAG hallucination detection on
RAGTruth: highlighting exactly which characters of a generated response are
unsupported by the retrieved context, not just a binary verdict. Powers this project's
live demo (paste-your-own text, and a real RAG pipeline over a small Wikipedia corpus).
Not validated outside RAGTruth's three task types (QA, Summary, Data2txt), and not
intended for production moderation decisions without further evaluation on your own
data and threshold.

## Training data

[RAGTruth](https://github.com/ParticleMedia/RAGTruth) (Niu et al., 2024, ACL,
[arXiv:2401.00396](https://arxiv.org/abs/2401.00396)) — MIT-licensed, reproduced in
this project's [docs/THIRD_PARTY_LICENSES.md](https://github.com/hugoomez/rag-hallucination-detector/blob/main/docs/THIRD_PARTY_LICENSES.md).
13,578 train / 1,512 val / 2,700 test rows (one more val row than Track A — see ADR-006/
ADR-011), tokenized at `max_length=4096` (0% of rows truncated). Per-token binary labels: 0 = supported, 1 = hallucinated (any character
overlap with a gold annotated span); context and special tokens are ignored in the
loss (label -100). Plain cross-entropy, **no class weighting** — an earlier 3-class BIO
scheme with inverse-frequency weighting on an ultra-rare class caused near-total span
fragmentation (0.037 F1); this binary/unweighted redesign fixed it.

## Metrics (RAGTruth test set, n=2700)

Response-level (a response is "predicted hallucinated" iff any response token is
predicted positive) — reported figures from this project's unified cross-system
evaluation, matching the deployed model and the README's comparison table:

| Metric | Value |
|---|---|
| Precision | 0.8359 |
| Recall | 0.7020 |
| F1 | 0.7631 |
| Accuracy | 0.8478 |

This matches [LettuceDetect-base](https://arxiv.org/abs/2502.17125)'s published
example-level F1 of 76.07%, from `results/finetuned_track_b_token_level_metrics.json`.

Span-level (character-overlap, LettuceDetect's headline metric; from
`results/finetuned_track_b_token_level_metrics.json`):

| Metric | Value |
|---|---|
| Precision | 0.6474 |
| Recall | 0.4517 |
| F1 | **0.5321** |

vs. LettuceDetect-base's published span-level F1 of 55.44%.

Per task_type (response-level derived):

| Task | F1 | Recall |
|---|---|---|
| Data2txt | 0.8675 | 0.8256 |
| QA | 0.6708 | 0.6688 |
| Summary | 0.4904 | 0.3775 |

**Recipe correction ([ADR-020](https://github.com/hugoomez/rag-hallucination-detector/blob/main/docs/decisions.md#adr-020-acws-ablation-results--recipe-fix-arm-b-adopted-noise-down-weighting-arm-c-rejected)):**
these are the weights from a controlled ablation's arm (b) — a faithful
LettuceDetect-recipe replication (lr=1e-5, effective batch 8, 6 epochs, checkpoint
selection on **span-level F1** instead of response-level F1). The prior deployed
weights (arm a) selected the checkpoint on response-level F1, which is structurally
unable to distinguish tight spans from sloppy ones and was suppressing span-level
performance; switching the selection metric alone (no architecture or data change)
raised span-F1 from 0.5113 to 0.5321 (+2.1 points) and response precision from 0.7873
to 0.8359, at the cost of some recall (0.7381 → 0.7020). A companion arm testing
Annotation-Confidence-Weighted Supervision (down-weighting the training loss on
annotator-flagged "implicit_true" spans) did not clear its pre-registered bar and was
rejected as a tested negative result — see the ADR for both arms' full numbers.

## Limitations

- **Summary remains the weakest task** (F1 0.490, recall 0.377 — misses more than half
  of hallucinated summaries), though improved precision over Track A/Approach 1.
- **Overconfident, not paranoid — more so than the prior arm-a weights**: misses 29.8%
  of hallucinated responses but false-alarms on only 7.4% of faithful ones (281 FN vs
  130 FP in this project's error analysis, up from 247 FN vs 188 FP under arm-a) — the
  token-level decision rule (flag if any token crosses P≥0.5) structurally favors
  silence over alarm, and the span-F1 checkpoint selection recipe (ADR-020) traded
  recall for precision, widening this skew rather than closing it.
- **"Subtle" hallucinations are the hardest case, and arm-b misses more of them**:
  responses annotated only with "Subtle" span types are missed 48.1% of the time (vs
  40.3% under arm-a), and evident-only spans are missed 30.6% of the time (vs 27.0%) —
  the same precision/recall tradeoff concentrated on the deployment scenario (strong
  generators, subtle errors) where detection matters most. The GPT-3.5/GPT-4-specific
  F1 drop (≈0.48–0.52 under arm-a) has not yet been re-measured for arm-b.
- **Known false positive on close paraphrase**: during live demo testing (against the
  prior arm-a weights; not yet re-tested on arm-b), the model flagged a factually
  *correct* grounded answer that paraphrased the source ("second of six children, five
  siblings") as unsupported at score 0.99 — plausibly triggered by surface-form
  sensitivity rather than a genuine factual disagreement. Noted as a known limitation,
  not further investigated (`docs/notes.md`, Phase 5 section). This is the inverse of
  the "overconfident" pattern above: mostly the model under-flags, but this shows it
  can occasionally over-flag on close paraphrases.
- **TODO (visible, not silent):** this project's README contains qualitative examples
  (error-analysis case studies, live-demo transcripts) and an ADR-017 ensemble analysis
  (F1 0.7701) that still reference arm-a's predictions/weights, not arm-b's. Full
  re-analysis is deferred until after the upcoming ModernBERT-large experiment
  concludes, to avoid redoing this work twice if that experiment supersedes Track B
  again.
- **The decision threshold is a product tradeoff, not a fixed answer**: F1 is nearly
  flat across thresholds 0.2–0.7, so the same checkpoint can be run as a high-precision
  "block" mode (t=0.9: precision 0.879, recall 0.602) or a high-recall "warn" mode
  (t=0.1: recall 0.843, precision 0.642). Even at the most aggressive setting, ~16% of
  hallucinations slip through — a risk reducer, not a guarantee.

## How to use

This snippet reproduces the response-level score (max P(hallucinated) over response
tokens); see `src/models/predict.py` in the project repo for full character-span
reconstruction (`merge_predicted_spans`).

```python
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

# The tokenizer is loaded from the base ModernBERT repo, not the fine-tuned one: this
# repo's tokenizer_config.json was written by a newer transformers than some pinned
# installs can parse. Training data was built with this exact base tokenizer, so the
# substitution is not just safe but exact.
TOKENIZER_ID = "answerdotai/ModernBERT-base"
MODEL_ID = "hugoomezz/modernbert-ragtruth-token-level-binary"
SUPPORTED, HALLUCINATED = 0, 1

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
model = AutoModelForTokenClassification.from_pretrained(MODEL_ID, attn_implementation="sdpa").eval()

context = "The Eiffel Tower was completed in 1889 for the World's Fair in Paris."
response = "The Eiffel Tower was completed in 1889 and is located in Berlin, Germany."

encoding = tokenizer(
    context, response, max_length=4096, truncation="only_first",
    return_offsets_mapping=True, return_token_type_ids=False, return_tensors="pt",
)
sequence_ids = encoding.sequence_ids(0)
encoding.pop("offset_mapping")  # only needed for span reconstruction, not the score

with torch.no_grad():
    logits = model(**encoding).logits[0]
probs_hallucinated = torch.softmax(logits, dim=-1)[:, HALLUCINATED].tolist()

# Response-level score: max P(hallucinated) over response tokens (sequence_id == 1).
response_probs = [p for p, sid in zip(probs_hallucinated, sequence_ids) if sid == 1]
score = max(response_probs) if response_probs else 0.0
print(f"hallucination score: {score:.4f}  ({'FLAGGED' if score >= 0.5 else 'clean'})")
```

## Citation

```bibtex
@inproceedings{niu2024ragtruth,
  title     = {RAGTruth: A Hallucination Corpus for Developing and Evaluating RAG Systems},
  author    = {Niu, Cheng and Wu, Yuanhao and Zhu, Juno and Xu, Siliang and Shum, Kashun and Zhong, Randy and Song, Juntong and Zhang, Tong},
  booktitle = {Proceedings of ACL 2024},
  year      = {2024},
  eprint    = {2401.00396}
}
@article{kovacs2025lettucedetect,
  title   = {LettuceDetect: A Hallucination Detection Framework for RAG Applications},
  author  = {Kov{\'a}cs, {\'A}d{\'a}m and Bakos, Zsolt},
  journal = {arXiv preprint arXiv:2502.17125},
  year    = {2025}
}
```
