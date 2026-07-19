---
license: mit
datasets:
- ragtruth
language:
- en
base_model: answerdotai/ModernBERT-base
pipeline_tag: text-classification
tags:
- hallucination-detection
- rag
- text-classification
- modernbert
---

# modernbert-ragtruth-response-level (Approach 1)

Fine-tuned `answerdotai/ModernBERT-base`, binary sequence classifier for
**response-level** RAG hallucination detection: given a `(context, response)` pair,
predicts whether the response contains *any* hallucinated content relative to the
context. Label 0 = faithful, label 1 = hallucinated. Same task as Track A
(`hugoomezz/deberta-v3-ragtruth-hallucination`), on a truncation-free backbone.

This is **not** the best-performing system in this project — see
[`hugoomezz/modernbert-ragtruth-token-level-binary`](https://huggingface.co/hugoomezz/modernbert-ragtruth-token-level-binary)
(Track B), which supersedes it and is the model deployed in the live demo.

## Intended use

Research / portfolio demonstration of RAG hallucination detection on RAGTruth, and a
long-context (ModernBERT, 4096 tokens) comparison point against Track A's
DeBERTa-v3-base (512 tokens). Not validated outside RAGTruth's three task types (QA,
Summary, Data2txt), and not intended for production moderation decisions without
further evaluation.

## Training data

[RAGTruth](https://github.com/ParticleMedia/RAGTruth) (Niu et al., 2024, ACL,
[arXiv:2401.00396](https://arxiv.org/abs/2401.00396)) — MIT-licensed, reproduced in
this project's [docs/THIRD_PARTY_LICENSES.md](https://github.com/hugoomez/rag-hallucination-detector/blob/main/docs/THIRD_PARTY_LICENSES.md).
13,578 train / 1,512 val / 2,700 test response-level rows (one more val row than Track A —
its 4096-token budget didn't need to drop the ADR-006 response-length outlier), class-weighted
cross-entropy.
Tokenized at `max_length=4096`, which eliminates truncation entirely on RAGTruth (0.00%
of rows truncated, vs. 70.34% at DeBERTa-v3's 512-token limit).

## Metrics (RAGTruth test set, n=2700; from `results/finetuned_approach1_modernbert_metrics.json`)

| Metric | Value |
|---|---|
| Precision | 0.6839 |
| Recall | 0.7731 |
| F1 | 0.7257 |
| Accuracy | 0.7959 |

Per task_type:

| Task | F1 | Recall |
|---|---|---|
| Data2txt | 0.8506 | 0.8705 |
| QA | 0.5924 | 0.6813 |
| Summary | 0.5088 | 0.5686 |

Note: this project's unified cross-system comparison table (`README.md`) reports F1
0.7254 for this model, a ~0.0004 discrepancy from the 0.7257 above traced to exactly
one borderline test example flipping prediction between the original Kaggle training
run and a later re-inference pass — consistent with ordinary GPU inference
nondeterminism (batch composition / kernel selection), not a data or pipeline bug (see
`docs/notes.md`).

## Limitations

- **Summary recall (0.5686) improved substantially over Track A (0.2451)** by moving to
  a longer context window, but Summary F1 (0.5088) is still the weakest of the three
  task types.
- **Precision is slightly lower than Track A's** (0.6839 vs 0.7367) — the longer context
  window trades some precision for the large recall gain, concentrated almost entirely
  in Summary.
- **Response-level only** — cannot localize *which part* of a response is hallucinated.
  For span-level localization, use Track B instead.

## How to use

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_ID = "hugoomezz/modernbert-ragtruth-response-level"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID).eval()

context = "The Eiffel Tower was completed in 1889 for the World's Fair in Paris."
response = "The Eiffel Tower was completed in 1889 and stands in Paris, France."

inputs = tokenizer(context, response, truncation="only_first", max_length=4096, return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits
probs = torch.softmax(logits, dim=-1)[0]
print({"faithful": probs[0].item(), "hallucinated": probs[1].item()})
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
```
