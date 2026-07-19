---
license: mit
datasets:
- ragtruth
language:
- en
base_model: microsoft/deberta-v3-base
pipeline_tag: text-classification
tags:
- hallucination-detection
- rag
- text-classification
- deberta-v3
---

# deberta-v3-ragtruth-hallucination (Track A)

Fine-tuned `microsoft/deberta-v3-base`, binary sequence classifier for **response-level**
RAG hallucination detection: given a `(context, response)` pair, predicts whether the
response contains *any* hallucinated content relative to the context. Label 0 =
faithful, label 1 = hallucinated.

This is **not** the best-performing system in this project — see
[`hugoomezz/modernbert-ragtruth-token-level-binary`](https://huggingface.co/hugoomezz/modernbert-ragtruth-token-level-binary)
(Track B), which supersedes it and is the model deployed in the live demo. Track A is
kept published for the project's 4-system comparison.

## Intended use

Research / portfolio demonstration of RAG hallucination detection on RAGTruth. Given a
retrieved context and a generated response, scores whether the response is faithful to
that context. Not validated on any domain outside RAGTruth's three task types (QA,
Summary, Data2txt), and not intended for production moderation decisions without
further evaluation on your own data.

## Training data

[RAGTruth](https://github.com/ParticleMedia/RAGTruth) (Niu et al., 2024, ACL,
[arXiv:2401.00396](https://arxiv.org/abs/2401.00396)) — MIT-licensed, reproduced in
this project's [docs/THIRD_PARTY_LICENSES.md](https://github.com/hugoomez/rag-hallucination-detector/blob/main/docs/THIRD_PARTY_LICENSES.md).
13,578 train / 1,511 val / 2,700 test response-level rows (~55/45 faithful/hallucinated
split in train; val is one row lower than the ModernBERT-based models below because
DeBERTa-v3's 512-token budget forced dropping a response-length outlier that they didn't
need to — see ADR-006/ADR-011), class-weighted cross-entropy to counter the imbalance. Context-only
truncation at 512 tokens (ADR-004): the response is always fully preserved, only the
context's head is truncated when the combined length exceeds the budget (this affects
70.34% of rows at DeBERTa-v3's 512-token limit).

## Metrics (RAGTruth test set, n=2700; from `results/finetuned_track_a_metrics.json`)

| Metric | Value |
|---|---|
| Precision | 0.7367 |
| Recall | 0.6882 |
| F1 | 0.7116 |
| Accuracy | 0.8052 |

Per task_type:

| Task | F1 | Recall |
|---|---|---|
| Data2txt | 0.8476 | 0.8549 |
| QA | 0.5859 | 0.6500 |
| Summary | 0.3322 | 0.2451 |

## Limitations

- **Summary is a significant weak spot** (recall 0.245 — misses 3 out of 4 hallucinated
  summaries), the inverse of the zero-shot baseline's near-perfect recall.
- **Context truncation** affects 70.34% of RAGTruth rows at 512 tokens; its cost is
  precision-driven (over-flags faithful responses), not recall-driven, per this
  project's ADR-010 diagnostic.
- **Response-level only** — flags a response as hallucinated or not, but cannot
  localize *which part* of the response is hallucinated. For span-level localization,
  use Track B instead.

## How to use

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_ID = "hugoomezz/deberta-v3-ragtruth-hallucination"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID).eval()

context = "The Eiffel Tower was completed in 1889 for the World's Fair in Paris."
response = "The Eiffel Tower was completed in 1889 and stands in Paris, France."

inputs = tokenizer(context, response, truncation="only_first", max_length=512, return_tensors="pt")
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
