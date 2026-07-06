\# Learning Notes



Running summary of key concepts learned from the papers/resources in `docs/theory.md`.

Meant as a quick-reference, not a replacement for the original sources.



\---



\## NLI task vs. SNLI dataset



NLI (Natural Language Inference) is the \*task\*: given a premise and a hypothesis, classify

the relation as entailment, contradiction, or neutral. SNLI is one specific \*dataset\*

built for that task (like ImageNet is a dataset for the image classification task).

MultiNLI is another. In this project: premise = retrieved context, hypothesis = answer

claim; a hallucination tends to be contradiction (contradicts context) or neutral

(unsupported by context).



\## The \[CLS] token



A special token prepended to the input (`\[CLS] premise \[SEP] hypothesis \[SEP]`). After

passing through the transformer, its output vector is used as a fixed-size summary of the

whole input pair, fed into the final classification head. It isn't inherently meaningful —

it becomes meaningful because training forces it to encode whatever the classifier needs.



\## Disentangled attention (DeBERTa)



Standard transformers (BERT/RoBERTa) merge content and position into a single vector per

token. DeBERTa keeps them separate (H = content, P = relative position) and computes

attention as a sum of three components: content-to-content, content-to-position, and

position-to-content. This lets the model reason about content and relative position

independently — relevant because word relationships depend on both.



\## DeBERTa v1 vs v3



Same architecture (disentangled attention + Enhanced Mask Decoder). What changes is the

pretraining objective:

\- v1: MLM (mask 15% of tokens, predict them) — only 15% of tokens give training signal.

\- v3: RTD, ELECTRA-style (a small generator swaps some words for plausible fakes; the

&#x20; main model — the discriminator, which is the DeBERTa we use — predicts original vs.

&#x20; replaced for every token) — 100% of tokens give training signal, more efficient

&#x20; pretraining for the same compute.



\## Fine-tuning hyperparameter reference (from the DeBERTa paper, Table 9)



Starting point for `TrainingArguments` in Phase 3, for the `base` model size:

\- Learning rate: 1.5e-5 to 4e-5

\- Batch size: 16, 32, 48, or 64

\- Max epochs: up to 10

