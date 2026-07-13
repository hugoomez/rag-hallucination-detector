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

---

## Phase 1 — RAGTruth EDA findings

Key numbers from exploring RAGTruth before preprocessing (see
`notebooks/01_eda_ragtruth.ipynb` for full detail and saved charts in `results/`):

- Class balance: 56.9% faithful responses, 43.1% with at least one
  hallucination span — reasonably balanced, but still justifies using F1
  over accuracy (per `theory.md` block F).
- Hallucination rate varies sharply by generating model: GPT-3.5/GPT-4 ~13-14%,
  Llama-2 variants 47-62%, Mistral-7B-instruct highest at 65.9%.
- Span-level label type distribution is dominated by "evident" errors:
  Evident Baseless Info (6237) and Evident Conflict (5324) far outnumber
  Subtle Baseless Info (2527) and especially Subtle Conflict (201) — matching
  RAGTruth's published statistics.
- Context length varies drastically by task_type (DeBERTa-v3 tokens):
  QA mean 307 (max 617, safest), Data2txt mean 761 (most consistent, nearly
  all exceed 512), Summary mean 690 but with the widest spread (std 398.7,
  max 2189 — the longest tail). Responses are almost always short (mean 160).
- 70.34% of rows exceed 512 tokens when combining context+response+special
  tokens; hallucinated rows are over-represented among truncated ones
  (50.52% vs 43.08% globally) — truncation disproportionately affects the
  cases hardest to verify. This finding drove ADR-004.

## Phase 1 — Code review lessons (src/data/preprocess.py)

A code review caught two silent-failure risks worth remembering as patterns:
- Clamping a truncation budget to a minimum (e.g., "context tokens = 0") can
  mask a downstream constraint violation (total sequence length) if nothing
  asserts the final invariant. Always assert the property you actually care
  about (final length <= max_length), not just the intermediate one you
  computed towards it.
- A merge/join row-count check (e.g., "no fan-out duplication") does NOT
  catch unmatched keys producing NaN rows that survive silently until a much
  later, harder-to-trace error. Check both directions explicitly.

## Phase 2 — NLI baseline: independent max-entailment and max-contradiction tracking

When aggregating NLI scores across multiple context sentences for a single response
sentence, it's tempting to just keep the (entailment, neutral, contradiction) triple
from whichever context sentence had the highest entailment. This is wrong: since NLI
outputs are a 3-way softmax (always summing to 1) computed independently per context
sentence, the sentence with the highest entailment will almost always have a low
contradiction score by construction — but a DIFFERENT context sentence entirely may be
the one that actually contradicts the claim. Softmax constraints apply within one
comparison, not across different comparisons.

Fix: track max entailment and max contradiction independently, potentially from two
different context sentences. The resulting (entailment, contradiction) pair is no
longer a valid probability distribution (won't sum to 1 with the implied neutral) —
it's two independent signals, not one sentence's full output — but this is necessary
and correct for the aggregation to actually catch contradictions.

## Track B — token-level BIO preprocessing verified

`preprocess_token_level.py`'s alignment diagnostic was visually confirmed correct
across 6 sample rows on the real Kaggle run, including edge cases: a
subword-split span ("Wi-Fi," -> 4 separate tokens, all correctly tagged
B-HALL/I-HALL), a 24-token-long span (a full sentence), and a zero-hallucination
row (all tokens O, confirming the negative case). Consistent with ADR-011,
truncation was 0.00% across all three splits and task_types for the token-level
pipeline as well. Rough character-based class-balance estimate (Option A
diagnostic, pre-tokenization): ~94.7% O tokens vs. ~5.3% B-HALL+I-HALL combined
-- expect real token-level class weighting to matter significantly more here
than it did for response-level classification.

## Phase 4 — minor inference nondeterminism note

When recomputing Track A / Approach 1 metrics from freshly-collected predictions
(for the unified comparison table) versus the original Kaggle training run's
test-set evaluation, Approach 1's F1 differs by 0.0004 (0.72537 vs. published
0.72573) — traced to exactly one borderline example flipping from true negative
to false positive (recall, TP, and FN counts are identical; only this single
example's discrete prediction differs). Consistent with expected GPU inference
nondeterminism (batch composition / kernel selection), not a data or pipeline
bug. Considered acceptable noise; not investigated further.

## Phase 5 — observed false positive on paraphrase (Riemann sibling count)

During ablation-mode demo testing, Track B flagged a factually CORRECT
grounded answer ("second of six children, five siblings") as unsupported
(score 0.99), likely due to sensitivity to paraphrasing rather than genuine
factual disagreement. Noted as a known limitation, not investigated further --
consistent with Phase 4's finding that the detector is "overconfident" rather
than paranoid overall, but shows this can occasionally cut the other way on
close paraphrases.

