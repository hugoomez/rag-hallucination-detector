# Research: long-context truncation and the RAGTruth encoder landscape

**Status:** reconstructed from in-repo sources (Phase 1 EDA, ADR-001/004/010/011/012/013/014/021,
`docs/theory.md`, `docs/notes.md`, `README.md`), plus the external literature figures in
§2 and §4 restored from the original research pass. Every number about *this project* is
traceable to a committed artifact. Figures about *other systems* were **verified against
primary sources on 2026-07-25** — see [Provenance](#provenance) for which are confirmed
at origin and which remain secondary. That pass found the three source papers disagree
with each other on the same systems by up to 22 F1 points; §4.0.1 documents it.

---

## 1. The problem, quantified

RAGTruth pairs a retrieved context with a generated response and asks whether the
response is faithful to that context. Verifying faithfulness requires the model to see
both. DeBERTa-v3-base (ADR-001) accepts 512 tokens. RAGTruth does not fit.

Phase 1's EDA (`notebooks/01_eda_ragtruth.ipynb`, summarised in `docs/notes.md`)
measured the gap in DeBERTa-v3 tokens:

| | context mean | context max | notes |
|---|---|---|---|
| QA | 307 | 617 | safest task type |
| Data2txt | 761 | — | most consistent; nearly all exceed 512 |
| Summary | 690 (std 398.7) | 2189 | widest spread, longest tail |

Responses are almost always short (mean 160 tokens), which is what makes the
"never truncate the response" rule affordable.

Combining context + response + special tokens:

- **70.34%** of all rows exceed 512 tokens.
- Non-uniform by task type: **QA 34.31%**, **Summary 75.77%**, **Data2txt 99.89%**.
- Truncation is **biased toward the hard cases**: 50.52% of truncated rows are
  hallucinated, vs 43.08% globally. The rows where evidence is most needed are the rows
  most likely to lose it.

One row (`source_id` 11845, Summary) has a response that alone tokenizes to 770 tokens,
which breaks the "never truncate the response" guarantee at any context budget. It was
dropped rather than weakening the rule (ADR-006).

## 2. Options surveyed

ADR-004 records four families of solution that were considered before committing.

### 2.1 Smarter truncation

Sun et al. (2019), *How to Fine-Tune BERT for Text Classification?*, proposes head-only,
tail-only, and head+tail truncation. **Rejected as a long-term fix.** Its motivating
assumption — that salient information clusters at the start and end of a document —
does not hold for RAGTruth: Data2txt contexts are structured records with no
narrative ordering, and Summary evidence is scattered throughout. Head+tail would
optimise for a document shape this dataset does not have.

### 2.2 Sliding window over a 512-token encoder (Luna-style)

Keep a 512-token encoder and slide a window across the context. Luna (Belyi et al.,
Galileo, COLING 2025) is the reference implementation: a DeBERTa-v3-large NLI backbone
(440M params), context chunked into overlapping windows, **question and response repeated
in every window**, token-level support prediction per window.

The part worth stealing is the aggregation, which is not the obvious one:
**max-support over windows, then min over tokens.** The naive rule — flag if *any*
window flags — is actively wrong here, because a window that simply does not contain a
claim's evidence will report that claim unsupported. Every claim is unsupported by most
windows almost by construction. Taking the max support a token achieves across windows
first, and only then taking the weakest token as the response verdict, keeps a single
piece of evidence anywhere in the context sufficient to acquit a claim.

**Deprioritised, never built.** It is a valid demonstration of working within the ADR-001
constraint, but it costs more implementation complexity than switching backbone, for
comparable expected gain. ADR-004 kept it as a fallback conditional on Approach 1
underperforming; Approach 1 did not underperform, so it was never revisited.

### 2.3 Long-context encoders

ModernBERT, Longformer, and BigBird were surveyed. **ModernBERT-base was chosen**
(149M params, native 8,192-token context), for three reasons recorded in ADR-004:

1. It removes the constraint outright rather than working around it.
2. It is the backbone of the then-current encoder SOTA on RAGTruth (LettuceDetect),
   so the comparison is against a published, reproducible recipe rather than an
   independently invented one.
3. It fits a free Colab/Kaggle T4 with `attn_implementation="sdpa"` — FlashAttention 2
   is unsupported on Turing GPUs, which is the practical constraint that decided
   ModernBERT-base over ModernBERT-large at that point in the project.

**Longformer and BigBird were not pursued.** Both solve the same problem — sparse /
windowed attention to make long sequences tractable — and either would have removed the
512-token ceiling. Neither had RAGTruth-specific validation to compare against, whereas
ModernBERT did, via LettuceDetect: choosing it meant a published recipe and a published
number to be measured against, instead of an independently invented setup whose
underperformance could not be attributed. ModernBERT is also the more recent design and
was built for efficient inference on the class of GPU actually available here. Reason 2
above is doing most of the work in this decision; the architectures themselves were not
benchmarked against each other.

### 2.4 Claim decomposition + retrieval + NLI (AlignScore / MiniCheck / RefChecker style)

Split the response into claims, retrieve or score against context chunks, run an
entailment check per (chunk, claim) pair, and aggregate. This sidesteps the token limit
by construction and was ADR-004's intended flagship ("Approach 3"). Three reference
systems, at three granularities:

- **AlignScore** (Zha et al., ACL 2023, [arXiv:2305.16739](https://arxiv.org/abs/2305.16739))
  — a 355M RoBERTa-based *alignment function* trained on 4.7M examples unified from seven
  task families (NLI, QA, paraphrase, fact verification, information retrieval, semantic
  similarity, summarization). Context is split into ~350-token chunks; each claim sentence
  is scored against **all** chunks, taking the max per sentence and then averaging across
  sentences. Matches or exceeds ChatGPT- and GPT-4-based metrics on its benchmarks at a
  fraction of the size — the direct evidence that a small trained alignment model can beat
  a large prompted one on this task shape.
- **MiniCheck** (Tang, Laban & Durrett, EMNLP 2024,
  [arXiv:2404.10774](https://arxiv.org/abs/2404.10774)) — the efficiency argument taken
  further. The best variant, MiniCheck-FT5 (770M), reaches near-GPT-4 fact-checking
  accuracy at **>400× lower inference cost**. Grounding is **claim-level, not
  span-level**: it tells you a claim is unsupported, not which characters are the problem.
- **RefChecker** (Amazon, [arXiv:2405.14486](https://arxiv.org/abs/2405.14486)) — the
  finest granularity of the three. An LLM extracts **subject–predicate–object claim
  triplets** from the response, and each triplet is verified against the context
  independently. More precise than sentence-level checking, at the cost of an LLM
  extraction step in the loop.

Note what none of them produce: character offsets into the response. AlignScore and
MiniCheck return a score per sentence or per claim; RefChecker returns a verdict per
triplet. Track B's per-token labelling is what actually yields the character spans the
demo highlights — which is the concrete sense in which it superseded Approach 3 rather
than merely substituting for it.

**Never built.** ADR-004's status note records why: once Track B (binary token-level
classification, ADR-013/014) matched LettuceDetect and became the project's best model,
it already delivered exact-span granularity on truncation-free input — Approach 3's main
motivation — at lower implementation cost.

The zero-shot NLI baseline that *was* built (`src/models/nli_baseline.py`) is a
degenerate relative of this family: same premise/hypothesis decomposition, but comparing
every response sentence against every context chunk instead of a retrieved top-k.
Its failure mode (ADR-009) is informative for anyone reconsidering Approach 3 —
see §4.

## 3. What was decided, and what actually happened

ADR-004 committed to a phased plan:

| Stage | Plan | Outcome |
|---|---|---|
| MVP | Keep DeBERTa-v3-base, truncate context only, never the response; evaluate per `task_type` to price the truncation honestly | Shipped as Track A. Test F1 0.7116 |
| Approach 1 | Swap backbone to ModernBERT-base, ~4,096-token inputs | Shipped. Truncation eliminated (ADR-011); test F1 0.7254 (ADR-012) |
| Approach 3 | Claim decomposition + retrieval + NLI | Never built; superseded by Track B |

### 3.1 Truncation was eliminated, exactly as hypothesised (ADR-011)

At `max_length=4096`, **0.00%** of rows require truncation — across all three task types
and all three splits. Verified twice, independently: by a pre-tokenization length
diagnostic (max observed combined length **2,618 tokens**) and by the `was_truncated`
flag computed during real tokenization. The 770-token response outlier dropped under
DeBERTa (ADR-006) fits comfortably.

The later error analysis re-confirms this from the other end: the longest
model-visible test sequence is **2,388 tokens** against a 4,096-token window, so no test
row is truncated (README, Error analysis point 3).

### 3.2 But the *mechanism* of the gain was predicted wrong

This is the part worth carrying forward.

ADR-010 ran a correlational diagnostic on Track A's own predictions
(`scripts/analyze_track_a_predictions.py`), correlating `was_truncated` with per-row
correctness. It found the opposite of the intuitive story: truncated rows had **higher**
recall on hallucinated examples than untruncated ones (Summary 0.278 vs 0.151; QA 0.750
vs 0.576) but **lower** overall accuracy (0.778 vs 0.859). Conclusion: truncation's cost
is concentrated in **precision** — the model over-flags faithful content when it cannot
confirm support — plausibly reinforced by the mildly hallucination-favouring class
weights `[0.90, 1.12]` used in training.

ADR-012 then measured the real across-architecture effect, and the prediction did not
hold:

| | Track A (DeBERTa, 512) | Approach 1 (ModernBERT, 4096) |
|---|---|---|
| Precision | 0.7367 | 0.6839 |
| Recall | 0.6882 | 0.7731 |
| F1 | 0.7116 | 0.7257 |
| Summary recall | 0.245 | 0.569 |
| Summary F1 | 0.332 | 0.509 |

Precision *fell*; recall rose sharply, with the gain concentrated almost entirely in
Summary — the task type with the longest, most dispersed evidence. The mechanism that
actually improved was the model's ability to **locate scattered evidence**, not reduced
false-positive behaviour under partial-context uncertainty.

**Methodological lesson (ADR-012):** a correlational diagnostic on a fixed architecture
does not predict the causal effect of changing that architecture. Both findings are
valid; they answer different questions. Anyone reading ADR-010's "truncation costs
precision" as a forecast rather than a description will mis-plan the next experiment.

### 3.3 Summary's weakness was never a truncation problem

ADR-010 also noted that Summary's low recall held for truncated (0.278) *and*
untruncated (0.151) rows, so truncation could not be the primary cause. ModernBERT's
long context did resolve a large part of it (0.245 → 0.569 recall), but Summary remains
the weakest task at every stage of the project — Track B (arm-b) scores F1 0.4904,
recall 0.3775 on Summary against F1 0.8675 on Data2txt. The residual cause is
hypothesised to be RAGTruth's "Subtle" span types; see
[`02-implicit-true-audit.md`](02-implicit-true-audit.md), which shows that most of the
"Subtle Baseless Info" category (73.6%) is `implicit_true` — ungrounded-but-true content
that RAGTruth's annotators marked as low-severity. Note this does **not** make the Subtle
weakness a measurement artifact: excluding that subclass *raises* the miss rate
([`04-subtle-only-reconciliation.md`](04-subtle-only-reconciliation.md)).

## 4. The RAGTruth landscape as this project used it

### 4.0 Where the encoders actually sit

Response/example-level F1 on RAGTruth, as surveyed:

> **This table cannot be read as a leaderboard.** Verification at primary sources
> (2026-07-25) found that the three papers report *different F1 values for the same
> nominal systems* — see §4.0.1. Rows below carry the figure from the **originating**
> paper wherever one exists, with the source named per row.

| System | Type | Backbone | RAGTruth F1 | Primary source |
|---|---|---|---|---|
| **RAG-HAT** (Song et al., EMNLP 2024) | decoder, DPO-tuned | Llama-3-8B | **83.9%** | ✅ Song Table 2 (P 87.3 / R 80.8) |
| LettuceDetect-large | encoder, token-level | ModernBERT-large | 79.22% | ✅ Kovács & Recski Table 2 |
| RAGTruth paper baseline (Niu et al., ACL 2024) | decoder, fine-tuned | Llama-2-13B | **80.7%** | ✅ Niu Table 5 (P 88.6 / R 74.1). *Song reports 78.7 for the same system — see §4.0.1* |
| LettuceDetect-base | encoder, token-level | ModernBERT-base | 76.07% | ✅ Kovács & Recski Table 2 |
| **This project — Track B (arm-b)** | encoder, token-level | ModernBERT-base | **76.31%** | `results/arm_b_metrics.json` |
| Prompted GPT-4-turbo | prompted decoder | GPT-4-turbo | **68.3%** | ✅ Niu Table 5 (P 54.8 / R 90.5). *Song reports 76.7, Kovács 63.4 — see §4.0.1* |
| Luna (Belyi et al., COLING 2025) | encoder, sliding window | DeBERTa-v3-large (440M) | 65.4% | ⚠️ **secondary only** — Kovács & Recski Table 2; Luna's own paper unverifiable |
| SelfCheckGPT w/ GPT-4-turbo | sampling-based | GPT-4-turbo | **60.5%** | ✅ Niu Table 5 (P 49.5 / R 77.7) |

### 4.0.1 The same systems do not get the same score across papers

This is the load-bearing caveat, and it was only discovered by checking primaries rather
than trusting a comparison table:

| System | Niu et al. (RAGTruth, Table 5) | Song et al. (RAG-HAT, Table 2) | Kovács & Recski (LettuceDetect, Table 2) |
|---|---:|---:|---:|
| Prompt GPT-4-turbo | **68.3** | **76.7** | **63.4** |
| SelfCheckGPT gpt-3.5-turbo | **36.6** | — | **58.8** |
| SelfCheckGPT gpt-4-turbo | **60.5** | — | — |
| Finetuned Llama-2-13B | **80.7** *(self-report)* | **78.7** *(reproduction)* | 78.7 *(copied from Song)* |
| RAG-HAT | — | **83.9** | 83.9 *(copied from Song)* |

Prompt GPT-4-turbo spans **13.3 F1 points** across three papers. SelfCheckGPT
gpt-3.5-turbo spans **22.2 points** (36.6 vs 58.8 — and the recall figures, 28.0 vs 71.9,
are not close enough to be the same experiment). LettuceDetect's Llama-2-13B and RAG-HAT
rows carry precision/recall **identical** to RAG-HAT's Table 2, confirming they were
copied rather than recomputed.

Niu et al. describe their Overall column as an **average** F1; Song et al. label theirs
OVERALL without defining the aggregation. That is a candidate explanation for part of the
spread, but it does not account for the SelfCheckGPT recall gap, and no source states its
protocol precisely enough to reconcile them. **Treat every cross-paper delta in the table
above as indicative only.**

This finding is independent of the `implicit_true` work and arguably belongs in the
write-up on its own terms: for a paper arguing that RAGTruth numbers do not mean what they
appear to mean, a demonstration that the field's own published figures disagree by up to
22 F1 points is directly on-thesis.

**Correction log for this section.** On 2026-07-25 an earlier pass "corrected" the
SelfCheckGPT row from *"w/ GPT-4-turbo, ~60.5%"* to LettuceDetect's `gpt-3.5-turbo` row
(58.8), on the assumption that LettuceDetect's table was authoritative. **That was wrong
in both directions**: the original figure was correct at primary source (Niu Table 5,
60.5), and the substitute was a different system configuration taken from a secondary
source. The row is restored. Two rows that *were* genuinely misattributed to Niu et al.
are now fixed: Llama-2-13B (78.7 → **80.7**) and Prompt GPT-4-turbo (63.4 → **68.3**);
neither 78.7 nor 63.4 appears anywhere in the RAGTruth paper.

Three things this table settles, and they are the reason it is worth reproducing here
rather than leaving as a list of names:

1. **The encoder track is not the frontier — it is the cheap frontier.** RAG-HAT's 83.9%
   beats every encoder on this list by a wide margin, and it does so with a fine-tuned
   Llama-3-8B and Hallucination-Aware Tuning (a DPO-based objective), not an encoder at
   all. Even the RAGTruth paper's own fine-tuned Llama-2-13B baseline (80.7% self-reported,
   78.7% as reproduced by Song et al.) outscores LettuceDetect-base on either figure.
   Any claim this project makes about being "near SOTA" must be qualified as *near encoder
   SOTA* — a ~150M-parameter model reaching 76% against an 8B decoder's 84% is the honest
   framing, and it is a good one, because the encoder runs on a T4 and returns character
   spans.
2. **Prompting a frontier model is not competitive.** In the benchmark's own paper,
   GPT-4-turbo prompted directly scores 68.3% and SelfCheckGPT over GPT-4-turbo 60.5% —
   both far below the fine-tuned Llama-2-13B's 80.7%. The conclusion survives the
   cross-paper disagreement documented in §4.0.1: on no source's numbers does a prompted
   frontier model approach a fine-tuned one. Fine-tuning on RAGTruth is not a formality;
   it is most of the signal. (The Luna comparison in the original version of this point is
   dropped — at 65.4% from a secondary source, it sits inside the disagreement band and
   cannot carry an ordering claim.)
3. **Sliding-window over a 512-token encoder underperforms a native long-context
   encoder.** Luna uses a *larger* backbone than LettuceDetect (DeBERTa-v3-large, 440M vs
   ModernBERT-base, 149M) and still scores ~11 points lower. This retroactively supports
   ADR-004's §2.2 decision to deprioritise the sliding-window approach in favour of
   switching backbone — the decision was made on implementation-cost grounds before this
   comparison was available, and the numbers happen to agree.

### LettuceDetect (Kovács & Recski 2025, arXiv:2502.17125)

The reference system throughout. ModernBERT backbone, **binary** token classification
(supported / hallucinated), plain cross-entropy, spans reconstructed at inference by
merging consecutive positive tokens.

Published numbers used as targets in this repo:

| Metric | LettuceDetect-base | LettuceDetect-large | This project (Track B, arm-b) |
|---|---|---|---|
| Example/response-level F1 | 76.07% | 79.22% | **76.31%** |
| Span-level F1 (char-overlap) | 55.44% | 58.93% | **53.21%** |

Their published base→large gap (+3.15 example-F1, +3.49 span-F1) is the target that
ADR-021's 3-seed scaling comparison was run against; it replicated on response-level F1
(+3.11 across 3/3 seeds) and was exceeded at span level (+4.08).

Two facts about their evaluation mattered more than the modelling:

1. **Their span metric is character-overlap micro P/R/F1**, not strict entity match:
   sum character overlap over all pred×gold span pairs; P = overlap / total predicted
   chars, R = overlap / total gold chars. Implemented verbatim as `char_span_prf` in
   `src/models/train_token_level.py`.
2. **Their headline 76–79% is example-level, not span-level.** ADR-013 records that
   Track B's first run was being compared against the wrong number — strict seqeval
   exact-entity F1 against their example-level F1 — an unfair comparison independent of
   the modelling bug that also existed.

Their exact recipe (paper + repo): lr 1e-5, batch 8, 6 epochs, A100, no class weighting,
max_len 4096, checkpoint selection on token-level F1. Track B's first binary run
deviated on all of lr (2e-5), effective batch (16), epochs (8) and selection metric
(response-level F1); ADR-020's arm-b re-aligned them and gained +2.1 span-F1 points from
the selection metric alone. See [`03-candidate-methods.md`](03-candidate-methods.md).

### RAG-HAT (Song et al., EMNLP 2024)

**Shared authorship with the benchmark — five of seven authors, not one.** Verified at
primary source 2026-07-25 (ACL Anthology `2024.emnlp-industry.113`, DOI
`10.18653/v1/2024.emnlp-industry.113`): RAG-HAT's author list is Song, Wang, Zhu, Wu,
Cheng, Zhong, **Niu**. Cross-checked against RAGTruth's verified author list (Niu, Wu, Zhu,
Xu, Shum, Zhong, Song, Zhang — arXiv:2401.00396 primary source), **five of RAG-HAT's seven
authors — Niu, Wu, Zhu, Zhong, and Song — also appear on the RAGTruth paper**, not Cheng
Niu alone as an earlier pass of this note recorded. Both are industry work from the same
group. This is a decisive fact, not a cosmetic one: with 5/7 authorship overlap, RAG-HAT is
not a separate group's replication that happens to share a name — it is substantially the
same team publishing a follow-up system on their own benchmark. It means RAG-HAT's 83.9% is
not an independent external validation of the benchmark, and a write-up that presents the
§4.0 table as a field-wide leaderboard should say so plainly. It also bears on the
`implicit_true` question: the group closest to the annotation scheme published no code, so
whether their pipeline reads the field is unverifiable.

The strongest system surveyed, at 83.9% F1. Llama-3-8B fine-tuned with
**Hallucination-Aware Tuning**, a DPO-based objective: rather than training a separate
detector, it tunes the *generator* on preference pairs so that it prefers faithful
continuations. A decoder LLM, not an encoder — and a different problem framing, since it
changes the model being audited rather than auditing it from outside. Not reproducible
under this project's constraints (an 8B DPO run is not a free-tier T4 workload), and not
directly substitutable either: it produces better generations, not span annotations over
someone else's generations.

### RAGTruth's own baselines (Niu et al., ACL 2024, arXiv:2401.00396)

The benchmark paper's reference points, all reproduced in the §4.0 table: fine-tuned
Llama-2-13B at **80.7%**, prompted GPT-4-turbo at **68.3%**, SelfCheckGPT with GPT-4-turbo
at **60.5%** (all Table 5, verified at source 2026-07-25; the paper also reports
SelfCheckGPT gpt-3.5-turbo at 36.6% and LMvLM gpt-4-turbo at 50.1%). The gap between the
first and the rest is the paper's own argument for
fine-tuning on the benchmark rather than prompting a frontier model at it.

### Luna (Belyi et al., Galileo, COLING 2025)

65.4% F1. Design covered in §2.2 — DeBERTa-v3-large, sliding windows with the
question+response repeated per window, max-support-over-windows then min-over-tokens
aggregation. Its relevance here is as the counterfactual for the ADR-004 decision: a
larger backbone worked around the context limit and still lost to a smaller one that
never had the limit.

### AlignScore, MiniCheck, RefChecker

The claim-decomposition family, covered in §2.4. Named in ADR-004 and in
`src/models/nli_baseline.py`'s docstring as the stylistic reference for the baseline's
sentence-splitting and per-pair NLI scoring. **None of them report RAGTruth numbers in
what was surveyed** — they are evaluated on their own benchmark suites, so they do not
appear in the §4.0 table. They informed the *design* of Approach 3, not a target score.

### Why the baseline's failure matters for this family

The zero-shot NLI baseline scores test F1 0.5234, barely above the trivial
"always hallucinated" F1 0.5177. ADR-009 diagnosed the cause as calibration of the raw
per-sentence scores, **not** the aggregation rule:

- The "contradicted" flag fires on 55.7% of genuinely faithful sentences vs 53.8% of
  hallucinated ones — almost no discriminative signal.
- Median max-entailment for faithful sentences is only 0.169 (25th percentile 0.030):
  faithful responses routinely synthesise across several context chunks, and no single
  chunk entails the resulting sentence.
- Switching to a proportion-based aggregation rule moved val F1 only 0.611 → 0.632,
  ruling out aggregation as the cause.

That second point is the load-bearing one for Approach 3: a retrieval step selects
*which* chunks to compare, but does not fix a scorer whose entailment signal collapses on
multi-chunk synthesis. Data2txt is the exception (baseline F1 0.783), where ADR-008's
task-type-aware chunking turns structured fields into clean `key: value` evidence — a
hint that the decomposition family works when chunks are atomic facts and struggles when
they are prose.

## 5. ModernBERT vs DeBERTa-v3, as decided here

**Why DeBERTa-v3-base first (ADR-001):** strongest available encoder at its size on MNLI
specifically (He et al. 2021), which is the task shape of the detector; disentangled
attention separates content and position, relevant for negation and subject/object
distinctions; v3's ELECTRA-style RTD pretraining gives signal on 100% of tokens rather
than the 15% MLM masks; and `base` fits a free T4.

**Why ModernBERT-base replaced it:**

| | DeBERTa-v3-base | ModernBERT-base |
|---|---|---|
| Native context | 512 | 8,192 (used at 4,096) |
| Params | ~184M (86M backbone) | ~149M |
| RAGTruth rows truncated | 70.34% | 0.00% |
| Response-level test F1 (same recipe) | 0.7116 | 0.7254 |
| T4 constraint | — | needs `attn_implementation="sdpa"`; FA2 unsupported on Turing |

The comparison in the last row of the table is the honest one: same task, same training
recipe, different backbone and context length. It is a joint backbone+context effect, not
an isolated context-length ablation — the repo never ran ModernBERT at 512 tokens to
separate the two.

ModernBERT-**large** was tried later, at 3 seeds, and is a clear further gain
(+3.1 response-F1, +4.1 span-F1 on paired seeds). See
[`../EXPERIMENT_LEDGER.md`](../EXPERIMENT_LEDGER.md).

---

## Provenance

| Claim | Source in repo |
|---|---|
| Context/response length stats, 70.34% exceedance, per-task exceedance, 50.52% vs 43.08% | `docs/notes.md` (Phase 1 EDA), `README.md` Dataset section, ADR-004 |
| Dropped 770-token outlier | ADR-006, `src/data/preprocess.py` |
| Truncation options surveyed and rejected | ADR-004 |
| 0.00% truncation, 2,618-token max | ADR-011 |
| 2,388-token max visible test sequence | `README.md`, Error analysis point 3 |
| Track A vs Approach 1 mechanism | ADR-010, ADR-012, `results/finetuned_track_a_metrics.json`, `results/finetuned_approach1_modernbert_metrics.json` |
| Baseline calibration diagnosis | ADR-009, `scripts/diagnose_baseline_flagging.py`, `results/baseline_nli_metrics.json` |
| LettuceDetect recipe, metrics, char-overlap definition | ADR-013, ADR-014, ADR-020, `src/models/train_token_level.py` (`char_span_prf`) |
| Track B / arm-b numbers | `results/arm_b_metrics.json` |
| DeBERTa rationale | ADR-001, `docs/theory.md` §B, `docs/notes.md` |
| Base→large scaling replication (+3.11 / +4.08) | ADR-021, `results/seed_aggregate.json` |

### External figures — provenance and status

Everything in the §4.0 table and the system descriptions in §2.2, §2.4, and §4 is a
**published figure restored from the original research pass** (2026-07, author's notes),
not a repo artifact and not a re-derivation:

| System | Figures used | Citation given in the notes |
|---|---|---|
| RAG-HAT | 83.9% F1 (P 87.3 / R 80.8); Llama-3-8B + DPO-based Hallucination-Aware Tuning | ✅ **Primary, verified 2026-07-25**: Song et al., EMNLP 2024 industry track, Table 2 |
| RAGTruth baselines | Llama-2-13B **80.7%**; GPT-4-turbo prompted **68.3%**; SelfCheckGPT+GPT-4-turbo **60.5%** | ✅ **Primary, verified 2026-07-25**: Niu et al., ACL 2024, arXiv:2401.00396, Table 5 (full table reproduced in §4.0.1) |
| LettuceDetect | base 76.07 / 55.44; large 79.22 / 58.93 | Kovács & Recski 2025, arXiv:2502.17125 |
| Luna | 65.4% F1 (P 52.7 / R 86.1) | ⚠️ **Secondary only** — Kovács & Recski Table 2. Luna's own paper could not be text-extracted and releases no code/weights; not confirmed at origin |
| AlignScore | 355M RoBERTa-based; 4.7M examples / 7 task families; ~350-token chunks; max-per-sentence then average | Zha et al., ACL 2023, arXiv:2305.16739 |
| MiniCheck | MiniCheck-FT5 770M; near-GPT-4 accuracy at >400× lower cost; claim-level | Tang, Laban & Durrett, EMNLP 2024, arXiv:2404.10774 |
| RefChecker | subject–predicate–object triplet extraction + per-triplet verification | Amazon, arXiv:2405.14486 |

**Verification status (2026-07-25).** RAG-HAT, the RAGTruth baselines, and LettuceDetect
have now been **confirmed at their originating papers** (Song Table 2; Niu Table 5;
Kovács & Recski Table 2). **Luna remains secondary-only** — its paper resists text
extraction and it releases no code or weights, so its 65.4% is cited via LettuceDetect.
AlignScore, MiniCheck, and RefChecker report no RAGTruth numbers and are cited for design,
not score.

The concern flagged in the original version of this note turned out to be **correct and
worse than expected**: the §4.0 table did place numbers side by side that were not
computed under identical conditions, and §4.0.1 now quantifies the damage — up to 22 F1
points of disagreement between papers for the same nominal system. Cross-system deltas in
§4.0 are indicative only.

## Known gaps

- **The §4.0 table is a survey, not a controlled comparison — and now demonstrably so.**
  §4.0.1 shows the three source papers reporting different F1s for the same nominal
  systems (Prompt GPT-4-turbo: 68.3 / 76.7 / 63.4; SelfCheckGPT gpt-3.5-turbo: 36.6 vs
  58.8). Treat cross-system deltas as indicative only.
- **No source states its aggregation protocol precisely enough to reconcile the spread.**
  Niu et al. say "average"; Song et al. say "OVERALL" without definition. Whether every
  row uses the same positive-class definition and the same test split is still unverified,
  and probably unverifiable from the published text alone.
- **Luna is the one remaining unverified figure.** Its 65.4% is cited via LettuceDetect's
  Table 2; the paper could not be text-extracted and no code or weights are released.
- **Longformer / BigBird were never benchmarked**, against ModernBERT or against each
  other. §2.3's reasoning is a decision rationale, not a measurement.
- **The claim-decomposition family has no RAGTruth numbers** in what was surveyed, so
  Approach 3's expected performance on this benchmark was never estimated from
  literature — one of the reasons it stayed a plan rather than a prediction.
