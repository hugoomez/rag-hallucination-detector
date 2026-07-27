# Two Kinds of Hallucination, One Positive Class: A Concentrated Conflation in RAGTruth

*Working draft. Chapter-by-chapter, following [`07-paper-outline.md`](07-paper-outline.md)
structure and word budgets. Venue undecided.*

---

## Abstract

Every published evaluation on RAGTruth (Niu et al., 2024) scores its gold hallucination
class as a single, undifferentiated positive: a span is hallucinated or it is not. The
benchmark's own annotators mark a distinction that every public evaluation discards. The
`implicit_true` field records spans that are unsupported by the retrieved context but
happen to be true — 13.5% of gold spans by count, 14.56% by character mass — and marks
them, in 90.6% of cases, with the annotators' own low-intensity severity prefix, against
4.4% for other gold spans. These two subclasses are marked as differing in severity by the
benchmark's own annotators, and are nonetheless scored identically. The conflation is not a
scattering of annotator disagreement: it concentrates in one span type — nearly three
quarters of spans typed "Subtle Baseless Info" — and is unevenly distributed by task type,
the structure of a real, recognized subclass rather than of noise. We quantify what this
structured conflation costs under RAGTruth's official scoring: because the protocol counts
every flagged character directly, any prediction set that excludes the low-intensity
subclass is capped at 90.5% char-overlap span recall against gold, independent of any
model — a direct consequence of the conflation, not a second, independent result. We also
tested whether the conflation could be exploited at training time, by down-weighting the
low-intensity subclass in the loss under a pre-registered decision rule; it produced no
measurable improvement in clean-span F1 at the one setting tested, a negative result we
report as one rather than repackage as a further finding. We report this as a
benchmark-quality contribution, not a detection method, and show that the metric needed to
distinguish these subclasses already exists in data the benchmark ships but no published
evaluation uses.

**Keywords:** RAGTruth; hallucination detection; retrieval-augmented generation; benchmark
evaluation; label-class conflation; annotation metadata

---

## 1. Introduction

Every published evaluation on RAGTruth (Niu et al., 2024) treats the benchmark's gold
annotations as a single, undifferentiated positive class. A span is hallucinated or it is
not; a response is hallucinated or it is not; the reported F1 aggregates over that binary.
This is how the benchmark's own reference baseline scores itself, how LettuceDetect
(Kovács & Recski, 2025) scores itself, and how the comparison tables in this literature are
assembled.

That positive class is not homogeneous. RAGTruth annotates under a faithfulness
objective — it targets claims that are *unsupported or contradictory* with respect to the
retrieved context — and under that objective a span can earn a positive label in two
materially different ways. It can assert something the context does not support and that is
also false: **ungrounded-and-false**. Or it can assert something the context does not
support but which happens to be true: **ungrounded-but-true**. Both are hallucinations under
the stated objective, and RAGTruth labels both positive, correctly. These two subclasses
are marked as differing in severity by the benchmark's own annotators, and are
nonetheless scored identically.

The benchmark ships the field that separates them. `implicit_true`, defined in the corpus
README as marking a span that "is correct while the info is not mentioned in the context,"
is set on 1,928 of 14,289 gold spans — 13.5% by span count, and 14.56% of gold-span
character mass. The two figures are different quantities and are reported here as such;
an earlier version of this work conflated them.

This yields the gap the paper addresses. **Published RAGTruth evaluations score
ungrounded-and-false content and ungrounded-but-true content identically, even though the
benchmark's own annotators mark them as differing in severity — so no reported F1
distinguishes performance on the two subclasses.**

The distinction matters most where a detector's output feeds a human reviewer rather than
an automated block: such a deployment would reasonably want to prioritize
ungrounded-and-false content — the subclass a reviewer must catch — over
ungrounded-but-true content, which is wrong about grounding but not about the world. We do
not evidence how production RAG deployments actually triage flagged output, and the point
below does not depend on that evidence. It is narrower: RAGTruth's official scoring cannot
express such a priority even where a deployment holds it. The protocol counts a decision to
deprioritize the low-severity subclass as an equivalent miss to failing on the high-severity
one, so a detector built for triage is scored by the same metric as one that draws no such
distinction. No published evaluation on this benchmark can say whether a system was built
with this priority in mind or simply happened to miss fewer ungrounded-but-true spans.

A reader may reasonably respond that every benchmark carries some label noise, so what? The
objection does not apply here, for two reasons, and it is worth settling before the evidence
rather than after it. First, nothing in this paper is *estimated* noise. `implicit_true` is
a self-declared, machine-readable field that the benchmark publishes with the data, and that
the systems trained on that data discard: we verified in code (2026-07-25) that
LettuceDetect's `preprocess_ragtruth.py` reads only `start`, `end`, and `label_type`, and
that RAGTruth's own vendored reference baseline contains no occurrence of the field at all.
Luna (Belyi et al., 2024/2025) and RAG-HAT (Song et al., 2024) publish neither code nor
weights; their treatment of the field is unverified and we do not assume it. Second, and
more importantly, the field does not mark an error. Two independent sources establish that
it is a **severity qualifier applied within the positive class**, not a retraction of the
label. The README's own wording separates correctness from groundedness. And RAGTruth's
per-span `meta` field — a single annotator-written string, whose first line is a severity
prefix and whose remainder is free text — carries both halves of the same signal: it
consistently asserts that the content is true but ungrounded ("these details are correct,
however, not directly mentioned in the passages"), and its severity prefix opens with `LOW`
for 90.6% of flagged spans, against 4.4% of all other gold spans. The labels are right. What
is at issue is that two error types of very different consequence are being scored as one.
We therefore use the term **label-class conflation** throughout, and not "label noise" or
"annotator disagreement."

The magnitude is exact, because the RAGTruth test set is the object of study rather than a
sample from which anything is estimated. Flagged spans account for 8,151 of 85,877
characters of test gold span mass — 9.49%. Stated as what it actually bounds: **under
RAGTruth's official scoring, any prediction set that omits the low-intensity subclass from
its positives is capped at 90.5% char-overlap span recall against gold** — a constraint on
the scoring protocol, not a claim about what any detector does. This is a bound on achieving
two goals at once, not a ceiling on a metric in isolation — a system that flags everything
still reaches 100% recall — and every statement of it in this paper carries that
conditional. At the coarser response level the same argument gives a corroborating
instance: 49 of the 943 hallucinated test responses consist *entirely* of flagged spans, so
the same protocol-level argument caps response-level recall at 94.8% for any prediction set
that omits the low-intensity subclass.

One clarification matters before the paper puts this bound to use. It is a property of
RAGTruth's scoring protocol — computable directly from the released gold labels,
independent of whether any detector can, or should, restrict its predictions to the
high-severity subclass. It does not propose such a detector, and no experiment in this
paper builds one. What it shows is that any prediction set scored against official gold
while omitting the low-intensity subclass is capped at that recall by construction, before
any model exists.

§4 establishes what this property of the benchmark actually is: not a scattering of
annotator disagreement but a **structured subclass** — concentrated in one span type,
unevenly distributed by task, and already marked as low-severity by the annotators who
labelled it. That structure is this paper's central finding. Two things follow from it, and
neither is inflated into a co-equal second finding. §6 quantifies what the conflation costs
under RAGTruth's official scoring — a direct arithmetic consequence of §4, computable before
any model is trained: the scoring protocol counts every flagged character directly,
undiluted, so it fixes what any reported RAGTruth recall figure can mean once the
low-intensity subclass is excluded from what counts as caught. §5 asks whether the same
conflation can instead be *exploited* at training time, by teaching a model to discount the
low-severity subclass through loss down-weighting; under a pre-registered decision rule, at
the setting tested, it could not — a negative result, reported as one, not repackaged as a
second leg. The two are worth holding side by side for one reason, developed in §7.1 as a
discussion point rather than asserted here as the thesis: the same quantity that is too
small to move a training process is large enough to bend a scoring function, so their
difference is not a contradiction.

This is a benchmark-quality contribution, not a detection method. No number reported here
is a new record, and none needs to be. The token-level detector used as the instrument
throughout — a ModernBERT-based (Warner et al., 2024/2025) binary token classifier
replicating LettuceDetect's recipe — is prior, separately developed infrastructure of this
project, documented in the repository (ADR-013, ADR-014, ADR-020) and not re-argued here.

---

## 2. Background and Related Work

### 2.1 RAGTruth's construction and faithfulness objective

RAGTruth (Niu et al., 2024) is a hallucination corpus of 17,790 LLM responses spanning
three retrieval-augmented task types — question answering, summarization, and data-to-text
generation — with human annotation at the word/span level. Its annotation target is
**faithfulness, not factuality**: annotators mark claims that are *unsupported or
contradictory* with respect to the retrieved context. A claim that is true of the world but
absent from the context is therefore in scope, by design. Spans carry a `label_type` drawn
from four categories along two axes, intensity (Evident/Subtle) and kind (Baseless
Info/Conflict), and a free-text `meta` comment. The official split is 15,090 train
responses and 2,700 test responses, of which 943 are hallucinated and 1,757 faithful; all
test figures in this paper are on that official test split. Span-level scoring in the
token-level detector literature is character-overlap micro-F1, and response-level scoring
derives a response label from its token predictions.

### 2.2 The detector landscape

Response/example-level F1 on RAGTruth, each figure resolved to the paper that originates it:

| System | Type | Backbone | RAGTruth F1 | Primary source |
|---|---|---|---|---|
| RAG-HAT (Song et al., 2024) | decoder, DPO-tuned | Llama-3-8B | **83.9%** | Song Table 2 (P 87.3 / R 80.8) |
| RAGTruth baseline (Niu et al., 2024) | decoder, fine-tuned | Llama-2-13B | **80.7%** | Niu Table 5 (P 88.6 / R 74.1) |
| LettuceDetect-large (Kovács & Recski, 2025) | encoder, token-level | ModernBERT-large | 79.22% | Kovács & Recski Table 2 |
| LettuceDetect-base | encoder, token-level | ModernBERT-base | 76.07% | Kovács & Recski Table 2 |
| Prompted GPT-4-turbo | prompted decoder | GPT-4-turbo | **68.3%** | Niu Table 5 (P 54.8 / R 90.5) |
| Luna (Belyi et al., 2024/2025) | encoder, sliding window | DeBERTa-v3-large | 65.4% | ⚠️ **secondary only** — Kovács & Recski Table 2 |
| SelfCheckGPT w/ GPT-4-turbo | sampling-based | GPT-4-turbo | **60.5%** | Niu Table 5 (P 49.5 / R 77.7) |

The ordering settles what the encoder track is: **the cheap frontier, not the frontier.**
RAG-HAT's 8B-decoder DPO pipeline and the benchmark's own fine-tuned Llama-2-13B baseline
both outscore every encoder here. Two caveats attach to the table rather than to any system
in it. Luna's figure is taken from a secondary source — its paper releases no code or
weights — and is labelled so wherever it appears. RAG-HAT shares **five of its seven
authors** with RAGTruth's (Niu, Wu, Zhu, Zhong, Song), a caveat about independence, not
about the figure. **All figures in this table are response-level.** This paper's own bound
(§6) is stated at the span level first, with response-level as a corroborating instance —
the coarser granularity here, and the finer granularity there, should not be read as
directly comparable numbers. The claim-decomposition family — AlignScore (Zha et al.,
2023), MiniCheck (Tang, Laban & Durrett, 2024), and RefChecker (Hu et al., 2024) — informs
this project's own baseline design but reports no RAGTruth figures and cannot anchor a
comparison here.

This table deliberately excludes this project's own instrument, arm b (§5.1,
response-F1 76.31%): it is the replication baseline for the training-side ablation, not a
system submitted for comparison against the literature, and placing it in a table of
externally published results would misrepresent what it is.

### 2.3 On the comparability of published RAGTruth figures

Assembling the comparison above required resolving each figure to its originating paper.
Three discrepancies emerged. We report them because they bear on how such tables should be
read, not because three cases support a general claim.

First, prompted GPT-4-turbo is reported at three different response-level F1 values on
RAGTruth: **68.3** by Niu et al. (2024), the benchmark's own authors; **76.7** by Song et
al. (2024); and **63.4** by Kovács and Recski (2025) — a spread of 13.3 F1 points. The
aggregation conventions differ or are undefined. Niu et al. describe their overall column
as an *average* across task types; Song et al. label theirs OVERALL without definition;
Kovács and Recski state no convention. We could not reconcile the three from the published
text.

Second, SelfCheckGPT over gpt-3.5-turbo is reported at **36.6** by Niu et al. and **58.8**
by Kovács and Recski — a spread of 22.2 F1 points, the largest we found, and one where
recall diverges too (28.0 versus 71.9). Aggregation convention does not move the recall of
the same predictions, so that explanation is unavailable here; we could not reconcile the
two and do not claim to know the cause.

Third, the Llama-2-13B and RAG-HAT rows in Kovács and Recski's comparison table carry
precision and recall identical to Song et al.'s Table 2 (76.9 / 80.7 and 87.3 / 80.8
respectively), including the **78.7** F1 that appears in Song et al. for the RAGTruth
baseline rather than the **80.7** that Niu et al. self-report for the same system. This is
consistent with figures being carried across rather than recomputed — though it is not
proof of it. Independent evaluation could coincide, and neither paper states which was
done.

We draw no conclusion about the comparability of the RAGTruth literature as a whole; three
cases cannot support one. What they suggest is narrower: that verification of external
figures against originating sources may be undervalued in this literature, and that a
systematic audit — resolving every published RAGTruth figure to its primary source and
stated evaluation protocol — would be worth conducting. That is beyond this paper's scope.
We contribute only the observation that prompted the question.

### 2.4 `implicit_true` has no literature

The conceptual distinction this paper turns on — between a claim's *faithfulness* to its
source and its *factuality* in the world — is not new. Maynez, Narayan, Bohnet & McDonald
(2020) established it for abstractive summarization: a generated claim can be unfaithful to
its source document while still being true of the world, and conflating the two
overstates or understates a summarizer's hallucination rate depending on which direction
the conflation runs. RAGTruth's ungrounded-but-true / ungrounded-and-false split is the same
distinction, applied to retrieval-augmented generation instead of summarization. What we
claim as novel is narrower than the distinction itself: **the conceptual distinction is
established; the benchmark ships a machine-readable field encoding it; no published
RAGTruth evaluation conditions on that field.**

The field this paper audits was added to the **data**, not to the paper. RAGTruth's corpus
README dates it to February 2024; the ACL 2024 paper (Niu et al., arXiv:2401.00396) does
not mention it anywhere, checked against the full text on 2026-07-25. No source in our
review reports any RAGTruth metric conditioned on it.

There is, however, a direct precedent for metadata-conditioned scoring, and the benchmark's
own authors set it. The RAGTruth paper offers users the option to **include or exclude
`due_to_null` spans** when evaluating — spans where the model invented a value for a null
field. The authors thus recognized that one metadata field materially changes what a score
means, and built the affordance to score with and without it. They built it for one field
and not for the other. What this paper proposes for `implicit_true` — report the metric
stratified, so a reader can see which subclass a number is about — is the treatment
RAGTruth already gives `due_to_null`, applied to the field that did not get it.

What is verified about the field's handling, and what is not:

| System | Field handling | Basis (checked 2026-07-25) |
|---|---|---|
| RAGTruth reference baseline | discards both fields | **Verified in code** — buckets by `label_type` alone |
| LettuceDetect | discards both fields | **Verified in code** — `preprocess_ragtruth.py` reads `start`, `end`, `label_type` |
| Luna | unknown | **UNVERIFIED** — no public code or weights |
| RAG-HAT | unknown | **UNVERIFIED** — no public code or weights |

The claim is exactly the top two rows; we do not extend it to the leaderboard as a whole.

---

## 3. Methods

Three subsections, deliberately kept apart so that pre-registered and post-hoc work are
visibly distinct: the annotation audit that establishes the central finding (§3.1), the
pre-registered ablation behind §5's training-time attempt (§3.2), and the exploratory
re-scoring behind §6's case study (§3.3).

### 3.1 Annotation audit computation

The audit operates on RAGTruth's released `response.jsonl` and `source_info.jsonl`, one row
per (response, gold span), joined to task type through `source_id`. It is a direct
enumeration over 17,790 responses and 14,289 gold spans; the full computation is ~40 lines
of standard library Python. It is not paraphrased here: it is reproduced verbatim as
Supplement S1 (`docs/research/02-implicit-true-audit.md`, §5 "Exact computation," in the
repository named in the Data and Code Availability statement), so that every count below
can be re-derived from the raw corpus without this project's other code.

Three definitional choices govern the numbers.

**The flag predicate.** The pipeline predicate is `is_implicit_true_span()` — strict flag,
defined as `implicit_true and not due_to_null`. It was originally named `is_noisy_span()`;
the rename accompanies the reframing in §1 and §4, and the identifier is stated here because
the earlier name appears in this project's commit history. Raw `implicit_true` alone is
reported alongside the strict flag throughout, because the two differ by only 6 spans.

**Character mass is the right denominator, and it is not the span count.** Span-level
scoring on this benchmark is character-overlap micro-F1, so a long span counts for more than
a short one, and any bound expressed against a span metric must be computed over character
mass. `implicit_true` covers **13.5% of gold spans by count** and **14.56% of gold-span
character mass** — two different quantities. An earlier version of this work quoted 13.5%
as the character-mass figure; that was a conflation of the two, corrected here, and nothing
downstream depends on which is quoted because the pipeline flag is computed per token from
the spans directly. RAGTruth contains overlapping annotations (115 of 17,790 responses,
taking span offsets as half-open `[start, end)` intervals), so summing `end - start` over
raw spans double-counts overlaps: the §1/§4 figures are raw sums,
while the pipeline's per-token flag uses union-normalised regions and is the overlap-free
version — 14.49% of train positive tokens against 14.53% of raw flagged character mass.

**Flagging runs against raw spans, not normalised ones.** A token is flagged only if
*every* raw gold span covering it is flagged. A token backed by any unqualified annotation
keeps full weight. The audit enters the pipeline as an auxiliary column
(`is_implicit_true`, emitted alongside `labels`) and never as a label edit; labels, metrics,
and the test set are untouched at every stage of this paper.

### 3.2 Pre-registered ablation protocol

ACWS — Annotation-Confidence-Weighted Supervision — down-weights flagged positions in the
training loss by a factor λ, leaving labels and evaluation untouched:

```
L   = Σ_t (w_t · l_t) / clamp(Σ_t w_t, 1e-8)
l_t = per-token cross-entropy (reduction="none", ignore_index=-100)
w_t = 0   where label == -100 (context / special / padding)
    = λ   where the token is flagged
    = 1   elsewhere
```

Two boundary properties are unit-tested rather than asserted: **λ = 1.0 reduces exactly to
plain mean cross-entropy** (the default path does not route through the weighted branch, so
it stays bit-identical to the pre-ACWS model), and **λ = 0.0 is exactly loss-masking**.
λ = 0 was never run — §5 states why, §8 records it as a limitation.

What ACWS asks the model to exploit is deliberately partial. The model has no way to
verify, from context and response text alone, whether an unsupported claim is true —
RAGTruth's context is exactly what the claim is unsupported by, so there is no oracle for
correctness at inference time, and none is assumed here. What down-weighting can plausibly
teach is narrower: a correlate of the ungrounded-but-true subclass that survives in the
model's own parametric world-knowledge priors — the signal that makes some unsupported
claims *look* more plausible than others, independent of whether the context confirms them.
Whether that partial, imperfect signal is strong enough for loss down-weighting to extract
is exactly what the ablation tests. A null result does not mean the model cannot distinguish
true from false in general; it means this intervention did not surface whatever
prior-knowledge correlate exists, at the one setting tested.

ACWS could not be tested alone. A code audit had separately found this project's detector
deviating from LettuceDetect's published recipe on four axes at once — learning rate,
effective batch, epochs, and checkpoint-selection metric — so testing ACWS on top of a
divergent recipe would have confounded the two. Hence three arms, b and c differing in
exactly one thing:

| Arm | Description | lr | eff. batch | epochs | checkpoint metric | λ |
|---|---|---|---|---|---|---|
| **a** | the then-production model | 2e-5 | 16 | 8 | response F1 | 1.0 |
| **b** | faithful LettuceDetect-recipe replication | 1e-5 | 8 | 6 | **token F1** | 1.0 |
| **c** | arm b **+ ACWS** | 1e-5 | 8 | 6 | token F1 | **0.25** |

Scoring is stratified into five blocks joined back to the raw test slice. Two matter here:
`official`, char-overlap span and response P/R/F1 scored identically to training, and
**`clean_span`**, the same span metric with flagged intervals subtracted from *both* gold
and predictions — the metric the hypothesis is actually about. The emitted keys `clean_span`
and `noisy_char_mass_share` are frozen deliberately: they are the pre-registered on-disk
contract of the rule below, and renaming them to match §1's terminology would sever
traceability to the rule as written before the arms ran.

**The decision rule was written into the harness before any arm was run**, and printed as
PASS/FAIL, so the verdict was not eyeballed afterward. Adopt (c) over (b) iff all three
hold:

```
clean_span_f1(c) > clean_span_f1(b)
response_f1(c)   > response_f1(b)
official_span_recall(b) − official_span_recall(c) ≤ noisy_char_mass_share
```

The third clause is the honesty clause: ACWS is *expected* to cost some official span
recall, because it deliberately teaches the model not to predict over flagged text the
official metric still scores as gold, and the rule grants a budget for that cost.

**That tolerance was computed from the data at runtime, never hardcoded.** This is what
makes the rule binding rather than decorative. The budget is the flagged fraction of
official gold character mass, recomputed by the harness on every run, so it could not have
been widened after the fact to accommodate a larger recall loss than the rule intended. A
hardcoded constant would leave a reader no way to tell a pre-registered tolerance from a
post-hoc one.

Before any arm was trusted, the harness had to reproduce the *published* numbers of the
already-released model (arm a) from its prediction dump. It did — span-F1 **0.5114** against
a published 0.5113, response-F1 **0.7619** against 0.7619 — and only then were b and c
scored.

One thing about this protocol must be stated head-on rather than left for a reader to find.
**The hypothesis it pre-registers was premised on a misreading.** ACWS was designed under a
reading of `implicit_true` as marking, in the original decision record's words,
"annotator-acknowledged noise" — positives the labels had got wrong. §1 and §4 refute that
reading, and it is withdrawn. The premise was wrong; the experiment was not invalidated by
it, because the arms, the recipe, the decision rule, the reproduction gate, and every number
they produced are independent of why the hypothesis seemed worth testing. What changes is
what a null result *means* (§5). This is precisely what pre-registration is for: the record
shows what was believed and when, legibly enough to be corrected in public rather than
quietly re-motivated after the fact.

### 3.3 Post-hoc stratified re-scoring

The analysis in §6's case study is **exploratory, not pre-registered**, and is reported as
such. It was designed after the arms had run, in response to a question the audit raised
but did not answer.

The cohort is the test set's "Subtle-only" responses: those with at least one gold span
where *every* gold span's `label_type` begins with `Subtle` — no Evident spans present.
There are 77, all gold-hallucinated by construction. It splits disjointly into
**all-flagged** responses, where every gold span is `implicit_true` (n = 47), and
**authentic-Subtle** responses, where at least one gold span is not (n = 30). Arm b's
committed per-row prediction dump is joined back to the raw span metadata by the same
pattern the ablation harness uses — merged dataframe, filter to test, reset index,
positional index equals the dump's `row_index` — and the response-level miss rate (a
gold-hallucinated response predicted negative) is computed over each subset. The join is
self-checking: it reproduces the previously published 48.1% headline figure exactly, and the
47/77 split independently matches the audit's count. The script prints to stdout and emits
no metrics file, so this result resolves to a script plus a committed prediction dump rather
than to a JSON artifact; we say so rather than let a uniform citation style imply otherwise.

**The census/inference boundary is declared here, and it governs how §4 and §6 report every
number.** The RAGTruth test set is this paper's object of study, not a sample drawn from a
population. A claim that counts something *about that test set* — how many spans carry the
flag, how much character mass they cover, what a miss rate is over a closed, fully
enumerated cohort — is arithmetic, and takes **no confidence interval**, because nothing is
being estimated. The moment a sentence generalizes beyond this benchmark — asserting
something about how a model behaves on a *kind* of input — it changes register, and must
carry its uncertainty. §6 contains one claim of each type, side by side, and labels them.
Reporting a census figure with an interval would invite a reader to dismiss an exact result
as noise; reporting an inferential claim without one would assert what the data cannot
support. We do neither.

---

## 4. The conflated label class

This section establishes the paper's central finding: the structured conflation that §5 and
§6 each build on. Everything in it is a census over RAGTruth as released: exact counts over
a closed corpus, carrying no intervals, per §3.3.

Across the whole corpus, **1,928 of 14,289 gold spans carry `implicit_true`** — 13.5% by
count, and 110,171 of 756,461 characters, **14.56% of gold-span character mass**. Under the
strict flag (excluding the 6 spans that also carry `due_to_null`) the figures are 1,922 and
14.53%. That is the magnitude. Its *structure* is what makes it a conflated class rather
than a scattering.

**The flag is a severity qualifier, and the annotators' own intensity markers show it.**
RAGTruth's free-text `meta` comments open with a hallucination-intensity prefix. Flagged and
unflagged gold spans have close to inverted distributions:

| Span set | n | `LOW` | `HIGH` | none / other |
|---|---:|---:|---:|---:|
| `implicit_true = True` | 1,928 | **90.6%** | 0.8% | 8.6% |
| all other gold spans | 12,361 | 4.4% | **46.4%** | 49.2% |

The predicate is exact, not approximate, and worth stating precisely because a plausible
variant changes the numbers: a span counts as `LOW` (respectively `HIGH`) iff
`meta.upper().startswith("LOW")` (`"HIGH"`), with no leading-whitespace stripping. Stripping
before the check — an equally defensible reading — moves 17 flagged spans whose `meta`
string opens with a literal newline before the tag, and a comparable share of unflagged
spans, yielding 91.5% / 47.0% in place of 90.6% / 4.4%. Both variants preserve the
finding — flagged spans are overwhelmingly `LOW`, unflagged spans are not — but we report
the unstripped predicate because it is what §5's script and the pipeline's own
`is_implicit_true_span()` companion code actually run, not because it is the only
defensible choice.

Nine in ten flagged spans are marked low-intensity by the annotator who labelled them,
against fewer than one in twenty elsewhere. Read alongside the README's definition and the
comments themselves — *"these details are correct, however, not directly mentioned in the
passages"* — this is what rules out reading the field as a retraction. It marks a subclass
*within* the positive class, not an exception to it.

**The comparison above is confounded by span kind, and a narrower cut removes the
confound.** RAGTruth's four `label_type` values split into two kinds — Baseless Info
("introduction of new information") and Conflict — and the two use different annotator
vocabularies for severity. Conflict spans structurally almost never carry a `LOW`/`HIGH`
prefix at all: 99.7% of Evident Conflict and 100% of Subtle Conflict spans fall in
`none/other`, because conflict annotations are written as `"EVIDENT CONFLICT: ..."`, a
different comment convention that carries no severity gradient. Including Conflict spans in
the "all other gold spans" row therefore pads that row's `none/other` share for reasons that
have nothing to do with severity. Restricting the comparison to Baseless Info spans only —
where both `LOW` and `HIGH` are live options either way — removes that confound: of 1,921
flagged Baseless Info spans, **90.9% are `LOW`**; of 6,843 unflagged Baseless Info spans,
**83.7% are `HIGH`** and only **8.0% are `LOW`**. The inverted distribution survives the
narrower, unconfounded cut — if anything the flagged/unflagged gap on `LOW` alone (90.9%
against 8.0%) is a cleaner statement of the same severity split than the original
all-gold-spans comparison, which mixed in a span kind that could not have scored `LOW`
regardless of true severity.

**The subclass is concentrated in one span type, not spread across four.**

| `label_type` | spans | flagged | share |
|---|---:|---:|---:|
| Evident Baseless Info | 6,237 | 60 | 0.96% |
| Evident Conflict | 5,324 | 6 | 0.11% |
| **Subtle Baseless Info** | **2,527** | **1,861** | **73.64%** |
| Subtle Conflict | 201 | 1 | 0.50% |

Nearly three quarters of "Subtle Baseless Info" is ungrounded-but-true content. That
category is therefore not a homogeneous detection target: it is predominantly
ungrounded-but-true with a minority of ungrounded-and-false cases mixed in. Any figure reported
over it — and Subtle/Evident is the benchmark's own severity axis, carried in every gold
span's `label_type` — is a figure over a mixture whose composition is nowhere stated.

**It is also concentrated by task type**, which matters because RAGTruth results are
routinely reported per task:

| Split | Task type | flagged chars / total | share |
|---|---|---:|---:|
| train | QA | 63,280 / 305,474 | **20.72%** |
| train | Data2txt | 30,008 / 262,339 | 11.44% |
| train | Summary | 8,732 / 102,771 | 8.50% |
| test | QA | 5,557 / 31,622 | **17.57%** |
| test | Data2txt | 1,855 / 36,264 | 5.12% |
| test | Summary | 739 / 17,991 | 4.11% |

On the test split, ungrounded-but-true content is more than four times as dense in QA gold
mass as in Summary gold mass (17.57% against 4.11%). A per-task comparison on RAGTruth is
thus not comparing like with like: the QA column is substantially more about the
ungrounded-but-true subclass than the Summary column is. This uneven distribution is also the reason filtering
flagged spans out of the *test* set was rejected as an option — it would change the
benchmark by task type unevenly and break comparability with every published RAGTruth
number. Nothing in this paper alters the test set.

**The contrasting case is `due_to_null`, and it is instructive.** RAGTruth's other span-level
metadata field marks spans where the model invented a value for a null field: **all 1,642
sit in Data2txt, 98.3% of them Evident Baseless Info**. Those are ungrounded-and-false
hallucinations by any reading — high-severity positives that must keep full weight, which is
why this project's flag predicate excludes them even where `implicit_true` is also set. The
two fields are not interchangeable metadata. One marks the ungrounded-but-true subclass and
one marks a severe one, and only the second has an authors'-provided scoring affordance
(§2.4).

One split-level difference is worth stating plainly and then leaving alone. The flagged
subclass is **more concentrated in train (14.49% of positive tokens) than in test
(8.95%)**. For a paper whose sharper claim (§6) is about evaluation, that cuts against the
tidy story: the phenomenon is denser exactly where §5's intervention produced no measurable
effect. We report it as a methodological fact and decline to recruit it as support for
either §5 or §6, in either direction. Sensitivity to a subclass at training time is not
required to scale with that subclass's concentration, and we have no measurement that would
establish it does. §5's null is scoped to the setting tested, and this table does not widen
it.

---

## 5. Attempting to exploit the conflation at training time: the ACWS null

The training-side question is whether a model can be taught to discount the
ungrounded-but-true subclass by the simplest available mechanism — down-weighting those
positions in the loss. This does not ask the model to verify truth it has no way to check;
it asks whether a partial, correlational plausibility signal — grounded in the model's own
parametric priors, not in ground-truth access — can be exploited well enough by that
mechanism to move a downstream metric. Under the pre-registered rule of §3.2, at λ = 0.25,
it could not.

### 5.1 Arm b: the recipe fix, isolated

Arm b is the LettuceDetect-recipe replication with no ACWS (λ = 1.0), and it exists to give
arm c a clean baseline. It is also, on its own, the larger effect in this ablation. Against
arm a — the same architecture and data under this project's earlier recipe — arm b reaches
**span-F1 0.5321** against 0.5113, a gain of 2.1 points, with response-F1 **0.7631**,
response precision 0.8359 against 0.7873, and a false-positive rate on faithful responses of
7.4% against 10.7%. ADR-020 attributes that gain to checkpoint selection on span-level
rather than response-level F1, and §7.3 gives the structural reason it is the plausible
cause. The attribution is an interpretation, not an isolated measurement: arm b changed four
recipe axes at once — learning rate, effective batch, epochs, and selection metric (§3.2) —
so the ablation does not partition the 2.1 points among them. Nothing in that result
involves `implicit_true`, and we do not present it as
part of the paper's argument — it is the control arm, reported so that the comparison arm c
is judged against is visible.

### 5.2 Arm c: ACWS at λ = 0.25

Arm c's numbers are reported from ADR-020: a committed, timestamped decision record written
under this study's pre-registered rule at the time the arm was run. No prediction dump or
metrics file was retained for the run, although the tooling to produce one existed and was
used for its sibling arm — the omission is ours, not a limitation of the setup. Alone among
the experiments reported here, arm c's figures cannot be independently re-derived without
retraining, and we mark them as such wherever they appear.

Arm c differs from arm b in exactly one parameter, `--implicit_true_weight` 1.0 → 0.25. The
pre-registered rule's three clauses evaluate as follows:

| Clause | Arm b | Arm c *(ADR-020, no artifact)* | Verdict |
|---|---:|---:|---|
| `clean_span_f1(c) > clean_span_f1(b)` | 0.5307 | 0.5262 | **FAIL** (−0.0045) |
| `response_f1(c) > response_f1(b)` | 0.7631 | 0.7633 | pass (+0.0002) |
| official span-recall loss ≤ flagged char-mass share | — | — | not reached |

The first clause fails outright, so the rule rejects arm c and the remaining clauses do not
matter. Clean-span F1 — the metric the hypothesis is specifically about, with flagged
intervals removed from both gold and predictions — moved in the wrong direction. The
response-level gain of 0.0002 is not a countervailing signal at this scale.

### 5.3 Reading the null

The immediate question is whether −0.0045 is a result or a fluctuation. §5.2's own
comparison anchored this against the base recipe's seed-to-seed spread on *official* span
F1 — a mismatched metric, since the b→c delta itself is on *clean* span F1. These same
three seeded arm-b runs are also the ModernBERT-base arm of Appendix A's scaling
comparison — the variance anchor used here and the base arm reported there are the
identical three training runs, read for two different purposes.

A matched-metric anchor is now available, and it replaces the mismatched one. Arm b's
clean-span F1 across its three real seeds (42, 123, 456) is **0.5307 / 0.5281 / 0.5336** —
a spread of **0.0055**, computed directly by `scripts/ablation_report.py` from each seed's
committed prediction dump. Two of arm c's three seeds also have real, committed prediction
dumps now (`results/arm_c_seed42_preds.json`, `results/arm_c_seed123_preds.json`, recovered
via Hub inference after the original run's metrics-save step was lost); the third
(seed 456) does not, and is out of scope here per the deferred RunPod pass. Scored on the
same matched metric, clean-span F1 for arm c is **0.5171** (seed 42) and **0.5209**
(seed 123), giving matched b→c deltas of **−0.0136** and **−0.0072** — both negative, both
several times larger than the 0.0055 seed-only spread, and consistent in direction with
ADR-020's original single-seed finding. *Source for every figure in this paragraph:
`results/clean_span_seed_variance.json`, written by
`scripts/ablation_report.py --arm b42=... --arm b123=... --arm b456=... --arm c42=... --arm c123=...`
against the five prediction dumps named above and `results/base_seed456_preds.json`.*

One discrepancy is worth disclosing rather than absorbing. The recovered seed-42 clean-span
F1 (0.5171) does not match ADR-020's originally reported figure for the same nominal
setting (0.5262) — a 0.0091 gap, larger than the 0.0055 seed spread it is being compared
against. Response F1 is close between the two (0.7623 recovered against 0.7633 originally
reported), so the discrepancy is specific to the span-level metric rather than a wholesale
mismatch, and we do not know its cause. §5.2's numbers remain sourced to ADR-020, unchanged;
this is a separate, independently re-derivable computation from the recovered prediction
dumps, and the two are not claimed to be the same measurement. Reconciling the gap is out of
scope here and deferred alongside the rest of arm c's re-verification.

With that caveat stated, the matched-metric comparison strengthens rather than weakens
§5.2's conclusion: on the metric the hypothesis is actually about, arm c's clean-span F1 was
lower than arm b's at both seeds where a direct comparison is now possible, by margins that
exceed the base recipe's own seed noise. This is a post-hoc robustness check on the null's
*direction*, not a re-run of the pre-registered decision rule — that rule's verdict (§5.2)
still rests on the single seed it was evaluated against, and the four constraints below
describe that rule's scope, not this supplementary check's.

Five constraints bound what this null can be read to mean, and none of them is
recoverable from the data we have.

**One λ, one seed, one architecture.** The result is λ = 0.25, seed 42, ModernBERT-base. No
sweep followed, and §8 records this as a limitation rather than an oversight: the result was
not directionally encouraging, so a sweep would have been a search for a value that happened
to land, with the tuning risk that implies.

**The intervention is small.** Flagged tokens are roughly 0.7–0.8% of supervised tokens, and
λ = 0.25 shifts on the order of **0.2% of total loss mass**. This is per-token gradient
scaling on a thin slice of the objective, not a re-specification of it.

**Checkpoint selection is on the wrong metric for what the hypothesis asks.** Arms b and c
both select checkpoints on token-level F1 (§3.2), which scores flagged tokens the same as
any other positive; the ACWS hypothesis is specifically about clean-span F1, which does not.
Down-weighting flagged tokens in the loss shifts the token-F1 landscape across training
steps, so checkpoint selection on token F1 may discard exactly the checkpoints where ACWS
would have helped on clean-span F1 — an unresolved confound between what arm c was selected
for and what it was evaluated on, stated here as acknowledged and not something this paper
attempts to fix.

**The null cannot distinguish two explanations.** It is equally consistent with the model
being insensitive to this distinction and with the perturbation falling below the resolution
of the training process. Separating them requires λ = 0 — exact loss-masking, the largest
version of this intervention — which was never run and is not planned. **The confound
therefore stays stated and unresolved.** We do not claim the model is robust to the
distinction; we claim only that this intervention, at this magnitude, did not move the
metric it targeted.

**Nothing here licenses a general robustness claim.** In particular, this is not evidence
about a model's tolerance for mislabelled training data, and cannot be — §4 establishes that
the flagged subclass is not mislabelled. What was tested is whether a real, low-severity
error subclass can be discounted by simple loss reweighting. At the one setting tested, it
could not.

The defensible claim, stated exactly: **at λ = 0.25, one seed, one architecture,
down-weighting the ungrounded-but-true subclass produced no measurable improvement in
clean-span F1.** A reviewer will reasonably find that thin, and the honest response is that
the chapter's contribution is not the result but the protocol around it — a decision rule
fixed in code before the arms ran, a harness validated against published numbers first, and
a stated inability to distinguish two explanations that a less disciplined write-up could
have collapsed into one. That, and not the −0.0045, is what §7 carries forward.

---

## 6. Quantifying the conflation's cost at evaluation time

§4 established that the conflation is real and structured. This section quantifies what it
costs under RAGTruth's official scoring — a direct, model-independent consequence of that
structure, not a second, independent result. The reason is arithmetic rather than empirical:
RAGTruth's official scoring counts the ungrounded-but-true subclass as gold, so a detector
that declines to flag it is scored as having missed it.

### 6.1 The span-level bound

Span-level P/R/F1 under character overlap is the metric token-level detectors on this
benchmark are principally scored on, so it is where the bound bites hardest and where we
state it first. Flagged spans cover 8,151 of the 85,877 characters of gold span mass in the
test split — 9.49%. Therefore:

> **Under RAGTruth's official scoring, any prediction set that excludes the
> annotator-flagged low-intensity subclass from its positives is capped at 90.5%
> char-overlap span recall against gold — a property of the scoring protocol, computable
> from the released labels alone, independent of whether any detector can or should make
> that exclusion.**

The figure is (85,877 − 8,151) / 85,877 = 90.5%. It is a census over a closed, fully
enumerated test split, computed directly from the released corpus, and it carries no
confidence interval because nothing is being estimated (§3.3). It is not an empirical
finding about any particular system; it is a property of the benchmark that holds before
any model is trained.

### 6.2 The response-level bound

The same argument applies at coarser granularity, and we report it as a corroborating
instance rather than as a second result. A response counts as hallucinated if it carries any
gold span; 49 of the 943 hallucinated test responses consist *entirely* of flagged spans.
Therefore, under the same protocol-level argument, **any prediction set excluding the
low-intensity subclass is capped at (943 − 49) / 943 = 94.8% response-level recall against
gold.** The response-level bound is the weaker of the two — a response survives if even one
of its spans is ungrounded-and-false — which is precisely why the span-level figure leads.

### 6.3 What the bound is and is not

The statement above is a bound on **achieving two goals at once**. It is not a ceiling on
recall in isolation, and any paraphrase that drops the conditional is false: a system that
flags every token still reaches 100% recall on this benchmark, trivially and uselessly. What
cannot be done is to reach past 90.5% span recall *while* declining to flag
ungrounded-but-true content. Every statement of the bound in this paper carries that
conjunction, and we have written it out in full each time rather than abbreviate it after
first use.

Two consequences follow. First, the bound is a property of the scoring function's
relationship to the gold labels, not of any system that might be scored by it. Stated in
terms of a hypothetical detector's *objective*, it is indifferent to model class, size, or
training data, and a frontier-scale system is bound by it exactly as tightly as a 150M
encoder.

Second, and this is the sentence in this paper with the widest reach: recall figures on
this benchmark are not comparable across detectors with different implicit objectives, and
that applies to every system in §2.2's landscape table, not only the instrument used here.
RAGTruth's official scoring does not ask what a system was built to catch; it asks only
whether a flagged span overlaps a gold span, of whatever severity. A system tuned to
surface ungrounded-and-false errors and a system tuned to maximize official recall are
scored on the same axis while optimizing different things, and nothing in a published F1
distinguishes them — the metric cannot tell a low-severity span missed by design from one
missed by failure. This is not a claim that any figure in §2.2 is wrong; each is a correct
report of what its official metric measures. It is a claim about what that metric cannot
tell a reader: a higher published F1 does not imply a system catches more
ungrounded-and-false content specifically, because the aggregate score is silent on which
subclass drove the difference. Two systems separated by several F1 points could differ
almost entirely in how much ungrounded-but-true content each flags, with their handling of
the ungrounded-and-false subclass nearly identical — or the reverse — and no published number
distinguishes the two cases. Resolving this for any specific pair of published systems
would require re-scoring their released predictions against a stratified metric, which
this paper does not attempt for the literature as a whole; that is a distinct exercise from
§2.3's narrower concern about resolving figures to primary sources. We state this
consequence as this paper's clearest implication for how the field should read its own
leaderboard, not as an audit of that leaderboard.

### 6.4 Subordinate case study: the Subtle cohort

The bound is aggregate. Where the conflation actually lands in a specific detector's error
profile is a separate, narrower question, and we answer it for one model — arm b — as a
worked example. This subsection is scoped to that model and is not a second headline.

§4 showed the flagged subclass concentrates in Subtle Baseless Info. Arm b's weakest
reported cohort is Subtle-only responses, at a 48.1% miss rate. The natural inference is
that part of that figure is not a real capability gap but the model declining to flag
content a detector limited to ungrounded-and-false content would arguably be right to leave
alone. We tested it, and it does not hold.

| Cohort | n | missed | miss rate |
|---|---:|---:|---:|
| Full Subtle-only cohort (as reported) | 77 | 37 | **48.1%** |
| Authentic-Subtle only (all-flagged excluded) | 30 | 16 | **53.3%** |
| All-flagged only (reference) | 47 | 21 | 44.7% |

**Excluding the flagged responses raises the miss rate, from 48.1% to 53.3%.** This is a
census claim — exact counts over a closed 77-response cohort — and it carries no interval.
The direction is the point: the flagged subclass was *diluting* the headline figure toward a
better-looking number, not inflating it. Arm b's real difficulty with authentic Subtle
hallucinations is understated by the metric as reported, not overstated.

The mechanism that would explain this lives in a different register and is **not
established**. Arm b catches 26 of 47 flagged Subtle responses (55.3%) against 14 of 30
authentic ones (46.7%) — an 8.7-point gap, on which Fisher's exact test, two-sided, gives
**p = 0.491**. That is not distinguishable from chance. We report it because omitting it
would leave the census result unmotivated, and we label it as a candidate mechanism for
future work rather than a finding. The census claim is the result; the mechanism is a guess
at why, and keeping them in separate registers is what stops the guess's weak evidence from
dragging down the exact result.

Two readings this data does not support, stated explicitly because both are the natural
thing to conclude and both are wrong. It does **not** show that the Subtle weakness is a
labeling artifact inflating the miss rate — the measured direction is the opposite. And it
does **not** establish that the model is better at ungrounded-but-true cases than at
authentic ones; that is true of this cohort as counted, but the p-value above is exactly the
reason it cannot be asserted as a property of the model.

The case study's reach is genuinely limited, and the limitation is worth naming rather than
leaving for a reviewer. No published system reports a Subtle-stratified miss rate, so the
distortion demonstrated here lands on a metric only this project reports. That is why the
case study is subordinate: it illustrates where the conflation concentrates inside one
detector's errors, while §6.1's bound is the claim with reach beyond this paper, because it
constrains every system scored on this benchmark regardless of what any of them chooses to
report.

---

## 7. Discussion

### 7.1 The training-time null and the evaluation-time bound are independent claims

A reader could reasonably wonder whether §5's null and §6's bound are in tension: if the
conflated subclass is large enough to cap recall at 90.5% (§6), why did down-weighting it in
training move nothing (§5)? They are not in tension, and neither result bears on the
other's validity — the reason is that the two stages weight the same tokens by entirely
different quantities.

At training time, the flagged subclass enters through the loss. It is roughly 0.7–0.8% of
supervised tokens, and down-weighting it at λ = 0.25 moves on the order of 0.2% of total
loss mass — a perturbation that competes with every other gradient signal in the objective
and is diluted across the whole of training. At evaluation time, the same subclass enters
through the scoring function, where it is 9.49% of test gold character mass and is diluted
by nothing at all. Each flagged character is counted once, directly, against any detector
that declines to flag it.

A quantity can be too small to steer a training process and large enough to bend a metric,
and here it demonstrably is both. The stronger reason to treat the two as independent is
that §6's bound needs no model: it is arithmetic over the released corpus and holds before
any system is trained, so it cannot be undermined by a training-time null. Symmetrically,
§5's null is not weakened by §6's bound: a model's inability to exploit a training signal
says nothing about the scoring protocol's arithmetic.

### 7.2 The arm-b complication

One result cuts against the tidy version of this story, and we include it deliberately.
Adopting arm b's corrected recipe — the change §5.1 reports as a 2.1-point span-F1 gain —
made the detector *worse* on exactly the cases this paper cares most about. Subtle
hallucination miss rate rose from 40.3% to 48.1%, Evident from 27.0% to 30.6%, and the
FP:FN ratio shifted from 0.76 to 0.46, meaning the model bought its precision by
under-flagging rather than by flagging more accurately.

This is awkward, and it strengthens the argument rather than undercutting it. If Subtle
performance moved by nearly eight points in response to a change that has nothing to do with
the annotation — a checkpoint-selection metric, a learning rate, a batch size — then Subtle
detection is fragile to training decisions that are not about the phenomenon at all. Label-
class conflation does not explain arm b's Subtle weakness, and this paper does not claim it
does. Subtle detection is a genuinely unresolved problem, and §6.4's finding is that the
metric reporting it is *also* distorted, not that the distortion is the whole of the
difficulty. A reader who came away thinking the conflation accounts for the Subtle gap would
have read us wrongly, and this complication is the clearest evidence against that reading.

### 7.3 A transferable lesson about checkpoint selection

The mechanism behind arm b's gain generalizes past this benchmark, and we record it here as
a Discussion observation rather than as a contribution of the paper.

The earlier recipe selected checkpoints on response-level F1 — the deployment-relevant
metric, and the defensible-looking choice. Changing that selection, together with three
other recipe axes, recovered 2.1 span-F1 points (§5.1), and selection is the axis with a
structural reason to have caused it: response-level F1 asks only whether
*any* token in a response was flagged, so it cannot distinguish a tight, well-placed span
from a sloppy one that happens to overlap the gold region. Two checkpoints that are
identical at response level can differ substantially at span level, and selecting on the
coarse metric picks between them arbitrarily.

The general form: **when a system has both a coarse and a fine success criterion, selecting
on the coarse one does not merely fail to optimize the fine one — it discards the
information needed to choose between candidates that tie at the coarse level.** Any pipeline
with nested granularities of correctness can fall into this, and the failure is invisible in
the metric being selected on. We flag it because nothing in the coarse metric's behaviour
signalled the loss — the suppressed checkpoints scored normally on the metric being used to
choose them — and because the cost was recoverable once the fine metric was consulted at
selection time.

### 7.4 Implication for benchmark reporting

The narrow implication of §6 is that a RAGTruth recall figure does not, on its own, tell a
reader what kind of error the detector is catching. A detector that reports 85% span recall
may have caught predominantly ungrounded-and-false content, or predominantly
ungrounded-but-true content, and the number is identical either way.

Stratified reporting would resolve this: publish the metric separately over
ungrounded-and-false and ungrounded-but-true gold, alongside the aggregate. The benchmark
already ships the field this requires, and its authors already established the pattern for
the neighbouring field (§2.4). We offer this as an implication of what we measured rather
than as a recommendation we have standing to make — one audit and one null ablation do not
establish what a benchmark's reporting conventions should be, and we have not evaluated what
stratified reporting would cost in practice or whether it would change any published
ranking. What we can say is narrower and, we think, sufficient: the information needed to
report that way already exists in the data, unused.

The following table makes the proposal concrete with a number already on hand rather than a
hypothetical one. Both columns are arm b (§5.1), computed over the same test-split
predictions; the second column subtracts the flagged (ungrounded-but-true) intervals from
both gold and predictions before rescoring, exactly as §5.2 uses to evaluate the ACWS
hypothesis.

| Metric | Aggregate (official) | Flagged-excluded (`clean_span`) |
|---|---:|---:|
| Span-F1 | 0.5321 | 0.5307 |

*Source: aggregate column from `results/arm_b_metrics.json` (`test.span_char_level.f1`);
flagged-excluded column from the `clean_span` block of `scripts/ablation_report.py`,
reported in §5.2 and recorded in `docs/EXPERIMENT_LEDGER.md` (E6) and ADR-020. No new computation
was run to produce either figure.*

The two columns differ by 0.0014 — small at this scale, and in this instance the direction
happens to run the other way from the concern that opened §6 (excluding flagged spans here
costs recall on a subclass more entwined with the true positives than it inflates the
headline). The magnitude of the gap is not the point and we do not read it as evidence either
way about arm b specifically. The point is narrower: the two columns are both already
computable from released annotations, neither requires retraining or additional inference,
and reporting only the first — as this benchmark's convention currently does — discards
information that was sitting in the same prediction dump used to produce it.

We do not have a response-level counterpart to sit beside this table. The aggregate
response-F1 (0.7631, same source) has a published analogue, but `ablation_report.py`'s
`clean_span` block recomputes only the char-span metric with flagged intervals removed — it
was never extended to derive a parallel flagged-excluded response-level figure, and no other
artifact in this project contains one. Producing it would mean computing a new number rather
than surfacing an existing one, so we leave the response-level half of this demonstration for
future work and note the gap rather than approximate it.

---

## 8. Limitations

**The training-side null (§5).**

- **One setting only.** λ = 0.25, seed 42, ModernBERT-base. No sweep, no replication.
- **Arm c has no artifact.** No prediction dump or metrics file was retained; its figures
  resolve to ADR-020 and cannot be re-derived without retraining the arm. Every other
  experiment here resolves to a metrics file.
- **λ = 0 was not run.** This was a decision, not an oversight — the result was not
  directionally encouraging, so a sweep was declined. The consequence is that the confound
  below stays open.
- **The intervention may be below the resolution of training.** Flagged tokens are ~0.7–0.8%
  of supervised tokens and λ = 0.25 shifts ~0.2% of loss mass, so the null cannot
  distinguish model insensitivity from an intervention too small to measure.

**The arm-b comparison (§5.1, §7.3).**

- **Four recipe axes changed at once** — learning rate, effective batch, epochs, and
  selection metric. The 2.1-point span-F1 gain is not partitioned among them. Attributing
  it to checkpoint selection rests on a structural argument, not on an isolating arm.

**The evaluation-side case study (§6.4).**

- **n = 30 in the authentic-Subtle cohort.** A small denominator limits what one cohort can
  illustrate. It does not make the count inexact: no confidence interval is reported,
  deliberately, and an earlier ±18pp interval was withdrawn as a misapplication of
  inferential statistics to a census. That withdrawal is not to be reversed.
- **The mechanism is unestablished** (Fisher's exact p = 0.491) and is reported only as a
  candidate explanation.
- **Reach.** No published system reports a Subtle-stratified miss rate, so this case study
  lands on a metric only this project reports.

**Related work (§2).**

- **Luna and RAG-HAT are unverified.** Neither publishes code or weights, so their handling
  of `implicit_true` could not be checked. The novelty claim is scoped to the two systems
  whose preprocessing is publicly inspectable.

**The scaling replication (Appendix A) and the variance anchor.**

- **E10's epoch cap is not uniform.** Seed 42 ran at 6 epochs, seeds 123 and 456 at 4; seed
  123 selected its final epoch and may not have converged. The sweep is "seed, with a
  differing epoch cap on one arm", not "seed only".
- **Large seed 123's configuration is unrecorded** and its predictions were never dumped, so
  stratified re-analysis covers two of three large seeds.
- **The seed-42 large checkpoint is not published to the Hub**, so the reporting-model claim
  is not yet externally verifiable.

**Self-disclosed correction.**

- **Earlier versions of this work quoted 13.5% as a character-mass share.** It is a span
  *count* share; the character-mass figure is 14.56%. Corrected in §3.1.

---

## 9. Conclusion

RAGTruth's positive class aggregates two materially different error types, and the benchmark
ships the field that separates them. §4 showed this is not a scattering of annotator
disagreement: the conflation concentrates in one span type — nearly three quarters of spans
typed "Subtle Baseless Info" — and is unevenly distributed by task, the structure of a real,
annotator-recognized subclass. That structured finding is what the rest of the paper builds
on.

Its consequence at evaluation time is exact and model-independent: flagged spans are 8,151
of the 85,877 characters of test gold span mass, so **under RAGTruth's official scoring, any
prediction set that omits the low-intensity subclass is capped at 90.5% char-overlap span
recall against gold** — a property of the scoring protocol, not a claim about what any
detector does, and a direct arithmetic consequence of §4 rather than an independent second
result. At the coarser response level, 49 of 943 hallucinated test responses are entirely
flagged, giving the corroborating 94.8% figure. Both are census counts over a closed, fully
enumerated test split, computed from the released corpus and carrying no confidence
interval, because nothing is being estimated. Neither depends on any model, and both hold
before a detector is trained.

We also tested whether the same conflation could be exploited at training time. Down-weighting
the ungrounded-but-true subclass in the loss at λ = 0.25, under a decision rule fixed in
code before the arms ran, produced no measurable improvement in clean-span F1 — at one λ,
one seed, one architecture, with an intervention touching roughly 0.2% of loss mass and a
confound we could not resolve because λ = 0 was never run. This is a negative result,
narrower and weaker than the finding it sits beside, and we claim it as such rather than as
a second leg.

The evaluation-time bound and the training-time null are not in tension, for the reason
§7.1 gives: the same quantity that is too small to steer a training process (0.2% of loss
mass) is large enough to bend a scoring function (9.49% of test character mass; 8.95% of
test positive tokens is the related but distinct train/test density figure from §4), so a
real finding and a real null coexist without contradiction. They are two different
questions asked of the same structured fact, not one result measured twice.

We close with an implication we have not demonstrated and offer as speculation rather than
as a lesson. `implicit_true` exists because RAGTruth's annotators recorded not only *that* a
span was ungrounded but *how severe* that was — and having recorded it, the field
went unused by the systems trained on the data and unreported by the papers that scored on
it. Annotation-confidence and annotation-severity metadata may be a generally under-used
signal in benchmark construction: cheap to collect while annotators are already reading the
text, and expensive to reconstruct afterward. Whether that generalizes past this one corpus
is not something one audit and one null ablation can establish, and we do not claim it.

---

## Data and Code Availability

All code, configuration, and result artifacts for this project are in a public
repository: https://github.com/hugoomez/rag-hallucination-detector.

Every quantitative result reported in this paper resolves to a committed artifact in that
repository — a metrics file, a prediction dump, or a decision record — with **one named
exception**. Arm c (§5.2, ACWS at λ = 0.25) resolves only to ADR-020, a committed,
timestamped decision record written under this study's pre-registered rule at the time
the arm was run; no prediction dump or metrics file was retained, and its figures cannot
be independently re-derived without retraining. This is the paper's one open
reproducibility gap. It is disclosed wherever arm c's numbers appear (§5.2, §8) and not
only here; regenerating the missing artifact by retraining under the same recipe is
planned as a follow-up, tracked internally as Fase D, and is independent of every other
result in this paper.

Arm a's metrics file, arm b's metrics and prediction dump, the E10 three-seed variance
anchor, and Appendix A's scaling comparison for large seeds 42 and 456 are all committed
under `results/`. Arm a has no committed prediction dump — the harness used one to
reproduce arm a's published numbers at run time (§3.2), but it was not retained — so
arm a's per-row predictions are not independently re-derivable from this repository.
Appendix A's own two partial gaps — large seed 123's predictions were not dumped and its
checkpoint was not published to the Hub — are disclosed in §8 and do not affect any claim
in the paper's main argument (§4–§7), which Appendix A is explicitly severable from.

The annotation audit underlying §3.1 and §4 is a direct, standard-library computation over
the released RAGTruth corpus, reproduced in full as Supplement S1
(`docs/research/02-implicit-true-audit.md`, §5) rather than paraphrased, so every count in
this paper can be re-derived from the raw corpus independently of this project's other
code.

## References

Belyi, M., Friel, R., Shao, S., & Sanyal, A. (2024/2025). Luna: An evaluation foundation
model to catch language model hallucinations with high accuracy and low cost / Luna: A
lightweight evaluation model to catch language model hallucinations with high accuracy and
low cost. arXiv:2406.00975 (preprint) / *Proceedings of the 31st International Conference
on Computational Linguistics: Industry Track* (COLING 2025, camera-ready).

Hu, X., Ru, D., Qiu, L., Guo, Q., Zhang, T., Xu, Y., Luo, Y., Liu, P., Zhang, Y., & Zhang,
Z. (2024). RefChecker: Reference-based fine-grained hallucination checker and benchmark
for large language models. arXiv:2405.14486.

Kovács, Á., & Recski, G. (2025). LettuceDetect: A hallucination detection framework for
RAG applications. arXiv:2502.17125.

Maynez, J., Narayan, S., Bohnet, B., & McDonald, R. (2020). On faithfulness and factuality
in abstractive summarization. *Proceedings of the 58th Annual Meeting of the Association
for Computational Linguistics*, 1906–1919. https://doi.org/10.18653/v1/2020.acl-main.173

Niu, C., Wu, Y., Zhu, J., Xu, S., Shum, K., Zhong, R., Song, J., & Zhang, T. (2024).
RAGTruth: A hallucination corpus for developing trustworthy retrieval-augmented language
models. *Proceedings of the 62nd Annual Meeting of the Association for Computational
Linguistics (Volume 1: Long Papers)*, 10862–10878. arXiv:2401.00396.

Song, J., Wang, X., Zhu, J., Wu, Y., Cheng, X., Zhong, R., & Niu, C. (2024). RAG-HAT: A
hallucination-aware tuning pipeline for LLM in retrieval-augmented generation.
*Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing:
Industry Track*, 1548–1558. https://doi.org/10.18653/v1/2024.emnlp-industry.113

Tang, L., Laban, P., & Durrett, G. (2024). MiniCheck: Efficient fact-checking of LLMs on
grounding documents. *Proceedings of the 2024 Conference on Empirical Methods in Natural
Language Processing*. https://doi.org/10.18653/v1/2024.emnlp-main.499

Warner, B., Chaffin, A., Clavié, B., Weller, O., Hallström, O., Taghadouini, S., Gallagher,
A., Biswas, R., Ladhak, F., Aarsen, T., Cooper, N., Adams, G., Howard, J., & Poli, I.
(2024/2025). Smarter, better, faster, longer: A modern bidirectional encoder for fast,
memory efficient, and long context finetuning and inference. arXiv:2412.13663. Also:
*Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics*
(2025).

Zha, Y., Yang, Y., Li, R., & Hu, Z. (2023). AlignScore: Evaluating factual consistency with
a unified alignment function. *Proceedings of the 61st Annual Meeting of the Association
for Computational Linguistics (Volume 1: Long Papers)*.
https://doi.org/10.18653/v1/2023.acl-long.634

---

## Appendix A — Resource-constrained scaling replication

**This appendix is methodologically separate from the paper's argument and is not part of
it.** It reports a backbone-scaling comparison that shares infrastructure with the
experiments above but tests nothing about `implicit_true`, contributes no evidence for or
against §4, §5, or §6, and would be removed without affecting any claim the paper
makes. It is included because the data exists and is worth reporting on its own terms, not
because it supports anything else here.

**What was run.** A three-seed comparison between ModernBERT-base and ModernBERT-large
under arm b's recipe, intended as matched: the same seed should mean the same
weight-initialization draw and data shuffle, so the backbone is the only difference. That
intent is verifiable for two of the three seeds. Seeds 42 and 456 have full, recorded
hyperparameters on both backbones and match on every axis (learning rate, epochs,
checkpoint-selection metric, `implicit_true_weight`) other than backbone size — **2
verified matched seeds**. Seed 123's large-backbone run was produced by the existing-model
inference path rather than a training run's own reporting; its metrics file records only
`{"seed": 123}` under hyperparameters, so its recipe cannot be confirmed to match the base
arm — **1 seed of unrecorded training configuration**, included in the counts below because
it was run and its output is real, but not verified as matched. Deltas are computed per
seed and then aggregated. The base arm of this scaling comparison is the same three
training runs used as §5.3's variance anchor for the training-side null — not a separate
re-run, the identical runs read for a different purpose here.

| Metric | Base mean | Large mean | Δ mean | Δ range | Direction | Published Δ (Kovács & Recski) |
|---|---:|---:|---:|---|---|---:|
| Response F1 | 0.7637 | 0.7948 | **+0.0311** | +0.0292 … +0.0323 | 3/3 seeds\* | +0.0315 |
| Span F1 (char-overlap) | 0.5325 | 0.5733 | **+0.0408** | +0.0379 … +0.0446 | 3/3 seeds\* | +0.0349 |

*\*2 of 3 seeds (42, 456) are verified recipe-matched across backbones; seed 123's
large-backbone run is of unrecorded training configuration (see "What was run," above, and
the caveats below). "3/3" is the observed direction across all three runs as executed, not
a claim that all three are verified matched.*

At response level the observed gap is close to LettuceDetect's published base→large gap
(+0.0311 against +0.0315), so "replicates" is a fair description of that row. At span level
the observed gap is **larger** than the published one (+0.0408 against +0.0349); the row
does not merely reproduce the published difference, and we report the discrepancy rather
than round it toward agreement. We do not know its cause and did not investigate it. Nothing
in this comparison is a claim about either system's quality as a method, and none of these
figures is presented as a record.

**No significance is claimed, and none is available at n = 3.** The aggregation script emits
raw per-seed values, means, min/max ranges, and per-seed paired deltas by construction, and
no p-values. The reportable statement is that the direction was consistent across all 3
seeds run on both metrics, with the means and ranges above — on a base of 2 verified
matched seeds (42, 456) plus one seed (123) of unrecorded training configuration.

**The gain is recall-led.** Response recall improves in 3/3 seeds (mean +0.0516), while
response precision improves in only 2/3 (mean +0.0055, with one seed at −0.0028) and
char-span precision likewise in 2/3 (mean +0.0217, one seed at −0.0207). Only recall is
consistently better. The largest per-task movement is on Summary (+0.0931 mean F1, 3/3
seeds), against QA (+0.0375) and Data2txt (+0.0146).

**What this is worth reporting for.** LettuceDetect published its base→large scaling gap as
a single point estimate with no variance information. This appendix supplies three matched
seeds with means, ranges, and per-seed paired deltas for the same comparison, under
materially more constrained compute: a single consumer GPU, and 4 epochs against their 6.
The contribution, such as it is, is evidence about what a published scaling result looks
like when re-run under constraint and with variance reported — not a statement about which
system is better, which this design could not establish and does not attempt.

**Caveats specific to this appendix**, in addition to §8's list:

- The base arm's epoch cap is not uniform: seed 42 ran at 6 epochs, seeds 123 and 456 at 4.
  Seed 123 selected its final epoch and its ceiling under a longer cap is unknown. The base
  response-F1 spread across seeds is 0.0012 and the base→large gap is roughly 0.031, an
  order of magnitude larger, so this is unlikely to affect the direction — but the sweep is
  "seed, with a differing epoch cap on one base arm", not "seed only".
- Large seed 123's metrics file records only its seed under hyperparameters and carries no
  validation block: it was produced by the existing-model inference path rather than by a
  training run's own reporting, so that run's cap and selected epoch are recorded nowhere.
- Per-row predictions were dumped for large seeds 42 and 456 but not for seed 123, so any
  stratified re-analysis of the large arm covers two of three seeds.
- The seed-42 large checkpoint is not published to the Hub.

**Relationship to the rest of the paper: none.** The scaling comparison and the annotation
findings are independent results that happen to share a codebase and a benchmark. Neither
corroborates the other, and this appendix should not be read as strengthening §4, §5, or §6
in any way. It is severable in full.
