# Research: candidate methods for exploiting RAGTruth's annotation metadata

Follow-on from [`02-implicit-true-audit.md`](02-implicit-true-audit.md). The audit
established that ~13.5% of RAGTruth's gold spans (14.6% of character mass) are marked
`implicit_true` — *"correct while the info is not mentioned in the context"*, i.e.
ungrounded-but-true content, correctly labelled a hallucination under RAGTruth's
faithfulness objective and flagged as low-severity — and that the two systems whose
preprocessing code is public both discard that signal. This document covers what was
proposed to do about it, what was built, and what it measured.

> **Terminology correction (2026-07-25).** Earlier versions of this document described
> `implicit_true` as **label noise** and the candidates below as noise correction. That
> reading is withdrawn ([`02-implicit-true-audit.md`](02-implicit-true-audit.md) §2.1:
> RAGTruth's README definition, the annotator comments, and a 90.6%-vs-4.4% LOW-severity
> split). The correct framing is **label-class conflation** — RAGTruth's positive class
> aggregates ungrounded-and-false with ungrounded-but-true content and ships the field
> that separates them. All three candidates are re-described below as attempts to teach a
> model that *distinction*, not to correct bad labels. No experiment, arm, decision rule,
> or number changes.

**Status:** Candidate 1 (ACWS) is fully backed by the repo — hypothesis, harness,
pre-registered decision rule, results, and verdict. Candidates 2 and 3 (metadata-
calibrated confident learning, dual weighting) were **proposed but never built**; they
are restored here from the original research pass and have no code, ADR, test, or branch
behind them.

---

## Candidate 1 — Annotation-Confidence-Weighted Supervision (ACWS)

### Hypothesis

**As originally stated (superseded).** If a meaningful fraction of positive labels are
ones the annotators disagreed with, a model trained to fit them is pushed to over-flag
exactly the kind of text a deployed detector should leave alone.

**As it must now be read.** The premise above was wrong: the annotators did not disagree
with these labels. What is true is that a meaningful fraction of positive labels mark a
*distinguishable, low-severity subclass* — ungrounded-but-true content. A deployed
detector whose purpose is to surface **consequential** hallucinations arguably should
leave that subclass alone, even though official scoring rewards flagging it.
Down-weighting those positions in the loss should, if the hypothesis holds, improve
detection of the consequential subclass without a proportionate loss of overall recall.

The experiment is unaffected by the correction — the same tokens are down-weighted by the
same amount — but what a null result *means* changes. It is not evidence about robustness
to label noise. It is evidence about whether this distinction can be taught by simple loss
reweighting at all. See the revised verdict below.

Attractive properties, and the reason this was picked as Candidate 1:

- It uses a signal that is **already in the benchmark** and that both code-verifiable
  systems discard — LettuceDetect (`preprocess_ragtruth.py` reads only `start`, `end`,
  `label_type`) and RAGTruth's own reference baseline (no occurrence of the field
  anywhere). Luna and RAG-HAT publish no code; their treatment is unverified.
- It is **training-time only**. Labels, evaluation, and the test set are untouched, so
  results stay comparable to every published RAGTruth number.
- It is **one hyperparameter**, cheap to run as a controlled arm on a Kaggle T4 session.

### Mechanism

A continuous per-token loss weight, not a filter:

```
L = Σ_t (w_t · l_t) / clamp(Σ_t w_t, 1e-8)

l_t = per-token cross-entropy (reduction="none", ignore_index=-100)
w_t = 0                     where label == -100  (context / special / padding)
    = implicit_true_weight  where the token is annotator-flagged
    = 1                     elsewhere
```

Implemented as `weighted_token_ce` + `WeightedTokenTrainer` in
`src/models/train_token_level.py`, exposed as `--implicit_true_weight` (default 1.0).
Unit-tested properties (`tests/test_train_token_level_loss.py`):

- **λ = 1.0** reduces exactly to plain mean cross-entropy — and `main()` never routes
  through the weighted path at all in that case, so the default code path stays
  bit-identical to the pre-ACWS model.
- **λ = 0.0** is exactly loss-masking: flagged tokens leave both numerator and
  denominator.
- Flagged tokens are ~0.7–0.8% of supervised tokens, so the denominator shift is <1% —
  the mechanism is effectively per-token gradient scaling, not a re-normalisation of the
  objective.

The flag itself (`is_implicit_true`) is built in preprocessing and applied to the
**train split only**; eval batches never carry the mask. A token is flagged only if
*every* raw gold span covering it is flagged (`implicit_true` and not `due_to_null`),
computed against raw spans rather than the union-normalised ones so that a token backed
by any unqualified annotation keeps full weight.

### How it was tested: the 3-arm ablation

ACWS could not be tested alone, because a code audit had separately found that Track B's
recipe deviated from LettuceDetect's on four axes at once (lr, effective batch, epochs,
and checkpoint-selection metric). Testing ACWS on top of a divergent recipe would have
confounded the two. Hence three arms:

| Arm | Description | lr | eff. batch | epochs | checkpoint metric | λ |
|---|---|---|---|---|---|---|
| **a** | the then-production Track B model | 2e-5 | 16 | 8 | response F1 | 1.0 |
| **b** | faithful LettuceDetect-recipe replication | 1e-5 | 8 | 6 | **token F1** | 1.0 |
| **c** | arm b **+ ACWS** | 1e-5 | 8 | 6 | token F1 | **0.25** |

Arms b and c differ in exactly one thing, so the ACWS effect is isolated.

**Gate 4 (run before trusting any arm):** the new stratified-evaluation harness had to
reproduce the *published* arm-a numbers from its prediction dump before being used to
judge anything. It did — span-F1 0.5114 vs published 0.5113, response-F1 0.7619 vs
0.7619.

**Five stratified comparison blocks** per arm (`scripts/ablation_report.py`), joining
each arm's per-example prediction dump back to the raw RAGTruth test slice (where
`implicit_true` / `due_to_null` / `label_type` live):

1. `official` — char-overlap span P/R/F1 + response-level P/R/F1. The headline numbers,
   scored identically to training so arm a reproduces its published figures.
2. `clean_span` — the same char-span metric with flagged intervals subtracted from
   **both** gold and predictions. This is the metric the hypothesis is actually about.
3. `noisy_recall_only` — char-overlap **recall** against the flagged-only gold mass.
   Precision is meaningless here; lower = the arm predicts less over ungrounded-but-true
   text. (Key name frozen from the pre-registered harness; read "noisy" as
   "implicit_true" — see `scripts/ablation_report.py`'s TERMINOLOGY note.)
4. `by_task_severity` — char-overlap recall per (task_type × Evident/Subtle) gold cell.
   Recall only, for the same reason: a *predicted* span carries no severity.
5. `response_precision` — response-level precision plus the false-positive rate among
   faithful responses (the paraphrase-FP failure mode).

### Pre-registered decision rule

Written into the harness before the arms were run, printed as PASS/FAIL, so the verdict
is not eyeballed after the fact. Adopt (c) over (b) iff **all three** hold:

```
clean_span_f1(c) > clean_span_f1(b)
response_f1(c)   > response_f1(b)
official_span_recall(b) − official_span_recall(c) ≤ noisy_char_mass_share
```

The third clause is the honesty clause: ACWS is *expected* to cost some official span
recall, because it deliberately teaches the model not to predict over flagged text that
the official metric still scores as gold. The tolerance is the `implicit_true` fraction of
official gold character mass, **computed from the data at runtime, never hardcoded** — so
the budget cannot be quietly widened to fit the result. (The identifier
`noisy_char_mass_share` is frozen as the pre-registered on-disk key; the Python function
that computes it is now `implicit_true_char_mass_share`.)

### Results

| | arm a (production) | arm b (recipe fix) | arm c (b + ACWS λ=0.25) |
|---|---|---|---|
| Official span-F1 | 0.5113 | **0.5321** | — |
| Clean-span F1 | — | 0.5307 | 0.5262 |
| Response F1 | 0.7611 | 0.7631 | 0.7633 |
| Response precision | 0.7873 | **0.8359** | — |
| FP rate on faithful responses | 10.7% | **7.4%** | — |

**Arm b — the recipe fix alone, no ACWS — is the win.** +2.1 span-F1 points over
production, driven by checkpoint selection on span-level F1 instead of response-level F1.
Response-level F1 is structurally unable to distinguish a tight span from a sloppy one
that happens to overlap, so selecting on it was suppressing span performance. This
confirmed the code audit's hypothesis and had nothing to do with ACWS, the architecture,
or the data pipeline.

**Arm c — ACWS at λ=0.25 — failed the rule.** Clean-span F1 was *worse* than arm b
(0.5262 vs 0.5307), and response F1 only trivially better (0.7633 vs 0.7631). The first
clause fails outright, so the rest does not matter. The hypothesis was not supported at
this weight.

### Verdict, and why no λ sweep followed

Arm b's recipe was **adopted** as the Track B production model. ACWS at λ=0.25 was
**rejected and documented as a null result** (ADR-020): well-motivated, pre-registered,
cleanly falsified.

**What the null licenses, and what it does not (revised 2026-07-25).** The original
verdict read this as evidence that "pretrained transformers are surprisingly robust to
moderate, even structured, label noise". That reading is withdrawn with the label-noise
framing: `implicit_true` is not noise, so the result is not about noise robustness. Nor
does it support a general robustness claim, for a separate and more damaging reason —
**the intervention may simply have been too small to measure.** Flagged tokens are
~0.7–0.8% of supervised tokens, and λ=0.25 shifts roughly **0.2% of total loss mass**.
A null is equally consistent with:

- the model being insensitive to this distinction, and
- the perturbation falling below the resolution of the training process.

λ=0 (full masking) was never run, so the two cannot be separated. The defensible claim is
narrow: **at one λ, one seed, one architecture, down-weighting the ungrounded-but-true
subclass produced no measurable improvement in clean-span F1.** Anything stronger
overstates it.

**One piece of evidence that the null is not merely noise.** The b→c clean-span delta is
−0.0045. The base recipe's span-F1 spread across three seeds (E10) is 0.0006 — the delta
is roughly **7× the observed seed-to-seed floor, and points the wrong way**. This is a
mismatched comparison (official span F1 vs clean span F1; arm c is a single unreplicated
seed) and is offered as the most conservative anchor available rather than the most
favourable one. It suggests a real, small, negative effect at this setting; it does not
establish that the effect would replicate.

No further λ values (0, 0.5) were tested. The result was not even *directionally*
encouraging — clean-span F1 moved the wrong way — so a sweep would have been searching
for a value that happened to land, with the tuning risk that implies, at the cost of
additional Kaggle sessions. Testing λ=0 remains the single most informative follow-up,
because it is the one setting that would separate the two explanations above.

### Addendum: arm b is a trade-off, not a strict improvement

A post-adoption reconciliation of arm-b's per-row predictions found something the
aggregate metrics hide (ADR-020 addendum):

| | arm a | arm b |
|---|---|---|
| Subtle-hallucination miss rate | 40.3% | **48.1%** |
| Evident-hallucination miss rate | 27.0% | **30.6%** |
| FP:FN ratio | 0.76 | 0.46 |

Arm b buys its precision by under-flagging, and the recall it gives up is concentrated on
the hardest cases — Subtle hallucinations from strong generators, which this project's own
analysis identifies as the deployment scenario that matters most.

**The interaction with the audit — and the inference that turned out to be wrong.**
§2.3 of [`02-implicit-true-audit.md`](02-implicit-true-audit.md) shows 61% of the test
Subtle-only cohort is entirely `implicit_true`. The natural inference was that part of the
48.1% "miss rate" is therefore not a real capability gap. **That was tested and refuted**:
excluding those 47 responses *raises* the miss rate to 53.3% (16/30), because arm-b flags
the ungrounded-but-true cohort at least as often as the authentic one. The 48.1% figure
**understates** arm-b's difficulty with authentic Subtle hallucinations rather than
inflating it. See [`04-subtle-only-reconciliation.md`](04-subtle-only-reconciliation.md),
which also sets out why the 48.1% → 53.3% shift is an exact census fact while the
mechanism behind it (55.3% vs 46.7%, Fisher exact p = 0.491) is not established.

---

## Candidate 2 — Metadata-calibrated subclass extrapolation

**Not pursued.** Documented for completeness and as a possible follow-up.

> Originally written as "metadata-calibrated confident learning", framing the target as
> undetected *label errors*. Rewritten 2026-07-25 under the label-class-conflation
> reading: the target is undetected members of a **correctly-labelled subclass**, which
> changes both the mechanism's justification and its risk profile. The design is otherwise
> as proposed.

### Idea

Candidate 1 only ever touches spans the annotators explicitly flagged. But annotators
applied `implicit_true` at their own discretion, and there is no reason to assume they
marked every ungrounded-but-true span in the corpus — particularly in the 12,361 spans
that carry no `implicit_true` flag but may still describe content that happens to be true.
Candidate 2 tries to find the unmarked members of that subclass, using the flagged set as
a **calibration reference** rather than as the whole target.

Two stages:

1. **Characterise.** Measure the model's own per-example predicted-probability
   distribution separately over two known populations: flagged (ungrounded-but-true)
   positives, and unflagged positives. If the flagged population is systematically
   predicted with lower confidence, that difference is a usable signature of the subclass.
2. **Extrapolate.** Apply that signature to the unflagged positives to identify
   suspected-but-unmarked members of the same subclass, and treat them the way Candidate 1
   treats the marked ones.

This is confident learning *in mechanism* (using model confidence to find a target
population), but not in purpose: it is not hunting label errors, because there are none to
hunt. It is trying to recover the benchmark's own severity distinction where the
annotators left it implicit — calibrated against a known subpopulation the benchmark
supplies, rather than bootstrapped from scratch.

### Why it was not attempted first

**Circularity is a real failure mode here, not a hypothetical one — and the reframing
makes it worse, not better.** Stage 2 asks the model to identify positives it is least
confident about, then reduces the training pressure to fit exactly those. Under the old
"label error" framing, a bad signature merely failed to find errors. Under the corrected
framing the risk is sharper: **low model confidence is not evidence that a span is
ungrounded-but-true.** It is at least as likely to mark a *consequential* hallucination
the model is simply bad at — precisely the cases the detector exists to catch. Down-weight
those and the model gets worse at them, entrenching the failure. The flagged set provides
a check (the signature must validate on held-out flagged examples before being trusted),
but that constrains the risk rather than eliminating it.

There is also a validity question the original framing hid: the signature would need to
distinguish "ungrounded-but-true" from "hard", and nothing establishes that model
confidence separates those two at all.

It is also strictly more complex than Candidate 1 and depends on it: Stage 2 inherits
whatever weighting scheme Stage 1's population turns out to warrant.

Recommended sequencing was therefore Candidate 1 first, Candidate 2 as a follow-up.
Candidate 1's null result weakens rather than strengthens the case: if down-weighting the
**explicitly marked** subclass at λ=0.25 does not help, a mechanism whose payoff depends
on finding *more* of the same subclass — inferentially, with a signature of unproven
validity — has a smaller expected return and a larger risk surface.

## Candidate 3 — Dual (opposite) weighting

**Not pursued.** Documented for completeness.

### Idea

A two-sided intervention: simultaneously **up-weight** confidently-consequential
positives and **down-weight** flagged (ungrounded-but-true) positives, rather than
Candidate 1's one-sided down-weighting. The intuition is that sharpening the contrast
between the two subclasses from both directions gives a stronger training signal than
pushing from one side alone.

Under the corrected reading this is the most direct expression of what all three
candidates are actually attempting: not noise removal, but teaching the model the
**severity distinction RAGTruth's positive class collapses**.

### Why it was rejected as a first move

Explicitly the riskiest of the three, for two reasons:

1. **It cannot run independently.** The up-weighting side needs a reference point, and
   that reference is Candidate 1's chosen λ. Candidate 3 is therefore serialised behind
   Candidate 1 — it cannot be run in parallel with it, and it cannot be interpreted
   without Candidate 1's result in hand.
2. **It changes two things at once.** Any observed effect could come from the
   up-weighting, the down-weighting, or their interaction, and a single run cannot
   separate them. This is precisely the mistake this project has already paid for:
   ADR-013's post-mortem of the original BIO failure found several changes had moved
   together (label granularity, class weighting, epoch budget, evaluation metric), which
   is why a near-total failure took a code review and a SOTA comparison to diagnose
   instead of being obvious from the run. ADR-020's 3-arm design — arm b changes the
   recipe, arm c changes exactly one further thing — is the corrective, and Candidate 3
   would have walked straight back into the original error.

The machinery partly exists: `--class_weight_cap` and `--implicit_true_weight` do compose
in `weighted_token_ce`. (Note the documented caveat — with class weights set, the
denominator uses only `w_t` rather than PyTorch's class-weighted-mean convention.
Irrelevant to the ablation arms, which used plain CE, but it would matter here.) What
does not exist is a per-token *up*-weighting input for "confidently-consequential"
positives: that population is not annotated — RAGTruth marks the low-severity subclass,
not the high-severity one — and would have to come from Candidate 2.

## Ordering, in retrospect

The three candidates form a dependency chain — 1 → 2 → 3 — with strictly increasing
complexity, risk, and attribution difficulty. Candidate 1 was chosen first because it was
the only one that could be run and interpreted on its own, in one controlled arm, in a
free-tier GPU session. It returned a clean null. ADR-020's decision not to sweep λ applies
with more force to Candidates 2 and 3, which are both amplifications of a mechanism that
showed no directional effect at its simplest setting.

One caveat on that ordering, visible only after the reframing: Candidate 1's null is
confounded by intervention magnitude (~0.2% of loss mass), so it is weaker evidence
against Candidates 2 and 3 than it looks. A λ=0 arm would settle whether the mechanism is
inert or merely under-powered, and is a cheaper next step than either follow-on.

**Provenance note:** Candidates 2 and 3 are restored from the original research pass
(author's notes, 2026-07). Unlike Candidate 1, they have **no footprint in the repo** —
no code, no ADR, no test, no branch. Nothing in this section was run or measured; it
records what was proposed and why it was deprioritised, and should be read as design
rationale, not as a result.

---

## Provenance

| Claim | Source |
|---|---|
| ACWS hypothesis, arms, findings, verdict, λ-sweep reasoning | ADR-020 |
| Loss formula, λ=0/1 properties, ~0.7–0.8% flagged-token share | `src/models/train_token_level.py` (`weighted_token_ce`, `WeightedTokenTrainer`), `tests/test_train_token_level_loss.py` |
| Flag construction, train-split-only application | `src/data/preprocess_token_level.py`, `src/models/train_token_level.py` (`with_implicit_mask`) |
| Five stratified blocks, pre-registered decision rule, Gate 4 | `scripts/ablation_report.py` module docstring, `tests/test_ablation_report.py` |
| Arm a / arm b metric values | `results/arm_a_original_metrics.json` (recovered from commit `b54604d`, where the shared filename was later overwritten by arm b); `results/arm_b_metrics.json` |
| Arm c metric values | ADR-020 prose only — **no arm-c artifact exists**, so these numbers are not independently re-derivable without retraining (see [`../EXPERIMENT_LEDGER.md`](../EXPERIMENT_LEDGER.md)) |
| Candidates 2 and 3 | Original research pass (author's notes, 2026-07). **No repo footprint** — proposed, never built |
| Subtle/Evident miss rates, FP:FN ratio | ADR-020 addendum, `README.md`, `docs/model_cards/track_b.md` |
| 48.1% → 53.3% refutation of the dilution inference | [`04-subtle-only-reconciliation.md`](04-subtle-only-reconciliation.md), ADR-022, `scripts/subtle_only_miss_rate.py` |
| E10 seed spread (span-F1 range 0.0006) used as the noise floor | ADR-021, `results/seed_aggregate.json` |
| Label-class-conflation reframing (README definition, annotator comments, 90.6% vs 4.4% severity split) | [`02-implicit-true-audit.md`](02-implicit-true-audit.md) §2.1; ADR-020 correction addendum |
| LettuceDetect discards the metadata | Their `preprocess_ragtruth.py` on GitHub — checked 2026-07-12, **re-verified 2026-07-25** |
| RAGTruth's own baseline discards it | `data/raw/ragtruth/baseline/` (vendored), checked 2026-07-25 |
| Luna / RAG-HAT treatment | **UNVERIFIED** — no public code or weights, 2026-07-25 |
