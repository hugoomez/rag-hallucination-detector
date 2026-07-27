# Chapter Plan — technical report / preprint

Output of an ARS `academic-paper` plan-mode session (2026-07-25). Produced by Socratic
negotiation; every INSIGHT below is in the author's own words.

**Budget: ~8,300 words main body + ~700 appendix = ~9,000 total.** This is the sum of the
per-chapter allocations in §3, not an independent target — an earlier header said "~7,500",
which no arrangement of those chapters could reach. Venue is deliberately undecided until
the preprint exists; if a venue with a hard limit is chosen later, Ch. 3 (1,300) and Ch. 6
(1,200) are the two with slack, and Appendix A is severable in full by design.

---

## 1. Thesis and framing

### [INSIGHT: thesis_statement]

> RAGTruth's `implicit_true` annotation behaves **asymmetrically across the pipeline**:
> benign at training time (down-weighting it yields no measurable benefit — with
> intervention-magnitude caveats), distorting at evaluation time (it systematically biases
> what the reported metric means). The asymmetry is itself the finding — two measurements
> of one underlying fact at two pipeline stages.

### [INSIGHT: contribution_claim]

> Neither RAGTruth's paper nor LettuceDetect's tells a reader that every published RAGTruth
> F1 — including LettuceDetect's own headline numbers — aggregates two distinguishable error
> types (consequential/false vs. benign/ungrounded-but-true) under a single label, at a
> magnitude we quantify exactly (9.49% of test gold span character mass), imposing a 90.5%
> span-recall ceiling on any detector that tries to report only the consequential kind; nor
> do they report what happens when you actually try to teach a model that distinction via a
> pre-registered loss-reweighting intervention (a null result, informative about the
> difficulty of the distinction itself).

### [INSIGHT: framing_discipline]

> Benchmark-quality / data-quality contribution. **Not** a new detection method. No SOTA
> claim. No single number here is a new record, and the paper does not need one.

### [INSIGHT: scope]

> Ablation + scaling work only. The earlier project arc (NLI baseline, Track A, Approach 1)
> appears as setup, not as results. E10 (3-seed base) → main body as variance anchor.
> E11/E12 (base→large scaling) → labeled appendix, explicitly outside the argument.

### [INSIGHT: reframing_ruling] — the session's most consequential decision

> We misread `implicit_true`. RAGTruth's README defines it as *"correct while the info is
> not mentioned in the context"*; annotator comments confirm it; 90.6% of flagged spans
> carry a LOW-severity prefix vs 4.4% of all other spans. It is a **severity qualifier, not
> a retraction** — the label is correctly applied under RAGTruth's stated faithfulness
> objective. We retract every use of "noise", "annotator disagreement", and the
> `is_noisy_span()` naming, and reframe from **label noise** to **label-class conflation**.

---

## 2. Claims ledger — what register each claim lives in

The paper's defensibility rests on never mixing these three registers. Reviewers kill
papers that state census facts with confidence intervals and inferential claims without them.

| Claim | Register | Status |
|---|---|---|
| 13.5% of gold spans / 14.56% of gold char mass are `implicit_true` | **Census** — exact, whole corpus | Verified, recomputed 2026-07-24 |
| 9.49% of *test* gold span char mass is flagged → 90.5% span-recall bound | **Census** — exact | Computed this session |
| 49/943 test hallucinated responses are entirely flagged → 94.8% response-recall bound | **Census** — exact | Verified |
| Excluding flagged responses raises Subtle-only miss rate 48.1% → 53.3% | **Census** — exact count over a closed 77-response cohort. **No CI. No p-value.** | Verified (ADR-022) |
| Model detects flagged Subtle cases more often than authentic ones (55% vs 47%) | **Inferential** — generalization claim | **Fisher exact p = 0.491.** Not established. Label as candidate mechanism only |
| ACWS at λ=0.25 produced no measurable clean-span improvement | **Single-run observation** | No artifact (E7). Cite ADR-020 as contemporaneous prose |
| The b→c delta (−0.0045) exceeds base-seed spread (0.0006) ~7× | **Suggestive** — metric/arm mismatch stated | Use as evidence against "pure noise", not as replication |
| Large > base on response F1, 3/3 seeds, mean +0.031 | **Descriptive, n=3** | No significance claimed or possible |
| RAGTruth baseline + LettuceDetect discard the field | **Verified in code** | Both re-checked 2026-07-25 |
| Luna / RAG-HAT treatment of the field | **UNVERIFIED** | No public code. Must be stated as unverified, never implied |

**Register rule for drafting:** a census claim about the RAGTruth test set takes no
uncertainty interval, because there is no population being estimated. The moment a sentence
generalizes beyond this benchmark, it moves to the inferential register and must carry its
uncertainty. Q13's ruling is the load-bearing methodological commitment of the paper.

---

## 3. Chapter plan

### Ch. 1 — Introduction (~900 words)

**Core argument.** Every published evaluation on RAGTruth treats its gold labels as a
single undifferentiated positive class. That class in fact contains two materially
different error types, and the benchmark itself ships the field that separates them.

**Gap sentence (author's words, adapted to the Q14 reframing).** Published RAGTruth
evaluations score ungrounded-and-false content and ungrounded-but-true content
identically, though the benchmark distinguishes them — so no reported F1 says how a
detector performs on the class that carries downstream harm.

**Evidence deployed:** the three census numbers (13.5% / 9.49% / 49-of-943); the
verification table showing both public implementations discard the field.

**Must not say:** "noise", "annotators disagreed", "labels are wrong", "beats SOTA",
"hard ceiling on recall".

**Weakest point.** A reader may respond "all benchmarks have label noise, so what?" The
answer must arrive in Ch. 1, not Ch. 7: this is not estimated noise, it is a
**self-declared, machine-readable field the benchmark publishes and the field ignores** —
and it marks a *correctly labeled* subclass, not an error.

---

### Ch. 2 — Background and Related Work (~1,200 words)

**Core argument.** The encoder track on RAGTruth is the cheap frontier, not the frontier;
and the metadata this paper uses has no prior literature.

**Structure:**
1. RAGTruth's construction and faithfulness objective ("unsupported *or* contradictory").
2. The detector landscape (§4.0 table from `01-long-context-truncation.md`), now carrying
   primary-source figures.
3. **On the comparability of published RAGTruth figures** — the ~350-word passage drafted
   at [§4b](#4b-related-work-subsection--comparability-of-published-ragtruth-figures).
   Secondary and motivating; not evidence for the thesis.
4. **`implicit_true` has no literature.** The field postdates the ACL paper (added to data
   Feb 2024) and appears only in the repo README. Note that the RAGTruth paper *does*
   provide an include/exclude option for `due_to_null` — metadata-conditioned scoring is
   precedent the authors established for one field and not the other. This is the single
   best support for the paper's proposal and should be given room.
5. The verification table, stated at its true scope.

**Hard gate — CLOSED 2026-07-25.** Every external figure verified at its originating paper
or explicitly marked secondary. RAG-HAT 83.9 confirmed; Llama-2-13B corrected 78.7 → 80.7;
Prompt GPT-4-turbo corrected 63.4 → 68.3; SelfCheckGPT restored to 60.5 (w/ GPT-4-turbo).
**Luna's 65.4% remains the one secondary-only figure** — its paper resists text extraction
and it releases no code or weights — and must be labelled as such wherever it appears.

**Weakest point.** An unverified comparison table in a paper about measurement rigor is a
self-inflicted contradiction. This is the chapter most likely to sink the paper on a
detail unrelated to its contribution. The verification pass closed the exposure; §4b is
the honest disclosure of what that pass found, and its scoping is the thing to guard —
it is one sentence away from becoming an overclaim the evidence cannot carry.

---

### Ch. 3 — Methods (~1,300 words)

Three subsections, deliberately not folded together, so that pre-registered and post-hoc
work are visibly distinct.

**3.1 Annotation audit computation (~350w).** The exact computation
(`02-implicit-true-audit.md` §5), the `is_noisy_span` predicate **renamed** per the Q14
ruling, per-token flagging against raw rather than union-normalised spans, and the
character-mass-vs-span-count distinction — stating plainly that an earlier version of this
work conflated the two and that 13.5% is by count, 14.56% by character mass.

**3.2 Pre-registered ablation protocol (~600w) — the paper's strongest methodological asset.**
Arms a/b/c; ACWS loss formula; λ=1 reduces to plain CE and λ=0 to masking; the three-clause
decision rule; Gate 4 reproduction of published arm-a numbers before any arm was trusted.
**Foreground that the tolerance was computed from data at runtime, never hardcoded** — that
is what proves it could not be widened post hoc. Give it its own paragraph, not a clause.

**3.3 Post-hoc stratified re-scoring (~350w).** Explicitly labeled exploratory, in contrast
to 3.2's pre-registered status. The n=77 → 47/30 cohort split; the census/inference
boundary declared here rather than in Results.

**Weakest point.** §3.2 describes a hypothesis premised on a misreading. Handle it head-on:
pre-registration means the record shows exactly what was believed and when. State the
premise, state that it was wrong, and note that this is precisely what pre-registration is
for. A reviewer who finds this unaided will treat it as concealment.

---

### Ch. 4 — The conflated label class (~800 words)

**Core argument.** Establishes the shared fact both legs measure. Corpus-level audit; the
73.6% concentration in Subtle Baseless Info; per-task distribution (QA 17.57% vs Summary
4.11% of test gold char mass); the LOW/HIGH severity evidence (90.6% vs 4.4%) that grounds
the reframing; `due_to_null` as the contrasting case — 98.3% Evident Baseless Info in
Data2txt, genuine positives that keep full weight.

**Evidence:** `02-implicit-true-audit.md`; the severity table computed this session; raw
`response.jsonl`.

**Weakest point.** Train (14.49%) is noisier than test (8.95%) — awkward for a claim about
evaluation. The author's ruling: report it plainly as a methodological fact, do not recruit
it as support. Insensitivity does not scale with concentration.

---

### Ch. 5 — Training-side leg: the ACWS null (~1,000 words)

**Core argument.** Down-weighting the benign subclass at λ=0.25 produced no measurable
improvement in clean-span F1 — the model could not be taught this distinction by simple
loss reweighting at the setting tested.

**Evidence:** arms b/c; the decision rule's PASS/FAIL; E10's seed spread (0.0006) as the
noise floor against which the −0.0045 delta is read.

**Opening sentences of the arm-c results subsection — FIXED WORDING, Gate 5, adopted
2026-07-25.** This is the first thing a reader learns about arm c. Not a footnote, not
mid-paragraph, not hedged:

> Arm c's numbers are reported from ADR-020: a committed, timestamped decision record
> written under this study's pre-registered rule at the time the arm was run. **No
> prediction dump or metrics file was retained for the run, although the tooling to
> produce one existed and was used for its sibling arm — the omission is ours, not a
> limitation of the setup.** Alone among the experiments reported here, arm c's figures
> cannot be independently re-derived without retraining, and we mark them as such wherever
> they appear.

The bolded sentence is the author's adopted wording and is not to be softened, moved, or
paraphrased. It deliberately declines an available excuse: the dump tooling shipped in
commit `64422f4` alongside the `--implicit_true_weight` flag that defines arm c, and arm b
was dumped from the same harness on the same day. An earlier draft of this sentence
attributed the gap to tooling that "predates" the run; that was false and was caught
before it reached the paper.

**Required hedges, non-negotiable (Q5/Q6):**
- One λ, one seed, one architecture, ~0.7–0.8% of supervised tokens, ~0.2% of loss mass.
- Cannot distinguish "model is insensitive" from "intervention below resolution". λ=0
  untested and, per the author's ruling, not planned — the confound stays stated and
  unresolved.
- **Arm c has no artifact.** Cite ADR-020 as contemporaneous prose and say so. Every other
  arm resolves to a metrics file; this one resolves to a document, and the reader must see
  the difference rather than have it smoothed by uniform citation style.
- The seed-spread comparison is *official* span F1 vs *clean* span F1, arm c unreplicated.
  Author's framing to preserve verbatim in spirit: **"the most conservative point of
  comparison available, not the most favorable one."**

**Weakest point.** This is the chapter a reviewer will call thin. Its defense is not the
result but the protocol: a pre-registered rule, a validated harness, and a stated inability
to distinguish two explanations. Claim exactly that and nothing more.

---

### Ch. 6 — Evaluation-side leg: the divergence bound (~1,200 words)

**Core argument, leading with the span-level bound per Q12.** A detector optimized to flag
only consequential (false) hallucinations cannot exceed **90.5% span-level recall** under
RAGTruth's official scoring, because the benchmark also scores benign ungrounded-but-true
content as positive. The response-level bound (94.8%) follows as a corroborating instance
at coarser granularity.

**Statement discipline (Q12).** This is a **joint-achievement bound between two goals**,
not a ceiling on a metric in isolation — an always-flag system still reaches 100% recall.
The bound is on *simultaneously* satisfying a consequential-only objective and the official
scoring. Any sentence that drops the conditional is false.

**Subordinate case study (Q10).** The Subtle cohort, explicitly scoped to this model and
presented as a worked example of where the effect concentrates:
- Version 1, **census, exact, no interval**: excluding flagged responses raises the
  Subtle-only miss rate from 48.1% (37/77) to 53.3% (16/30).
- Version 2, **inferential, hedged**: the candidate mechanism (55% vs 47%) at
  **Fisher exact p = 0.491** — descriptive, motivating future work, not demonstrated.

**Weakest point.** The Subtle leg's reach is limited: per the paper's own literature review,
nobody publishes a stratified Subtle miss rate. Keeping it subordinate to the aggregate
bound is what prevents that limitation from touching the main claim.

---

### Ch. 7 — Discussion (~900 words)

**Core argument.** The two legs describe one property of the benchmark measured at two
stages, and the property matters more for evaluation than for training.

**Content:**
- Why the asymmetry is coherent rather than an artifact of measuring twice.
- **The arm-b complication, included deliberately (author's ruling).** Adopting the
  corrected recipe worsened Subtle miss rate 40.3% → 48.1% and shifted FP:FN 0.76 → 0.46.
  It strengthens the argument: Subtle detection is fragile and sensitive to seemingly
  unrelated training decisions, so Subtle performance is a genuinely unresolved problem,
  not an artifact fully explained by label-class conflation.
- The transferable engineering lesson: checkpoint selection under multi-granularity
  objectives. Selecting on response-level F1 — the deployment-relevant metric — silently
  suppressed span-level performance, because response-level F1 structurally cannot
  distinguish a tight span from a sloppy overlapping one. Any system with coarse and
  fine success criteria can fall into this.
- Implication for benchmark reporting, hedged: stratified reporting by
  ungrounded-and-false vs ungrounded-but-true would make published numbers interpretable.

**Weakest point.** The temptation to promote the engineering lesson to a headline. It is a
Discussion item. It does not belong in the abstract.

---

### Ch. 8 — Limitations (~600 words)

Consolidated and unflinching; this chapter is a load-bearing part of the paper's credibility,
not a formality.

Single λ; single seed for arm c; no arm-c artifact; λ=0 untested; the intervention-magnitude
confound (0.2% of loss mass); n=30 authentic-Subtle cohort; Luna/RAG-HAT unverified;
E10's non-uniform epoch cap (seed 42 at 6, seeds 123/456 at 4); large seed-123 config
unrecorded and its predictions never dumped; the seed-42 large checkpoint not published to
the Hub; the earlier count-vs-character-mass conflation, now corrected.

---

### Ch. 9 — Conclusion (~400 words)

Leads with **(a)**: the divergence bound as an exact, verified fact about RAGTruth — the
census claim, no interval. Closes with a brief, explicitly speculative gesture toward
**(b)**: annotation-confidence metadata as a generally under-used signal in benchmark
construction, framed as an implication rather than a demonstrated lesson.

**Deliberately excluded: (c).** No call on benchmark maintainers to act. Per the author:
neither the standing nor the evidence, and doing so would overreach what one audit and one
null ablation support.

---

### Appendix A — Resource-constrained scaling replication (~700 words)

**Opening sentence must state that this is methodologically separate and not part of the
paper's argument.**

3-seed matched base→large comparison. Value retained: LettuceDetect published a single
point estimate with no variance; this provides 3 matched seeds with mean, range, and
per-seed paired deltas (response F1 +0.0311 mean vs their published +0.0315), under
materially more constrained compute — a single consumer GPU, 4 epochs rather than 6.
Evidence about reproducibility under constraint.

**Explicitly avoided:** any language that this "beats" or "validates" the detector as a
method; any implication that the scaling result and the annotation findings support each
other; any suggestion this is central.

---

## 4. Pre-drafting gates

Blockers. Drafting should not begin until each is closed.

1. ✅ **DONE 2026-07-25 — Terminology sweep.** "Noise" and "annotator disagreement"
   retired across `02`, `03`, `04`, ADR-020/021/022 (correction addenda, not rewrites),
   the Track B model card, README, the ledger, and code. `is_noisy_span()` →
   `is_implicit_true_span()`; `is_noisy()` → `is_implicit_true()`;
   `noisy_char_mass_share()` → `implicit_true_char_mass_share()`; region/local names to
   `flagged`/`unflagged`. **Emitted JSON keys and the `clean_span` block name are frozen
   deliberately** — they are the pre-registered on-disk contract of ADR-020's decision
   rule, and renaming them would sever traceability to the rule as written before the arms
   ran. 250 tests pass; `subtle_only_miss_rate.py` re-run and reproduces 48.1% / 53.3% /
   44.7% exactly.
2. ✅ **DONE 2026-07-25 — Related-work verification pass, at primary sources.**
   - **RAG-HAT 83.9% CONFIRMED** at Song et al. Table 2 (P 87.3 / R 80.8).
   - **RAGTruth baselines CORRECTED** to Niu et al. Table 5: Llama-2-13B **80.7%**
     (was 78.7 — that figure is Song's *reproduction*, not Niu's self-report), Prompt
     GPT-4-turbo **68.3%** (was 63.4 — that is LettuceDetect's own number). Neither 78.7
     nor 63.4 appears anywhere in the RAGTruth paper.
   - **SelfCheckGPT restored to 60.5% w/ GPT-4-turbo** (Niu Table 5). An earlier pass
     wrongly "corrected" this correct primary-sourced figure to LettuceDetect's
     `gpt-3.5-turbo` row (58.8) — a different system from a secondary source. Logged as a
     correction in `01` §4.0.1.
   - **Luna remains secondary-only** and is labelled as such.
   - **New finding, see below.**
3. ✅ **DONE — Novelty claim narrowed.** Verified in code 2026-07-25: LettuceDetect's
   `preprocess_ragtruth.py` and RAGTruth's own vendored baseline both discard the field.
   Luna and RAG-HAT marked **UNVERIFIED** (no public code). "No published system uses this
   field" is retired.
4. ✅ **DONE — Subtle re-scoring artifact committed.** `scripts/subtle_only_miss_rate.py`,
   `04-subtle-only-reconciliation.md`, and ADR-022 are all tracked. Note the analysis still
   emits no results JSON — it prints to stdout — so Ch. 6's census claim resolves to a
   script plus a committed prediction dump rather than to a metrics file. Acceptable, but
   state it that way.
5. ✅ **DONE 2026-07-25 — arm-c disclosure sentence adopted.** Wording, placement, and
   register are all settled; the passage is fixed verbatim in
   [Ch. 5](#ch-5--training-side-leg-the-acws-null-1000-words) and must not be softened,
   moved, or paraphrased:

   > No prediction dump or metrics file was retained for the run, although the tooling to
   > produce one existed and was used for its sibling arm — the omission is ours, not a
   > limitation of the setup.

   It opens the arm-c results subsection as the first thing a reader learns about that
   result. A proposed earlier version attributed the missing artifact to tooling that
   "predates" the run; the repository refutes that — commit `64422f4` (2026-07-22) added
   `scripts/dump_token_predictions.py` **and** the `--implicit_true_weight` flag defining
   arm c in the same commit, `train_token_level.py` dumps predictions by default
   (lines 764, 777), and `results/token_preds_arm_b.json` was committed the same day from
   the sibling arm. The adopted wording states what is true and declines the excuse.

---

## 4b. Related Work subsection — comparability of published RAGTruth figures

**Decision (author, 2026-07-25): option (b).** A ~350-word Related Work subsection,
explicitly framed as a **secondary, motivating observation** — not a verified systematic
finding, and not a third leg of the argument. The scoping discipline is the same one
governing the rest of the paper (Q9/Q13): state exactly what was checked, and do not
extrapolate past what two data points support.

**Placement:** Ch. 2, after the detector-landscape table, before the `implicit_true`
literature gap. Supporting detail lives in
[`01-long-context-truncation.md`](01-long-context-truncation.md) §4.0.1.

### Draft text

> **On the comparability of published RAGTruth figures.**
>
> Assembling the comparison above required resolving each figure to its originating paper.
> Three discrepancies emerged. We report them because they bear on how such tables should
> be read, not because three cases support a general claim.
>
> First, prompted GPT-4-turbo is reported at three different response-level F1 values on
> RAGTruth: **68.3** by Niu et al. (2024), the benchmark's own authors; **76.7** by Song et
> al. (2024); and **63.4** by Kovács and Recski (2025) — a spread of 13.3 F1 points. The
> aggregation conventions differ or are undefined. Niu et al. describe their overall column
> as an *average* across task types; Song et al. label theirs OVERALL without definition;
> Kovács and Recski state no convention. We could not reconcile the three from the
> published text.
>
> Second, SelfCheckGPT over gpt-3.5-turbo is reported at **36.6** by Niu et al. and
> **58.8** by Kovács and Recski — a spread of 22.2 F1 points, the largest we found, and one
> where recall diverges too (28.0 versus 71.9). Aggregation convention does not move the
> recall of the same predictions, so that explanation is unavailable here; we could not
> reconcile the two and do not claim to know the cause.
>
> Third, the Llama-2-13B and RAG-HAT rows in Kovács and Recski's comparison table carry
> precision and recall identical to Song et al.'s Table 2 (76.9 / 80.7 and 87.3 / 80.8
> respectively), including the **78.7** F1 that appears in Song et al. for the RAGTruth
> baseline rather than the **80.7** that Niu et al. self-report for the same system. This
> is consistent with figures being carried across rather than recomputed — though it is not
> proof of it. Independent evaluation could coincide, and neither paper states which was
> done.
>
> We draw no conclusion about the comparability of the RAGTruth literature as a whole;
> three cases cannot support one. What they suggest is narrower: that verification of external
> figures against originating sources may be undervalued in this literature, and that a
> systematic audit — resolving every published RAGTruth figure to its primary source and
> stated evaluation protocol — would be worth conducting. That is beyond this paper's
> scope. We contribute only the observation that prompted the question.

### Constraints on any revision of this passage

- **Do not** write "the field's leaderboard is not comparable", or any equivalent
  conclusion. Three cases do not support it.
- **Do not** offer a cause for the SelfCheckGPT gap. The recall divergence rules
  *aggregation convention* out as a sufficient explanation; it does not rule anything in.
- **Do not** assert that Kovács and Recski copied rather than recomputed. State the
  coincidence and its most likely reading; stop there.
- **Do** keep both figures attributed to the papers that print them, and keep "we could
  not reconcile" rather than "cannot be reconciled".
- The passage is motivating context for why this paper verified its own citations. It is
  not evidence for the thesis and must not be recruited as such.

## 5. Open items

- **λ=0 ACWS arm — discarded.** Proposed during planning as the one run that would
  separate "model is insensitive to the distinction" from "intervention below the
  resolution of training". Discarded by the author; not pursued. Ch. 5 therefore states
  the confound and leaves it unresolved, which is the honest position given no further
  arm was run.

- **Venue.** Not settled. The shape — audit + pre-registered null + measurement critique,
  no positive result — fits a benchmark/dataset or analysis track better than a main
  track. Worth deciding before the abstract, since it governs how much of Appendix A
  survives.
- **Title.** Not attempted this session. It should name the conflation, not the null.
