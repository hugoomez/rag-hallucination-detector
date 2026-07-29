# Paper outline — section-level, with evidence map

Output of an ARS `academic-paper` `outline-only`-mode pass (2026-07-25). Follows
[`05-paper-chapter-plan.md`](05-paper-chapter-plan.md)'s chapter structure exactly, per that
document's status as the authoritative spine from the Socratic planning session — nothing
here re-derives or renegotiates that plan. Evidence citations use
[`06-lit-review-annotated-bibliography.md`](06-lit-review-annotated-bibliography.md) for
external sources and this repo's own `docs/research/02`, `04`, `EXPERIMENT_LEDGER.md`, and
`docs/decisions.md` for internal ones. Terminology follows the Q14/Gate-1 ruling:
**label-class conflation**, never "label noise" or "annotator disagreement."

**Title and venue: still open** (chapter plan §5). Not attempted in this pass; a title
should name the conflation, not the null.

**Word-count table (updated 2026-07-25 — trim pass).** Two chapters were trimmed at the
author's request; every other chapter's budget and content is unchanged from the prior
pass.

| Chapter | Prior budget | Current budget | Δ | Note |
|---|---:|---:|---:|---|
| Ch. 1 — Introduction | 900 | 900 | — | untouched |
| Ch. 2 — Background/Related Work | 1,200 | **900** | **−300** | §2.2/§2.4 prose compressed; §2.3 (§4b) untouched, still 300–350 |
| Ch. 3 — Methods | 1,300 | 1,300 | — | untouched |
| Ch. 4 — The conflated label class | 800 | 800 | — | untouched |
| Ch. 5 — Training-side leg (ACWS null) | 1,000 | 1,000 | — | untouched |
| Ch. 6 — Evaluation-side leg (divergence bound) | 1,200 | 1,200 | — | untouched |
| Ch. 7 — Discussion | 900 | 900 | — | untouched |
| Ch. 8 — Limitations | 600 | **400** | **−200** | converted to tight bullets; all 11 items retained |
| Ch. 9 — Conclusion | 400 | 400 | — | untouched |
| **Main body total** | **8,300** | **7,800** | **−500** | |
| Appendix A | 700 | 700 | — | untouched |
| **Grand total** | **9,000** | **8,500** | **−500** | |

**IMRaD mapping**, since the chapter plan's 9 chapters don't map 1:1 to the four IMRaD
blocks:

| IMRaD block | Chapters |
|---|---|
| Introduction | Ch. 1 |
| Background / Related Work | Ch. 2 |
| Methods | Ch. 3 |
| Results | Ch. 4 (shared fact), Ch. 5 (training-side leg), Ch. 6 (evaluation-side leg) |
| Discussion | Ch. 7 |
| Limitations | Ch. 8 (kept separate from Discussion — author's ruling, chapter plan §3) |
| Conclusion | Ch. 9 |
| Supplementary | Appendix A |

---

## Ch. 1 — Introduction (~900 words)

**Argument arc.**
1. Every published RAGTruth evaluation treats the gold label as one undifferentiated
   positive class.
2. That class in fact contains two materially different error types
   (ungrounded-and-false vs. ungrounded-but-true), and the benchmark ships the field
   (`implicit_true`) that separates them.
3. Gap sentence: published RAGTruth evaluations score both identically, though the
   benchmark distinguishes them — so no reported F1 says how a detector performs on the
   class that carries downstream harm.
4. Anticipate the "all benchmarks have label noise" objection **here**, not in Ch. 7: this
   is not estimated noise, it is a self-declared, machine-readable field the benchmark
   publishes and the field ignores, and it marks a *correctly labeled* subclass, not an
   error.
5. State the thesis (training/evaluation asymmetry) and the two-leg structure the paper
   will follow.
6. One-sentence forward pointer to Track B as prior, separately-developed project
   infrastructure (repo-cited, not re-argued) — per the scope ruling.

**Must not say:** "noise", "annotators disagreed", "labels are wrong", "beats SOTA", "hard
ceiling on recall" (unqualified).

**Evidence map**

| Claim | Register | Support |
|---|---|---|
| 13.5% of gold spans / 14.56% of gold char mass are `implicit_true` | Census, whole corpus | `docs/research/02-implicit-true-audit.md` §1 |
| 9.49% of test gold char mass flagged → 90.5% span-recall bound | Census, exact | `02-implicit-true-audit.md` §3 ("Span-level bound") |
| 49/943 test hallucinated responses entirely flagged → 94.8% response-recall bound | Census, exact | `02-implicit-true-audit.md` §3 |
| Both public implementations (RAGTruth baseline, LettuceDetect) discard the field | Verified in code, 2026-07-25 | `02-implicit-true-audit.md` §4 "Comparability note"; Gate 3, chapter plan §4 item 3 |
| Thesis statement (asymmetry) | Framing, not a data claim | `05-paper-chapter-plan.md` §1 `thesis_statement` |

**Weakest point (carried forward, not to be dropped in drafting):** the "all benchmarks
have noise" pre-emption is load-bearing and must land in this chapter.

---

## Ch. 2 — Background and Related Work (~900 words)

**Structure**

**2.1 RAGTruth's construction and faithfulness objective (~150w).**
- 17,790 responses, QA/Summary/Data2txt, span-level annotation under "unsupported *or*
  contradictory."
- Source: Niu et al. 2024.

**2.2 The detector landscape (~250w). [trimmed from ~350w — table and every figure
below are unchanged; only the surrounding prose is compressed.]**
- Table reproduced verbatim from `01-long-context-truncation.md` §4.0, primary-source
  verified: RAG-HAT 83.9; LettuceDetect-large 79.22 / base 76.07; RAGTruth baseline
  Llama-2-13B 80.7; this project's Track B arm-b 76.31; prompted GPT-4-turbo 68.3; Luna
  65.4 (flagged secondary); SelfCheckGPT+GPT-4-turbo 60.5. No figure or citation cut.
- One framing sentence: the encoder track is the cheap frontier, not the frontier —
  RAG-HAT's 8B-decoder DPO pipeline beats every encoder by a wide margin.

**2.3 On the comparability of published RAGTruth figures (~300–350w). [NOT touched by
this trim — word count and text are locked by chapter plan §4b.]**
- Insert the **exact draft text already fixed** in
  [`05-paper-chapter-plan.md` §4b](05-paper-chapter-plan.md#4b-related-work-subsection--comparability-of-published-ragtruth-figures) —
  do not redraft.
- Three cases only: GPT-4-turbo (68.3 / 76.7 / 63.4), SelfCheckGPT gpt-3.5-turbo (36.6 vs
  58.8, recall 28.0 vs 71.9), Llama-2-13B/RAG-HAT rows in Kovács & Recski matching Song et
  al.'s P/R exactly.
- **Placement is fixed:** after 2.2, before 2.4. **Scope is fixed:** motivating context
  only, not a third spine leg, no generalization to "the field's numbers aren't
  comparable."

**2.4 `implicit_true` has no literature (~200w). [trimmed from ~300w — the `due_to_null`
precedent argument is kept at full strength, not thinned; the cut comes out of the
surrounding framing only.]**
- Field postdates the ACL paper (added Feb 2024, README-only), verified against
  arXiv:2401.00396 full text.
- **`due_to_null` include/exclude precedent, full strength, unthinned:** RAGTruth's own
  authors built exactly this affordance for one metadata field and not the other — "the
  single best support for the paper's proposal," give it room.
- Verification table at its true, narrowed scope: RAGTruth baseline + LettuceDetect
  verified in code; Luna and RAG-HAT explicitly UNVERIFIED, no public code.

**Evidence map**

| Claim | Register | Support |
|---|---|---|
| RAGTruth construction, faithfulness objective | Descriptive | Niu et al. 2024 (`06-lit-review...md`, entry 1) |
| Detector landscape table figures | Verified primary-source, 2026-07-25 | `01-long-context-truncation.md` §4.0 |
| §4b comparability cases (3) | Motivating, explicitly not generalized | `01-long-context-truncation.md` §4.0.1; fixed text in `05-paper-chapter-plan.md` §4b |
| `implicit_true` absent from ACL paper, added Feb 2024 | Verified | `02-implicit-true-audit.md` §2.2 |
| `due_to_null` include/exclude precedent | Verified | Niu et al. 2024, evaluation section; `02-implicit-true-audit.md` §2.2 |
| RAGTruth baseline + LettuceDetect discard the field, verified in code | Verified in code, 2026-07-25 | `02-implicit-true-audit.md` §4; `06-lit-review...md` entries for Niu, Kovács & Recski |
| Luna / RAG-HAT treatment of the field | **UNVERIFIED — state as such** | `06-lit-review...md` gap #4; RAG-HAT 5/7-author overlap with RAGTruth noted as a *separate* independence caveat, not a field-verification claim |
| RAG-HAT shares 5 of 7 authors with RAGTruth | Verified, 2026-07-25 | `06-lit-review-annotated-bibliography.md`, Song et al. entry, §1 row 3 cross-check |

**Weakest point:** this chapter is "most likely to sink the paper on a detail unrelated to
its contribution" — the verification pass is what closes that exposure; §2.3's scoping is
the thing to actively guard while drafting.

---

## Ch. 3 — Methods (~1,300 words)

**3.1 Annotation audit computation (~350w).**
- Exact computation per `02-implicit-true-audit.md` §5.
- Predicate renamed per Q14: `is_implicit_true_span()` (not `is_noisy_span()`).
- Per-token flagging against **raw**, not union-normalized, spans.
- State plainly: count share (13.5%) ≠ character-mass share (14.56%); an earlier version
  of this work conflated the two.

**3.2 Pre-registered ablation protocol (~600w) — strongest methodological asset.**
- Arms a (production)/b (LettuceDetect-recipe replication)/c (b + ACWS λ=0.25).
- ACWS loss formula; λ=1 → plain CE, λ=0 → masking (untested, not planned).
- The three-clause pre-registered decision rule.
- Gate 4: reproduction of published arm-a numbers (span-F1 0.5114 vs. published 0.5113;
  response-F1 0.7619 vs. 0.7619) before any arm was trusted.
- **Foreground, own paragraph:** the tolerance was computed from data at runtime, never
  hardcoded — this is what proves it wasn't widened post hoc.
- Head-on handling of the pre-registration-under-a-misreading issue: state the premise
  (label noise), state it was wrong, note this is exactly what pre-registration is for.

**3.3 Post-hoc stratified re-scoring (~350w).**
- Explicitly labeled exploratory, contrasted with 3.2's pre-registered status.
- n=77 Subtle-only cohort → 47 (all-`implicit_true`) / 30 (authentic) split.
- Declare the census/inference boundary **here**, not in Results.

**Evidence map**

| Claim | Register | Support |
|---|---|---|
| Audit computation, predicate definitions | Methods, reproducible | `02-implicit-true-audit.md` §5; `src/data/preprocess_token_level.py` |
| Count (13.5%) vs char-mass (14.56%) distinction | Correction, stated plainly | `02-implicit-true-audit.md` §1 correction note |
| Arm a/b/c protocol, ACWS formula, decision rule | Pre-registered methods | ADR-020 (`docs/decisions.md`); `docs/research/03-candidate-methods.md` |
| Gate 4 reproduction (0.5114/0.5113, 0.7619/0.7619) | Verified | ADR-020 "Findings"; `EXPERIMENT_LEDGER.md` E5/E6 note on the two arm-a numbers |
| Tolerance computed at runtime, not hardcoded | Methods integrity claim | `scripts/ablation_report.py` (cited per chapter plan §3, Ch.3 weakest-point note) |
| n=77 → 47/30 split, join method | Methods, reproducible | `04-subtle-only-reconciliation.md` §1 |

**Weakest point:** §3.2 describes a hypothesis premised on a misreading — the chapter plan
is explicit this must be handled head-on, not smoothed over.

---

## Ch. 4 — The conflated label class (~800 words)

**Core argument.** Establishes the shared fact both legs (Ch. 5, Ch. 6) measure.

**Content, in order:**
- Corpus-level audit headline numbers (13.5% / 14.56%).
- 73.6% concentration in Subtle Baseless Info (2,527 spans, 1,861 flagged).
- Per-task distribution: QA 17.57% vs. Summary 4.11% of test gold char mass.
- LOW/HIGH severity evidence (90.6% vs. 4.4%) — the evidentiary core of the reframing.
- `due_to_null` as the contrasting case: 98.3% Evident Baseless Info in Data2txt, genuine
  positives, kept at full weight.
- Train (14.49%) vs. test (8.95%) asymmetry: report plainly as a methodological fact, do
  **not** recruit it as support for anything.

**Evidence map**

| Claim | Register | Support |
|---|---|---|
| 73.64% of Subtle Baseless Info is `implicit_true` | Census, exact | `02-implicit-true-audit.md` §2.3 table |
| Per-task char-mass shares (QA 17.57% test / 20.72% train; Summary 4.11% / 8.50%; Data2txt 5.12% / 11.44%) | Census, exact | `02-implicit-true-audit.md` §3 "Character mass by split × task type" |
| Severity distribution 90.6% LOW (flagged) vs. 4.4% LOW (other) | Computed 2026-07-25 | `02-implicit-true-audit.md` §2.1(c) |
| `due_to_null`: 98.3% Evident Baseless Info in Data2txt | Census | `02-implicit-true-audit.md` §3 "two fields that mark something else" |
| Train 14.49% vs. test 8.95% flagged positive-token share | Census, exact, no CI | `02-implicit-true-audit.md` §5 "Live re-check" table |

**Weakest point:** train-noisier-than-test is awkward for an evaluation-focused claim —
state it, don't lean on it. "Insensitivity does not scale with concentration."

---

## Ch. 5 — Training-side leg: the ACWS null (~1,000 words)

**Core argument.** Down-weighting the ungrounded-but-true subclass at λ=0.25 produced no
measurable improvement in clean-span F1.

**Structure:**
- Arms b/c results.
- **Arm-c results subsection opens with the fixed Gate-5 sentence, verbatim, unhedged, not
  moved, not a footnote:**

  > Arm c's numbers are reported from ADR-020: a committed, timestamped decision record
  > written under this study's pre-registered rule at the time the arm was run. No
  > prediction dump or metrics file was retained for the run, although the tooling to
  > produce one existed and was used for its sibling arm — the omission is ours, not a
  > limitation of the setup. Alone among the experiments reported here, arm c's figures
  > cannot be independently re-derived without retraining, and we mark them as such
  > wherever they appear.

- E10's seed spread (0.0006 clean-span F1) as the noise floor against which arm b→c's
  −0.0045 delta is read — "the most conservative point of comparison available, not the
  most favorable one" (official span F1 vs. clean span F1, arm c unreplicated — state the
  mismatch).
- Required hedges (non-negotiable): one λ, one seed, one architecture; ~0.7–0.8% of
  supervised tokens, ~0.2% of loss mass; cannot distinguish "insensitive" from "below
  resolution"; λ=0 untested and not planned, confound stays unresolved.

**Evidence map**

| Claim | Register | Support |
|---|---|---|
| Arm b span-F1 0.5321 / response-F1 0.7631 | Verified | `EXPERIMENT_LEDGER.md` E6; `results/arm_b_metrics.json` |
| Arm c clean-span F1 0.5262 vs. arm b 0.5307; response F1 0.7633 vs. 0.7631 | Single-run, **no artifact** | ADR-020 "Findings"; `EXPERIMENT_LEDGER.md` E7 note ("E7: arm c has no artifact") |
| Arm-c disclosure sentence | Fixed wording, Gate 5 closed 2026-07-25 | `05-paper-chapter-plan.md` Ch.5, verbatim |
| E10 seed spread: response-F1 range 0.0012, span-F1 range 0.0006 | Descriptive, n=3 | `EXPERIMENT_LEDGER.md` E10; ADR-021 correction addendum |
| Intervention magnitude: ~0.7–0.8% supervised tokens, ~0.2% loss mass | Stated hedge | ADR-020 correction addendum |
| λ=0 discarded, not pursued | Scoping decision | `05-paper-chapter-plan.md` §5 "Open items" |

**Weakest point:** this chapter will read thin to a reviewer. Its defense is the protocol
(pre-registered rule + validated harness + stated inability to distinguish two
explanations), not the result. Claim exactly that.

---

## Ch. 6 — Evaluation-side leg: the divergence bound (~1,200 words)

**Core argument, span-level bound leads (Q12).**
- **No detector can restrict itself to ungrounded-and-false hallucinations and simultaneously
  score char-overlap span recall above 90.5%** under RAGTruth's official scoring.
- Response-level bound (94.8%) as corroborating instance at coarser granularity, secondary.
- **Statement discipline, non-negotiable:** joint-achievement bound between two goals, not
  a ceiling on a metric alone (an always-flag system still reaches 100% recall). Every
  sentence stating the bound must carry the conditional.

**Subordinate case study — the Subtle cohort (Q10), explicitly scoped to this model:**
- Version 1 (census, exact, no interval): excluding flagged responses raises the
  Subtle-only miss rate 48.1% (37/77) → 53.3% (16/30).
- Version 2 (inferential, hedged): candidate mechanism — 55.3% (26/47) vs. 46.7% (14/30)
  detection — Fisher's exact p = 0.491. Descriptive, motivating, not demonstrated.
- Explicitly state what the finding does and does not support: it makes arm-b *look
  better* at Subtle detection than it is on authentic cases, not worse; two paraphrases are
  banned outright (per `04-subtle-only-reconciliation.md` §4): "the Subtle weakness is a
  labeling artifact that inflates the miss rate" (direction is opposite), and "the model is
  better at ungrounded-but-true cases" as a property of the model (not established).

**Evidence map**

| Claim | Register | Support |
|---|---|---|
| Span-level bound: 90.5% (8,151/85,877 test char mass) | Census, exact, no CI | `02-implicit-true-audit.md` §3 "Span-level bound" |
| Response-level bound: 94.8% (49/943) | Census, exact, no CI | `02-implicit-true-audit.md` §3 "Responses that are entirely ungrounded-but-true" |
| Subtle miss rate 48.1%→53.3% (37/77 → 16/30) | Census, exact, **no CI** (±18pp caveat formally withdrawn) | `04-subtle-only-reconciliation.md` §2–3.1; ADR-022 correction addendum |
| 55.3% (26/47) vs. 46.7% (14/30) detection gap | Inferential, **not established**, Fisher p=0.491 | `04-subtle-only-reconciliation.md` §3.2 |
| Two banned paraphrases | Drafting constraint | `04-subtle-only-reconciliation.md` §4 |

**Weakest point:** no published system reports a stratified Subtle miss rate — this is a
genuine limit on the Subtle case study's reach. Keeping it explicitly subordinate to the
aggregate bound is what prevents that limitation from touching the main claim.

---

## Ch. 7 — Discussion (~900 words)

**Content, in order:**
1. Why the training/evaluation asymmetry is coherent, not an artifact of measuring the
   same thing twice.
2. **Arm-b complication, included deliberately (ADR-020 addendum):** adopting the
   corrected recipe *worsened* Subtle miss rate 40.3% → 48.1% (Evident 27.0% → 30.6%);
   FP:FN shifted 0.76 → 0.46. State plainly this strengthens rather than undercuts the
   argument — Subtle detection is fragile to unrelated training decisions, so it's a
   genuinely unresolved problem, not fully explained by label-class conflation.
3. Transferable engineering lesson: checkpoint selection under multi-granularity
   objectives — response-F1 selection silently suppressed span-level performance, because
   response-F1 structurally can't distinguish a tight span from a sloppy overlapping one.
   **Discussion-only; must not be promoted to the abstract.**
4. Hedged implication for benchmark reporting: stratified reporting by
   ungrounded-and-false vs. ungrounded-but-true would make published numbers
   interpretable.

**Evidence map**

| Claim | Register | Support |
|---|---|---|
| Arm-b Subtle/Evident miss-rate shift (40.3→48.1 / 27.0→30.6), FP:FN 0.76→0.46 | Verified | ADR-020 "Addendum (post-deployment reconciliation)" |
| Checkpoint-selection lesson (response-F1 selection suppressed span-F1) | Interpretive, grounded in ADR-013/014/020 | ADR-013 (diagnosis of the original defect), ADR-020 (arm-b's fix and its cost) |
| Stratified-reporting implication | Hedged proposal, not a demand | `05-paper-chapter-plan.md` Ch.7 |

**Weakest point:** resist promoting the checkpoint-selection lesson to a headline — it is a
Discussion item.

---

## Ch. 8 — Limitations (~400 words)

**Format, changed by this trim: tight bullet list, not prose paragraphs.** Every item
below survived the cut — only wording is compressed, not content:

- Single λ (0.25) tested; single seed for arm c; **no arm-c artifact**.
- λ=0 untested — discarded by author decision, not oversight.
- Intervention-magnitude confound: flagged tokens ~0.2% of loss mass.
- n=30 authentic-Subtle cohort — small denominator; no CI (ADR-022 correction; do not
  reintroduce one).
- Luna/RAG-HAT: field-handling unverified, no public code.
- E10 epoch cap non-uniform: seed 42 at 6 epochs, seeds 123/456 at 4.
- Large seed-123: config unrecorded, predictions never dumped.
- Seed-42 large checkpoint not published to the Hub.
- Count (13.5%) vs. char-mass (14.56%) conflation — self-disclosed, now corrected.

**Evidence map**

| Claim | Register | Support |
|---|---|---|
| E10 epoch-cap non-uniformity | Documented caveat | `EXPERIMENT_LEDGER.md` "Caveats" section, E10; ADR-021 |
| Large seed-123 unrecorded config, no prediction dump | Documented caveat | `EXPERIMENT_LEDGER.md` E11 note; ADR-021 "Status" open follow-ups |
| Seed-42 large checkpoint not on Hub | Documented caveat | ADR-021 "Status" |
| Count-vs-char-mass conflation, corrected | Self-disclosed correction | `02-implicit-true-audit.md` §1 correction note |

---

## Ch. 9 — Conclusion (~400 words)

- Leads with **(a)**: the divergence bound as an exact, verified, census fact about
  RAGTruth — no interval.
- Closes with a brief, explicitly speculative gesture toward **(b)**: annotation-confidence
  metadata as an under-used signal in benchmark construction generally — framed as
  implication, not demonstrated lesson.
- **Deliberately excluded — (c):** no call to action directed at benchmark maintainers.
  Per the author's ruling: neither the standing nor the evidence supports it.

**Evidence map:** restates Ch. 4/Ch. 6 census figures only; no new claims introduced.

---

## Appendix A — Resource-constrained scaling replication (~700 words)

**Opening sentence must state:** this is methodologically separate, not part of the
paper's argument.

- 3-seed (42, 123, 456) matched base→large comparison.
- Response F1 mean **+0.0311** (range +0.0292…+0.0323) vs. LettuceDetect's published
  +0.0315 — response-level "replicates"; span-F1 delta **+0.0408** (range +0.0379…+0.0446)
  is *larger* than their published +0.0349 — span-level does not merely replicate, it
  exceeds.
- Value framed as: LettuceDetect published a single point estimate with no variance; this
  supplies 3 matched seeds with mean/range/per-seed paired deltas, under materially more
  constrained compute (single consumer GPU, 4 epochs vs. their 6).
- **Explicitly avoid:** any "beats"/"validates" language; any implication the scaling
  result and the annotation findings support each other; any suggestion this is central.

**Evidence map**

| Claim | Register | Support |
|---|---|---|
| Response F1 +0.0311 mean, 3/3 seeds | Descriptive, n=3, no significance claimed | `EXPERIMENT_LEDGER.md` E12; ADR-021 "Findings" |
| Span F1 +0.0408 mean, 3/3 seeds | Descriptive, n=3 | ADR-021 "Findings" |
| LettuceDetect published deltas (+0.0315 / +0.0349) | External, primary | `06-lit-review-annotated-bibliography.md`, Kovács & Recski entry |
| Epoch-cap / unrecorded-config caveats | Documented | ADR-021 "Caveat" |

---

## Cross-cutting checks applied while building this outline

- **No chapter states a census figure with a confidence interval** (Ch. 4/6/8 all carry
  the "no CI" annotation explicitly, per Q13/ADR-022).
- **No chapter states the Fisher-exact mechanism without its p-value attached** (Ch. 6).
- **§4b (now Ch. 2.3) is bounded to ~300–350 words and does not appear as a Ch. 4/5/6
  claim anywhere** — checked against all three evidence maps above; no leakage found.
- **Appendix A is the only place E11/E12 appear** — Ch. 4–7 evidence maps cite only E10 (as
  variance anchor, per chapter plan §1 `scope`) — checked, no leakage found.
- **"Label noise" / "annotator disagreement" do not appear in any evidence-map row above** —
  every register is either Census, Inferential, Single-run/no-artifact, or Descriptive-n=3,
  per the Ch. 2 claims ledger.

## Open items carried into drafting (not resolved by this outline)

1. **Title** — should name the conflation, not the null (chapter plan §5).
2. **Venue** — undecided; governs how much of Appendix A survives if a hard word limit
   applies later (chapter plan header note).
