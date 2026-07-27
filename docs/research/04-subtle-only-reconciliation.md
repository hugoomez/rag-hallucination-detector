# Research: does the `implicit_true` subclass inflate Track B's Subtle-only miss rate?

Follow-on from [`02-implicit-true-audit.md`](02-implicit-true-audit.md). That audit found
that 61% of the test set's 77 "Subtle-only" responses (every gold span's `label_type`
starting with `Subtle`) consist entirely of `implicit_true` spans — content RAGTruth marks
as *correct while the info is not mentioned in the context*, i.e. ungrounded-but-true. It
flagged, but did not answer, the natural follow-up: does that subclass explain part of
arm-b's reported 48.1% Subtle-only miss rate (vs. 30.6% for Evident-only, per
[`03-candidate-methods.md`](03-candidate-methods.md))?

**Status:** answered. The result runs opposite to the naive hypothesis — see §2.
Terminology corrected 2026-07-25: an earlier version of this document called the
`implicit_true` subclass "annotation noise". It is not noise — see
[`02-implicit-true-audit.md`](02-implicit-true-audit.md) §2.1. Only the wording and the
statistical framing in §3 changed; every count below is unaltered.

---

## 1. Question and method

**Question tested:** if the 47 all-`implicit_true` responses are removed from the
Subtle-only cohort's miss-rate denominator, does the reported 48.1% miss rate go down?
That is: is the headline number inflated by the model declining to flag content a
consequential-only detector would arguably be right to leave alone?

**Method:**

1. Take the test-set "Subtle-only" cohort: responses with ≥1 gold span, where every gold
   span's `label_type` starts with `"Subtle"` (no Evident spans present). n = 77, all
   gold-hallucinated by construction.
2. Split it into two disjoint subsets:
   - **all-`implicit_true`** (n = 47): every gold span in the response is flagged.
   - **authentic-Subtle** (n = 30): the complement — at least one gold span is *not*
     flagged.
3. Use arm-b's (Track B / production model, ADR-020) predictions from
   `results/token_preds_arm_b.json`, joined back to the raw span metadata via the same
   pattern as `scripts/ablation_report.py::build_test_meta` —
   `load_merged_dataframe()` → filter `split == 'test'` → `reset_index(drop=True)`;
   positional index equals `row_index` in the prediction dump.
4. Compute the response-level miss rate (false-negative rate: `resp_pred == 0` on a
   gold-hallucinated response) over each subset.

Script: [`scripts/subtle_only_miss_rate.py`](../../scripts/subtle_only_miss_rate.py).
Reproduce with `python scripts/subtle_only_miss_rate.py` from the repo root (no GPU or
Kaggle needed — it reads the committed prediction dump and joins to
`data/raw/ragtruth/`).

## 2. Result

| Cohort | N | FN | Miss rate |
|---|---:|---:|---:|
| (a) Full Subtle-only cohort — **as currently reported** | 77 | 37 | **48.1%** |
| (b) Authentic-Subtle only (all-`implicit_true` excluded) | 30 | 16 | **53.3%** |
| (c) All-`implicit_true`-only (for reference) | 47 | 21 | 44.7% |

Row (a) reproduces the published 48.1% exactly (37/77), and the 47/77 (61.0%) split
matches §2.3 of the audit — confirming the join is correct and consistent with the earlier
result.

**Finding, stated precisely:** removing the flagged responses does **not** reduce the miss
rate — it *increases* it, from 48.1% to 53.3%. This runs opposite to the naive
dilution hypothesis. **The 48.1% headline figure understates, rather than overstates,
arm-b's real difficulty with authentic Subtle hallucinations.**

## 3. Two claims, two statistical registers

This analysis produces two claims. They are not equally supported, and reporting them the
same way would be an error in both directions.

### 3.1 The census claim — exact, no interval

> On RAGTruth's test set, excluding all-`implicit_true` responses raises the Subtle-only
> miss rate from **48.1% (37/77)** to **53.3% (16/30)**.

For a benchmark-measurement question, the RAGTruth test set is the **object of study**,
not a sample drawn from a wider population. This is an arithmetic fact about a closed,
fully-enumerated cohort of 77 responses. It takes **no confidence interval**, because
nothing is being estimated.

> **Withdrawn (2026-07-25):** an earlier version of this section attached a
> "±18 percentage points at 95%" interval to the 53.3% figure. That was a misapplication
> of inferential statistics to a census. It implied an estimate where none was being made,
> and — since the resulting interval (36.1–69.8) contains 48.1% — it would have let a
> reader dismiss an exact result as noise. The interval is removed, not recomputed.

### 3.2 The mechanistic claim — inferential, and NOT established

> Arm-b detects flagged Subtle cases more reliably (26/47, 55.3%) than authentic ones
> (14/30, 46.7%).

This *is* a generalization beyond the test set: it asserts something about how the model
behaves on a kind of input. It therefore carries inferential uncertainty, and the
uncertainty is fatal to it:

| Test | Value |
|---|---|
| Fisher's exact, two-sided, 26/47 vs 14/30 | **p = 0.491** (OR 1.42) |
| Gap | 8.7 pp |
| 95% Wilson CI, flagged caught | 41.2 – 68.6% |
| 95% Wilson CI, authentic caught | 30.2 – 63.9% |

Computed 2026-07-25. The gap is **not distinguishable from chance**. Report this as a
candidate mechanism that motivates future work — it is the intuitive explanation for why
§3.1 happens, and omitting it would leave the census result unmotivated — but never as a
demonstrated effect.

**Why the split matters.** §3.1 is the finding; §3.2 is the explanation. Keeping them in
separate registers is what stops the explanation's weak evidence from dragging down the
finding's exact status.

## 4. Reading for the paper

The audit's original point stands in corrected form: a meaningful share of what is scored
as a "Subtle miss" is content RAGTruth marks as ungrounded-but-true, so the 48.1% figure
is not cleanly interpretable as a pure capability gap in either direction. What this
analysis adds is the **direction** of the effect on the headline number: it makes arm-b
look *better* at Subtle detection than it is on authentic cases, not worse.

Two paraphrases that are **not supported** by this data and must not be used:

- *"The Subtle weakness is partly a labeling artifact that inflates the true miss rate."*
  The measured direction is the opposite.
- *"The model is better at ungrounded-but-true cases than at authentic ones."* True of
  this cohort as counted, but not established as a property of the model (§3.2).

For the write-up's structure, this result is a **subordinate case study** under the
evaluation-side leg, scoped to this model — not a second headline. Its reach is limited by
the fact that no published system reports a Subtle-stratified miss rate, so the
distortion it demonstrates lands on a metric only this project reports. The
aggregate-bound result (§3 of the audit) is the claim with field-wide reach.

---

## Provenance

| Claim | Source |
|---|---|
| n=77 Subtle-only cohort, 47/77 (61%) all-`implicit_true` | `scripts/subtle_only_miss_rate.py`, cross-checked against `02-implicit-true-audit.md` §2.3 |
| 48.1% / 53.3% / 44.7% miss rates | `scripts/subtle_only_miss_rate.py`, run against `results/token_preds_arm_b.json` |
| 48.1% published headline figure | `docs/research/03-candidate-methods.md`, `docs/decisions.md` (ADR-020 addendum) |
| Join-back pattern (`load_merged_dataframe` → test split → `row_index` alignment) | `scripts/ablation_report.py::build_test_meta` / `load_arm` |
| Fisher's exact p = 0.491, Wilson intervals | Computed 2026-07-25 via `scipy.stats.fisher_exact` on the 2×2 table in §3.2 (not scripted into the repo) |
| Census-vs-inference framing; withdrawal of the ±18pp interval | ADR-022 correction addendum |
