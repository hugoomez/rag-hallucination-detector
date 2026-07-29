# Experiment ledger

Every experiment actually run in this project, in the order it was run. One row per run,
with the exact artifact path so any number in the README, the model cards, or the ADRs
can be traced back to the file it came from.

**Conventions**

- All test metrics are on RAGTruth's official test split, n = 2,700 (943 hallucinated /
  1,757 faithful). Positive class = hallucinated.
- `val` is a 10% group-stratified carve-out of the official train split, grouped by
  `source_id` (ADR-005). Response-level pipelines report 1,511 val rows (one outlier
  dropped, ADR-006); token-level and ModernBERT pipelines report 1,512.
- "Response F1" for token-level models is *derived*: a response is predicted hallucinated
  iff any response token crosses P(hallucinated) ≥ 0.5.
- "Span F1" is character-overlap micro-F1 (`char_span_prf`), matching LettuceDetect's
  span metric — not strict entity match.
- Paths are repo-relative. `⚠` marks an artifact that is **not** committed.

---

## The table

| ID | Experiment | What was varied | Seed(s) | Headline test result | Results file | ADR |
|---|---|---|---|---|---|---|
| **E0** | Trivial baselines | — (analytic / fixed rule) | n/a | Always-hallucinated F1 0.5177; random-50/50 F1 0.4107 | embedded in every `results/*_metrics.json` | — |
| **E1** | Zero-shot NLI baseline | Aggregation rule + two decision thresholds (`ent_thr`, `con_thr`) over a 12-point val grid; no training | n/a (frozen `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`) | **F1 0.5234** (P 0.3547 / R 0.9979). Best thresholds ent 0.4 / con 0.4 | `results/baseline_nli_metrics.json`; per-sentence scores in `results/nli_scores_{val,test}.json` | ADR-007, ADR-008, ADR-009, ADR-015 |
| **E1b** | Baseline flagging diagnostic | Re-analysis of E1's cached val scores; proportion-based vs any-sentence aggregation | n/a | Aggregation ruled out as the cause (val F1 0.611 → 0.632); contradiction flag fires on 55.7% of faithful vs 53.8% of hallucinated sentences | no artifact — `scripts/diagnose_baseline_flagging.py` recomputes from `results/nli_scores_val.json` | ADR-009 |
| **E2** | **Track A** — DeBERTa-v3-base, response-level | Fine-tuning vs zero-shot. lr 2e-5, batch 16, 5 epochs, class weights [0.9028, 1.1207], max_len 512, context-only truncation | 42 | **F1 0.7116** (P 0.7367 / R 0.6882 / acc 0.8052) | `results/finetuned_track_a_metrics.json` | ADR-001, ADR-004, ADR-005, ADR-006 |
| **E2b** | Truncation-correlation diagnostic | Correlating `was_truncated` with per-row correctness on E2's predictions | n/a | Truncation cost is precision-driven, not recall-driven; Summary's weakness is *not* explained by truncation | `results/track_a_test_predictions.json` (+ `scripts/analyze_track_a_predictions.py`) | ADR-010 |
| **E3** | **Approach 1** — ModernBERT-base, response-level | Backbone + context length only (same recipe family as E2). lr 2e-5, per-device batch 4 × accum 4 = 16, 5 epochs, same class weights, max_len 4096 | 42 | **F1 0.7254** (P 0.6839 / R 0.7731 / acc 0.7959). Truncation 0.00%; Summary recall 0.245 → 0.569 | `results/finetuned_approach1_modernbert_metrics.json` | ADR-011, ADR-012 |
| **E4** | **Track B v0** — ModernBERT-base, 3-class BIO token labels | Task reframed to token classification. 5 epochs, inverse-frequency class weights [0.34, 95.24, 12.53] | 42 | **Failed**: seqeval exact-entity span F1 **0.037**; token recall ~0.12–0.14 with precision 0.007–0.025 | ⚠ **no artifact committed** — numbers survive only in ADR-013 | ADR-013 |
| **E5** | **Track B v1 (arm a)** — binary token labels | vs E4: binary labels (no B-/I-), plain unweighted CE, char-overlap span metric, 8 epochs. lr 2e-5, batch 4 × accum 4 = 16, checkpoint selection on **response** F1 | 42 | Span F1 **0.5113**, response **F1 0.7611** (P 0.7856 / R 0.7381 / acc 0.8381) | `results/arm_a_original_metrics.json` — recovered from commit `b54604d` after E6 overwrote the shared filename at `29f6b03`. Per-row preds (`token_preds_arm_a.json`) ⚠ not committed | ADR-013, ADR-014 |
| **E6** | **ACWS ablation, arm b** — LettuceDetect-recipe replication | vs E5, four axes at once: lr 1e-5, batch 4 × accum 2 = 8, 6 epochs, checkpoint selection on **token** F1. λ = 1.0 (no ACWS) | 42 | **Span F1 0.5321** (+2.1 over arm a), **response F1 0.7631** (P 0.8359 / R 0.7020 / acc 0.8478). FP rate on faithful 7.4% vs 10.7%. **Adopted as production Track B** | `results/arm_b_metrics.json` (identical copy now at `results/finetuned_track_b_token_level_metrics.json`); per-row preds `results/token_preds_arm_b.json` | ADR-020 |
| **E7** | **ACWS ablation, arm c** — arm b + ACWS | vs E6, exactly one thing: `--implicit_true_weight` 1.0 → **0.25** | 42 | **Rejected** by the pre-registered rule: clean-span F1 0.5262 vs arm b's 0.5307 (worse); response F1 0.7633 vs 0.7631 | ⚠ **no artifact exists** — see [the arm-c note](#e7-arm-c-has-no-artifact-and-cite-it-accordingly) | ADR-020 |
| **E8** | Threshold tuning | Global decision threshold, and per-task thresholds, tuned on val and applied to test once | n/a (inference on E5's model) | **Did not generalise.** Global 0.45: test F1 0.7609 vs untuned-0.5's 0.7619. Per-task (Summary 0.15 / QA 0.45 / Data2txt 0.65): 0.7462 | `results/threshold_ensemble_tuning.json` → `threshold_tuning` | ADR-017 |
| **E9** | 3-system ensemble | Weighted score fusion of `baseline_nli` (0.55), `approach_1_modernbert` (0.25), `track_b_modernbert` (0.20), threshold 0.75; `track_a_deberta` excluded for val-split misalignment | n/a (inference on E1/E3/E5) | **Generalised.** Val F1 0.7997 → **test F1 0.7701** (P 0.7446 / R 0.7975), +0.82 over Track B alone. Not deployed (3 models per prediction) | `results/threshold_ensemble_tuning.json` → `ensemble`; inputs in `results/unified_predictions.parquet` | ADR-017 |
| **E10** | Multi-seed **ModernBERT-base** (arm-b recipe) | Seed only — but see the caveat below on the epoch cap | **42, 123, 456** | Response F1 mean **0.7637** (range 0.7631–0.7643); span F1 mean **0.5325** (range 0.5321–0.5327) | seed 42 = `results/arm_b_metrics.json`; `results/base_seed123_metrics.json`; `results/base_seed456_metrics.json`. Per-row preds: `results/token_preds_arm_b.json`, `results/base_seed123_preds.json`, `results/base_seed456_preds.json` | ADR-021 |
| **E11** | Multi-seed **ModernBERT-large** | Backbone size (base → large), matched by seed to E10. lr 1e-5, batch 2 × accum 4 = 8, 4 epochs, token-F1 selection | **42, 123, 456** | Response F1 mean **0.7948** (range 0.7930–0.7965); span F1 mean **0.5733** (range 0.5705–0.5767) | seed 42 `results/modernbert_large_metrics.json`; `results/large_seed123_metrics.json`; `results/large_seed456_metrics.json`. Per-row preds: `results/token_preds_modernbert_large.json` (seed 42), `results/large_seed456_preds.json`; ⚠ **seed-123 preds not dumped** | ADR-021 |
| **E12** | Paired base-vs-large aggregation | Matched-seed differencing of E10 vs E11 | 42, 123, 456 | Large wins on **3/3** seeds: response F1 **+0.0311** mean (+0.0292 … +0.0323), response recall **+0.0516** mean, span F1 **+0.0408** mean. Largest per-task gain on Summary (+0.0931, 3/3). Precision is the one metric where base wins a seed (2/3) | `results/seed_aggregate.json` (written by `scripts/aggregate_seeds.py`) | ADR-021 |

---

## Caveats that the table cannot carry

**E10's third variable.** The three base seeds are not a pure seed sweep. Seed 42
(= arm b) was configured with `num_train_epochs=6`; seeds 123 and 456 with `4`. Best
checkpoints landed at epochs 3, 4 and 3 respectively — so seed 123 selected its
*final* epoch and may have kept improving under a 6-epoch cap. The observed spread is
tiny (response-F1 range 0.0012), so this is unlikely to change any conclusion, but the
sweep should be described as "seed, with a differing epoch cap on one arm", not
"seed only".

**E11's seed 123 was scored, not trained, by the same code path.**
`results/large_seed123_metrics.json` carries only `{"seed": 123}` under
`hyperparameters` and has no `val` block — it was produced by the existing-model
inference path added in commit `d5d9598`, not by a fresh training run's reporting. Its
full hyperparameters are therefore not recorded anywhere. The `model_name` field in that
file also needed a post-hoc correction (commit `5b2f2b9`).

**E5's headline file was overwritten — now recovered.**
`results/finetuned_track_b_token_level_metrics.json` holds arm b's numbers, not arm a's;
commit `29f6b03` overwrote it in place. Arm a's original report has been restored from
commit `b54604d` as `results/arm_a_original_metrics.json`, so the 0.5113 / 0.7611 figures
have a live artifact again rather than existing only in git history. Two consequences
worth knowing: `results/arm_b_metrics.json` and
`results/finetuned_track_b_token_level_metrics.json` are now **byte-identical duplicates**
of arm b (verified by hash), and the latter's name no longer says which arm it holds —
cite `arm_a_original_metrics.json` / `arm_b_metrics.json` by preference.

**Arm a has two slightly different response-level numbers, and both are correct.** The
training-run report gives F1 0.7611 / P 0.7856 (`arm_a_original_metrics.json`); the
unified cross-system collection gives F1 0.7619 / P 0.7873
(`results/threshold_ensemble_tuning.json`, `results/unified_predictions.parquet`, and
ADR-020's Gate-4 check). Same model, two evaluation paths — the same ~0.0004-scale
inference nondeterminism already documented in `docs/notes.md` (Phase 4 note). The
ensemble and threshold experiments (E8/E9) were run against the unified figures, so
those are the ones to quote alongside them.

### E7: arm c has no artifact, and cite it accordingly

Arm c ran and produced the numbers ADR-020 reports, but **nothing was written to disk and
committed** — no `token_preds_arm_c.json`, no `ablation_report.json`. Stated plainly:

- **Arm c's numbers are not independently re-derivable.** They cannot be recomputed,
  re-scored under a different metric, or re-stratified. Reproducing them requires
  **retraining the arm** (ModernBERT-base, arm-b recipe, `--implicit_true_weight 0.25`,
  seed 42).
- **ADR-020 is the citable source for them**, and it is a legitimate one: a committed,
  version-controlled document written at the time of the experiment, under this project's
  pre-registered decision rule. It is contemporaneous evidence, not a recollection.
- **A write-up must not imply a JSON artifact backs those numbers.** Where arm-c figures
  appear (clean-span F1 0.5262, response F1 0.7633), cite ADR-020 directly. Every other
  arm in this ledger resolves to a metrics file; arm c resolves to prose, and the
  difference should be visible to the reader rather than smoothed over by a uniform
  citation style.

**Nothing after ADR-020 was documented as a decision — now closed.** E10–E12 are covered
by **ADR-021**, which also closes ADR-020's forward reference. Two follow-ups remain open
and are recorded there rather than claimed as done: the seed-42 large checkpoint is **not
published to the Hub** (no `hugoomezz/*-large` repo exists), and **large seed 123 has no
prediction dump**, so stratified re-analysis covers only 2 of the 3 large seeds.

**Two comparisons in the README still reference superseded predictions.** E9's ensemble
was tuned and measured against E5 (arm a), not E6 (arm b), and the README's qualitative
error-analysis examples and live-demo transcripts were produced against arm-a weights.
Both are flagged as visible TODOs in the README and the Track B model card rather than
silently reconciled.

---

## Not experiments

For completeness, these produced committed artifacts but are not runs:

| Artifact | What it is |
|---|---|
| `results/unified_predictions.parquet` | Per-row test (and val, where available) predictions for all four systems on identical footing, 16,847 rows — the join table behind the README's comparison table. Written by `scripts/collect_predictions.py` |
| `results/confusion_matrix_*.png`, `results/pr_curve_comparison.png` | Evaluation plots (`scripts/generate_evaluation_plots.py`) |
| `results/eda_*.png` | Phase 1 EDA charts (`notebooks/01_eda_ragtruth.ipynb`) |
| `results/demo_*.png` | Phase 6 demo screenshots, captured against the live system |
| `docs/research/04-subtle-only-reconciliation.md` | Reconciliation of the `implicit_true` audit (E-none, `02-implicit-true-audit.md`) with arm-b's (E6) Subtle-only miss rate: splits the n=77 Subtle-only test cohort into all-`implicit_true` (n=47) vs. genuinely-Subtle (n=30) subsets using `results/token_preds_arm_b.json`. Finding: excluding the all-`implicit_true` subset raises the miss rate (48.1% → 53.3%), not lowers it — the ungrounded-but-true subclass dilutes the headline figure toward a better-looking number rather than inflating it. Written by `scripts/subtle_only_miss_rate.py`. See ADR-022 |
