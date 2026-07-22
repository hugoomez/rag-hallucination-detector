# Architecture Decision Records (ADR)

Log of key technical decisions made throughout the project, with their rationale.

---

## ADR-001: Use DeBERTa-v3-base as the base encoder

**Context:** Need to choose a pretrained transformer encoder to fine-tune for hallucination detection, framed as an NLI-style classification task (premise = retrieved context, hypothesis = answer claim).

**Decision:** Use `microsoft/deberta-v3-base`.

**Rationale:**

- He et al. (2021) show DeBERTa consistently outperforms BERT, RoBERTa, and XLNet specifically on MNLI (the NLI benchmark), not just on average across tasks.
- DeBERTa's disentangled attention mechanism separates content and position representations, which helps capture syntactic nuances (subject vs. object, negation) relevant to detecting contradiction vs. entailment.
- v3 (vs. v1) uses ELECTRA-style Replaced Token Detection pretraining instead of MLM, giving a training signal on 100% of tokens instead of only the masked 15% — a stronger pretrained model at the same size.
- `base` size runs comfortably on a free Colab T4 GPU (16GB), no paid compute required.

**Alternatives considered:** RoBERTa-base (weaker on MNLI per the DeBERTa paper).

---

## ADR-002: Train on Google Colab (free tier)

**Context:** Need to decide where to run training.

**Decision:** Google Colab, free T4 GPU tier.

**Rationale:** Sufficient VRAM (16GB) for DeBERTa-v3-base fine-tuning; no cost; no local GPU available. Kaggle Notebooks considered as a backup option if Colab quota runs out.

---

## ADR-003: Repository language — English

**Context:** The project is a public portfolio piece.

**Decision:** All code, comments, commit messages, and documentation in English; working conversations with the AI copilot in Spanish.

**Rationale:** English is the industry standard for public repos reviewed by recruiters and other engineers.

---

## ADR-004: Long-context truncation strategy for DeBERTa-v3 input

**Context:** EDA on RAGTruth (Phase 1) showed that 70.34% of rows exceed DeBERTa-v3's 512-token limit when concatenating context + response + special tokens. This is highly non-uniform across task types: QA 34.31% exceed, Summary 75.77%, Data2txt 99.89%. Naive truncation also disproportionately harms hallucinated rows (50.52% of truncated rows are hallucinated vs. 43.08% globally), since evidence needed to verify faithfulness is lost. Responses are almost always under 512 tokens alone (mean 160), so the response should never be the one truncated. Extended research surveyed the RAGTruth leaderboard (Luna, LettuceDetect, RAGTruth/RAG-HAT baselines), truncation techniques (Sun et al. 2019 head/tail/head+tail), long-context encoders (ModernBERT, Longformer, BigBird), and claim-decomposition + retrieval approaches (AlignScore, MiniCheck, RefChecker).

**Decision:** Phased approach across three stages:

1. **MVP (now):** Keep DeBERTa-v3-base (per ADR-001). Truncate the CONTEXT only, always preserving the full response + question. Train/evaluate separately by `task_type` (QA / Summary / Data2txt) to quantify the real cost of truncation per task before solving it further. This is the honest baseline against which later gains are measured.
2. **Approach 1 (next):** Reproduce the LettuceDetect recipe by switching the backbone to ModernBERT-base (149M params, native 8,192-token context, fits the free Colab T4 with `attn_implementation="sdpa"` since FlashAttention 2 is unsupported on Turing GPUs). Token-classification head, ~4,096-token inputs, non-response tokens masked with -100. This eliminates ~99% of the truncation problem with comparatively low engineering effort, matching current encoder SOTA on RAGTruth (~76-79% F1).
3. **Approach 3 (advanced/capstone):** Claim/sentence decomposition + retrieval + NLI, AlignScore/MiniCheck-style: split the response into sentences, retrieve top-k relevant context chunks per sentence with a small bi-encoder, run the fine-tuned DeBERTa-v3 NLI check per (chunk, sentence) pair, aggregate (max-entailment per sentence). Sidesteps the token limit entirely by design and is the most system-design-mature of the three approaches considered — the intended flagship deliverable of the project.

**Alternatives considered:**

- Sliding-window DeBERTa-v3 (Luna-style, per-window label propagation + max-support aggregation): valid and demonstrates deep mastery of the ADR-001 constraint, but higher implementation complexity than Approach 1 for comparable gains. Deprioritized in favor of Approach 1 → Approach 3, revisited only if Approach 1's results are unsatisfactory.
- Head+tail truncation (Sun et al. 2019): rejected as a long-term fix — its motivating assumption (salient info clusters at document start/end) does not hold for RAGTruth's Data2txt (structured data) or Summary (evidence scattered throughout).

**Status:** MVP shipped (Phase 1-2, Track A). Approach 1 shipped — see ADR-011 (truncation eliminated on ModernBERT) and ADR-012 (results). Approach 3 (claim decomposition + retrieval) was never built: once Track B (binary token-level classification, ADR-013/ADR-014) matched LettuceDetect and became the project's best-performing model, it superseded Approach 3 as the flagship deliverable — Track B already delivers exact-span granularity on truncation-free ModernBERT input, which was Approach 3's main motivation, at lower implementation cost. This entry is kept as the original decision trail, not as a live roadmap.

---

## ADR-005: Group-stratified train/val split by source_id

**Context:** RAGTruth has exactly 6 responses per source_id (one per generating
model). A naive row-level stratified split for the train/val carve-out risks
placing sibling responses of the same source_id in both train and val, which
would fail the no-leakage requirement even though each row is nominally
"different data."

**Decision:** Group rows by source_id first, then stratify on each group's
majority label_response, assigning whole groups to train or val. This makes
the no-leakage guarantee structural (by construction) rather than something
only caught by a post-hoc assertion.

**Status:** Implemented in `src/data/preprocess.py`.

---

## ADR-006: Excluding the response-token-overflow outlier

**Context:** One row (source_id 11845, task_type Summary) has a response that
alone tokenizes to 770 tokens — exceeding the 512-token budget even with zero
context tokens reserved. This breaks the ADR-004 guarantee that the response
is never truncated.

**Decision:** Drop this row from the dataset entirely (from whichever split it
falls into) rather than breaking the "never truncate response" rule for this
one case. A permanent assertion (`len(input_ids) <= max_length`) was also
added as a safety net in case similar rows appear in future data.

**Status:** Implemented in `src/data/preprocess.py`. 1 row dropped (from train,
which affected val after the split, since the group-stratified split had
assigned that source_id's group to val).

---

## ADR-007: Independent max-entailment/max-contradiction aggregation in the NLI baseline

**Context:** When scoring a response sentence against multiple context sentences,
each comparison produces a softmax triple (entailment, neutral, contradiction)
that sums to 1. Naively keeping the full triple from whichever context sentence
had the highest entailment score would almost always fail to detect real
contradictions: a sentence with high entailment structurally tends to have low
contradiction (softmax constraint), while the actual contradicting evidence is
typically a DIFFERENT context sentence entirely.

**Decision:** Track max entailment and max contradiction independently across
context sentences (potentially from two different sentences), rather than
taking one sentence's full triple. Flag priority: contradicted → supported →
unverifiable. The resulting (entailment, contradiction) pair does not sum to 1
with an implied neutral value — it represents two independent signals, not one
sentence's complete output — but this is necessary for the aggregation to
actually surface contradictions found anywhere in the context.

**Alternatives considered:** Using only the best-entailment sentence's full
triple (simpler, matches a naive reading of AlignScore/MiniCheck's "take the
max" approach) — rejected because it would systematically under-detect
contradictions due to the softmax constraint described above.

**Status:** Implemented in src/models/nli_baseline.py, verified via a real-model
smoke test (contradicting date claim correctly flagged despite low entailment
from the best-supporting context sentence).

---

## ADR-008: Task-type-aware context chunking for the NLI baseline

**Context:** A diagnostic on 5 rows per task_type revealed that generic nltk
sentence-tokenization of the normalized "context" string is only valid for Summary
(real prose, ~28 clean sentences). For QA, the "Question: ... Passages: ..." format
produces a single undecomposed unit (no real chunking benefit). For Data2txt, nltk
splits raw JSON syntax as if it were prose, producing semantically meaningless
premise fragments (e.g., a chunk of "{\"name\": \"...").

**Decision:** Chunk context based on task_type, operating on the RAW source_info
structure (before Phase 1's flattening into a single string), not on the flattened
string itself:
- Summary: nltk.sent_tokenize on the raw text (unchanged, already correct).
- QA: split into individual retrieved passages first (preserving the original
  passage boundaries from source_info), then sentence-tokenize within each passage.
- Data2txt: chunk by structured field (one chunk per scalar key-value pair);
  for list-valued fields containing natural-language text (e.g., reviews),
  sentence-tokenize each entry individually rather than JSON-dumping the whole dict.

Implementation note: real Data2txt rows also contain nested dicts (`hours`,
`attributes`, incl. the `BusinessParking: null` evidence) and lists of dicts
(`review_info`), which the summary above doesn't call out. The implementation recurses
into nested dicts (each leaf scalar becomes a `"{key}: {value}"` chunk) and, for
list-of-dict entries, sentence-tokenizes string fields (e.g. `review_text`) while
emitting `"{key}: {value}"` for numeric fields. Values render JSON-style
(`null`/`true`/`false`); no branch ever serializes a container, so JSON syntax never
leaks into a chunk.

**Status:** Implemented in `src/data/context_chunking.py`
(`chunk_context(task_type, source_info)`), decoupled from `NLIHallucinationDetector`,
which now takes pre-chunked `context_chunks`. Verified on real RAGTruth rows: Summary
28 prose chunks, QA passage-level chunks (question excluded), Data2txt 50 field/prose
chunks with zero JSON-syntax leaks.

---

## ADR-009: Zero-shot NLI baseline's aggregation-independent failure mode

**Context:** The zero-shot NLI baseline (F1=0.523 on test) barely outperformed the
"always hallucinated" trivial baseline (F1=0.518). A diagnostic on cached val scores
(scripts/diagnose_baseline_flagging.py) found the root cause is NOT the aggregation
rule ("any sentence not-supported → hallucinated"), but poor calibration of the raw
per-sentence NLI scores themselves:
- Among genuinely faithful (label_response=0) sentences, the contradicted flag fires
  on 55.7% of them, nearly identical to the 53.8% rate among hallucinated sentences
  — meaning contradiction carries almost no discriminative signal.
- Median max_entailment for faithful sentences is only 0.169 (25th percentile: 0.030)
  — most truly-supported claims score low on entailment against any single context
  chunk, likely because faithful responses often synthesize information across
  multiple chunks, which no single-chunk comparison can fully capture.
- Switching from "any not-supported" to a proportion-based rule (e.g., "hallucinated
  if >75% of sentences are not-supported") only improved F1 marginally (0.611 to
  0.632 on val), confirming the problem is in the underlying scores, not how they're
  aggregated.

**Decision:** Report this baseline's result along with this diagnosed failure mode
in the Phase 2 README section, rather than tuning the aggregation rule further to
cosmetically improve the number. This finding is treated as empirical justification
for proceeding with the fine-tuned approaches (ADR-004's Approach 1/3): a model
fine-tuned on RAGTruth should learn domain-appropriate support/contradiction
calibration that a generic zero-shot NLI model checking isolated sentence-chunk
pairs cannot achieve.

**Status:** Documented. scripts/diagnose_baseline_flagging.py kept in the repo as
a reusable diagnostic for future baseline/model comparisons.

---

## ADR-010: Empirical truncation impact on Track A is precision-driven, not recall-driven

**Context:** ADR-004 hypothesized that context truncation would primarily harm recall
(the model missing hallucinations because supporting/contradicting evidence gets cut
off). A diagnostic on the fine-tuned Track A model's test predictions
(scripts/analyze_track_a_predictions.py), correlating was_truncated with per-row
correctness, tested this directly.

**Finding:** The opposite pattern was observed. Truncated rows have HIGHER recall on
hallucinated examples than untruncated rows, in both task types where the comparison
is possible (Summary: 0.278 truncated vs. 0.151 untruncated; QA: 0.750 truncated vs.
0.576 untruncated). However, truncated rows have LOWER overall accuracy (0.778 vs.
0.859), implying the truncation cost is concentrated in PRECISION (more false
positives — faithful responses flagged as hallucinated) rather than recall. A
plausible mechanism: when the model has less context to confirm a claim is
supported, it appears biased toward predicting "hallucinated" rather than "faithful"
under uncertainty, likely reinforced by the mildly hallucination-favoring class
weights ([0.90, 1.12]) used in training.

A separate implication: Summary's low overall recall (0.245, the model's weakest
metric) is NOT well explained by truncation — both truncated (0.278) and untruncated
(0.151) Summary rows show similarly poor recall, with the untruncated subset actually
worse. This suggests Summary's weakness has a different primary cause than context
truncation, possibly the prevalence of "subtle" hallucination types (RAGTruth's
rarest label category) which may be inherently harder to detect regardless of
context completeness.

**Decision:** ADR-004's roadmap (Approach 1: ModernBERT, Approach 3: claim
decomposition + retrieval) remains justified — truncation does measurably hurt
accuracy via precision, and eliminating it is still expected to help. However, the
original framing ("truncation causes missed hallucinations") is corrected to
"truncation causes over-flagging of faithful content." This changes what we should
watch for when evaluating Approach 1/3: expect precision gains on truncation-heavy
task types (Data2txt, Summary) rather than assuming recall will be the primary
metric that improves. Separately, Summary's recall weakness should be treated as a
partially independent problem, worth investigating on its own terms (e.g., checking
performance specifically on "subtle" vs. "evident" hallucination sub-types) rather
than assumed to be fully solved by a longer-context backbone alone.

**Status:** Documented. Informs Phase 4's evaluation design and Approach 1/3
expectations.

---

## ADR-011: ModernBERT eliminates truncation entirely on RAGTruth

**Context:** ADR-004 hypothesized that switching to a long-context encoder
(ModernBERT, 8192 native context, used here at max_length=4096) would substantially
reduce or eliminate the truncation problem quantified in Phase 1's EDA (70.34% of
rows exceeded DeBERTa-v3's 512-token limit).

**Finding:** Confirmed directly. At max_length=4096, 0.00% of rows require any
truncation across all three task_types (Summary, QA, Data2txt) and all three splits
(train/val/test) — verified both by an independent pre-truncation diagnostic
(report_combined_length_exceedance, max observed combined length: 2618 tokens) and
by the actual was_truncated flag computed during real tokenization. The single
response-length outlier excluded in the DeBERTa pipeline (Phase 1, ADR-006) did not
need exclusion here, as its 770 tokens fit comfortably within the 4096 budget.

**Decision:** Proceed to train a response-level classifier
(src/models/train_modernbert.py) on this truncation-free data, to directly test
ADR-010's hypothesis: does eliminating truncation reduce the false-positive
(precision) cost previously observed on truncated rows, particularly for Data2txt
and Summary?

**Status:** Data pipeline complete and verified. Training pending.

---

## ADR-012: ModernBERT Approach 1 results — recall-driven improvement, not precision-driven

**Context:** ADR-010 hypothesized that eliminating context truncation (via ADR-011's
ModernBERT pipeline) would primarily improve PRECISION, based on the observation
that DeBERTa Track A's truncated rows had lower accuracy driven by more false
positives under uncertainty.

**Finding:** The real controlled comparison (same task, same training recipe,
different backbone/context length) showed a different mechanism than predicted.
Overall test F1 improved (0.7116 → 0.7257), but PRECISION actually decreased
slightly (0.7367 → 0.6839) while RECALL improved substantially (0.6882 → 0.7731).
The improvement is heavily concentrated in Summary, where recall more than doubled
(0.245 → 0.569, +0.324) and F1 rose from 0.332 to 0.509 — the task type with the
longest, most dispersed evidence requirements, and the one ADR-010 flagged as having
a recall weakness NOT well-explained by truncation status alone under the 512-token
architecture. QA and Data2txt saw only marginal changes (already performing well
under truncation or already resilient to it).

**Interpretation:** ADR-010's within-architecture "truncated vs. untruncated rows"
comparison did not fully predict the effect of an across-architecture, truncation-
free redesign. The mechanism that actually improved was the model's ability to
locate evidence scattered across a full long document (raising recall), rather than
reduced false-positive behavior under partial-context uncertainty (which would have
raised precision). This is a useful methodological lesson: correlational diagnostics
on a fixed architecture (ADR-010) do not necessarily predict the causal effect of
changing that architecture (ADR-012) — both findings are valid but answer different
questions.

**Decision:** Approach 1 (ModernBERT, response-level) is adopted as the stronger
response-level model going forward, given its clear overall F1 gain and the
resolution of Summary's severe recall weakness. Track B (token-level span
detection) should be pursued next on this ModernBERT backbone rather than DeBERTa,
both because it is now the stronger base model and because it matches the actual
LettuceDetect SOTA recipe referenced throughout this project's research.

**Status:** Response-level comparison complete. Track B (token-level, ModernBERT)
planned next.

---

## ADR-013: Pivot Track B from 3-class BIO to binary token labels (LettuceDetect parity)

**Context:** Track B's first training run (5 epochs, BIO scheme O/B-HALL/I-HALL,
inverse-frequency class weights [0.34, 95.24, 12.53]) produced a very low
span-level seqeval F1 (0.037), though it was still monotonically improving with
no plateau. A code review + SOTA comparison against LettuceDetect
(arXiv:2502.17125) found two compounding issues: (1) LettuceDetect uses BINARY
token labels (supported/hallucinated), not BIO — our 3-class scheme creates an
ultra-rare B-HALL class (0.35% of tokens); (2) the resulting 95x inverse-frequency
weight on B-HALL actively rewards fragmenting real spans into many separate
B-HALL predictions (each scored as a failed entity by seqeval's exact-match
scoring), which is consistent with the observed precision collapse
(0.007-0.025) despite reasonable token-level recall (~0.12-0.14). Additionally,
our reported metric (strict seqeval exact-entity F1) was being compared against
LettuceDetect's EXAMPLE-level F1 (76-79%), not their actual span-level metric
(55.4-58.9%, computed via character-overlap, a more lenient standard than exact
match) — an unfair comparison independent of the modeling issue.

**Decision:** Redesign Track B to match LettuceDetect's actual recipe:
- Binary token labels (0=supported, 1=hallucinated) instead of BIO, eliminating
  the ultra-rare B-HALL class entirely.
- No class weighting by default (matching LettuceDetect's plain cross-entropy),
  with capped weighting (max ~10x) as a fallback only if recall proves poor.
- Spans reconstructed at inference by merging consecutive positive-token
  predictions (threshold 0.5), moving boundary handling to post-processing
  instead of the training objective.
- Report BOTH the derived response-level F1 (already implemented, comparable to
  LettuceDetect's 76-79% example-level F1) and a new character-overlap span F1
  (comparable to their 55.4-58.9% span-level F1) — retire strict seqeval
  exact-match as the headline metric, keep it only as a secondary strict measure.
- Train for 6-8 epochs (vs. the prior run's 5, which was still improving when
  capped), with checkpoint selection on the derived response-level F1 rather
  than the previously noisy/degenerate span metric.

**Alternatives considered:** Simply training the existing BIO scheme longer —
rejected as the primary fix, since the weighting issue (not epoch count alone)
was diagnosed as actively working against correct span formation; more epochs
alone would not resolve the fragmentation incentive.

**Status:** Redesign in progress.

---

## ADR-014: ADR-013's binary-scheme pivot validated — Track B matches LettuceDetect

**Context:** ADR-013 pivoted Track B from 3-class BIO (which produced a
pathological 0.037 span-F1, diagnosed as caused by an extreme 95x class weight
on the ultra-rare B-HALL token fragmenting real spans) to binary token
classification, unweighted loss, char-overlap span evaluation, and 8 epochs
(up from 5), matching LettuceDetect's actual recipe (arXiv:2502.17125).

**Finding:** The redesign succeeded, closely matching published LettuceDetect-base
numbers on the same benchmark:
- Response/example-level F1: 76.11% (ours) vs. 76.07% (LettuceDetect-base
  published) — a 0.04-point difference, functionally equivalent.
- Span-level (char-overlap) F1: 51.13% (ours) vs. 55.44% (LettuceDetect-base
  published) — close, somewhat below, plausibly explainable by their A100
  training vs. our T4-constrained batch size/epochs, or minor implementation
  differences in span merging.
- This also makes Track B the best-performing system in this project at
  response level, surpassing Track A (0.7116) and Approach 1 (0.7257) despite
  being trained on a harder, more granular objective (exact token spans) than
  either.

**Interpretation:** This confirms the code review's diagnosis (ADR-013) was
correct: the original BIO scheme's near-total failure was caused by the
training recipe (weighting + label granularity), not by ModernBERT, the
truncation-free data pipeline (ADR-011), or the task's inherent difficulty.
Matching the actual SOTA recipe rather than a plausible-sounding but
independently-designed BIO variant was the key correction.

**Status:** Complete. Track B (binary token classification) adopted as the
project's best-performing model. Model published at
hugoomezz/modernbert-ragtruth-token-level-binary.

---

## ADR-015: Baseline y_score = max over sentences of max(contradiction, 1 - entailment)

**Context:** Phase 4's unified comparison needs one probability-like score per response
from every system. Fine-tuned classifiers provide softmax P(hallucinated) directly, but
the zero-shot NLI baseline produces per-sentence (max_entailment, max_contradiction)
pairs (ADR-007) plus a two-threshold decision rule — there is no single native score.

**Decision:** y_score = max over response sentences of max(contradiction, 1 - entailment).
The flag rule is a disjunction (not-supported iff contradiction >= con_thr OR
entailment < ent_thr), and this is its faithful one-dimensional reduction: sweeping a
single threshold t over the score walks the coupled rule family con_thr = t,
ent_thr = 1 - t, while preserving ADR-007's contradiction-priority (high contradiction
dominates the max even under high entailment). Empty responses score 0.0, matching the
vacuously-not-hallucinated convention.

**Alternatives considered:** max_contradiction alone (rejected: blind to "unverifiable"
hallucinations, which drive the baseline's 0.997 recall); 1 - min_entailment alone
(rejected: blind to ADR-007's core case of a claim entailed by one context sentence but
contradicted by another). Note the tuned operating point (ent_thr=0.4, con_thr=0.4) is
not on the coupled family, so the recorded y_pred (operational decision) is not exactly
a threshold on y_score; y_pred and y_score serve different purposes (point metrics vs
PR curves).

**Status:** Implemented in scripts/collect_predictions.py (baseline mode); verified by
reproducing results/baseline_nli_metrics.json test metrics from the unified table.

---

## ADR-016: No-context ablation mode for RAG demo hallucination demonstration

**Context:** Phase 5's Step 5A.5 requires demonstrating the detector catching
an induced hallucination in the live RAG pipeline. Live testing of the
grounded prompt (RAG_PROMPT, requiring the model to answer only from context
or explicitly refuse) across ~12 adversarial bait questions with
openai/gpt-oss-20b and qwen produced ZERO natural hallucinations — the model
either answered correctly from genuinely retrieved context, or used the
sanctioned refusal. This is evidence the prompt engineering succeeded, not a
pipeline failure — but it leaves no natural demonstration case.

**Decision:** Add an explicit, clearly-labeled ablation mode to RAGPipeline:
retrieve the REAL context via the retriever (so the detector has genuine
ground truth to check the answer against), but generate the answer WITHOUT
showing the model that context, forcing it to answer from parametric
knowledge alone. This creates an honest, real mismatch for the detector to
catch — the same principle used in hallucination red-teaming (deliberately
removing grounding to test a safety mechanism), not a fabricated or
cherry-picked example. The pipeline's output must clearly label when this
mode was used (e.g. an "ablation": true field), so it's never confused with
normal grounded operation.

**Alternatives considered:** Weakening the RAG_PROMPT's grounding instruction
— rejected, since it would undermine the very mechanism that makes normal
pipeline operation trustworthy, for the sake of an easier demo. Canned/
pre-recorded hallucinated examples from the RAGTruth test set — valid as a
supplementary backup, but doesn't demonstrate the LIVE pipeline catching a
real-time hallucination, which is Phase 5's specific goal.

**Status:** Implemented as `RAGPipeline.answer(question, no_context=True)` in
src/rag/pipeline.py: the retriever still runs for real (the detector checks
against genuine context), but `generate()` is called with the separate
NO_CONTEXT_PROMPT template instead of RAG_PROMPT, so the model never sees
that context. The returned dict carries `"ablation": true` when used.
Covered by hermetic tests in tests/test_pipeline.py.

---

## ADR-017: Threshold tuning failed to generalize; simple ensemble gave a real, modest win

**Context:** Following the research report's recommendations #1 (threshold
tuning) and #4 (simple ensemble), both were tuned strictly on val and applied
to test exactly once, per this project's established discipline.

**Findings:**
- Threshold tuning did NOT generalize. Global threshold tuned on val (0.45,
  val-optimal) scored F1=0.7609 on test, slightly WORSE than the untuned
  default (0.5, F1=0.7619). Per-task thresholds performed worse still
  (F1=0.7462), driven by Summary overfitting (best val F1 only 0.60 on a
  small ~500-row subset). Smaller, noisier subsets amplify the risk of
  fitting val-specific noise rather than a transferable pattern.
- The 3-system ensemble (baseline_nli, approach_1_modernbert,
  track_b_modernbert; track_a_deberta excluded due to val misalignment) DID
  generalize: val F1 0.7997 → test F1 0.7701, beating Track B alone (0.7619)
  by +0.82 points, mainly via improved recall on Summary and QA — directly
  addressing Phase 4's diagnosed weak spots.

**Decision:** Adopt the 3-system ensemble's exact weights/threshold as
documented in results/threshold_ensemble_tuning.json as an available
"best measured" configuration, reported alongside Track B alone in the
README comparison table. Do NOT adopt tuned thresholds (global or per-task)
as the new operating point for Track B — the untuned 0.5 default remains
the reported operating point for that model specifically.

**Status:** Complete. Whether the ensemble becomes the model that powers the
live RAG demo (Phase 5/6) is a separate decision (three models running per
prediction vs. one — complexity/latency trade-off), tracked separately.

---

## ADR-018: Skip public HF Spaces deployment in favor of local Docker reproduction

**Context:** Phase 6 originally planned a public Hugging Face Space (Gradio
SDK, CPU Basic hardware) per the phase document. During deployment, HF
confirmed (via a 402 Payment Required response and their own official docs)
that hosting a personal Gradio Space — on CPU Basic OR ZeroGPU — now
requires a PRO subscription ($9/month, recurring, confirmed not a one-time
fee). This is a recent platform change (corroborated by a matching HF forum
thread from the same week) not anticipated by the original phase document.

**Decision:** Do not pay for a recurring subscription to host a demo with
uncertain traffic value. Instead: package both api/main.py (FastAPI) and
app/app.py (Gradio) with Docker/docker-compose for one-command local
reproduction, and document clear setup instructions in the README, alongside
captured screenshots/GIF evidence of the working system (captured while the
implementation is fresh, rather than requiring a live demo to prove
functionality).

**Alternatives considered:**
- HF Pro subscription ($9/month recurring) — rejected as an ongoing cost
  disproportionate to expected demo traffic for a portfolio project.
- Self-hosting on Oracle Cloud's Always Free tier (VM, systemd, security
  groups) — rejected: Oracle's Always Free tier itself has no recurring
  fee, but self-managing it trades HF Pro's small recurring cost for a
  larger ongoing maintenance/security burden instead (patching, uptime, SSH
  key management), a poor trade for a project meant to showcase ML
  engineering, not infrastructure operations.

**Status:** Implemented via Docker packaging (see below).


---

## ADR-019: Replace the Gradio demo with a custom same-origin frontend

**Context:** The Phase 6 demo (app/app.py) is a Gradio app. It works, and its
callbacks and helpers are tested, but every Gradio app looks like every other
Gradio app. As the primary artifact a reader sees first, that is a real cost
for a portfolio project: the visual layer says nothing about what the system
does or how much care went into it. Two things blocked a custom UI. First,
api/main.py only wrapped Detector.predict() via POST /detect — the retriever,
RAGPipeline, Groq client and the demo_cache.json presets existed only inside
the Gradio process, so a browser had no way to reach live RAG. Second, nothing
in the repo served static files.

**Decision:** Build a standalone HTML/CSS/JS frontend under frontend/ and serve
it from the FastAPI process itself, mounted at "/" and registered after every
API route. Extend the API with POST /ask (live RAG, with the ADR-016 ablation
exposed as a real request field), GET /presets (the checked-in cache, needing
neither model nor key), and pipeline_loaded on /health. Keep the Gradio app as
a working fallback on :7860.

**Rationale:** Serving the frontend from the API process makes the browser's
fetch("/detect") same-origin by construction, so CORSMiddleware is never added
at all. A separate origin would have required a compose service, an
allow_origins list, and a build- or run-time mechanism to tell the JS where the
API lives — three pieces of configuration that can only ever be wrong, bought
for no benefit, since there is no CDN, no separate deploy target and no other
API consumer. It also costs almost nothing structurally: ADR-018 already builds
ONE image that compose runs two ways, and the `app` service never talked to
`api` over HTTP anyway (it loads the Detector in-process), so there was no
cross-origin call to preserve. docker-compose.yml is unchanged; the Dockerfile
gains one COPY line.

Route ordering is the one real hazard, since the mount is a catch-all at "/".
Starlette matches routes in registration order, so the mount must stay last in
api/main.py. That is verified rather than assumed: tests/test_api.py asserts
that /health, /presets, /docs and /openapi.json still resolve while "/" serves
index.html.

Two smaller decisions fell out of this. GenerationError gained a `kind`
attribute (rate_limit / connection / api_error / empty_completion) so /ask can
map failures to status codes (429 for a rate limit, 502 otherwise) and the
frontend can style them, without matching on message text; the messages
themselves are unchanged. And the three preset questions — previously
duplicated across app/app.py, app/precompute_cache.py and pipeline.main() in
three drifting shapes — moved to app/presets.py, which the API imports instead
of app/app.py (importing the latter would drag gradio into the API process).

**Design:** The frontend takes its identity from the subject rather than from a
template. The page is a near-monochrome cool grey, and the three traffic-light
colors are the only saturated things on it, so it stays silent until the
detector speaks — which is what integrates the traffic light into the design
instead of bolting an emoji onto a card. Three IBM Plex cuts are each reserved
for one meaning: condensed sans is the interface talking, serif is text the
model read or wrote, mono is numbers the model produced. The signature element
is the assay gauge, which draws the verdict as what it actually is — a
threshold on one scalar — with the 0.45-0.50 amber band at true scale, i.e. 5%
of the width. That makes "borderline is a hair's breadth" something the reader
sees rather than something we claim. Fonts are self-hosted woff2 (OFL, ~69 KB
total) so the design renders identically offline, which a CDN link would not,
and which ADR-018's "identical experience locally" claim depends on.

**Alternatives considered:**
- A separate static origin (nginx or a third compose service) with
  CORSMiddleware — rejected above: pure configuration cost, no benefit here.
- Deleting the Gradio app — rejected for now. It is working, tested code and a
  genuine fallback; keeping it costs a compose service that was already there.
- A JS framework and build step — rejected. The only real frontend logic is
  wrapping character offsets in <mark> tags, which is a dozen lines of string
  slicing. A toolchain would add a build to a repo that currently has none.

**Status:** Implemented in frontend/, api/main.py (/ask, /presets,
pipeline_loaded, static mount), app/presets.py, and src/rag/pipeline.py
(GenerationError.kind). Frontend JS interaction is not unit-tested (no JS
toolchain in this repo); it is scoped the same way app/app.py's Gradio
callbacks were — logic worth testing is tested in Python, and the integration
risk that matters (mount ordering, endpoint contracts) is covered in
tests/test_api.py.


---

## ADR-020: ACWS ablation results — recipe fix (arm b) adopted, noise down-weighting (arm c) rejected

**Context:** Following the research report's Candidate 1 hypothesis (Annotation-
Confidence-Weighted Supervision), a controlled 3-arm ablation was run on Track B:
(a) our existing production model; (b) a faithful LettuceDetect-recipe replication
(lr=1e-5, effective batch 8, 6 epochs, checkpoint selection on span-level F1
instead of response-level F1, no implicit_true weighting); (c) identical to (b)
plus implicit_true_weight=0.25 (down-weighting training loss on RAGTruth's own
annotator-flagged "implicit_true" spans, per this project's original finding that
13.5% of gold hallucination-span character mass is annotator-acknowledged noise).

**Findings:**
- Gate 4 (evaluation pipeline reproduction of the published model) passed exactly
  (span-F1 0.5114 vs. published 0.5113; response-F1 0.7619 vs. 0.7619), validating
  the new stratified-evaluation infrastructure before trusting arms b/c.
- Arm (b) — the recipe fix ALONE, no ACWS — substantially outperformed the
  current production model (arm a): span-F1 0.5321 (+2.1 points), response
  precision 0.8359 vs 0.7873, false-positive rate on faithful responses 7.4% vs
  10.7%. This confirms the code audit's hypothesis that checkpoint selection on
  response-level F1 (structurally unable to distinguish tight spans from sloppy
  ones) was suppressing our span-level performance relative to LettuceDetect-base
  (0.5544 published), not an architectural or data limitation.
- Arm (c) — ACWS at lambda=0.25 — did NOT pass the pre-registered decision rule.
  Clean-span F1 was slightly WORSE than arm (b) (0.5262 vs 0.5307), response-F1
  only marginally better (0.7633 vs 0.7631). The hypothesis that down-weighting
  annotator-flagged noisy positives would measurably improve detection of
  genuinely-clean hallucinations was not supported at this weight value.

**Decision:**
- ADOPT arm (b)'s recipe (span-F1 checkpoint selection, LettuceDetect's exact
  hyperparameters) as the new Track B production model, replacing the current
  Hub model. This is a genuine, measured improvement via honest recipe
  correction, independent of the ACWS hypothesis.
- REJECT ACWS at lambda=0.25 as tested. Document this as a legitimate null
  result: the hypothesis was well-motivated (13.5% of gold span character mass
  is annotator-acknowledged noise, unexploited by any published RAGTruth
  system), pre-registered before testing, and cleanly falsified at this
  setting -- consistent with literature showing pretrained transformers can be
  surprisingly robust to moderate, even structured, label noise.
- Do not pursue further lambda sweeps (0, 0.5) for this specific mechanism --
  the result wasn't even directionally encouraging, and further tuning risk
  outweighs the modest research report note that additional Kaggle sessions
  cost more than they're likely to recover here.

**Status:** Complete. Arm (b) pending deployment (Hub push, README/model-card
update, live demo swap). ACWS (simple down-weighting) closed as a tested,
documented negative result -- see ADR-021 for a follow-up direction informed
by this finding.

**Addendum (post-deployment reconciliation):** A light reconciliation of
arm-b's exact per-row predictions found a trade-off not visible in the
aggregate metrics: while arm-b improves overall precision (false-positive
rate on faithful responses: 7.4% vs. arm-a's 10.7%) and span-F1, its
recall trade-off is NOT uniform -- it disproportionately worsens detection
of the hardest cases specifically. Subtle-hallucination miss rate rose
from 40.3% (arm-a) to 48.1% (arm-b); Evident-hallucination miss rate rose
from 27.0% to 30.6%. The FP:FN ratio shifted from 0.76 to 0.46 (more
under-flagging, less over-flagging). Since Subtle hallucinations from
strong generators are the deployment scenario this project's own analysis
identifies as mattering most, arm-b's adoption is a genuine precision/
recall trade-off, not a strict improvement -- documented here rather than
only in the aggregate F1 gain, so the decision is auditable on its own
terms.
