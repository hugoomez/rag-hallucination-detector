\# Architecture Decision Records (ADR)



Log of key technical decisions made throughout the project, with their rationale.



\---



\## ADR-001: Use DeBERTa-v3-base as the base encoder



\*\*Context:\*\* Need to choose a pretrained transformer encoder to fine-tune for hallucination

detection, framed as an NLI-style classification task (premise = retrieved context,

hypothesis = answer claim).



\*\*Decision:\*\* Use `microsoft/deberta-v3-base`.



\*\*Rationale:\*\*

\- He et al. (2021) show DeBERTa consistently outperforms BERT, RoBERTa, and XLNet

&#x20; specifically on MNLI (the NLI benchmark), not just on average across tasks.

\- DeBERTa's disentangled attention mechanism separates content and position

&#x20; representations, which helps capture syntactic nuances (subject vs. object, negation)

&#x20; relevant to detecting contradiction vs. entailment.

\- v3 (vs. v1) uses ELECTRA-style Replaced Token Detection pretraining instead of MLM,

&#x20; giving a training signal on 100% of tokens instead of only the masked 15% — a stronger

&#x20; pretrained model at the same size.

\- `base` size runs comfortably on a free Colab T4 GPU (16GB), no paid compute required.



\*\*Alternatives considered:\*\* RoBERTa-base (weaker on MNLI per the DeBERTa paper).



\---



\## ADR-002: Train on Google Colab (free tier)



\*\*Context:\*\* Need to decide where to run training.



\*\*Decision:\*\* Google Colab, free T4 GPU tier.



\*\*Rationale:\*\* Sufficient VRAM (16GB) for DeBERTa-v3-base fine-tuning; no cost; no local

GPU available. Kaggle Notebooks considered as a backup option if Colab quota runs out.



\---



\## ADR-003: Repository language — English



\*\*Context:\*\* The project is a public portfolio piece.



\*\*Decision:\*\* All code, comments, commit messages, and documentation in English; working

conversations with the AI copilot in Spanish.



\*\*Rationale:\*\* English is the industry standard for public repos reviewed by recruiters

and other engineers.



\---



\## ADR-004: Long-context truncation strategy for DeBERTa-v3 input



\*\*Context:\*\* EDA on RAGTruth (Phase 1) showed that 70.34% of rows exceed

DeBERTa-v3's 512-token limit when concatenating context + response + special tokens.

This is highly non-uniform across task types: QA 34.31% exceed, Summary 75.77%, Data2txt

99.89%. Naive truncation also disproportionately harms hallucinated rows (50.52% of

truncated rows are hallucinated vs. 43.08% globally), since evidence needed to verify

faithfulness is lost. Responses are almost always under 512 tokens alone (mean 160), so

the response should never be the one truncated.



Extended research (see `docs/research/long-context-truncation.md` or equivalent)

surveyed the RAGTruth leaderboard (Luna, LettuceDetect, RAGTruth/RAG-HAT baselines),

truncation techniques (Sun et al. 2019 head/tail/head+tail), long-context encoders

(ModernBERT, Longformer, BigBird), and claim-decomposition + retrieval approaches

(AlignScore, MiniCheck, RefChecker).



\*\*Decision:\*\* Phased approach across three stages:



1\. \*\*MVP (now):\*\* Keep DeBERTa-v3-base (per ADR-001). Truncate the CONTEXT only, always

&#x20; preserving the full response + question. Train/evaluate separately by `task_type` (QA /

&#x20; Summary / Data2txt) to quantify the real cost of truncation per task before solving it

&#x20; further. This is the honest baseline against which later gains are measured.

2\. \*\*Approach 1 (next):\*\* Reproduce the LettuceDetect recipe by switching the backbone

&#x20; to ModernBERT-base (149M params, native 8,192-token context, fits the free Colab T4 with

&#x20; `attn_implementation="sdpa"` since FlashAttention 2 is unsupported on Turing GPUs).

&#x20; Token-classification head, ~4,096-token inputs, non-response tokens masked with -100.

&#x20; This eliminates ~99% of the truncation problem with comparatively low engineering

&#x20; effort, matching current encoder SOTA on RAGTruth (~76-79% F1).

3\. \*\*Approach 3 (advanced/capstone):\*\* Claim/sentence decomposition + retrieval + NLI,

&#x20; AlignScore/MiniCheck-style: split the response into sentences, retrieve top-k relevant

&#x20; context chunks per sentence with a small bi-encoder, run the fine-tuned DeBERTa-v3 NLI

&#x20; check per (chunk, sentence) pair, aggregate (max-entailment per sentence). Sidesteps the

&#x20; token limit entirely by design and is the most system-design-mature of the three

&#x20; approaches considered — the intended flagship deliverable of the project.



\*\*Alternatives considered:\*\*



\- Sliding-window DeBERTa-v3 (Luna-style, per-window label propagation + max-support

&#x20; aggregation): valid and demonstrates deep mastery of the ADR-001 constraint, but higher

&#x20; implementation complexity than Approach 1 for comparable gains. Deprioritized in favor

&#x20; of Approach 1 → Approach 3, revisited only if Approach 1's results are unsatisfactory.

\- Head+tail truncation (Sun et al. 2019): rejected as a long-term fix — its motivating

&#x20; assumption (salient info clusters at document start/end) does not hold for RAGTruth's

&#x20; Data2txt (structured data) or Summary (evidence scattered throughout).



\*\*Status:\*\* MVP in progress. Approach 1 planned as next milestone. Approach 3

planned as capstone/advanced deliverable.

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
taking one sentence's full triple. Flag priority: contradicted -> supported ->
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
rule ("any sentence not-supported -> hallucinated"), but poor calibration of the raw
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

