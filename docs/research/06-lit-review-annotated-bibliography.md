# Literature review — annotated bibliography for Ch. 2 (Related Work)

Output of an ARS `academic-paper` `lit-review`-mode pass (2026-07-25), scoped to the three
themes the chapter plan assigns to Ch. 2: (a) token-level span detection on RAGTruth,
(b) label-class conflation in benchmark design — faithfulness vs. factuality collapsed
into one label — and (c) precedent for metadata-conditioned scoring
(`due_to_null`/`implicit_true`). This document supplies the annotated bibliography,
literature matrix, and citation-verification record behind Ch. 2's claims; it does not
redraft §4b or restate the comparability discrepancies beyond the scope
[`05-paper-chapter-plan.md`](05-paper-chapter-plan.md) §4b already sets. Most primary-source
figures were already gathered and verified in
[`01-long-context-truncation.md`](01-long-context-truncation.md) §4 — this pass adds
existence/identity verification (title, authors, venue, identifier) for every citation
the chapter plan requires, and does not re-derive scores already settled there.

**Terminology note.** This document uses *label-class conflation*, not *label noise* or
*annotator disagreement*, per the reframing ruling in
[`05-paper-chapter-plan.md`](05-paper-chapter-plan.md) §1 (`reframing_ruling`) and its
evidentiary basis in `02-implicit-true-audit.md` §2.1: `implicit_true` is a severity
qualifier on a correctly-applied label, not a data-quality defect. It also does not
reintroduce "label-noise robustness in pretrained transformers" as a framing — that claim
was tested (ADR-020 arm c), found unsupported at λ=0.25, and its original robustness
framing was withdrawn in the same correction addendum that retired "noise" terminology.

---

## 1. Deterministic citation verification gate

Every reference below was checked against a live resolver — arXiv, ACL Anthology (Crossref
DOI), Semantic Scholar, or OpenAlex — on 2026-07-25, before being admitted to the
bibliography. No citation below passed silently; the row that failed initial verification
(RAGTruth's author order) is shown with the discrepancy and its resolution.

| # | Citation key | Resolver(s) queried | Result | Notes |
|---|---|---|---|---|
| 1 | Niu et al. 2024 (RAGTruth) | Semantic Scholar (`arXiv:2401.00396`), arXiv abstract page | **VERIFIED, with correction** | Semantic Scholar's `authors` field lists **Yuanhao Wu first**, Cheng Niu fifth — an S2 metadata ordering anomaly. **arXiv primary source confirms Cheng Niu is first author**: Niu, Wu, Zhu, Xu, Shum, Zhong, Song, Zhang. Venue: ACL 2024 (`2024.acl-long.585`). Cited as **Niu et al.**, per primary source, not per the S2 record. |
| 2 | Kovács & Recski 2025 (LettuceDetect) | arXiv abstract page (`arXiv:2502.17125`) | **VERIFIED** | Ádám Kovács, Gábor Recski. Submitted 24 Feb 2025. No venue/DOI found beyond arXiv at verification time (preprint). |
| 3 | Song et al. 2024 (RAG-HAT) | ACL Anthology (`2024.emnlp-industry.113`) | **VERIFIED** | Authors: Juntong Song, Xingguang Wang, Juno Zhu, Yuanhao Wu, Xuxin Cheng, Randy Zhong, **Cheng Niu**. EMNLP 2024 Industry Track. DOI `10.18653/v1/2024.emnlp-industry.113`. |
| 4 | Belyi et al. 2025 (Luna) | arXiv abstract page (`arXiv:2406.00975`), ACL Anthology (`2025.coling-industry.34`) | **VERIFIED (existence only)** | Masha Belyi, Robert Friel, Shuai Shao, Atindriyo Sanyal. arXiv v1 3 Jun 2024, v2 5 Jun 2024, titled "...Evaluation **Foundation** Model..."; COLING 2025 Industry Track camera-ready titled "...Evaluation **Lightweight** Model..." (title changed between preprint and camera-ready — not a discrepancy in substance). **Per this project's existing scoping (chapter plan §2, gate closed 2026-07-25), this verification confirms the paper exists and its authorship — it does NOT constitute independent verification of Luna's reported figures.** The 65.4% F1 figure used elsewhere in this project is cited via LettuceDetect Table 2 (secondary), unchanged by this pass. No public code or weights located. |
| 5 | Zha et al. 2023 (AlignScore) | arXiv abstract page (`arXiv:2305.16739`), ACL Anthology (`2023.acl-long.634`) | **VERIFIED** | Yuheng Zha, Yichi Yang, Ruichen Li, Zhiting Hu. ACL 2023. DOI `10.18653/v1/2023.acl-long.634`. |
| 6 | Tang, Laban & Durrett 2024 (MiniCheck) | arXiv abstract page (`arXiv:2404.10774`), ACL Anthology (`2024.emnlp-main.499`) | **VERIFIED** | Liyan Tang, Philippe Laban, Greg Durrett. EMNLP 2024. DOI `10.18653/v1/2024.emnlp-main.499`. |
| 7 | Hu et al. 2024 (RefChecker) | arXiv abstract page (`arXiv:2405.14486`), OpenAlex | **VERIFIED, one field unconfirmed** | Xiangkun Hu, Dongyu Ru, Lin Qiu, Qipeng Guo, Tianhang Zhang, Yang Xu, Yun Luo, Pengfei Liu, Yue Zhang, Zheng Zhang. Submitted 23 May 2024. **Amazon affiliation could not be confirmed from arXiv/OpenAlex metadata** (OpenAlex returns no institution field for these authors); it is stated only on the project's GitHub org page, a secondary, non-bibliographic source. Cited here as Hu et al. without asserting affiliation as a verified fact. |
| 8 | Warner et al. 2024/2025 (ModernBERT) | arXiv abstract search (`arXiv:2412.13663`), ACL Anthology (`2025.acl-long.127`) | **VERIFIED** | Benjamin Warner + 13 co-authors. arXiv posted 18 Dec 2024; accepted ACL 2025. |
| 9 | "RAGTruth++" / "RAGTruth-Enhance" | arXiv search, Semantic Scholar/OpenAlex (via web search) | **NOT FOUND** | No benchmark by either name located as of 2026-07-25. Not cited; not asserted to exist. |
| 10 | Maynez, Narayan, Bohnet & McDonald 2020 (faithfulness/factuality) | arXiv abstract page (`arXiv:2005.00661`), ACL Anthology (`2020.acl-main.173`) | **VERIFIED** | Joshua Maynez, Shashi Narayan, Bernd Bohnet, Ryan McDonald. ACL 2020, pp. 1906–1919. DOI `10.18653/v1/2020.acl-main.173`. Added 2026-07-27 to close theme (b)'s conceptual-precedent gap (§4, item 2 below). |

Row 2's rate-limit note: parallel Semantic Scholar queries for rows 2, 5, 6, 7 returned
HTTP 429; those four were instead verified directly at arXiv (and, for the three that are
formally published, at their ACL Anthology / Crossref DOI record), which are equally
admissible primary resolvers and in this case more authoritative than an aggregator record
— exactly as row 1's S2 anomaly illustrates.

---

## 2. Annotated bibliography

### Niu, C., Wu, Y., Zhu, J., Xu, S., Shum, K., Zhong, R., Song, J., & Zhang, T. (2024). RAGTruth: A hallucination corpus for developing trustworthy retrieval-augmented language models. *Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, 10862–10878. arXiv:2401.00396.

- **Type**: Conference paper (ACL 2024), benchmark/dataset paper.
- **Method**: 17,790 LLM responses across QA, summarization, and data-to-text tasks,
  human-annotated at the word/span level for hallucination, under a stated faithfulness
  objective ("unsupported *or* contradictory" relative to the retrieved context).
- **Key Findings**: Establishes RAGTruth as the reference token/span-level hallucination
  benchmark for RAG; reports response-level F1 for a fine-tuned Llama-2-13B baseline
  (80.7%), prompted GPT-4-turbo (68.3%), and SelfCheckGPT+GPT-4-turbo (60.5%), all Table 5.
  Offers an include/exclude evaluation option for `due_to_null` spans — the paper's own
  precedent for metadata-conditioned scoring.
- **Relevance**: The benchmark this entire project builds on; source of the gold labels,
  the `due_to_null` precedent (theme c), and the paper's own baseline figures (theme a).
  The `implicit_true` field this project's Ch. 4 audits does **not** appear anywhere in
  this paper — it postdates it (added to the released data February 2024, repo-README
  only) — which is itself part of the novelty claim, not a citation to look up.
- **Quality**: Primary source for every figure attributed to it; author order corrected at
  verification (see §1, row 1).
- **Potential Use**: Ch. 1 (gap sentence), Ch. 2 (construction + faithfulness objective +
  `due_to_null` precedent), Ch. 4 (label definitions).

### Maynez, J., Narayan, S., Bohnet, B., & McDonald, R. (2020). On faithfulness and factuality in abstractive summarization. *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics*, 1906–1919. DOI: 10.18653/v1/2020.acl-main.173. arXiv:2005.00661.

- **Type**: Conference paper (ACL 2020).
- **Method**: Large-scale human evaluation of abstractive summarization systems,
  distinguishing a generated claim's *faithfulness* to its source document from its
  *factuality* in the world — the two can diverge in either direction.
- **Key Findings**: Neural abstractive summarizers are highly prone to generating content
  unfaithful to the source, and conflating faithfulness with factuality misrepresents a
  system's actual hallucination behavior; pretrained models produce more faithful summaries
  than non-pretrained alternatives.
- **Relevance**: The conceptual precedent for theme (b). RAGTruth's ungrounded-but-true /
  ungrounded-and-false split is the same faithfulness/factuality distinction, applied to
  retrieval-augmented generation instead of summarization. This paper establishes that the
  *distinction* is not novel; what remains novel and is claimed as such is narrower —
  RAGTruth ships a machine-readable field encoding the distinction at the span level, and no
  published RAGTruth evaluation conditions on it. Closes the theme (b) gap noted in §4,
  item 2 (originally: no external source addressed the faithfulness/factuality distinction
  itself, only RAGTruth's own construction did).
- **Quality**: High — ACL 2020, human-evaluation methodology, widely cited foundational
  work in the faithfulness/factuality literature.
- **Potential Use**: Ch. 2 (Related Work, novelty rescoping), Ch. 1 (gap sentence,
  optionally).

### Kovács, Á., & Recski, G. (2025). LettuceDetect: A hallucination detection framework for RAG applications. arXiv:2502.17125.

- **Type**: Preprint (arXiv, Feb 2025); no venue/DOI beyond arXiv found at verification.
- **Method**: ModernBERT-backbone binary token classifier (supported/hallucinated) over
  RAGTruth, spans reconstructed by merging consecutive positive tokens at inference,
  char-overlap span metric.
- **Key Findings**: Published base/large response-level F1 76.07%/79.22%, span-level F1
  55.44%/58.93% (their Table 2). This project's `arm-b` recipe reproduces their exact
  hyperparameters (lr 1e-5, batch 8, 6 epochs, token-F1 checkpoint selection) after an
  earlier recipe (response-F1 selection) cost 2.1 span-F1 points.
- **Relevance**: The direct methodological baseline for theme (a) — token-level span
  detection on RAGTruth. Its `preprocess_ragtruth.py` was checked in code (2026-07-25) and
  **discards `implicit_true` and `due_to_null` entirely** — the Gate 3 finding this
  project's novelty claim rests on for this system specifically.
- **Quality**: High — public code and weights, exact recipe reproducible; this project's
  Gate 4 (arm-a reproduction of published numbers) passed to within 0.0001 F1.
- **Potential Use**: Ch. 2 (detector landscape, discard-the-field verification), Ch. 3.2
  (methods baseline recipe), Appendix A (base→large scaling target).

### Song, J., Wang, X., Zhu, J., Wu, Y., Cheng, X., Zhong, R., & Niu, C. (2024). RAG-HAT: A hallucination-aware tuning pipeline for LLM in retrieval-augmented generation. *Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing: Industry Track*, 1548–1558. DOI: 10.18653/v1/2024.emnlp-industry.113.

- **Type**: Conference paper (EMNLP 2024 Industry Track).
- **Method**: DPO-based Hallucination-Aware Tuning of the *generator* itself (Llama-3-8B),
  not an external detector — a decoder trained to prefer faithful continuations over
  hallucinated ones, using GPT-4-turbo-corrected preference pairs.
- **Key Findings**: 83.9% response-level F1 on RAGTruth (Table 2: P 87.3 / R 80.8),
  strongest system in this project's §4.0 landscape table by a wide margin.
- **Relevance**: Establishes theme (a)'s ceiling (the encoder track is the *cheap*
  frontier, not *the* frontier) — but **five of its seven authors also appear on the
  RAGTruth author list** (Niu, Wu, Zhu, Zhong, Song — verified at §1 row 3 against
  row 1), a materially larger overlap than "shares an author with the benchmark paper."
  Not independent external validation of the benchmark; whether its pipeline reads
  `implicit_true`/`due_to_null` is unverifiable (no public code).
- **Quality**: Primary source for its own figure (confirmed, Table 2). Not independently
  reproducible under this project's constraints (8B DPO run).
- **Potential Use**: Ch. 2 (detector landscape table, shared-authorship caveat).

### Belyi, M., Friel, R., Shao, S., & Sanyal, A. (2024/2025). Luna: An evaluation foundation model to catch language model hallucinations with high accuracy and low cost / Luna: A lightweight evaluation model to catch language model hallucinations with high accuracy and low cost. arXiv:2406.00975 (preprint) / *Proceedings of the 31st International Conference on Computational Linguistics: Industry Track* (COLING 2025, camera-ready).

- **Type**: Preprint + industry-track conference paper; **secondary source only for its
  reported figure in this project**.
- **Method** (as characterized via LettuceDetect's Table 2, per this project's existing
  scoping — not independently re-verified here): DeBERTa-v3-large (440M) encoder,
  sliding-window inference over 512-token windows with the question+response repeated per
  window, max-support-over-windows then min-over-tokens aggregation.
- **Key Findings**: 65.4% response-level F1 on RAGTruth as reported by LettuceDetect's
  Table 2 (P 52.7 / R 86.1) — **this project has not confirmed this figure at Luna's own
  paper**; no code or weights are public, and the paper resists text extraction (per this
  project's prior verification attempts, unchanged by this pass).
- **Relevance**: Theme (a) counterfactual — a *larger* backbone than LettuceDetect's using
  a workaround for context length still underperforms a native long-context encoder by
  roughly 11 points, supporting this project's earlier decision (ADR-004) to switch
  backbone rather than tile context.
- **Quality**: **Existence and authorship verified this pass** (§1 row 4); **figures
  remain unverified at source** and must be presented as such wherever cited. Do not
  imply independent verification.
- **Potential Use**: Ch. 2 (detector landscape table, flagged secondary), §2.2-style
  sliding-window discussion.

### Zha, Y., Yang, Y., Li, R., & Hu, Z. (2023). AlignScore: Evaluating factual consistency with a unified alignment function. *Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*. DOI: 10.18653/v1/2023.acl-long.634. arXiv:2305.16739.

- **Type**: Conference paper (ACL 2023).
- **Method**: 355M RoBERTa-based unified alignment function trained on 4.7M examples
  across 7 factual-consistency task families; scores an input pair by decomposing into
  chunk-level alignment, then aggregating (max-per-sentence, then average).
- **Key Findings**: Competitive with much larger LLM-based factual-consistency metrics at
  a fraction of the size; not evaluated on RAGTruth in the source paper.
- **Relevance**: Design precedent for the claim-decomposition family (theme a's
  alternative architecture class) — informed this project's NLI baseline's
  sentence-splitting and per-pair scoring design (`src/models/nli_baseline.py`
  docstring), not a benchmark comparison point. Reports no RAGTruth numbers.
- **Quality**: High — peer-reviewed, widely cited, public artifact.
- **Potential Use**: Ch. 2 (claim-decomposition family, cited for design not score).

### Tang, L., Laban, P., & Durrett, G. (2024). MiniCheck: Efficient fact-checking of LLMs on grounding documents. *Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing*. DOI: 10.18653/v1/2024.emnlp-main.499. arXiv:2404.10774.

- **Type**: Conference paper (EMNLP 2024).
- **Method**: Small fact-checking models (MiniCheck-FT5, 770M) trained to approach
  GPT-4-level grounding-verification accuracy at ~400× lower inference cost; introduces
  the LLM-AggreFact benchmark.
- **Key Findings**: Near-GPT-4 accuracy at a fraction of the cost, evaluated on its own
  aggregated benchmark suite; no RAGTruth numbers reported.
- **Relevance**: Same role as AlignScore — claim-decomposition family design precedent,
  not a RAGTruth score to compare against.
- **Quality**: High — peer-reviewed, public artifact.
- **Potential Use**: Ch. 2 (claim-decomposition family, design citation only).

### Hu, X., Ru, D., Qiu, L., Guo, Q., Zhang, T., Xu, Y., Luo, Y., Liu, P., Zhang, Y., & Zhang, Z. (2024). RefChecker: Reference-based fine-grained hallucination checker and benchmark for large language models. arXiv:2405.14486.

- **Type**: Preprint (arXiv, May 2024); no formal venue confirmed at verification.
- **Method**: Extracts claims from LLM responses as subject–predicate–object triplets,
  then verifies each triplet against reference evidence individually.
- **Key Findings**: Introduces a fine-grained hallucination-checking benchmark and
  methodology; reports no RAGTruth numbers.
- **Relevance**: Third member of the claim-decomposition family cited for design
  (triplet-level granularity as an alternative to this project's token-level framing), not
  score comparison.
- **Quality**: Medium — preprint only at verification; affiliation claim (Amazon) not
  independently confirmed (§1 row 7).
- **Potential Use**: Ch. 2 (claim-decomposition family, design citation only).

### Warner, B., Chaffin, A., Clavié, B., Weller, O., Hallström, O., Taghadouini, S., Gallagher, A., Biswas, R., Ladhak, F., Aarsen, T., Cooper, N., Adams, G., Howard, J., & Poli, I. (2024). Smarter, better, faster, longer: A modern bidirectional encoder for fast, memory efficient, and long context finetuning and inference. arXiv:2412.13663. Also: *Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics* (2025).

- **Type**: Preprint (Dec 2024), accepted ACL 2025.
- **Method**: A modernized bidirectional encoder — RoPE, GLU activations, alternating
  local/global attention, native 8192-token context, trained on 2T tokens.
- **Key Findings**: Pareto improvement over prior BERT-family encoders on classification
  and retrieval tasks, including long-context settings, at competitive inference cost.
- **Relevance**: The backbone this project's Approach 1 and Track B are built on — the
  architectural precondition for theme (a)'s token-level span detection without
  truncation (0.00% truncation at 4096 tokens, per this project's own ADR-011).
- **Quality**: High — peer-reviewed (ACL 2025), public weights and code.
- **Potential Use**: Ch. 2 (backbone justification), Ch. 3 (methods), Appendix A
  (base→large scaling).

---

## 3. Literature matrix

| Source | (a) Token-level span detection on RAGTruth | (b) Label-class/objective conflation | (c) Metadata-conditioned scoring precedent | Method | Quality |
|---|:---:|:---:|:---:|---|---|
| Niu et al. 2024 (RAGTruth) | x | main | **main** | Corpus/annotation | High — primary |
| Maynez et al. 2020 (faithfulness/factuality) | — | **main** (conceptual precedent) | — | Human evaluation | High — foundational |
| Kovács & Recski 2025 (LettuceDetect) | **main** | x (discards the field, verified in code) | x (discards the field) | Encoder, token-level | High — reproducible |
| Song et al. 2024 (RAG-HAT) | x | — | — | Decoder, DPO tuning | Primary for own figure; not independent validation |
| Belyi et al. 2024/2025 (Luna) | x | — | — | Encoder, sliding window | Secondary-only figure |
| Zha et al. 2023 (AlignScore) | (design only) | — | — | Claim decomposition | High |
| Tang, Laban & Durrett 2024 (MiniCheck) | (design only) | — | — | Claim decomposition | High |
| Hu et al. 2024 (RefChecker) | (design only) | — | — | Claim decomposition | Medium — preprint |
| Warner et al. 2024/2025 (ModernBERT) | (backbone) | — | — | Encoder architecture | High |

No source outside RAGTruth itself addresses theme (c) — this is the gap the paper's
contribution claim rests on, not an artifact of an incomplete search (see §4).

---

## 4. Identified gaps

1. **`implicit_true` has no literature to cite.** Confirmed again this pass: it appears in
   none of the 9 verified sources except as the field this project audits. It was added to
   RAGTruth's released data in February 2024, after the ACL paper (arXiv:2401.00396,
   verified above to contain no mention of it), and is documented only in the corpus
   README — a gap in the published record, not a search failure.
2. **No source stratifies RAGTruth performance by the ungrounded-but-true /
   ungrounded-and-false distinction.** None of RAGTruth, RAG-HAT, or LettuceDetect reports
   a metric conditioned on `implicit_true`; only `due_to_null` has an authors'-own
   include/exclude option (RAGTruth, theme c), and no verified source exercises it in a
   published table. **Partially closed 2026-07-27**: the *conceptual* distinction
   (faithfulness vs. factuality) is not itself novel — Maynez et al. 2020 (§1 row 10)
   establishes it for summarization. What remains true, and is now the precisely-scoped
   novelty claim, is narrower: no published *RAGTruth* evaluation conditions on the field
   that encodes this distinction for this benchmark specifically.
3. **The claim-decomposition family (AlignScore, MiniCheck, RefChecker) has no RAGTruth
   figures at all**, confirmed again at each paper's abstract this pass — they inform
   architectural design for theme (a)'s alternatives but cannot anchor a score comparison.
4. **RAG-HAT and Luna's treatment of `implicit_true`/`due_to_null` is unverifiable.**
   Neither publishes code; RAG-HAT's author overlap with RAGTruth (5 of 7 authors,
   confirmed this pass) and Luna's lack of any public artifact both block independent
   confirmation.
5. **No successor benchmark folds this distinction in.** A targeted search for
   "RAGTruth++" / "RAGTruth-Enhance" (§1, row 9) returned nothing — the gap is current as
   of 2026-07-25, not stale.

---

## 5. Recommended sources by paper section

| Section | Key sources |
|---|---|
| Ch. 1 (Introduction) | Niu et al. 2024 |
| Ch. 2 (Background/Related Work) | Niu et al. 2024; Maynez et al. 2020; Kovács & Recski 2025; Song et al. 2024; Belyi et al. (secondary); Zha et al. 2023; Tang, Laban & Durrett 2024; Hu et al. 2024 |
| Ch. 3 (Methods) | Kovács & Recski 2025 (recipe); Warner et al. 2024/2025 (backbone) |
| Ch. 4 (The conflated label class) | Niu et al. 2024 |
| Appendix A (scaling replication) | Kovács & Recski 2025; Warner et al. 2024/2025 |

---

## 6. Coverage distribution advisory

Venue-family concentration: 8 of 10 sources are ACL-family peer-reviewed venues (ACL/EMNLP/
COLING) = 80%, exceeding the 70% threshold. This reflects the field's actual publication
pattern for RAG hallucination detection rather than a narrow search — the 2 non-ACL-family
entries (LettuceDetect, RefChecker) are both arXiv preprints already included. No search
expansion undertaken; advisory only, non-blocking.

No time-distribution or methodological-distribution skew triggered (sources span
2020–2025 after adding Maynez et al.; methods split across encoder/token-level,
decoder/DPO, claim-decomposition, and human-evaluation families).
