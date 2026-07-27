# Research: the `implicit_true` label-class audit

**What this is.** RAGTruth ships three per-span metadata fields that this project's
detectors did not use, and that the two systems whose preprocessing code is publicly
available also discard: `implicit_true`, `due_to_null`, and a per-response `quality`
field. This audit measures how much of the gold hallucination signal those fields cover.

**What `implicit_true` means.** RAGTruth's own README defines it verbatim as:

> `implicit_true` means this span is **correct while the info is not mentioned in the
> context**.

That is: **true in the world, absent from the retrieved context.** Under RAGTruth's stated
objective — the corpus targets "unsupported *or* contradictory claims" — such a span is
**correctly labelled a hallucination**. The flag records that the ungrounded content
happens also to be true; it does not retract the label.

**Correction to an earlier reading of this document (2026-07-25).** Previous versions of
this audit described `implicit_true` as marking spans "actually true *given the context* —
a positive label the annotator disagreed with", and framed the field as **label noise**.
That was a misreading, and §2.1 sets out the evidence that refutes it. The field is a
**severity qualifier, not a retraction**. Every use of "noise" and "annotator
disagreement" in the framing of this work is withdrawn. The correct framing is
**label-class conflation**: RAGTruth's positive class aggregates two materially different
error types — *ungrounded-and-false* and *ungrounded-but-true* — under one label, and
ships the field that separates them. None of the counts, shares, or downstream results
below change; only what they are claimed to mean.

**Status:** first run 2026-07-12; **re-verified 2026-07-24** against
`data/raw/ragtruth/dataset/response.jsonl` (17,790 responses, 14,289 gold spans);
**reframed 2026-07-25** per §2.1. Every number below is reproduced by the script in
[§5](#5-exact-computation), which is the computation, not a paraphrase of it.

---

## 1. Headline numbers

| Quantity | Value |
|---|---|
| Gold spans total | 14,289 |
| `implicit_true=True` spans | **1,928 (13.49%)** |
| ... of which also `due_to_null` | 6 |
| Strict-flag spans (`implicit_true` **and not** `due_to_null`) | 1,922 (13.45%) |
| Gold span **character mass** total | 756,461 chars |
| `implicit_true` character mass | **110,171 (14.56%)** |
| Strict-flag character mass | 109,899 (14.53%) |
| `due_to_null` spans | 1,642 (11.49%), 41,955 chars (5.55%) |

> **Correction to the figure quoted elsewhere in this repo.** ADR-020 and
> `src/data/preprocess_token_level.py`'s `print_implicit_true_report` docstring say
> "13.5% of gold hallucination-span **character mass**". 13.5% is the share by **span
> count**; by character mass it is **14.56%** (14.53% for the strict-flag definition, and
> 14.49% / 14.70% of positive *tokens* in the train / val parquets). The two framings
> were conflated at some point. Nothing downstream depends on which is quoted — the
> flag is per-token and computed from the spans directly — but the character-mass claim
> should read 14.6%, not 13.5%.

## 2. What the field marks — and what it does not

### 2.1 The evidence that `implicit_true` is a severity qualifier

Three independent sources agree, and together they rule out the "label noise" reading.

**(a) RAGTruth's README** (quoted in full above): *"this span is correct while the info is
not mentioned in the context."* Correctness and groundedness are named as separate
properties. The span is ungrounded; that is what makes it a hallucination under a
faithfulness objective.

**(b) The annotators' own comments.** 1,846 of the 1,928 flagged spans carry a free-text
`meta` comment. They consistently assert *both* halves — true, and not in the context:

> *"These details are correct, however, not directly mentioned in the passages."*
>
> *"Passages did not mention specific food products ... however, this is likely to be true
> based on the context."*
>
> *"...might be true and helpful but it was not mentioned."*
>
> *"While it is very likely from the context that the scale used is Fahrenheit, it is not
> specifically mentioned in the original passages."*

Not one of these retracts the label. Each explains why the span is *low-intensity*.

**(c) The severity prefix distribution — decisive.** RAGTruth's `meta` comments open with a
hallucination-intensity marker (`LOW` / `HIGH` INTRO OF NEW INFO, etc.). Flagged and
unflagged spans have almost inverted distributions:

| Span set | n | `LOW` | `HIGH` | none / other |
|---|---:|---:|---:|---:|
| `implicit_true = True` | 1,928 | **90.6%** | 0.8% | 8.6% |
| all other gold spans | 12,361 | 4.4% | **46.4%** | 49.2% |

`implicit_true` is, in practice, how RAGTruth's annotators mark **low-severity ungrounded
additions**. It is a severity qualifier applied *within* the positive class, not a
correction *of* it.

**Consequence for this project.** The audit's counts stand; the interpretation changes.
The field does not identify mislabelled data. It identifies a **distinguishable subclass**
of correctly-labelled positives — ungrounded-but-true content — that every published
RAGTruth score aggregates together with ungrounded-and-false content.

### 2.2 The field has no literature

`implicit_true` was added to the **data**, not the paper. RAGTruth's README dates it to
**February 2024**; the ACL 2024 paper (Niu et al., arXiv:2401.00396) **does not mention it
anywhere** — checked 2026-07-25. The field is documented only in the corpus README.

There is, however, a **direct precedent for metadata-conditioned scoring by the benchmark's
own authors**: the RAGTruth paper offers users an option to include or exclude
`due_to_null` spans when evaluating. The authors built exactly this affordance for one
metadata field and not for the other. That precedent is the strongest available support
for treating `implicit_true` the same way.

### 2.3 Where the flag concentrates: the "Subtle" category

Flagging is not spread evenly across RAGTruth's four span types. It is almost entirely
concentrated in one:

| `label_type` | spans | `implicit_true` | share |
|---|---:|---:|---:|
| Evident Baseless Info | 6,237 | 60 | 0.96% |
| Evident Conflict | 5,324 | 6 | 0.11% |
| **Subtle Baseless Info** | **2,527** | **1,861** | **73.64%** |
| Subtle Conflict | 201 | 1 | 0.50% |

**Nearly three quarters of "Subtle Baseless Info" spans are ungrounded-but-true content.**
"Subtle Baseless Info" is therefore not a single homogeneous detection class: it is
predominantly the low-severity subclass, mixed with a minority of genuinely consequential
cases.

This bears on a finding the project had reported as a model weakness. Phase 4's error
analysis found the detector misses Subtle-only responses far more often than Evident-only
ones (40.3% vs 27.0% under arm-a; 48.1% vs 30.6% under arm-b), and the audit shows
**47 of the 77 test "Subtle-only" responses (61%) consist entirely of `implicit_true`
spans**.

The obvious inference — that the Subtle miss rate is inflated by cases the model is
arguably right to leave unflagged — **was tested and does not hold.** Removing those 47
responses *raises* the miss rate to 53.3%, because arm-b flags the ungrounded-but-true
cohort somewhat more often than the genuinely unambiguous one. See
[`04-subtle-only-reconciliation.md`](04-subtle-only-reconciliation.md). The composition of
the cohort is a fact about the benchmark; it does not license a claim about the model's
capability in either direction.

## 3. Where the flagged subclass sits

### Responses that are *entirely* ungrounded-but-true

A response counts as hallucinated if it has any gold span. Some have nothing but flagged
spans:

| Split | hallucinated responses | all-`implicit_true` | share |
|---|---:|---:|---:|
| official `train` (= our train + val) | 6,721 | **605** | 9.00% |
| official `test` | 943 | **49** | 5.20% |

Under the stricter definition (`implicit_true and not due_to_null`) the train figure is
603 and test is unchanged at 49. Recomputed at the token level from the committed
parquets — which apply that definition and our own train/val split — this lands at
**550 train + 53 val = 603**, matching exactly.

These responses are where the conflation bites hardest. A detector built to flag only
*consequential* (ungrounded-and-false) hallucinations would leave all 49 test responses
unflagged, and official scoring would record 49 false negatives.

**The resulting bound, stated precisely.** This is not a ceiling on recall in isolation —
a system that flags everything still reaches 100% recall. It is a bound on achieving two
goals *at once*: **no detector can restrict itself to consequential hallucinations and
simultaneously score response-level recall above (943 − 49) / 943 = 94.8%** under
RAGTruth's official scoring. The span-level version of the same bound is larger — see the
character-mass table below.

### Character mass by split × task type

| Split | Task type | flagged chars / total | share |
|---|---|---:|---:|
| train | QA | 63,280 / 305,474 | **20.72%** |
| train | Data2txt | 30,008 / 262,339 | 11.44% |
| train | Summary | 8,732 / 102,771 | 8.50% |
| test | QA | 5,557 / 31,622 | **17.57%** |
| test | Data2txt | 1,855 / 36,264 | 5.12% |
| test | Summary | 739 / 17,991 | 4.11% |

QA is where the flagged subclass concentrates — roughly a fifth of QA gold character mass
in train, and a sixth in test. This is the main reason filtering flagged spans out of the
**test** set was rejected: it would change the benchmark by task type unevenly and break
comparability with every published RAGTruth number.

**Span-level bound.** Across the whole test split, flagged spans account for
**8,151 / 85,877 = 9.49%** of gold character mass. By the same joint-achievement argument
as above: **no detector can restrict itself to consequential hallucinations and
simultaneously score char-overlap span recall above 90.5%.** This is the larger of the two
bounds, and it applies to the metric — span-level P/R/F1 — that token-level detectors on
this benchmark are principally scored on.

### The two fields that mark something else entirely

- **`due_to_null`** (1,642 spans): **98.3% are Evident Baseless Info in Data2txt** —
  consequential hallucinations where the model invented a value for a null JSON field.
  These are high-severity positives and must keep full training weight. This is why the
  pipeline's flag predicate excludes them even when `implicit_true` is also set (6 spans).
- **`quality`**: 144 `incorrect_refusal` + 29 `truncated` responses across the corpus
  (25 of them in test: 24 + 1). Kept everywhere; the volume is negligible.

## 4. What was done with it

Three options were considered:

1. **Filter flagged spans from test** — rejected. Breaks benchmark comparability
   (see §3), and unevenly by task type.
2. **Filter from train only** — legitimate, but risks losing recall on the flagged test
   positives that remain in the benchmark.
3. **Down-weight in the training loss only** — the option taken forward, as
   Annotation-Confidence-Weighted Supervision (ACWS). Labels and metrics are untouched;
   only the per-token loss weight changes.

The audit is wired into the pipeline as an **auxiliary column**, never as a label edit:
`src/data/preprocess_token_level.py` emits `is_implicit_true` alongside `labels`, and
`print_implicit_true_report()` re-checks the audit numbers on every preprocessing run.
A token is flagged only if **every** raw gold span covering it is flagged — the check runs
against the raw spans, not the union-normalised ones, so a token backed by any
unqualified annotation keeps full weight.

ACWS was then tested and **rejected** at λ=0.25 (ADR-020). See
[`03-candidate-methods.md`](03-candidate-methods.md).

> **Note on the ACWS hypothesis and the reframing.** ACWS was motivated by the
> label-noise reading refuted in §2.1. Under the corrected reading it is not a noise
> correction but an attempt to teach the model to discount a *real, low-severity* error
> subclass. The experiment, its pre-registered decision rule, and its null result are
> unaffected — only the hypothesis's rationale changes, and that change is recorded in
> ADR-020's addendum rather than edited into the original text.

**Comparability note (verified 2026-07-25).** Two systems' preprocessing code is publicly
available, and **both discard the field**:
[LettuceDetect's `preprocess_ragtruth.py`](https://github.com/KRLabsOrg/LettuceDetect)
reads only `start`, `end`, and `label_type`; RAGTruth's **own reference baseline**
(vendored at `data/raw/ragtruth/baseline/`) contains zero occurrences of `implicit_true`
or `due_to_null` — `prepare_dataset.py` buckets spans by `label_type` alone, and
`predict_and_evaluate.py` sets `is_halu = len(labels) > 0`.
**Luna and RAG-HAT release no code or weights and could not be checked; their treatment of
the field is UNVERIFIED and must not be assumed.** The conflation is therefore shared
across at least the code-verifiable part of the leaderboard — a property of how the
benchmark is used, not a handicap unique to this project.

## 5. Exact computation

Run from the repo root with the raw corpus present
(`python src/data/download.py` fetches it into `data/raw/ragtruth/`). No dependencies
beyond the standard library.

```python
import io, json
from collections import Counter, defaultdict
from pathlib import Path

RAW = Path("data/raw/ragtruth/dataset")
load = lambda name: [json.loads(l) for l in io.open(RAW / name, encoding="utf-8")]

responses = load("response.jsonl")
sources = {s["source_id"]: s for s in load("source_info.jsonl")}
for r in responses:
    r["task_type"] = sources[r["source_id"]]["task_type"]

# One row per (response, gold span). A response with no spans is faithful.
spans = [(r, s) for r in responses for s in r["labels"]]

# --- span counts ---
n_it = sum(1 for _, s in spans if s.get("implicit_true"))
n_strict = sum(1 for _, s in spans if s.get("implicit_true") and not s.get("due_to_null"))

# --- Sec 2.1(c): severity prefix of the annotator's `meta` comment ---
def severity(s):
    m = str(s.get("meta", "")).upper()
    for tag in ("LOW", "HIGH", "MEDIUM"):
        if m.startswith(tag):
            return tag
    return "none/other"

for flagged in (True, False):
    sub = [s for _, s in spans if bool(s.get("implicit_true")) is flagged]
    c = Counter(severity(s) for s in sub)
    print("implicit_true" if flagged else "other", len(sub),
          {k: f"{v} ({v / len(sub):.1%})" for k, v in c.most_common()})

# --- character mass: spans are [start, end) character offsets into the response ---
chars = lambda pred: sum(s["end"] - s["start"] for _, s in spans if pred(s))
total_chars = chars(lambda s: True)
it_chars = chars(lambda s: s.get("implicit_true"))

# --- by label_type ---
by_type = defaultdict(lambda: [0, 0])
for _, s in spans:
    by_type[s["label_type"]][0] += 1
    by_type[s["label_type"]][1] += bool(s.get("implicit_true"))

# --- responses whose gold spans are ALL flagged ---
for split in ("train", "test"):
    rows = [r for r in responses if r["split"] == split and r["labels"]]
    all_it = [r for r in rows if all(s.get("implicit_true") for s in r["labels"])]
    print(split, len(all_it), "/", len(rows))

# --- the test "Subtle-only" cohort ---
subtle_only = [r for r in responses
               if r["split"] == "test" and r["labels"]
               and all("Subtle" in s["label_type"] for s in r["labels"])]
print(len(subtle_only),
      sum(all(s.get("implicit_true") for s in r["labels"]) for r in subtle_only))
```

Definitions that matter:

- **Character mass**, not span count, is the meaningful denominator for span-level
  metrics: `char_span_prf` scores character overlap, so a long span counts more than a
  short one. Span offsets are `[start, end)` into the response string; RAGTruth contains
  overlapping annotations (115 of 17,790 responses), so summing `end - start` over raw
  spans **double-counts overlaps**. The figures in §1 are raw sums, matching how the
  original audit was computed; the pipeline's per-token flag uses union-normalised
  regions and is therefore the overlap-free version (14.49% of train positive tokens vs
  14.53% of raw flagged char mass — the gap is small because overlap is rare).
- **"Strict flag"** = `implicit_true and not due_to_null`, the predicate used by
  `src/data/preprocess_token_level.py`. Raw `implicit_true` alone is reported alongside it
  because the two differ by only 6 spans. (Predicate rename pending — see the Gate-1
  terminology sweep.)
- **Official `train`** (15,090 responses) is what this project splits into our train
  (13,578) + val (1,512); `test` (2,700) is untouched. The 605 figure is over the
  official train split.

### Live re-check

`src/data/preprocess_token_level.py` prints, per split, the flagged share of positive
tokens and the number of all-flagged responses on every run. Current values:

| Split | flagged / positive tokens | share | all-flagged responses |
|---|---:|---:|---:|
| train | 18,352 / 126,646 | 14.49% | 550 |
| val | 2,057 / 13,992 | 14.70% | 53 |
| test | 1,597 / 17,844 | 8.95% | 49 |

The docstring's stated expectation ("~13.5% ... and ~605") should be read as 14.5% and
605 = 550 + 53 across our train/val split, per the correction in §1.

---

## Provenance

| Claim | Source |
|---|---|
| All numbers in §1–§3 | Recomputed 2026-07-24 from `data/raw/ragtruth/dataset/response.jsonl` + `source_info.jsonl` via §5's script |
| Token-level shares, all-flagged counts per split | `data/processed/token_level_binary_{train,val,test}.parquet` |
| Flag definition, per-token flagging rule | `src/data/preprocess_token_level.py` (`print_implicit_true_report`) |
| 40.3% / 27.0% and 48.1% / 30.6% Subtle-vs-Evident miss rates | `README.md` Error analysis, ADR-020 addendum, `docs/model_cards/track_b.md` |
| §2.1(a) `implicit_true` definition | `data/raw/ragtruth/README.md`, response.jsonl field table — quoted verbatim |
| §2.1(b) annotator `meta` comments | Sampled 2026-07-25 from `data/raw/ragtruth/dataset/response.jsonl` (1,846 of 1,928 flagged spans carry one) |
| §2.1(c) severity distribution (90.6% / 4.4%) | Computed 2026-07-25 via §5's `severity()` snippet |
| §2.2 field absent from the ACL paper; added Feb 2024 | `data/raw/ragtruth/README.md` Updates §2; full-text check of arXiv:2401.00396, 2026-07-25 — no occurrence |
| §2.2 `due_to_null` include/exclude precedent | RAGTruth paper (Niu et al., ACL 2024), evaluation §; checked 2026-07-25 |
| 9.49% test flagged char mass, 90.5% / 94.8% bounds | Computed 2026-07-25 from the §3 character-mass table |
| LettuceDetect uses all spans unfiltered | Their `preprocess_ragtruth.py` on GitHub — checked 2026-07-12, **re-verified 2026-07-25** |
| RAGTruth's own baseline ignores the field | `data/raw/ragtruth/baseline/` (vendored): zero occurrences of `implicit_true`/`due_to_null`; `prepare_dataset.py:16–22`, `predict_and_evaluate.py:85`. Checked 2026-07-25 |
| Luna / RAG-HAT treatment of the field | **UNVERIFIED** — no public code or weights located, 2026-07-25 |
| Filtering options 1–3 and the choice of ACWS | ADR-020, `scripts/ablation_report.py` |
