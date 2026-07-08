"""Zero-shot NLI hallucination detector for RAGTruth responses.

Uses an off-the-shelf NLI model (MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli) to check
each response sentence against the retrieved context, with no fine-tuning. This is the
zero-shot foundation for ADR-004's Approach 3 (sentence-decomposition + NLI, in the
AlignScore/MiniCheck style): it provides the sentence-splitting, per-pair NLI scoring,
and max-aggregation plumbing that the fine-tuned capstone will reuse.

For each response sentence we score (context sentence -> response sentence) NLI pairs
against every context sentence and keep the max entailment and max contradiction
independently (the two maxima may come from different context sentences). The per-
sentence flag is "supported" / "contradicted" / "unverifiable"; a response counts as
hallucinated if any of its sentences is not "supported".
"""

from dataclasses import dataclass

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.data.context_chunking import split_sentences

DEFAULT_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
MAX_LENGTH = 512
DEFAULT_ENT_THR = 0.5
DEFAULT_CON_THR = 0.5

_REQUIRED_LABELS = ("entailment", "neutral", "contradiction")


@dataclass
class SentenceVerdict:
    """NLI verdict for a single response sentence against the whole context.

    ``entailment`` is the max entailment probability over context sentences and
    ``contradiction`` is the max contradiction probability over context sentences; these
    two maxima may come from different context sentences, so the triple does not sum to 1
    by design. ``neutral`` is reported from the best-entailment context sentence and is
    informational only (the flag never depends on it).
    """

    sentence: str
    entailment: float
    neutral: float
    contradiction: float
    flag: str  # "supported" | "unverifiable" | "contradicted"


@dataclass
class DetectionResult:
    """Response-level detection result: the aggregate flag plus per-sentence verdicts."""

    response_hallucinated: bool
    verdicts: list[SentenceVerdict]


def flag_from_scores(entailment: float, contradiction: float, ent_thr: float, con_thr: float) -> str:
    """Map aggregated entailment/contradiction scores to a support flag.

    Contradiction is checked before entailment (see ADR-007): because entailment and
    contradiction are maxed independently over context sentences, a claim can score high
    on both. A contradicted claim is a hallucination even when some other context
    sentence partially supports it, so "contradicted" takes priority over "supported".
    """
    if contradiction >= con_thr:
        return "contradicted"
    if entailment >= ent_thr:
        return "supported"
    return "unverifiable"


def apply_thresholds(all_scores: list[list[tuple[float, float]]], ent_thr: float, con_thr: float) -> list[bool]:
    """Turn precomputed raw scores into response_hallucinated flags at given thresholds.

    ``all_scores`` is a list of per-response score-lists, each element being the
    ``[(max_entailment, max_contradiction), ...]`` returned by
    ``NLIHallucinationDetector.score_response``. Returns one ``response_hallucinated``
    bool per response: True if any sentence is not "supported" (empty score-list -> False,
    matching the vacuously-not-hallucinated convention).

    This is the cheap, model-free half of evaluation: run the NLI model once to get
    ``all_scores``, then call this repeatedly while sweeping thresholds.
    """
    return [
        any(flag_from_scores(ent, con, ent_thr, con_thr) != "supported" for ent, con in scores) for scores in all_scores
    ]


class NLIHallucinationDetector:
    """Zero-shot NLI faithfulness checker over sentence-decomposed context and response."""

    def __init__(self, model, tokenizer, device: str = "cpu") -> None:
        # Read the label order dynamically from the model config; never assume that a
        # given index means "entailment". Different NLI checkpoints order labels
        # differently, and hardcoding positions silently corrupts the scores.
        label2idx = {name.lower(): idx for idx, name in model.config.id2label.items()}
        missing = [label for label in _REQUIRED_LABELS if label not in label2idx]
        if missing:
            raise ValueError(
                f"Model config id2label is missing required NLI labels {missing}; " f"got {model.config.id2label}"
            )

        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.device = device
        self._ent_idx = label2idx["entailment"]
        self._neu_idx = label2idx["neutral"]
        self._con_idx = label2idx["contradiction"]

    @classmethod
    def from_pretrained(cls, model_name: str = DEFAULT_MODEL, device: str | None = None) -> "NLIHallucinationDetector":
        """Load the tokenizer and NLI model from the HF hub and build a detector."""
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        return cls(model, tokenizer, device=device)

    def _score_pairs(self, premises: list[str], hypotheses: list[str]) -> list[tuple[float, float, float]]:
        """Run NLI on (premise, hypothesis) pairs, returning (ent, neu, con) per pair."""
        encoded = self.tokenizer(
            premises,
            hypotheses,
            max_length=MAX_LENGTH,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with torch.no_grad():
            logits = self.model(**encoded).logits
        probs = torch.softmax(logits, dim=-1)
        return [
            (
                float(row[self._ent_idx]),
                float(row[self._neu_idx]),
                float(row[self._con_idx]),
            )
            for row in probs
        ]

    def _aggregate_sentence(self, response_sentence: str, context_chunks: list[str]) -> tuple[float, float, float]:
        """Aggregate NLI scores for one response sentence over all context chunks.

        Returns ``(max_entailment, neutral, max_contradiction)``. Entailment and
        contradiction are maxed independently across context chunks (ADR-007), so the two
        maxima may come from different chunks; ``neutral`` is taken from the best-entailment
        chunk and is informational only. Empty context returns ``(0.0, 0.0, 0.0)`` (nothing
        supports or contradicts the sentence).

        This is the single, threshold-free NLI core shared by ``score_response`` and
        ``detect``, so the model runs exactly once per response sentence regardless of how
        many threshold settings are later applied.
        """
        if not context_chunks:
            return (0.0, 0.0, 0.0)

        scores = self._score_pairs(context_chunks, [response_sentence] * len(context_chunks))

        best_ent_idx = max(range(len(scores)), key=lambda i: scores[i][0])
        max_entailment = scores[best_ent_idx][0]
        neutral = scores[best_ent_idx][1]
        max_contradiction = max(contra for _, _, contra in scores)
        return (max_entailment, neutral, max_contradiction)

    def score_response(self, context_chunks: list[str], response: str) -> list[tuple[float, float]]:
        """Return raw aggregated ``(max_entailment, max_contradiction)`` per response sentence.

        ``context_chunks`` is the pre-chunked context (from ``chunk_context``); chunking is
        the caller's responsibility since the right strategy is task-type-dependent (ADR-008).
        These are the threshold-free scores, before any flag decision. Compute them once per
        (context, response) pair and reuse them across many threshold settings via
        ``apply_thresholds`` instead of re-running the NLI model for every threshold — the
        expensive model work happens here, threshold tuning is then model-free.
        """
        response_sentences = split_sentences(response)
        return [
            (entailment, contradiction)
            for entailment, _neutral, contradiction in (
                self._aggregate_sentence(sentence, context_chunks) for sentence in response_sentences
            )
        ]

    def detect(
        self,
        context_chunks: list[str],
        response: str,
        ent_thr: float = DEFAULT_ENT_THR,
        con_thr: float = DEFAULT_CON_THR,
    ) -> DetectionResult:
        """Verify each response sentence against the context chunks and aggregate a flag.

        ``context_chunks`` is the pre-chunked context (from ``chunk_context``). Reuses the
        same NLI aggregation as ``score_response`` (computed once per sentence), then applies
        ``flag_from_scores`` with the given thresholds. A response counts as hallucinated if
        any of its sentences is not "supported"; an empty response (no sentences) is vacuously
        not hallucinated.
        """
        response_sentences = split_sentences(response)

        verdicts = []
        for sentence in response_sentences:
            entailment, neutral, contradiction = self._aggregate_sentence(sentence, context_chunks)
            flag = flag_from_scores(entailment, contradiction, ent_thr, con_thr)
            verdicts.append(SentenceVerdict(sentence, entailment, neutral, contradiction, flag))

        response_hallucinated = any(verdict.flag != "supported" for verdict in verdicts)
        return DetectionResult(response_hallucinated, verdicts)


COLOR = {"supported": "🟢", "unverifiable": "🟡", "contradicted": "🔴"}


def response_color(result: DetectionResult) -> str:
    """Return the response-level traffic-light color for a DetectionResult.

    Priority: any contradicted sentence -> 🔴; else any unverifiable sentence -> 🟡;
    else 🟢. An empty verdict list (no sentences) returns 🟢, consistent with the
    "vacuously not hallucinated" behavior in DetectionResult.

    This function has no model or tensor dependencies by design: it operates purely on
    an already-computed DetectionResult, so it can be reused as-is in the Phase 6 demo
    UI without importing torch/transformers.
    """
    flags = {verdict.flag for verdict in result.verdicts}
    if "contradicted" in flags:
        return COLOR["contradicted"]
    if "unverifiable" in flags:
        return COLOR["unverifiable"]
    return COLOR["supported"]
