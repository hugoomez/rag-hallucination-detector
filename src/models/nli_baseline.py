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

import nltk
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

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


def _ensure_punkt() -> None:
    """Make sure an nltk sentence tokenizer model is available, downloading if needed."""
    for resource in ("punkt_tab", "punkt"):
        try:
            nltk.data.find(f"tokenizers/{resource}")
            return
        except LookupError:
            if nltk.download(resource, quiet=True):
                return


def split_sentences(text: str) -> list[str]:
    """Split text into sentences with nltk; returns [] for blank/whitespace input."""
    if not text or not text.strip():
        return []
    _ensure_punkt()
    return nltk.sent_tokenize(text)


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

    def _verify_sentence(
        self,
        response_sentence: str,
        context_sentences: list[str],
        ent_thr: float,
        con_thr: float,
    ) -> SentenceVerdict:
        """Score one response sentence against every context sentence and aggregate."""
        if not context_sentences:
            # No evidence to check against: nothing supports or contradicts it.
            return SentenceVerdict(response_sentence, 0.0, 0.0, 0.0, "unverifiable")

        scores = self._score_pairs(context_sentences, [response_sentence] * len(context_sentences))

        best_ent_idx = max(range(len(scores)), key=lambda i: scores[i][0])
        max_entailment = scores[best_ent_idx][0]
        neutral = scores[best_ent_idx][1]
        max_contradiction = max(contra for _, _, contra in scores)

        flag = flag_from_scores(max_entailment, max_contradiction, ent_thr, con_thr)
        return SentenceVerdict(response_sentence, max_entailment, neutral, max_contradiction, flag)

    def detect(
        self,
        context: str,
        response: str,
        ent_thr: float = DEFAULT_ENT_THR,
        con_thr: float = DEFAULT_CON_THR,
    ) -> DetectionResult:
        """Verify each response sentence against the context and aggregate a response flag.

        A response counts as hallucinated if any of its sentences is not "supported".
        An empty response (no sentences) is vacuously not hallucinated.
        """
        context_sentences = split_sentences(context)
        response_sentences = split_sentences(response)

        verdicts = [
            self._verify_sentence(sentence, context_sentences, ent_thr, con_thr) for sentence in response_sentences
        ]
        response_hallucinated = any(verdict.flag != "supported" for verdict in verdicts)
        return DetectionResult(response_hallucinated, verdicts)
