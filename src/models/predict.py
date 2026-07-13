"""Track B span-level hallucination Detector for the Phase 5 demo (Step 5A.4).

Wraps the best fine-tuned model, hugoomezz/modernbert-ragtruth-token-level-binary (the
Track B token-level binary classifier, ADR-013), behind a single predict(context, response)
call. Unlike the phase doc's sequence-classification stub, this model labels every response
token supported(0)/hallucinated(1), so we can return real character-level span highlights,
not just a response-level score.

predict() tokenizes (context, response) exactly as src/data/preprocess_token_level.py did at
training time (truncation="only_first", return_offsets_mapping=True, sequence_ids() to find
response tokens), runs one forward pass, and then:
  - reconstructs character spans by reusing train_token_level.merge_predicted_spans verbatim,
    with the ignore-mask derived from sequence_ids() standing in for the gold labels it uses
    at eval time (a response token -> a real label, everything else -> IGNORE_LABEL);
  - derives the response-level score as the max per-token P(hallucinated) over the response's
    real tokens, identical to collect_predictions.collect_track_b_modernbert, so the demo's
    headline number matches how this model was reported throughout Phase 4;
  - maps that score to a traffic-light color via color_from_score, mirroring the glyphs and
    the pure/torch-free design of nli_baseline.response_color so the Phase 6 UI can reuse it.

The color thresholds are deliberately minimal (binary-honest): red_threshold defaults to 0.5,
the model's only validated decision boundary (argmax over two classes), so red <=> at least
one predicted span; a narrow yellow "near-miss" band sits just below it. Both thresholds are
constructor/from_pretrained args, and setting yellow_threshold == red_threshold collapses the
scale to pure green/red.
"""

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from src.data.preprocess_token_level import (
    HALLUCINATED_LABEL,
    IGNORE_LABEL,
    MAX_LENGTH,
    MODEL_NAME as BASE_MODEL_NAME,
    SUPPORTED_LABEL,
)
from src.models.train_token_level import merge_predicted_spans

DEFAULT_MODEL = "hugoomezz/modernbert-ragtruth-token-level-binary"
# The tokenizer is loaded from the base ModernBERT repo, not the fine-tuned one: the
# fine-tuned repo's tokenizer_config.json was written by a newer transformers than the
# pinned install can parse (raises "Tokenizer class TokenizersBackend does not exist"),
# the same version-skew collect_predictions.py works around via --tokenizer_id. Here the
# substitution is not just safe but exact: preprocess_token_level.py built the training
# data with this very tokenizer (MODEL_NAME), so its tokenization/offsets are identical.
DEFAULT_TOKENIZER = BASE_MODEL_NAME
DEFAULT_ATTN_IMPLEMENTATION = "sdpa"
DEFAULT_RED_THRESHOLD = 0.5
DEFAULT_YELLOW_THRESHOLD = 0.45

GREEN, YELLOW, RED = "🟢", "🟡", "🔴"


def color_from_score(
    score: float,
    red_threshold: float = DEFAULT_RED_THRESHOLD,
    yellow_threshold: float = DEFAULT_YELLOW_THRESHOLD,
) -> str:
    """Map a response-level hallucination score to a traffic-light glyph.

    score >= red_threshold -> 🔴; else score >= yellow_threshold -> 🟡; else 🟢. With the
    defaults, 🔴 coincides exactly with the model's positive decision (argmax at 0.5), so a
    red banner always has at least one highlighted span. Setting yellow_threshold ==
    red_threshold removes the yellow band entirely. Pure and torch-free by design (like
    nli_baseline.response_color) so the demo UI can import it without transformers.
    """
    if score >= red_threshold:
        return RED
    if score >= yellow_threshold:
        return YELLOW
    return GREEN


class Detector:
    """Token-level hallucination detector: predict(context, response) -> score/color/spans."""

    def __init__(
        self,
        model,
        tokenizer,
        device: str = "cpu",
        red_threshold: float = DEFAULT_RED_THRESHOLD,
        yellow_threshold: float = DEFAULT_YELLOW_THRESHOLD,
    ) -> None:
        # Constructor injection (mirrors NLIHallucinationDetector / Retriever) keeps tests
        # hermetic: a fake model/tokenizer can be passed in with no Hub download.
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.device = device
        self.red_threshold = red_threshold
        self.yellow_threshold = yellow_threshold

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = DEFAULT_MODEL,
        device: str | None = None,
        attn_implementation: str = DEFAULT_ATTN_IMPLEMENTATION,
        tokenizer_id: str = DEFAULT_TOKENIZER,
        red_threshold: float = DEFAULT_RED_THRESHOLD,
        yellow_threshold: float = DEFAULT_YELLOW_THRESHOLD,
    ) -> "Detector":
        """Load the token-classification model + tokenizer from the Hub and build a Detector.

        Device auto-detects (cuda if available, else cpu), matching
        NLIHallucinationDetector.from_pretrained. attn_implementation defaults to "sdpa" for
        consistency with training (train_token_level.py); it is valid on CPU and can be
        overridden (e.g. to "eager") if a particular CPU/transformers combination objects.
        tokenizer_id defaults to the base ModernBERT tokenizer (see DEFAULT_TOKENIZER); pass
        the fine-tuned repo id to force-load its own tokenizer copy instead.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
        model = AutoModelForTokenClassification.from_pretrained(model_name, attn_implementation=attn_implementation)
        return cls(
            model,
            tokenizer,
            device=device,
            red_threshold=red_threshold,
            yellow_threshold=yellow_threshold,
        )

    def predict(self, context: str, response: str) -> dict:
        """Detect hallucinated spans in `response` given `context`.

        Returns {"score": float, "color": str, "spans": [{"start", "end", "text"}]}, where
        span offsets are relative to `response` and text = response[start:end] (no stripping,
        so offsets stay identical to the training-time char-overlap metric).
        """
        encoding = self.tokenizer(
            context,
            response,
            max_length=MAX_LENGTH,
            truncation="only_first",
            return_offsets_mapping=True,
            return_token_type_ids=False,
        )
        sequence_ids = encoding.sequence_ids()
        offsets = encoding["offset_mapping"]
        token_starts = [start for start, _end in offsets]
        token_ends = [end for _start, end in offsets]

        input_ids = torch.tensor([encoding["input_ids"]], device=self.device)
        attention_mask = torch.tensor([encoding["attention_mask"]], device=self.device)
        with torch.no_grad():
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
        # (seq,) P(hallucinated) per token, the column collect_predictions also reads.
        probs_hallucinated = torch.softmax(logits, dim=-1)[0, :, HALLUCINATED_LABEL].tolist()

        # sequence_ids() stands in for the gold labels merge_predicted_spans consumes at eval
        # time: response tokens (seq_id 1) become a real label, everything else IGNORE_LABEL,
        # so context/special tokens are skipped exactly as during training.
        label_mask = [SUPPORTED_LABEL if seq_id == 1 else IGNORE_LABEL for seq_id in sequence_ids]
        # A token is predicted hallucinated iff P >= 0.5 (argmax over the two classes), the
        # model's native boundary and the same rule used at training/eval time.
        pred_row = [HALLUCINATED_LABEL if prob >= 0.5 else SUPPORTED_LABEL for prob in probs_hallucinated]

        char_spans = merge_predicted_spans(pred_row, label_mask, token_starts, token_ends)
        spans = [{"start": start, "end": end, "text": response[start:end]} for start, end in char_spans]

        response_probs = [prob for prob, seq_id in zip(probs_hallucinated, sequence_ids) if seq_id == 1]
        score = float(max(response_probs)) if response_probs else 0.0
        color = color_from_score(score, self.red_threshold, self.yellow_threshold)

        return {"score": score, "color": color, "spans": spans}
