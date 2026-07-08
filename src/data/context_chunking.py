"""Task-type-aware context chunking for the NLI hallucination baseline (ADR-008).

The generic "split the whole context into sentences" approach only works for
natural-language prose (Summary). For QA the normalized context is a question plus
concatenated passages, and for Data2txt it is structured data — running nltk sentence
tokenization over those produces meaningless premises (undivided passage blobs, JSON
fragments). This module produces evidence chunks tailored to each task_type so that every
chunk is a real, checkable statement the NLI model can use as a premise.

Home of ``split_sentences`` (moved here from src/models/nli_baseline.py): it only needs
nltk, so keeping it in this lightweight data module avoids dragging torch/transformers in
just to split sentences. nli_baseline imports it from here.
"""

import re

import nltk

_PASSAGE_MARKER = re.compile(r"passage\s*\d+\s*:", re.IGNORECASE)


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


def _fmt_scalar(value) -> str:
    """Render a scalar as evidence text (JSON-ish null/true/false rather than Python repr)."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _field_chunks(key: str, value) -> list[str]:
    """Turn one structured (key, value) field into evidence chunks, recursing as needed.

    Scalars become a single ``"{key}: {value}"`` chunk. Nested dicts recurse (each leaf
    scalar becomes its own ``"{leaf_key}: {value}"`` chunk, so evidence like
    ``BusinessParking: null`` is preserved). Lists yield one group of chunks per entry:
    string entries are sentence-tokenized; dict entries have their string fields
    sentence-tokenized (prose such as review_text) and their non-string scalars emitted as
    ``"{key}: {value}"``. No branch ever serializes a container, so JSON syntax never leaks
    into a chunk.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return [f"{key}: {_fmt_scalar(value)}"]

    if isinstance(value, dict):
        chunks: list[str] = []
        for sub_key, sub_value in value.items():
            chunks.extend(_field_chunks(sub_key, sub_value))
        return chunks

    if isinstance(value, list):
        chunks = []
        for entry in value:
            if isinstance(entry, str):
                chunks.extend(split_sentences(entry))
            elif isinstance(entry, dict):
                for sub_key, sub_value in entry.items():
                    if isinstance(sub_value, str):
                        chunks.extend(split_sentences(sub_value))
                    else:
                        chunks.extend(_field_chunks(sub_key, sub_value))
            else:
                chunks.append(_fmt_scalar(entry))
        return chunks

    return [f"{key}: {_fmt_scalar(value)}"]


def _qa_chunks(source_info: dict) -> list[str]:
    """Split QA passages into per-passage sentence chunks; the question is not evidence."""
    passages = source_info["passages"]
    chunks: list[str] = []
    for part in _PASSAGE_MARKER.split(passages):
        part = part.strip()
        if part:
            chunks.extend(split_sentences(part))
    return chunks


def chunk_context(task_type: str, source_info) -> list[str]:
    """Break a RAGTruth source into task-type-appropriate evidence chunks.

    - ``Summary``: ``source_info`` is a prose string -> nltk sentence chunks.
    - ``QA``: ``source_info`` is ``{"question", "passages"}`` where ``passages`` is one
      string of ``passage N:``-prefixed passages -> split into passages, then sentence
      chunks per passage (question excluded).
    - ``Data2txt``: ``source_info`` is a dict of structured fields -> ``"{key}: {value}"``
      per scalar (recursing into nested dicts) and sentence chunks for list-entry prose.

    Raises ``ValueError`` for any other ``task_type``.
    """
    if task_type == "Summary":
        return split_sentences(source_info)
    if task_type == "QA":
        return _qa_chunks(source_info)
    if task_type == "Data2txt":
        chunks: list[str] = []
        for key, value in source_info.items():
            chunks.extend(_field_chunks(key, value))
        return chunks
    raise ValueError(f"Unrecognized task_type: {task_type!r}")
