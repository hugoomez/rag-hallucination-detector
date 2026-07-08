"""Unit tests for task-type-aware context chunking (ADR-008).

Hermetic: ``split_sentences`` is monkeypatched to a deterministic regex splitter, so the
tests exercise the chunking logic without depending on nltk's punkt data or the network.
"""

import re

import pytest

from src.data import context_chunking
from src.data.context_chunking import chunk_context


def _fake_split(text):
    """Deterministic stand-in for nltk sentence tokenization."""
    if not text or not text.strip():
        return []
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text.strip()) if part.strip()]


@pytest.fixture(autouse=True)
def deterministic_split(monkeypatch):
    monkeypatch.setattr(context_chunking, "split_sentences", _fake_split)


# --- Summary --------------------------------------------------------------------------


def test_summary_returns_sentence_chunks():
    assert chunk_context("Summary", "Alpha runs fast. Beta is slow.") == [
        "Alpha runs fast.",
        "Beta is slow.",
    ]


# --- QA -------------------------------------------------------------------------------


def test_qa_splits_passages_into_sentences_excluding_question():
    source_info = {
        "question": "what color is alpha?",
        "passages": "passage 1:Alpha is red. Alpha is large.\n\npassage 2:Beta is blue.",
    }
    chunks = chunk_context("QA", source_info)

    assert chunks == ["Alpha is red.", "Alpha is large.", "Beta is blue."]
    # The question is not evidence and must not appear as a chunk.
    assert all("what color" not in chunk for chunk in chunks)
    # The "passage N:" markers must be stripped, not leaked into chunks.
    assert all("passage" not in chunk.lower() for chunk in chunks)


def test_qa_without_passage_markers_falls_back_to_sentence_split():
    source_info = {"question": "q?", "passages": "Just one block of text. Two sentences here."}
    assert chunk_context("QA", source_info) == ["Just one block of text.", "Two sentences here."]


# --- Data2txt -------------------------------------------------------------------------


def _data2txt_example():
    return {
        "name": "Joe's Diner",
        "business_stars": 4.5,
        "attributes": {
            "BusinessParking": None,  # the canonical hallucination-evidence field
            "WiFi": "no",
            "Ambience": {"casual": True},  # doubly-nested dict
        },
        "review_info": [
            {
                "review_stars": 5.0,
                "review_date": "2022-01-01 10:00:00",
                "review_text": "Great food. Nice staff.",
            }
        ],
    }


def test_data2txt_scalars_nested_dicts_and_list_of_dicts():
    chunks = chunk_context("Data2txt", _data2txt_example())

    assert chunks == [
        "name: Joe's Diner",
        "business_stars: 4.5",
        "BusinessParking: null",  # nested dict, None -> "null"
        "WiFi: no",
        "casual: true",  # doubly-nested dict, True -> "true"
        "review_stars: 5.0",  # non-string field of a list-of-dict entry
        "2022-01-01 10:00:00",  # string field, sentence-tokenized (single chunk)
        "Great food.",  # review_text prose split into sentences
        "Nice staff.",
    ]


def test_data2txt_no_json_syntax_leaks_into_any_chunk():
    chunks = chunk_context("Data2txt", _data2txt_example())
    # No chunk should contain raw JSON/dict syntax (braces or quoted-key separators).
    for chunk in chunks:
        assert "{" not in chunk
        assert "}" not in chunk
        assert '": ' not in chunk


def test_data2txt_list_of_plain_strings_is_sentence_tokenized():
    source_info = {"notes": ["First note. Second note.", "Third note."]}
    assert chunk_context("Data2txt", source_info) == ["First note.", "Second note.", "Third note."]


# --- Errors ---------------------------------------------------------------------------


def test_unrecognized_task_type_raises():
    with pytest.raises(ValueError, match="Unrecognized task_type"):
        chunk_context("Translation", "some text")
