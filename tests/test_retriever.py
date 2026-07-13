"""Tests for src/rag/retriever.py.

Hermetic and offline: never constructs a real SentenceTransformer. Retriever.build()
and .retrieve() are exercised through a fake embedder (_FakeEmbedder below) that hashes
words into a small bag-of-words vector, so lexical overlap between a query and a chunk
actually drives cosine similarity -- the retrieval-correctness tests need shared words
to be *why* a chunk ranks first, not coincidence.
"""

import zlib

import numpy as np
import pytest

from src.rag.retriever import Retriever, chunk_words

_FAKE_DIM = 32


class _FakeEmbedder:
    """Deterministic bag-of-words hashing embedder: shared words -> higher cosine similarity.

    Uses zlib.crc32 rather than the builtin hash() because str hashing is randomized
    per-process unless PYTHONHASHSEED is fixed -- crc32 keeps this reproducible across runs.
    """

    def encode(self, texts, normalize_embeddings=True, convert_to_numpy=True):
        vectors = []
        for text in texts:
            vec = np.zeros(_FAKE_DIM, dtype="float32")
            for word in text.lower().split():
                vec[zlib.crc32(word.encode()) % _FAKE_DIM] += 1.0
            norm = np.linalg.norm(vec)
            if normalize_embeddings and norm > 0:
                vec = vec / norm
            vectors.append(vec)
        return np.array(vectors, dtype="float32")


def _write_corpus(tmp_path):
    (tmp_path / "cats.txt").write_text(
        "Cats are small domesticated carnivorous mammals. Cats enjoy chasing mice and yarn. "
        "Many cats sleep most of the day curled up in warm sunny spots.",
        encoding="utf-8",
    )
    (tmp_path / "rockets.txt").write_text(
        "Rockets use combustion to generate thrust for spaceflight. Liquid fuel rockets "
        "carry oxidizer and propellant in separate tanks for orbital launches.",
        encoding="utf-8",
    )
    return tmp_path


# --- 1. chunk_words: word-count/overlap behavior (hand-traced) -------------------------


def test_chunk_words_produces_hand_traced_boundaries():
    # 12 words, chunk_size=5, overlap=2 -> step=3
    text = " ".join(f"w{i}" for i in range(12))
    chunks = chunk_words(text, source="doc.txt", chunk_size=5, overlap=2)
    assert [c.text for c in chunks] == [
        "w0 w1 w2 w3 w4",
        "w3 w4 w5 w6 w7",
        "w6 w7 w8 w9 w10",
        "w9 w10 w11",  # short tail chunk, expected
    ]
    assert [c.chunk_id for c in chunks] == [0, 1, 2, 3]
    assert all(c.source == "doc.txt" for c in chunks)


def test_chunk_words_empty_text_returns_empty_list():
    assert chunk_words("", source="doc.txt") == []
    assert chunk_words("   ", source="doc.txt") == []  # whitespace-only


def test_chunk_words_rejects_overlap_gte_chunk_size():
    with pytest.raises(AssertionError, match="overlap"):
        chunk_words("a b c", source="doc.txt", chunk_size=3, overlap=3)


# --- 2. Retriever.build + retrieve on a synthetic corpus -------------------------------


def test_retrieve_returns_most_relevant_chunk_for_matching_query(tmp_path):
    _write_corpus(tmp_path)
    retriever = Retriever(embedder=_FakeEmbedder())
    retriever.build(tmp_path, chunk_size=50, overlap=5)  # small window; corpus is tiny

    results = retriever.retrieve("Tell me about cats and mice", k=2)

    assert results[0].chunk.source == "cats.txt"  # topically matching doc wins
    assert results[0].score > results[-1].score  # ranked, not coincidental order


def test_retrieve_clamps_k_to_available_chunks(tmp_path):
    _write_corpus(tmp_path)
    retriever = Retriever(embedder=_FakeEmbedder())
    retriever.build(tmp_path, chunk_size=500, overlap=50)  # whole doc fits in 1 chunk each

    results = retriever.retrieve("cats", k=100)

    assert len(results) == len(retriever.chunks)  # not 100


# --- 3. Error paths ----------------------------------------------------------------------


def test_retrieve_before_build_raises():
    retriever = Retriever(embedder=_FakeEmbedder())
    with pytest.raises(RuntimeError, match="build"):
        retriever.retrieve("anything")


def test_build_on_empty_corpus_leaves_index_unset(tmp_path):
    retriever = Retriever(embedder=_FakeEmbedder())
    retriever.build(tmp_path, chunk_size=50, overlap=5)  # tmp_path has no .txt files

    with pytest.raises(RuntimeError, match="build"):
        retriever.retrieve("anything")
