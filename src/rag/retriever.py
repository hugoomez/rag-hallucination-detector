"""Dense passage retrieval over the mathematician corpus (Phase 5, Step 5A.2).

Consumes the plain-text Wikipedia extracts produced by src/rag/fetch_corpus.py
(data/corpus/*.txt) and builds a single FAISS IndexFlatIP over sentence-transformer
embeddings of word-count sliding-window chunks, so a query can be matched against the
most relevant passage from any of the five articles.

Chunking here is deliberately a standalone word-count sliding window rather than a reuse
of src/data/context_chunking.py: that module is sentence-based (nltk) and task-type-aware,
built around RAGTruth's structured source_info dicts, and isn't a good fit for flat,
uniform-length passages carved out of long prose that also contains embedded
MathML/LaTeX noise from the Wikipedia extract -- a word-count window tolerates that noise
as just more tokens instead of breaking on it.

Embeddings are L2-normalized before indexing, so FAISS's inner-product index
(IndexFlatIP) is equivalent to cosine similarity search.
"""

from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CHUNK_SIZE = 500  # words per chunk
DEFAULT_OVERLAP = 50  # words of overlap between consecutive chunks
DEFAULT_K = 3
CORPUS_DIR = Path("data/corpus")


@dataclass
class Chunk:
    """One sliding-window passage plus its provenance, ready for embedding/indexing.

    `source` is the corpus filename (e.g. "gauss.txt") the chunk was extracted from --
    kept as the literal filename rather than the stripped stem so retrieval results are
    traceable back to the exact file on disk without the caller needing to reconstruct a
    path (`corpus_dir / chunk.source` always round-trips). `chunk_id` is the chunk's
    0-based index within its own source document (not global), useful for debugging
    chunk-boundary/overlap behavior per document.
    """

    text: str
    source: str
    chunk_id: int


@dataclass
class RetrievalResult:
    """A retrieved chunk paired with its similarity score against the query.

    `score` is the raw FAISS inner-product value, which is a cosine similarity in
    [-1, 1] because both query and chunk embeddings are L2-normalized before indexing.
    """

    chunk: Chunk
    score: float


def chunk_words(
    text: str,
    source: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[Chunk]:
    """Slide a fixed-size word window over `text`, tagging each chunk with its `source`.

    Word-count chunking rather than sentence chunking because the corpus is flat prose
    with embedded MathML/LaTeX noise (see fetch_corpus.py) that would break sentence
    tokenizers unpredictably; a word-count window tolerates that noise as just more
    tokens, keeps chunks a model-length-friendly size for all-MiniLM-L6-v2, and overlap
    avoids losing context that straddles a chunk boundary.
    """
    assert overlap < chunk_size, f"overlap ({overlap}) must be smaller than chunk_size ({chunk_size})"
    assert chunk_size > 0, f"chunk_size must be positive, got {chunk_size}"

    words = text.split()
    if not words:
        return []

    step = chunk_size - overlap
    chunks: list[Chunk] = []
    start = 0
    while start < len(words):
        window = words[start : start + chunk_size]
        chunks.append(Chunk(text=" ".join(window), source=source, chunk_id=len(chunks)))
        if start + chunk_size >= len(words):
            break
        start += step
    return chunks


class Retriever:
    """Dense retriever: chunks a text corpus, embeds it, and serves top-k FAISS lookups."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, embedder: object | None = None) -> None:
        # `embedder` injection point: production code leaves it None and lazily loads a
        # real SentenceTransformer; tests pass a fake object exposing just `.encode(...)`
        # so no model download ever happens under test (mirrors NLIHallucinationDetector's
        # constructor-injected model/tokenizer pattern in src/models/nli_baseline.py).
        self.model_name = model_name
        self.embedder = embedder if embedder is not None else SentenceTransformer(model_name)
        self.index: faiss.IndexFlatIP | None = None
        self.chunks: list[Chunk] = []

    def build(
        self,
        corpus_dir: Path,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        overlap: int = DEFAULT_OVERLAP,
    ) -> None:
        """Load every .txt file in corpus_dir, chunk it, embed all chunks, and index them."""
        corpus_dir = Path(corpus_dir)
        txt_paths = sorted(corpus_dir.glob("*.txt"))  # sorted for deterministic order across runs

        self.chunks = []
        for path in txt_paths:
            text = path.read_text(encoding="utf-8")
            self.chunks.extend(chunk_words(text, source=path.name, chunk_size=chunk_size, overlap=overlap))

        if not self.chunks:
            # Empty corpus: leave index as None so retrieve() rejects via the same
            # "not built" error path rather than building a degenerate 0-row index.
            self.index = None
            return

        texts = [c.text for c in self.chunks]
        embeddings = np.asarray(
            self.embedder.encode(texts, normalize_embeddings=True, convert_to_numpy=True),
            dtype="float32",
        )

        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)
        print(f"Indexed {len(self.chunks)} chunks from {len(txt_paths)} documents (dim={embeddings.shape[1]}).")

    def retrieve(self, query: str, k: int = DEFAULT_K) -> list[RetrievalResult]:
        """Embed `query` and return the top-k most similar chunks by cosine similarity."""
        if self.index is None:
            raise RuntimeError("Retriever.build() must be called before retrieve()")

        query_vec = np.asarray(
            self.embedder.encode([query], normalize_embeddings=True, convert_to_numpy=True),
            dtype="float32",
        )

        k = min(k, len(self.chunks))  # clamp: avoid FAISS -1 padding when k > ntotal
        scores, indices = self.index.search(query_vec, k)

        return [
            RetrievalResult(chunk=self.chunks[idx], score=float(score))
            for score, idx in zip(scores[0], indices[0])
            if idx != -1  # defensive: -1 marks "no result" if k somehow still exceeds ntotal
        ]


def main() -> None:
    retriever = Retriever()
    retriever.build(CORPUS_DIR)

    queries = [
        "What did Gauss contribute to number theory?",
        "What did von Neumann do for computing?",
        "What is the Riemann hypothesis about?",
    ]
    for query in queries:
        print(f"\nQuery: {query}")
        for result in retriever.retrieve(query, k=3):
            preview = result.chunk.text[:200].replace("\n", " ")
            print(f"  [{result.score:.3f}] ({result.chunk.source}) {preview}...")


if __name__ == "__main__":
    main()
