"""Regenerate app/demo_cache.json by running the real RAG pipeline (Phase 6, Step 6.2).

One-shot build tool, NOT run at app startup. It calls the live Groq API (needs GROQ_API_KEY
in .env) once per preset question and freezes the results, so the Gradio demo's preset
buttons render instantly, deterministically, and offline -- see the caching rationale in
app/app.py's module docstring (generator temperature=0.6 makes live answers non-reproducible;
HF Spaces cold starts would otherwise fire live calls and risk rate limits).

Re-run this by hand whenever the pipeline, prompts, retriever corpus, or the preset list
changes:

    python app/precompute_cache.py

The three presets mirror src/rag/pipeline.py::main() -- an in-corpus grounded question, an
out-of-corpus refusal, and the ADR-016 no-context ablation (von Neumann's second-marriage
date), which is the primary live-hallucination demo and should come back 🔴.
"""

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.models.predict import Detector  # noqa: E402
from src.rag.pipeline import RAGPipeline, create_groq_client  # noqa: E402
from src.rag.retriever import CORPUS_DIR, Retriever  # noqa: E402

CACHE_PATH = Path(__file__).resolve().parent / "demo_cache.json"

# (question, no_context) -- keep in sync with TAB2_PRESETS in app/app.py.
PRESETS = [
    ("What did Gauss contribute to number theory?", False),
    ("Who won the 2022 FIFA World Cup?", False),
    ("On what exact date did John von Neumann marry his second wife, Klara Dan?", True),
]


def main() -> None:
    client = create_groq_client()
    retriever = Retriever()
    retriever.build(CORPUS_DIR)
    detector = Detector.from_pretrained()
    pipeline = RAGPipeline(retriever, detector, client)

    cache: dict[str, dict] = {}
    for question, no_context in PRESETS:
        result = pipeline.answer(question, no_context=no_context)
        cache[question] = result
        verdict = result["verdict"]
        tag = " (ablation)" if result.get("ablation") else ""
        print(f"{verdict['color']} score={verdict['score']:.4f}{tag}  {question}")
        print(f"    -> {result['answer']}")

    CACHE_PATH.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {len(cache)} cached results to {CACHE_PATH}")


if __name__ == "__main__":
    main()
