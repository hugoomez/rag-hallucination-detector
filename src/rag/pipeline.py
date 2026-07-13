"""End-to-end RAG pipeline with hallucination detection (Phase 5, Step 5A.3).

Wires the three Phase 5 components together: the FAISS Retriever over the mathematician
corpus (src/rag/retriever.py), a generator LLM served by the Groq API, and the Track B
span-level Detector (src/models/predict.py). RAGPipeline.answer(question) runs
retrieve -> build context -> generate -> detect and returns the dict the Phase 6 demo
UI renders.

Generator model: openai/gpt-oss-20b, NOT the llama models the phase doc suggests --
Groq deprecated llama-3.1-8b-instant / llama-3.3-70b-versatile on 2026-06-17 with
shutdown on 2026-08-16, and gpt-oss-20b is their official migration target. It is also
the better demo fit: the smallest/fastest production model (friendliest free-tier rate
limits), with a documented high hallucination rate on factual QA (OpenAI's gpt-oss model
card), so it will confidently invent facts for out-of-corpus questions instead of always
refusing -- which is what gives the detector something to catch. gpt-oss is a reasoning
model, but on Groq the chain-of-thought arrives in a separate message.reasoning field,
so message.content is only the final answer and the detector never sees CoT; we ask for
reasoning_effort="low" (fast, cheap, less self-correction) and include_reasoning=False.

RAG_PROMPT makes grounding the explicit contract: the model is told to answer ONLY from
the provided context and given a sanctioned refusal sentence for anything the context
doesn't cover. Per the phase doc's fair-evaluation guidance, that matters because a
hallucination is then a genuine failure of the instruction, not the model legitimately
drawing on parametric knowledge we never told it to avoid. The same context string goes
to both the generator and the detector, so the verdict is computed against exactly what
the generator saw.

GROQ_API_KEY is loaded from .env via python-dotenv inside create_groq_client() (never at
import time, so tests stay env-free) and is never printed or logged anywhere.

RAGPipeline.answer(question, no_context=True) runs the ADR-016 ablation mode: the
retriever still runs for real, so the detector checks the answer against genuine
retrieved context, but the generator is prompted with NO_CONTEXT_PROMPT instead of
RAG_PROMPT and never sees that context -- forcing it to answer from parametric
knowledge alone. The resulting context/answer mismatch is what the demo needs to show
the detector catching a real, unscripted hallucination (see docs/decisions.md ADR-016).
"""

import os

from dotenv import load_dotenv
from groq import APIConnectionError, Groq, GroqError, RateLimitError

from src.rag.retriever import CORPUS_DIR, DEFAULT_K, RetrievalResult

DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"
# Reasoning tokens count against this cap even when hidden from message.content: qwen's
# CoT alone runs ~500-900 tokens on RAG prompts (verified live; 512 truncated to an empty
# answer with finish_reason="length"), so the budget must fit CoT + answer.
DEFAULT_MAX_COMPLETION_TOKENS = 2048
DEFAULT_TEMPERATURE = 0.6

RAG_PROMPT = """\
Answer the question using ONLY the information in the context below.
If the context does not contain the information needed to answer, reply exactly:
"I cannot answer this question from the provided context."
Do not use any outside knowledge. Answer concisely, in complete sentences.

Context:
{context}

Question: {question}

Answer:"""

NO_CONTEXT_PROMPT = """\
Answer the following question using your own knowledge. Answer concisely, in complete
sentences.

Question: {question}

Answer:"""


class GenerationError(Exception):
    """Groq call failed after the SDK's built-in retries; message is user-facing and key-free."""


def create_groq_client() -> Groq:
    """Load GROQ_API_KEY from .env (via python-dotenv) and return a Groq client.

    Lives outside RAGPipeline so the pipeline itself never touches env/dotenv -- tests
    inject a fake client instead. The key is passed straight to the client constructor
    and never printed, logged, or embedded in any error message.
    """
    load_dotenv()
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and paste your key from "
            "https://console.groq.com/keys (the free tier is enough)."
        )
    return Groq(api_key=api_key)


def build_context(results: list[RetrievalResult]) -> str:
    """Render retrieved chunks as one prompt-ready context block.

    Each chunk is prefixed with its corpus filename so answers stay traceable to a
    source document, and chunks are separated by a `---` rule so the LLM sees them as
    distinct passages rather than one run-on text.
    """
    return "\n\n---\n\n".join(f"[source: {r.chunk.source}]\n{r.chunk.text}" for r in results)


class RAGPipeline:
    """Retrieve -> generate (Groq) -> detect (Track B): the object the demo UI drives."""

    def __init__(
        self,
        retriever,
        detector,
        groq_client,
        k: int = DEFAULT_K,
        model_name: str = DEFAULT_GROQ_MODEL,
    ) -> None:
        # Constructor injection of all three collaborators (mirrors Retriever/Detector):
        # tests pass fakes, main() passes the real ones.
        self.retriever = retriever
        self.detector = detector
        self.groq_client = groq_client
        self.k = k
        self.model_name = model_name

    def generate(self, question: str, context: str | None = None) -> str:
        """Ask the Groq-served model to answer `question`.

        With `context` given, uses RAG_PROMPT so the model is grounded and instructed to
        refuse when the context doesn't cover the question. With `context=None` (the
        ADR-016 ablation path), uses NO_CONTEXT_PROMPT instead -- a plain question with no
        grounding instruction, so the model answers from parametric knowledge rather than
        (incorrectly) triggering RAG_PROMPT's refusal sentence over an empty context.

        The SDK already retries connection errors and 429s with backoff, so this layer
        only translates final failures into actionable, key-free GenerationErrors
        (chained with `from` so tracebacks keep the original exception).
        """
        prompt = (
            NO_CONTEXT_PROMPT.format(question=question)
            if context is None
            else RAG_PROMPT.format(context=context, question=question)
        )
        # Reasoning knobs are per-model-family (Groq 400s on mismatches), so they are
        # applied conditionally to keep model_name a true one-string swap: gpt-oss takes
        # reasoning_effort/include_reasoning; qwen/deepseek take reasoning_format, and
        # without "hidden" they leak their <think> CoT into message.content (verified
        # live), burning the token budget and feeding CoT to the detector; anything else
        # gets no reasoning kwargs at all.
        if self.model_name.startswith("openai/gpt-oss"):
            reasoning_kwargs = {"reasoning_effort": "low", "include_reasoning": False}
        elif any(family in self.model_name for family in ("qwen", "deepseek")):
            reasoning_kwargs = {"reasoning_format": "hidden"}
        else:
            reasoning_kwargs = {}
        try:
            response = self.groq_client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=DEFAULT_TEMPERATURE,
                max_completion_tokens=DEFAULT_MAX_COMPLETION_TOKENS,
                **reasoning_kwargs,
            )
        except RateLimitError as exc:
            raise GenerationError(
                "Groq rate limit hit (free tier is ~30 requests/min); wait a minute and retry."
            ) from exc
        except APIConnectionError as exc:
            raise GenerationError("Could not reach the Groq API -- check your network connection.") from exc
        except GroqError as exc:
            # Auth failures, 4xx/5xx, decommissioned model, etc. SDK messages carry
            # status/model info but never the API key.
            raise GenerationError(f"Groq API call failed: {exc}") from exc

        content = response.choices[0].message.content
        if content is None or not content.strip():
            # finish_reason="length" here means hidden reasoning exhausted the completion
            # budget before any answer was emitted (observed live with qwen at a 512 cap).
            reason = getattr(response.choices[0], "finish_reason", "unknown")
            raise GenerationError(
                f"Groq returned an empty completion for model {self.model_name} (finish_reason={reason})."
            )
        return content.strip()

    def answer(self, question: str, no_context: bool = False) -> dict:
        """Run the full pipeline and return the demo-facing result dict.

        In normal operation the detector judges the generated answer against the exact
        context string the generator saw, so the verdict and the generation are grounded
        in the same evidence. With `no_context=True` (ADR-016 ablation), the retriever
        still runs and the detector still checks the answer against that REAL retrieved
        context, but the generator never saw it -- so any mismatch the detector flags is
        a genuine, unscripted hallucination rather than a fabricated demo case.
        """
        results = self.retriever.retrieve(question, k=self.k)
        context = build_context(results)
        answer = self.generate(question, context=None if no_context else context)
        verdict = self.detector.predict(context, answer)
        result = {
            "question": question,
            "contexts": [{"source": r.chunk.source, "text": r.chunk.text, "score": r.score} for r in results],
            "answer": answer,
            "verdict": verdict,
        }
        if no_context:
            result["ablation"] = True
        return result


def main() -> None:
    """Demo entrypoint: real retriever + detector + Groq client over probe questions.

    The grounded questions are chosen to exercise the three interesting outcomes:
    answerable from the corpus (expect a grounded answer), about a corpus mathematician
    but absent from the corpus (hallucination bait -- the detector should light up if
    the model invents), and fully out-of-corpus (the prompt's sanctioned refusal is the
    correct behavior). The last question repeats the first in ADR-016 ablation mode: same
    real retrieved context, but the model never sees it, giving a real, unscripted
    hallucination case for the detector to catch (see docs/decisions.md ADR-016).
    """
    from src.models.predict import Detector
    from src.rag.retriever import Retriever

    client = create_groq_client()
    retriever = Retriever()
    retriever.build(CORPUS_DIR)
    detector = Detector.from_pretrained()
    pipeline = RAGPipeline(retriever, detector, client)

    questions = [
        ("What did Gauss contribute to number theory?", False),
        ("What prizes did Emmy Noether win for her work on topology?", False),
        ("Who won the 2022 FIFA World Cup?", False),
        ("What did Gauss contribute to number theory?", True),
    ]
    for question, no_context in questions:
        label = "  [ABLATION: no-context]" if no_context else ""
        print(f"\n{'=' * 70}\nQ: {question}{label}")
        try:
            result = pipeline.answer(question, no_context=no_context)
        except GenerationError as exc:
            print(f"  generation failed: {exc}")
            continue
        sources = ", ".join(c["source"] for c in result["contexts"])
        verdict = result["verdict"]
        print(f"A: {result['answer']}")
        print(f"   sources: {sources}")
        if result.get("ablation"):
            print("   (ablation mode: model did NOT see the context above)")
        print(f"   verdict: {verdict['color']} score={verdict['score']:.4f}")
        for span in verdict["spans"]:
            print(f"   span [{span['start']:>4},{span['end']:>4}) -> {span['text']!r}")


if __name__ == "__main__":
    main()
