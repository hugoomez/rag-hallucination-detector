"""FastAPI wrapper around the Track B span-level Detector, plus the demo frontend (Phase 6).

Exposes the model and the RAG pipeline over HTTP, and serves the custom frontend from the
same origin:
  - POST /detect   {context, response}      -> {score, color, spans:[{text, start, end}]}
  - POST /ask      {question, no_context}   -> {question, contexts, answer, verdict, ablation}
  - GET  /presets                           -> {presets:[{id, label, question, result}]}
  - GET  /health                            -> {status, model_loaded, pipeline_loaded}
  - GET  /                                  -> frontend/index.html (ADR-019)

The response schema mirrors Detector.predict()'s real dict exactly: `color` is a
traffic-light emoji glyph (not a color name), and each span carries start/end/text
only -- there is no per-span "label" field (the token-level model doesn't produce one,
unlike the sequence-classification stub the phase doc assumed).

Models are loaded ONCE at startup via FastAPI's lifespan context manager. Loading is
best-effort at two levels, mirroring app/app.py::load_models(): if the checkpoint can't be
fetched, /health reports model_loaded=false and /detect + /ask return 503; if the Groq key
is absent or the corpus can't be indexed, the Detector still serves /detect while /ask
returns 503 and /presets keeps working from the checked-in cache. Every failure is
observable rather than crashing the process.

All three model-dependent surfaces resolve through get_detector / get_pipeline /
get_presets, which tests override with fakes so the real checkpoint is never downloaded
(same hermetic-by-default convention as tests/test_predict.py and tests/test_pipeline.py).

The frontend is mounted at "/" and is deliberately registered LAST: Starlette matches routes
in registration order, so every API route above still wins over the catch-all mount. Serving
it from this process makes the browser's fetch("/detect") same-origin by construction, which
is why no CORSMiddleware appears anywhere in this file (ADR-019).
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.presets import PRESETS, Preset
from app.presets import load_cache as _load_preset_cache
from src.models.predict import Detector
from src.rag.pipeline import GenerationError, RAGPipeline, create_groq_client
from src.rag.retriever import Retriever

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = _REPO_ROOT / "frontend"
# src/rag/retriever.py's CORPUS_DIR is a bare relative Path("data/corpus"), so it only
# resolves when the process happens to be launched from the repo root. Anchor it to this
# file instead -- uvicorn's CWD is not something this app should depend on.
CORPUS_DIR = _REPO_ROOT / "data" / "corpus"

# GenerationError.kind -> HTTP status. A rate limit is the caller's cue to wait and retry
# (429); every other generation failure is an upstream fault we can't fix here (502).
_ERROR_STATUS = {"rate_limit": 429}
_DEFAULT_ERROR_STATUS = 502

NO_KEY_MESSAGE = (
    "Live RAG is unavailable because GROQ_API_KEY is not set. Add it to .env and restart "
    "to ask your own questions. The cached demo questions below still work."
)


# --- Schemas --------------------------------------------------------------------------


class DetectRequest(BaseModel):
    context: str
    response: str


class Span(BaseModel):
    text: str
    start: int
    end: int


class DetectResponse(BaseModel):
    score: float
    color: str  # traffic-light emoji glyph: 🟢 / 🟡 / 🔴
    spans: list[Span]


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    # ADR-016 ablation: retrieve for real, but blind the generator to the context.
    no_context: bool = False


class Context(BaseModel):
    source: str  # corpus filename, e.g. "gauss.txt"
    text: str
    score: float  # cosine similarity in [-1, 1]


class RagResult(BaseModel):
    question: str
    contexts: list[Context]
    answer: str
    verdict: DetectResponse
    # RAGPipeline.answer() OMITS this key unless no_context=True; the default fills it in so
    # the wire contract always carries the field and the frontend needn't guess.
    ablation: bool = False


class PresetEntry(BaseModel):
    id: str
    label: str
    question: str
    result: RagResult


class PresetsResponse(BaseModel):
    presets: list[PresetEntry]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    pipeline_loaded: bool


# --- App + model lifecycle ------------------------------------------------------------


def _build_pipeline(detector: Detector) -> RAGPipeline | None:
    """Build the RAG pipeline, or return None if anything it needs is unavailable.

    Groq is optional by design (mirrors app/app.py::load_models()): with no GROQ_API_KEY,
    create_groq_client() raises RuntimeError, /ask returns 503, and the cached presets carry
    the demo on their own.
    """
    try:
        retriever = Retriever()
        retriever.build(CORPUS_DIR)
        return RAGPipeline(retriever, detector, create_groq_client())
    except Exception:
        logger.exception("RAG pipeline unavailable; /ask will return 503")
        return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the Detector and RAG pipeline once at startup; degrade gracefully if either fails."""
    try:
        app.state.detector = Detector.from_pretrained()
    except Exception:
        logger.exception("Detector failed to load at startup; /detect will return 503")
        app.state.detector = None

    # The pipeline needs the detector, so a failed checkpoint rules it out too.
    app.state.pipeline = _build_pipeline(app.state.detector) if app.state.detector is not None else None

    yield
    app.state.detector = None
    app.state.pipeline = None


app = FastAPI(title="RAG Hallucination Detector", lifespan=lifespan)


def get_detector(request: Request) -> Detector | None:
    """Return the loaded Detector, or None if it failed to load.

    One of three injection seams for the API: tests override these to inject fakes, so the
    real checkpoint is never downloaded and lifespan never runs.
    """
    return getattr(request.app.state, "detector", None)


def get_pipeline(request: Request) -> RAGPipeline | None:
    """Return the loaded RAGPipeline, or None if Groq/the corpus was unavailable."""
    return getattr(request.app.state, "pipeline", None)


def get_presets() -> list[Preset]:
    """Return the preset definitions. Overridden in tests so /presets never reads the disk cache."""
    return PRESETS


def get_cache() -> dict:
    """Read the checked-in preset cache; {} if it hasn't been generated yet.

    Reuses app/presets.py's loader (shared with the Gradio app) rather than reimplementing
    it. Tests override this seam with an in-memory dict, so /presets never depends on
    demo_cache.json existing on disk.
    """
    return _load_preset_cache()


# --- Endpoints ------------------------------------------------------------------------


@app.post("/detect", response_model=DetectResponse)
def detect(req: DetectRequest, detector: Detector | None = Depends(get_detector)) -> dict:
    if detector is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return detector.predict(req.context, req.response)


@app.post("/ask", response_model=RagResult)
def ask(req: AskRequest, pipeline: RAGPipeline | None = Depends(get_pipeline)) -> dict:
    """Run the live RAG pipeline: retrieve -> generate (Groq) -> detect.

    GenerationError messages are user-facing and key-free by pipeline.py's contract, so they
    are surfaced verbatim; `kind` lets the frontend style the failure. No stack trace ever
    reaches the client.
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail={"kind": "no_key", "message": NO_KEY_MESSAGE})
    try:
        return pipeline.answer(req.question, no_context=req.no_context)
    except GenerationError as exc:
        status = _ERROR_STATUS.get(exc.kind, _DEFAULT_ERROR_STATUS)
        raise HTTPException(status_code=status, detail={"kind": exc.kind, "message": str(exc)}) from exc


@app.get("/presets", response_model=PresetsResponse)
def presets(defs: list[Preset] = Depends(get_presets), cache: dict = Depends(get_cache)) -> dict:
    """Serve the precomputed preset results.

    Needs no Detector and no Groq key -- the cache is checked in -- so the demo's preset path
    keeps working even when the model fails to load. Presets missing from the cache are
    omitted rather than erroring; an empty cache yields an empty list and the frontend simply
    hides the preset row.
    """
    entries = [
        {"id": p.id, "label": p.label, "question": p.question, "result": cache[p.question]}
        for p in defs
        if p.question in cache
    ]
    return {"presets": entries}


@app.get("/health", response_model=HealthResponse)
def health(
    detector: Detector | None = Depends(get_detector),
    pipeline: RAGPipeline | None = Depends(get_pipeline),
) -> dict:
    """Report what actually loaded, so the frontend can disable live RAG up front."""
    return {
        "status": "ok",
        "model_loaded": detector is not None,
        "pipeline_loaded": pipeline is not None,
    }


# --- Frontend (must stay last: see module docstring) -----------------------------------

if FRONTEND_DIR.is_dir():
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
else:  # pragma: no cover - only when the frontend dir is missing from a build
    logger.warning("frontend/ not found at %s; serving the API only", FRONTEND_DIR)
