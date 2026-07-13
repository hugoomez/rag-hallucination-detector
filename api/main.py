"""FastAPI wrapper around the Track B span-level Detector (Phase 6, Step 6.1).

Exposes src/models/predict.py's Detector over HTTP so the Phase 6 demo can call it:
  - POST /detect  {context, response} -> {score, color, spans:[{text, start, end}]}
  - GET  /health  -> {status, model_loaded}

The response schema mirrors Detector.predict()'s real dict exactly: `color` is a
traffic-light emoji glyph (not a color name), and each span carries start/end/text
only -- there is no per-span "label" field (the token-level model doesn't produce one,
unlike the sequence-classification stub the phase doc assumed).

The model is loaded ONCE at startup via FastAPI's lifespan context manager (the current
pattern; @app.on_event("startup") is deprecated). Loading is best-effort: if the
checkpoint can't be fetched (e.g. no network), the app still starts, /health reports
model_loaded=false, and /detect returns 503, so the failure is observable rather than
crashing the process.

Both endpoints resolve the Detector through the single get_detector dependency, which
tests override with a fake so the real checkpoint is never downloaded (same
hermetic-by-default convention as tests/test_predict.py and tests/test_pipeline.py).
"""

import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel

from src.models.predict import Detector

logger = logging.getLogger(__name__)


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


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


# --- App + model lifecycle ------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the Detector once at startup; degrade gracefully if it can't load."""
    try:
        app.state.detector = Detector.from_pretrained()
    except Exception:
        logger.exception("Detector failed to load at startup; /detect will return 503")
        app.state.detector = None
    yield
    app.state.detector = None


app = FastAPI(title="RAG Hallucination Detector", lifespan=lifespan)


def get_detector(request: Request) -> Detector | None:
    """Return the loaded Detector, or None if it failed to load.

    The single injection seam for the API: tests override this to inject a fake
    detector, so the real checkpoint is never downloaded and lifespan never runs.
    """
    return getattr(request.app.state, "detector", None)


# --- Endpoints ------------------------------------------------------------------------


@app.post("/detect", response_model=DetectResponse)
def detect(req: DetectRequest, detector: Detector | None = Depends(get_detector)) -> dict:
    if detector is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return detector.predict(req.context, req.response)


@app.get("/health", response_model=HealthResponse)
def health(detector: Detector | None = Depends(get_detector)) -> dict:
    return {"status": "ok", "model_loaded": detector is not None}
