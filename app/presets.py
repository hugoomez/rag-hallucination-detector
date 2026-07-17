"""The demo's three preset questions -- single source of truth (Phase 6).

These pairs were previously duplicated three ways, in shapes that had drifted apart:
app/app.py::TAB2_PRESETS held (label, question), app/precompute_cache.py::PRESETS held
(question, no_context), and src/rag/pipeline.py::main() held all three inline. Adding a
fourth consumer (api/main.py's /presets endpoint) made one definition worth having.

The API imports this module rather than app/app.py deliberately: app/app.py imports gradio
at module scope, and the API process has no business loading a UI toolkit to learn three
strings.

`id` is written out rather than slugified from `label`, because the labels carry "·" and
parentheses that no slug rule handles gracefully -- and an id that appears in a URL should
be stable when someone rewords a label.

`no_context=True` on the von Neumann preset is the ADR-016 ablation: the retriever still
runs, but the generator never sees the context, so the detector catches a real hallucination
rather than a scripted one. Keep in sync with app/demo_cache.json by re-running
`python app/precompute_cache.py` whenever this list changes.

The cache reader lives here too, for the same reason: it is preset data, and both the Gradio
app and the API need it without either one importing the other.
"""

import json
from dataclasses import dataclass
from pathlib import Path

CACHE_PATH = Path(__file__).resolve().parent / "demo_cache.json"


@dataclass(frozen=True)
class Preset:
    """One cached demo question. `id` is the URL-safe handle the API and frontend use."""

    id: str
    label: str
    question: str
    no_context: bool


PRESETS: list[Preset] = [
    Preset(
        id="grounded-in-corpus",
        label="Grounded · in corpus",
        question="What did Gauss contribute to number theory?",
        no_context=False,
    ),
    Preset(
        id="grounded-out-of-corpus-refusal",
        label="Grounded · out-of-corpus refusal",
        question="Who won the 2022 FIFA World Cup?",
        no_context=False,
    ),
    Preset(
        id="ablation-no-context",
        label="Ablation · no context (live-hallucination demo)",
        question="On what exact date did John von Neumann marry his second wife, Klara Dan?",
        no_context=True,
    ),
]


def load_cache(path: Path = CACHE_PATH) -> dict:
    """Load the precomputed preset cache; return {} if it hasn't been generated yet."""
    path = Path(path)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def get_cached_result(question: str, cache: dict) -> dict | None:
    """Return the stored result dict for a preset question, or None if not cached."""
    return cache.get(question)
