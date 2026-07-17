"""Offline unit tests for the Phase 6 Gradio demo helpers (Step 6.2).

Full Gradio UI interaction isn't practically unit-testable the way our other components are,
so these tests target the pure functions the Gradio callbacks delegate to -- highlight
construction, verdict formatting, context rendering, and cache lookup -- not Gradio itself.

Hermetic by the same convention as test_predict.py / test_api.py: importing app.app builds
no UI and loads no model (load_models() runs only under __main__), so no checkpoint is
downloaded and no Groq key is needed. get_cached_result is tested against an in-memory dict,
not the checked-in demo_cache.json, so the tests don't depend on that file existing.
"""

import pytest

from app.app import (
    HALLUCINATION_LABEL,
    build_highlight,
    format_contexts,
    format_verdict,
    get_cached_result,
)

GREEN, YELLOW, RED = "🟢", "🟡", "🔴"


# --- build_highlight ------------------------------------------------------------------


def test_build_highlight_maps_spans_to_entities():
    response = "Gauss published Disquisitiones Arithmeticae in 1801 and won the Fields Medal in 1820."
    spans = [{"start": 51, "end": 85, "text": response[51:85]}]

    out = build_highlight(response, spans)

    assert out["text"] == response  # full response preserved verbatim for the widget
    assert out["entities"] == [{"entity": HALLUCINATION_LABEL, "start": 51, "end": 85}]


def test_build_highlight_empty_spans_yields_no_entities():
    out = build_highlight("A fully supported answer.", [])

    assert out == {"text": "A fully supported answer.", "entities": []}


def test_build_highlight_offsets_slice_back_to_span_text():
    response = "He won the Fields Medal in 1820 and also the Abel Prize."
    spans = [
        {"start": 7, "end": 31, "text": response[7:31]},
        {"start": 45, "end": 55, "text": response[45:55]},
    ]

    out = build_highlight(response, spans)

    # The widget re-slices response[start:end]; that must equal the detector's span text.
    for span, entity in zip(spans, out["entities"]):
        assert out["text"][entity["start"] : entity["end"]] == span["text"]


def test_build_highlight_drops_label_field_never_leaks():
    spans = [{"start": 0, "end": 3, "text": "abc"}]

    entity = build_highlight("abcdef", spans)["entities"][0]

    assert set(entity) == {"entity", "start", "end"}  # no per-span "label" invented


# --- format_verdict -------------------------------------------------------------------


@pytest.mark.parametrize(
    "color,label",
    [(RED, "Likely hallucination"), (YELLOW, "Borderline"), (GREEN, "Looks supported")],
)
def test_format_verdict_renders_color_label_and_score(color, label):
    banner = format_verdict({"color": color, "score": 0.5})

    assert color in banner
    assert label in banner
    assert "0.500" in banner  # score formatted to 3 dp


# --- format_contexts ------------------------------------------------------------------


def test_format_contexts_lists_sources_and_scores():
    contexts = [
        {"source": "gauss.txt", "text": "Gauss was a mathematician.", "score": 0.71},
        {"source": "riemann.txt", "text": "Riemann studied analysis.", "score": 0.42},
    ]

    md = format_contexts(contexts)

    assert "gauss.txt" in md and "riemann.txt" in md
    assert "0.710" in md
    assert "Ablation mode" not in md  # no ablation banner when ablation=False


def test_format_contexts_shows_ablation_banner():
    contexts = [{"source": "von_neumann.txt", "text": "von Neumann text.", "score": 0.6}]

    md = format_contexts(contexts, ablation=True)

    assert "Ablation mode" in md
    assert "did NOT see" in md


# --- get_cached_result ----------------------------------------------------------------


def test_get_cached_result_hit_and_miss():
    cache = {"What did Gauss contribute?": {"answer": "number theory", "verdict": {}}}

    assert get_cached_result("What did Gauss contribute?", cache) == {
        "answer": "number theory",
        "verdict": {},
    }
    assert get_cached_result("Unknown question", cache) is None
