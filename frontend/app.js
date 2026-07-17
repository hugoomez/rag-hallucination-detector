/* RAG Hallucination Detector — frontend behaviour (ADR-019)
 *
 * Same-origin with the API, so every call is a relative path and no CORS is involved.
 *
 * Both modes converge on renderReadout(): "paste your own" supplies the evidence and claim
 * by hand, live RAG has the pipeline supply them, and from that point the examination is
 * the same act — one gauge, one set of marks, one renderer.
 *
 * The wire contract this file is written against (see api/main.py):
 *   verdict = {score: float, color: "🟢"|"🟡"|"🔴", spans: [{start, end, text}]}
 * Spans are CHARACTER offsets into the claim, half-open, unstripped, with no per-span label
 * and no per-span score. That shapes two decisions below: the claim is rendered as plain
 * text (never markdown — parsing it would desynchronise the offsets), and every mark is
 * styled identically because the model gives us nothing to rank them by.
 */

const VERDICT_LABELS = {
  "🔴": "Likely hallucination",
  "🟡": "Borderline",
  "🟢": "Looks supported",
};

const VERDICT_VARS = {
  "🔴": "var(--signal-red)",
  "🟡": "var(--signal-amber)",
  "🟢": "var(--signal-green)",
};

// Real (context, response) pairs with real verdicts — mirrors app/app.py::TAB1_EXAMPLES.
const EXAMPLES = [
  {
    chip: "🟢 supported",
    context:
      "Carl Friedrich Gauss (1777-1855) was a German mathematician. He made major " +
      "contributions to number theory, including his 1801 work Disquisitiones " +
      "Arithmeticae, and proved the fundamental theorem of algebra.",
    response:
      "Gauss contributed to number theory and published Disquisitiones Arithmeticae in 1801.",
  },
  {
    chip: "🔴 invented award",
    context:
      "Carl Friedrich Gauss (1777-1855) was a German mathematician who made major " +
      "contributions to number theory and published Disquisitiones Arithmeticae in 1801.",
    response:
      "Gauss published Disquisitiones Arithmeticae in 1801 and won the Fields Medal in 1820.",
  },
  {
    chip: "🔴 invented biography",
    context:
      "Marie Curie was a physicist and chemist who conducted pioneering research on " +
      "radioactivity. She was born in Warsaw in 1867 and won two Nobel Prizes.",
    response: "Marie Curie was born in Paris in 1901 and personally invented the telephone.",
  },
  {
    chip: "🔴 false positive",
    note:
      "The fourth example is a known false positive: that answer is correct, and the " +
      "detector is over-sensitive to paraphrase. It ships here because a demo that only " +
      "shows the wins is a brochure (see docs/notes.md).",
    context:
      "Bernhard Riemann was born in 1826 in Breselenz. He was the second of six children " +
      "of his father Friedrich Bernhard Riemann, a pastor.",
    response: "Riemann was the second of six children, meaning he had five siblings.",
  },
];

const $ = (id) => document.getElementById(id);

const el = {
  status: $("status"),
  examine: $("examine"),
  pasteContext: $("paste-context"),
  pasteResponse: $("paste-response"),
  examples: $("examples"),
  exampleNote: $("example-note"),
  question: $("question"),
  ask: $("ask"),
  blindedNote: $("blinded-note"),
  presetsWrap: $("presets-wrap"),
  presets: $("presets"),
  evidence: $("evidence"),
  evidenceBody: $("evidence-body"),
  evidenceFlag: $("evidence-flag"),
  readout: $("readout"),
  verdictLabel: $("verdict-label"),
  needle: $("needle"),
  score: $("score"),
  markedBody: $("marked-body"),
  markedNote: $("marked-note"),
  spanCount: $("span-count"),
  notice: $("notice"),
};

let blinded = false;
let pipelineReady = false;

/* --- Rendering --------------------------------------------------------------------- */

/**
 * Wrap each detector span in a <mark>, by character offset.
 *
 * Escapes before wrapping (claims contain markdown and could contain "<"), walks the spans
 * in offset order, and skips any span overlapping one already emitted so a malformed pair
 * can't produce broken nesting. Returns HTML.
 */
function renderMarks(text, spans) {
  const esc = (s) =>
    s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

  let out = "";
  let cursor = 0;
  let i = 0;

  for (const span of [...spans].sort((a, b) => a.start - b.start)) {
    if (span.start < cursor) continue;
    out += esc(text.slice(cursor, span.start));
    out += `<mark class="mark" style="--i:${i++}">${esc(text.slice(span.start, span.end))}</mark>`;
    cursor = span.end;
  }
  return out + esc(text.slice(cursor));
}

/** Render "chars 21–29, 44–61" — the offsets are the model's output, so they are shown. */
function formatSpanCount(spans) {
  if (!spans.length) return "no spans";
  const ranges = spans.map((s) => `${s.start}–${s.end}`).join(", ");
  return `${spans.length} span${spans.length > 1 ? "s" : ""} · chars ${ranges}`;
}

/** The shared readout: gauge + marked claim. Both modes end here. */
function renderReadout(claim, verdict) {
  hideNotice();
  const color = VERDICT_VARS[verdict.color] || "var(--ink)";
  el.readout.hidden = false;
  el.readout.style.setProperty("--verdict", color);

  el.verdictLabel.textContent = `${VERDICT_LABELS[verdict.color] || "Unknown"} ${verdict.color}`;
  el.verdictLabel.style.color = color;

  el.score.textContent = verdict.score.toFixed(3);
  el.spanCount.textContent = formatSpanCount(verdict.spans);
  el.markedBody.innerHTML = renderMarks(claim, verdict.spans);

  // An empty result is a finding, not a blank space — say what the model concluded.
  el.markedNote.hidden = verdict.spans.length > 0;
  el.markedNote.textContent =
    "No passages marked. The detector found nothing in this claim that the evidence " +
    "does not support.";

  const pct = Math.min(Math.max(verdict.score, 0), 1);

  // Keep the value legible at the extremes: centred on the needle normally, but tucked
  // inside the track once the needle nears either end (a 0.988 verdict is the common case).
  el.score.style.transform =
    pct > 0.92 ? "translateX(-100%)" : pct < 0.08 ? "translateX(0)" : "translateX(-50%)";

  // Park the needle at 0, then let a frame elapse so the browser has a start value to
  // animate from — otherwise it jumps straight to the score with no travel.
  el.needle.style.left = "0%";
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      el.needle.style.left = `${pct * 100}%`;
    });
  });
}

function renderEvidence(contexts, ablation) {
  el.evidence.hidden = false;
  el.evidence.classList.toggle("evidence--blinded", ablation);
  el.evidenceFlag.hidden = !ablation;
  el.evidenceBody.replaceChildren(
    ...contexts.map((c) => {
      const node = document.createElement("div");
      node.className = "chunk";
      const head = document.createElement("div");
      head.className = "chunk__head mono";
      const src = document.createElement("span");
      src.textContent = c.source;
      const sim = document.createElement("span");
      sim.textContent = c.score.toFixed(3);
      head.append(src, sim);
      const body = document.createElement("p");
      body.className = "chunk__text prose";
      body.textContent = c.text.slice(0, 400).trim() + (c.text.length > 400 ? "…" : "");
      node.append(head, body);
      return node;
    })
  );
}

function renderRagResult(result) {
  renderEvidence(result.contexts, result.ablation);
  renderReadout(result.answer, result.verdict);
}

/* --- Notices ----------------------------------------------------------------------- */

function showNotice(title, message, tone = "error") {
  el.notice.hidden = false;
  el.notice.dataset.tone = tone;
  el.notice.replaceChildren();
  const h = document.createElement("div");
  h.className = "notice__title";
  h.textContent = title;
  const p = document.createElement("p");
  p.style.margin = "0";
  p.textContent = message;
  el.notice.append(h, p);
}

function hideNotice() {
  el.notice.hidden = true;
}

const NO_KEY_NOTICE =
  "GROQ_API_KEY is not set, so no new answers can be generated. The cached questions below " +
  "still work — they were precomputed and never call the API. To ask your own, add the key " +
  "to .env and restart.";

// Titles are the interface's voice: they name what happened, and never apologise.
const ERROR_TITLES = {
  rate_limit: "Rate limit reached",
  connection: "Could not reach Groq",
  api_error: "Generation failed",
  empty_completion: "Generation failed",
  no_key: "Live RAG unavailable",
};

/**
 * Turn a failed response into a styled notice.
 *
 * Handles both detail shapes the API emits: /ask sends {kind, message} so failures can be
 * told apart, while /detect sends a plain string. Never dumps raw JSON at the reader.
 */
async function reportFailure(res) {
  let detail;
  try {
    detail = (await res.json()).detail;
  } catch {
    detail = null;
  }

  if (detail && typeof detail === "object" && detail.message) {
    const tone = detail.kind === "rate_limit" ? "wait" : "error";
    showNotice(ERROR_TITLES[detail.kind] || "Request failed", detail.message, tone);
    return;
  }
  if (res.status === 503) {
    showNotice(
      "Model not loaded",
      "The detector checkpoint could not be loaded, so nothing can be examined. Check the " +
        "server logs, then reload this page.",
    );
    return;
  }
  showNotice(
    "Request failed",
    typeof detail === "string" ? detail : `The server returned ${res.status}.`,
  );
}

/* --- Actions ----------------------------------------------------------------------- */

async function withPending(button, label, fn) {
  const original = button.textContent;
  button.disabled = true;
  button.textContent = label;
  try {
    await fn();
  } catch {
    showNotice(
      "Could not reach the server",
      "The request never completed. Check that the API is still running, then try again.",
    );
  } finally {
    button.disabled = false;
    button.textContent = original;
  }
}

async function examine() {
  const context = el.pasteContext.value.trim();
  const response = el.pasteResponse.value.trim();
  if (!context || !response) {
    showNotice("Nothing to examine", "Fill in both the evidence and the claim, then examine.", "wait");
    return;
  }
  await withPending(el.examine, "Examining", async () => {
    const res = await fetch("/detect", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ context, response }),
    });
    if (!res.ok) return reportFailure(res);
    renderReadout(response, await res.json());
  });
}

async function ask() {
  const question = el.question.value.trim();
  if (!question) {
    showNotice("No question yet", "Type a question about the corpus, then ask.", "wait");
    return;
  }
  await withPending(el.ask, "Asking", async () => {
    const res = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, no_context: blinded }),
    });
    if (!res.ok) return reportFailure(res);
    renderRagResult(await res.json());
  });
}

/* --- Wiring ------------------------------------------------------------------------ */

function selectMode(mode) {
  for (const tab of document.querySelectorAll(".mode")) {
    tab.setAttribute("aria-selected", String(tab.dataset.mode === mode));
  }
  $("panel-paste").hidden = mode !== "paste";
  $("panel-rag").hidden = mode !== "rag";
  el.readout.hidden = true;
  el.evidence.hidden = true;
  hideNotice();

  // Explain a missing key on arrival, rather than leaving the reader to deduce it from a
  // greyed-out button.
  if (mode === "rag" && !pipelineReady) {
    showNotice(ERROR_TITLES.no_key, NO_KEY_NOTICE, "wait");
  }
}

function buildExamples() {
  el.examples.replaceChildren(
    ...EXAMPLES.map((ex) => {
      const b = document.createElement("button");
      b.className = "chip";
      b.type = "button";
      b.textContent = ex.chip;
      b.addEventListener("click", () => {
        el.pasteContext.value = ex.context;
        el.pasteResponse.value = ex.response;
        el.exampleNote.textContent = ex.note || "";
        el.readout.hidden = true;
        hideNotice();
      });
      return b;
    })
  );
}

/** Presets render from the payload fetched at load: clicking one makes no request. */
function buildPresets(presets) {
  if (!presets.length) return;
  el.presetsWrap.hidden = false;
  el.presets.replaceChildren(
    ...presets.map((p) => {
      const b = document.createElement("button");
      b.className = "chip";
      b.type = "button";
      b.textContent = p.label;
      b.addEventListener("click", () => {
        el.question.value = p.question;
        setBlinded(p.result.ablation);
        renderRagResult(p.result);
      });
      return b;
    })
  );
}

function setBlinded(value) {
  blinded = value;
  for (const opt of document.querySelectorAll(".switch__opt")) {
    opt.setAttribute("aria-checked", String((opt.dataset.blinded === "true") === value));
  }
  el.blindedNote.hidden = !value;
}

function setStatus(health) {
  const ok = health && health.model_loaded;
  el.status.dataset.state = ok ? "ok" : "down";
  el.status.querySelector(".status__text").textContent = ok ? "model loaded" : "model unavailable";

  pipelineReady = Boolean(health && health.pipeline_loaded);
  el.examine.disabled = !ok;
  el.ask.disabled = !pipelineReady;
  el.question.disabled = !pipelineReady;
}

async function init() {
  buildExamples();

  for (const tab of document.querySelectorAll(".mode")) {
    tab.addEventListener("click", () => selectMode(tab.dataset.mode));
  }
  for (const opt of document.querySelectorAll(".switch__opt")) {
    opt.addEventListener("click", () => setBlinded(opt.dataset.blinded === "true"));
  }
  el.examine.addEventListener("click", examine);
  el.ask.addEventListener("click", ask);
  el.question.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !el.ask.disabled) ask();
  });

  const [health, presets] = await Promise.all([
    fetch("/health").then((r) => r.json()).catch(() => null),
    fetch("/presets").then((r) => r.json()).catch(() => ({ presets: [] })),
  ]);

  setStatus(health);
  buildPresets(presets.presets || []);
}

init();
