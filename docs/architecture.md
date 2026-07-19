# Architecture

The real, deployed system (Phase 5/6/ADR-019).

```mermaid
flowchart TD
    User(["User (browser)"]) -->|":8000, same-origin"| FE["frontend/ (index.html + app.js)"]
    FE --> API["FastAPI — api/main.py<br/>/detect · /ask · /presets · /health"]

    API -->|"/detect: context + response"| DET["Track B Detector<br/>src/models/predict.py"]
    API -->|"/ask: question, no_context flag"| PIPE["RAGPipeline<br/>src/rag/pipeline.py"]
    API -->|"/presets"| PRESETS["app/presets.py<br/>cached demo Q&A"]

    PIPE --> RET["Retriever: FAISS + MiniLM embeddings<br/>src/rag/retriever.py"]
    RET -->|"top-k chunks"| CORPUS[("5-mathematician Wikipedia corpus<br/>Gauss, Euler, Riemann, Hilbert, von Neumann")]
    PIPE -->|"question + retrieved context<br/>(withheld from generator if no_context=True)"| GROQ["Groq generator<br/>openai/gpt-oss-20b"]
    GROQ -->|"generated answer"| DET
    RET -->|"retrieved context always passed to detector,<br/>even in no_context ablation mode"| DET
    DET -->|"score + color + char-offset spans"| API

    User2(["User (:7860 fallback)"]) -.-> GRADIO["Gradio app — app/app.py"]
    GRADIO --> PIPE
    GRADIO --> DET

    subgraph COMPOSE["docker compose (docker-compose.yml)"]
        FE
        API
        GRADIO
        CACHE[("hf-cache volume<br/>Track B + tokenizer + MiniLM,<br/>~600MB, pulled on first start")]
    end
    API -. "downloads on first start, reused after" .-> CACHE
    GRADIO -. "downloads on first start, reused after" .-> CACHE
```

## Notes

- **Same-origin, no CORS** (ADR-019): the custom frontend is static files served
  directly by the FastAPI process at `:8000`, not a separate service.
- **Gradio (`:7860`) is a fallback**, kept from Phase 6 alongside the custom frontend,
  not the primary demo.
- **No public hosting** (ADR-018): HF now PRO-gates personal Gradio Spaces, so the
  system runs entirely via local `docker compose up` — there is no public URL.
- **The detector always checks the real retrieved context**, even in `no_context`
  ablation mode (ADR-016) — only the *generator* is blinded to it, so the demo can show
  the detector catching a live hallucination rather than a scripted example.
- **Models are not baked into the Docker image** — they download from the HF Hub into
  the named `hf-cache` volume on first `docker compose up` and are reused after that.
