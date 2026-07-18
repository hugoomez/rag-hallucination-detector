PYTHON ?= python

.PHONY: install data train evaluate app test lint

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

# Regenerates data/processed/*.parquet from RAGTruth (data/raw/ is gitignored).
# Run in order: download the raw dataset once, then all 3 preprocessing variants
# (response-level/DeBERTa, response-level/ModernBERT, token-level/ModernBERT).
data:
	$(PYTHON) -m src.data.download
	$(PYTHON) -m src.data.preprocess
	$(PYTHON) -m src.data.preprocess_modernbert
	$(PYTHON) -m src.data.preprocess_token_level

# Fine-tunes all 3 models. Needs a GPU (see scripts/KAGGLE_SETUP.md) -- on CPU
# these are impractically slow. Requires `make data` to have run first.
# Hub pushing is off by default; add --push_to_hub --hub_model_id <repo> to
# any of these to publish (see docs/model_cards/ for the cards to pair with them).
train:
	$(PYTHON) -m src.models.train
	$(PYTHON) -m src.models.train_modernbert
	$(PYTHON) -m src.models.train_token_level

# Reproduces the README's comparison table and ADR-017's ensemble result.
# evaluate_baseline writes results/baseline_nli_metrics.json directly; the
# fine-tuned systems are pulled from the Hub (Kaggle-trained) into
# results/unified_predictions.parquet by collect_predictions.py, one system
# at a time, before tune_threshold_and_ensemble builds the final comparison.
evaluate:
	$(PYTHON) scripts/evaluate_baseline.py
	$(PYTHON) scripts/collect_predictions.py baseline
	$(PYTHON) scripts/collect_predictions.py track_a
	$(PYTHON) scripts/collect_predictions.py approach_1
	$(PYTHON) scripts/collect_predictions.py track_b_modernbert
	$(PYTHON) scripts/tune_threshold_and_ensemble.py

# Starts the full demo (custom frontend + API on :8000, Gradio fallback on :7860).
# Needs Docker + Docker Compose, and GROQ_API_KEY in .env (see README).
app:
	docker compose up

test:
	pytest -q

lint:
	ruff check src/ tests/
	black --check src/ tests/
