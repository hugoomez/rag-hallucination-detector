\# Architecture Decision Records (ADR)



Log of key technical decisions made throughout the project, with their rationale.



\---



\## ADR-001: Use DeBERTa-v3-base as the base encoder



\*\*Context:\*\* Need to choose a pretrained transformer encoder to fine-tune for hallucination

detection, framed as an NLI-style classification task (premise = retrieved context,

hypothesis = answer claim).



\*\*Decision:\*\* Use `microsoft/deberta-v3-base`.



\*\*Rationale:\*\*

\- He et al. (2021) show DeBERTa consistently outperforms BERT, RoBERTa, and XLNet

&#x20; specifically on MNLI (the NLI benchmark), not just on average across tasks.

\- DeBERTa's disentangled attention mechanism separates content and position

&#x20; representations, which helps capture syntactic nuances (subject vs. object, negation)

&#x20; relevant to detecting contradiction vs. entailment.

\- v3 (vs. v1) uses ELECTRA-style Replaced Token Detection pretraining instead of MLM,

&#x20; giving a training signal on 100% of tokens instead of only the masked 15% — a stronger

&#x20; pretrained model at the same size.

\- `base` size runs comfortably on a free Colab T4 GPU (16GB), no paid compute required.



\*\*Alternatives considered:\*\* RoBERTa-base (weaker on MNLI per the DeBERTa paper).



\---



\## ADR-002: Train on Google Colab (free tier)



\*\*Context:\*\* Need to decide where to run training.



\*\*Decision:\*\* Google Colab, free T4 GPU tier.



\*\*Rationale:\*\* Sufficient VRAM (16GB) for DeBERTa-v3-base fine-tuning; no cost; no local

GPU available. Kaggle Notebooks considered as a backup option if Colab quota runs out.



\---



\## ADR-003: Repository language — English



\*\*Context:\*\* The project is a public portfolio piece.



\*\*Decision:\*\* All code, comments, commit messages, and documentation in English; working

conversations with the AI copilot in Spanish.



\*\*Rationale:\*\* English is the industry standard for public repos reviewed by recruiters

and other engineers.

