\# Theory Checklist



You don't need to master everything before starting to code, but you should build a solid

base for each block below. Study this in parallel with the coding phases. Priority resources

are marked with ⭐.



\## A. Natural Language Inference (NLI) — the conceptual core of the detector



Key idea: given a \*premise\* and a \*hypothesis\*, does the hypothesis follow (entailment),

contradict (contradiction), or is it neutral relative to the premise? In this project,

`premise = retrieved context` and `hypothesis = claim in the answer`. A hallucinated claim

tends to be \*contradiction\* or \*neutral\* (unsupported).



\- ⭐ Bowman et al. (2015), \*A large annotated corpus for learning natural language inference\* (SNLI). https://arxiv.org/abs/1508.05326

\- Williams et al. (2018), \*A Broad-Coverage Challenge Corpus for Sentence Understanding through Inference\* (MultiNLI). https://arxiv.org/abs/1704.05426

\- ⭐ Jurafsky \& Martin, \*Speech and Language Processing\* (3rd ed. draft), chapter on semantic inference. https://web.stanford.edu/\~jurafsky/slp3/

\- MoritzLaurer's model cards and blog on zero-shot NLI with DeBERTa: https://huggingface.co/MoritzLaurer



\## B. DeBERTa and why it's strong at NLI



DeBERTa improves on BERT/RoBERTa with \*\*disentangled attention\*\* (separates content and

position representations) and an \*enhanced mask decoder\*. DeBERTa-v3 adds ELECTRA-style

pretraining. It's among the strongest encoders for NLI at moderate size.



\- ⭐ He et al. (2021), \*DeBERTa: Decoding-enhanced BERT with Disentangled Attention\* (ICLR). https://arxiv.org/abs/2006.03654

\- He et al. (2021), \*DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing\*. https://arxiv.org/abs/2111.09543

\- Model card: https://huggingface.co/microsoft/deberta-v3-base



\## C. Fine-tuning transformers with Hugging Face (review this well)



The rustiest but most important part for an ML Engineer profile. Master: loading

tokenizer/model, `datasets`, tokenization with `truncation`/`padding`, the `Trainer` API with

`TrainingArguments`, `compute\_metrics`, and saving/uploading to the Hub.



\- ⭐ Hugging Face NLP Course, chapter 3 (\*Fine-tuning a pretrained model\*) and chapter 7 (\*Token classification\*, for the token-level track in Phase 3). https://huggingface.co/learn/nlp-course

\- ⭐ Tutorial \*Fine-tune a pretrained model\*: https://huggingface.co/docs/transformers/training

\- `Trainer` and `TrainingArguments` docs: https://huggingface.co/docs/transformers/main\_classes/trainer



\## D. RAG (Retrieval-Augmented Generation)



Understand the two components (retriever + generator), why RAG is used (giving the LLM

up-to-date context), and \*\*where hallucinations come from\*\*: the generator can ignore,

contradict, or invent beyond the retrieved context (a \*faithfulness\* failure).



\- ⭐ Lewis et al. (2020), \*Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks\* (original RAG paper). https://arxiv.org/abs/2005.11401

\- Gao et al. (2023), \*Retrieval-Augmented Generation for Large Language Models: A Survey\*. https://arxiv.org/abs/2312.10997

\- Practical docs of LlamaIndex or LangChain (for Phase 5).



\## E. Hallucination taxonomy (faithfulness vs. factuality)



\- \*\*Faithfulness\*\*: is the answer consistent with the given context? \*This is what your

&#x20; detector measures in RAG.\*

\- \*\*Factuality\*\*: is the answer true in the real world? (independent of the context).

\- \*\*Intrinsic vs. extrinsic\*\*: contradicts the context vs. adds unverifiable info.



Resources:

\- ⭐ Ji et al. (2023), \*Survey of Hallucination in Natural Language Generation\* (ACM Computing Surveys). https://arxiv.org/abs/2202.03629

\- Huang et al. (2023), \*A Survey on Hallucination in Large Language Models\*. https://arxiv.org/abs/2311.05232

\- Maynez et al. (2020), \*On Faithfulness and Factuality in Abstractive Summarization\*. https://arxiv.org/abs/2005.00661

\- RAGTruth's specific 4-type taxonomy is detailed in Phase 1.



\## F. Evaluation metrics



\- Classification: \*precision, recall, F1\* and why \*\*F1\*\* matters more than \*accuracy\* when

&#x20; classes are imbalanced (hallucinations are usually the minority class).

\- Span/token-level: sequence metrics with `seqeval` (F1 over BIO spans).

\- Resources: scikit-learn docs on classification metrics

&#x20; (https://scikit-learn.org/stable/modules/model\_evaluation.html) and Hugging Face's

&#x20; `evaluate` library.

