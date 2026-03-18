# Agent-Orchestrated Personalized News Generator

A modular, production-ready Python pipeline for personalized news summarization using:
- RAG with hybrid retrieval (semantic + keyword + entity filtering)
- Hierarchical chunking (paragraph and sentence levels)
- NER and entity normalization
- Event pattern extraction and genetic optimization
- Adversarial defense and post-generation fact checking

## Project Structure

```text
project/
|-- data/
|   |-- raw/
|   |-- processed/
|-- agents/
|   |-- ingestion.py
|   |-- chunking.py
|   |-- ner.py
|   |-- retrieval.py
|   |-- event_extraction.py
|   |-- evolution.py
|   |-- defense.py
|   |-- summarizer.py
|   |-- personalization.py
|   |-- fact_check.py
|   |-- training.py
|-- utils/
|   |-- schema.py
|   |-- text_utils.py
|-- config.py
|-- main.py
|-- colab_bootstrap.py
|-- requirements-colab.txt
|-- requirements.txt
```

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

3. Create `.env` (or copy from `.env.example`) and configure keys/paths.

## Run

```bash
python main.py
```

By default, the project uses built-in dummy/test data so it runs without external datasets.
Set `USE_HF_DATASETS=true` to ingest Hugging Face datasets.

## Full Model Training (Train Then Infer)

Enable complete model training with environment flags:

```bash
export RUN_TRAINING=true
export USE_HF_DATASETS=true
export MAX_DOCS_PER_SOURCE=500
python main.py
```

When training mode is enabled, the pipeline performs:
- Retriever fine-tuning from summary-document pairs
- Summarizer fine-tuning with base + domain pairs
- Checkpoint saving under `checkpoints/retriever` and `checkpoints/summarizer`
- Inference using trained checkpoints in the same run

For Colab, set the same environment variables before running `python colab_bootstrap.py`.

## Optional: Kaggle News Category Dataset

Set `KAGGLE_NEWS_CSV_PATH` in `.env` and the ingestion stage will filter only ENTERTAINMENT rows.

## Google Colab

Fast path (Option 1):

```python
%cd /content/Project
!python colab_bootstrap.py
```

To match the user-input flow (topic, reading level, neutrality, summary length), set environment variables before running:

```python
import os
os.environ["DEFAULT_NEWS_TOPIC"] = "latest entertainment awards and streaming partnerships"
os.environ["DEFAULT_READING_LEVEL"] = "simple"  # simple | medium | advanced
os.environ["DEFAULT_BIAS_CONTROL"] = "neutral"  # neutral | balanced
os.environ["DEFAULT_SUMMARY_LENGTH"] = "short"  # short | medium | long
```

For complete train-then-infer runs in Colab:

```python
os.environ["RUN_TRAINING"] = "true"
os.environ["USE_HF_DATASETS"] = "true"
os.environ["MAX_DOCS_PER_SOURCE"] = "500"
!python main.py
```

## Performance Notes

- NER uses batched processing with `spaCy nlp.pipe`.
- Retrieval scoring is vectorized and computes top-k with `numpy argpartition`.
- Embedding batch size is configurable via `EMBEDDING_BATCH_SIZE`.
