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
|   |-- personalization.py
|   |-- fact_check.py
|   |-- training.py
|-- utils/
|   |-- schema.py
|   |-- text_utils.py
|-- config.py
|-- GenAI_Notebook.ipynb
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

Use [GenAI_Notebook.ipynb](GenAI_Notebook.ipynb) as the final execution entrypoint.

In Colab:
1. Open [GenAI_Notebook.ipynb](GenAI_Notebook.ipynb).
2. Run cells in order.
3. Provide your topic, reading level, and summary length when prompted.

## Train Once, Then Serve Frontend

This project now supports a persistent workflow:
- Train once and save checkpoints.
- Launch a simple frontend for user inputs and summary generation.

### 1) Train and save checkpoints

In Colab:

```python
%cd /content/Project
!pip install -q -r requirements-colab.txt
!python -m spacy download en_core_web_sm
```

Set environment values before training:

```python
import os
os.environ["RUN_TRAINING"] = "true"
os.environ["USE_HF_DATASETS"] = "true"
os.environ["MAX_DOCS_PER_SOURCE"] = "500"
os.environ["KAGGLE_NEWS_CSV_PATH"] = "/content/Project/data/raw/news_category_entertainment.csv"
os.environ["TRAINING_OUTPUT_DIR"] = "/content/drive/MyDrive/rag_news_checkpoints"
```

Run training:

```python
!python train_once.py
```

### 2) Launch frontend (no retraining)

```python
import os
os.environ["RUN_TRAINING"] = "false"
os.environ["USE_HF_DATASETS"] = "true"
os.environ["MAX_DOCS_PER_SOURCE"] = "500"
os.environ["KAGGLE_NEWS_CSV_PATH"] = "/content/Project/data/raw/news_category_entertainment.csv"
os.environ["TRAINING_OUTPUT_DIR"] = "/content/drive/MyDrive/rag_news_checkpoints"

!python frontend_app.py
```

The frontend asks for:
- News topic
- Reading level (`simple` or `advanced`)
- Summary length (`short` or `long`)
- Neutrality preference (`neutral` or `balanced`)

## Full Model Training (Train Then Infer)

Enable complete model training by setting environment variables in notebook cells before the training cell.

When training mode is enabled, the pipeline performs:
- Retriever fine-tuning from summary-document pairs
- Summarizer fine-tuning with base + domain pairs
- Checkpoint saving under `checkpoints/retriever` and `checkpoints/summarizer`
- Inference using trained checkpoints in the same run

For Colab, set the same environment variables directly in notebook cells.

## Optional: Kaggle News Category Dataset

Set `KAGGLE_NEWS_CSV_PATH` in `.env` and the ingestion stage will filter only ENTERTAINMENT rows.

## Google Colab

Fast path:
1. Clone the repository in Colab.
2. Install dependencies from `requirements-colab.txt`.
3. Run notebook cells in [GenAI_Notebook.ipynb](GenAI_Notebook.ipynb).

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
# then run the notebook training/inference cell
```

## Performance Notes

- NER uses batched processing with `spaCy nlp.pipe`.
- Retrieval scoring is vectorized and computes top-k with `numpy argpartition`.
- Embedding batch size is configurable via `EMBEDDING_BATCH_SIZE`.
