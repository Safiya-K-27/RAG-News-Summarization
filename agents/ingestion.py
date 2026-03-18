"""Stage 1: Data ingestion across multiple news datasets."""

from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd

from config import AppConfig
from utils.schema import Document
from utils.text_utils import clean_text


class DataIngestionAgent:
    """Loads and preprocesses documents from configured data sources."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def load_documents(self) -> List[Document]:
        """Load from Hugging Face datasets and optional Kaggle CSV, with dummy fallback."""
        docs: List[Document] = []

        if self.config.use_hf_datasets:
            docs.extend(self._load_hf_cnn_dailymail())
            docs.extend(self._load_hf_xsum())
            docs.extend(self._load_hf_multi_news())

        docs.extend(self._load_kaggle_entertainment())

        if not docs:
            docs = self._load_dummy_documents()

        return self.preprocess_documents(docs)

    def preprocess_documents(self, docs: List[Document]) -> List[Document]:
        """Run simple text cleaning and remove empty examples."""
        processed: List[Document] = []
        for d in docs:
            text = clean_text(d.text)
            if not text:
                continue
            d.text = text
            d.title = clean_text(d.title)
            processed.append(d)
        return processed

    def _load_hf_cnn_dailymail(self) -> List[Document]:
        return self._load_hf_dataset(
            dataset_name="cnn_dailymail",
            subset="3.0.0",
            text_field="article",
            title_field=None,
            source="cnn_dailymail",
        )

    def _load_hf_xsum(self) -> List[Document]:
        return self._load_hf_dataset(
            dataset_name="xsum",
            subset=None,
            text_field="document",
            title_field=None,
            source="xsum",
        )

    def _load_hf_multi_news(self) -> List[Document]:
        return self._load_hf_dataset(
            dataset_name="multi_news",
            subset=None,
            text_field="document",
            title_field=None,
            source="multi_news",
        )

    def _load_hf_dataset(
        self,
        dataset_name: str,
        subset: str | None,
        text_field: str,
        title_field: str | None,
        source: str,
    ) -> List[Document]:
        """Best-effort loader for Hugging Face datasets."""
        docs: List[Document] = []
        try:
            from datasets import load_dataset

            if subset:
                ds = load_dataset(dataset_name, subset, split="train")
            else:
                ds = load_dataset(dataset_name, split="train")

            for idx, row in enumerate(ds.select(range(min(self.config.max_docs_per_source, len(ds))))):
                text = str(row.get(text_field, "")).strip()
                if not text:
                    continue
                title = str(row.get(title_field, f"{source} article {idx}")) if title_field else f"{source} article {idx}"
                docs.append(
                    Document(
                        doc_id=f"{source}_{idx}",
                        source=source,
                        title=title,
                        text=text,
                    )
                )
        except Exception:
            # Keep pipeline robust in offline or restricted environments.
            return []
        return docs

    def _load_kaggle_entertainment(self) -> List[Document]:
        """Load Kaggle news category CSV and keep only ENTERTAINMENT rows."""
        docs: List[Document] = []
        csv_path = self.config.kaggle_news_csv_path
        if not csv_path:
            return docs

        p = Path(csv_path)
        if not p.exists():
            return docs

        try:
            df = pd.read_csv(p)
        except Exception:
            return docs

        category_col = None
        for col in ["category", "Category", "SECTION", "section"]:
            if col in df.columns:
                category_col = col
                break
        if not category_col:
            return docs

        text_col = "headline" if "headline" in df.columns else ("short_description" if "short_description" in df.columns else None)
        if not text_col:
            return docs

        ent_df = df[df[category_col].astype(str).str.upper() == "ENTERTAINMENT"].head(self.config.max_docs_per_source)
        for idx, row in ent_df.iterrows():
            title = str(row.get("headline", f"kaggle entertainment {idx}"))
            body = str(row.get("short_description", ""))
            full_text = f"{title}. {body}".strip()
            docs.append(
                Document(
                    doc_id=f"kaggle_ent_{idx}",
                    source="kaggle_news",
                    title=title,
                    text=full_text,
                    metadata={"category": "ENTERTAINMENT"},
                )
            )
        return docs

    def _load_dummy_documents(self) -> List[Document]:
        """Small test corpus to make the pipeline runnable out-of-the-box."""
        return [
            Document(
                doc_id="dummy_1",
                source="dummy",
                title="Shah Rukh Khan announces new film",
                text=(
                    "Shah Rukh Khan announced a new action-drama in Mumbai on Tuesday. "
                    "The production is backed by Red Chillies Entertainment and directed by an acclaimed filmmaker. "
                    "Industry analysts expect a worldwide release in early 2027."
                ),
            ),
            Document(
                doc_id="dummy_2",
                source="dummy",
                title="Streaming platform signs Bollywood stars",
                text=(
                    "A major streaming platform signed multi-film deals with several Bollywood actors in India. "
                    "Executives said the strategy focuses on family-friendly stories and global distribution. "
                    "The announcement followed strong growth in South Asian subscriptions last quarter."
                ),
            ),
            Document(
                doc_id="dummy_3",
                source="dummy",
                title="Film festival expands international lineup",
                text=(
                    "The London film festival expanded its international lineup and added new premieres from Asia. "
                    "Organizers confirmed partnerships with studios and independent producers. "
                    "The event is scheduled for October and will host industry roundtables."
                ),
            ),
        ]
