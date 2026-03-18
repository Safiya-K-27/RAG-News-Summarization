"""Stage 3: NER extraction and entity normalization."""

from __future__ import annotations

from typing import Dict, List

import spacy

from config import AppConfig
from utils.schema import Chunk


class NERAgent:
    """Extracts and normalizes entities for each chunk."""

    TARGET_LABELS = {"PERSON", "ORG", "GPE", "DATE"}

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.aliases = {k.lower(): v for k, v in config.entity_aliases.items()}
        self.nlp = self._load_spacy_model(config.spacy_model)

    def annotate_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        texts = [chunk.text for chunk in chunks]
        try:
            docs = self.nlp.pipe(texts, batch_size=64)
        except Exception:
            docs = (self.nlp(text) for text in texts)

        for chunk, doc in zip(chunks, docs):
            entities = {label: [] for label in self.TARGET_LABELS}
            try:
                for ent in doc.ents:
                    if ent.label_ in self.TARGET_LABELS:
                        entities[ent.label_].append(self.normalize_entity(ent.text))
            except Exception:
                # Keep chunk even if entity extraction fails.
                pass

            chunk.entities = {k: sorted(set(v)) for k, v in entities.items() if v}
        return chunks

    def normalize_entity(self, entity_text: str) -> str:
        key = entity_text.strip().lower()
        return self.aliases.get(key, entity_text.strip())

    @staticmethod
    def _load_spacy_model(model_name: str):
        try:
            return spacy.load(model_name)
        except Exception:
            # Fallback tokenizer-only pipeline to avoid hard failures.
            return spacy.blank("en")
