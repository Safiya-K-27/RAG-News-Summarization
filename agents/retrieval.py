"""Stages 4 and 5: Embeddings, vector store, and hybrid RAG retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from config import AppConfig
from utils.schema import Chunk, RetrievedChunk
from utils.text_utils import normalize_token


@dataclass
class VectorStoreState:
    chunks: List[Chunk]
    embeddings: np.ndarray


class HybridRetrievalAgent:
    """Combines semantic similarity, keyword relevance, and entity filtering."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.state: Optional[VectorStoreState] = None
        self._embedder = None
        self._faiss_index = None
        self._tfidf = TfidfVectorizer(stop_words="english")
        self._tfidf_matrix = None

    def build_index(self, chunks: List[Chunk]) -> None:
        if not chunks:
            raise ValueError("No chunks available for indexing.")

        texts = [c.text for c in chunks]
        embeddings = self._encode_texts(texts)
        embeddings = self._l2_normalize(embeddings)
        self.state = VectorStoreState(chunks=chunks, embeddings=embeddings)

        self._build_faiss_index(embeddings)
        self._tfidf_matrix = self._tfidf.fit_transform([normalize_token(t) for t in texts])

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        required_entities: Optional[Dict[str, List[str]]] = None,
    ) -> List[RetrievedChunk]:
        if self.state is None:
            raise RuntimeError("Index not built. Call build_index first.")

        top_k = top_k or self.config.top_k_retrieval
        query_emb = self._l2_normalize(self._encode_texts([query]))

        semantic_scores = self._semantic_search(query_emb, len(self.state.chunks))
        keyword_scores = self._keyword_search(query)
        entity_scores = np.array(
            [self._entity_score(chunk, required_entities) for chunk in self.state.chunks],
            dtype=np.float32,
        )
        final_scores = 0.55 * semantic_scores + 0.35 * keyword_scores + 0.10 * entity_scores

        safe_top_k = min(top_k, len(self.state.chunks))
        if safe_top_k <= 0:
            return []

        top_indices = np.argpartition(final_scores, -safe_top_k)[-safe_top_k:]
        top_indices = top_indices[np.argsort(final_scores[top_indices])[::-1]]
        combined: List[RetrievedChunk] = []

        for idx in top_indices:
            chunk = self.state.chunks[int(idx)]
            combined.append(
                RetrievedChunk(
                    chunk=chunk,
                    semantic_score=float(semantic_scores[idx]),
                    keyword_score=float(keyword_scores[idx]),
                    entity_score=float(entity_scores[idx]),
                    final_score=float(final_scores[idx]),
                )
            )

        return combined

    def _encode_texts(self, texts: List[str]) -> np.ndarray:
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer

            self._embedder = SentenceTransformer(self.config.embedding_model)
        emb = self._embedder.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=self.config.embedding_batch_size,
        )
        return emb.astype(np.float32)

    @staticmethod
    def _l2_normalize(x: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-9
        return x / norms

    def _build_faiss_index(self, embeddings: np.ndarray) -> None:
        try:
            import faiss

            dim = embeddings.shape[1]
            index = faiss.IndexFlatIP(dim)
            index.add(embeddings)
            self._faiss_index = index
        except Exception:
            self._faiss_index = None

    def _semantic_search(self, query_emb: np.ndarray, k: int) -> np.ndarray:
        if self._faiss_index is not None:
            scores, indices = self._faiss_index.search(query_emb, k)
            dense = np.zeros((len(self.state.chunks),), dtype=np.float32)
            for rank, idx in enumerate(indices[0]):
                dense[idx] = max(float(scores[0][rank]), 0.0)
            return dense

        # Fallback pure numpy cosine scores
        return np.clip(self.state.embeddings @ query_emb[0], 0.0, 1.0)

    def _keyword_search(self, query: str) -> np.ndarray:
        q = self._tfidf.transform([normalize_token(query)])
        sims = (self._tfidf_matrix @ q.T).toarray().ravel()
        if sims.max() > 0:
            sims = sims / sims.max()
        return sims.astype(np.float32)

    @staticmethod
    def _entity_score(chunk: Chunk, required_entities: Optional[Dict[str, List[str]]]) -> float:
        if not required_entities:
            return 0.5

        score = 0.0
        checks = 0
        for ent_type, values in required_entities.items():
            checks += 1
            chunk_values = {v.lower() for v in chunk.entities.get(ent_type, [])}
            wanted = {v.lower() for v in values}
            score += 1.0 if chunk_values.intersection(wanted) else 0.0

        return score / max(checks, 1)
