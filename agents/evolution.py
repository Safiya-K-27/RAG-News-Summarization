"""Stage 7: Evolutionary optimization over extracted event patterns."""

from __future__ import annotations

import random
from collections import Counter
from typing import List

import networkx as nx
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from utils.schema import EventPattern


class EvolutionaryOptimizationAgent:
    """Genetic algorithm for selecting and refining event patterns."""

    def __init__(self, seed: int = 42) -> None:
        self.random = random.Random(seed)

    def optimize(
        self,
        patterns: List[EventPattern],
        query: str,
        generations: int = 6,
        retain_top_k: int = 5,
    ) -> List[EventPattern]:
        if not patterns:
            return []

        population = patterns[:]
        for _ in range(generations):
            self._score_population(population, query)
            parents = self._select(population)
            children = self._crossover_population(parents, target_size=len(population))
            population = parents + children

        self._score_population(population, query)
        population.sort(key=lambda p: p.fitness, reverse=True)
        unique: List[EventPattern] = []
        seen = set()
        for pat in population:
            signature = (pat.actor.lower(), pat.action.lower(), pat.location.lower(), pat.time.lower())
            if signature in seen:
                continue
            seen.add(signature)
            unique.append(pat)
            if len(unique) >= retain_top_k:
                break
        return unique

    def _score_population(self, population: List[EventPattern], query: str) -> None:
        agreement = self._cross_document_agreement(population)
        for idx, pat in enumerate(population):
            tfidf_score = self._tfidf_relevance(pat, query)
            textrank_score = self._textrank_signal(pat)
            entity_consistency = self._entity_consistency(pat)
            cross_doc = agreement[idx]

            pat.fitness = float(
                0.35 * tfidf_score
                + 0.25 * textrank_score
                + 0.20 * entity_consistency
                + 0.20 * cross_doc
            )

    def _select(self, population: List[EventPattern]) -> List[EventPattern]:
        ranked = sorted(population, key=lambda p: p.fitness, reverse=True)
        keep = max(2, len(ranked) // 2)
        return ranked[:keep]

    def _crossover_population(self, parents: List[EventPattern], target_size: int) -> List[EventPattern]:
        children: List[EventPattern] = []
        if len(parents) < 2:
            return children

        while len(children) + len(parents) < target_size:
            p1, p2 = self.random.sample(parents, 2)
            child = self._crossover(p1, p2, child_id=f"child_{len(children)}")
            children.append(child)

        return children

    def _crossover(self, a: EventPattern, b: EventPattern, child_id: str) -> EventPattern:
        """Merge event attributes from two parent patterns."""
        choose = lambda x, y: x if self.random.random() > 0.5 else y

        return EventPattern(
            pattern_id=child_id,
            source_doc_ids=sorted(set(a.source_doc_ids + b.source_doc_ids)),
            type=choose(a.type, b.type),
            actor=choose(a.actor, b.actor),
            action=choose(a.action, b.action),
            location=choose(a.location, b.location),
            time=choose(a.time, b.time),
            evidence=list(dict.fromkeys(a.evidence + b.evidence))[:3],
        )

    @staticmethod
    def _tfidf_relevance(pattern: EventPattern, query: str) -> float:
        corpus = [
            f"{pattern.type} {pattern.actor} {pattern.action} {pattern.location} {pattern.time}",
            query,
        ]
        vec = TfidfVectorizer(stop_words="english")
        mat = vec.fit_transform(corpus).toarray()
        sim = np.dot(mat[0], mat[1]) / ((np.linalg.norm(mat[0]) * np.linalg.norm(mat[1])) + 1e-9)
        return float(max(sim, 0.0))

    @staticmethod
    def _textrank_signal(pattern: EventPattern) -> float:
        if not pattern.evidence:
            return 0.0

        vec = TfidfVectorizer(stop_words="english")
        mat = vec.fit_transform(pattern.evidence).toarray()
        n = mat.shape[0]
        if n == 1:
            return 0.6

        g = nx.Graph()
        for i in range(n):
            g.add_node(i)
        for i in range(n):
            for j in range(i + 1, n):
                sim = float(np.dot(mat[i], mat[j]) / ((np.linalg.norm(mat[i]) * np.linalg.norm(mat[j])) + 1e-9))
                if sim > 0:
                    g.add_edge(i, j, weight=sim)

        if g.number_of_edges() == 0:
            return 0.2
        scores = nx.pagerank(g, weight="weight")
        return float(np.mean(list(scores.values())))

    @staticmethod
    def _entity_consistency(pattern: EventPattern) -> float:
        fields = [pattern.actor, pattern.action, pattern.location, pattern.time]
        non_empty = sum(1 for x in fields if x and x != "Unknown")
        return non_empty / 4.0

    @staticmethod
    def _cross_document_agreement(population: List[EventPattern]) -> List[float]:
        signatures = [f"{p.actor}|{p.action}|{p.location}|{p.time}" for p in population]
        counts = Counter(signatures)
        max_count = max(counts.values()) if counts else 1
        return [counts[s] / max_count for s in signatures]
