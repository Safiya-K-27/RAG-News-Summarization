"""Stage 11: Lightweight fact-checking against retrieved evidence."""

from __future__ import annotations

import re
from typing import List, Set

from utils.schema import RetrievedChunk


class FactCheckingAgent:
    """Prunes unsupported lines from generated summaries."""

    def fact_check(self, summary: str, retrieved_chunks: List[RetrievedChunk]) -> str:
        if not summary.strip():
            return summary

        context_text = " ".join(x.chunk.text for x in retrieved_chunks)
        context_tokens = self._tokenize(context_text)
        known_entities = self._collect_known_entities(retrieved_chunks)

        checked_lines: List[str] = []
        for line in [ln.strip() for ln in summary.splitlines() if ln.strip()]:
            if self._is_supported(line, context_tokens, known_entities):
                checked_lines.append(line)

        if not checked_lines:
            return "Summary could not be fully verified with retrieved context."
        return "\n".join(checked_lines)

    @staticmethod
    def _tokenize(text: str) -> Set[str]:
        return set(re.findall(r"\b[a-zA-Z]{3,}\b", text.lower()))

    @staticmethod
    def _collect_known_entities(retrieved_chunks: List[RetrievedChunk]) -> Set[str]:
        ents = set()
        for item in retrieved_chunks:
            for values in item.chunk.entities.values():
                for value in values:
                    ents.add(value.lower())
        return ents

    def _is_supported(self, line: str, context_tokens: Set[str], known_entities: Set[str]) -> bool:
        tokens = self._tokenize(line)
        if not tokens:
            return False

        overlap_ratio = len(tokens.intersection(context_tokens)) / max(len(tokens), 1)

        entities = self._extract_capitalized_entities(line)
        unknown_entities = [e for e in entities if e.lower() not in known_entities]

        return overlap_ratio >= 0.35 and len(unknown_entities) <= 1

    @staticmethod
    def _extract_capitalized_entities(text: str) -> List[str]:
        # Simple heuristic for named entities when parser is unavailable.
        return re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text)
