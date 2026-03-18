"""Stage 8: Adversarial defense and robust re-ranking."""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Tuple

from utils.schema import RetrievedChunk
from utils.text_utils import remove_homoglyph_noise


class AdversarialDefenseAgent:
    """Mitigates lead bias, lexical noise, and contradictory fact patterns."""

    def __init__(self) -> None:
        self.synonym_map = {
            "movie": "film",
            "cinema": "film",
            "flick": "film",
            "announce": "announced",
            "reveal": "announced",
        }

    def defend_and_rerank(self, retrieved: List[RetrievedChunk]) -> List[RetrievedChunk]:
        if not retrieved:
            return []

        cleaned = self._sanitize_noise(retrieved)
        contradictions = self._detect_contradictions(cleaned)

        for item in cleaned:
            base = item.final_score
            lead_factor = self._lead_bias_penalty(item.chunk.position)
            importance = self._importance_score(item)
            contradiction_penalty = 0.75 if item.chunk.chunk_id in contradictions else 1.0
            item.final_score = base * lead_factor * importance * contradiction_penalty

        cleaned.sort(key=lambda x: x.final_score, reverse=True)
        return cleaned

    def _sanitize_noise(self, retrieved: List[RetrievedChunk]) -> List[RetrievedChunk]:
        for item in retrieved:
            text = remove_homoglyph_noise(item.chunk.text)
            for k, v in self.synonym_map.items():
                text = text.replace(k, v).replace(k.title(), v)
            item.chunk.text = text
        return retrieved

    @staticmethod
    def _lead_bias_penalty(position: int) -> float:
        # Slightly down-weight lead-only chunks and reward informative later chunks.
        if position == 0:
            return 0.9
        if position == 1:
            return 1.0
        return 1.05

    @staticmethod
    def _importance_score(item: RetrievedChunk) -> float:
        entity_count = sum(len(v) for v in item.chunk.entities.values())
        length_bonus = min(len(item.chunk.text.split()) / 40.0, 1.0)
        return 0.9 + 0.15 * min(entity_count, 4) / 4.0 + 0.15 * length_bonus

    @staticmethod
    def _signature(item: RetrievedChunk) -> Tuple[str, str]:
        persons = item.chunk.entities.get("PERSON", [])
        orgs = item.chunk.entities.get("ORG", [])
        actor = persons[0] if persons else (orgs[0] if orgs else "unknown")

        gpe = item.chunk.entities.get("GPE", ["unknown"])[0]
        return actor.lower(), gpe.lower()

    def _detect_contradictions(self, retrieved: List[RetrievedChunk]) -> set:
        """Mark minority location claims for the same actor as potentially contradictory."""
        actor_to_locs: Dict[str, List[str]] = {}
        actor_to_items: Dict[str, List[RetrievedChunk]] = {}

        for item in retrieved:
            actor, loc = self._signature(item)
            actor_to_locs.setdefault(actor, []).append(loc)
            actor_to_items.setdefault(actor, []).append(item)

        flagged = set()
        for actor, locs in actor_to_locs.items():
            if len(set(locs)) <= 1:
                continue
            majority_loc, _ = Counter(locs).most_common(1)[0]
            for item in actor_to_items[actor]:
                _, loc = self._signature(item)
                if loc != majority_loc and loc != "unknown":
                    flagged.add(item.chunk.chunk_id)

        return flagged
