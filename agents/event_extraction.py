"""Stage 6: Rule-based event pattern extraction from retrieved chunks."""

from __future__ import annotations

import re
from typing import List

from utils.schema import EventPattern, RetrievedChunk


class EventExtractionAgent:
    """Extracts structured event patterns from retrieved evidence chunks."""

    WEAK_ACTIONS = {
        "is", "are", "was", "were", "be", "been", "being", "has", "have", "had", "does", "do", "did"
    }

    def extract_event_patterns(self, retrieved_chunks: List[RetrievedChunk]) -> List[EventPattern]:
        events: List[EventPattern] = []

        for i, item in enumerate(retrieved_chunks):
            chunk = item.chunk
            text = chunk.text
            entities = chunk.entities

            actor = (
                self._first_entity(entities, "PERSON")
                or self._first_entity(entities, "ORG")
                or self._extract_actor_from_text_or_title(text, chunk.metadata.get("title", ""))
            )
            location = self._first_entity(entities, "GPE") or self._extract_location(text)
            time = self._first_entity(entities, "DATE") or self._extract_time(text)
            action = self._extract_action(text)
            event_type = self._infer_type(action)

            if actor == "Unknown" and action in self.WEAK_ACTIONS and location == "Unknown":
                # Skip patterns that are likely noise and not event-like.
                continue

            events.append(
                EventPattern(
                    pattern_id=f"event_{i}",
                    source_doc_ids=[chunk.doc_id],
                    type=event_type,
                    actor=actor,
                    action=action,
                    location=location,
                    time=time,
                    evidence=[text],
                )
            )

            # Attempt to extract a second micro-pattern if text contains conjunctions.
            if " and " in text.lower():
                events.append(
                    EventPattern(
                        pattern_id=f"event_{i}_b",
                        source_doc_ids=[chunk.doc_id],
                        type=event_type,
                        actor=actor,
                        action=self._extract_secondary_action(text),
                        location=location,
                        time=time,
                        evidence=[text],
                    )
                )

                return self._deduplicate_events(events)

    @staticmethod
    def _first_entity(entities, key: str) -> str:
        values = entities.get(key, [])
        return values[0] if values else ""

    @staticmethod
    def _extract_action(text: str) -> str:
        verbs = [
            "announced", "launched", "signed", "confirmed", "expanded", "released",
            "acquired", "partnered", "invested", "scheduled", "hosted",
            "won", "nominated", "premiered", "joined", "unveiled", "opened",
        ]
        lower = text.lower()
        for verb in verbs:
            if verb in lower:
                return verb
        # Fallback to first verb-like token.
        candidates = re.findall(r"\b\w+(ed|ing|s)\b", lower)
        for cand in candidates:
            if cand not in EventExtractionAgent.WEAK_ACTIONS and len(cand) > 3:
                return cand
        return "reported"

    @staticmethod
    def _extract_secondary_action(text: str) -> str:
        lower = text.lower()
        parts = lower.split(" and ")
        if len(parts) > 1:
            return EventExtractionAgent._extract_action(parts[-1])
        return EventExtractionAgent._extract_action(lower)

    @staticmethod
    def _extract_time(text: str) -> str:
        m = re.search(
            r"\b(today|yesterday|tomorrow|this year|last year|next year|"
            r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|\d{4})\b",
            text,
            flags=re.IGNORECASE,
        )
        return m.group(0) if m else "Unknown"

    @staticmethod
    def _extract_actor_from_text_or_title(text: str, title: str) -> str:
        # Prefer multi-word proper names in the text, then the title.
        for source in [text, title]:
            names = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2}\b", source)
            if names:
                return names[0]

            org_like = re.findall(
                r"\b([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*\s+"
                r"(?:Studios|Entertainment|Pictures|Festival|Network|Platform|Media))\b",
                source,
            )
            if org_like:
                return org_like[0]
        return "Unknown"

    @staticmethod
    def _extract_location(text: str) -> str:
        # Lightweight location fallback when NER misses GPE labels.
        m = re.search(r"\bin\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b", text)
        return m.group(1) if m else "Unknown"

    @staticmethod
    def _deduplicate_events(events: List[EventPattern]) -> List[EventPattern]:
        seen = set()
        deduped: List[EventPattern] = []
        for ev in events:
            key = (ev.actor.lower(), ev.action.lower(), ev.location.lower(), ev.time.lower())
            if key in seen:
                continue
            seen.add(key)
            deduped.append(ev)
        return deduped

    @staticmethod
    def _infer_type(action: str) -> str:
        mapping = {
            "announced": "announcement",
            "launched": "launch",
            "signed": "agreement",
            "confirmed": "confirmation",
            "expanded": "expansion",
            "released": "release",
            "partnered": "partnership",
            "invested": "investment",
            "scheduled": "schedule",
            "hosted": "event",
        }
        return mapping.get(action.lower(), "general_event")
