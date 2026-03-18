"""Stage 6: Rule-based event pattern extraction from retrieved chunks."""

from __future__ import annotations

import re
from typing import List

from utils.schema import EventPattern, RetrievedChunk


class EventExtractionAgent:
    """Extracts structured event patterns from retrieved evidence chunks."""

    def extract_event_patterns(self, retrieved_chunks: List[RetrievedChunk]) -> List[EventPattern]:
        events: List[EventPattern] = []

        for i, item in enumerate(retrieved_chunks):
            chunk = item.chunk
            text = chunk.text
            entities = chunk.entities

            actor = self._first_entity(entities, "PERSON") or self._first_entity(entities, "ORG") or "Unknown"
            location = self._first_entity(entities, "GPE") or "Unknown"
            time = self._first_entity(entities, "DATE") or self._extract_time(text)
            action = self._extract_action(text)
            event_type = self._infer_type(action)

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

        return events

    @staticmethod
    def _first_entity(entities, key: str) -> str:
        values = entities.get(key, [])
        return values[0] if values else ""

    @staticmethod
    def _extract_action(text: str) -> str:
        verbs = [
            "announced", "launched", "signed", "confirmed", "expanded", "released",
            "acquired", "partnered", "invested", "scheduled", "hosted",
        ]
        lower = text.lower()
        for verb in verbs:
            if verb in lower:
                return verb
        # Fallback to first verb-like token.
        match = re.search(r"\b\w+(ed|ing|s)\b", lower)
        return match.group(0) if match else "reported"

    @staticmethod
    def _extract_secondary_action(text: str) -> str:
        lower = text.lower()
        parts = lower.split(" and ")
        if len(parts) > 1:
            return EventExtractionAgent._extract_action(parts[-1])
        return EventExtractionAgent._extract_action(lower)

    @staticmethod
    def _extract_time(text: str) -> str:
        m = re.search(r"\b(today|yesterday|tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|sunday|\d{4})\b", text, flags=re.IGNORECASE)
        return m.group(0) if m else "Unknown"

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
