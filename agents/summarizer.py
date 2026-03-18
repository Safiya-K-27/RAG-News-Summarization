"""Stage 9: LLM-based summary generation using optimized events + context."""

from __future__ import annotations

from pathlib import Path
from typing import List

from config import AppConfig
from utils.schema import EventPattern, RetrievedChunk, UserPreferences


class LLMSummarizerAgent:
    """Generates final structured summary with OpenAI or Hugging Face fallback."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def generate_summary(
        self,
        optimized_patterns: List[EventPattern],
        retrieved_chunks: List[RetrievedChunk],
        preferences: UserPreferences,
        personalization_controls: str,
    ) -> str:
        event_block = self._format_events(optimized_patterns)
        context_block = "\n".join(f"- {r.chunk.text}" for r in retrieved_chunks[:10])

        prompt = self._build_prompt(event_block, context_block, personalization_controls, preferences)

        if self.config.llm_provider.lower() == "openai":
            summary = self._generate_openai(prompt)
            if summary:
                return summary

        summary = self._generate_hf(prompt)
        if summary:
            return summary

        return self._deterministic_fallback(optimized_patterns, retrieved_chunks, preferences)

    def _build_prompt(self, events: str, context: str, controls: str, preferences: UserPreferences) -> str:
        """Sample prompt template for grounded personalized summarization."""
        return (
            "You are a factual news summarization assistant.\n"
            "Use only the provided context and optimized event patterns.\n"
            "Do not invent facts or entities.\n"
            f"User preferences: length={preferences.length}, tone={preferences.tone}, "
            f"bias={preferences.bias_control}, reading_level={preferences.reading_level}\n"
            f"Style controls:\n{controls}\n\n"
            f"Optimized events:\n{events}\n\n"
            f"Retrieved context:\n{context}\n\n"
            "Output format:\n"
            "1) Headline\n2) Summary\n3) Key events bullets\n"
        )

    def _generate_openai(self, prompt: str) -> str:
        if not self.config.openai_api_key:
            return ""
        try:
            from openai import OpenAI

            client = OpenAI(api_key=self.config.openai_api_key)
            resp = client.chat.completions.create(
                model=self.config.openai_model,
                messages=[
                    {"role": "system", "content": "You produce concise factual summaries."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            return ""

    def _generate_hf(self, prompt: str) -> str:
        try:
            from transformers import pipeline

            model_name = self.config.hf_summarization_model
            trained_dir = Path(self.config.trained_summarizer_dir)
            if trained_dir.exists() and any(trained_dir.iterdir()):
                model_name = str(trained_dir)

            try:
                summarizer = pipeline("summarization", model=model_name)
                out = summarizer(prompt[:3500], max_length=220, min_length=80, do_sample=False)
                return out[0]["summary_text"].strip()
            except Exception:
                generator = pipeline("text2text-generation", model=model_name)
                out = generator(prompt[:3500], max_length=220, do_sample=False)
                return out[0]["generated_text"].strip()
        except Exception:
            return ""

    @staticmethod
    def _format_events(patterns: List[EventPattern]) -> str:
        if not patterns:
            return "- No high-confidence event patterns found."

        lines = []
        for p in patterns:
            lines.append(
                f"- type={p.type}; actor={p.actor}; action={p.action}; "
                f"location={p.location}; time={p.time}; fitness={p.fitness:.3f}"
            )
        return "\n".join(lines)

    @staticmethod
    def _deterministic_fallback(
        optimized_patterns: List[EventPattern],
        retrieved_chunks: List[RetrievedChunk],
        preferences: UserPreferences,
    ) -> str:
        """Fallback summary builder for environments without model/API access."""
        headline = "Entertainment News Brief"
        key_events = optimized_patterns[:3]

        summary_lines = [f"Headline: {headline}", "Summary:"]
        if key_events:
            for ev in key_events:
                summary_lines.append(
                    f"{ev.actor} {ev.action} ({ev.type}) in {ev.location} around {ev.time}."
                )
        else:
            summary_lines.append("No reliable event pattern was extracted from the available context.")

        summary_lines.append("Key events:")
        for idx, ev in enumerate(key_events, start=1):
            summary_lines.append(f"{idx}. {ev.type} | {ev.actor} | {ev.action} | {ev.location} | {ev.time}")

        if preferences.length == "short":
            return "\n".join(summary_lines[:5])
        if preferences.length == "medium":
            return "\n".join(summary_lines[:8])
        return "\n".join(summary_lines + [f"Context snippets used: {len(retrieved_chunks[:8])}"])
