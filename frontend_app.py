"""Simple frontend for personalized news summaries using trained checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import gradio as gr
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from config import AppConfig
from agents.chunking import HierarchicalChunkingAgent
from agents.defense import AdversarialDefenseAgent
from agents.event_extraction import EventExtractionAgent
from agents.evolution import EvolutionaryOptimizationAgent
from agents.fact_check import FactCheckingAgent
from agents.ingestion import DataIngestionAgent
from agents.ner import NERAgent
from agents.personalization import PersonalizationAgent
from agents.retrieval import HybridRetrievalAgent
from utils.schema import RetrievedChunk, UserPreferences


class NewsFrontendService:
    """Loads pipeline resources once and serves user requests interactively."""

    def __init__(self) -> None:
        self.config = AppConfig()
        self.config.run_training = False

        self.ingestion_agent = DataIngestionAgent(self.config)
        self.chunking_agent = HierarchicalChunkingAgent()
        self.ner_agent = NERAgent(self.config)
        self.retrieval_agent = HybridRetrievalAgent(self.config)
        self.extraction_agent = EventExtractionAgent()
        self.evolution_agent = EvolutionaryOptimizationAgent(seed=42)
        self.defense_agent = AdversarialDefenseAgent()
        self.personalization_agent = PersonalizationAgent()
        self.fact_check_agent = FactCheckingAgent()

        self.tokenizer, self.model = self._load_summarizer_checkpoint()

        self._build_runtime_index()

    def _load_summarizer_checkpoint(self):
        ckpt_dir = Path(self.config.trained_summarizer_dir)
        model_name = str(ckpt_dir) if ckpt_dir.exists() and any(ckpt_dir.iterdir()) else self.config.hf_summarization_model
        print(f"[Frontend] Summarizer source: {model_name}")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        model.eval()
        return tokenizer, model

    def _build_runtime_index(self) -> None:
        documents = self.ingestion_agent.load_documents()
        chunks = self.chunking_agent.chunk_documents(documents)
        chunks = self.ner_agent.annotate_chunks(chunks)
        self.retrieval_agent.build_index(chunks)
        print(f"[Frontend] Runtime index ready with {len(chunks)} chunks")

    def _generate_with_model(
        self,
        topic: str,
        preferences: UserPreferences,
        prompt_controls: str,
        retrieved_chunks: List[RetrievedChunk],
    ) -> str:
        context = "\n".join([f"- {x.chunk.text}" for x in retrieved_chunks[:10]])
        prompt = (
            f"Summarize entertainment news for topic: {topic}\n"
            f"Constraints:\n{prompt_controls}\n"
            "Stay factual and use only the provided context.\n"
            f"Context:\n{context}"
        )

        max_new_tokens = 120 if preferences.length == "short" else 260
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        output_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            max_new_tokens=max_new_tokens,
            num_beams=4,
            early_stopping=True,
        )
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

    def summarize(self, topic: str, reading_level: str, summary_length: str, neutrality: str) -> Tuple[str, str]:
        topic = (topic or "latest entertainment news").strip()

        preferences = UserPreferences(
            length=summary_length,
            tone="formal",
            bias_control=neutrality,
            reading_level=reading_level,
        )

        retrieved = self.retrieval_agent.retrieve(query=topic, top_k=self.config.top_k_retrieval)
        event_patterns = self.extraction_agent.extract_event_patterns(retrieved)
        optimized_patterns = self.evolution_agent.optimize(event_patterns, query=topic, generations=6, retain_top_k=5)
        defended_chunks = self.defense_agent.defend_and_rerank(retrieved)

        controls = self.personalization_agent.build_prompt_controls(preferences)
        raw_summary = self._generate_with_model(topic, preferences, controls, defended_chunks)
        checked_summary = self.fact_check_agent.fact_check(raw_summary, defended_chunks)

        if optimized_patterns:
            event_lines = [
                f"- {p.actor} | {p.action} | {p.location} | {p.time} | fitness={p.fitness:.3f}"
                for p in optimized_patterns
            ]
            events_view = "\n".join(event_lines)
        else:
            events_view = "- No optimized events found"

        return checked_summary, events_view


def launch_app() -> None:
    service = NewsFrontendService()

    with gr.Blocks(title="Personalized News Generator") as demo:
        gr.Markdown("# Personalized Entertainment News Summarizer")
        gr.Markdown("Train once, then generate summaries with user-controlled reading level and length.")

        with gr.Row():
            topic = gr.Textbox(label="News Topic", value="latest entertainment awards and film releases")

        with gr.Row():
            reading_level = gr.Radio(["simple", "advanced"], value="simple", label="Reading Level")
            summary_length = gr.Radio(["short", "long"], value="short", label="Summary Length")
            neutrality = gr.Radio(["neutral", "balanced"], value="neutral", label="Neutrality Preference")

        generate_btn = gr.Button("Generate Summary", variant="primary")
        summary_out = gr.Textbox(label="Fact-Checked Summary", lines=12)
        events_out = gr.Textbox(label="Optimized Event Patterns", lines=8)

        generate_btn.click(
            fn=service.summarize,
            inputs=[topic, reading_level, summary_length, neutrality],
            outputs=[summary_out, events_out],
        )

    demo.launch(share=True)


if __name__ == "__main__":
    launch_app()
