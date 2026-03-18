"""Main entrypoint for the agent-orchestrated personalized news generator."""

from __future__ import annotations

from pprint import pprint

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
from agents.summarizer import LLMSummarizerAgent
from agents.training import NewsModelTrainer
from utils.schema import UserPreferences


def run_pipeline() -> None:
    config = AppConfig()

    # User preferences can come from API/UI in a production deployment.
    preferences = UserPreferences(
        length=config.default_length,
        tone=config.default_tone,
        bias_control=config.default_bias_control,
        reading_level=config.default_reading_level,
    )
    query = config.default_news_topic
    required_entities = None

    ingestion_agent = DataIngestionAgent(config)

    # Stage 1
    documents = ingestion_agent.load_documents()
    print(f"[Stage 1] Loaded documents: {len(documents)}")

    if config.run_training:
        trainer = NewsModelTrainer(config)
        training_result = trainer.train_all(documents)
        config.embedding_model = training_result.embedding_model_path
        config.hf_summarization_model = training_result.summarizer_model_path
        config.llm_provider = "hf"
        print(
            "[Training] Completed with "
            f"{training_result.trained_pairs} supervised pairs | "
            f"retriever={config.embedding_model} | summarizer={config.hf_summarization_model}"
        )

    chunking_agent = HierarchicalChunkingAgent()
    ner_agent = NERAgent(config)
    retrieval_agent = HybridRetrievalAgent(config)
    extraction_agent = EventExtractionAgent()
    evolution_agent = EvolutionaryOptimizationAgent(seed=42)
    defense_agent = AdversarialDefenseAgent()
    personalization_agent = PersonalizationAgent()
    summarizer_agent = LLMSummarizerAgent(config)
    fact_check_agent = FactCheckingAgent()

    # Stage 2
    chunks = chunking_agent.chunk_documents(documents)
    print(f"[Stage 2] Created chunks: {len(chunks)}")

    # Stage 3
    chunks = ner_agent.annotate_chunks(chunks)
    print("[Stage 3] NER annotation complete")

    # Stage 4
    retrieval_agent.build_index(chunks)
    print("[Stage 4] Embeddings and vector index built")

    # Stage 5
    retrieved = retrieval_agent.retrieve(query=query, top_k=config.top_k_retrieval, required_entities=required_entities)
    print(f"[Stage 5] Retrieved chunks: {len(retrieved)}")

    # Stage 6
    event_patterns = extraction_agent.extract_event_patterns(retrieved)
    print(f"[Stage 6] Event patterns extracted: {len(event_patterns)}")

    # Stage 7
    optimized_patterns = evolution_agent.optimize(event_patterns, query=query, generations=6, retain_top_k=5)
    print(f"[Stage 7] Optimized event patterns: {len(optimized_patterns)}")

    # Stage 8
    defended_chunks = defense_agent.defend_and_rerank(retrieved)
    print("[Stage 8] Defense re-ranking applied")

    # Stage 10 (used before Stage 9 to shape prompt)
    prompt_controls = personalization_agent.build_prompt_controls(preferences)

    # Stage 9
    summary = summarizer_agent.generate_summary(
        optimized_patterns=optimized_patterns,
        retrieved_chunks=defended_chunks,
        preferences=preferences,
        personalization_controls=prompt_controls,
    )
    print("[Stage 9] Summary generated")

    # Stage 11
    checked_summary = fact_check_agent.fact_check(summary, defended_chunks)
    print("[Stage 11] Fact-check complete")

    print("\n=== Optimized Event Patterns ===")
    for p in optimized_patterns:
        pprint(
            {
                "type": p.type,
                "actor": p.actor,
                "action": p.action,
                "location": p.location,
                "time": p.time,
                "fitness": round(p.fitness, 4),
            }
        )

    print("\n=== Final Fact-Checked Summary ===")
    print(checked_summary)


if __name__ == "__main__":
    run_pipeline()
