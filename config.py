"""Application configuration for the multi-agent personalized news generator."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

try:
    from dotenv import load_dotenv
except Exception:
    def load_dotenv() -> None:
        return None

load_dotenv()


@dataclass
class AppConfig:
    """Centralized runtime configuration loaded from environment variables."""

    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parent)
    data_dir: Path = field(init=False)
    raw_dir: Path = field(init=False)
    processed_dir: Path = field(init=False)

    # Data ingestion controls
    use_hf_datasets: bool = os.getenv("USE_HF_DATASETS", "false").lower() == "true"
    kaggle_news_csv_path: str = os.getenv("KAGGLE_NEWS_CSV_PATH", "")
    max_docs_per_source: int = int(os.getenv("MAX_DOCS_PER_SOURCE", "50"))
    
    # Training controls
    run_training: bool = os.getenv("RUN_TRAINING", "false").lower() == "true"
    train_sample_limit: int = int(os.getenv("TRAIN_SAMPLE_LIMIT", "4000"))
    domain_sample_limit: int = int(os.getenv("DOMAIN_SAMPLE_LIMIT", "1200"))
    retriever_train_epochs: int = int(os.getenv("RETRIEVER_TRAIN_EPOCHS", "1"))
    summarizer_train_epochs: int = int(os.getenv("SUMMARIZER_TRAIN_EPOCHS", "1"))
    train_batch_size: int = int(os.getenv("TRAIN_BATCH_SIZE", "8"))
    summarizer_max_input_length: int = int(os.getenv("SUMMARIZER_MAX_INPUT_LENGTH", "512"))
    summarizer_max_target_length: int = int(os.getenv("SUMMARIZER_MAX_TARGET_LENGTH", "96"))
    training_output_dir_name: str = os.getenv("TRAINING_OUTPUT_DIR", "checkpoints")
    base_summarizer_train_model: str = os.getenv("BASE_SUMMARIZER_TRAIN_MODEL", "google/flan-t5-small")
    training_output_dir: Path = field(init=False)
    trained_embedding_dir: Path = field(init=False)
    trained_summarizer_dir: Path = field(init=False)

    # NLP and retrieval
    spacy_model: str = os.getenv("SPACY_MODEL", "en_core_web_sm")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    embedding_batch_size: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "64"))
    top_k_retrieval: int = int(os.getenv("TOP_K_RETRIEVAL", "8"))

    # LLM settings
    llm_provider: str = os.getenv("LLM_PROVIDER", "openai")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    hf_summarization_model: str = os.getenv("HF_SUMMARY_MODEL", "facebook/bart-large-cnn")

    # Personalization defaults
    default_length: str = os.getenv("DEFAULT_SUMMARY_LENGTH", "medium")
    default_tone: str = os.getenv("DEFAULT_SUMMARY_TONE", "formal")
    default_bias_control: str = os.getenv("DEFAULT_BIAS_CONTROL", "balanced")
    default_reading_level: str = os.getenv("DEFAULT_READING_LEVEL", "medium")
    default_news_topic: str = os.getenv(
        "DEFAULT_NEWS_TOPIC",
        "latest entertainment industry announcements and partnerships",
    )

    # Entity normalization aliases
    entity_aliases: Dict[str, str] = field(
        default_factory=lambda: {
            "srk": "Shah Rukh Khan",
            "uk": "United Kingdom",
            "usa": "United States",
            "u.s.": "United States",
            "u.s": "United States",
            "bolly": "Bollywood",
        }
    )

    def __post_init__(self) -> None:
        self.data_dir = self.project_root / "data"
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.training_output_dir = self.project_root / self.training_output_dir_name
        self.trained_embedding_dir = self.training_output_dir / "retriever"
        self.trained_summarizer_dir = self.training_output_dir / "summarizer"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.training_output_dir.mkdir(parents=True, exist_ok=True)
        self.trained_embedding_dir.mkdir(parents=True, exist_ok=True)
        self.trained_summarizer_dir.mkdir(parents=True, exist_ok=True)
