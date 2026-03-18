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
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
