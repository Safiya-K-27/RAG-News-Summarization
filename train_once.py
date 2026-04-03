"""Train models once and persist checkpoints for frontend inference."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from config import AppConfig
from agents.ingestion import DataIngestionAgent
from agents.training import NewsModelTrainer


def main() -> None:
    config = AppConfig()

    ingestion = DataIngestionAgent(config)
    documents = ingestion.load_documents()
    print(f"[Train] Loaded documents: {len(documents)}")

    trainer = NewsModelTrainer(config)
    result = trainer.train_all(documents)

    meta = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "trained_pairs": result.trained_pairs,
        "retriever_checkpoint": result.embedding_model_path,
        "summarizer_checkpoint": result.summarizer_model_path,
        "training_output_dir": str(config.training_output_dir),
    }

    meta_path = Path(config.training_output_dir) / "training_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("[Train] Completed")
    print(json.dumps(meta, indent=2))
    print(f"[Train] Metadata saved to: {meta_path}")


if __name__ == "__main__":
    main()
