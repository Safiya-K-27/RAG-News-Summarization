"""Training module for end-to-end retriever and summarizer fine-tuning."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from config import AppConfig
from utils.schema import Document


@dataclass
class TrainingResult:
    embedding_model_path: str
    summarizer_model_path: str
    trained_pairs: int


class NewsModelTrainer:
    """Trains retriever and summarizer checkpoints before inference."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def train_all(self, documents: List[Document]) -> TrainingResult:
        pairs = self._build_pairs(documents)
        if not pairs:
            return TrainingResult(
                embedding_model_path=self.config.embedding_model,
                summarizer_model_path=self.config.hf_summarization_model,
                trained_pairs=0,
            )

        base_pairs, domain_pairs = self._split_base_domain_pairs(pairs)
        embedding_model_path = self._train_retriever(base_pairs + domain_pairs)
        summarizer_model_path = self._train_summarizer(base_pairs, domain_pairs)

        return TrainingResult(
            embedding_model_path=embedding_model_path,
            summarizer_model_path=summarizer_model_path,
            trained_pairs=len(base_pairs) + len(domain_pairs),
        )

    def _build_pairs(self, documents: List[Document]) -> List[Tuple[str, str, str]]:
        pairs: List[Tuple[str, str, str]] = []
        for doc in documents:
            src_text = (doc.text or "").strip()
            tgt_summary = (doc.summary or doc.title or "").strip()
            if len(src_text.split()) < 25:
                continue
            if len(tgt_summary.split()) < 4:
                continue
            pairs.append((src_text, tgt_summary, doc.source))

        # Cap for practical Colab training speed.
        return pairs[: self.config.train_sample_limit]

    def _split_base_domain_pairs(self, pairs: List[Tuple[str, str, str]]) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
        base: List[Tuple[str, str]] = []
        domain: List[Tuple[str, str]] = []

        for text, summary, source in pairs:
            if source == "kaggle_news":
                domain.append((text, summary))
            else:
                base.append((text, summary))

        domain = domain[: self.config.domain_sample_limit]
        return base, domain

    def _train_retriever(self, pairs: List[Tuple[str, str]]) -> str:
        if len(pairs) < 32:
            return self.config.embedding_model

        try:
            from sentence_transformers import InputExample, SentenceTransformer, losses
            from torch.utils.data import DataLoader

            model = SentenceTransformer(self.config.embedding_model)
            examples = [InputExample(texts=[summary, text]) for text, summary in pairs]
            loader = DataLoader(examples, batch_size=self.config.train_batch_size, shuffle=True)
            loss = losses.MultipleNegativesRankingLoss(model)
            warmup_steps = max(1, int(0.1 * len(loader)))

            model.fit(
                train_objectives=[(loader, loss)],
                epochs=self.config.retriever_train_epochs,
                warmup_steps=warmup_steps,
                show_progress_bar=True,
            )
            model.save(str(self.config.trained_embedding_dir))
            return str(self.config.trained_embedding_dir)
        except Exception as exc:
            print(f"[Training][Retriever] Falling back to base embedding model due to: {exc}")
            return self.config.embedding_model

    def _train_summarizer(self, base_pairs: List[Tuple[str, str]], domain_pairs: List[Tuple[str, str]]) -> str:
        train_pairs = base_pairs
        if domain_pairs:
            # Upweight domain adaptation with light oversampling.
            train_pairs = base_pairs + (domain_pairs * 2)

        if len(train_pairs) < 64:
            return self.config.hf_summarization_model

        try:
            from datasets import Dataset
            from transformers import (
                AutoModelForSeq2SeqLM,
                AutoTokenizer,
                DataCollatorForSeq2Seq,
                Seq2SeqTrainer,
                Seq2SeqTrainingArguments,
            )

            base_model = self.config.base_summarizer_train_model
            tokenizer = AutoTokenizer.from_pretrained(base_model)
            model = AutoModelForSeq2SeqLM.from_pretrained(base_model)

            texts = [x[0] for x in train_pairs]
            summaries = [x[1] for x in train_pairs]
            dataset = Dataset.from_dict({"text": texts, "summary": summaries})

            def _tokenize(batch: Dict[str, List[str]]) -> Dict[str, List[List[int]]]:
                model_inputs = tokenizer(
                    batch["text"],
                    max_length=self.config.summarizer_max_input_length,
                    truncation=True,
                )
                labels = tokenizer(
                    text_target=batch["summary"],
                    max_length=self.config.summarizer_max_target_length,
                    truncation=True,
                )
                model_inputs["labels"] = labels["input_ids"]
                return model_inputs

            tokenized = dataset.map(_tokenize, batched=True, remove_columns=["text", "summary"])
            data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

            args = Seq2SeqTrainingArguments(
                output_dir=str(self.config.trained_summarizer_dir),
                num_train_epochs=self.config.summarizer_train_epochs,
                learning_rate=3e-5,
                per_device_train_batch_size=self.config.train_batch_size,
                gradient_accumulation_steps=1,
                save_total_limit=1,
                logging_steps=25,
                report_to=[],
                fp16=False,
                predict_with_generate=False,
            )

            trainer = Seq2SeqTrainer(
                model=model,
                args=args,
                train_dataset=tokenized,
                data_collator=data_collator,
            )
            trainer.train()

            trainer.save_model(str(self.config.trained_summarizer_dir))
            tokenizer.save_pretrained(str(self.config.trained_summarizer_dir))
            return str(self.config.trained_summarizer_dir)
        except Exception as exc:
            print(f"[Training][Summarizer] Falling back to configured HF model due to: {exc}")
            return self.config.hf_summarization_model
