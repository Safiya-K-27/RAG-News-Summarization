"""Stage 2: Hierarchical chunking for paragraph and sentence granularity."""

from typing import List

from utils.schema import Chunk, Document
from utils.text_utils import split_paragraphs, split_sentences


class HierarchicalChunkingAgent:
    """Builds paragraph-level and sentence-level chunks with metadata."""

    def chunk_documents(self, documents: List[Document]) -> List[Chunk]:
        chunks: List[Chunk] = []

        for doc in documents:
            paragraphs = split_paragraphs(doc.text)
            for p_idx, paragraph in enumerate(paragraphs):
                para_chunk = Chunk(
                    chunk_id=f"{doc.doc_id}_p_{p_idx}",
                    doc_id=doc.doc_id,
                    source=doc.source,
                    level="paragraph",
                    text=paragraph,
                    paragraph_id=p_idx,
                    position=p_idx,
                    metadata={"title": doc.title},
                )
                chunks.append(para_chunk)

                sentences = split_sentences(paragraph)
                for s_idx, sentence in enumerate(sentences):
                    chunks.append(
                        Chunk(
                            chunk_id=f"{doc.doc_id}_p_{p_idx}_s_{s_idx}",
                            doc_id=doc.doc_id,
                            source=doc.source,
                            level="sentence",
                            text=sentence,
                            paragraph_id=p_idx,
                            position=s_idx,
                            metadata={"title": doc.title},
                        )
                    )
        return chunks
