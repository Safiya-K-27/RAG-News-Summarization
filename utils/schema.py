"""Shared dataclasses for pipeline communication."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Document:
    doc_id: str
    source: str
    title: str
    text: str
    summary: str = ""
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    source: str
    level: str  # paragraph | sentence
    text: str
    paragraph_id: int
    position: int
    metadata: Dict[str, str] = field(default_factory=dict)
    entities: Dict[str, List[str]] = field(default_factory=dict)


@dataclass
class RetrievedChunk:
    chunk: Chunk
    semantic_score: float
    keyword_score: float
    entity_score: float
    final_score: float


@dataclass
class EventPattern:
    pattern_id: str
    source_doc_ids: List[str]
    type: str
    actor: str
    action: str
    location: str
    time: str
    evidence: List[str] = field(default_factory=list)
    fitness: float = 0.0


@dataclass
class UserPreferences:
    length: str = "medium"  # short | medium | long
    tone: str = "formal"  # formal | casual
    bias_control: str = "balanced"  # neutral | balanced
    reading_level: str = "medium"  # simple | medium | advanced


@dataclass
class PipelineArtifacts:
    documents: List[Document] = field(default_factory=list)
    chunks: List[Chunk] = field(default_factory=list)
    retrieved_chunks: List[RetrievedChunk] = field(default_factory=list)
    event_patterns: List[EventPattern] = field(default_factory=list)
    optimized_patterns: List[EventPattern] = field(default_factory=list)
    summary: Optional[str] = None
    fact_checked_summary: Optional[str] = None
