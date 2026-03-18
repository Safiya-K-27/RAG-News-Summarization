"""Text utility helpers for cleaning and segmentation."""

import re
from typing import List


def clean_text(text: str) -> str:
    """Normalize whitespace and remove obvious artifacts."""
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_paragraphs(text: str) -> List[str]:
    """Split text into paragraphs while preserving useful content."""
    rough = re.split(r"\n{2,}", text)
    paragraphs = [p.strip() for p in rough if p.strip()]
    if paragraphs:
        return paragraphs
    return [x.strip() for x in re.split(r"(?<=[.!?])\s{2,}", text) if x.strip()]


def split_sentences(text: str) -> List[str]:
    """Simple sentence splitter as a fallback when parser-based split is unavailable."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


def normalize_token(token: str) -> str:
    """Normalize tokens for lexical matching and keyword search."""
    token = token.lower().strip()
    token = re.sub(r"[^a-z0-9\s]", "", token)
    token = re.sub(r"\s+", " ", token)
    return token


def remove_homoglyph_noise(text: str) -> str:
    """Basic homoglyph cleanup for common Cyrillic/Greek lookalike characters."""
    replacements = {
        "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H", "О": "O", "Р": "P", "С": "C", "Т": "T", "Х": "X",
        "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "х": "x", "у": "y",
    }
    return "".join(replacements.get(ch, ch) for ch in text)
