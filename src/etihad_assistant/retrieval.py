"""Load policy snapshots and rank current passages with local BM25 search."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from langchain_core.documents import Document

from .config import Settings
from .schemas import Citation

TOKEN_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "a", "about", "an", "and", "are", "can", "do", "does", "for", "from", "how",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "policy", "tell",
    "that", "the", "this", "to", "what", "when", "where", "which", "with",
}
POLICY_TERMS = {
    "acm", "adm", "ancillary", "apis", "baggage", "bag", "blind", "booking", "bus",
    "chauffeur", "child", "codeshare", "cooling", "correction", "emd", "expectant",
    "fare", "falcon", "gds", "group", "gst", "hajj", "infant", "insurance", "interline",
    "luggage", "medical", "minor", "name", "no-show", "passenger", "pet", "priority",
    "rail", "refund", "residence", "seat", "stretcher", "tax", "ticket", "visa",
    "wheelchair",
}


def _normalise_token(token: str) -> str:
    """Reduce common word endings for a small, dependency-free lexical index."""
    if len(token) > 5 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 5 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 4 and token.endswith("ed"):
        return token[:-2]
    if len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _tokens(text: str) -> list[str]:
    """Extract searchable terms while dropping broad question words."""
    return [
        _normalise_token(token)
        for token in TOKEN_RE.findall(text.lower())
        if token not in STOPWORDS and len(token) > 1
    ]


@dataclass(frozen=True)
class PolicyChunk:
    """One citable policy passage with its source and publication metadata."""
    id: str
    topic: str
    document_title: str
    version_id: str
    is_current: bool
    effective_date_status: str
    section_path: str
    snapshot_page: int | None
    passage: str
    table_cell_ids: list[str]
    scope: dict[str, Any]
    specificity: int
    source_url: str | None = None
    fixture: bool = False

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> PolicyChunk:
        allowed = cls.__dataclass_fields__
        return cls(**{key: item for key, item in value.items() if key in allowed})

    def as_document(self) -> Document:
        """Expose a passage to LangChain-compatible retrieval code."""
        return Document(
            page_content=self.passage,
            metadata={
                "id": self.id,
                "title": self.document_title,
                "section_path": self.section_path,
                "version_id": self.version_id,
                "source": self.source_url,
            },
        )

    def as_citation(self) -> Citation:
        """Build the source object returned beside a grounded answer."""
        return Citation(
            id=self.id,
            source_type="policy",
            document_title=self.document_title,
            section_path=self.section_path,
            version_id=self.version_id,
            effective_date_status=self.effective_date_status,
            snapshot_page=self.snapshot_page,
            passage=self.passage,
            table_cell_ids=self.table_cell_ids,
            canonical_url=self.source_url,
            retrieved_at=datetime.now(UTC),
            fixture=self.fixture,
        )


class PolicyRepository:
    """Local LangChain-document repository with BM25 retrieval and no model calls."""

    def __init__(self, settings: Settings):
        """Load the primary snapshot and optional official supplements into one index."""
        corpus_text = settings.corpus_file.read_text(encoding="utf-8")
        supplement_paths = sorted(settings.corpus_file.parent.glob("etihad_official_*.json"))
        supplements = [
            (path.name, path.read_text(encoding="utf-8")) for path in supplement_paths
        ]
        self.corpus_digest = hashlib.sha256(
            (corpus_text + "".join(name + text for name, text in supplements)).encode("utf-8")
        ).hexdigest()
        raw = json.loads(corpus_text)
        self.warning: str = raw["warning"]
        self.source_url: str | None = raw.get("source_url")
        self.scraped_at: str | None = raw.get("scraped_at")
        self.chunks = [PolicyChunk.from_dict(item) for item in raw["chunks"]]
        for _, supplement_text in supplements:
            supplement = json.loads(supplement_text)
            self.chunks.extend(PolicyChunk.from_dict(item) for item in supplement["chunks"])
        self.chunks_by_id = {chunk.id: chunk for chunk in self.chunks}
        self.documents = [chunk.as_document() for chunk in self.chunks]
        self._tokens = [
            _tokens(f"{chunk.document_title} {chunk.section_path} {chunk.passage}")
            for chunk in self.chunks
        ]
        self._counts = [Counter(tokens) for tokens in self._tokens]
        self._lengths = [len(tokens) for tokens in self._tokens]
        self._average_length = sum(self._lengths) / max(len(self._lengths), 1)
        self._document_frequency: Counter[str] = Counter()
        for tokens in self._tokens:
            self._document_frequency.update(set(tokens))

    def _bm25(self, query_tokens: list[str], index: int) -> float:
        """Score term frequency while down-weighting common terms and long chunks."""
        count = self._counts[index]
        length = self._lengths[index]
        total = len(self.chunks)
        score = 0.0
        for token in query_tokens:
            frequency = count[token]
            if not frequency:
                continue
            document_frequency = self._document_frequency[token]
            inverse_frequency = math.log(
                1 + (total - document_frequency + 0.5) / (document_frequency + 0.5)
            )
            denominator = frequency + 1.5 * (
                1 - 0.75 + 0.75 * length / self._average_length
            )
            score += inverse_frequency * frequency * 2.5 / denominator
        return score

    def retrieve(
        self,
        query: str,
        _topic: str = "policy",
        _state: dict[str, Any] | None = None,
        limit: int = 5,
    ) -> tuple[list[PolicyChunk], dict[str, Any]]:
        """Return current passages and diagnostic scores without an LLM call."""
        query_tokens = _tokens(query)
        if not query_tokens:
            return [], {"backend": "local_bm25", "selected_ids": [], "llm_calls": 0}
        title_vocabulary = {
            token for chunk in self.chunks for token in _tokens(chunk.document_title)
        }
        if not (set(query_tokens) & (title_vocabulary | POLICY_TERMS)):
            # A vocabulary gate limits accidental matches on generic travel words.
            return [], {
                "backend": "local_bm25",
                "candidate_count": len(self.chunks),
                "selected_ids": [],
                "scores": {},
                "llm_calls": 0,
                "reason": "no_policy_vocabulary_match",
                "source_url": self.source_url,
                "scraped_at": self.scraped_at,
            }

        scored: list[tuple[float, int]] = []
        query_token_set = set(query_tokens)
        for index, chunk in enumerate(self.chunks):
            if not chunk.is_current:
                continue
            score = self._bm25(query_tokens, index)
            title_tokens = set(_tokens(chunk.document_title))
            title_overlap = len(query_token_set & title_tokens)
            score += title_overlap * 2.25
            # Policy headings should outrank incidental mentions inside long passages.
            score += (title_overlap / max(len(title_tokens), 1)) * 3.0
            if chunk.document_title.lower() in query.lower():
                score += 4.0
            scored.append((score, index))

        scored.sort(reverse=True)
        if not scored or scored[0][0] < 1.0:
            selected: list[PolicyChunk] = []
        else:
            best_score = scored[0][0]
            selected = [
                self.chunks[index]
                for score, index in scored
                if score >= max(1.0, best_score * 0.34)
            ][:limit]

        details = {
            "backend": "local_bm25",
            "candidate_count": len(scored),
            "selected_ids": [item.id for item in selected],
            "scores": {
                self.chunks[index].id: round(score, 4)
                for score, index in scored[: len(selected)]
            },
            "llm_calls": 0,
            "source_url": self.source_url,
            "scraped_at": self.scraped_at,
        }
        return selected, details
