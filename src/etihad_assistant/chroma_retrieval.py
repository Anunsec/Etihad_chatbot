"""Hybrid policy search combining persistent Chroma vectors with local BM25."""

from __future__ import annotations

import hashlib
from typing import Any

from langchain_openai import OpenAIEmbeddings

from .config import Settings
from .retrieval import PolicyChunk, PolicyRepository


class ChromaPolicyRepository(PolicyRepository):
    """Persistent policy vectors, fused with the existing precise BM25 retrieval."""

    def __init__(
        self,
        settings: Settings,
        embeddings: Any = None,
        chroma_client: Any = None,
    ):
        super().__init__(settings)
        self.settings = settings
        self.embedding_calls = 0
        self.indexed_count = 0
        if not settings.openai_api_key and embeddings is None:
            raise ValueError("OpenAI key is required to embed the policy corpus")
        if chroma_client is None:
            import chromadb

            settings.chroma_dir.mkdir(parents=True, exist_ok=True)
            chroma_client = chromadb.PersistentClient(path=str(settings.chroma_dir))
        self.embeddings = embeddings or OpenAIEmbeddings(
            model=settings.openai_embedding_model,
            api_key=settings.openai_api_key,
        )
        digest = hashlib.sha256(
            f"{self.corpus_digest}:{settings.openai_embedding_model}".encode()
        ).hexdigest()[:20]
        # A corpus/model change gets a new collection, avoiding stale embeddings.
        self.collection = chroma_client.get_or_create_collection(name=f"etihad-policy-{digest}")
        self._index_ready = False

    def ensure_index(self) -> int:
        """Embed only current chunks missing from this collection."""
        if self._index_ready:
            return 0
        ids = [chunk.id for chunk in self.chunks if chunk.is_current]
        present = set(self.collection.get(ids=ids, include=[])["ids"])
        missing = [self.chunks_by_id[item] for item in ids if item not in present]
        for start in range(0, len(missing), 32):
            batch = missing[start : start + 32]
            documents = [
                f"{chunk.document_title}\n{chunk.section_path}\n{chunk.passage}"
                for chunk in batch
            ]
            vectors = self.embeddings.embed_documents(documents)
            self.embedding_calls += 1
            self.collection.add(
                ids=[chunk.id for chunk in batch],
                documents=documents,
                embeddings=vectors,
                metadatas=[
                    {
                        "title": chunk.document_title,
                        "section_path": chunk.section_path,
                        "version_id": chunk.version_id,
                        "source_url": chunk.source_url or "",
                    }
                    for chunk in batch
                ],
            )
        self.indexed_count += len(missing)
        self._index_ready = True
        return len(missing)

    def retrieve(
        self,
        query: str,
        _topic: str = "policy",
        _state: dict[str, Any] | None = None,
        limit: int = 5,
    ) -> tuple[list[PolicyChunk], dict[str, Any]]:
        """Fuse independent lexical and semantic rankings, falling back to BM25."""
        lexical, details = super().retrieve(query, _topic, _state, limit=max(limit, 8))
        if not query.strip():
            return lexical, details
        try:
            indexed = self.ensure_index()
            if not self.collection.count():
                details["vector_status"] = "empty_index; lexical fallback"
                return lexical[:limit], details
            vector = self.embeddings.embed_query(query)
            self.embedding_calls += 1
            result = self.collection.query(
                query_embeddings=[vector],
                n_results=min(12, self.collection.count()),
                include=["distances"],
            )
            vector_ids = result["ids"][0]
        except Exception:
            details["vector_status"] = "unavailable; lexical fallback"
            return lexical[:limit], details

        # Search both indexes independently: a BM25 miss must not hide a semantic match.
        # Weighted rank fusion preserves exact policy matches while allowing semantic ones.
        ranks: dict[str, float] = {}
        for rank, chunk in enumerate(lexical):
            ranks[chunk.id] = ranks.get(chunk.id, 0) + 2 / (rank + 1)
        for rank, chunk_id in enumerate(vector_ids):
            chunk = self.chunks_by_id.get(chunk_id)
            if chunk and chunk.is_current:
                ranks[chunk_id] = ranks.get(chunk_id, 0) + 1 / (rank + 1)
        selected_ids = sorted(ranks, key=ranks.get, reverse=True)[:limit]
        selected = [self.chunks_by_id[item] for item in selected_ids]
        details.update(
            backend="chroma_hybrid",
            selected_ids=selected_ids,
            indexed_new_chunks=indexed,
            embedding_calls=1 + (indexed + 31) // 32,
            vector_candidates=len(vector_ids),
            lexical_candidates=len(lexical),
        )
        return selected, details
