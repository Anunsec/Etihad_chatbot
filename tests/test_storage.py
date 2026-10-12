"""Persistence tests for Redis, Chroma, conversation ownership, and checkpoints."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

from etihad_assistant.cache import PolicyAnswerCache
from etihad_assistant.chroma_retrieval import ChromaPolicyRepository
from etihad_assistant.config import PROJECT_ROOT, Settings
from etihad_assistant.db import Database
from etihad_assistant.retrieval import PolicyRepository
from etihad_assistant.schemas import ChatRequest
from etihad_assistant.service import AssistantService


class FakeRedis:
    def __init__(self):
        self.values = {}
        self.ttls = {}

    def get(self, key):
        return self.values.get(key)

    def set(self, key, value, ex):
        self.values[key] = value
        self.ttls[key] = ex

    def incr(self, key):
        self.values[key] = str(int(self.values.get(key, 0)) + 1)

    def scan_iter(self, match):
        prefix = match.rstrip("*")
        return (key for key in self.values if key.startswith(prefix))


class FakeEmbeddings:
    def __init__(self):
        self.document_batches = 0
        self.queries = 0

    @staticmethod
    def vector(text):
        digest = hashlib.sha256(text.lower().encode()).digest()
        return [value / 255 for value in digest[:8]]

    def embed_documents(self, texts):
        self.document_batches += 1
        return [self.vector(text) for text in texts]

    def embed_query(self, text):
        self.queries += 1
        return self.vector(text)


def test_shared_redis_cache_and_tls_guard(tmp_path: Path):
    settings = Settings(
        database_path=tmp_path / "test.db",
        redis_url="redis://default:placeholder@example.invalid:6379",
        redis_require_tls=True,
    )
    db = Database(settings)
    db.initialize()
    guarded = PolicyAnswerCache(settings)
    assert guarded.backend == "unavailable"
    assert "TLS" in guarded.warning
    assert guarded.get("missing") == (None, "unavailable")
    assert guarded.put("missing", {"answer_markdown": "policy"}, ["source"]) == "unavailable"
    assert guarded.stats()["entries"] == 0

    client = FakeRedis()
    cache = PolicyAnswerCache(settings, redis_client=client)
    assert cache.put("abc", {"answer_markdown": "policy"}, ["source"]) == "redis"
    assert json.loads(client.values[cache.PREFIX + "abc"])["source_ids"] == ["source"]
    assert client.ttls[cache.PREFIX + "abc"] == settings.redis_cache_ttl_seconds
    another_instance = PolicyAnswerCache(settings, redis_client=client)
    value, backend = another_instance.get("abc")
    assert backend == "redis"
    assert value["draft"]["answer_markdown"] == "policy"
    assert another_instance.stats()["hits"] == 1


def test_redis_cache_recovers_after_startup_failure(monkeypatch, tmp_path: Path):
    class RecoveringRedis(FakeRedis):
        available = False

        def ping(self):
            if not self.available:
                raise TimeoutError("temporary outage")
            return True

    client = RecoveringRedis()
    monkeypatch.setattr("redis.Redis.from_url", lambda *_args, **_kwargs: client)
    settings = Settings(
        database_path=tmp_path / "test.db",
        redis_url="redis://default:placeholder@localhost:6379",
    )
    db = Database(settings)
    db.initialize()
    cache = PolicyAnswerCache(settings)
    assert cache.backend == "unavailable"
    assert "timed out" in cache.warning
    assert cache.put("during-outage", {"answer_markdown": "no"}, ["source"]) == "unavailable"
    assert cache.get("during-outage") == (None, "unavailable")

    client.available = True
    cache._next_probe_at = 0
    assert cache.put("recovered", {"answer_markdown": "yes"}, ["source"]) == "redis"
    assert cache.backend == "redis"
    assert cache.warning is None


def test_redis_request_failure_never_writes_sqlite(tmp_path: Path):
    class BrokenRedis(FakeRedis):
        def get(self, key):
            raise TimeoutError("temporary outage")

        def set(self, key, value, ex):
            raise TimeoutError("temporary outage")

    settings = Settings(database_path=tmp_path / "test.db", redis_url=None)
    db = Database(settings)
    db.initialize()
    cache = PolicyAnswerCache(settings, redis_client=BrokenRedis())
    assert cache.get("key") == (None, "unavailable")
    assert cache.put("key", {"answer_markdown": "answer"}, ["source"]) == "unavailable"
    assert cache.stats()["backend"] == "unavailable"
    with db.connect() as conn:
        assert conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'policy_answer_cache'"
        ).fetchone() is None


def test_chroma_policy_index_persists_without_reembedding(tmp_path: Path):
    settings = Settings(
        openai_api_key="test-key",
        policy_corpus_path=PROJECT_ROOT / "data/corpus/etihad_hub_policy_library.json",
        chroma_persist_directory=tmp_path / "chroma",
    )
    first_embeddings = FakeEmbeddings()
    first = ChromaPolicyRepository(settings, embeddings=first_embeddings)
    chunks, details = first.retrieve("What is the pets in cabin policy?")
    assert chunks
    assert details["backend"] == "chroma_hybrid"
    assert first.collection.count() == len([chunk for chunk in first.chunks if chunk.is_current])
    assert first_embeddings.document_batches > 0

    second_embeddings = FakeEmbeddings()
    second = ChromaPolicyRepository(settings, embeddings=second_embeddings)
    second.retrieve("What is the pets in cabin policy?")
    assert second_embeddings.document_batches == 0
    assert second_embeddings.queries == 1


def test_chroma_semantic_search_runs_after_bm25_miss(tmp_path: Path):
    class SemanticEmbeddings(FakeEmbeddings):
        @staticmethod
        def vector(text):
            if "furry companions aboard" in text.lower() or "pets in cabin" in text.lower():
                return [1.0, 0.0]
            return [0.0, 1.0]

    settings = Settings(
        openai_api_key="test-key",
        policy_corpus_path=PROJECT_ROOT / "data/corpus/etihad_hub_policy_library.json",
        chroma_persist_directory=tmp_path / "chroma",
    )
    embeddings = SemanticEmbeddings()
    repository = ChromaPolicyRepository(settings, embeddings=embeddings)
    question = "Can furry companions come aboard?"
    lexical, _ = PolicyRepository.retrieve(repository, question)
    assert lexical == []

    chunks, details = repository.retrieve(question)
    assert any("Animals and pets" == chunk.document_title for chunk in chunks)
    assert details["backend"] == "chroma_hybrid"
    assert details["lexical_candidates"] == 0
    assert details["vector_candidates"] > 0
    assert embeddings.queries == 1


def test_chroma_vector_failure_falls_back_to_bm25(tmp_path: Path):
    class BrokenQueryEmbeddings(FakeEmbeddings):
        def embed_query(self, text):
            raise TimeoutError("embedding service unavailable")

    settings = Settings(
        openai_api_key="test-key",
        policy_corpus_path=PROJECT_ROOT / "data/corpus/etihad_hub_policy_library.json",
        chroma_persist_directory=tmp_path / "chroma",
    )
    repository = ChromaPolicyRepository(settings, embeddings=BrokenQueryEmbeddings())
    chunks, details = repository.retrieve("What is the pets in cabin policy?")
    assert chunks
    assert details["backend"] == "local_bm25"
    assert details["vector_status"] == "unavailable; lexical fallback"

    missing, details = repository.retrieve("Can furry companions come aboard?")
    assert missing == []
    assert details["vector_status"] == "unavailable; lexical fallback"


def test_chat_history_is_scoped_to_conversation_owner(tmp_path: Path):
    settings = Settings(database_path=tmp_path / "test.db")
    db = Database(settings)
    db.initialize()
    user = db.authenticate(settings.demo_user_username, settings.demo_user_password)
    admin = db.authenticate(settings.demo_admin_username, settings.demo_admin_password)
    assert user and admin
    conversation, _ = db.load_or_create_conversation(None, user.id)
    db.add_message(conversation, "user", "My baggage question")
    db.add_message(conversation, "assistant", "Policy answer")
    assert [item["content"] for item in db.recent_messages(conversation, user.id)] == [
        "My baggage question", "Policy answer"
    ]
    assert db.recent_messages(conversation, admin.id) == []


def test_langgraph_sqlite_checkpoint_survives_service_restart(monkeypatch, tmp_path: Path):
    settings = Settings(
        database_path=tmp_path / "test.db", rag_backend="bm25", redis_url=None,
    )
    db = Database(settings)
    db.initialize()
    user = db.authenticate(settings.demo_user_username, settings.demo_user_password)
    assert user
    first_service = AssistantService(settings, db)
    try:
        first = first_service.chat(
            ChatRequest(message="What is the status of flight EY 219?"), user
        )
        assert first.decision == "clarify"
        thread_id = f"{user.id}:{first.conversation_id}"
        config = {"configurable": {"thread_id": thread_id}}
        saved = first_service.graph.get_state(config).values
        assert saved["app_state"]["pending_intent"] == "flight_status"
    finally:
        first_service.close()

    with sqlite3.connect(settings.checkpoint_db_file) as conn:
        assert conn.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0] > 0

    restarted_service = AssistantService(settings, db)
    try:
        assert restarted_service.graph.get_state(config).values["app_state"] == saved["app_state"]
        calls = []

        def fake_mcp(_provider, tool_name, arguments):
            calls.append((tool_name, arguments))
            return {"status": "not_found", "notice": "No matching AviationStack flight."}

        monkeypatch.setattr("etihad_assistant.graph.invoke_flight_mcp_sync", fake_mcp)
        follow_up = restarted_service.chat(
            ChatRequest(message="2026-10-10", conversation_id=first.conversation_id), user
        )
        assert follow_up.decision == "not_found"
        assert calls == [
            ("get_flight_status", {"flight_number": "EY219", "departure_date": "2026-10-10"})
        ]
    finally:
        restarted_service.close()

