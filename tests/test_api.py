"""API tests for login, role restrictions, and authenticated chat behavior."""

from __future__ import annotations

import os

from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import FakeListChatModel

os.environ["AVIATIONSTACK_API_KEY"] = ""
os.environ["JWT_SECRET"] = "test-only-jwt-secret-with-at-least-32-bytes"

from etihad_assistant.api import app, database, settings  # noqa: E402


def _token(client: TestClient, username: str, password: str) -> str:
    response = client.post("/auth/token", data={"username": username, "password": password})
    assert response.status_code == 200
    return response.json()["access_token"]


def test_api_rbac_and_chat(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(settings, "database_path", tmp_path / "api.db")
    monkeypatch.setattr(database, "path", tmp_path / "api.db")
    monkeypatch.setattr(settings, "openai_api_key", "test-openai-key")
    monkeypatch.setattr(settings, "rag_backend", "bm25")
    monkeypatch.setattr(settings, "redis_url", None)
    monkeypatch.setattr(
        "etihad_assistant.llm.ChatOpenAI",
        lambda **_kwargs: FakeListChatModel(responses=["Etihad's refund conditions apply."]),
    )
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        user_token = _token(client, "user", "user-change-me")
        admin_token = _token(client, "admin", "admin-change-me")

        user_headers = {"Authorization": f"Bearer {user_token}"}
        admin_headers = {"Authorization": f"Bearer {admin_token}"}
        assert client.get("/admin/audit-logs", headers=user_headers).status_code == 403
        assert client.get("/admin/audit-logs", headers=admin_headers).status_code == 200
        assert client.get("/admin/dashboard", headers=user_headers).status_code == 403
        dashboard = client.get("/admin/dashboard", headers=admin_headers)
        assert dashboard.status_code == 200
        assert {"llm", "cost", "application", "recent_calls", "configuration"}.issubset(
            dashboard.json()
        )

        response = client.post(
            "/chat",
            headers=user_headers,
            json={"message": "What is the Etihad refund policy?"},
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["decision"] == "answer"
        assert any(item["document_title"] == "Refunds" for item in payload["citations"])
        assert "refund conditions" in payload["answer_markdown"]
        assert database.admin_dashboard()["llm"]["total_calls"] == 1
        conversation_id = payload["conversation_id"]
        listed = client.get("/conversations", headers=user_headers)
        assert listed.status_code == 200
        assert listed.json()[0]["id"] == conversation_id
        assert listed.json()[0]["title"] == "What is the Etihad refund policy?"
        assert listed.json()[0]["message_count"] == 2
        detail = client.get(f"/conversations/{conversation_id}", headers=user_headers)
        assert detail.status_code == 200
        assert [message["role"] for message in detail.json()["messages"]] == [
            "user", "assistant"
        ]
        assert detail.json()["messages"][1]["answer_id"] == payload["answer_id"]
        assert detail.json()["messages"][1]["citations"] == payload["citations"]
        assert client.get("/conversations", headers=admin_headers).json() == []
        assert client.get(
            f"/conversations/{conversation_id}", headers=admin_headers
        ).status_code == 404

    with TestClient(app) as restarted_client:
        assert restarted_client.get(
            f"/conversations/{conversation_id}", headers=user_headers
        ).status_code == 200
