"""Streamlit interaction tests for chat reruns, greeting, and admin costs."""

from __future__ import annotations

from pathlib import Path

import httpx
from streamlit.testing.v1 import AppTest


def test_chat_reply_renders_without_a_second_full_page_rerun(monkeypatch) -> None:
    calls = []

    def fake_request(method: str, url: str, **kwargs) -> httpx.Response:
        calls.append((method, url))
        request = httpx.Request(method, url)
        if url.endswith("/conversations?limit=50"):
            return httpx.Response(200, json=[], request=request)
        if url.endswith("/chat"):
            assert kwargs["json"]["message"] == "What is the baggage policy?"
            return httpx.Response(
                200,
                json={
                    "conversation_id": "chat-1",
                    "answer_id": "answer-1",
                    "answer_markdown": "The baggage policy depends on your fare.",
                    "decision": "answer",
                    "citations": [],
                },
                request=request,
            )
        raise AssertionError(f"Unexpected request: {method} {url}")

    monkeypatch.setattr(httpx, "request", fake_request)
    app = AppTest.from_file(
        Path(__file__).resolve().parents[1] / "ui/streamlit_app.py", default_timeout=60
    )
    app.session_state["token"] = "test-token"
    app.session_state["role"] = "user"
    app.session_state["guest_name"] = "Aisha"
    app.session_state["guest_email"] = "aisha@example.com"
    app.run()
    assert not app.exception

    app.chat_input[0].set_value("What is the baggage policy?").run()
    assert not app.exception
    assert app.session_state["conversation_id"] == "chat-1"
    assert [message["role"] for message in app.session_state["messages"]] == [
        "user", "assistant"
    ]
    assert app.session_state["messages"][1]["content"] == (
        "The baggage policy depends on your fare."
    )
    assert sum(url.endswith("/chat") for _method, url in calls) == 1
    # AppTest.run() simulates a full script run even for widgets in fragments;
    # a third history fetch here would indicate an explicit post-answer rerun.
    assert sum(url.endswith("/conversations?limit=50") for _method, url in calls) == 2


def test_guest_introduction_greets_by_name_and_clears_on_sign_out(monkeypatch) -> None:
    calls = []

    def fake_request(method: str, url: str, **_kwargs) -> httpx.Response:
        calls.append((method, url))
        assert url.endswith("/conversations?limit=50")
        return httpx.Response(200, json=[], request=httpx.Request(method, url))

    monkeypatch.setattr(httpx, "request", fake_request)
    app = AppTest.from_file(
        Path(__file__).resolve().parents[1] / "ui/streamlit_app.py", default_timeout=60
    )
    app.session_state["token"] = "test-token"
    app.session_state["role"] = "user"
    app.run()
    assert not app.exception
    assert not app.chat_input

    app.get_by_key("guest_name_input").set_value("  Aisha   Khan  ")
    app.get_by_key("guest_email_input").set_value("invalid-email")
    app.get_by_key("guest_start").click().run()
    assert "guest_name" not in app.session_state
    assert app.error

    app.get_by_key("guest_email_input").set_value("AISHA@example.com")
    app.get_by_key("guest_start").click().run()
    assert not app.exception
    assert app.session_state["guest_name"] == "Aisha Khan"
    assert app.session_state["guest_email"] == "aisha@example.com"
    assert app.chat_input
    assert any("Hello, Aisha Khan!" in item.value for item in app.markdown)
    assert all(not url.endswith("/chat") for _method, url in calls)

    next(button for button in app.button if "New conversation" in button.label).click().run()
    assert app.session_state["guest_name"] == "Aisha Khan"
    assert app.chat_input

    next(button for button in app.button if button.label == "Sign out").click().run()
    assert not app.exception
    assert "guest_name" not in app.session_state
    assert "guest_email" not in app.session_state


def test_admin_dashboard_displays_estimated_chat_cost(monkeypatch) -> None:
    dashboard = {
        "configuration": {
            "openai_configured": True,
            "openai_model": "gpt-4o-mini",
            "langsmith_tracing": False,
            "aviationstack_configured": False,
        },
        "llm": {
            "total_calls": 1,
            "successful_calls": 1,
            "failed_calls": 0,
            "total_tokens": 25,
            "input_tokens": 20,
            "output_tokens": 5,
            "cached_input_tokens": 3,
            "reasoning_tokens": 0,
            "average_latency_ms": 125.0,
        },
        "cost": {
            "estimated_total_usd": 0.000005775,
            "priced_calls": 1,
            "unpriced_calls": 0,
            "pricing_source": "https://developers.openai.com/api/docs/pricing",
            "pricing_as_of": "2026-10-11",
        },
        "application": {
            "policy_cache_entries": 0,
            "policy_cache_hits": 0,
            "users": 2,
            "conversations": 1,
            "messages": 2,
            "feedback_total": 0,
        },
        "daily": [],
        "by_model": [],
        "by_agent": [],
        "decisions": [],
        "recent_calls": [],
    }

    def fake_request(method: str, url: str, **_kwargs) -> httpx.Response:
        request = httpx.Request(method, url)
        if url.endswith("/conversations?limit=50"):
            payload = []
        elif url.endswith("/admin/dashboard?limit=250"):
            payload = dashboard
        elif url.endswith("/admin/audit-logs?limit=250"):
            payload = []
        else:
            raise AssertionError(f"Unexpected request: {method} {url}")
        return httpx.Response(200, json=payload, request=request)

    monkeypatch.setattr(httpx, "request", fake_request)
    app = AppTest.from_file(
        Path(__file__).resolve().parents[1] / "ui/streamlit_app.py", default_timeout=60
    )
    app.session_state["token"] = "admin-token"
    app.session_state["role"] = "admin"
    app.session_state["active_view"] = "Admin dashboard"
    app.session_state["admin_dashboard"] = {"llm": {"total_calls": 1}}
    app.run()
    assert not app.exception
    assert any(
        metric.label == "Estimated chat cost" and metric.value == "$0.00000578"
        for metric in app.metric
    )

