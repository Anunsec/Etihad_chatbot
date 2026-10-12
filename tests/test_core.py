"""End-to-end assistant tests for routing, RAG, caching, safety, and usage."""

from __future__ import annotations

import json
from pathlib import Path

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from etihad_assistant.config import PROJECT_ROOT, Settings
from etihad_assistant.db import Database
from etihad_assistant.graph import _extract_context, _missing_flight_slot, _policy_cache_key
from etihad_assistant.guardrails import contains_unsupported_promise, is_off_topic
from etihad_assistant.ingestion import CATEGORY_NAMES, build_corpus
from etihad_assistant.llm import LLMInvocation, LoggedLangChainLLM
from etihad_assistant.schemas import ChatRequest, Role
from etihad_assistant.service import AssistantService

CORPUS = PROJECT_ROOT / "data/corpus/etihad_hub_policy_library.json"


class FakeRedis:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}

    def get(self, key: str) -> str | None:
        return self.values.get(key)

    def set(self, key: str, value: str, ex: int) -> None:
        self.values[key] = value

    def incr(self, key: str) -> None:
        self.values[key] = str(int(self.values.get(key, "0")) + 1)

    def scan_iter(self, match: str):
        prefix = match.removesuffix("*")
        return (key for key in self.values if key.startswith(prefix))


def attach_fake_redis(service: AssistantService, client: FakeRedis) -> None:
    service.cache.client = client
    service.cache._redis_candidate = client
    service.cache.backend = "redis"
    service.cache.warning = None


def make_service(
    tmp_path: Path, openai_api_key: str | None = None, *, use_fake_redis: bool = True
) -> tuple[AssistantService, Database]:
    settings = Settings(
        openai_api_key=openai_api_key,
        database_path=tmp_path / "test.db",
        audit_log_path=tmp_path / "audit.jsonl",
        policy_corpus_path=CORPUS,
        jwt_secret="test-secret",
        demo_user_password="test-user",
        demo_admin_password="test-admin",
        langsmith_tracing=False,
        aviationstack_api_key=None,
        rag_backend="bm25",
        redis_url=None,
    )
    database = Database(settings)
    database.initialize()
    service = AssistantService(settings, database)
    if use_fake_redis:
        attach_fake_redis(service, FakeRedis())
    return service, database


def test_corpus_is_complete_etihad_hub_snapshot() -> None:
    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
    assert corpus["record_count"] == 55
    assert corpus["chunk_count"] == 194
    assert list(corpus["categories"]) == CATEGORY_NAMES
    assert sum(corpus["categories"].values()) == 55
    assert all(record["html"] and record["text"] for record in corpus["records"])
    assert all(chunk["fixture"] is False for chunk in corpus["chunks"])


def test_policy_answer_is_refined_then_shared_cache_skips_llm(
    monkeypatch, tmp_path: Path
) -> None:
    service, database = make_service(tmp_path, openai_api_key="test-openai-key")
    monkeypatch.setattr(
        "etihad_assistant.llm.ChatOpenAI",
        lambda **_kwargs: FakeListChatModel(
            responses=["Etihad permits cats and dogs in the cabin subject to its pet rules."]
        ),
    )
    user = database.authenticate("user", "test-user")
    admin = database.authenticate("admin", "test-admin")
    assert user and user.role == Role.user
    assert admin and admin.role == Role.admin

    response = service.chat(ChatRequest(message="What are the rules for pets in cabin?"), user)

    assert response.decision.value == "answer"
    assert response.answer_markdown.startswith("Etihad permits cats and dogs")
    assert response.citations
    assert any("Pets in Cabin" in citation.document_title for citation in response.citations)
    assert all(citation.fixture is False for citation in response.citations)
    assert all(citation.canonical_url for citation in response.citations)
    assert len(database.list_llm_calls()) == 1
    assert database.list_llm_calls()[0]["agent_name"] == "policy_rag_refinement"
    assert service.audit.tail()[-1]["retrieval"]["llm_calls"] == 1

    restarted_service = AssistantService(service.settings, database)
    attach_fake_redis(restarted_service, service.cache.client)
    cached = restarted_service.chat(
        ChatRequest(message="  what are the rules for pets in cabin  "), admin
    )
    assert cached.decision.value == "answer"
    assert cached.answer_markdown == response.answer_markdown
    assert {item.id for item in cached.citations} == {item.id for item in response.citations}
    assert len(database.list_llm_calls()) == 1
    assert service.audit.tail()[-1]["retrieval"]["backend"] == "policy_answer_cache"
    assert service.audit.tail()[-1]["retrieval"]["llm_calls"] == 0
    assert service.cache.stats()["entries"] == 1
    assert service.cache.stats()["hits"] == 1


def test_policy_answer_is_not_cached_when_redis_is_unavailable(
    monkeypatch, tmp_path: Path
) -> None:
    service, database = make_service(
        tmp_path, openai_api_key="test-openai-key", use_fake_redis=False
    )
    monkeypatch.setattr(
        "etihad_assistant.llm.ChatOpenAI",
        lambda **_kwargs: FakeListChatModel(responses=["Pets are subject to Etihad's rules."]),
    )
    user = database.authenticate("user", "test-user")
    assert user

    question = ChatRequest(message="What are the rules for pets in cabin?")
    first = service.chat(question, user)
    second = service.chat(question, user)

    assert first.decision.value == second.decision.value == "answer"
    assert len(database.list_llm_calls()) == 2
    assert service.cache.stats()["backend"] == "unavailable"
    assert service.cache.stats()["entries"] == 0
    with database.connect() as conn:
        assert conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'policy_answer_cache'"
        ).fetchone() is None


def test_policy_cache_key_invalidates_on_source_or_model_change() -> None:
    question = "What is the baggage policy?"
    first = _policy_cache_key(question, "corpus-a", "model-a")
    assert first == _policy_cache_key("  what is the baggage policy  ", "corpus-a", "model-a")
    assert first != _policy_cache_key(question, "corpus-b", "model-a")
    assert first != _policy_cache_key(question, "corpus-a", "model-b")
    assert _policy_cache_key("What is my booking policy?", "corpus-a", "model-a") is None


def test_generic_policy_questions_are_not_rejected_by_intent_router(
    monkeypatch, tmp_path: Path
) -> None:
    service, database = make_service(tmp_path, openai_api_key="test-openai-key")
    monkeypatch.setattr(
        "etihad_assistant.llm.ChatOpenAI",
        lambda **_kwargs: FakeListChatModel(responses=["A concise source-grounded answer."]),
    )
    user = database.authenticate("user", "test-user")
    assert user

    refund = service.chat(ChatRequest(message="What is the Etihad refund policy?"), user)
    assert refund.decision.value == "answer"
    assert any(citation.document_title == "Refunds" for citation in refund.citations)

    correction = service.chat(ChatRequest(message="How can a passenger name be corrected?"), user)
    assert correction.decision.value == "answer"
    assert any("Name correction" in citation.document_title for citation in correction.citations)


def test_platinum_member_benefits_are_in_scope_and_use_official_rag(
    monkeypatch, tmp_path: Path
) -> None:
    service, database = make_service(tmp_path, openai_api_key="test-openai-key")
    user = database.authenticate("user", "test-user")
    assert user
    prompts: list[str] = []

    def fake_invoke(_self, **kwargs) -> LLMInvocation:
        prompts.append(kwargs["user_prompt"])
        return LLMInvocation(
            text=(
                "Etihad Guest Platinum includes 100% bonus miles, priority airport services, "
                "lounge access and free onboard surfing. Members also select five Custom "
                "Benefits, while Beyond Benefits unlock at 175,000 Tier Miles, subject to terms."
            ),
            attempted=True,
        )

    monkeypatch.setattr(LoggedLangChainLLM, "invoke", fake_invoke)

    first = service.chat(
        ChatRequest(message="I am a platinum member what are the benefits?"), user
    )
    second = service.chat(
        ChatRequest(message="I am a platinum member of etihad what are the benefits?"), user
    )

    assert not is_off_topic("I am a platinum member what are the benefits?")
    assert first.decision.value == second.decision.value == "answer"
    assert all(
        citation.document_title == "Etihad Guest Platinum benefits"
        for response in (first, second)
        for citation in response.citations
    )
    assert all(response.citations for response in (first, second))
    assert all("core benefits" in prompt.casefold() for prompt in prompts)
    assert all("custom benefits" in prompt.casefold() for prompt in prompts)
    assert all("beyond benefits" in prompt.casefold() for prompt in prompts)


def test_food_options_are_in_scope_and_use_onboard_meal_sources(
    monkeypatch, tmp_path: Path
) -> None:
    service, database = make_service(tmp_path, openai_api_key="test-openai-key")
    user = database.authenticate("user", "test-user")
    admin = database.authenticate("admin", "test-admin")
    assert user and admin
    prompts: list[str] = []

    def fake_invoke(_self, **kwargs) -> LLMInvocation:
        prompts.append(kwargs["user_prompt"])
        return LLMInvocation(
            text=(
                "Etihad offers complimentary Halal meals or snacks in Economy, "
                "à la carte dining in Business and gourmet dining in First. "
                "Special dietary and children's meals can be requested in advance, "
                "subject to flight length and availability."
            ),
            attempted=True,
        )

    monkeypatch.setattr(LoggedLangChainLLM, "invoke", fake_invoke)

    response = service.chat(ChatRequest(message="what are the food options"), user)
    cached = service.chat(ChatRequest(message="what are the food options"), admin)

    assert not is_off_topic("what are the food options")
    assert response.decision.value == cached.decision.value == "answer"
    assert response.answer_markdown == cached.answer_markdown
    assert len(prompts) == 1
    assert "Economy" in prompts[0]
    assert "Kosher" in prompts[0]
    assert "Children's" in prompts[0]
    assert "pet food" not in prompts[0].casefold()
    assert response.citations
    assert all(citation.canonical_url.startswith("https://www.etihad.com/") for citation in response.citations)


def test_baggage_refinement_prompt_is_limited_to_matching_policy_excerpt(
    monkeypatch, tmp_path: Path
) -> None:
    service, database = make_service(tmp_path, openai_api_key="test-openai-key")
    user = database.authenticate("user", "test-user")
    assert user
    captured: dict[str, str] = {}

    def fake_invoke(_self, **kwargs) -> LLMInvocation:
        captured.update(kwargs)
        return LLMInvocation(
            text=(
                "For Economy Delhi–Abu Dhabi, Etihad allows one cabin bag up to 7 kg. "
                "Checked baggage depends on the fare: Basic has no included allowance; "
                "Value 25 kg, Comfort 30 kg, and Deluxe 40 kg."
            ),
            attempted=True,
        )

    monkeypatch.setattr(LoggedLangChainLLM, "invoke", fake_invoke)
    response = service.chat(ChatRequest(
        message="I am travelling from Delhi to Abu Dhabi in economy class what is the baggage policy"
    ), user)

    assert response.decision.value == "answer"
    assert "Economy Delhi" in response.answer_markdown
    assert "Value 25 kg" in response.answer_markdown
    assert "Economy baggage allowance by fare" in captured["user_prompt"]
    assert "Value" in captured["user_prompt"]
    assert "Comfort" in captured["user_prompt"]
    assert "Extra Seat" not in captured["user_prompt"]
    assert "No-show Policy" not in captured["user_prompt"]
    assert len(captured["user_prompt"]) < 4000
    assert response.citations
    assert all(item.document_title == "Economy baggage allowance by fare" for item in response.citations)
    admin = database.authenticate("admin", "test-admin")
    assert admin
    cached = service.chat(ChatRequest(
        message="  i am travelling from delhi to abu dhabi in economy class what is the baggage policy?  "
    ), admin)
    assert cached.answer_markdown == response.answer_markdown
    assert service.audit.tail()[-1]["retrieval"]["backend"] == "policy_answer_cache"


def test_first_delhi_london_flight_uses_mcp_not_policy_or_general_llm(
    monkeypatch, tmp_path: Path
) -> None:
    service, database = make_service(tmp_path, openai_api_key="test-openai-key")
    user = database.authenticate("user", "test-user")
    assert user
    calls: list[tuple[str, dict[str, str]]] = []

    def fake_mcp(_provider, tool_name: str, arguments: dict[str, str]) -> dict:
        calls.append((tool_name, arguments))
        return {
            "status": "ok", "provider": "aviationstack",
            "retrieved_at": "2026-10-07T12:00:00+00:00",
            "query": {"origin_iata": "DEL", "destination_iata": "LHR",
                      "travel_date": arguments["travel_date"]},
            "first_flight": {
                "flight": {"iata": "BA256"},
                "departure": {"iata": "DEL", "scheduled": "2026-10-07T06:00:00+05:30"},
                "arrival": {"iata": "LHR"},
            },
            "matching_flights": 2, "schedule_scope": "flight_data",
        }

    monkeypatch.setattr("etihad_assistant.graph.invoke_flight_mcp_sync", fake_mcp)
    monkeypatch.setattr(
        "etihad_assistant.llm.ChatOpenAI",
        lambda **_kwargs: FakeListChatModel(responses=["AviationStack provided the departure time."]),
    )
    response = service.chat(ChatRequest(message="tell me the first flight from delhi to london"), user)
    assert calls and calls[0][0] == "get_first_flight"
    assert calls[0][1]["origin_iata"] == "DEL"
    assert calls[0][1]["destination_iata"] == "LHR"
    assert response.decision.value == "answer"
    assert "BA256" in response.answer_markdown
    assert "Heathrow (LHR)" in response.answer_markdown
    assert database.list_llm_calls()[0]["agent_name"] == "aviationstack_mcp_answer"


def test_first_flight_without_provider_key_fails_closed(tmp_path: Path) -> None:
    service, database = make_service(tmp_path)
    user = database.authenticate("user", "test-user")
    assert user
    response = service.chat(ChatRequest(message="tell me the first flight from delhi to london"), user)
    assert response.decision.value == "tool_unavailable"
    assert "AVIATIONSTACK_API_KEY" in response.answer_markdown
    assert database.list_llm_calls() == []


def test_abu_dhabi_spellings_go_straight_to_first_flight_mcp(
    monkeypatch, tmp_path: Path
) -> None:
    service, database = make_service(tmp_path)
    user = database.authenticate("user", "test-user")
    assert user
    calls: list[tuple[str, dict[str, str]]] = []

    def fake_mcp(_provider, tool_name: str, arguments: dict[str, str]) -> dict:
        calls.append((tool_name, arguments))
        return {"status": "not_found", "notice": "No matching AviationStack flight data."}

    monkeypatch.setattr("etihad_assistant.graph.invoke_flight_mcp_sync", fake_mcp)
    for question in (
        "whats the first flight from delhi to abudhabi",
        "i want info about first flight details from delhi to abudhabi",
        "what's the first flight from Delhi to Abu-Dhabi?",
    ):
        response = service.chat(ChatRequest(message=question), user)
        assert response.decision.value == "not_found"
        assert response.state["trip"] == {"origin": "DEL", "destination": "AUH"}
    assert [tool_name for tool_name, _ in calls] == ["get_first_flight"] * 3
    assert all(args["origin_iata"] == "DEL" and args["destination_iata"] == "AUH"
               for _, args in calls)
    assert database.list_llm_calls() == []


def test_destination_clarification_reply_keeps_first_flight_intent(
    monkeypatch, tmp_path: Path
) -> None:
    service, database = make_service(tmp_path)
    user = database.authenticate("user", "test-user")
    assert user
    calls: list[tuple[str, dict[str, str]]] = []

    def fake_mcp(_provider, tool_name: str, arguments: dict[str, str]) -> dict:
        calls.append((tool_name, arguments))
        return {"status": "not_found", "notice": "No matching AviationStack flight data."}

    monkeypatch.setattr("etihad_assistant.graph.invoke_flight_mcp_sync", fake_mcp)
    first = service.chat(
        ChatRequest(message="whats the first flight from delhi to my destination"), user
    )
    assert first.decision.value == "clarify"
    assert first.state["pending_slot"] == "trip.destination"

    second = service.chat(
        ChatRequest(message="i said abu dhabi", conversation_id=first.conversation_id), user
    )
    assert second.decision.value == "not_found"
    assert calls == [("get_first_flight", {
        "origin_iata": "DEL", "destination_iata": "AUH",
        "travel_date": second.state["flight"]["departure_date"],
    })]
    assert "pending_intent" not in second.state
    assert database.list_llm_calls() == []


def test_pending_flight_does_not_hijack_new_policy_question() -> None:
    state = {"pending_intent": "first_flight", "pending_slot": "trip.destination",
             "trip": {"origin": "DEL"}}
    intent, updated = _extract_context("What is baggage policy in Abu Dhabi?", state)
    assert intent == "policy"
    assert _missing_flight_slot(intent, updated) is None


def test_policy_cache_is_scoped_by_question_and_model(monkeypatch, tmp_path: Path) -> None:
    service, database = make_service(tmp_path, openai_api_key="test-openai-key")
    user = database.authenticate("user", "test-user")
    assert user
    monkeypatch.setattr(
        "etihad_assistant.llm.ChatOpenAI",
        lambda **_kwargs: FakeListChatModel(responses=["A concise source-grounded answer."]),
    )
    service.chat(ChatRequest(message="What are the rules for pets in cabin?"), user)
    service.chat(ChatRequest(message="What is the baggage policy?"), user)
    service.settings.openai_chat_model = "different-test-model"
    service.chat(ChatRequest(message="What are the rules for pets in cabin?"), user)
    assert len(database.list_llm_calls()) == 3
    assert service.cache.stats()["entries"] == 3


def test_failed_policy_refinement_is_not_cached(monkeypatch, tmp_path: Path) -> None:
    service, database = make_service(tmp_path)
    user = database.authenticate("user", "test-user")
    assert user
    question = ChatRequest(message="What is the Etihad refund policy?")
    first = service.chat(question, user)
    assert first.decision.value == "tool_unavailable"
    assert service.cache.stats()["entries"] == 0

    service.settings.openai_api_key = "test-openai-key"
    monkeypatch.setattr(
        "etihad_assistant.llm.ChatOpenAI",
        lambda **_kwargs: FakeListChatModel(responses=["The refund terms depend on the fare."]),
    )
    second = service.chat(question, user)
    assert second.decision.value == "answer"
    assert len(database.list_llm_calls()) == 1
    assert service.cache.stats()["entries"] == 1


def test_role_and_audit_records_policy_refinement(monkeypatch, tmp_path: Path) -> None:
    service, database = make_service(tmp_path, openai_api_key="test-openai-key")
    monkeypatch.setattr(
        "etihad_assistant.llm.ChatOpenAI",
        lambda **_kwargs: FakeListChatModel(responses=["The cooling-off policy is described."]),
    )
    admin = database.authenticate("admin", "test-admin")
    assert admin and admin.role == Role.admin
    response = service.chat(ChatRequest(message="Explain the 24 hour cooling-off policy"), admin)
    assert response.decision.value == "answer"
    logs = service.audit.tail()
    assert logs[-1]["retrieval"]["backend"] == "local_bm25"
    assert logs[-1]["retrieval"]["llm_calls"] == 1
    assert "message" not in logs[-1]


def test_flight_status_fails_closed_without_aviationstack_key(tmp_path: Path) -> None:
    service, database = make_service(tmp_path)
    user = database.authenticate("user", "test-user")
    assert user
    response = service.chat(ChatRequest(message="Is flight EY20 on time today?"), user)
    assert response.decision.value == "tool_unavailable"
    assert "AVIATIONSTACK_API_KEY" in response.answer_markdown
    assert response.citations == []


def test_flight_status_date_followup_keeps_intent_and_reports_provider_status(
    monkeypatch, tmp_path: Path
) -> None:
    service, database = make_service(tmp_path, openai_api_key="test-openai-key")
    user = database.authenticate("user", "test-user")
    assert user
    calls: list[tuple[str, dict[str, str]]] = []

    def fake_mcp(_provider, tool_name: str, arguments: dict[str, str]) -> dict:
        calls.append((tool_name, arguments))
        return {
            "status": "ok",
            "provider": "aviationstack",
            "retrieved_at": "2026-10-07T12:00:00+00:00",
            "query": {"flight_number": "EY219", "departure_date": "2026-10-10"},
            "flight_status": "scheduled",
            "flight": {"flight_date": "2026-10-10", "flight": {"iata": "EY219"}},
        }

    monkeypatch.setattr("etihad_assistant.graph.invoke_flight_mcp_sync", fake_mcp)
    monkeypatch.setattr(
        "etihad_assistant.llm.ChatOpenAI",
        lambda **_kwargs: FakeListChatModel(responses=["AviationStack lists this flight as scheduled."]),
    )

    first = service.chat(ChatRequest(message="what the flight status of ey 219"), user)
    assert first.decision.value == "clarify"
    assert first.state["pending_intent"] == "flight_status"
    assert first.state["flight"]["number"] == "EY219"
    assert database.list_llm_calls() == []

    second = service.chat(
        ChatRequest(message="2026-10-10", conversation_id=first.conversation_id), user
    )
    assert calls == [
        (
            "get_flight_status",
            {"flight_number": "EY219", "departure_date": "2026-10-10"},
        )
    ]
    assert second.decision.value == "answer"
    assert second.answer_markdown.startswith("**EY219 on 2026-10-10: scheduled**")
    assert second.citations[0].document_title == "AviationStack"
    assert "pending_intent" not in second.state
    assert len(database.list_llm_calls()) == 1
    assert database.list_llm_calls()[0]["agent_name"] == "aviationstack_mcp_answer"


def test_missing_flight_status_does_not_call_llm(monkeypatch, tmp_path: Path) -> None:
    service, database = make_service(tmp_path, openai_api_key="test-openai-key")
    user = database.authenticate("user", "test-user")
    assert user

    def fake_mcp(_provider, _tool_name: str, _arguments: dict[str, str]) -> dict:
        return {
            "status": "not_found",
            "flight_status": None,
            "notice": "AviationStack has no matching status for EY219 on 2026-10-10.",
        }

    monkeypatch.setattr("etihad_assistant.graph.invoke_flight_mcp_sync", fake_mcp)
    response = service.chat(
        ChatRequest(message="flight status of EY219 on 2026-10-10"), user
    )
    assert response.decision.value == "not_found"
    assert "no matching status" in response.answer_markdown
    assert response.citations == []
    assert database.list_llm_calls() == []


def test_future_flight_airport_clarification_remains_in_same_conversation(
    monkeypatch, tmp_path: Path
) -> None:
    service, database = make_service(tmp_path, openai_api_key="test-openai-key")
    user = database.authenticate("user", "test-user")
    assert user
    calls: list[dict[str, str]] = []

    def fake_mcp(_provider, _tool_name: str, arguments: dict[str, str]) -> dict:
        calls.append(arguments)
        if "origin_iata" not in arguments:
            return {
                "status": "clarify",
                "question": "What is the departure airport IATA code for EY219 on 2026-10-10?",
            }
        return {
            "status": "ok",
            "provider": "aviationstack",
            "retrieved_at": "2026-10-07T12:00:00+00:00",
            "query": {"flight_number": "EY219", "departure_date": "2026-10-10"},
            "flight_status": "scheduled",
            "status_source": "confirmed_future_schedule_not_live_status",
        }

    monkeypatch.setattr("etihad_assistant.graph.invoke_flight_mcp_sync", fake_mcp)
    monkeypatch.setattr(
        "etihad_assistant.llm.ChatOpenAI",
        lambda **_kwargs: FakeListChatModel(responses=["Confirmed by AviationStack."]),
    )
    first = service.chat(
        ChatRequest(message="flight status EY219 on 2026-10-10"), user
    )
    assert first.decision.value == "clarify"
    assert first.state["pending_slot"] == "trip.origin"

    second = service.chat(ChatRequest(message="AUH", conversation_id=first.conversation_id), user)
    assert second.decision.value == "answer"
    assert calls[-1]["origin_iata"] == "AUH"
    assert "confirmed future schedule, not a live operational status" in second.answer_markdown


def test_delhi_arrivals_route_to_mcp_and_answer_from_provider(monkeypatch, tmp_path: Path) -> None:
    service, database = make_service(tmp_path, openai_api_key="test-openai-key")
    user = database.authenticate("user", "test-user")
    assert user
    calls: list[tuple[str, dict[str, str]]] = []

    def fake_mcp(_provider, tool_name: str, arguments: dict[str, str]) -> dict:
        calls.append((tool_name, arguments))
        return {
            "status": "ok",
            "provider": "aviationstack",
            "retrieved_at": "2026-10-07T12:00:00+00:00",
            "airport_iata": "DEL",
            "notice": "Showing up to 20 matching records from the current AviationStack board.",
            "arrivals": [
                {
                    "flight_iata": "EY218",
                    "from_iata": "AUH",
                    "scheduled_arrival": "2026-10-07T18:30:00",
                    "flight_status": "scheduled",
                }
            ],
        }

    monkeypatch.setattr("etihad_assistant.graph.invoke_flight_mcp_sync", fake_mcp)
    monkeypatch.setattr(
        "etihad_assistant.llm.ChatOpenAI",
        lambda **_kwargs: FakeListChatModel(responses=["One arrival is listed by AviationStack."]),
    )
    response = service.chat(ChatRequest(message="Show flights arriving at Delhi airport."), user)

    assert calls == [("list_airport_arrivals", {"airport_iata": "DEL"})]
    assert response.decision.value == "answer"
    assert "Current arrivals at DEL" in response.answer_markdown
    assert "EY218" in response.answer_markdown
    assert "No-show Policy" not in response.answer_markdown
    assert [citation.document_title for citation in response.citations] == ["AviationStack"]
    assert len(database.list_llm_calls()) == 1
    assert service.audit.tail()[-1]["intent"] == "airport_arrivals"


def test_delhi_arrivals_fails_closed_without_aviationstack_key(tmp_path: Path) -> None:
    service, database = make_service(tmp_path)
    user = database.authenticate("user", "test-user")
    assert user
    response = service.chat(ChatRequest(message="Show flights arriving at Delhi airport."), user)
    assert response.decision.value == "tool_unavailable"
    assert "AVIATIONSTACK_API_KEY" in response.answer_markdown
    assert response.citations == []
    assert database.list_llm_calls() == []


def test_arrivals_ask_for_airport_and_keep_intent(monkeypatch, tmp_path: Path) -> None:
    service, database = make_service(tmp_path, openai_api_key="test-openai-key")
    user = database.authenticate("user", "test-user")
    assert user

    def fake_mcp(_provider, _tool_name: str, arguments: dict[str, str]) -> dict:
        return {"status": "not_found", "notice": f"No arrivals at {arguments['airport_iata']}."}

    monkeypatch.setattr("etihad_assistant.graph.invoke_flight_mcp_sync", fake_mcp)
    first = service.chat(ChatRequest(message="Show airport arrivals."), user)
    assert first.decision.value == "clarify"
    assert first.state["pending_intent"] == "airport_arrivals"
    second = service.chat(ChatRequest(message="DEL", conversation_id=first.conversation_id), user)
    assert second.decision.value == "not_found"
    assert "No arrivals at DEL" in second.answer_markdown
    assert database.list_llm_calls() == []


def test_off_topic_question_is_declined_without_llm(tmp_path: Path) -> None:
    service, database = make_service(tmp_path)
    user = database.authenticate("user", "test-user")
    assert user
    response = service.chat(ChatRequest(message="Explain quantum entanglement"), user)
    assert response.decision.value == "off_topic"
    assert "Etihad travel policies and flight information" in response.answer_markdown
    assert response.citations == []
    assert database.list_llm_calls() == []


def test_unknown_etihad_policy_is_honestly_refused_without_llm(tmp_path: Path) -> None:
    service, database = make_service(tmp_path)
    user = database.authenticate("user", "test-user")
    assert user

    response = service.chat(
        ChatRequest(message="What is Etihad's blockchain mining policy?"), user
    )

    assert response.decision.value == "not_found"
    assert "don't have enough information" in response.answer_markdown
    assert response.citations == []
    assert database.list_llm_calls() == []


def test_unsupported_refund_promise_is_blocked_and_not_cached(
    monkeypatch, tmp_path: Path
) -> None:
    service, database = make_service(tmp_path, openai_api_key="test-openai-key")
    user = database.authenticate("user", "test-user")
    assert user
    calls: list[str] = []

    def fake_invoke(_self, **kwargs) -> LLMInvocation:
        calls.append(kwargs["agent_name"])
        return LLMInvocation(text="You will receive a full refund.", attempted=True)

    monkeypatch.setattr(LoggedLangChainLLM, "invoke", fake_invoke)
    request = ChatRequest(message="What is the Etihad refund policy?")
    first = service.chat(request, user)
    second = service.chat(request, user)

    assert first.decision.value == second.decision.value == "not_found"
    assert "can't confirm or promise" in first.answer_markdown
    assert first.citations == second.citations == []
    assert calls == ["policy_rag_refinement", "policy_rag_refinement"]
    assert service.cache.stats()["entries"] == 0


def test_scope_and_promise_rules_keep_conditional_policy_explanations() -> None:
    assert is_off_topic("Tell me a joke about Etihad")
    assert is_off_topic("What is your name?")
    assert not is_off_topic("What is the Etihad baggage policy?")
    assert contains_unsupported_promise("Etihad will issue you a refund.")
    assert contains_unsupported_promise("I can waive your fee.")
    assert contains_unsupported_promise("You qualify for compensation.")
    assert contains_unsupported_promise("Your fee has been waived.")
    assert not contains_unsupported_promise(
        "A refund may be available if the written fare conditions are met."
    )


def test_ingestion_rejects_partial_page() -> None:
    try:
        build_corpus("<html><body>partial</body></html>")
    except ValueError as exc:
        assert "Expected 5 policy groups" in str(exc)
    else:
        raise AssertionError("Partial policy pages must fail closed")


def test_admin_dashboard_aggregates_llm_usage(tmp_path: Path) -> None:
    _, database = make_service(tmp_path)
    database.log_llm_call(
        {
            "id": "call-1",
            "timestamp_utc": "2026-10-06T12:00:00+00:00",
            "user_id": "user-1",
            "conversation_id": "conversation-1",
            "turn_id": "turn-1",
            "agent_name": "general_answer",
            "provider": "openai",
            "model": "gpt-4o-mini",
            "prompt_version": "general-v1",
            "input_tokens": 20,
            "output_tokens": 5,
            "cached_input_tokens": 3,
            "reasoning_tokens": 0,
            "latency_ms": 125.0,
            "status": "ok",
        }
    )
    dashboard = database.admin_dashboard()
    assert dashboard["llm"]["total_calls"] == 1
    assert dashboard["llm"]["total_tokens"] == 25
    assert dashboard["llm"]["cached_input_tokens"] == 3
    assert dashboard["cost"]["estimated_total_usd"] == 0.000005775
    assert dashboard["cost"]["priced_calls"] == 1
    assert dashboard["cost"]["unpriced_calls"] == 0
    assert dashboard["by_model"][0]["estimated_cost_usd"] == 0.000005775
    assert dashboard["daily"][0]["estimated_cost_usd"] == 0.000005775
    assert dashboard["recent_calls"][0]["estimated_cost_usd"] == 0.000005775
    assert dashboard["by_agent"][0]["agent_name"] == "general_answer"

    database.log_llm_call(
        {
            "id": "call-2",
            "timestamp_utc": "2026-10-06T12:01:00+00:00",
            "user_id": "user-1",
            "conversation_id": "conversation-1",
            "turn_id": "turn-2",
            "agent_name": "general_answer",
            "provider": "openai",
            "model": "custom-model",
            "prompt_version": "general-v1",
            "input_tokens": 20,
            "output_tokens": 5,
            "latency_ms": 125.0,
            "status": "ok",
        }
    )
    dashboard = database.admin_dashboard()
    assert dashboard["cost"]["estimated_total_usd"] == 0.000005775
    assert dashboard["cost"]["unpriced_calls"] == 1
    assert dashboard["cost"]["unpriced_reasons"] == {"unsupported_model": 1}
    unpriced_model = next(row for row in dashboard["by_model"] if row["model"] == "custom-model")
    assert unpriced_model["estimated_cost_usd"] is None
    assert unpriced_model["unpriced_calls"] == 1


def test_langchain_openai_wrapper_logs_attempt(monkeypatch, tmp_path: Path) -> None:
    settings = Settings(
        openai_api_key="test-key",
        database_path=tmp_path / "llm.db",
        audit_log_path=tmp_path / "audit.jsonl",
        policy_corpus_path=CORPUS,
    )
    database = Database(settings)
    database.initialize()
    monkeypatch.setattr(
        "etihad_assistant.llm.ChatOpenAI",
        lambda **_kwargs: FakeListChatModel(responses=["LangChain answer"]),
    )
    result = LoggedLangChainLLM(settings, database).invoke(
        agent_name="general_answer",
        system_prompt="Answer concisely.",
        user_prompt="Hello",
        user_id="user-1",
        conversation_id="conversation-1",
        turn_id="turn-1",
        trace_id="trace-1",
        prompt_version="test-v1",
    )
    assert result.text == "LangChain answer"
    rows = database.list_llm_calls()
    assert len(rows) == 1
    assert rows[0]["status"] == "ok"
    assert rows[0]["agent_name"] == "general_answer"
