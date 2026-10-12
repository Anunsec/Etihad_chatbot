"""Orchestrate one authenticated turn and persist its messages, state, and audit."""

from __future__ import annotations

import os
import sqlite3
import time
from datetime import datetime
from uuid import uuid4

from langgraph.checkpoint.sqlite import SqliteSaver
from langsmith import traceable

from .audit import AuditLogger
from .cache import PolicyAnswerCache
from .chroma_retrieval import ChromaPolicyRepository
from .config import Settings
from .db import Database
from .flights import AviationStackProvider
from .graph import build_graph
from .retrieval import PolicyRepository
from .schemas import ChatRequest, ChatResponse, Citation, UserPrincipal


class AssistantService:
    """Own shared retrieval, cache, flight, graph, and checkpoint dependencies."""

    def __init__(self, settings: Settings, database: Database):
        self.settings = settings
        self.database = database
        self.audit = AuditLogger(settings.audit_file)
        self.rag_warning: str | None = None
        if settings.rag_backend == "chroma":
            # Search remains available through BM25 if vector setup fails.
            try:
                self.repository = ChromaPolicyRepository(settings)
            except Exception:
                self.repository = PolicyRepository(settings)
                self.rag_warning = "Chroma initialization failed; using local BM25 retrieval."
        else:
            self.repository = PolicyRepository(settings)
        self.cache = PolicyAnswerCache(settings)
        if settings.langsmith_tracing:
            os.environ["LANGSMITH_TRACING"] = "true"
            os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
            if settings.langsmith_api_key:
                os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
        provider = AviationStackProvider(settings)
        settings.checkpoint_db_file.parent.mkdir(parents=True, exist_ok=True)
        self._checkpoint_connection = sqlite3.connect(
            settings.checkpoint_db_file, check_same_thread=False, timeout=30
        )
        self.checkpointer = SqliteSaver(self._checkpoint_connection)
        self.graph = build_graph(
            self.repository, provider, settings, database, self.cache,
            checkpointer=self.checkpointer,
        )

    def close(self) -> None:
        """Release the SQLite connection held by LangGraph's checkpointer."""
        self._checkpoint_connection.close()

    @traceable(name="etihad_chat_turn", run_type="chain")
    def chat(self, request: ChatRequest, user: UserPrincipal) -> ChatResponse:
        """Resume the user's graph thread, run a turn, and save the grounded reply."""
        started = time.perf_counter()
        conversation_id, app_state = self.database.load_or_create_conversation(
            request.conversation_id, user.id
        )
        thread_id = f"{user.id}:{conversation_id}"
        # User-scoped thread IDs prevent two accounts from sharing graph memory.
        checkpoint_config = {"configurable": {"thread_id": thread_id}}
        saved_state = self.graph.get_state(checkpoint_config).values
        if isinstance(saved_state.get("app_state"), dict):
            app_state = saved_state["app_state"]
        turn_id = str(uuid4())
        trace_id = str(uuid4())
        recent = self.database.recent_messages(
            conversation_id, user.id, self.settings.chat_history_messages
        )
        self.database.add_message(conversation_id, "user", request.message)
        result = self.graph.invoke(
            {
                "message": request.message,
                "conversation_id": conversation_id,
                "user_id": user.id,
                "turn_id": turn_id,
                "trace_id": trace_id,
                "app_state": app_state,
                "conversation_history": recent,
            },
            config={
                "configurable": {"thread_id": thread_id},
                "run_name": "etihad-passenger-turn",
                "tags": ["local-mvp", user.role.value],
                "metadata": {
                    "conversation_id": conversation_id,
                    "turn_id": turn_id,
                    "user_role": user.role.value,
                },
            },
        )
        draft = result["draft"]
        updated_state = result.get("app_state", app_state)
        self.database.save_conversation_state(conversation_id, updated_state)

        citations: list[Citation] = []
        referenced = {citation_id for claim in draft.claims for citation_id in claim.citation_ids}
        # Show only sources that the final draft actually used.
        for chunk in result.get("retrieved", []):
            if chunk.id in referenced:
                citations.append(chunk.as_citation())
        if "aviationstack-live" in referenced:
            flight_result = result["flight_result"]
            citations.append(
                Citation(
                    id="aviationstack-live",
                    source_type="flight_tool",
                    document_title="AviationStack",
                    section_path="Normalized live flight-data response",
                    version_id="aviationstack-api-v1",
                    passage=str(flight_result),
                    retrieved_at=datetime.fromisoformat(flight_result["retrieved_at"]),
                    fixture=False,
                )
            )

        answer_id = str(uuid4())
        response = ChatResponse(
            **draft.model_dump(),
            answer_id=answer_id,
            conversation_id=conversation_id,
            citations=citations,
            state=updated_state,
            trace_id=trace_id,
        )
        self.database.add_message(
            conversation_id,
            "assistant",
            response.answer_markdown,
            response.model_dump(mode="json"),
            message_id=answer_id,
        )
        self.audit.write(
            "chat_turn",
            trace_id=trace_id,
            conversation_id=conversation_id,
            turn_id=turn_id,
            user_id=user.id,
            role=user.role.value,
            intent=result.get("intent"),
            decision=response.decision.value,
            citation_ids=[item.id for item in citations],
            retrieval=result.get("retrieval_details", {}),
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
        )
        return response
