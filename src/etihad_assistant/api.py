"""FastAPI routes, authentication gates, and application lifecycle."""

from __future__ import annotations

from contextlib import asynccontextmanager

import uvicorn
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm

from .auth import create_access_token, get_current_user, require_roles
from .config import get_settings
from .db import Database
from .schemas import (
    ChatRequest,
    ChatResponse,
    ConversationDetail,
    ConversationSummary,
    FeedbackRequest,
    FeedbackResponse,
    Role,
    TokenResponse,
    UserPrincipal,
)
from .service import AssistantService

settings = get_settings()
database = Database(settings)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Open shared app services on startup and close the checkpoint DB on shutdown."""
    database.initialize()
    service = AssistantService(settings, database)
    app.state.service = service
    try:
        yield
    finally:
        service.close()


app = FastAPI(
    title="Etihad Passenger Travel Assistant — Local MVP",
    version="0.1.0",
    description=(
        "Conditional local RAG plus LangChain/OpenAI answers and AviationStack MCP tools."
    ),
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)


@app.get("/health")
def health() -> dict[str, str | bool]:
    """Report configured capabilities without exposing API keys or passwords."""
    return {
        "status": "ok",
        "policy_data": "etihad_hub_snapshot",
        "policy_answer_mode": "rag_openai_refinement_with_shared_cache",
        "fallback_answer_mode": "langchain_openai",
        "openai_configured": bool(settings.openai_api_key),
        "langsmith_tracing": settings.langsmith_tracing,
        "flight_provider": "aviationstack",
        "flight_transport": "mcp",
        "live_flight_data": settings.enable_live_flight_data,
    }


@app.post("/auth/token", response_model=TokenResponse)
def login(form: OAuth2PasswordRequestForm = Depends()) -> TokenResponse:
    """Exchange local credentials for a role-bearing bearer token."""
    user = database.authenticate(form.username, form.password)
    if not user:
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    return TokenResponse(access_token=create_access_token(user, settings), role=user.role)


@app.get("/me", response_model=UserPrincipal)
def me(user: UserPrincipal = Depends(get_current_user)) -> UserPrincipal:
    """Return the identity resolved from the caller's bearer token."""
    return user


@app.post("/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    user: UserPrincipal = Depends(require_roles(Role.user, Role.admin)),
) -> ChatResponse:
    """Run one graph turn for an authenticated user or administrator."""
    return app.state.service.chat(request, user)


@app.get("/conversations", response_model=list[ConversationSummary])
def list_conversations(
    limit: int = 50,
    user: UserPrincipal = Depends(require_roles(Role.user, Role.admin)),
) -> list[ConversationSummary]:
    """List only conversations owned by the signed-in user."""
    return [
        ConversationSummary.model_validate(item)
        for item in database.list_conversations(user.id, limit)
    ]


@app.get("/conversations/{conversation_id}", response_model=ConversationDetail)
def get_conversation(
    conversation_id: str,
    user: UserPrincipal = Depends(require_roles(Role.user, Role.admin)),
) -> ConversationDetail:
    """Load a saved conversation after the database checks ownership."""
    messages = database.get_conversation_messages(conversation_id, user.id)
    if messages is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return ConversationDetail(conversation_id=conversation_id, messages=messages)


@app.post("/feedback", response_model=FeedbackResponse)
def feedback(
    request: FeedbackRequest,
    user: UserPrincipal = Depends(require_roles(Role.user, Role.admin)),
) -> FeedbackResponse:
    """Record feedback only for the caller's answer, unless the caller is admin."""
    if not database.answer_belongs_to_user(request.answer_id, user.id) and user.role != Role.admin:
        raise HTTPException(status_code=404, detail="Answer not found")
    feedback_id = database.add_feedback(
        request.answer_id, user.id, request.rating, request.reason_code, request.free_text
    )
    app.state.service.audit.write(
        "feedback",
        feedback_id=feedback_id,
        answer_id=request.answer_id,
        user_id=user.id,
        rating=request.rating,
        reason_code=request.reason_code,
    )
    return FeedbackResponse(stored=True, feedback_id=feedback_id)


@app.get("/admin/audit-logs")
def audit_logs(
    limit: int = 100,
    _: UserPrincipal = Depends(require_roles(Role.admin)),
):
    """Return a bounded tail of local audit events to admins."""
    return app.state.service.audit.tail(max(1, min(limit, 1000)))


@app.get("/admin/llm-calls")
def llm_calls(
    limit: int = 100,
    _: UserPrincipal = Depends(require_roles(Role.admin)),
):
    """Return recent persisted model-call records to admins."""
    return database.list_llm_calls(max(1, min(limit, 1000)))


@app.get("/admin/dashboard")
def admin_dashboard(
    limit: int = 100,
    _: UserPrincipal = Depends(require_roles(Role.admin)),
):
    """Combine database usage metrics with live cache and retrieval diagnostics."""
    payload = database.admin_dashboard(max(1, min(limit, 1000)))
    service = app.state.service
    cache_stats = service.cache.stats()
    payload["application"]["policy_cache_entries"] = cache_stats["entries"]
    payload["application"]["policy_cache_hits"] = cache_stats["hits"]
    payload["application"]["policy_cache_backend"] = cache_stats["backend"]
    payload["application"]["rag_backend"] = (
        "chroma_hybrid" if service.repository.__class__.__name__ == "ChromaPolicyRepository"
        else "local_bm25"
    )
    payload["application"]["rag_indexed_new_chunks"] = getattr(
        service.repository, "indexed_count", 0
    )
    payload["application"]["rag_embedding_calls"] = getattr(
        service.repository, "embedding_calls", 0
    )
    payload["application"]["storage_warnings"] = [
        message for message in (cache_stats.get("warning"), service.rag_warning) if message
    ]
    payload["configuration"] = {
        "openai_configured": bool(settings.openai_api_key),
        "openai_model": settings.openai_chat_model,
        "langsmith_tracing": settings.langsmith_tracing,
        "langsmith_project": settings.langsmith_project,
        "aviationstack_configured": bool(settings.aviationstack_api_key),
    }
    return payload


def run() -> None:
    """Start the development API server with automatic source reload."""
    uvicorn.run("etihad_assistant.api:app", host="127.0.0.1", port=8000, reload=True)


if __name__ == "__main__":
    run()
