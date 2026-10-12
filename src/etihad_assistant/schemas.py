"""Shared API contracts for users, conversations, cited answers, and tools."""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field


class Role(StrEnum):
    """Roles accepted by the FastAPI authorization dependencies."""
    user = "user"
    admin = "admin"


class Decision(StrEnum):
    """Outcome categories surfaced to the UI and admin dashboard."""
    answer = "answer"
    clarify = "clarify"
    not_found = "not_found"
    off_topic = "off_topic"
    tool_unavailable = "tool_unavailable"
    conflicting_policy = "conflicting_policy"


class UserPrincipal(BaseModel):
    """Authenticated account identity read from the application database."""
    id: str
    username: str
    role: Role


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: Role


class ChatRequest(BaseModel):
    """A single user turn, optionally continuing a saved conversation."""
    message: str = Field(min_length=1, max_length=4000)
    conversation_id: str | None = None


class Citation(BaseModel):
    """Policy or live-tool evidence attached to a generated claim."""
    id: str
    source_type: Literal["policy", "flight_tool"]
    document_title: str
    section_path: str
    version_id: str
    effective_date_status: str = "not_stated"
    snapshot_page: int | None = None
    passage: str
    table_cell_ids: list[str] = Field(default_factory=list)
    canonical_url: str | None = None
    retrieved_at: datetime
    fixture: bool = False


class Claim(BaseModel):
    claim: str
    citation_ids: list[str]


class GroundedDraft(BaseModel):
    """Graph result before conversation IDs and source objects are attached."""
    decision: Decision
    answer_markdown: str
    claims: list[Claim] = Field(default_factory=list)
    clarification_question: str | None = None
    refusal_reason: str | None = None


class ChatResponse(GroundedDraft):
    """Public reply with trace, conversation state, and supporting citations."""
    answer_id: str
    conversation_id: str
    citations: list[Citation] = Field(default_factory=list)
    state: dict[str, Any] = Field(default_factory=dict)
    trace_id: str


class ConversationSummary(BaseModel):
    id: str
    title: str
    created_at: datetime
    updated_at: datetime
    message_count: int


class ConversationMessage(BaseModel):
    """Saved message shape used when reopening a conversation in Streamlit."""
    role: Literal["user", "assistant"]
    content: str
    decision: Decision | None = None
    citations: list[Citation] = Field(default_factory=list)
    answer_id: str | None = None


class ConversationDetail(BaseModel):
    conversation_id: str
    messages: list[ConversationMessage]


class FeedbackRequest(BaseModel):
    answer_id: str
    rating: Literal[-1, 1]
    reason_code: str | None = Field(default=None, max_length=80)
    free_text: str | None = Field(default=None, max_length=1000)


class FeedbackResponse(BaseModel):
    stored: bool
    feedback_id: str


class FlightStatusRequest(BaseModel):
    flight_number: str = Field(pattern=r"^[A-Za-z]{2}\s?\d{1,4}$")
    departure_date: date
    origin_iata: str | None = Field(default=None, pattern=r"^[A-Za-z]{3}$")


class FlightScheduleRequest(BaseModel):
    origin_iata: str = Field(pattern=r"^[A-Za-z]{3}$")
    destination_iata: str = Field(pattern=r"^[A-Za-z]{3}$")
    date: date


class RouteRequest(BaseModel):
    origin_iata: str | None = Field(default=None, pattern=r"^[A-Za-z]{3}$")
    destination_iata: str | None = Field(default=None, pattern=r"^[A-Za-z]{3}$")
    airline_iata: str = Field(default="EY", pattern=r"^[A-Za-z0-9]{2,3}$")
