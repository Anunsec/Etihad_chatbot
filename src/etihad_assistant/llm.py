"""LangChain OpenAI invocation with local usage accounting and LangSmith metadata."""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from .config import Settings
from .db import Database


@dataclass(frozen=True)
class LLMInvocation:
    """Text plus whether a provider call was attempted and its failure type."""
    text: str | None
    attempted: bool
    error_type: str | None = None


def _message_text(message: Any) -> str:
    """Extract plain text from either string or structured model output."""
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and item.get("type") in {"text", "output_text"}
        ]
        return "\n".join(part for part in parts if part).strip()
    return str(content).strip()


class LoggedLangChainLLM:
    """LangChain OpenAI wrapper that stores normalized usage for the admin dashboard."""

    def __init__(self, settings: Settings, database: Database):
        self.settings = settings
        self.database = database

    def invoke(
        self,
        *,
        agent_name: str,
        system_prompt: str,
        user_prompt: str,
        user_id: str,
        conversation_id: str,
        turn_id: str,
        trace_id: str,
        prompt_version: str,
    ) -> LLMInvocation:
        """Call the model and persist usage, latency, and errors for every attempt."""
        if not self.settings.openai_api_key:
            return LLMInvocation(text=None, attempted=False, error_type="missing_openai_api_key")

        started = time.perf_counter()
        record: dict[str, Any] = {
            "id": str(uuid4()),
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "user_id": user_id,
            "conversation_id": conversation_id,
            "turn_id": turn_id,
            "agent_name": agent_name,
            "provider": "openai",
            "model": self.settings.openai_chat_model,
            "provider_request_id": None,
            "prompt_version": prompt_version,
            "input_tokens": None,
            "output_tokens": None,
            "cached_input_tokens": None,
            "reasoning_tokens": None,
            "status": "error",
            "error_type": None,
        }
        try:
            prompt = ChatPromptTemplate.from_messages(
                [("system", system_prompt), ("human", "{user_prompt}")]
            )
            model = ChatOpenAI(
                model=self.settings.openai_chat_model,
                api_key=self.settings.openai_api_key,
                temperature=0,
                timeout=self.settings.openai_timeout_seconds,
                max_retries=self.settings.openai_max_retries,
            )
            message = (prompt | model).invoke(
                {"user_prompt": user_prompt},
                config={
                    "run_name": agent_name,
                    "tags": ["openai", agent_name],
                    "metadata": {
                        "trace_id": trace_id,
                        "conversation_id": conversation_id,
                        "turn_id": turn_id,
                        "prompt_version": prompt_version,
                    },
                },
            )
            usage = getattr(message, "usage_metadata", None) or {}
            response_metadata = getattr(message, "response_metadata", None) or {}
            token_usage = response_metadata.get("token_usage", {})
            input_details = usage.get("input_token_details", {}) or {}
            output_details = usage.get("output_token_details", {}) or {}
            record.update(
                {
                    "provider_request_id": getattr(message, "id", None)
                    or response_metadata.get("id"),
                    "input_tokens": usage.get("input_tokens")
                    or token_usage.get("prompt_tokens")
                    or token_usage.get("input_tokens"),
                    "output_tokens": usage.get("output_tokens")
                    or token_usage.get("completion_tokens")
                    or token_usage.get("output_tokens"),
                    "cached_input_tokens": input_details.get("cache_read")
                    or input_details.get("cached_tokens"),
                    "reasoning_tokens": output_details.get("reasoning")
                    or output_details.get("reasoning_tokens"),
                    "status": "ok",
                }
            )
            text = _message_text(message)
            if not text:
                raise ValueError("OpenAI returned an empty response")
            return LLMInvocation(text=text, attempted=True)
        except Exception as exc:
            record["status"] = "error"
            record["error_type"] = type(exc).__name__
            return LLMInvocation(text=None, attempted=True, error_type=type(exc).__name__)
        finally:
            # This record is written even when model invocation or parsing fails.
            record["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
            self.database.log_llm_call(record)
