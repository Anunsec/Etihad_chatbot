"""SQLite storage for users, chat history, feedback, and model-call metrics."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any
from uuid import uuid4

from pwdlib import PasswordHash

from .config import Settings
from .cost import PRICING_AS_OF, PRICING_SOURCE, estimate_chat_cost_usd
from .schemas import Role, UserPrincipal

password_hash = PasswordHash.recommended()


class Database:
    """Small SQLite persistence layer for the local MVP."""

    def __init__(self, settings: Settings):
        self.path = settings.db_file
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.settings = settings

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        """Open a short-lived connection with row objects and foreign keys enabled."""
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def initialize(self) -> None:
        """Create local tables and seed the configured demo user and admin."""
        with self.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL CHECK (role IN ('user', 'admin')),
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL REFERENCES users(id),
                    state_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL REFERENCES conversations(id),
                    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    payload_json TEXT,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS feedback (
                    id TEXT PRIMARY KEY,
                    answer_id TEXT NOT NULL,
                    user_id TEXT NOT NULL REFERENCES users(id),
                    rating INTEGER NOT NULL CHECK (rating IN (-1, 1)),
                    reason_code TEXT,
                    free_text TEXT,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS llm_calls (
                    id TEXT PRIMARY KEY,
                    timestamp_utc TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    conversation_id TEXT NOT NULL,
                    turn_id TEXT NOT NULL,
                    agent_name TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    provider_request_id TEXT,
                    prompt_version TEXT NOT NULL,
                    input_tokens INTEGER,
                    output_tokens INTEGER,
                    cached_input_tokens INTEGER,
                    reasoning_tokens INTEGER,
                    latency_ms REAL NOT NULL,
                    status TEXT NOT NULL,
                    error_type TEXT
                );
                -- Policy answers are Redis-only; discard the former SQLite cache table.
                DROP TABLE IF EXISTS policy_answer_cache;
                """
            )
        self._seed_user(
            self.settings.demo_user_username,
            self.settings.demo_user_password,
            Role.user,
        )
        self._seed_user(
            self.settings.demo_admin_username,
            self.settings.demo_admin_password,
            Role.admin,
        )

    def _seed_user(self, username: str, password: str, role: Role) -> None:
        """Create a demo account once without overwriting an existing password."""
        with self.connect() as conn:
            exists = conn.execute("SELECT id FROM users WHERE username = ?", (username,)).fetchone()
            if exists:
                return
            conn.execute(
                "INSERT INTO users VALUES (?, ?, ?, ?, ?)",
                (
                    str(uuid4()),
                    username,
                    password_hash.hash(password),
                    role.value,
                    datetime.now(UTC).isoformat(),
                ),
            )

    def authenticate(self, username: str, password: str) -> UserPrincipal | None:
        """Verify a stored password hash and return the associated role."""
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        if not row or not password_hash.verify(password, row["password_hash"]):
            return None
        return UserPrincipal(id=row["id"], username=row["username"], role=Role(row["role"]))

    def get_user(self, user_id: str) -> UserPrincipal | None:
        """Reload the current account by token subject ID."""
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        if not row:
            return None
        return UserPrincipal(id=row["id"], username=row["username"], role=Role(row["role"]))

    def load_or_create_conversation(
        self, conversation_id: str | None, user_id: str
    ) -> tuple[str, dict[str, Any]]:
        """Continue only a conversation owned by this user, else create a new one."""
        now = datetime.now(UTC).isoformat()
        with self.connect() as conn:
            if conversation_id:
                row = conn.execute(
                    "SELECT state_json FROM conversations WHERE id = ? AND user_id = ?",
                    (conversation_id, user_id),
                ).fetchone()
                if row:
                    return conversation_id, json.loads(row["state_json"])
            new_id = str(uuid4())
            conn.execute(
                "INSERT INTO conversations VALUES (?, ?, '{}', ?, ?)",
                (new_id, user_id, now, now),
            )
        return new_id, {}

    def save_conversation_state(self, conversation_id: str, state: dict[str, Any]) -> None:
        """Persist extracted flight slots and update sidebar recency."""
        with self.connect() as conn:
            conn.execute(
                "UPDATE conversations SET state_json = ?, updated_at = ? WHERE id = ?",
                (json.dumps(state), datetime.now(UTC).isoformat(), conversation_id),
            )

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        payload: dict[str, Any] | None = None,
        message_id: str | None = None,
    ) -> str:
        """Append a message, optionally retaining its complete answer payload."""
        item_id = message_id or str(uuid4())
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO messages VALUES (?, ?, ?, ?, ?, ?)",
                (
                    item_id,
                    conversation_id,
                    role,
                    content,
                    json.dumps(payload) if payload else None,
                    datetime.now(UTC).isoformat(),
                ),
            )
        return item_id

    def recent_messages(
        self, conversation_id: str, user_id: str, limit: int = 8
    ) -> list[dict[str, str]]:
        """Return recent messages in chronological order for prompt context."""
        if limit <= 0:
            return []
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT m.role, m.content FROM messages AS m
                JOIN conversations AS c ON c.id = m.conversation_id
                WHERE c.id = ? AND c.user_id = ? AND m.role IN ('user', 'assistant')
                ORDER BY m.created_at DESC, m.rowid DESC LIMIT ?
                """,
                (conversation_id, user_id, limit),
            ).fetchall()
        return [dict(row) for row in reversed(rows)]

    def list_conversations(self, user_id: str, limit: int = 50) -> list[dict[str, Any]]:
        """Build this user's sidebar list using each chat's first question as title."""
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT c.id, c.created_at, c.updated_at,
                       COUNT(m.id) AS message_count,
                       (SELECT first.content FROM messages AS first
                        WHERE first.conversation_id = c.id AND first.role = 'user'
                        ORDER BY first.created_at, first.rowid LIMIT 1) AS first_question
                FROM conversations AS c
                JOIN messages AS m ON m.conversation_id = c.id
                WHERE c.user_id = ?
                GROUP BY c.id
                ORDER BY c.updated_at DESC, c.id DESC
                LIMIT ?
                """,
                (user_id, max(1, min(limit, 100))),
            ).fetchall()
        return [
            {
                "id": row["id"],
                "title": (row["first_question"] or "Conversation").strip()[:70],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "message_count": row["message_count"],
            }
            for row in rows
        ]

    def get_conversation_messages(
        self, conversation_id: str, user_id: str
    ) -> list[dict[str, Any]] | None:
        """Return messages only after verifying conversation ownership."""
        with self.connect() as conn:
            owned = conn.execute(
                "SELECT 1 FROM conversations WHERE id = ? AND user_id = ?",
                (conversation_id, user_id),
            ).fetchone()
            if not owned:
                return None
            rows = conn.execute(
                """
                SELECT m.id, m.role, m.content, m.payload_json
                FROM messages AS m
                WHERE m.conversation_id = ? AND m.role IN ('user', 'assistant')
                ORDER BY m.created_at, m.rowid
                """,
                (conversation_id,),
            ).fetchall()
        messages = []
        for row in rows:
            payload = json.loads(row["payload_json"]) if row["payload_json"] else {}
            messages.append(
                {
                    "role": row["role"],
                    "content": row["content"],
                    "decision": payload.get("decision"),
                    "citations": payload.get("citations", []),
                    "answer_id": payload.get("answer_id", row["id"])
                    if row["role"] == "assistant" else None,
                }
            )
        return messages

    def add_feedback(
        self,
        answer_id: str,
        user_id: str,
        rating: int,
        reason_code: str | None,
        free_text: str | None,
    ) -> str:
        """Save a helpful/not-helpful rating linked to one answer."""
        feedback_id = str(uuid4())
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO feedback VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    feedback_id,
                    answer_id,
                    user_id,
                    rating,
                    reason_code,
                    free_text,
                    datetime.now(UTC).isoformat(),
                ),
            )
        return feedback_id

    def answer_belongs_to_user(self, answer_id: str, user_id: str) -> bool:
        """Prevent a user from rating another account's answer."""
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT 1
                FROM messages m
                JOIN conversations c ON c.id = m.conversation_id
                WHERE m.id = ? AND m.role = 'assistant' AND c.user_id = ?
                """,
                (answer_id, user_id),
            ).fetchone()
        return row is not None

    def log_llm_call(self, record: dict[str, Any]) -> None:
        """Store normalized provider usage without saving prompts or responses."""
        columns = [
            "id",
            "timestamp_utc",
            "user_id",
            "conversation_id",
            "turn_id",
            "agent_name",
            "provider",
            "model",
            "provider_request_id",
            "prompt_version",
            "input_tokens",
            "output_tokens",
            "cached_input_tokens",
            "reasoning_tokens",
            "latency_ms",
            "status",
            "error_type",
        ]
        values = [record.get(column) for column in columns]
        with self.connect() as conn:
            conn.execute(
                f"INSERT INTO llm_calls ({', '.join(columns)}) VALUES ({', '.join('?' for _ in columns)})",
                values,
            )

    def list_llm_calls(self, limit: int = 100) -> list[dict[str, Any]]:
        """Return recent usage rows for the admin-only API."""
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM llm_calls ORDER BY timestamp_utc DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(row) for row in rows]

    def admin_dashboard(self, recent_limit: int = 100) -> dict[str, Any]:
        """Aggregate calls, tokens, outcomes, and estimated chat-model costs."""
        with self.connect() as conn:
            llm_totals = dict(
                conn.execute(
                    """
                    SELECT
                        COUNT(*) AS total_calls,
                        SUM(CASE WHEN status = 'ok' THEN 1 ELSE 0 END) AS successful_calls,
                        SUM(CASE WHEN status != 'ok' THEN 1 ELSE 0 END) AS failed_calls,
                        COALESCE(SUM(input_tokens), 0) AS input_tokens,
                        COALESCE(SUM(output_tokens), 0) AS output_tokens,
                        COALESCE(SUM(cached_input_tokens), 0) AS cached_input_tokens,
                        COALESCE(SUM(reasoning_tokens), 0) AS reasoning_tokens,
                        COALESCE(AVG(latency_ms), 0) AS average_latency_ms
                    FROM llm_calls
                    """
                ).fetchone()
            )
            by_model = [
                dict(row)
                for row in conn.execute(
                    """
                    SELECT model, COUNT(*) AS calls,
                           COALESCE(SUM(input_tokens), 0) AS input_tokens,
                           COALESCE(SUM(output_tokens), 0) AS output_tokens
                    FROM llm_calls GROUP BY model ORDER BY calls DESC
                    """
                ).fetchall()
            ]
            by_agent = [
                dict(row)
                for row in conn.execute(
                    """
                    SELECT agent_name, COUNT(*) AS calls,
                           COALESCE(SUM(input_tokens), 0) AS input_tokens,
                           COALESCE(SUM(output_tokens), 0) AS output_tokens,
                           COALESCE(AVG(latency_ms), 0) AS average_latency_ms
                    FROM llm_calls GROUP BY agent_name ORDER BY calls DESC
                    """
                ).fetchall()
            ]
            daily = [
                dict(row)
                for row in conn.execute(
                    """
                    SELECT substr(timestamp_utc, 1, 10) AS day,
                           COUNT(*) AS calls,
                           COALESCE(SUM(input_tokens + output_tokens), 0) AS total_tokens
                    FROM llm_calls GROUP BY day ORDER BY day DESC LIMIT 30
                    """
                ).fetchall()
            ][::-1]
            recent_calls = [
                dict(row)
                for row in conn.execute(
                    "SELECT * FROM llm_calls ORDER BY timestamp_utc DESC LIMIT ?",
                    (recent_limit,),
                ).fetchall()
            ]
            cost_rows = [
                dict(row)
                for row in conn.execute(
                    """
                    SELECT id, timestamp_utc, agent_name, provider, model,
                           input_tokens, output_tokens, cached_input_tokens
                    FROM llm_calls
                    """
                ).fetchall()
            ]
            app_counts = {
                "users": conn.execute("SELECT COUNT(*) FROM users").fetchone()[0],
                "conversations": conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0],
                "messages": conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0],
                "feedback_total": conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0],
                "feedback_positive": conn.execute(
                    "SELECT COUNT(*) FROM feedback WHERE rating = 1"
                ).fetchone()[0],
                "feedback_negative": conn.execute(
                    "SELECT COUNT(*) FROM feedback WHERE rating = -1"
                ).fetchone()[0],
            }
            payload_rows = conn.execute(
                "SELECT payload_json FROM messages WHERE role = 'assistant' AND payload_json IS NOT NULL"
            ).fetchall()

        decisions: dict[str, int] = {}
        for row in payload_rows:
            try:
                decision = json.loads(row["payload_json"]).get("decision", "unknown")
            except (json.JSONDecodeError, TypeError):
                decision = "unknown"
            decisions[decision] = decisions.get(decision, 0) + 1
        llm_totals["total_tokens"] = (
            llm_totals["input_tokens"] + llm_totals["output_tokens"]
        )
        estimated_total = Decimal("0")
        # Keep unpriced calls visible; unknown token usage must not look like $0.
        unpriced_reasons: dict[str, int] = {}
        model_costs: dict[str, Decimal] = {}
        agent_costs: dict[str, Decimal] = {}
        daily_costs: dict[str, Decimal] = {}
        model_unpriced: dict[str, int] = {}
        agent_unpriced: dict[str, int] = {}
        daily_unpriced: dict[str, int] = {}
        call_costs: dict[str, float | None] = {}
        priced_calls = 0
        for row in cost_rows:
            amount, reason = estimate_chat_cost_usd(row)
            day = row["timestamp_utc"][:10]
            if amount is None:
                unpriced_reasons[reason or "unknown"] = unpriced_reasons.get(
                    reason or "unknown", 0
                ) + 1
                model_unpriced[row["model"]] = model_unpriced.get(row["model"], 0) + 1
                agent_unpriced[row["agent_name"]] = agent_unpriced.get(row["agent_name"], 0) + 1
                daily_unpriced[day] = daily_unpriced.get(day, 0) + 1
                call_costs[row["id"]] = None
                continue
            call_costs[row["id"]] = float(amount)
            estimated_total += amount
            priced_calls += 1
            model_costs[row["model"]] = model_costs.get(row["model"], Decimal("0")) + amount
            agent_costs[row["agent_name"]] = (
                agent_costs.get(row["agent_name"], Decimal("0")) + amount
            )
            daily_costs[day] = daily_costs.get(day, Decimal("0")) + amount
        for row in by_model:
            amount = model_costs.get(row["model"])
            row["estimated_cost_usd"] = float(amount) if amount is not None else None
            row["unpriced_calls"] = model_unpriced.get(row["model"], 0)
        for row in by_agent:
            amount = agent_costs.get(row["agent_name"])
            row["estimated_cost_usd"] = float(amount) if amount is not None else None
            row["unpriced_calls"] = agent_unpriced.get(row["agent_name"], 0)
        for row in daily:
            amount = daily_costs.get(row["day"])
            row["estimated_cost_usd"] = float(amount) if amount is not None else None
            row["unpriced_calls"] = daily_unpriced.get(row["day"], 0)
        for row in recent_calls:
            row["estimated_cost_usd"] = call_costs[row["id"]]
        return {
            "llm": llm_totals,
            "cost": {
                "estimated_total_usd": float(estimated_total),
                "priced_calls": priced_calls,
                "unpriced_calls": sum(unpriced_reasons.values()),
                "unpriced_reasons": unpriced_reasons,
                "currency": "USD",
                "pricing_as_of": PRICING_AS_OF,
                "pricing_source": PRICING_SOURCE,
                "scope": "Recorded OpenAI chat-model calls only; excludes embeddings and other services.",
            },
            "by_model": by_model,
            "by_agent": by_agent,
            "daily": daily,
            "recent_calls": recent_calls,
            "application": app_counts,
            "decisions": [
                {"decision": decision, "count": count}
                for decision, count in sorted(decisions.items())
            ],
        }
