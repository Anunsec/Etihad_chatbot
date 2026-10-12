"""Redis-only cache for reusable, source-backed policy answers."""

from __future__ import annotations

import json
import time
from typing import Any
from urllib.parse import urlsplit

from .config import Settings


class PolicyAnswerCache:
    """Shared Redis-only policy cache; never expose connection secrets."""

    PREFIX = "etihad:policy-answer:v1:"
    HITS_KEY = "etihad:policy-answer:v1-hits"
    RETRY_SECONDS = 30

    @staticmethod
    def _connection_warning(error: Exception) -> str:
        error_type = type(error).__name__
        if error_type == "AuthenticationError":
            return "Redis authentication failed. Policy answers will not be cached."
        if error_type == "TimeoutError":
            return "Redis connection timed out. Policy answers will not be cached."
        return "Redis unavailable. Policy answers will not be cached."

    def __init__(self, settings: Settings, redis_client: Any = None):
        self.settings = settings
        self.client = redis_client
        self._redis_candidate = redis_client
        self._next_probe_at = 0.0
        self.backend = "unavailable"
        self.warning: str | None = None
        if redis_client is not None:
            self.backend = "redis"
        elif not settings.redis_url:
            self.warning = "Redis is not configured. Policy answers will not be cached."
        else:
            parsed = urlsplit(settings.redis_url)
            local = parsed.hostname in {"localhost", "127.0.0.1", "::1"}
            if parsed.scheme not in {"redis", "rediss"}:
                self.warning = "Redis URL must use redis:// or rediss://. Policy answers will not be cached."
            elif settings.redis_require_tls and parsed.scheme != "rediss" and not local:
                self.warning = "Remote Redis requires TLS (rediss://). Policy answers will not be cached."
            else:
                try:
                    import redis

                    self._redis_candidate = redis.Redis.from_url(
                        settings.redis_url,
                        decode_responses=True,
                        socket_connect_timeout=3,
                        socket_timeout=3,
                    )
                    self._redis_candidate.ping()
                    self.client = self._redis_candidate
                    self.backend = "redis"
                except Exception as exc:
                    self.client = None
                    self.warning = self._connection_warning(exc)
                    self._next_probe_at = time.monotonic() + self.RETRY_SECONDS

    def _ensure_redis(self) -> None:
        """Retry a failed connection after a cooldown instead of probing every turn."""
        if self.client is not None or self._redis_candidate is None:
            return
        if time.monotonic() < self._next_probe_at:
            return
        self._next_probe_at = time.monotonic() + self.RETRY_SECONDS
        try:
            self._redis_candidate.ping()
        except Exception as exc:
            self.warning = self._connection_warning(exc)
        else:
            self.client = self._redis_candidate
            self.backend = "redis"
            self.warning = None

    def _degrade(self, warning: str) -> None:
        """Stop using Redis after a request error until the next retry window."""
        self.client = None
        self.backend = "unavailable"
        self.warning = warning
        self._next_probe_at = time.monotonic() + self.RETRY_SECONDS

    def get(self, key: str) -> tuple[dict[str, Any] | None, str]:
        """Return a validated cached draft and its backend, or a cache miss."""
        self._ensure_redis()
        if self.client is not None:
            try:
                raw = self.client.get(self.PREFIX + key)
                if raw:
                    value = json.loads(raw)
                    if isinstance(value, dict) and {"draft", "source_ids"} <= value.keys():
                        self.client.incr(self.HITS_KEY)
                        return value, "redis"
                return None, "redis"
            except Exception:
                self._degrade("Redis request failed. Policy answers will not be cached.")
        return None, "unavailable"

    def put(self, key: str, draft: dict[str, Any], source_ids: list[str]) -> str:
        """Store the answer with source IDs and a configured expiry."""
        self._ensure_redis()
        if self.client is not None:
            try:
                self.client.set(
                    self.PREFIX + key,
                    json.dumps({"draft": draft, "source_ids": source_ids}, ensure_ascii=False),
                    ex=self.settings.redis_cache_ttl_seconds,
                )
                return "redis"
            except Exception:
                self._degrade("Redis request failed. Policy answers will not be cached.")
        return "unavailable"

    def stats(self) -> dict[str, Any]:
        """Expose cache health and counters without disclosing the Redis URL."""
        self._ensure_redis()
        if self.client is not None:
            try:
                entries = sum(1 for _ in self.client.scan_iter(match=self.PREFIX + "*"))
                return {
                    "backend": "redis",
                    "entries": entries,
                    "hits": int(self.client.get(self.HITS_KEY) or 0),
                    "ttl_seconds": self.settings.redis_cache_ttl_seconds,
                    "warning": self.warning,
                }
            except Exception:
                self._degrade("Redis metrics unavailable. Policy answers will not be cached.")
        return {
            "backend": "unavailable",
            "entries": 0,
            "hits": 0,
            "warning": self.warning,
        }
