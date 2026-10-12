"""Estimated OpenAI chat-model spend from the usage recorded by this app."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

PRICING_AS_OF = "2026-10-11"
PRICING_SOURCE = "https://developers.openai.com/api/docs/pricing"

# Standard text-token USD rates per million tokens, not batch or priority pricing.
# Snapshot aliases are included only where their model page lists them.
MODEL_RATES_USD_PER_MILLION = {
    "gpt-4o-mini": ("0.15", "0.075", "0.60"),
    "gpt-4o-mini-2024-07-18": ("0.15", "0.075", "0.60"),
    "gpt-4o": ("2.50", "1.25", "10.00"),
    "gpt-4o-2024-08-06": ("2.50", "1.25", "10.00"),
    "gpt-4o-2024-11-20": ("2.50", "1.25", "10.00"),
}


def estimate_chat_cost_usd(call: dict[str, Any]) -> tuple[Decimal | None, str | None]:
    """Return estimated cost and, when unavailable, the reason it cannot be priced."""
    rates = MODEL_RATES_USD_PER_MILLION.get(call.get("model"))
    if call.get("provider") != "openai" or rates is None:
        return None, "unsupported_model"

    input_tokens = call.get("input_tokens")
    output_tokens = call.get("output_tokens")
    cached_tokens = call.get("cached_input_tokens") or 0
    if input_tokens is None or output_tokens is None:
        return None, "missing_usage"
    if (
        input_tokens < 0
        or output_tokens < 0
        or cached_tokens < 0
        or cached_tokens > input_tokens
    ):
        return None, "invalid_usage"

    input_rate, cached_rate, output_rate = map(Decimal, rates)
    # Cached input tokens are a subset of input tokens, not an extra charge.
    cost = (
        Decimal(input_tokens - cached_tokens) * input_rate
        + Decimal(cached_tokens) * cached_rate
        + Decimal(output_tokens) * output_rate
    ) / Decimal(1_000_000)
    return cost, None
