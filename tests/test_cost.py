"""Cost-estimation tests for cached tokens and unpriceable calls."""

from __future__ import annotations

from decimal import Decimal

from etihad_assistant.cost import estimate_chat_cost_usd


def test_chat_cost_subtracts_cached_tokens_from_regular_input() -> None:
    amount, reason = estimate_chat_cost_usd(
        {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "input_tokens": 20,
            "cached_input_tokens": 3,
            "output_tokens": 5,
        }
    )
    assert reason is None
    assert amount == Decimal("0.000005775")


def test_chat_cost_does_not_guess_unknown_prices_or_usage() -> None:
    base = {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "input_tokens": 20,
        "output_tokens": 5,
    }
    assert estimate_chat_cost_usd({**base, "model": "unknown-model"}) == (
        None, "unsupported_model"
    )
    assert estimate_chat_cost_usd({**base, "output_tokens": None}) == (
        None, "missing_usage"
    )
    assert estimate_chat_cost_usd({**base, "cached_input_tokens": 21}) == (
        None, "invalid_usage"
    )

