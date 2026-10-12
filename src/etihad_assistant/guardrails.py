"""Deterministic scope and benefit checks around retrieved and generated answers."""

from __future__ import annotations

import re

OFF_TOPIC_REPLY = (
    "I can help with Etihad travel policies and flight information, but I can't "
    "help with that unrelated request."
)
POLICY_UNAVAILABLE_REPLY = (
    "I don't have enough information in the available Etihad policy sources to "
    "answer that question reliably. I can't confirm a policy or outcome without "
    "written support."
)
BENEFIT_UNAVAILABLE_REPLY = (
    "I can't confirm or promise a refund, fee waiver, or compensation for your "
    "booking from the available written policy. Eligibility and any amount must "
    "be confirmed by Etihad."
)

_TRAVEL_TERMS = {
    "airline", "airlines", "aircraft", "airport", "arrival", "arrivals",
    "baggage", "bag", "boarding", "booking", "cabin", "cancel", "cancellation",
    "checkin", "check-in", "chauffeur", "codeshare", "cooling",
    "compensation", "delay", "departure", "destination", "economy", "etihad",
    "dining", "fare", "flight", "flights", "fly", "flying", "food", "gate", "halal", "infant",
    "itinerary", "jetlag", "journey", "layover", "lounge", "luggage",
    "benefit", "benefits", "bronze", "emerald", "gold", "guest", "loyalty",
    "meal", "meals", "member", "membership", "miles", "no-show", "passport", "passenger", "pet",
    "platinum", "policy", "refund", "rewards", "schedule", "silver", "seat",
    "snack", "snacks", "terminal", "ticket", "tier", "travel", "travelling",
    "traveling", "trip", "upgrade", "visa", "waiver", "waive",
    "vegan", "vegetarian", "wheelchair", "auh", "del", "lhr", "ey",
}
# These deterministic checks bound scope and catch clear promises; they do not
# establish whether a policy claim is true, which still requires source evidence.
_OFF_TOPIC_REQUEST = re.compile(
    r"\b(?:write|generate|debug|explain)\s+(?:me\s+)?(?:a\s+)?(?:python|javascript|code|script|program)\b"
    r"|\b(?:recipe|joke|poem|homework|stock price|weather forecast)\b",
    re.I,
)
_FOLLOW_UP = re.compile(
    r"\s*(?:what about|how about|and\b|that\b|it\b|this\b|those\b|them\b|does that|is it)",
    re.I,
)
_GENERAL_ADVICE = re.compile(
    r"\b(?:tips|advice|prepare|comfortable|jet\s*lag|packing tips)\b", re.I
)
_SOURCE_DEPENDENT = re.compile(
    r"\b(?:etihad|policy|allowance|allowed|refund|waiv\w*|compensat\w*|"
    r"fee|charge|fare|booking|ticket|baggage|luggage|status|schedule|visa|"
    r"requirement|eligible|entitled|guarantee)\b",
    re.I,
)
_BENEFIT_REQUEST = re.compile(r"\b(?:refund|waiv\w*|compensat\w*)\b", re.I)
_UNSUPPORTED_PROMISE = re.compile(
    r"\b(?:you|your booking|we|i|etihad)\s+(?:will|shall|can|guarantee|promise)\s+"
    r"(?:receive|get|be given|be paid|be refunded|be compensated|refund|"
    r"waive|compensat\w*|approve|issue|pay|grant|provide)\b"
    r"|\byou(?:'ll|\s+(?:are|will))\s+(?:(?:be\s+)?eligible\s+for|"
    r"entitled\s+to|receive|get)\s+(?:a\s+)?(?:refund|waiver|compensation)\b"
    r"|\b(?:you|your booking)\s+(?:qualify|qualifies|is eligible)\s+for\s+"
    r"(?:a\s+)?(?:refund|waiver|compensation)\b"
    r"|\byour\s+(?:refund|waiver|compensation)\s+(?:is|has been)\s+"
    r"(?:approved|confirmed|guaranteed|issued)\b"
    r"|\byour\s+(?:fee|charge)\s+has been\s+waived\b"
    r"|\b(?:i|we|etihad)\s+(?:have|has|already)\s+"
    r"(?:approved|issued|waived|refunded|compensated)\b"
    r"|\b(?:refund|waiver|compensation|fee waiver)\s+"
    r"(?:is|are|will be)\s+(?:guaranteed|approved|issued|paid|automatic)\b"
    r"|\b(?:guaranteed|entitled to)\s+(?:a\s+)?(?:refund|waiver|compensation)\b",
    re.I,
)


def is_off_topic(message: str, has_travel_history: bool = False) -> bool:
    """Decline clear non-travel requests without asking the model to decide scope."""
    if _OFF_TOPIC_REQUEST.search(message):
        return True
    if has_travel_history and _FOLLOW_UP.match(message):
        return False
    words = set(re.findall(r"[a-z]+(?:-[a-z]+)?", message.casefold()))
    return not bool(words & _TRAVEL_TERMS)


def can_answer_from_general_knowledge(message: str) -> bool:
    """Allow only evergreen travel advice without claims about a specific policy."""
    return bool(_GENERAL_ADVICE.search(message)) and not bool(_SOURCE_DEPENDENT.search(message))


def asks_about_benefits(message: str) -> bool:
    """Identify refund/waiver/compensation questions needing extra caution."""
    return bool(_BENEFIT_REQUEST.search(message))


def contains_unsupported_promise(answer: str) -> bool:
    """Catch language that guarantees an individual benefit or approval."""
    return bool(_UNSUPPORTED_PROMISE.search(answer))
