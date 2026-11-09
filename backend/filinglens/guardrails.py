"""Checks applied before a question reaches the model, plus a daily usage limit."""
import re
import threading
from collections import defaultdict
from datetime import date

ADVICE_MESSAGE = (
    "FilingLens describes what company filings say and can't give investment advice. "
    "Try asking about a company's results, risks, or strategy instead."
)
INJECTION_MESSAGE = "That looks like an attempt to change FilingLens's instructions, so it wasn't processed."

ADVICE = re.compile(
    r"\b(should\s+i\s+(buy|sell|hold|invest|short)|is\s+\S+(\s+\S+)?\s+a\s+(good|bad)\s+(buy|investment|stock)|"
    r"(buy|sell)\s+or\s+(hold|sell|buy)|price\s+target|what\s+(stock|stocks|shares)\s+should|"
    r"will\s+(the\s+)?(stock|share\s+price)\s+(go|rise|fall))\b",
    re.IGNORECASE,
)
INJECTION = re.compile(
    r"(ignore|disregard|forget|override)\s+(all\s+|any\s+|the\s+|your\s+|previous\s+|prior\s+|above\s+)*"
    r"(instructions?|prompts?|rules?)|"
    r"(reveal|show|print|repeat)\s+(me\s+)?(your\s+|the\s+)(system\s+prompt|instructions?)",
    re.IGNORECASE,
)


def blocked_reason(question: str) -> str | None:
    if INJECTION.search(question):
        return INJECTION_MESSAGE
    if ADVICE.search(question):
        return ADVICE_MESSAGE
    return None


class DailyLimiter:
    """Caps questions per visitor per day on a public demo. Zero turns the limit off."""

    def __init__(self, limit: int):
        self.limit = limit
        self._lock = threading.Lock()
        self._day = date.today()
        self._counts: dict[str, int] = defaultdict(int)

    def allow(self, visitor: str, today: date | None = None) -> bool:
        if self.limit <= 0:
            return True
        today = today or date.today()
        with self._lock:
            if today != self._day:
                self._day = today
                self._counts.clear()
            if self._counts[visitor] >= self.limit:
                return False
            self._counts[visitor] += 1
            return True
