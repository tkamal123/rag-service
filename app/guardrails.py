"""Input and output guardrails. Deliberately simple, readable and unit-tested."""
import re


class GuardrailViolation(Exception):
    """Raised when a request should be rejected before it reaches the LLM."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


_INJECTION_PATTERNS = [
    r"ignore (all |any |the )?(previous|prior|above|earlier) (instructions|prompts?|rules)",
    r"disregard (all |any |the )?(previous|prior|above|earlier|your) (instructions|prompts?|rules)",
    r"(reveal|show|print|repeat|leak) (me )?(your |the )?(system|hidden|initial) (prompt|instructions|message)",
    r"you are now (?!able)",
    r"\bdeveloper mode\b",
    r"\bjailbreak\b",
    r"act as (if you (have|had) )?no (rules|restrictions|guidelines)",
]
_INJECTION_RE = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE)

_PII_PATTERNS = {
    "email": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "card": re.compile(r"\b(?:\d[ -]?){13,16}\b"),
    "phone": re.compile(r"(?<!\w)\+?\d[\d\s().-]{8,}\d(?!\w)"),
}


def validate_input(question: str, max_chars: int = 1000) -> str:
    """Return the cleaned question or raise GuardrailViolation."""
    q = (question or "").strip()
    if not q:
        raise GuardrailViolation("Question is empty.")
    if len(q) > max_chars:
        raise GuardrailViolation(f"Question is too long (max {max_chars} characters).")
    if _INJECTION_RE.search(q):
        raise GuardrailViolation("Question looks like a prompt-injection attempt.")
    return q


def redact_pii(text: str) -> tuple[str, list[str]]:
    """Replace emails, card-like numbers and phone numbers. Returns (text, kinds_found)."""
    found: list[str] = []
    for kind, pattern in _PII_PATTERNS.items():
        if pattern.search(text):
            found.append(kind)
            text = pattern.sub(f"[REDACTED_{kind.upper()}]", text)
    return text, found
