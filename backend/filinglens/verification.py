"""Check that the figures in an answer actually appear in the sources it cites."""
import re
from dataclasses import dataclass

from .answer import citable_units

SCALES = {
    "thousand": 1e3, "lakh": 1e5, "million": 1e6, "m": 1e6, "crore": 1e7, "cr": 1e7,
    "billion": 1e9, "b": 1e9, "trillion": 1e12,
}
PERCENT_UNITS = {"%", "percent"}
FILING_SCALES = (1.0, 1e6, 1e7)  # filings print amounts whole, in millions, or in crore

NUMBER = re.compile(
    r"(?<![\w.])([$₹])?(\d[\d,]*(?:\.\d+)?)"
    r"(?:\s?(billion|million|trillion|thousand|crore|lakh|percent|cr\b|B\b|M\b)|(%))?",
    re.IGNORECASE,
)
CITATION = re.compile(r"\[(?:Sources?\s*)?(\d+(?:\s*,\s*\d+)*)\]", re.IGNORECASE)
WORD = re.compile(r"[a-z][a-z'-]{2,}")
STOPWORDS = {"the", "and", "was", "were", "for", "that", "with", "from", "its", "has", "had", "fiscal"}


@dataclass(frozen=True)
class Claim:
    value: float
    tolerance: float  # half of the last digit shown, so rounded figures still match
    is_percent: bool


def _parse_number(match: re.Match) -> Claim | None:
    dollar, digits, word_unit, percent_sign = match.groups()
    unit = (word_unit or percent_sign or "").lower()
    value = float(digits.replace(",", ""))
    is_plain_year = not dollar and not unit and "," not in digits and "." not in digits and 1900 <= value <= 2100
    is_small_count = not dollar and not unit and "," not in digits and value < 100
    if is_plain_year or is_small_count:
        return None

    decimals = len(digits.split(".")[1]) if "." in digits else 0
    scale = SCALES.get(unit, 1.0)
    return Claim(value * scale, 0.5 * 10**-decimals * scale, unit in PERCENT_UNITS)


def extract_claims(text: str) -> list[Claim]:
    return [claim for claim in map(_parse_number, NUMBER.finditer(text)) if claim]


def source_values(text: str) -> set[float]:
    values: set[float] = set()
    for match in NUMBER.finditer(text):
        number = float(match.group(2).replace(",", ""))
        values.update(number * scale for scale in FILING_SCALES)
    return values


def is_supported(claim: Claim, values: set[float]) -> bool:
    return any(abs(value - claim.value) <= claim.tolerance * 1.01 for value in values)


def best_unit(sentence: str, passage: dict) -> str:
    """Pick the row or sentence of a passage that a claim most likely came from."""
    claims = extract_claims(sentence)
    words = set(WORD.findall(sentence.lower())) - STOPWORDS
    best, best_score = "", 0
    for unit in citable_units(passage):
        values = source_values(unit)
        score = 5 * sum(is_supported(claim, values) for claim in claims)
        score += len(words & set(WORD.findall(unit.lower())))
        if score > best_score:
            best, best_score = unit, score
    return best


def split_sentences(raw_answer: str) -> list[str]:
    # Move citations written after the full stop to before it.
    normalized = re.sub(
        r"([.!?])\s*((?:\[(?:Sources?\s*)?\d[\d,\s]*\]\s*)+)",
        lambda m: f" {m.group(2).strip()}{m.group(1)} ",
        raw_answer,
    )
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z$\d])|\n+", normalized) if s.strip()]


def calculation_values(results: list[float]) -> set[float]:
    """A calculated result may be stated as a percentage or in millions, billions, or trillions."""
    factors = (1.0, 100.0, *SCALES.values())
    return {result * factor for result in results for factor in factors}


def check_answer(raw_answer: str, passages: list[dict], calculations: list[float]) -> dict:
    calculated = calculation_values(calculations)
    claims = supported = uncited = 0
    unsupported: list[str] = []

    for sentence in split_sentences(raw_answer):
        cited = {
            int(number) - 1
            for match in CITATION.finditer(sentence)
            for number in match.group(1).split(",")
        }
        text = CITATION.sub("", sentence)
        allowed = set(calculated)
        for index in cited:
            if 0 <= index < len(passages):
                allowed |= source_values(passages[index]["text"])

        sentence_claims = extract_claims(text)
        missing = [claim for claim in sentence_claims if not is_supported(claim, allowed)]
        claims += len(sentence_claims)
        supported += len(sentence_claims) - len(missing)
        if missing:
            unsupported.append(text.strip())
            if not cited:
                uncited += 1

    return {
        "claims": claims,
        "supported": supported,
        "uncited_sentences": uncited,
        "unsupported_sentences": unsupported,
    }
