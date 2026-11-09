"""Select and objectively grade numeric 10-K questions from the FinanceBench open dataset."""
import json
import random
import re
from dataclasses import dataclass

from . import config

DATA_DIR = config.DATA_DIR / "financebench"
SELECTION_SEED = 20261109
RELATIVE_TOLERANCE = 0.01

# Tickers no longer in the SEC's current list, or renamed since these filings.
CIK_OVERRIDES = {"ATVI": 718877, "XYZ": 1512673}
TICKERS = {
    "3M": "MMM", "ACTIVISIONBLIZZARD": "ATVI", "ADOBE": "ADBE", "AES": "AES", "AMAZON": "AMZN",
    "AMCOR": "AMCR", "AMD": "AMD", "AMERICANWATERWORKS": "AWK", "BESTBUY": "BBY", "BLOCK": "XYZ",
    "BOEING": "BA", "COCACOLA": "KO", "CORNING": "GLW", "COSTCO": "COST", "CVSHEALTH": "CVS",
    "GENERALMILLS": "GIS", "KRAFTHEINZ": "KHC", "LOCKHEEDMARTIN": "LMT", "MGMRESORTS": "MGM",
    "MICROSOFT": "MSFT", "NETFLIX": "NFLX", "NIKE": "NKE", "PEPSICO": "PEP", "WALMART": "WMT",
}

NUMBER = re.compile(
    r"(?<![\w.])\$?\(?(\d[\d,]*(?:\.\d+)?)\)?\s?(billion|million|thousand|trillion|%|percent)?",
    re.IGNORECASE,
)
SCALES = {"thousand": 1e3, "million": 1e6, "billion": 1e9, "trillion": 1e12}
EQUIVALENT_FACTORS = (1.0, 100.0, 0.01, 1e-3, 1e-6, 1e-9)  # raw, ratio/percent, or reporting units


@dataclass
class BenchmarkQuestion:
    id: str
    doc_name: str
    ticker: str
    fiscal_year: int
    question: str
    gold: str
    gold_value: float


def gold_value(answer: str) -> float | None:
    """The magnitude of a short numeric answer such as $1577.00, 1.9%, or -0.02."""
    match = re.fullmatch(r"\s*-?\$?(\d[\d,]*(?:\.\d+)?)\s*%?\s*", answer)
    return float(match.group(1).replace(",", "")) if match else None


def select_questions(records: list[dict], documents: dict[str, dict], count: int) -> list[BenchmarkQuestion]:
    """A fixed random sample of 10-K questions with non-zero numeric answers."""
    eligible = []
    for record in sorted(records, key=lambda r: r["financebench_id"]):
        document = documents[record["doc_name"]]
        prefix = record["doc_name"].rsplit("_", 2)[0]
        value = gold_value(record["answer"])
        if document["doc_type"].lower() != "10k" or prefix not in TICKERS or not value:
            continue
        eligible.append(
            BenchmarkQuestion(
                id=record["financebench_id"],
                doc_name=record["doc_name"],
                ticker=TICKERS[prefix],
                fiscal_year=int(document["doc_period"]),
                question=record["question"],
                gold=record["answer"],
                gold_value=value,
            )
        )
    random.Random(SELECTION_SEED).shuffle(eligible)
    return eligible[:count]


def load_selection(count: int = 40) -> list[BenchmarkQuestion]:
    records = [json.loads(line) for line in open(DATA_DIR / "financebench_open_source.jsonl")]
    documents = {
        row["doc_name"]: row
        for row in map(json.loads, open(DATA_DIR / "financebench_document_information.jsonl"))
    }
    return select_questions(records, documents, count)


def answer_numbers(text: str) -> list[float]:
    """Numbers stated in an answer, skipping bare years such as 2019."""
    numbers = []
    for match in NUMBER.finditer(text):
        digits, unit = match.groups()
        value = float(digits.replace(",", ""))
        if not unit and "," not in digits and "." not in digits and 1900 <= value <= 2100:
            continue
        numbers.append(value * SCALES.get((unit or "").lower(), 1.0))
    return numbers


def matches(value: float, gold: float) -> bool:
    return any(abs(value * factor - gold) <= RELATIVE_TOLERANCE * gold for factor in EQUIVALENT_FACTORS)


def grade(answer: str, gold: float) -> dict[str, bool]:
    """Strict: the first number stated is right. Lenient: any number stated is right."""
    numbers = answer_numbers(answer)
    return {
        "strict": bool(numbers) and matches(numbers[0], gold),
        "lenient": any(matches(number, gold) for number in numbers),
    }
