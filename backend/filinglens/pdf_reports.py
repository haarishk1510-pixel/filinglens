"""Read annual report PDFs, keeping page numbers and rebuilding financial table rows.

Indian annual reports are not available as structured SEC data, so key figures are read
from the consolidated statements. Amounts are reported in ₹ crore.
"""
import re
from dataclasses import dataclass

import pymupdf

from .parsing import MAX_CHUNK_WORDS, OVERLAP_WORDS, Chunk

CRORE = 10_000_000
NUMBER_LINE = re.compile(r"^\(?-?\d[\d,]*(?:\.\d+)?\)?$")
NOTE_REFERENCE_MAX = 99
TABLE_NUMBER_SHARE = 0.3
MIN_LINES_FOR_TABLE = 12
HEADER_LINES = 3


@dataclass
class ReportFact:
    concept: str
    label: str
    pattern: str
    is_flow: bool


CONSOLIDATED_FACTS = [
    ReportFact("Revenue", "Revenue from operations", r"^Revenue from Operations", True),
    ReportFact("ProfitLoss", "Profit for the year", r"^Profit for the Year", True),
    ReportFact("Assets", "Total assets", r"^Total Assets", False),
    ReportFact(
        "CashFlowsFromUsedInOperatingActivities",
        "Net cash from operating activities",
        r"^Net Cash Flows? (?:From|Generated From|from) Operating Activities",
        True,
    ),
]


def _number(text: str) -> float:
    negative = text.startswith("(") or text.startswith("-")
    value = float(text.strip("()-").replace(",", ""))
    return -value if negative else value


def rebuild_rows(lines: list[str]) -> list[tuple[str, list[str]]]:
    """Pair each label with the numbers that follow it, dropping small note references."""
    rows: list[tuple[str, list[str]]] = []
    label, numbers = None, []
    for line in lines:
        if NUMBER_LINE.match(line):
            if label is not None:
                numbers.append(line)
            continue
        if label is not None:
            rows.append((label, numbers))
        label, numbers = line, []
    if label is not None:
        rows.append((label, numbers))

    cleaned = []
    for row_label, values in rows:
        is_note = len(values) >= 3 and values[0].isdigit() and int(values[0]) <= NOTE_REFERENCE_MAX
        cleaned.append((row_label, values[1:] if is_note else values))
    return cleaned


def clean_lines(page_text: str) -> list[str]:
    # The report's rupee symbol is extracted as the letter I.
    text = page_text.replace("(I in crore)", "(₹ in crore)")
    return [line.strip() for line in text.split("\n") if line.strip()]


def is_table_page(lines: list[str]) -> bool:
    numeric = sum(bool(NUMBER_LINE.match(line)) for line in lines)
    return len(lines) >= MIN_LINES_FOR_TABLE and numeric / len(lines) >= TABLE_NUMBER_SHARE


def page_chunks(lines: list[str], page_number: int) -> list[Chunk]:
    section = f"Page {page_number}"
    if is_table_page(lines):
        title = " ".join(lines[:HEADER_LINES])
        rows = [
            f"{label} | {' | '.join(values)}" if values else label
            for label, values in rebuild_rows(lines[HEADER_LINES:])
        ]
        return [Chunk(section, "table", "\n".join([f"Table: {title}", *rows]))]

    words = " ".join(lines).split()
    step = MAX_CHUNK_WORDS - OVERLAP_WORDS
    return [
        Chunk(section, "text", " ".join(words[start : start + MAX_CHUNK_WORDS]))
        for start in range(0, len(words), step)
        if len(words[start : start + MAX_CHUNK_WORDS]) > 20
    ]


def parse_report(path) -> tuple[list[Chunk], list[list[str]]]:
    document = pymupdf.open(path)
    chunks: list[Chunk] = []
    pages: list[list[str]] = []
    for number, page in enumerate(document, start=1):
        lines = clean_lines(page.get_text())
        pages.append(lines)
        chunks.extend(page_chunks(lines, number))
    return chunks, pages


def extract_consolidated_facts(
    pages: list[list[str]], ticker: str, company: str, fiscal_year: int, filed: str, accession: str
) -> list[dict]:
    """Read figures from pages marked consolidated, using each table's two year columns."""
    found: dict[tuple[str, int], dict] = {}
    for lines in pages:
        if not any("Consolidated" in line for line in lines[:HEADER_LINES + 2]) or not is_table_page(lines):
            continue
        for row_label, values in rebuild_rows(lines):
            for fact in CONSOLIDATED_FACTS:
                if not re.match(fact.pattern, row_label, re.IGNORECASE) or len(values) < 2:
                    continue
                for offset, text in enumerate(values[:2]):
                    year = fiscal_year - offset
                    found.setdefault(
                        (fact.concept, year),
                        {
                            "ticker": ticker,
                            "company": company,
                            "concept": fact.concept,
                            "label": fact.label,
                            "unit": "INR",
                            "value": _number(text) * CRORE,
                            "period_start": f"{year - 1}-04-01" if fact.is_flow else None,
                            "period_end": f"{year}-03-31",
                            "fiscal_year": year,
                            "filed": filed,
                            "accession": accession,
                        },
                    )
    return list(found.values())
