"""Turn SEC XBRL company facts into one clean row per annual value."""
from datetime import date

ANNUAL_FORMS = ("10-K", "20-F")  # 20-F is the annual report foreign companies file with the SEC
ANNUAL_MIN_DAYS = 340
ANNUAL_MAX_DAYS = 380


def fiscal_year_of(period_end: str) -> int:
    """52/53-week years can end in the first days of January; count those as the prior year."""
    end = date.fromisoformat(period_end)
    return end.year - 1 if end.month == 1 and end.day <= 7 else end.year


def _is_annual(start: str | None, end: str) -> bool:
    if start is None:  # balance-sheet values are measured at a single date
        return True
    days = (date.fromisoformat(end) - date.fromisoformat(start)).days
    return ANNUAL_MIN_DAYS <= days <= ANNUAL_MAX_DAYS


def extract_annual_facts(payload: dict, ticker: str, company: str) -> list[dict]:
    """Keep 10-K values for full years, using the most recently filed figure if restated."""
    latest: dict[tuple, dict] = {}
    all_facts = payload.get("facts", {})
    concepts = {**all_facts.get("us-gaap", {}), **all_facts.get("ifrs-full", {})}
    for concept, data in concepts.items():
        for unit, entries in data.get("units", {}).items():
            for entry in entries:
                start = entry.get("start")
                if entry.get("form") not in ANNUAL_FORMS or not _is_annual(start, entry["end"]):
                    continue
                key = (concept, unit, start, entry["end"])
                if key not in latest or entry["filed"] > latest[key]["filed"]:
                    latest[key] = {
                        "ticker": ticker,
                        "company": company,
                        "concept": concept,
                        "label": data.get("label") or concept,
                        "unit": unit,
                        "value": entry["val"],
                        "period_start": start,
                        "period_end": entry["end"],
                        "fiscal_year": fiscal_year_of(entry["end"]),
                        "filed": entry["filed"],
                        "accession": entry["accn"],
                    }
    return list(latest.values())
