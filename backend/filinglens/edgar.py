"""Download annual reports (10-K filings) from SEC EDGAR."""
import time
from dataclasses import dataclass
from pathlib import Path

import requests

from . import config
from .facts import ANNUAL_FORMS, fiscal_year_of

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
COMPANY_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{document}"

# SEC's fair-access limit is 10 requests per second; stay well below it.
REQUEST_PAUSE_SECONDS = 0.25


class FilingNotFound(Exception):
    pass


@dataclass
class FilingRef:
    ticker: str
    company: str
    cik: int
    form: str
    accession: str
    filing_date: str
    report_date: str
    url: str

    @property
    def filing_id(self) -> str:
        return self.accession

    @property
    def fiscal_year(self) -> int:
        return fiscal_year_of(self.report_date)


def _get(url: str) -> requests.Response:
    time.sleep(REQUEST_PAUSE_SECONDS)
    response = requests.get(
        url, headers={"User-Agent": config.SEC_USER_AGENT}, timeout=30
    )
    response.raise_for_status()
    return response


def lookup_cik(ticker: str) -> tuple[int, str]:
    wanted = ticker.upper()
    for entry in _get(TICKERS_URL).json().values():
        if entry["ticker"].upper() == wanted:
            return int(entry["cik_str"]), entry["title"]
    raise FilingNotFound(f"No SEC company found for ticker {wanted}")


def latest_10k_refs(ticker: str, count: int = 1) -> list[FilingRef]:
    cik, company = lookup_cik(ticker)
    recent = _get(SUBMISSIONS_URL.format(cik=cik)).json()["filings"]["recent"]

    refs: list[FilingRef] = []
    for i, form in enumerate(recent["form"]):
        if form not in ANNUAL_FORMS:
            continue
        accession = recent["accessionNumber"][i]
        document = recent["primaryDocument"][i]
        refs.append(
            FilingRef(
                ticker=ticker.upper(),
                company=company,
                cik=cik,
                form=form,
                accession=accession,
                filing_date=recent["filingDate"][i],
                report_date=recent["reportDate"][i],
                url=ARCHIVE_URL.format(
                    cik=cik, accession=accession.replace("-", ""), document=document
                ),
            )
        )
        if len(refs) == count:
            return refs

    if not refs:
        raise FilingNotFound(f"No annual filings found for {ticker.upper()}")
    return refs


def _filing_tables(cik: int) -> tuple[str, list[dict]]:
    """A company's recent filings table, plus the older ones the SEC stores separately."""
    submissions = _get(SUBMISSIONS_URL.format(cik=cik)).json()
    tables = [submissions["filings"]["recent"]]
    for older in submissions["filings"].get("files", []):
        tables.append(_get(f"https://data.sec.gov/submissions/{older['name']}").json())
    return submissions["name"], tables


def ten_k_for_year(ticker: str, cik: int, fiscal_year: int) -> FilingRef:
    company, tables = _filing_tables(cik)
    for table in tables:
        for i, form in enumerate(table["form"]):
            if form == "10-K" and table["reportDate"][i] and fiscal_year_of(table["reportDate"][i]) == fiscal_year:
                accession = table["accessionNumber"][i]
                return FilingRef(
                    ticker=ticker,
                    company=company,
                    cik=cik,
                    form=form,
                    accession=accession,
                    filing_date=table["filingDate"][i],
                    report_date=table["reportDate"][i],
                    url=ARCHIVE_URL.format(
                        cik=cik, accession=accession.replace("-", ""), document=table["primaryDocument"][i]
                    ),
                )
    raise FilingNotFound(f"No FY{fiscal_year} 10-K found for {ticker}")


def company_facts(cik: int) -> dict:
    return _get(COMPANY_FACTS_URL.format(cik=cik)).json()


def download(ref: FilingRef) -> Path:
    """Fetch the filing HTML once and reuse the saved copy afterwards."""
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = config.RAW_DIR / f"{ref.accession}.html"
    if not path.exists():
        path.write_bytes(_get(ref.url).content)
    return path
