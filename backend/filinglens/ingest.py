"""Download, parse, embed, and store company 10-K filings."""
import logging

import requests

from . import edgar, facts, parsing, retrieval, store

logger = logging.getLogger(__name__)


def _ingest_facts(ref: edgar.FilingRef) -> None:
    """Structured data is optional: a filing stays searchable if the SEC has none."""
    try:
        payload = edgar.company_facts(ref.cik)
    except requests.RequestException:
        logger.warning("No structured financial data available for %s", ref.ticker)
        return
    store.save_facts(ref.ticker, facts.extract_annual_facts(payload, ref.ticker, ref.company))


def ingest_ref(ref: edgar.FilingRef, force: bool = False) -> dict:
    if store.filing_exists(ref.filing_id) and not force:
        return {"id": ref.filing_id, "ticker": ref.ticker, "status": "already_indexed"}

    chunks = parsing.parse_filing(edgar.download(ref).read_bytes(), ref.form)
    if not chunks:
        raise ValueError(f"No text could be extracted from {ref.url}")

    records = [{"section": c.section, "kind": c.kind, "text": c.text} for c in chunks]
    embeddings = retrieval.embed_passages(
        [
            retrieval.contextual_text(ref.company, ref.form, ref.fiscal_year, c.section, c.text)
            for c in chunks
        ]
    )
    store.save_filing(
        {
            "id": ref.filing_id,
            "ticker": ref.ticker,
            "company": ref.company,
            "form": ref.form,
            "fiscal_year": ref.fiscal_year,
            "filing_date": ref.filing_date,
            "url": ref.url,
        },
        records,
        embeddings,
    )
    retrieval.clear_cache()
    return {"id": ref.filing_id, "ticker": ref.ticker, "status": "indexed", "chunks": len(chunks)}


def ingest_ticker(ticker: str, count: int = 1, force: bool = False) -> list[dict]:
    refs = edgar.latest_10k_refs(ticker, count=count)
    if force or not store.has_facts(refs[0].ticker):
        _ingest_facts(refs[0])
    return [ingest_ref(ref, force=force) for ref in refs]


def ingest_year(ticker: str, cik: int, fiscal_year: int) -> dict:
    ref = edgar.ten_k_for_year(ticker, cik, fiscal_year)
    if not store.has_facts(ticker):
        _ingest_facts(ref)
    return ingest_ref(ref)
