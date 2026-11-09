"""Index Indian companies: SEC 20-F filers, plus annual report PDFs saved in data/india.

Name each PDF <TICKER>_<FISCAL YEAR>.pdf and add its details to PDF_REPORTS.
"""
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

from filinglens import config, pdf_reports, retrieval, store  # noqa: E402
from filinglens.ingest import ingest_ticker  # noqa: E402

SEC_FILERS = ["INFY", "WIT", "HDB", "IBN", "RDY"]
PDF_REPORTS = [
    {
        "ticker": "RELIANCE",
        "company": "Reliance Industries Limited",
        "fiscal_year": 2026,
        "filed": "2026-05-29",
        "url": "https://www.ril.com/reports/RIL-Integrated-Annual-Report-2025-26.pdf",
    },
]


def ingest_pdf(report: dict) -> dict:
    filing_id = f"{report['ticker']}-{report['fiscal_year']}"
    path = config.DATA_DIR / "india" / f"{report['ticker']}_{report['fiscal_year']}.pdf"
    chunks, pages = pdf_reports.parse_report(path)
    records = [{"section": c.section, "kind": c.kind, "text": c.text} for c in chunks]
    embeddings = retrieval.embed_passages(
        [
            retrieval.contextual_text(report["company"], "Annual Report", report["fiscal_year"], c.section, c.text)
            for c in chunks
        ]
    )
    store.save_filing(
        {
            "id": filing_id,
            "ticker": report["ticker"],
            "company": report["company"],
            "form": "Annual Report",
            "fiscal_year": report["fiscal_year"],
            "filing_date": report["filed"],
            "url": report["url"],
        },
        records,
        embeddings,
    )
    facts = pdf_reports.extract_consolidated_facts(
        pages, report["ticker"], report["company"], report["fiscal_year"], report["filed"], filing_id
    )
    store.save_facts(report["ticker"], facts)
    retrieval.clear_cache()
    return {"id": filing_id, "chunks": len(chunks), "facts": len(facts)}


def main() -> None:
    for ticker in SEC_FILERS:
        try:
            print(ticker, ingest_ticker(ticker), flush=True)
        except Exception as error:  # one company's problem shouldn't stop the rest
            print(ticker, "FAILED", repr(error), flush=True)
    for report in PDF_REPORTS:
        print(report["ticker"], ingest_pdf(report), flush=True)


if __name__ == "__main__":
    main()
