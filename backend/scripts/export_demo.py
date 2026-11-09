"""Save real results as static files for the public demo website.

  python scripts/export_demo.py                  export data and record answers (needs Ollama)
  python scripts/export_demo.py --skip-answers   export data only

Every recorded question is kept, including answers the verifier flags.
"""
import argparse
import json
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

from filinglens import api, config, local_agent, store  # noqa: E402

OUTPUT = BACKEND.parent / "frontend" / "public" / "demo"
DEMO_TICKERS = ("AAPL", "MSFT", "RELIANCE")
DEMO_QUESTIONS = [
    ("How much did Apple's total revenue grow from fiscal 2025 to fiscal 2026, in percent?", ("AAPL",)),
    ("Which company had higher net income in fiscal 2026, Apple or Microsoft, and what was the gap in USD millions?", ("AAPL", "MSFT")),
    ("What was Microsoft's operating margin in fiscal 2026?", ("MSFT",)),
    ("What supply chain risks does Apple describe?", ("AAPL",)),
    ("How does Microsoft describe its relationship with OpenAI?", ("MSFT",)),
    ("What cybersecurity risks does Microsoft describe?", ("MSFT",)),
    ("What was Reliance's revenue in FY2026 and how much did it grow from FY2025?", ("RELIANCE",)),
]


def write(path: str, data) -> None:
    target = OUTPUT / f"{path}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-answers", action="store_true")
    args = parser.parse_args()

    # The benchmark also indexes older filings; the demo shows each company's latest.
    latest = {}
    for filing in sorted(store.list_filings(), key=lambda f: f["fiscal_year"]):
        latest[filing["ticker"]] = filing
    filings = [latest[ticker] for ticker in DEMO_TICKERS]
    write("filings", filings)
    write("health", {"status": "ok", "llm_configured": True, "model": config.active_model()})
    write("evaluation", api.evaluation_results())
    write("fx", api.exchange_rate())

    # Some companies have filings but no structured figures, so they have no dashboard.
    companies = [company for company in api.companies() if store.has_facts(company["ticker"])]
    write("companies", companies)
    for company in companies:
        write(f"companies/{company['ticker']}", api.company_overview(company["ticker"]))

    if not args.skip_answers:
        recorded = []
        for question, tickers in DEMO_QUESTIONS:
            print(question, flush=True)
            filing_ids = [latest[ticker]["id"] for ticker in tickers]
            recorded.append({"question": question, "events": list(local_agent.stream_local_agent(question, filing_ids))})
        write("answers", recorded)
    print("Exported to", OUTPUT)


if __name__ == "__main__":
    main()
