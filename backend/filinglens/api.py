"""HTTP API for the FilingLens website."""
import json
import logging
import time
from collections.abc import Iterator
from datetime import date

import anthropic
import openai
import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from . import agent, answer, config, edgar, guardrails, local_agent, retrieval, store
from .ingest import ingest_ticker

logger = logging.getLogger(__name__)
limiter = guardrails.DailyLimiter(config.DAILY_QUESTION_LIMIT)
app = FastAPI(title="FilingLens API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

PUBLIC_PASSAGE_FIELDS = (
    "id", "ticker", "company", "form", "fiscal_year", "section", "kind", "text", "url",
)


class IngestRequest(BaseModel):
    ticker: str = Field(pattern=r"^[A-Za-z][A-Za-z.\-]{0,9}$")


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    filing_ids: list[str] = Field(min_length=1, max_length=6)
    mode: retrieval.Mode = "hybrid_rerank"
    top_k: int = Field(default=8, ge=1, le=20)
    # Off gives single-pass RAG, the baseline for measuring what the agent adds.
    use_agent: bool = True


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "llm_configured": config.llm_configured(), "model": config.active_model()}


@app.get("/api/filings")
def filings() -> list[dict]:
    return store.list_filings()


@app.post("/api/filings")
def add_filing(body: IngestRequest) -> list[dict]:
    try:
        return ingest_ticker(body.ticker)
    except edgar.FilingNotFound as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except requests.RequestException as error:
        raise HTTPException(status_code=502, detail="Could not download the filing from the SEC.") from error


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event)}\n\n"


def _llm_events(events: Iterator[dict]) -> Iterator[str]:
    try:
        for event in events:
            yield _sse(event)
    except openai.APIConnectionError:
        yield _sse({"type": "error", "message": "Could not reach the local model. Check that Ollama is running."})
    except openai.APIStatusError as error:
        yield _sse({"type": "error", "message": f"The model server returned an error ({error.status_code})."})
    except anthropic.AuthenticationError:
        yield _sse({"type": "error", "message": "The Anthropic API key was rejected. Check backend/.env."})
    except anthropic.RateLimitError:
        yield _sse({"type": "error", "message": "The AI service is rate limiting requests. Try again shortly."})
    except anthropic.APIConnectionError:
        yield _sse({"type": "error", "message": "Could not reach the AI service. Check your connection."})
    except anthropic.APIStatusError as error:
        yield _sse({"type": "error", "message": f"The AI service returned an error ({error.status_code})."})
    except Exception:
        logger.exception("Answer generation failed")
        yield _sse({"type": "error", "message": "Something went wrong while writing the answer."})


def _ask_events(body: AskRequest, visitor: str) -> Iterator[str]:
    reason = guardrails.blocked_reason(body.question)
    if reason:
        yield _sse({"type": "notice", "message": reason})
        yield _sse({"type": "done"})
        return
    if not limiter.allow(visitor):
        yield _sse({"type": "error", "message": "The daily question limit for this demo has been reached."})
        return

    if config.using_local_llm() and config.llm_configured():
        events = (
            local_agent.stream_local_agent(body.question, body.filing_ids)
            if body.use_agent
            else local_agent.stream_single_pass(body.question, body.filing_ids, body.mode)
        )
        yield from _llm_events(events)
        return

    if body.use_agent and config.llm_configured():
        yield from _llm_events(agent.stream_agent(body.question, body.filing_ids))
        return

    started = time.perf_counter()
    passages = retrieval.search(body.question, body.filing_ids, body.top_k, body.mode)
    retrieval_ms = round((time.perf_counter() - started) * 1000)
    yield _sse(
        {
            "type": "passages",
            "passages": [{k: p[k] for k in PUBLIC_PASSAGE_FIELDS} for p in passages],
            "retrieval_ms": retrieval_ms,
        }
    )

    if not passages:
        yield _sse({"type": "error", "message": "No indexed passages were found for those filings."})
        return

    if not config.llm_configured():
        yield _sse(
            {
                "type": "notice",
                "message": "AI answers are off because no model is available. Showing the most relevant passages.",
            }
        )
        yield _sse({"type": "done"})
        return

    yield from _llm_events(answer.stream_answer(body.question, passages))


OVERVIEW_METRICS = {
    "Revenue": ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "SalesRevenueNet", "Revenue"],
    "Operating income": ["OperatingIncomeLoss", "ProfitLossFromOperatingActivities"],
    "Net income": ["NetIncomeLoss", "ProfitLoss"],
    "Research and development": ["ResearchAndDevelopmentExpense"],
    "Operating cash flow": ["NetCashProvidedByUsedInOperatingActivities", "CashFlowsFromUsedInOperatingActivities"],
    "Total assets": ["Assets"],
    "Total liabilities": ["Liabilities"],
}
OVERVIEW_YEARS = 5
FX_URL = "https://api.frankfurter.dev/v1/latest?base=USD&symbols=INR"
EVALUATION_FILES = {
    "retrieval_development": "tune_after.json",
    "retrieval_holdout": "holdout_after.json",
    "answers": "answers_local_summary.json",
    "answers_holdout": "answers_local_holdout_summary.json",
    "answers_first_run": "answers_local_v1_summary.json",
    "volatility": "volatility.json",
    "finbert": "finbert.json",
    "extraction": "extraction.json",
}
RESULTS_DIR = config.BACKEND_DIR / "eval" / "results"


@app.get("/api/companies")
def companies() -> list[dict]:
    seen: dict[str, str] = {}
    for filing in store.list_filings():
        seen.setdefault(filing["ticker"], filing["company"])
    return [{"ticker": ticker, "company": name} for ticker, name in sorted(seen.items())]


@app.get("/api/companies/{ticker}")
def company_overview(ticker: str) -> dict:
    concepts = sorted({name for names in OVERVIEW_METRICS.values() for name in names})
    placeholders = ",".join("?" * len(concepts))
    with store.connect() as db:
        rows = db.execute(
            f"""SELECT concept, fiscal_year, value, company, unit FROM financial_facts
                WHERE ticker = ? AND unit IN ('USD', 'INR') AND concept IN ({placeholders})""",
            [ticker.upper(), *concepts],
        ).fetchall()
    if not rows:
        raise HTTPException(status_code=404, detail="No financial data for that company.")
    currency = "INR" if any(row["unit"] == "INR" for row in rows) else "USD"
    rows = [row for row in rows if row["unit"] == currency]

    values = {(row["concept"], row["fiscal_year"]): row["value"] for row in rows}
    years = sorted({row["fiscal_year"] for row in rows})[-OVERVIEW_YEARS:]
    metrics = [
        {
            "name": label,
            "values": [
                next((values[(c, year)] for c in names if (c, year) in values), None) for year in years
            ],
        }
        for label, names in OVERVIEW_METRICS.items()
    ]
    return {
        "ticker": ticker.upper(),
        "company": rows[0]["company"],
        "currency": currency,
        "years": years,
        "metrics": metrics,
    }


_fx_cache: dict = {}


@app.get("/api/fx")
def exchange_rate() -> dict:
    """Today's European Central Bank reference rate, cached for the day."""
    today = date.today().isoformat()
    if _fx_cache.get("fetched") != today:
        try:
            response = requests.get(FX_URL, timeout=5)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as error:
            raise HTTPException(status_code=503, detail="The exchange rate is unavailable.") from error
        _fx_cache.update(fetched=today, usd_inr=data["rates"]["INR"], date=data["date"])
    return {"usd_inr": _fx_cache["usd_inr"], "date": _fx_cache["date"]}


@app.get("/api/evaluation")
def evaluation_results() -> dict:
    results = {}
    for key, name in EVALUATION_FILES.items():
        path = RESULTS_DIR / name
        results[key] = json.loads(path.read_text()) if path.exists() else None
    return results


@app.post("/api/ask")
def ask(body: AskRequest, request: Request) -> StreamingResponse:
    visitor = request.client.host if request.client else "unknown"
    return StreamingResponse(
        _ask_events(body, visitor),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
