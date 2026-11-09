"""Retrieval benchmark.

Relevance labels are computed from the filing text and SEC data alone, so they don't
depend on any retriever being compared. A question counts as a hit at k when any chunk
carrying the answer appears in the top k results.
"""
import json
import re
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import retrieval, store

KS = (1, 3, 5, 10)
MODES = ("keyword", "vector", "hybrid", "hybrid_rerank")
BASELINE_MODE = "vector"
BOOTSTRAP_SAMPLES = 2000
SHORT_NAMES = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "NVDA": "NVIDIA",
    "GOOGL": "Alphabet",
    "WMT": "Walmart",
    "KO": "Coca-Cola",
}
MIN_FIGURE_MILLIONS = 100
FIGURE_YEARS = (2026, 2025)

# Worded the way a person would ask, deliberately not the SEC's tag names.
FIGURE_TEMPLATES = {
    "RevenueFromContractWithCustomerExcludingAssessedTax": "What was {company}'s total revenue in fiscal {year}?",
    "NetIncomeLoss": "How much profit did {company} make in fiscal {year}?",
    "OperatingIncomeLoss": "What was {company}'s operating income in fiscal {year}?",
    "GrossProfit": "What was {company}'s gross profit or gross margin in dollars for fiscal {year}?",
    "CostOfGoodsAndServicesSold": "What was {company}'s total cost of sales in fiscal {year}?",
    "CostOfRevenue": "What was {company}'s total cost of revenue in fiscal {year}?",
    "ResearchAndDevelopmentExpense": "How much did {company} spend on research and development in fiscal {year}?",
    "IncomeTaxExpenseBenefit": "What was {company}'s provision for income taxes in fiscal {year}?",
    "Assets": "What were {company}'s total assets at the end of fiscal {year}?",
    "Liabilities": "What were {company}'s total liabilities at the end of fiscal {year}?",
    "StockholdersEquity": "What was {company}'s total shareholders' equity at the end of fiscal {year}?",
    "CashAndCashEquivalentsAtCarryingValue": "How much cash and cash equivalents did {company} hold at the end of fiscal {year}?",
    "AccountsReceivableNetCurrent": "What were {company}'s net accounts receivable at the end of fiscal {year}?",
    "InventoryNet": "What was {company}'s inventory balance at the end of fiscal {year}?",
    "NetCashProvidedByUsedInOperatingActivities": "How much cash did {company} generate from operations in fiscal {year}?",
    "PaymentsToAcquirePropertyPlantAndEquipment": "How much did {company} spend on capital expenditures in fiscal {year}?",
    "PaymentsOfDividends": "How much did {company} pay in dividends in fiscal {year}?",
    "PaymentsForRepurchaseOfCommonStock": "How much did {company} spend on share buybacks in fiscal {year}?",
    "ShareBasedCompensation": "What was {company}'s stock-based compensation expense in fiscal {year}?",
}


@dataclass
class Question:
    id: str
    ticker: str
    category: str
    text: str
    gold_ids: set[str]


def figure_pattern(value: float) -> re.Pattern:
    """Match a whole-dollar value as filings print it, in millions with thousands commas."""
    return re.compile(rf"(?<![\d,.]){int(round(value / 1_000_000)):,}(?![\d,]|\.\d)")


def build_figure_questions(
    chunks: list[dict], facts: list[dict], latest_years: dict[str, int] | None = None
) -> tuple[list[Question], list[str]]:
    """Ask about each company's latest fiscal year and the one before it."""
    questions, skipped = [], []
    for fact in facts:
        template = FIGURE_TEMPLATES.get(fact["concept"])
        if latest_years is None:
            years = FIGURE_YEARS
        elif fact["ticker"] in latest_years:
            years = (latest_years[fact["ticker"]], latest_years[fact["ticker"]] - 1)
        else:
            continue
        if template is None or fact["unit"] != "USD" or fact["fiscal_year"] not in years:
            continue
        identifier = f"{fact['ticker']}-{fact['concept']}-{fact['fiscal_year']}".lower()
        if abs(fact["value"]) < MIN_FIGURE_MILLIONS * 1_000_000:
            skipped.append(f"{identifier}: number too small to identify reliably")
            continue
        pattern = figure_pattern(abs(fact["value"]))
        gold = {c["id"] for c in chunks if c["ticker"] == fact["ticker"] and pattern.search(c["text"])}
        if not gold:
            skipped.append(f"{identifier}: value not printed in this filing")
            continue
        text = template.format(company=SHORT_NAMES[fact["ticker"]], year=fact["fiscal_year"])
        questions.append(Question(identifier, fact["ticker"], "figure", text, gold))
    return questions, skipped


def narrative_gold(spec: dict, chunks: list[dict]) -> set[str]:
    """A chunk is relevant when it is in the right company and section and matches every pattern."""
    patterns = [re.compile(p, re.IGNORECASE) for p in spec["patterns"]]
    return {
        chunk["id"]
        for chunk in chunks
        if chunk["ticker"] == spec["ticker"]
        and chunk["section"].startswith(spec.get("section", ""))
        and all(p.search(chunk["text"]) for p in patterns)
    }


def build_narrative_questions(path: Path, chunks: list[dict]) -> tuple[list[Question], list[str]]:
    questions, skipped = [], []
    tickers = {chunk["ticker"] for chunk in chunks}
    for spec in json.loads(path.read_text()):
        if spec["ticker"] not in tickers:
            continue
        gold = narrative_gold(spec, chunks)
        if gold:
            questions.append(Question(spec["id"], spec["ticker"], "narrative", spec["question"], gold))
        else:
            skipped.append(f"{spec['id']}: no chunk matches its label rule")
    return questions, skipped


def first_hit_rank(ranked_ids: list[str], gold_ids: set[str]) -> int | None:
    return next((rank for rank, cid in enumerate(ranked_ids, start=1) if cid in gold_ids), None)


def hit_at(ranks: list[int | None], k: int) -> np.ndarray:
    return np.array([rank is not None and rank <= k for rank in ranks], dtype=float)


def reciprocal_ranks(ranks: list[int | None]) -> np.ndarray:
    return np.array([1 / rank if rank else 0.0 for rank in ranks])


def bootstrap_interval(values: np.ndarray, seed: int = 0) -> tuple[float, float]:
    """95% interval over which questions happened to be in the benchmark."""
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(values), size=(BOOTSTRAP_SAMPLES, len(values)))
    means = values[picks].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarize(ranks: list[int | None], latencies_ms: list[float]) -> dict:
    summary = {"questions": len(ranks)}
    for k in KS:
        hits = hit_at(ranks, k)
        low, high = bootstrap_interval(hits)
        summary[f"hit@{k}"] = {"value": float(hits.mean()), "ci95": [low, high]}
    summary["mrr"] = float(reciprocal_ranks(ranks).mean())
    if latencies_ms:
        summary["latency_ms"] = {
            "median": statistics.median(latencies_ms),
            "p95": float(np.percentile(latencies_ms, 95)),
        }
    return summary


def run_mode(questions: list[Question], mode: str, filing_ids_for) -> tuple[list[int | None], list[float], list[list[str]]]:
    ranks, latencies, rankings = [], [], []
    for question in questions:
        started = time.perf_counter()
        results = retrieval.search(question.text, filing_ids_for(question), top_k=max(KS), mode=mode)
        latencies.append((time.perf_counter() - started) * 1000)
        ids = [result["id"] for result in results]
        rankings.append(ids)
        ranks.append(first_hit_rank(ids, question.gold_ids))
    return ranks, latencies, rankings


def compare_to_baseline(ranks: list[int | None], baseline: list[int | None], k: int = 5) -> dict:
    difference = hit_at(ranks, k) - hit_at(baseline, k)
    low, high = bootstrap_interval(difference)
    return {"k": k, "difference": float(difference.mean()), "ci95": [low, high]}


def load_corpus() -> tuple[list[dict], list[dict], dict[str, str]]:
    chunks = [{k: v for k, v in c.items() if k != "embedding"} for c in store.load_chunks()]
    with store.connect() as db:
        facts = [dict(row) for row in db.execute("SELECT * FROM financial_facts").fetchall()]
    latest: dict[str, dict] = {}
    for filing in store.list_filings():
        if filing["ticker"] not in latest or filing["fiscal_year"] > latest[filing["ticker"]]["fiscal_year"]:
            latest[filing["ticker"]] = filing
    return chunks, facts, {ticker: filing["id"] for ticker, filing in latest.items()}
