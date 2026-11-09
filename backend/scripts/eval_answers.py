"""Measure answer accuracy on FinanceBench questions, agent vs. single-pass RAG.

  python scripts/eval_answers.py --ingest    index the filings the questions need
  python scripts/eval_answers.py --run       answer questions (safe to stop and resume)
  python scripts/eval_answers.py --report    grade saved answers and write a summary
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

from filinglens import config, edgar, evaluation, financebench as fb, local_agent, store  # noqa: E402
from filinglens.ingest import ingest_year  # noqa: E402

RESULTS_DIR = BACKEND / "eval" / "results"
ANSWERS_PATH = RESULTS_DIR / "answers_local.jsonl"
SYSTEMS = ("agent", "single_pass")
DEVELOPMENT_QUESTIONS = 40


def filing_id_for(question: fb.BenchmarkQuestion, filings: list[dict]) -> str | None:
    return next(
        (f["id"] for f in filings if f["ticker"] == question.ticker and f["fiscal_year"] == question.fiscal_year),
        None,
    )


def ingest(questions: list[fb.BenchmarkQuestion]) -> None:
    for ticker, year in sorted({(q.ticker, q.fiscal_year) for q in questions}):
        try:
            cik = fb.CIK_OVERRIDES.get(ticker) or edgar.lookup_cik(ticker)[0]
            print(ticker, year, ingest_year(ticker, cik, year), flush=True)
        except Exception as error:  # one missing filing shouldn't stop the rest
            print(ticker, year, "FAILED", repr(error), flush=True)


def completed() -> set[tuple[str, str]]:
    if not ANSWERS_PATH.exists():
        return set()
    rows = [json.loads(line) for line in ANSWERS_PATH.read_text().splitlines()]
    return {(row["id"], row["system"]) for row in rows if not row.get("error")}


def answer_one(question: fb.BenchmarkQuestion, filing_id: str, system: str) -> dict:
    started = time.perf_counter()
    if system == "agent":
        events = local_agent.stream_local_agent(question.question, [filing_id])
    else:
        events = local_agent.stream_single_pass(question.question, [filing_id], "hybrid_rerank")

    record = {"id": question.id, "system": system, "answer": "", "steps": [], "notices": []}
    try:
        for event in events:
            if event["type"] == "text":
                record["answer"] += event["text"]
            elif event["type"] == "step" and event["status"] != "running":
                record["steps"].append({"tool": event["tool"], "status": event["status"]})
            elif event["type"] == "notice":
                record["notices"].append(event["message"])
            elif event["type"] == "done":
                record["citation_check"] = event["citation_check"]
    except Exception as error:  # recorded and retried on the next run
        record["error"] = repr(error)
    record["seconds"] = round(time.perf_counter() - started, 1)
    return record


def run(questions: list[fb.BenchmarkQuestion]) -> None:
    filings = store.list_filings()
    done = completed()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for number, question in enumerate(questions, start=1):
        filing_id = filing_id_for(question, filings)
        if filing_id is None:
            print(f"{number}: {question.id} has no indexed filing", flush=True)
            continue
        for system in SYSTEMS:
            if (question.id, system) in done:
                continue
            record = answer_one(question, filing_id, system)
            with ANSWERS_PATH.open("a") as output:
                output.write(json.dumps(record) + "\n")
            print(f"{number}/{len(questions)} {system} {record['seconds']}s", flush=True)


def rate(flags: list[bool]) -> dict:
    values = np.array(flags, dtype=float)
    low, high = evaluation.bootstrap_interval(values)
    return {"n": len(flags), "rate": float(values.mean()), "ci95": [low, high]}


def report(questions: list[fb.BenchmarkQuestion], output_name: str = "answers_local_summary.json") -> None:
    golds = {q.id: q for q in questions}
    latest: dict[tuple[str, str], dict] = {}
    for line in ANSWERS_PATH.read_text().splitlines():
        row = json.loads(line)
        if row["id"] in golds and not row.get("error"):
            latest[(row["id"], row["system"])] = row

    both = sorted({i for i, _ in latest if (i, "agent") in latest and (i, "single_pass") in latest})
    summary: dict = {"model": config.LOCAL_LLM_MODEL, "paired_questions": len(both), "systems": {}}
    grades = {}
    for system in SYSTEMS:
        rows = [latest[(i, system)] for i in both]
        graded = [fb.grade(r["answer"], golds[r["id"]].gold_value) for r in rows]
        grades[system] = graded
        checks = [(r.get("citation_check") or {}, g["lenient"]) for r, g in zip(rows, graded)]
        clean = [ok for c, ok in checks if c.get("claims") and c["claims"] == c["supported"]]
        flagged = [ok for c, ok in checks if c.get("claims") and c["claims"] != c["supported"]]
        summary["systems"][system] = {
            "strict": rate([g["strict"] for g in graded]),
            "lenient": rate([g["lenient"] for g in graded]),
            "no_answer": sum(not r["answer"].strip() for r in rows),
            "median_seconds": float(np.median([r["seconds"] for r in rows])),
            "correct_when_all_figures_matched": {"n": len(clean), "correct": sum(clean)},
            "correct_when_figures_flagged": {"n": len(flagged), "correct": sum(flagged)},
        }
    for kind in ("strict", "lenient"):
        difference = np.array([a[kind] for a in grades["agent"]], float) - np.array(
            [s[kind] for s in grades["single_pass"]], float
        )
        low, high = evaluation.bootstrap_interval(difference)
        summary[f"agent_minus_single_pass_{kind}"] = {"difference": float(difference.mean()), "ci95": [low, high]}

    (RESULTS_DIR / output_name).write_text(json.dumps(summary, indent=2))
    print(output_name, json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ingest", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--count", type=int, default=40)
    args = parser.parse_args()

    questions = fb.load_selection(args.count)
    if args.ingest:
        ingest(questions)
    if args.run:
        run(questions)
    if args.report:
        # The agent was improved after reading failures on the first 40, so only the rest are unseen.
        report(questions[:DEVELOPMENT_QUESTIONS])
        if questions[DEVELOPMENT_QUESTIONS:]:
            report(questions[DEVELOPMENT_QUESTIONS:], "answers_local_holdout_summary.json")


if __name__ == "__main__":
    main()
