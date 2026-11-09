"""Compare retrieval modes. Needs no API key.

  python scripts/eval_retrieval.py --validate   check the labels only
  python scripts/eval_retrieval.py              run the benchmark and save results
"""
import argparse
import json
import sys
from datetime import date
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

from filinglens import evaluation as ev  # noqa: E402

NARRATIVE_PATH = BACKEND / "eval" / "narrative_questions.json"
RESULTS_DIR = BACKEND / "eval" / "results"
MAX_GOLD_FOR_NARRATIVE = 12


def percent(metric: dict) -> str:
    low, high = metric["ci95"]
    return f"{metric['value'] * 100:.1f}% ({low * 100:.0f}–{high * 100:.0f})"


def table(title: str, summaries: dict[str, dict]) -> str:
    rows = [
        f"### {title}",
        "",
        "| Mode | n | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR | Median ms | p95 ms |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for mode, s in summaries.items():
        latency = s.get("latency_ms", {"median": 0, "p95": 0})
        rows.append(
            f"| {mode} | {s['questions']} | {percent(s['hit@1'])} | {percent(s['hit@3'])} | "
            f"{percent(s['hit@5'])} | {percent(s['hit@10'])} | {s['mrr']:.3f} | "
            f"{latency['median']:.0f} | {latency['p95']:.0f} |"
        )
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--tickers", nargs="+", default=["AAPL", "MSFT"])
    parser.add_argument("--name", default="retrieval_latest", help="name for the saved results")
    args = parser.parse_args()

    chunks, facts, latest = ev.load_corpus()
    latest = {ticker: latest[ticker] for ticker in args.tickers}
    latest_ids = set(latest.values())
    chunks = [c for c in chunks if c["filing_id"] in latest_ids]
    latest_years = {c["ticker"]: c["fiscal_year"] for c in chunks}

    figures, skipped_figures = ev.build_figure_questions(chunks, facts, latest_years)
    narrative, skipped_narrative = ev.build_narrative_questions(NARRATIVE_PATH, chunks)
    questions = figures + narrative
    print(f"{len(figures)} figure and {len(narrative)} narrative questions")
    print("Skipped:", *(skipped_figures + skipped_narrative), sep="\n  ")

    if args.validate:
        for question in narrative:
            flag = "  <- broad" if len(question.gold_ids) > MAX_GOLD_FOR_NARRATIVE else ""
            print(f"{len(question.gold_ids):4d}  {question.id}{flag}")
        counts = sorted(len(q.gold_ids) for q in figures)
        print("figure gold counts min/median/max:", counts[0], counts[len(counts) // 2], counts[-1])
        return

    scopes = {
        "one_filing": lambda question: [latest[question.ticker]],
        "all_filings": lambda question: sorted(latest_ids),
    }
    output = {
        "date": date.today().isoformat(),
        "filings": sorted(latest),
        "questions": {"figure": len(figures), "narrative": len(narrative)},
        "skipped": skipped_figures + skipped_narrative,
        "scopes": {},
    }
    report = [f"# Retrieval benchmark ({output['date']})", ""]

    for scope, filing_ids_for in scopes.items():
        runs = {mode: ev.run_mode(questions, mode, filing_ids_for) for mode in ev.MODES}
        baseline_ranks = runs[ev.BASELINE_MODE][0]
        scope_result = {"overall": {}, "figure": {}, "narrative": {}, "vs_vector": {}, "misses": {}}

        for mode, (ranks, latencies, rankings) in runs.items():
            scope_result["overall"][mode] = ev.summarize(ranks, latencies)
            for category in ("figure", "narrative"):
                subset = [r for r, q in zip(ranks, questions) if q.category == category]
                if subset:
                    scope_result[category][mode] = ev.summarize(subset, [])
            scope_result["vs_vector"][mode] = ev.compare_to_baseline(ranks, baseline_ranks)
            scope_result["misses"][mode] = [
                q.id for q, rank in zip(questions, ranks) if rank is None or rank > 5
            ]
        output["scopes"][scope] = scope_result

        for key, title in (("overall", "All"), ("figure", "Figures"), ("narrative", "Narrative")):
            if scope_result[key]:
                report.append(table(f"{scope}: {title}", scope_result[key]))
                report.append("")
        report.append(f"Hit@5 change vs. vector ({scope}):")
        for mode, change in scope_result["vs_vector"].items():
            low, high = change["ci95"]
            report.append(f"- {mode}: {change['difference'] * 100:+.1f} points (95% CI {low * 100:+.0f} to {high * 100:+.0f})")
        report.append("")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"{args.name}.json").write_text(json.dumps(output, indent=2))
    (RESULTS_DIR / f"{args.name}.md").write_text("\n".join(report))
    print("\n".join(report))


if __name__ == "__main__":
    main()
