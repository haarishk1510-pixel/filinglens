"""Ask the agent a question from the command line: python scripts/ask.py AAPL MSFT -- "question" """
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from filinglens import local_agent, store  # noqa: E402


def ask(question: str, tickers: list[str]) -> None:
    ids = [f["id"] for f in store.list_filings() if f["ticker"] in tickers]
    started = time.perf_counter()
    print(f"\nQ: {question}")
    answer = ""
    for event in local_agent.stream_local_agent(question, ids):
        if event["type"] == "text":
            answer += event["text"]
        elif event["type"] == "citation":
            answer += f"[{event['passage_index'] + 1}]"
        elif event["type"] == "step" and event["status"] != "running":
            print(f"  {event['status']:5s} {event['tool']}: {json.dumps(event['input'])[:170]} -> {event['summary'][:90]}")
        elif event["type"] in ("notice", "error"):
            print(f"  {event['type']}: {event['message']}")
        elif event["type"] == "done":
            print(f"A: {answer.strip()}")
            print(f"  check: {event['citation_check']}")
            print(f"  {event['steps']} steps, {event['input_tokens']}+{event['output_tokens']} tokens, {time.perf_counter() - started:.0f}s")


if __name__ == "__main__":
    split = sys.argv.index("--")
    for question in sys.argv[split + 1:]:
        ask(question, sys.argv[1:split])
