"""Fine-tune a small model to extract financial figures from filing tables as JSON.

  python ml/extraction_lora.py prepare    build examples from filings and SEC data
  python ml/extraction_lora.py train      LoRA fine-tuning on Apple silicon (4-bit base model)
  python ml/extraction_lora.py evaluate   compare with the same model using prompting alone

Companies in the test set never appear in training. Labels come from SEC XBRL values.
"""
import argparse
import json
import random
import re
import subprocess
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

from filinglens import config, evaluation, store  # noqa: E402

BASE_MODEL = "mlx-community/Qwen2.5-3B-4bit"
DATA_PATH = config.DATA_DIR / "ml" / "extraction"
ADAPTER_PATH = BACKEND / "models" / "extraction-lora"
RESULTS_PATH = BACKEND / "eval" / "results" / "extraction.json"

SEED = 0
MIN_FIELDS = 3
MAX_WORDS = 500
MAX_TEST_EXAMPLES = 60
MAX_NEW_TOKENS = 220
ITERATIONS = 500

FIELDS = {
    "revenue": ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "SalesRevenueNet"],
    "cost_of_revenue": ["CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold"],
    "gross_profit": ["GrossProfit"],
    "operating_income": ["OperatingIncomeLoss"],
    "net_income": ["NetIncomeLoss", "ProfitLoss"],
    "total_assets": ["Assets"],
    "total_liabilities": ["Liabilities"],
    "shareholders_equity": ["StockholdersEquity"],
    "cash": ["CashAndCashEquivalentsAtCarryingValue"],
    "operating_cash_flow": ["NetCashProvidedByUsedInOperatingActivities"],
}

SYSTEM = (
    "Extract figures from a financial table for the requested fiscal year. Return only a JSON object with "
    f"exactly these keys: {', '.join(FIELDS)}. Use the value in millions of dollars as a number, "
    "or null if the table does not show it."
)


def build_target(table_text: str, values: dict[str, float]) -> dict[str, int | None]:
    """Keep a value only if the table really prints it, in millions."""
    target: dict[str, int | None] = dict.fromkeys(FIELDS)
    for field, value in values.items():
        if evaluation.figure_pattern(abs(value)).search(table_text):
            target[field] = int(round(value / 1_000_000))
    return target


def values_for_year(facts: list[dict], fiscal_year: int) -> dict[str, float]:
    by_concept = {
        row["concept"]: row["value"]
        for row in sorted(facts, key=lambda r: r["filed"])
        if row["fiscal_year"] == fiscal_year and row["unit"] == "USD"
    }
    values = {}
    for field, concepts in FIELDS.items():
        found = next((by_concept[c] for c in concepts if c in by_concept), None)
        if found is not None and abs(found) >= 1_000_000:
            values[field] = found
    return values


def split_tickers(tickers: list[str]) -> dict[str, str]:
    """Assign whole companies to one split, so a company's tables never leak across."""
    ordered = sorted(tickers)
    random.Random(SEED).shuffle(ordered)
    test_count = max(1, len(ordered) // 4)
    assignment = {ticker: "test" for ticker in ordered[:test_count]}
    assignment |= {ticker: "valid" for ticker in ordered[test_count : test_count + max(1, test_count // 2)]}
    return {ticker: assignment.get(ticker, "train") for ticker in ordered}


def conversation(table_text: str, fiscal_year: int, target: dict | None = None) -> list[dict]:
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Fiscal year: {fiscal_year}\n\n{table_text}"},
    ]
    if target is not None:
        messages.append({"role": "assistant", "content": json.dumps(target)})
    return messages


def prepare() -> None:
    with store.connect() as db:
        rows = [dict(r) for r in db.execute("SELECT * FROM financial_facts").fetchall()]
    facts_by_ticker: dict[str, list[dict]] = {}
    for row in rows:
        facts_by_ticker.setdefault(row["ticker"], []).append(row)

    filings = store.list_filings()
    splits = split_tickers(sorted({f["ticker"] for f in filings}))
    examples: dict[str, list[dict]] = {"train": [], "valid": [], "test": []}
    for filing in filings:
        facts = facts_by_ticker.get(filing["ticker"], [])
        values = values_for_year(facts, filing["fiscal_year"])
        for chunk in store.load_chunks([filing["id"]]):
            if chunk["kind"] != "table" or len(chunk["text"].split()) > MAX_WORDS:
                continue
            target = build_target(chunk["text"], values)
            if sum(value is not None for value in target.values()) >= MIN_FIELDS:
                examples[splits[filing["ticker"]]].append(
                    {
                        "ticker": filing["ticker"],
                        "fiscal_year": filing["fiscal_year"],
                        "text": chunk["text"],
                        "target": target,
                    }
                )

    DATA_PATH.mkdir(parents=True, exist_ok=True)
    random.Random(SEED).shuffle(examples["test"])
    for name, rows in examples.items():
        with (DATA_PATH / f"{name}.jsonl").open("w") as output:
            for example in rows:
                record = (
                    example
                    if name == "test"
                    else {"messages": conversation(example["text"], example["fiscal_year"], example["target"])}
                )
                output.write(json.dumps(record) + "\n")
        print(name, len(rows), "examples from", len({row["ticker"] for row in rows}), "companies")


def train() -> None:
    ADAPTER_PATH.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable, "-m", "mlx_lm", "lora",
            "--model", BASE_MODEL, "--train", "--data", str(DATA_PATH),
            "--adapter-path", str(ADAPTER_PATH), "--iters", str(ITERATIONS),
            "--batch-size", "1", "--learning-rate", "1e-4", "--steps-per-eval", "100",
            "--steps-per-report", "50", "--save-every", "100", "--max-seq-length", "1800", "--grad-checkpoint",
        ],
        check=True,
    )


def parse_json(text: str) -> dict | None:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def score_predictions(predictions: list[dict | None], targets: list[dict]) -> dict:
    true_positive = false_positive = false_negative = perfect = 0
    for predicted, target in zip(predictions, targets):
        predicted = predicted or {}
        all_right = bool(predicted)
        for field in FIELDS:
            expected, actual = target.get(field), predicted.get(field)
            if isinstance(actual, str) or isinstance(actual, bool):
                actual = "invalid"
            if actual is not None and expected is not None and actual == expected:
                true_positive += 1
                continue
            if actual is not None:
                false_positive += 1
            if expected is not None:
                false_negative += 1
            if actual != expected:
                all_right = False
        perfect += all_right
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    return {
        "examples": len(targets),
        "valid_json_rate": sum(p is not None for p in predictions) / max(1, len(predictions)),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(1e-9, precision + recall),
        "all_fields_correct_rate": perfect / max(1, len(targets)),
    }


def run_model(adapter: Path | None, examples: list[dict]) -> list[dict | None]:
    from mlx_lm import generate, load

    model, tokenizer = load(BASE_MODEL, adapter_path=str(adapter) if adapter else None)
    predictions = []
    for example in examples:
        prompt = tokenizer.apply_chat_template(
            conversation(example["text"], example["fiscal_year"]), add_generation_prompt=True, tokenize=False
        )
        predictions.append(parse_json(generate(model, tokenizer, prompt=prompt, max_tokens=MAX_NEW_TOKENS)))
        print(len(predictions), flush=True)
    return predictions


def evaluate() -> None:
    examples = [json.loads(line) for line in (DATA_PATH / "test.jsonl").read_text().splitlines()][:MAX_TEST_EXAMPLES]
    targets = [example["target"] for example in examples]
    results = {
        "base_model": BASE_MODEL,
        "test_companies": sorted({example["ticker"] for example in examples}),
        "prompting_only": score_predictions(run_model(None, examples), targets),
        "fine_tuned": score_predictions(run_model(ADAPTER_PATH, examples), targets),
    }
    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "train", "evaluate"])
    arguments = parser.parse_args()
    {"prepare": prepare, "train": train, "evaluate": evaluate}[arguments.command]()
