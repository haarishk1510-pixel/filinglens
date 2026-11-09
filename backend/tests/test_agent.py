import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from filinglens import agent, config, facts, store, tools  # noqa: E402


@pytest.fixture
def database(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "test.db")
    store.save_facts(
        "AAPL",
        [
            {
                "ticker": "AAPL", "company": "Apple Inc.", "concept": "NetIncomeLoss",
                "label": "Net income", "unit": "USD", "value": 100_000_000_000.0,
                "period_start": "2025-09-28", "period_end": "2026-09-26", "fiscal_year": 2026,
                "filed": "2026-11-03", "accession": "x",
            }
        ],
    )


def fact(end, start="2025-01-01", form="10-K", filed="2026-02-01", val=1):
    entry = {"end": end, "form": form, "filed": filed, "val": val, "accn": "a"}
    return entry | ({"start": start} if start else {})


def extract(entries):
    payload = {"facts": {"us-gaap": {"Sales": {"label": "Sales", "units": {"USD": entries}}}}}
    return facts.extract_annual_facts(payload, "X", "X Corp")


def test_facts_keep_full_years_and_balance_sheet_dates_only():
    rows = extract([
        fact("2025-12-31"),
        fact("2025-03-31"),
        fact("2025-12-31", start=None),
        fact("2025-12-31", form="10-Q"),
    ])
    assert len(rows) == 2


def test_restated_fact_uses_latest_filing():
    rows = extract([fact("2025-12-31", val=1), fact("2025-12-31", filed="2027-02-01", val=2)])
    assert [r["value"] for r in rows] == [2]


def test_early_january_year_end_counts_as_prior_fiscal_year():
    assert facts.fiscal_year_of("2026-01-03") == 2025
    assert facts.fiscal_year_of("2026-01-31") == 2026


def test_sql_reads_the_allowed_table(database):
    columns, rows, truncated = tools.run_readonly_sql("SELECT ticker, value FROM financial_facts")
    assert columns == ["ticker", "value"]
    assert rows == [("AAPL", 100_000_000_000.0)]
    assert not truncated


@pytest.mark.parametrize(
    "sql",
    [
        "DROP TABLE financial_facts",
        "DELETE FROM financial_facts",
        "INSERT INTO financial_facts(ticker) VALUES ('x')",
        "SELECT * FROM chunks",
        "SELECT * FROM sqlite_master",
        "ATTACH DATABASE '/tmp/x.db' AS x",
        "PRAGMA table_info(chunks)",
        "SELECT load_extension('x')",
        "SELECT 1; DROP TABLE financial_facts",
    ],
)
def test_sql_rejects_anything_but_reading_the_table(database, sql):
    with pytest.raises(tools.ToolError):
        tools.run_readonly_sql(sql)
    assert store.has_facts("AAPL")


def add_second_company():
    rows = [{
        "ticker": "MSFT", "company": "Microsoft", "concept": "NetIncomeLoss", "label": "Net income", "unit": "USD",
        "value": 1.0, "period_start": None, "period_end": "2026-06-30", "fiscal_year": 2026, "filed": "2026", "accession": "y",
    }]
    store.save_facts("MSFT", rows)


def test_sql_sees_only_selected_companies(database):
    add_second_company()
    _, rows, _ = tools.run_readonly_sql("SELECT ticker FROM financial_facts", ["AAPL"])
    assert rows == [("AAPL",)]


def test_sql_cannot_reach_around_the_company_limit(database):
    add_second_company()
    with pytest.raises(tools.ToolError):
        tools.run_readonly_sql("SELECT ticker FROM main.financial_facts", ["AAPL"])
    with pytest.raises(tools.ToolError):
        tools.run_readonly_sql("DROP VIEW financial_facts", ["AAPL"])


def test_results_must_say_which_metric_each_value_is(database):
    with pytest.raises(tools.ToolError):
        tools.query_financials(tools.SqlInput(sql="SELECT value FROM financial_facts"), tools.Evidence())
    outcome = tools.query_financials(
        tools.SqlInput(sql="SELECT ticker, concept, value FROM financial_facts"), tools.Evidence(), ["AAPL"]
    )
    assert outcome.summary == "1 rows"


def statement_facts(ticker, unit, concepts):
    return [{
        "ticker": ticker, "company": ticker, "concept": concept, "label": concept, "unit": unit, "value": value,
        "period_start": None, "period_end": "2026-06-30", "fiscal_year": 2026, "filed": "2026", "accession": "y",
    } for concept, value in concepts.items()]


def statement_table(ticker, years=(2026,), tickers=None):
    evidence = tools.Evidence()
    args = tools.StatementInput(ticker=ticker, fiscal_years=list(years))
    return tools.financial_statement(args, evidence, tickers), evidence


def test_statement_shows_standard_items_in_millions(database):
    store.save_facts("BA", statement_facts("BA", "USD", {
        "PropertyPlantAndEquipmentGross": 31_213_000_000.0,
        "PropertyPlantAndEquipmentNet": 12_645_000_000.0,
        "CostOfGoodsAndServicesSold": 2_500_000.0,
        "CostOfGoodsSold": 1_000_000.0,
    }))
    _, evidence = statement_table("ba")
    text = evidence.passages[0]["text"]
    assert "Net property, plant and equipment | 12,645" in text
    assert "Cost of sales | 2.5" in text
    assert "31,213" not in text


def test_statement_revenue_is_the_total_not_a_partial_line(database):
    store.save_facts("X", statement_facts("X", "USD", {
        "Revenues": 2_000_000.0, "RevenueFromContractWithCustomerExcludingAssessedTax": 9_000_000.0,
    }))
    _, evidence = statement_table("X")
    assert "Revenue (total) (" not in evidence.passages[0]["text"]
    assert "Revenue (total) | 9" in evidence.passages[0]["text"]


def test_statement_marks_missing_years_and_prefers_rupees(database):
    store.save_facts("INFY", [
        *statement_facts("INFY", "USD", {"Assets": 1_000_000.0}),
        *statement_facts("INFY", "INR", {"Assets": 80_000_000.0}),
    ])
    _, evidence = statement_table("INFY", (2026, 2025))
    text = evidence.passages[0]["text"]
    assert text.startswith("Line item (INR millions) | FY2026 | FY2025")
    assert "Total assets | 80 | not reported" in text


def test_statement_is_limited_to_selected_companies(database):
    with pytest.raises(tools.ToolError):
        statement_table("AAPL", tickers=["MSFT"])


def test_sql_stops_runaway_queries(database, monkeypatch):
    monkeypatch.setattr(tools, "SQL_TIME_LIMIT_SECONDS", 0.05)
    runaway = (
        "WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM n) "
        "SELECT COUNT(*) FROM n"
    )
    with pytest.raises(tools.ToolError):
        tools.run_readonly_sql(runaway)


@pytest.mark.parametrize(
    ("expression", "expected"),
    [("(478834-416161)/416161*100", (478834 - 416161) / 416161 * 100), ("round(2.345, 1)", 2.3), ("-2**2", -4)],
)
def test_calculator_evaluates_arithmetic(expression, expected):
    assert float(tools.calculate(tools.CalculateInput(expression=expression)).content) == expected


@pytest.mark.parametrize(
    "expression",
    ["__import__('os').system('id')", "1/0", "9**9**9", "1,000 + 2", "open('x')", "(1).real"],
)
def test_calculator_rejects_unsafe_or_invalid_input(expression):
    with pytest.raises(tools.ToolError):
        tools.calculate(tools.CalculateInput(expression=expression))


def tool_use(call_id, name, tool_input):
    return NS(type="tool_use", id=call_id, name=name, input=tool_input)


class ScriptedStream:
    def __init__(self, turn):
        self.turn = turn

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def __iter__(self):
        return iter(self.turn.get("events", []))

    def get_final_message(self):
        return NS(
            content=self.turn.get("content", []),
            stop_reason=self.turn["stop_reason"],
            model="claude-opus-5-5",
            usage=NS(input_tokens=100, output_tokens=10),
        )


class ScriptedClient:
    def __init__(self, turns):
        self.turns = list(turns)
        self.requests = []
        self.messages = NS(stream=self._stream)
        self.beta = NS(messages=NS(stream=self._stream))

    def _stream(self, **request):
        self.requests.append(request)
        return ScriptedStream(self.turns.pop(0))


def text_event(text):
    return NS(type="content_block_delta", delta=NS(type="text_delta", text=text))


def citation_event(source, cited):
    return NS(type="content_block_delta", delta=NS(
        type="citations_delta", citation=NS(source=source, cited_text=cited),
    ))


def test_agent_runs_tools_then_answers_with_citations(database):
    sql = "SELECT concept, value FROM financial_facts WHERE concept = 'NetIncomeLoss'"
    client = ScriptedClient([
        {"stop_reason": "tool_use", "content": [tool_use("t1", "query_financials", {"sql": sql})]},
        {"stop_reason": "tool_use", "content": [tool_use("t2", "calculate", {"expression": "1 + 1"})]},
        {
            "stop_reason": "end_turn",
            "content": [],
            "events": [text_event("Net income was $100B."), citation_event("sql:0", "100000000000")],
        },
    ])
    events = list(agent.stream_agent("Net income?", ["f1"], client=client))

    steps = [e for e in events if e["type"] == "step"]
    assert [(s["tool"], s["status"]) for s in steps] == [
        ("query_financials", "running"), ("query_financials", "done"),
        ("calculate", "running"), ("calculate", "done"),
    ]
    assert next(e for e in events if e["type"] == "citation")["passage_index"] == 0
    assert events[-1]["type"] == "done"
    assert events[-1]["steps"] == 2
    assert events[-1]["input_tokens"] == 300

    tool_result = client.requests[1]["messages"][-1]["content"][0]
    assert tool_result["content"][0]["type"] == "search_result"
    assert tool_result["content"][0]["citations"] == {"enabled": True}


def test_agent_returns_tool_errors_so_the_model_can_recover(database):
    client = ScriptedClient([
        {"stop_reason": "tool_use", "content": [tool_use("t1", "query_financials", {"sql": "SELECT * FROM chunks"})]},
        {"stop_reason": "tool_use", "content": [tool_use("t2", "calculate", {"wrong_field": 1})]},
        {"stop_reason": "end_turn", "content": []},
    ])
    events = list(agent.stream_agent("q?", ["f1"], client=client))
    assert [e["status"] for e in events if e["type"] == "step" and e["status"] != "running"] == ["error", "error"]
    assert client.requests[1]["messages"][-1]["content"][0]["is_error"] is True


def test_agent_must_answer_on_its_last_allowed_turn(database):
    call = {"stop_reason": "tool_use", "content": [tool_use("t", "calculate", {"expression": "1"})]}
    client = ScriptedClient([call] * (agent.MAX_MODEL_TURNS - 1) + [{"stop_reason": "end_turn", "content": []}])
    list(agent.stream_agent("q?", ["f1"], client=client))
    assert len(client.requests) == agent.MAX_MODEL_TURNS
    assert client.requests[-2]["tool_choice"] == {"type": "auto"}
    assert client.requests[-1]["tool_choice"] == {"type": "none"}
