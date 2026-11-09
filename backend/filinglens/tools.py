"""The tools the agent can call: filing search, read-only SQL, and a calculator."""
import ast
import math
import operator
import sqlite3
import time
from dataclasses import dataclass, field
from urllib.parse import quote

from pydantic import BaseModel, ConfigDict, Field

from . import config, retrieval, store
from .answer import citable_units

SEARCH_RESULTS = 6
MAX_SQL_ROWS = 40
SQL_TIME_LIMIT_SECONDS = 2.0
ALLOWED_SQL_TABLE = "financial_facts"
MAX_EXPONENT = 12
MAX_NUMBER = 1e30

MAX_STATEMENT_YEARS = 6
# Each line item maps to the names companies report it under, in order of preference.
LINE_ITEMS = {
    "Revenue (total)": [
        "Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueNet", "Revenue",
    ],
    "Cost of sales": ["CostOfGoodsAndServicesSold", "CostOfRevenue", "CostOfGoodsSold", "CostOfSales"],
    "Gross profit": ["GrossProfit"],
    "Operating income": ["OperatingIncomeLoss", "ProfitLossFromOperatingActivities"],
    "Net income attributable to shareholders": ["NetIncomeLoss", "ProfitLossAttributableToOwnersOfParent", "ProfitLoss"],
    "Depreciation and amortization": [
        "DepreciationDepletionAndAmortization", "DepreciationAmortizationAndAccretionNet",
        "DepreciationAndAmortization", "DepreciationAndAmortisationExpense",
    ],
    "Cash from operating activities": [
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
        "CashFlowsFromUsedInOperatingActivities",
    ],
    "Capital expenditure": [
        "PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets",
        "PurchaseOfPropertyPlantAndEquipmentClassifiedAsInvestingActivities",
    ],
    "Cash dividends paid": [
        "PaymentsOfDividends", "PaymentsOfDividendsCommonStock", "DividendsPaidClassifiedAsFinancingActivities",
    ],
    "Total assets": ["Assets"],
    "Total current assets": ["AssetsCurrent", "CurrentAssets"],
    "Total liabilities": ["Liabilities"],
    "Total current liabilities": ["LiabilitiesCurrent", "CurrentLiabilities"],
    "Shareholders' equity": [
        "StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest", "Equity",
    ],
    "Inventory": ["InventoryNet", "Inventories"],
    "Accounts receivable, net": ["AccountsReceivableNetCurrent", "ReceivablesNetCurrent", "TradeAndOtherCurrentReceivables"],
    "Accounts payable": ["AccountsPayableCurrent", "AccountsPayableTradeCurrent", "TradeAndOtherCurrentPayables"],
    "Net property, plant and equipment": ["PropertyPlantAndEquipmentNet", "PropertyPlantAndEquipment"],
    "Cash and cash equivalents": ["CashAndCashEquivalentsAtCarryingValue", "CashAndCashEquivalents"],
}

TOOL_DEFINITIONS = [
    {
        "name": "financial_statement",
        "description": (
            "Get a company's standard income statement, cash flow, and balance sheet line items for the fiscal "
            "years you list, as one table in millions. Use this first for revenue, cost of sales, operating "
            "income, net income, depreciation, operating cash flow, capital expenditure, dividends paid, assets, "
            "liabilities, equity, inventory, receivables, payables, and net PP&E. Ask for every year you need in "
            "one call, including the prior year for averages and growth."
        ),
        "eager_input_streaming": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "The company ticker."},
                "fiscal_years": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": f"Fiscal years to include, at most {MAX_STATEMENT_YEARS}.",
                },
            },
            "required": ["ticker", "fiscal_years"],
            "additionalProperties": False,
        },
    },
    {
        "name": "search_filings",
        "description": (
            "Search the text and tables of the selected 10-K filings. Use it for explanations, "
            "risks, strategy, segment detail, and anything not in structured financial data."
        ),
        "eager_input_streaming": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to look for, in natural language."},
                "ticker": {"type": "string", "description": "Optional: limit to one company ticker."},
                "fiscal_year": {"type": "integer", "description": "Optional: limit to one fiscal year."},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "query_financials",
        "description": (
            "Run one read-only SQLite SELECT against annual figures reported to the SEC. Use it for "
            "exact reported numbers, multi-year trends, and company comparisons. Always select concept and "
            "label. A broad LIKE search returns unrelated items, such as remaining contract revenue, so use it "
            "only to find the exact concept for the total you need, then query that concept.\n"
            "Table financial_facts(ticker, company, concept, label, unit, value, period_start, "
            "period_end, fiscal_year, filed, accession).\n"
            "Values are in whole units, so USD 478834000000 means $478.834 billion and "
            "INR 10756750000000 means ₹10,75,675 crore. Indian companies may report in INR, USD, or both, so "
            "select unit too. period_start is NULL for balance-sheet values. Filter ticker to the selected "
            "companies.\n"
            "Concept names differ by company. International filers use Revenue, ProfitLoss, Assets, Equity, and "
            "CashFlowsFromUsedInOperatingActivities. Others: Revenues, "
            "RevenueFromContractWithCustomerExcludingAssessedTax, NetIncomeLoss, OperatingIncomeLoss, "
            "ResearchAndDevelopmentExpense, Assets, Liabilities, StockholdersEquity, "
            "CashAndCashEquivalentsAtCarryingValue, EarningsPerShareDiluted. If a query returns no "
            "rows, look for the right name with concept LIKE or label LIKE."
        ),
        "eager_input_streaming": True,
        "input_schema": {
            "type": "object",
            "properties": {"sql": {"type": "string", "description": "A single SELECT statement."}},
            "required": ["sql"],
            "additionalProperties": False,
        },
    },
    {
        "name": "calculate",
        "description": (
            "Evaluate an arithmetic expression exactly. Use it for every growth rate, margin, ratio, "
            "sum, or difference. Supports + - * / ** % parentheses, abs, round, min, and max. "
            "Write plain numbers without commas."
        ),
        "eager_input_streaming": True,
        "input_schema": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
            "additionalProperties": False,
        },
    },
]


class ToolError(Exception):
    """A problem the agent can read and correct on its next attempt."""


class _Input(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SearchInput(_Input):
    query: str = Field(min_length=2, max_length=500)
    ticker: str | None = Field(default=None, max_length=10)
    fiscal_year: int | None = None


class SqlInput(_Input):
    sql: str = Field(min_length=6, max_length=2000)


class CalculateInput(_Input):
    expression: str = Field(min_length=1, max_length=300)


class StatementInput(_Input):
    ticker: str = Field(min_length=1, max_length=10)
    fiscal_years: list[int] = Field(min_length=1, max_length=MAX_STATEMENT_YEARS)


INPUT_MODELS = {
    "financial_statement": StatementInput,
    "search_filings": SearchInput,
    "query_financials": SqlInput,
    "calculate": CalculateInput,
}


@dataclass
class Evidence:
    """Every source the agent has read, in the order shown to the user."""

    passages: list[dict] = field(default_factory=list)
    _positions: dict[str, int] = field(default_factory=dict)

    def add(self, passage: dict) -> None:
        if passage["id"] not in self._positions:
            self._positions[passage["id"]] = len(self.passages)
            self.passages.append(passage)

    def index_of(self, source: str) -> int | None:
        return self._positions.get(source)


@dataclass
class ToolResult:
    content: str | list[dict]
    summary: str


def _citable_result(source: str, title: str, units: list[str]) -> dict:
    return {
        "type": "search_result",
        "source": source,
        "title": title,
        "content": [{"type": "text", "text": unit} for unit in units],
        "citations": {"enabled": True},
    }


def search_filings(args: SearchInput, filing_ids: list[str], evidence: Evidence) -> ToolResult:
    selected = [f for f in store.list_filings() if f["id"] in filing_ids]
    if args.ticker:
        selected = [f for f in selected if f["ticker"] == args.ticker.upper()]
    if args.fiscal_year:
        selected = [f for f in selected if f["fiscal_year"] == args.fiscal_year]
    if not selected:
        raise ToolError("No selected filing matches that ticker and fiscal year.")

    passages = retrieval.search(args.query, [f["id"] for f in selected], top_k=SEARCH_RESULTS)
    if not passages:
        return ToolResult("No matching passages were found.", "No passages found")

    results = []
    for passage in passages:
        evidence.add({key: passage[key] for key in PUBLIC_PASSAGE_FIELDS})
        title = (
            f"{passage['company']} {passage['form']} fiscal {passage['fiscal_year']}: "
            f"{passage['section']}"
        )
        results.append(_citable_result(passage["id"], title, citable_units(passage)))
    return ToolResult(results, f"Found {len(passages)} passages")


PUBLIC_PASSAGE_FIELDS = (
    "id", "ticker", "company", "form", "fiscal_year", "section", "kind", "text", "url",
)


def _authorize(action, table, _column, _database, _trigger):
    if action == sqlite3.SQLITE_SELECT:
        return sqlite3.SQLITE_OK
    if action == sqlite3.SQLITE_READ and table == ALLOWED_SQL_TABLE:
        return sqlite3.SQLITE_OK
    if action == sqlite3.SQLITE_FUNCTION and table != "load_extension":
        return sqlite3.SQLITE_OK
    return sqlite3.SQLITE_DENY


def _authorize_selected(action, table, column, database, view):
    """Allow only the temporary view limited to selected companies, not the table behind it."""
    if action == sqlite3.SQLITE_READ:
        allowed = table == ALLOWED_SQL_TABLE and (database == "temp" or view == ALLOWED_SQL_TABLE)
        return sqlite3.SQLITE_OK if allowed else sqlite3.SQLITE_DENY
    return _authorize(action, table, column, database, view)


def run_readonly_sql(sql: str, tickers: list[str] | None = None) -> tuple[list[str], list[tuple], bool]:
    """Run model-written SQL read-only, on one table, with a row and time limit.

    When tickers are given, the query can only see those companies.
    """
    connection = sqlite3.connect(f"file:{quote(str(config.DB_PATH))}?mode=ro", uri=True)
    deadline = time.monotonic() + SQL_TIME_LIMIT_SECONDS
    connection.set_progress_handler(lambda: time.monotonic() > deadline, 1000)
    if tickers:
        # Views can't take bound parameters; tickers come from our own filings table.
        quoted = ",".join("'" + ticker.replace("'", "''") + "'" for ticker in tickers)
        connection.execute(
            f"CREATE TEMP VIEW {ALLOWED_SQL_TABLE} AS "
            f"SELECT * FROM main.{ALLOWED_SQL_TABLE} WHERE ticker IN ({quoted})"
        )
    connection.set_authorizer(_authorize_selected if tickers else _authorize)
    try:
        cursor = connection.execute(sql)
        if cursor.description is None:
            raise ToolError("Only SELECT statements are allowed.")
        rows = cursor.fetchmany(MAX_SQL_ROWS + 1)
        columns = [column[0] for column in cursor.description]
    except sqlite3.Error as error:
        raise ToolError(f"SQL error: {error}. Only SELECT on financial_facts is allowed.") from error
    finally:
        connection.close()
    return columns, rows[:MAX_SQL_ROWS], len(rows) > MAX_SQL_ROWS


def query_financials(args: SqlInput, evidence: Evidence, tickers: list[str] | None = None) -> ToolResult:
    columns, rows, truncated = run_readonly_sql(args.sql, tickers)
    if not rows:
        return ToolResult("The query returned no rows. Check the concept name.", "0 rows")
    if not {"concept", "label"} & {column.lower() for column in columns}:
        # Without these, an unrelated figure can be mistaken for the one asked about.
        raise ToolError("Add the concept and label columns to the SELECT so each value can be identified.")

    lines = [" | ".join(columns)]
    for row in rows:
        lines.append(" | ".join(_cell(value) for value in row))

    source = f"sql:{len(evidence.passages)}"
    evidence.add(
        {
            "id": source,
            "ticker": "",
            "company": "SEC XBRL data",
            "form": "XBRL",
            "fiscal_year": 0,
            "section": "Financial data reported to the SEC",
            "kind": "table",
            "text": "\n".join(lines),
            "url": _company_filings_url(columns, rows, tickers),
        }
    )
    content: list[dict] = [_citable_result(source, "SEC XBRL annual financial data", lines)]
    if truncated:
        content.append({"type": "text", "text": f"Only the first {MAX_SQL_ROWS} rows are shown."})
    return ToolResult(content, f"{len(rows)} rows{' (limited)' if truncated else ''}")


def _in_millions(value: float) -> str:
    return f"{value / 1_000_000:,.3f}".rstrip("0").rstrip(".")


def financial_statement(args: StatementInput, evidence: Evidence, tickers: list[str] | None = None) -> ToolResult:
    """One table of standard line items, so the model doesn't have to choose among raw concepts."""
    ticker = args.ticker.upper()
    if tickers and ticker not in tickers:
        raise ToolError(f"{ticker} is not one of the selected companies: {', '.join(tickers)}.")
    years = sorted(set(args.fiscal_years), reverse=True)
    concepts = sorted({name for names in LINE_ITEMS.values() for name in names})
    with store.connect() as db:
        rows = db.execute(
            f"""SELECT concept, fiscal_year, unit, value FROM financial_facts
                WHERE ticker = ? AND unit IN ('USD', 'INR')
                AND concept IN ({",".join("?" * len(concepts))})
                AND fiscal_year IN ({",".join("?" * len(years))})
                ORDER BY period_end""",
            [ticker, *concepts, *years],
        ).fetchall()
    if not rows:
        return ToolResult(f"No standard financial data for {ticker} in those years.", "0 line items")

    currency = "INR" if any(row["unit"] == "INR" for row in rows) else "USD"
    values = {(r["concept"], r["fiscal_year"]): r["value"] for r in rows if r["unit"] == currency}

    def pick(label: str, names: list[str], year: int) -> float | None:
        found = [values[(name, year)] for name in names if (name, year) in values]
        if not found:
            return None
        # Some companies also tag a partial revenue line; the total is always the largest.
        return max(found) if label.startswith("Revenue") else found[0]

    lines = [f"Line item ({currency} millions) | " + " | ".join(f"FY{year}" for year in years)]
    for label, names in LINE_ITEMS.items():
        picked = [pick(label, names, year) for year in years]
        if any(value is not None for value in picked):
            cells = ["not reported" if value is None else _in_millions(value) for value in picked]
            lines.append(f"{label} | " + " | ".join(cells))

    source = f"statement:{ticker}:{','.join(map(str, years))}"
    evidence.add(
        {
            "id": source,
            "ticker": ticker,
            "company": "SEC XBRL data",
            "form": "XBRL",
            "fiscal_year": years[0],
            "section": "Standard financial statement line items",
            "kind": "table",
            "text": "\n".join(lines),
            "url": _company_filings_url(["ticker"], [(ticker,)], tickers),
        }
    )
    return ToolResult(
        [_citable_result(source, f"{ticker} financial statement line items", lines)],
        f"{len(lines) - 1} line items",
    )


def _company_filings_url(columns: list[str], rows: list[tuple], tickers: list[str] | None) -> str:
    lowered = [column.lower() for column in columns]
    ticker = rows[0][lowered.index("ticker")] if "ticker" in lowered else (tickers[0] if tickers else None)
    if not ticker:
        return "https://www.sec.gov/edgar/search/"
    return f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={quote(str(ticker))}&type=10-K"


def _cell(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).replace("|", "/")


_BINARY = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
}
_FUNCTIONS = {"abs": abs, "round": round, "min": min, "max": max}


def _evaluate(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        value = _evaluate(node.operand)
        return -value if isinstance(node.op, ast.USub) else value
    if isinstance(node, ast.BinOp):
        left, right = _evaluate(node.left), _evaluate(node.right)
        if isinstance(node.op, ast.Pow):
            if abs(right) > MAX_EXPONENT:
                raise ToolError(f"Exponents are limited to {MAX_EXPONENT}.")
            return left**right
        if type(node.op) in _BINARY:
            return _BINARY[type(node.op)](left, right)
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _FUNCTIONS
        and not node.keywords
    ):
        return _FUNCTIONS[node.func.id](*[_evaluate(arg) for arg in node.args])
    raise ToolError("Only numbers, arithmetic, abs, round, min, and max are allowed.")


def calculate(args: CalculateInput) -> ToolResult:
    try:
        result = _evaluate(ast.parse(args.expression, mode="eval"))
    except SyntaxError as error:
        raise ToolError("Could not read that expression. Use plain numbers without commas.") from error
    except (ZeroDivisionError, OverflowError, TypeError, ValueError, RecursionError) as error:
        raise ToolError(f"Could not calculate that: {error}.") from error
    if isinstance(result, complex) or not math.isfinite(result) or abs(result) > MAX_NUMBER:
        raise ToolError("The result is out of range.")
    return ToolResult(str(result), f"= {result:,.6g}")
