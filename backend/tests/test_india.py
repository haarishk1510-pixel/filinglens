import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from filinglens import facts, parsing, pdf_reports as p, verification as v  # noqa: E402

LINES = [
    "Consolidated Financial Statements", "Reliance Industries Limited", "Report", "(₹ in crore)", "Notes",
    "2025-26", "2024-25", "Income", "Revenue from Operations", "25", "10,75,675", "9,80,136",
    "Other Income", "26", "28,962", "17,978", "Changes in Inventories", "27", "(10,267)", "(15,124)",
    "Profit for the Year", "1,04,000", "90,000", "Total Assets", "20,00,000", "18,00,000",
]


def test_rows_pair_labels_with_numbers_and_drop_note_references():
    rows = dict(p.rebuild_rows(LINES))
    assert rows["Revenue from Operations"] == ["10,75,675", "9,80,136"]
    assert rows["Changes in Inventories"] == ["(10,267)", "(15,124)"]


def test_table_page_keeps_rows_and_page_number():
    chunk = p.page_chunks(LINES, 101)[0]
    assert (chunk.section, chunk.kind) == ("Page 101", "table")
    assert "Revenue from Operations | 10,75,675 | 9,80,136" in chunk.text


def test_text_page_is_split_into_passages():
    lines = ["word " * 900]
    chunks = p.page_chunks(lines, 7)
    assert len(chunks) >= 2 and all(c.kind == "text" and c.section == "Page 7" for c in chunks)


def test_rupee_symbol_is_restored():
    assert p.clean_lines("(I in crore)\nRevenue") == ["(₹ in crore)", "Revenue"]


def test_consolidated_facts_become_rupees_for_each_year():
    rows = p.extract_consolidated_facts([LINES], "RELIANCE", "RIL", 2026, "2026-05-29", "x")
    values = {(r["concept"], r["fiscal_year"]): r["value"] for r in rows}
    assert values[("Revenue", 2026)] == 1075675 * p.CRORE
    assert values[("Revenue", 2025)] == 980136 * p.CRORE
    assert values[("Assets", 2026)] == 2000000 * p.CRORE
    assert all(r["unit"] == "INR" for r in rows)


def test_standalone_pages_are_ignored():
    standalone = ["Standalone Financial Statements", *LINES[1:]]
    assert p.extract_consolidated_facts([standalone], "R", "R", 2026, "d", "x") == []


TABLE = {"kind": "table", "text": "Revenue from Operations | 10,75,675 | 9,80,136"}


def test_verifier_understands_crore_lakh_and_rupee_sign():
    for answer in (
        "Revenue was ₹10,75,675 crore [1].",
        "Revenue was ₹10.76 lakh crore [1].".replace("10.76 lakh crore", "1075675 cr"),
    ):
        assert v.check_answer(answer, [TABLE], [])["supported"] == 1


def test_verifier_flags_wrong_rupee_figure():
    assert v.check_answer("Revenue was ₹12,00,000 crore [1].", [TABLE], [])["supported"] == 0


def test_ifrs_and_20f_facts_are_included_in_each_currency():
    entry = {"start": "2025-04-01", "end": "2026-03-31", "form": "20-F", "filed": "2026-06-01", "accn": "a"}
    payload = {
        "facts": {
            "ifrs-full": {
                "Revenue": {"label": "Revenue", "units": {"INR": [entry | {"val": 100}], "USD": [entry | {"val": 2}]}}
            }
        }
    }
    rows = facts.extract_annual_facts(payload, "WIT", "Wipro")
    assert {(r["unit"], r["value"], r["fiscal_year"]) for r in rows} == {("INR", 100, 2026), ("USD", 2, 2026)}


def test_20f_sections_are_named():
    blocks = [parsing.Block("text", "ITEM 3. KEY INFORMATION"), parsing.Block("text", "Risk text.")]
    chunks = parsing.blocks_to_chunks(blocks, parsing.SECTION_NAMES_20F)
    assert chunks[0].section == "Item 3. Key Information and Risk Factors"
