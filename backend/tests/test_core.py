import sys
from pathlib import Path
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from filinglens import answer, parsing  # noqa: E402
from filinglens.retrieval import reciprocal_rank_fusion  # noqa: E402

FILING = """
<html><body>
<ix:header>hidden metadata</ix:header>
<table><tr><td>Item 1.</td><td>Business</td></tr><tr><td>Item 1A.</td><td>Risk Factors</td></tr>
<tr><td>Item 7.</td><td>MD&amp;A</td></tr></table>
<div>Item 1.    Business</div>
<div>We make <span>wid</span><span>gets</span>.</div>
<div>See Item 1A of this report for details.</div>
<div>Item 1A of this report discusses risk.</div>
<table><tr><td>Item 7.</td><td>Discussion</td></tr></table>
<table>
<tr><th>Metric</th><th>2026</th></tr>
<tr><td>Revenue</td><td>$</td><td>100</td></tr>
<tr><td>Loss</td><td>(</td><td>5</td><td>)</td></tr>
</table>
</body></html>
"""


def test_sections_ignore_contents_table_and_cross_references():
    chunks = parsing.parse_filing(FILING)
    business = [c for c in chunks if c.section == "Item 1. Business"]
    assert len(business) == 1
    assert "We make widgets." in business[0].text
    assert "Item 1A of this report discusses risk." in business[0].text
    assert not any(c.section.startswith("Item 1A") for c in chunks)
    assert "hidden metadata" not in " ".join(c.text for c in chunks)


def test_table_keeps_values_with_currency_and_closing_bracket():
    table = next(c for c in parsing.parse_filing(FILING) if "Revenue" in c.text)
    assert table.section == "Item 7. Management's Discussion and Analysis"
    assert "Revenue | $100" in table.text
    assert "5)" in table.text


def table_chunk_after(*text_blocks):
    blocks = [parsing.Block("text", text) for text in text_blocks]
    blocks.append(parsing.Block("table", "Revenue | 100"))
    return next(c for c in parsing.blocks_to_chunks(blocks) if c.kind == "table")


def test_table_gets_the_short_headings_above_it():
    chunk = table_chunk_after(
        "This long paragraph explains results.", "CONSOLIDATED STATEMENTS OF OPERATIONS", "(In millions)"
    )
    assert chunk.text == "Table: CONSOLIDATED STATEMENTS OF OPERATIONS (In millions)\nRevenue | 100"


def test_table_gets_its_lead_in_sentence():
    chunk = table_chunk_after("Gross margin for 2026 and 2025 were as follows (dollars in millions):")
    assert chunk.text.startswith("Table: Gross margin for 2026")


def test_table_ignores_ordinary_sentences_and_page_numbers():
    assert table_chunk_after("Revenue grew because demand rose.", "42").text == "Revenue | 100"


def test_every_piece_of_a_split_table_keeps_its_title():
    rows = "\n".join(f"Row {i} | one two three four five six seven eight" for i in range(200))
    blocks = [parsing.Block("text", "BALANCE SHEETS"), parsing.Block("table", f"Metric | Value\n{rows}")]
    pieces = [c for c in parsing.blocks_to_chunks(blocks) if c.kind == "table"]
    assert len(pieces) > 1
    assert all(p.text.startswith("Table: BALANCE SHEETS\nMetric | Value\n") for p in pieces)


def test_long_table_repeats_header():
    rows = "\n".join(f"Row {i} | one two three four five six seven eight" for i in range(200))
    pieces = parsing._split_table(f"Metric | Value\n{rows}")
    assert len(pieces) > 1
    assert all(piece.startswith("Metric | Value\n") for piece in pieces)


def test_long_paragraph_is_split_without_losing_words():
    words = [f"w{i}" for i in range(1000)]
    chunks = parsing._chunk_paragraphs([" ".join(words)])
    assert all(len(c.split()) <= parsing.MAX_CHUNK_WORDS for c in chunks)
    assert set(words) == set(" ".join(chunks).replace("\n", " ").split())


def test_rrf_rewards_agreement_between_rankings():
    assert reciprocal_rank_fusion([[1, 2, 3], [3, 2, 9]])[0] in {2, 3}
    assert reciprocal_rank_fusion([[1, 2], [2, 1], [2, 5]])[0] == 2


PASSAGE = {
    "company": "Apple Inc.", "form": "10-K", "fiscal_year": 2026,
    "section": "Item 8. Financial Statements", "text": "Total net sales | 478,834",
}


class FakeStream:
    def __init__(self, stop_reason="end_turn"):
        self.stop_reason = stop_reason

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def __iter__(self):
        yield NS(type="message_start")
        yield NS(type="content_block_delta", delta=NS(type="text_delta", text="Net sales were $478.8B."))
        yield NS(type="content_block_delta", delta=NS(
            type="citations_delta",
            citation=NS(document_index=0, cited_text="Total net sales | 478,834"),
        ))

    def get_final_message(self):
        return NS(stop_reason=self.stop_reason, model="claude-opus-5-5",
                  usage=NS(input_tokens=1000, output_tokens=100))


class FakeClient:
    def __init__(self, stop_reason="end_turn"):
        self.request = None
        self.stop_reason = stop_reason
        self.messages = NS(stream=self._stream)
        self.beta = NS(messages=NS(stream=self._stream))

    def _stream(self, **request):
        self.request = request
        return FakeStream(self.stop_reason)


def test_each_passage_is_a_citable_document():
    client = FakeClient()
    events = list(answer.stream_answer("What were net sales?", [PASSAGE], client=client))
    content = client.request["messages"][0]["content"]
    assert content[0]["citations"] == {"enabled": True}
    assert content[0]["source"]["content"][0]["text"] == PASSAGE["text"]
    assert content[-1] == {"type": "text", "text": "What were net sales?"}
    assert [e["type"] for e in events] == ["text", "citation", "done"]
    assert events[1]["passage_index"] == 0
    assert events[2]["cost_usd"] == (1000 * 4 + 100 * 20) / 1_000_000


def test_citable_units_are_rows_and_sentences():
    table = {"kind": "table", "text": "Metric | 2026\nRevenue | $100"}
    assert answer.citable_units(table) == ["Metric | 2026", "Revenue | $100"]
    text = {"kind": "text", "text": "Sales rose 5.2% in 2026. Costs fell.\n\nU.S. demand held."}
    assert answer.citable_units(text) == ["Sales rose 5.2% in 2026.", "Costs fell.", "U.S. demand held."]


def test_refusal_is_reported_to_the_user():
    events = list(answer.stream_answer("q?", [PASSAGE], client=FakeClient("refusal")))
    assert any(e["type"] == "notice" for e in events)
