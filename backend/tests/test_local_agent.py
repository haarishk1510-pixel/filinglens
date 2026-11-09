import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from filinglens import config, local_agent, store, tools, verification as v  # noqa: E402

TABLE = {"kind": "table", "text": "Metric | 2026 | 2025\nNet sales | 478,834 | 416,161\nNet income | 132,175 | 112,010"}


def check(answer, passages=(TABLE,), calculations=()):
    return v.check_answer(answer, list(passages), list(calculations))


@pytest.mark.parametrize(
    "answer",
    ["Net sales were $478.8 billion [1].", "Net sales were $478,834 million [1].", "Sales: $479 billion [1]."],
)
def test_reported_figures_match_even_when_rounded(answer):
    result = check(answer)
    assert (result["claims"], result["supported"]) == (1, 1)


def test_a_wrong_figure_is_flagged():
    result = check("Net sales were $500 billion [1].")
    assert result["supported"] == 0
    assert result["unsupported_sentences"] == ["Net sales were $500 billion ."]


def test_a_figure_cited_to_the_wrong_source_is_flagged():
    other = {"kind": "text", "text": "Nothing relevant."}
    assert check("Sales were $478.8 billion [2].", [TABLE, other])["supported"] == 0


def test_a_figure_with_no_citation_counts_as_uncited():
    result = check("Net sales were $478.8 billion.")
    assert result["uncited_sentences"] == 1


def test_calculated_percentages_are_accepted():
    assert check("Sales grew 15.1%.", calculations=[15.0598])["supported"] == 1
    assert check("Sales grew 15.1%.", calculations=[0.150598])["supported"] == 1


def test_years_and_small_counts_are_not_claims():
    assert check("In fiscal 2026 the 10-K listed 3 segments.")["claims"] == 0


def test_citation_after_the_full_stop_still_counts():
    assert check("Net sales were $478.8 billion. [1] Income rose.")["supported"] == 1


def test_best_unit_finds_the_row_with_the_figure():
    assert v.best_unit("Net income was $132.2 billion", TABLE).startswith("Net income")


def evidence_with(*passages):
    evidence = tools.Evidence()
    for number, passage in enumerate(passages):
        evidence.add({"id": str(number), **passage})
    return evidence


def test_stream_handles_citations_split_across_chunks():
    stream = local_agent.AnswerStream(evidence_with(TABLE))
    events = []
    for delta in ["Net sales were $478.8 billion [", "1", "]. Done"]:
        events += stream.feed(delta)
    events += stream.feed("", final=True)
    assert "".join(e["text"] for e in events if e["type"] == "text") == "Net sales were $478.8 billion . Done"
    citation = next(e for e in events if e["type"] == "citation")
    assert citation["passage_index"] == 0
    assert citation["cited_text"].startswith("Net sales")


def test_stream_drops_citations_to_sources_that_do_not_exist():
    stream = local_agent.AnswerStream(evidence_with(TABLE))
    events = stream.feed("Claim [9].", final=True)
    assert not [e for e in events if e["type"] == "citation"]
    assert stream.invalid_citations == 1


def test_stream_hides_thinking():
    stream = local_agent.AnswerStream(evidence_with(TABLE))
    events = stream.feed("<think>secret") + stream.feed(" plan</think>Answer", final=True)
    assert "".join(e["text"] for e in events) == "Answer"


@pytest.fixture
def database(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "test.db")
    store.save_facts("AAPL", [{
        "ticker": "AAPL", "company": "Apple Inc.", "concept": "NetIncomeLoss", "label": "Net income",
        "unit": "USD", "value": 132_175_000_000.0, "period_start": "2025-09-28",
        "period_end": "2026-09-26", "fiscal_year": 2026, "filed": "2026-11-03", "accession": "x",
    }])


def text_chunk(text, finish=None):
    return NS(usage=None, choices=[NS(delta=NS(content=text, tool_calls=None), finish_reason=finish)])


def tool_chunk(call_id, name, arguments):
    function = NS(name=name, arguments=json.dumps(arguments) if not isinstance(arguments, str) else arguments)
    call = NS(id=call_id, function=function)
    return NS(usage=None, choices=[NS(delta=NS(content=None, tool_calls=[call]), finish_reason="tool_calls")])


class FakeOpenAI:
    def __init__(self, turns):
        self.turns = list(turns)
        self.requests = []
        self.chat = NS(completions=NS(create=self._create))

    def _create(self, **request):
        self.requests.append({**request, "messages": list(request["messages"])})
        return iter(self.turns.pop(0))


SQL = "SELECT concept, fiscal_year, value FROM financial_facts"


def test_agent_uses_tools_then_verifies_its_answer(database):
    client = FakeOpenAI([
        [tool_chunk("a", "query_financials", {"sql": SQL})],
        [tool_chunk("b", "calculate", {"expression": "132175/1000"})],
        [text_chunk("Net income was $132.2 billion [1]."), text_chunk("", "stop")],
    ])
    events = list(local_agent.stream_local_agent("Net income?", ["f"], client=client))
    assert [e["status"] for e in events if e["type"] == "step"] == ["running", "done", "running", "done"]
    done = events[-1]
    assert done["citation_check"]["claims"] == done["citation_check"]["supported"] == 1
    assert done["steps"] == 2
    assert next(e for e in events if e["type"] == "citation")["cited_text"].endswith("132175000000")
    tool_message = client.requests[1]["messages"][-1]
    assert tool_message["role"] == "tool" and tool_message["content"].startswith("Source [1]")


def test_agent_recovers_from_bad_tool_arguments(database):
    client = FakeOpenAI([
        [tool_chunk("a", "query_financials", "{not json")],
        [tool_chunk("b", "query_financials", {"sql": SQL})],
        [text_chunk("Done [1].", "stop")],
    ])
    events = list(local_agent.stream_local_agent("q?", ["f"], client=client))
    statuses = [e["status"] for e in events if e["type"] == "step" and e["status"] != "running"]
    assert statuses == ["error", "done"]
    assert client.requests[1]["messages"][-1]["content"].startswith("Error:")


def test_agent_is_nudged_when_it_answers_without_evidence(database):
    client = FakeOpenAI([
        [text_chunk("I already know this: $1 trillion.", "stop")],
        [tool_chunk("a", "query_financials", {"sql": SQL})],
        [text_chunk("Net income was $132.2 billion [1].", "stop")],
    ])
    events = list(local_agent.stream_local_agent("q?", ["f"], client=client))
    text = "".join(e["text"] for e in events if e["type"] == "text")
    assert "trillion" not in text
    assert client.requests[1]["messages"][-1] == {"role": "user", "content": local_agent.NUDGE}


SOURCE = {"text": "Net sales | 478,834 | 416,161"}


def test_formula_is_computed_by_code_and_counts_as_verified():
    calculations = []
    text = local_agent.compute_formulas("Grew <<(478834-416161)/416161*100>>%.", [SOURCE], calculations)
    assert text == "Grew 15.06%."
    assert len(calculations) == 1


def test_formula_accepts_commas_and_dollar_signs():
    text = local_agent.compute_formulas("Gap <<$478,834-$416,161>>.", [SOURCE], [])
    assert text == "Gap 62,673."


def test_formula_with_invented_inputs_is_computed_but_not_verified():
    calculations = []
    assert local_agent.compute_formulas("<<999999/2>>", [SOURCE], calculations) == "499,999.5"
    assert calculations == []


def test_number_repeating_a_formula_result_is_removed():
    assert local_agent.compute_formulas("Ratio <<33/50>>0.66 [1].", [SOURCE], []) == "Ratio 0.66 [1]."
    assert local_agent.compute_formulas("Grew <<10/5>> = 2%.", [SOURCE], []) == "Grew 2%."


def test_number_after_a_formula_is_kept_when_it_differs():
    assert local_agent.compute_formulas("Then <<10/5>> 40 items.", [SOURCE], []) == "Then 2 40 items."


def test_unsafe_formula_is_not_run():
    text = local_agent.compute_formulas("<<__import__('os').getcwd()>>", [SOURCE], [])
    assert text == "(this figure could not be calculated)"


def test_agent_answer_shows_computed_figure_and_passes_the_check(database):
    client = FakeOpenAI([
        [tool_chunk("a", "query_financials", {"sql": SQL})],
        [text_chunk("That plus one is $<<132175000000/1000000000+1>> billion [1].", "stop")],
    ])
    events = list(local_agent.stream_local_agent("q?", ["f"], client=client))
    text = "".join(e["text"] for e in events if e["type"] == "text")
    assert "133.1" in text and "<<" not in text
    assert events[-1]["citation_check"]["unsupported_sentences"] == []


def test_agent_gives_up_cleanly_when_it_never_finds_evidence(database):
    client = FakeOpenAI([[text_chunk("Guess one.", "stop")], [text_chunk("Guess two.", "stop")]])
    events = list(local_agent.stream_local_agent("q?", ["f"], client=client))
    assert any(e["type"] == "notice" for e in events)
    assert not [e for e in events if e["type"] == "text"]


def test_agent_must_stop_calling_tools_at_the_turn_limit(database):
    call = [tool_chunk("x", "calculate", {"expression": "1+1"})]
    turns = [list(call) for _ in range(local_agent.MAX_MODEL_TURNS - 1)]
    client = FakeOpenAI([*turns, [text_chunk("Done.", "stop")]])
    list(local_agent.stream_local_agent("q?", ["f"], client=client))
    assert len(client.requests) == local_agent.MAX_MODEL_TURNS
    assert client.requests[-1]["tool_choice"] == "none"
