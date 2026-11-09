import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from filinglens import financebench as fb  # noqa: E402


@pytest.mark.parametrize(
    ("answer", "value"),
    [("$1577.00", 1577.0), ("1.9%", 1.9), ("-0.02", 0.02), ("0", 0.0), ("Yes, it did", None), ("$1 to $2", None)],
)
def test_gold_value(answer, value):
    assert fb.gold_value(answer) == value


@pytest.mark.parametrize(
    "answer",
    ["Capex was $1,577 million.", "Capex was $1.577 billion.", "It was 1577.", "$1,577,000,000 in total."],
)
def test_equivalent_forms_of_a_figure_are_correct(answer):
    assert fb.grade(answer, 1577.0) == {"strict": True, "lenient": True}


def test_ratio_and_percent_forms_match():
    assert fb.grade("The margin was 0.019.", 1.9)["strict"]
    assert fb.grade("The margin was 1.9%.", 1.9)["strict"]


def test_wrong_numbers_and_empty_answers_fail():
    assert fb.grade("Capex was $1,700 million.", 1577.0) == {"strict": False, "lenient": False}
    assert fb.grade("", 1577.0) == {"strict": False, "lenient": False}


def test_small_rounding_passes_but_large_error_does_not():
    assert fb.grade("Ratio 24.3", 24.26)["strict"]
    assert not fb.grade("Ratio 25", 24.26)["strict"]


def test_strict_needs_the_first_number_and_skips_years():
    answer = "In FY2018 and 2019 revenue was 900, while capex was $1,577."
    assert fb.grade(answer, 1577.0) == {"strict": False, "lenient": True}
    assert fb.grade("In 2018 capex was 1577.", 1577.0)["strict"]


def record(identifier, doc="3M_2018_10K", answer="$1.00"):
    return {"financebench_id": identifier, "doc_name": doc, "answer": answer, "question": "q"}


DOCUMENTS = {
    "3M_2018_10K": {"doc_type": "10k", "doc_period": 2018},
    "3M_2018_10Q": {"doc_type": "10q", "doc_period": 2018},
    "UNKNOWN_2018_10K": {"doc_type": "10k", "doc_period": 2018},
}


def test_selection_keeps_only_gradable_mapped_10k_questions():
    records = [
        record("keep"), record("quarter", doc="3M_2018_10Q"), record("unmapped", doc="UNKNOWN_2018_10K"),
        record("text", answer="Yes"), record("zero", answer="0"),
    ]
    selected = fb.select_questions(records, DOCUMENTS, 10)
    assert [q.id for q in selected] == ["keep"]
    assert (selected[0].ticker, selected[0].fiscal_year) == ("MMM", 2018)


def test_selection_is_repeatable_and_ignores_input_order():
    records = [record(f"q{i}") for i in range(30)]
    first = [q.id for q in fb.select_questions(records, DOCUMENTS, 5)]
    assert first == [q.id for q in fb.select_questions(list(reversed(records)), DOCUMENTS, 5)]
    assert len(first) == 5
