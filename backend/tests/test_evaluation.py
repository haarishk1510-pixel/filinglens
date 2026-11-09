import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from filinglens import evaluation as ev  # noqa: E402


def chunk(cid, text, ticker="AAPL", section="Item 7. MD&A"):
    return {"id": cid, "ticker": ticker, "section": section, "text": text}


def fact(concept="NetIncomeLoss", value=478_834_000_000.0, year=2026, unit="USD", ticker="AAPL"):
    return {"concept": concept, "value": value, "fiscal_year": year, "unit": unit, "ticker": ticker}


def test_figure_pattern_matches_whole_numbers_only():
    pattern = ev.figure_pattern(478_834_000_000)
    assert pattern.search("Total | $478,834 | 416,161")
    assert not pattern.search("1,478,834")
    assert not pattern.search("478,8340")
    assert not pattern.search("478,834,123")
    assert not pattern.search("478,834.5")


def test_figure_labels_are_scoped_to_the_right_company():
    chunks = [chunk("a", "Net income | 478,834"), chunk("m", "Other | 478,834", ticker="MSFT")]
    questions, skipped = ev.build_figure_questions(chunks, [fact()])
    assert questions[0].gold_ids == {"a"}
    assert not skipped


def test_figures_that_cannot_be_labeled_are_reported_not_dropped_silently():
    chunks = [chunk("a", "nothing here")]
    questions, skipped = ev.build_figure_questions(chunks, [fact(), fact(value=5_000_000.0, concept="Assets")])
    assert questions == []
    assert len(skipped) == 2


def test_narrative_rule_needs_every_pattern_and_the_right_section():
    chunks = [
        chunk("yes", "Epic Games filed suit", section="Item 3. Legal"),
        chunk("no-section", "Epic Games filed suit", section="Item 8. Notes"),
        chunk("no-pattern", "Games filed suit", section="Item 3. Legal"),
    ]
    spec = {"ticker": "AAPL", "section": "Item 3", "patterns": ["epic", "suit"]}
    assert ev.narrative_gold(spec, chunks) == {"yes"}


def test_rank_metrics():
    assert ev.first_hit_rank(["x", "g", "h"], {"g", "h"}) == 2
    assert ev.first_hit_rank(["x"], {"g"}) is None
    ranks = [1, 4, None, 2]
    assert ev.hit_at(ranks, 1).mean() == 0.25
    assert ev.hit_at(ranks, 3).mean() == 0.5
    assert ev.reciprocal_ranks(ranks).tolist() == [1.0, 0.25, 0.0, 0.5]


def test_bootstrap_interval_is_tight_for_constant_values_and_contains_mean():
    assert ev.bootstrap_interval(np.ones(10)) == (1.0, 1.0)
    values = np.array([0.0, 1.0] * 50)
    low, high = ev.bootstrap_interval(values)
    assert low < values.mean() < high
