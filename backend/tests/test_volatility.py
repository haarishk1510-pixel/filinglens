import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml import volatility as v  # noqa: E402

DAYS = 400
EVENT = 320


@pytest.fixture
def market_data():
    index = pd.bdate_range("2020-01-01", periods=DAYS)
    rng = np.random.default_rng(1)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, DAYS)))
    prices = pd.DataFrame(
        {"Close": close, "High": close * 1.01, "Low": close * 0.99, "Volume": rng.integers(1000, 2000, DAYS).astype(float)},
        index=index,
    )
    market = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.005, DAYS))), index=index)
    vix = pd.Series(rng.uniform(12, 30, DAYS), index=index)
    return prices, market, vix


def test_features_cannot_see_the_reaction_day_or_later(market_data):
    prices, market, vix = market_data
    before = v.compute_market_features(prices, market, vix, EVENT)

    changed = prices.copy()
    changed.iloc[EVENT:] *= 3
    changed_market, changed_vix = market.copy(), vix.copy()
    changed_market.iloc[EVENT:] *= 3
    changed_vix.iloc[EVENT:] *= 3
    assert v.compute_market_features(changed, changed_market, changed_vix, EVENT) == before


def test_features_do_change_with_the_past(market_data):
    prices, market, vix = market_data
    before = v.compute_market_features(prices, market, vix, EVENT)
    changed = prices.copy()
    changed.iloc[EVENT - 1, changed.columns.get_loc("Close")] *= 1.5
    assert v.compute_market_features(changed, market, vix, EVENT) != before


def test_label_rises_when_prices_swing_after_the_event(market_data):
    prices = market_data[0]
    calm = v.volatility_ratio(prices, EVENT)
    swings = prices.copy()
    moves = np.array([1.1, 0.9, 1.1, 0.9, 1.1])
    swings.iloc[EVENT : EVENT + 5, swings.columns.get_loc("Close")] *= moves
    assert v.volatility_ratio(swings, EVENT) > calm * 3


def test_label_needs_enough_future_days(market_data):
    assert v.volatility_ratio(market_data[0], DAYS - 2) is None


def test_after_close_release_counts_toward_next_session():
    days = pd.bdate_range("2024-01-01", periods=10)  # 2024-01-01 is a Monday
    assert days[v.reaction_index(days, "2024-01-03T16:05:00.000Z")] == pd.Timestamp("2024-01-04")
    assert days[v.reaction_index(days, "2024-01-03T07:00:00.000Z")] == pd.Timestamp("2024-01-03")
    assert days[v.reaction_index(days, "2024-01-05T17:00:00.000Z")] == pd.Timestamp("2024-01-08")


def test_history_features_use_only_earlier_events():
    events = pd.DataFrame(
        {
            "ticker": ["A", "A", "A", "B"],
            "event_date": pd.to_datetime(["2020-01-01", "2020-04-01", "2020-07-01", "2020-01-01"]),
            "ratio": [2.0, 1.0, 9.0, 3.0],
        }
    )
    result = v.add_history_features(events)
    a = result[result["ticker"] == "A"].reset_index(drop=True)
    assert np.isnan(a.loc[0, "prior_ratio_last"])
    assert a.loc[1, "prior_ratio_last"] == 2.0
    assert a.loc[2, "prior_ratio_mean_4"] == 1.5
    assert a.loc[2, "prior_high_rate_8"] == 0.5
    assert np.isnan(result[result["ticker"] == "B"]["prior_ratio_last"].iloc[0])


def test_split_is_chronological_with_a_gap():
    dates = pd.date_range("2021-01-01", "2024-12-31", freq="D")
    splits = v.time_split(pd.DataFrame({"event_date": dates}))
    assert splits["train"]["event_date"].max() < pd.Timestamp(v.TRAIN_END) - pd.Timedelta(days=v.EMBARGO_DAYS - 1)
    assert splits["validation"]["event_date"].min() >= pd.Timestamp(v.TRAIN_END)
    assert splits["test"]["event_date"].min() >= pd.Timestamp(v.VALIDATION_END)
    assert splits["validation"]["event_date"].max() < splits["test"]["event_date"].min()
