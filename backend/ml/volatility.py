"""Predict whether a stock turns unusually volatile after an earnings release.

  python ml/volatility.py build   download data and create the event table
  python ml/volatility.py train   train, compare with baselines, and save results

The label is whether realized volatility over the five trading days from the reaction
day exceeds 1.5 times the prior 60-day level. This is a research exercise about risk,
not a trading signal or investment advice.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

from filinglens import config, edgar  # noqa: E402

ML_DATA = config.DATA_DIR / "ml"
EVENTS_PATH = ML_DATA / "volatility_events.csv"
MODEL_PATH = BACKEND / "models" / "volatility_xgb.json"
RESULTS_PATH = BACKEND / "eval" / "results" / "volatility.json"

START = "2012-01-01"
TRAIN_END = "2023-01-01"
VALIDATION_END = "2024-01-01"
EMBARGO_DAYS = 10
POST_DAYS = 5
BASELINE_DAYS = 60
MIN_HISTORY_DAYS = 260
MIN_EVENT_SPACING_DAYS = 20
HIGH_VOL_RATIO = 1.5
MARKET_CLOSE_HOUR = 16
BOOTSTRAP_SAMPLES = 1000

# Large US companies across sectors. Choosing today's large companies adds survivorship bias.
UNIVERSE = """AAPL MSFT NVDA GOOGL AMZN META TSLA AVGO ORCL CRM ADBE CSCO INTC AMD QCOM TXN IBM NFLX
JPM BAC WFC C GS MS AXP BLK USB PNC V MA
JNJ UNH PFE MRK ABBV LLY TMO ABT BMY AMGN GILD CVS MDT
WMT COST PG KO PEP HD MCD NKE SBUX LOW TGT MDLZ CL GIS
XOM CVX COP SLB EOG
BA CAT GE HON UPS FDX LMT RTX MMM DE UNP
DIS CMCSA T VZ NEE DUK SO AMT""".split()

TECH_FEATURES = [
    "vol_20", "vol_60", "vol_252", "vol_20_over_60", "vol_60_over_252",
    "return_5", "return_20", "return_60", "excess_return_20",
    "volume_20_over_60", "range_20", "max_abs_move_20",
    "market_vol_20", "vix_level", "vix_change_20",
]
HISTORY_FEATURES = ["prior_ratio_mean_4", "prior_ratio_last", "prior_high_rate_8", "days_since_prior"]
FEATURES = TECH_FEATURES + HISTORY_FEATURES


def realized_vol(returns) -> float:
    return float(np.sqrt(np.mean(np.square(returns))))


def compute_market_features(prices: pd.DataFrame, market: pd.Series, vix: pd.Series, event_index: int) -> dict:
    """Features using only data through the close before the reaction day."""
    past = prices.iloc[:event_index]
    close, volume = past["Close"], past["Volume"]
    returns = np.log(close).diff().dropna()
    last_day = past.index[-1]
    market_past = market.loc[:last_day]
    vix_past = vix.loc[:last_day]

    vol_20, vol_60, vol_252 = (realized_vol(returns.iloc[-n:]) for n in (20, 60, 252))
    market_returns = np.log(market_past).diff().dropna()
    return_20 = float(close.iloc[-1] / close.iloc[-21] - 1)
    return {
        "vol_20": vol_20,
        "vol_60": vol_60,
        "vol_252": vol_252,
        "vol_20_over_60": vol_20 / vol_60,
        "vol_60_over_252": vol_60 / vol_252,
        "return_5": float(close.iloc[-1] / close.iloc[-6] - 1),
        "return_20": return_20,
        "return_60": float(close.iloc[-1] / close.iloc[-61] - 1),
        "excess_return_20": return_20 - float(market_past.iloc[-1] / market_past.iloc[-21] - 1),
        "volume_20_over_60": float(volume.iloc[-20:].mean() / volume.iloc[-60:].mean()),
        "range_20": float(((past["High"] - past["Low"]) / close).iloc[-20:].mean()),
        "max_abs_move_20": float(returns.iloc[-20:].abs().max()),
        "market_vol_20": realized_vol(market_returns.iloc[-20:]),
        "vix_level": float(vix_past.iloc[-1]),
        "vix_change_20": float(vix_past.iloc[-1] / vix_past.iloc[-21] - 1),
    }


def volatility_ratio(prices: pd.DataFrame, event_index: int) -> float | None:
    """Realized volatility after the event, relative to the 60 days before it."""
    if event_index + POST_DAYS > len(prices):
        return None
    returns = np.log(prices["Close"]).diff()
    before = returns.iloc[event_index - BASELINE_DAYS : event_index]
    after = returns.iloc[event_index : event_index + POST_DAYS]
    baseline = realized_vol(before)
    return realized_vol(after) / baseline if baseline > 0 else None


def reaction_index(trading_days: pd.DatetimeIndex, accepted_at: str) -> int | None:
    """The first session able to react: after-close releases count toward the next day."""
    # The SEC labels this timestamp UTC, but it is Eastern time.
    moment = pd.Timestamp(accepted_at[:19])
    day = moment.normalize()
    if moment.hour >= MARKET_CLOSE_HOUR:
        day += pd.Timedelta(days=1)
    position = int(trading_days.searchsorted(day))
    return position if position < len(trading_days) else None


def earnings_releases(ticker: str) -> list[str]:
    """Timestamps of SEC 8-K filings under Item 2.02, results of operations."""
    cik = edgar.lookup_cik(ticker)[0]
    _, tables = edgar._filing_tables(cik)
    releases = []
    for table in tables:
        for form, items, accepted in zip(table["form"], table["items"], table["acceptanceDateTime"]):
            if form == "8-K" and "2.02" in items.split(",") and accepted[:10] >= START:
                releases.append(accepted)
    return sorted(releases)


def add_history_features(events: pd.DataFrame) -> pd.DataFrame:
    """Each event sees only the outcomes of that company's earlier events."""
    events = events.sort_values(["ticker", "event_date"]).reset_index(drop=True)
    grouped = events.groupby("ticker")
    prior_ratio = grouped["ratio"].shift(1)
    events["prior_ratio_last"] = prior_ratio
    events["prior_ratio_mean_4"] = prior_ratio.groupby(events["ticker"]).transform(
        lambda s: s.rolling(4, min_periods=1).mean()
    )
    prior_high = (prior_ratio > HIGH_VOL_RATIO).where(prior_ratio.notna())
    events["prior_high_rate_8"] = prior_high.astype(float).groupby(events["ticker"]).transform(
        lambda s: s.rolling(8, min_periods=1).mean()
    )
    events["days_since_prior"] = grouped["event_date"].diff().dt.days
    return events


def build() -> None:
    import yfinance as yf

    ML_DATA.mkdir(parents=True, exist_ok=True)
    raw = yf.download(
        [*UNIVERSE, "SPY", "^VIX"], start=START, auto_adjust=True, progress=False, group_by="ticker", threads=True
    )
    market, vix = raw["SPY"]["Close"].dropna(), raw["^VIX"]["Close"].dropna()

    rows, skipped = [], {}
    for ticker in UNIVERSE:
        prices = raw[ticker].dropna(subset=["Close"])
        try:
            releases = earnings_releases(ticker)
        except Exception as error:  # one company's data problem shouldn't stop the rest
            skipped[ticker] = repr(error)
            continue
        previous = -MIN_EVENT_SPACING_DAYS
        for accepted in releases:
            index = reaction_index(prices.index, accepted)
            if index is None or index < MIN_HISTORY_DAYS or index - previous < MIN_EVENT_SPACING_DAYS:
                continue
            ratio = volatility_ratio(prices, index)
            if ratio is None:
                continue
            previous = index
            rows.append(
                {
                    "ticker": ticker,
                    "event_date": prices.index[index],
                    "ratio": ratio,
                    **compute_market_features(prices, market, vix, index),
                }
            )
        print(ticker, len(rows), flush=True)

    events = add_history_features(pd.DataFrame(rows))
    events["high_vol"] = (events["ratio"] > HIGH_VOL_RATIO).astype(int)
    events.to_csv(EVENTS_PATH, index=False)
    print(f"{len(events)} events, base rate {events['high_vol'].mean():.3f}; skipped: {skipped}")


def time_split(events: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Train on the past, test on the future, with a gap so labels cannot overlap."""
    train_cutoff = pd.Timestamp(TRAIN_END) - pd.Timedelta(days=EMBARGO_DAYS)
    validation_cutoff = pd.Timestamp(VALIDATION_END) - pd.Timedelta(days=EMBARGO_DAYS)
    dates = events["event_date"]
    return {
        "train": events[dates < train_cutoff],
        "validation": events[(dates >= TRAIN_END) & (dates < validation_cutoff)],
        "test": events[dates >= VALIDATION_END],
    }


def metrics(y, score, months) -> dict:
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

    y, score, months = np.asarray(y), np.clip(np.asarray(score), 1e-6, 1 - 1e-6), np.asarray(months)
    rng = np.random.default_rng(0)
    unique = np.unique(months)
    positions = {month: np.flatnonzero(months == month) for month in unique}
    samples = []
    for _ in range(BOOTSTRAP_SAMPLES):  # resample whole months, since events in a month are related
        picked = np.concatenate([positions[m] for m in rng.choice(unique, len(unique))])
        if 0 < y[picked].sum() < len(picked):
            samples.append(roc_auc_score(y[picked], score[picked]))
    return {
        "roc_auc": float(roc_auc_score(y, score)),
        "roc_auc_ci95": [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))],
        "pr_auc": float(average_precision_score(y, score)),
        "brier": float(brier_score_loss(y, score)),
    }


def train() -> None:
    import mlflow
    import xgboost as xgb
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    events = pd.read_csv(EVENTS_PATH, parse_dates=["event_date"])
    splits = time_split(events)
    y = {name: frame["high_vol"] for name, frame in splits.items()}
    x = {name: frame[FEATURES] for name, frame in splits.items()}
    test_months = splits["test"]["event_date"].dt.to_period("M").astype(str)

    linear_columns = ["vol_20_over_60", "vol_60_over_252", "prior_high_rate_8", "prior_ratio_mean_4"]
    linear = make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression(max_iter=1000))
    linear.fit(x["train"][linear_columns], y["train"])

    model = xgb.XGBClassifier(
        n_estimators=1500, learning_rate=0.02, max_depth=3, subsample=0.8, colsample_bytree=0.8,
        min_child_weight=20, reg_lambda=5.0, eval_metric="logloss", early_stopping_rounds=75, random_state=0,
    )
    model.fit(x["train"], y["train"], eval_set=[(x["validation"], y["validation"])], verbose=False)

    base_rate = float(y["train"].mean())
    prior_rate = splits["test"]["prior_high_rate_8"].fillna(base_rate)
    scores = {
        "training_base_rate": np.full(len(y["test"]), base_rate) + np.random.default_rng(0).normal(0, 1e-9, len(y["test"])),
        "company_prior_rate": prior_rate,
        "logistic_regression": linear.predict_proba(x["test"][linear_columns])[:, 1],
        "xgboost": model.predict_proba(x["test"])[:, 1],
    }
    results = {
        "events": {name: len(frame) for name, frame in splits.items()},
        "test_base_rate": float(y["test"].mean()),
        "best_iteration": int(model.best_iteration),
        "test": {name: metrics(y["test"], score, test_months) for name, score in scores.items()},
        "top_features": sorted(
            zip(FEATURES, map(float, model.feature_importances_)), key=lambda pair: pair[1], reverse=True
        )[:8],
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(MODEL_PATH)
    RESULTS_PATH.write_text(json.dumps(results, indent=2))

    mlflow.set_tracking_uri(f"sqlite:///{config.DATA_DIR / 'mlflow.db'}")
    mlflow.set_experiment("volatility")
    with mlflow.start_run(run_name="xgboost"):
        mlflow.log_params(model.get_params() | {"features": len(FEATURES), "train_end": TRAIN_END})
        for name, values in results["test"].items():
            mlflow.log_metric(f"test_auc_{name}", values["roc_auc"])
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["build", "train"])
    arguments = parser.parse_args()
    {"build": build, "train": train}[arguments.command]()
