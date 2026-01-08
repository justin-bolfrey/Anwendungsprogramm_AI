"""model_prophet.py
----------------
Prophet Forecast + Holdout Backtest (weekly)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd
from prophet import Prophet


@dataclass
class BacktestResult:
    forecast: pd.DataFrame
    metrics: Dict[str, float]
    train_df: pd.DataFrame
    test_df: pd.DataFrame


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    mask = y_true != 0
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def fit_forecast_weekly(
    train_df: pd.DataFrame,
    periods: int,
    week_freq: str = "W-SUN",
    seasonality_mode: str = "additive",
    changepoint_prior_scale: float = 0.05,
    seasonality_prior_scale: float = 3.0,
    yearly_seasonality: bool = True,
) -> pd.DataFrame:
    m = Prophet(
        seasonality_mode=seasonality_mode,
        changepoint_prior_scale=changepoint_prior_scale,
        seasonality_prior_scale=seasonality_prior_scale,
        yearly_seasonality=yearly_seasonality,
        weekly_seasonality=False,
        daily_seasonality=False,
    )
    m.fit(train_df)
    future = m.make_future_dataframe(periods=periods, freq=week_freq, include_history=True)
    return m.predict(future)[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()


def backtest_holdout(
    df: pd.DataFrame,
    test_weeks: int,
    horizon_weeks: int,
    week_freq: str = "W-SUN",
    seasonality_mode: str = "additive",
    cps: float = 0.05,
    sps: float = 3.0,
    yearly: bool = True,
) -> BacktestResult:
    df = df.sort_values("ds").reset_index(drop=True)
    if len(df) <= test_weeks + 10:
        raise ValueError("Zu wenig Daten: test_weeks reduzieren oder mehr Daten importieren.")

    train = df.iloc[:-test_weeks].copy()
    test = df.iloc[-test_weeks:].copy()

    fc = fit_forecast_weekly(
        train_df=train[["ds", "y"]],
        periods=test_weeks + horizon_weeks,
        week_freq=week_freq,
        seasonality_mode=seasonality_mode,
        changepoint_prior_scale=cps,
        seasonality_prior_scale=sps,
        yearly_seasonality=yearly,
    )

    fc_test = fc.set_index("ds").reindex(test["ds"])["yhat"].values
    y_true = test["y"].values.astype(float)
    y_pred = fc_test.astype(float)

    metrics = {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAPE_%": mape(y_true, y_pred),
    }
    return BacktestResult(forecast=fc, metrics=metrics, train_df=train, test_df=test)
