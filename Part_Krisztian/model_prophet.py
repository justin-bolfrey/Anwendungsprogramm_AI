"""model_prophet.py
----------------
Prophet Forecast + Holdout Backtest (weekly)
Logik-Layer: Keine UI, reine Berechnung.
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
    
    # Schutz vor Division durch Null (wenn Umsatz = 0 ist)
    mask = y_true > 0.001 
    if not np.any(mask):
        return float("nan")
    
    # MAPE Berechnung nur dort, wo Umsatz > 0
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
    """
    Trainiert das Prophet Modell und erstellt den Forecast.
    """
    m = Prophet(
        seasonality_mode=seasonality_mode,
        changepoint_prior_scale=changepoint_prior_scale,
        seasonality_prior_scale=seasonality_prior_scale,
        yearly_seasonality=yearly_seasonality,
        weekly_seasonality=False, # Wir haben bereits Wochen-Daten
        daily_seasonality=False,
    )
    m.fit(train_df)
    
    # Zukunfts-DataFrame bauen
    future = m.make_future_dataframe(periods=periods, freq=week_freq, include_history=True)
    
    # Vorhersage
    forecast = m.predict(future)
    
    # WICHTIG: Negative Werte abschneiden (Umsatz kann nicht < 0 sein)
    cols = ["ds", "yhat", "yhat_lower", "yhat_upper"]
    result = forecast[cols].copy()
    
    result["yhat"] = result["yhat"].clip(lower=0)
    result["yhat_lower"] = result["yhat_lower"].clip(lower=0)
    result["yhat_upper"] = result["yhat_upper"].clip(lower=0)
    
    return result


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
    """
    Hauptfunktion: Splittet Daten, trainiert Modell, testet Qualität.
    """
    # 1. Datenvorbereitung & Sicherheit
    df = df.copy()
    df["ds"] = pd.to_datetime(df["ds"]) # Zwingend Datetime erzwingen
    df = df.sort_values("ds").reset_index(drop=True)
    
    if len(df) <= test_weeks + 4:
        raise ValueError("Zu wenig Daten: Bitte 'Backtest Wochen' reduzieren.")

    # 2. Split (Train vs Test)
    train = df.iloc[:-test_weeks].copy()
    test = df.iloc[-test_weeks:].copy()

    # 3. Forecast
    # Wir müssen vorhersagen: Zeitraum des Tests (um zu vergleichen) + echte Zukunft
    total_periods = test_weeks + horizon_weeks

    fc = fit_forecast_weekly(
        train_df=train[["ds", "y"]],
        periods=total_periods,
        week_freq=week_freq,
        seasonality_mode=seasonality_mode,
        changepoint_prior_scale=cps,
        seasonality_prior_scale=sps,
        yearly_seasonality=yearly,
    )

    # 4. Metriken berechnen (Vergleich Forecast vs. Test-Daten)
    # Merge ist sicherer als reindex, falls Datumsformate leicht abweichen
    merged = pd.merge(test, fc, on="ds", how="left")
    
    # Nur Zeilen nehmen, wo wir beide Werte haben
    valid_metrics = merged.dropna(subset=["y", "yhat"])
    
    y_true = valid_metrics["y"].values
    y_pred = valid_metrics["yhat"].values

    metrics = {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAPE_%": mape(y_true, y_pred),
    }
    
    return BacktestResult(forecast=fc, metrics=metrics, train_df=train, test_df=test)