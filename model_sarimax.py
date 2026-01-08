"""model_sarimax.py
-----------------
SARIMAX Forecast + Holdout Backtest (weekly)

Warum SARIMAX?
- Klassisches Zeitreihenmodell (Baseline/Alternative zu Prophet)
- Gut bei kleinen Datensätzen (z.B. ~100 Wochen)
- Liefert Vorhersage + Konfidenzintervall

Erwartetes Input-Format:
- DataFrame mit Spalten: ds (datetime), y (float)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


# -----------------------------
# Ergebnis-Container (wie bei Prophet)
# -----------------------------
@dataclass
class BacktestResult:
    forecast: pd.DataFrame
    metrics: Dict[str, float]
    train_df: pd.DataFrame
    test_df: pd.DataFrame


# -----------------------------
# Metriken
# -----------------------------
def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MAPE in % (robuster Umgang mit 0-Werten).

    Wenn y_true == 0, wird der Punkt für MAPE übersprungen,
    da Prozentfehler sonst unendlich wäre.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


# -----------------------------
# Hilfsfunktionen
# -----------------------------
def _ensure_weekly_index(df: pd.DataFrame, week_freq: str) -> pd.DataFrame:
    """Sorgt für sauberen Weekly-Index (lückenlos), sortiert nach ds."""
    out = df.sort_values("ds").copy()
    out["ds"] = pd.to_datetime(out["ds"])
    out = out.set_index("ds")
    full_idx = pd.date_range(out.index.min(), out.index.max(), freq=week_freq)
    out = out.reindex(full_idx)
    # fehlende Wochen -> 0 (für Umsatz/Orders in Retail i.d.R. ok)
    out["y"] = out["y"].fillna(0.0)
    out.index.name = "ds"
    return out.reset_index()


def fit_forecast_sarimax_weekly(
    train_df: pd.DataFrame,
    steps_ahead: int,
    week_freq: str = "W-SUN",
    order: Tuple[int, int, int] = (1, 1, 1),
    seasonal_order: Tuple[int, int, int, int] = (0, 0, 0, 0),
) -> pd.DataFrame:
    """Fit SARIMAX auf train_df und erzeuge Forecast (inkl. In-Sample + Future).

    Returned: DataFrame mit Spalten [ds, yhat, yhat_lower, yhat_upper]
    über die gesamte Historie (Train) plus steps_ahead Wochen in der Zukunft.
    """
    train_df = train_df[["ds", "y"]].copy()
    train_df = _ensure_weekly_index(train_df, week_freq=week_freq)
    train_df["ds"] = pd.to_datetime(train_df["ds"])
    y = train_df.set_index("ds")["y"].astype(float)

    # Modell fitten
    model = SARIMAX(
        y,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    res = model.fit(disp=False)

    # In-sample Prediction (Train)
    pred_in = res.get_prediction(start=y.index[0], end=y.index[-1])
    pred_in_mean = pred_in.predicted_mean
    pred_in_ci = pred_in.conf_int()

    # Out-of-sample Forecast
    fc = res.get_forecast(steps=steps_ahead)
    fc_mean = fc.predicted_mean
    fc_ci = fc.conf_int()

    # Zusammenbauen (Index = ds)
    all_mean = pd.concat([pred_in_mean, fc_mean])
    all_ci = pd.concat([pred_in_ci, fc_ci])

    # conf_int Spaltennamen können je nach statsmodels-Version variieren
    # -> robust behandeln
    lower_col = [c for c in all_ci.columns if "lower" in c.lower()]
    upper_col = [c for c in all_ci.columns if "upper" in c.lower()]
    if lower_col and upper_col:
        lo = all_ci[lower_col[0]]
        hi = all_ci[upper_col[0]]
    else:
        # Fallback: wenn conf_int unerwartet ist
        lo = all_mean
        hi = all_mean

    out = pd.DataFrame(
        {
            "ds": all_mean.index,
            "yhat": all_mean.values.astype(float),
            "yhat_lower": lo.values.astype(float),
            "yhat_upper": hi.values.astype(float),
        }
    ).reset_index(drop=True)
    out["ds"] = pd.to_datetime(out["ds"])
    return out


def backtest_holdout_sarimax(
    df: pd.DataFrame,
    test_weeks: int,
    horizon_weeks: int,
    week_freq: str = "W-SUN",
    order: Tuple[int, int, int] = (1, 1, 1),
    seasonal_order: Tuple[int, int, int, int] = (0, 0, 0, 0),
) -> BacktestResult:
    """Holdout-Backtest (chronologischer Split) + Future-Forecast."""
    df = df.sort_values("ds").reset_index(drop=True)
    if len(df) <= test_weeks + 10:
        raise ValueError("Zu wenig Daten: test_weeks reduzieren oder mehr Daten importieren.")

    # Lückenloser Weekly-Index (wichtig für SARIMAX)
    df2 = _ensure_weekly_index(df[["ds", "y"]], week_freq=week_freq)
    df2 = df2.sort_values("ds").reset_index(drop=True)

    train = df2.iloc[:-test_weeks].copy()
    test = df2.iloc[-test_weeks:].copy()

    steps_ahead = int(test_weeks + horizon_weeks)
    fc = fit_forecast_sarimax_weekly(
        train_df=train,
        steps_ahead=steps_ahead,
        week_freq=week_freq,
        order=order,
        seasonal_order=seasonal_order,
    )

    # Test-Prognose aus Forecast ziehen (nur die Test-ds)
    fc_test = fc.set_index("ds").reindex(pd.to_datetime(test["ds"]))["yhat"].values
    y_true = test["y"].values.astype(float)
    y_pred = np.asarray(fc_test, dtype=float)

    metrics = {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAPE_%": mape(y_true, y_pred),
    }
    return BacktestResult(forecast=fc, metrics=metrics, train_df=train, test_df=test)
