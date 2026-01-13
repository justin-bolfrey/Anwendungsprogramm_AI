"""model_prophet.py 
---------------- 
Prophet Forecast + Holdout Backtest (weekly) 
Standard-Version: Flexibel (Additiv/Multiplikativ) & Robust
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
    
    # Schutz vor Division durch Null / extrem kleinen Werten
    mask = y_true > 10.0 
    if not np.any(mask): 
        return float("nan") 
     
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0) 

def remove_outliers(df: pd.DataFrame, lower_q=0.01, upper_q=0.99) -> pd.DataFrame:
    """
    Standard-Bereinigung: Entfernt nur die extremsten 1% Ausreißer (Datenfehler).
    """
    df_clean = df.copy()
    q_low = df_clean['y'].quantile(lower_q)
    q_high = df_clean['y'].quantile(upper_q)
    
    mask = (df_clean['y'] < q_low) | (df_clean['y'] > q_high)
    df_clean.loc[mask, 'y'] = None
    return df_clean

def fit_forecast_weekly( 
    train_df: pd.DataFrame, 
    periods: int, 
    week_freq: str = "W-SUN", 
    seasonality_mode: str = "multiplicative", 
    changepoint_prior_scale: float = 0.05, 
    seasonality_prior_scale: float = 10.0, 
    yearly_seasonality: bool = True, 
    country_holidays: str = "UK", 
) -> pd.DataFrame: 
    
    # 1. Cleaning
    df_train = train_df.copy()
    
    # Sicherheitshalber Nullen entfernen (machen Probleme bei Multiplikativ)
    if seasonality_mode == "multiplicative":
        df_train.loc[df_train['y'] <= 1, 'y'] = None
    
    # Moderate Outlier-Entfernung (1% oben/unten kappen)
    df_train = remove_outliers(df_train)

    # 2. Modell Setup (Standard)
    m = Prophet( 
        seasonality_mode=seasonality_mode, 
        changepoint_prior_scale=changepoint_prior_scale, 
        seasonality_prior_scale=seasonality_prior_scale, 
        yearly_seasonality=False, # Wir fügen es unten manuell hinzu (HD)
        weekly_seasonality=False, 
        daily_seasonality=False, 
    ) 

    # High-Res Saisonalität (hilft trotzdem, den Peak zu treffen)
    if yearly_seasonality:
        m.add_seasonality(name='yearly', period=365.25, fourier_order=15)

    # Feiertage sind immer gut
    if country_holidays:
        try:
            m.add_country_holidays(country_name=country_holidays)
        except AttributeError:
            pass

    m.fit(df_train) 

    # 3. Forecast
    future = m.make_future_dataframe(periods=periods, freq=week_freq, include_history=True) 
    forecast = m.predict(future) 
    
    cols = ["ds", "yhat", "yhat_lower", "yhat_upper"] 
    result = forecast[cols].copy() 
    
    # Keine negativen Werte
    result["yhat"] = result["yhat"].clip(lower=0) 
    result["yhat_lower"] = result["yhat_lower"].clip(lower=0) 
    result["yhat_upper"] = result["yhat_upper"].clip(lower=0) 
     
    return result 

def backtest_holdout( 
    df: pd.DataFrame, 
    test_weeks: int, 
    horizon_weeks: int, 
    week_freq: str = "W-SUN", 
    seasonality_mode: str = "multiplicative", 
    cps: float = 0.05,                        
    sps: float = 10.0,                        
    yearly: bool = True, 
) -> BacktestResult: 

    df = df.copy() 
    df["ds"] = pd.to_datetime(df["ds"]) 
    df = df.sort_values("ds").reset_index(drop=True) 
     
    if len(df) <= test_weeks + 4: 
        raise ValueError("Zu wenig Daten.") 

    train = df.iloc[:-test_weeks].copy() 
    test = df.iloc[-test_weeks:].copy() 

    total_periods = test_weeks + horizon_weeks 

    fc = fit_forecast_weekly( 
        train_df=train[["ds", "y"]], 
        periods=total_periods, 
        week_freq=week_freq, 
        seasonality_mode=seasonality_mode, 
        changepoint_prior_scale=cps, 
        seasonality_prior_scale=sps, 
        yearly_seasonality=yearly, 
        country_holidays="UK" 
    ) 

    merged = pd.merge(test, fc, on="ds", how="left") 
    valid_metrics = merged.dropna(subset=["y", "yhat"]) 
     
    y_true = valid_metrics["y"].values 
    y_pred = valid_metrics["yhat"].values 

    metrics = { 
        "MAE": mae(y_true, y_pred), 
        "RMSE": rmse(y_true, y_pred), 
        "MAPE_%": mape(y_true, y_pred), 
    } 
     
    return BacktestResult(forecast=fc, metrics=metrics, train_df=train, test_df=test)