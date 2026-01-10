"""
eda_analysis.py
================
Dieses Modul enthält alle Funktionen für Deskriptive Analyse & EDA.
"""

from __future__ import annotations

import pandas as pd
from Part_justin.src import db_manager as db




# 1) Basis: Daten holen (Wrapper)

def load_sales(start_date: str | None = None,
               end_date: str | None = None,
               country: str | None = None) -> pd.DataFrame:
    """
    Lädt Verkaufsdaten aus der DB über die bereitgestellte Schnittstelle.

    Rückgabe-Spalten (aus db_manager.get_sales_data):
    - invoice_date (datetime)
    - produkt (str)
    - land (str)
    - umsatz (float)

    Tipp: Diese Funktion kannst du in Streamlit überall nutzen.
    """
    df = db.get_sales_data(start_date=start_date, end_date=end_date, country=country)
    if df.empty:
        return df
    # Sicherheit: sortieren
    df = df.sort_values("invoice_date").reset_index(drop=True)
    return df



# 2) Zeitreihen-EDA

def weekly_revenue(start_date=None, end_date=None, country=None):
    return db.get_weekly_revenue(start_date=start_date, end_date=end_date, country=country)



def rolling_average(ts: pd.DataFrame, window: int = 4) -> pd.DataFrame:
    """
    Fügt einer Zeitreihe (ds,y) einen gleitenden Durchschnitt hinzu.
    Erwartet Spalten: ds, y
    Gibt zurück: ds, y, y_roll
    """
    if ts.empty:
        return ts
    out = ts.copy()
    out["y_roll"] = out["y"].rolling(window=window).mean()
    return out


def weekday_profile(start_date: str | None = None,
                    end_date: str | None = None,
                    country: str | None = None) -> pd.DataFrame:
    """
    Umsatz nach Wochentag (Mo-So).
    Ergebnis-Spalten:
    - weekday (Montag, Dienstag, ...)
    - revenue (Umsatz)
    """
    df = load_sales(start_date, end_date, country)
    if df.empty:
        return df

    df["weekday"] = df["invoice_date"].dt.day_name()

    out = (
        df.groupby("weekday", as_index=False)["umsatz"]
        .sum()
        .rename(columns={"umsatz": "revenue"})
    )

    # Reihenfolge Mo-So stabil machen
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    out["weekday"] = pd.Categorical(out["weekday"], categories=order, ordered=True)
    out = out.sort_values("weekday").reset_index(drop=True)

    return out

# 3) Retouren / Stornos:

def returns_summary(limit_preview: int = 100) -> dict:
    """
    Zusammenfassung über Stornos/Retouren.
    Nutzt db.get_cancellations() als Datenquelle.

    Rückgabe:
    - count_returns
    - refund_total (negativ oder Betrag)
    - preview_df (die letzten N Retouren)
    """
    canc = db.get_cancellations(limit=limit_preview)
    if canc.empty:
        return {"count_returns": 0, "refund_total": 0.0, "preview_df": canc}

    # "erstattung" ist negativ (quantity * price); wir zeigen Betrag positiv an:
    refund_total = float((-canc["erstattung"]).sum())

    return {
        "count_returns": int(len(canc)),
        "refund_total": refund_total,
        "preview_df": canc
    }


# 4) Kleine Utility-Funktion: KPI-Erweiterungen

def simple_kpis(start_date: str | None = None,
               end_date: str | None = None,
               country: str | None = None) -> dict:
    """
    Ein paar zusätzliche deskriptive Kennzahlen auf Basis von get_sales_data():
    - median Umsatz pro Zeile (robust gegen Ausreißer)
    - std (Streuung)
    - min/max
    """
    df = load_sales(start_date, end_date, country)
    if df.empty:
        return {"median": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}

    s = df["umsatz"].astype(float)
    return {
        "median": float(s.median()),
        "std": float(s.std()),
        "min": float(s.min()),
        "max": float(s.max()),
    }



