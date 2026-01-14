"""
Dieses Modul enthält alle Funktionen für EDA.
"""

from __future__ import annotations

import pandas as pd
from Part_justin.src import db_manager as db




# 1) Basis: Daten holen 

def load_sales(start_date: str | None = None,
               end_date: str | None = None,
               country: str | None = None) -> pd.DataFrame:
   
    df = db.get_sales_data(start_date=start_date, end_date=end_date, country=country)
    if df.empty:
        return df
    # Sicherheit: sortieren
    df = df.sort_values("invoice_date").reset_index(drop=True)
    return df



# 2) Zeitreihen-EDA

def weekly_revenue(start_date=None, end_date=None, country=None):
    return db.get_weekly_revenue(start_date=start_date, end_date=end_date, country=country)


def monthly_revenue(year=None, country=None):
    return db.get_monthly_revenue(year=year, country=country)




def rolling_average(ts: pd.DataFrame, window: int = 4) -> pd.DataFrame:
    if ts.empty:
        return ts
    out = ts.copy()
    out["y_roll"] = out["y"].rolling(window=window).mean()
    return out


def weekday_profile(start_date: str | None = None,
                    end_date: str | None = None,
                    country: str | None = None) -> pd.DataFrame:
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


