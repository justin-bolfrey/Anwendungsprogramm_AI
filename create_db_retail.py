"""create_db_retail.py
-------------------
Vorlesungsstil: DB-Erstellung + Import + get_data()

Enthält:
- SQLite Connection
- Tabellen anlegen (Schema)
- Import: Excel -> transactions_raw (optional) + weekly_kpis
- get_data() zum Laden von Tabellen für EDA/Streamlit

Anomalie-Erkennung wurde bewusst entfernt.
"""

import sqlite3
from pathlib import Path
from typing import Tuple, Dict, Any

import pandas as pd

DB_PATH = Path("retail_prophet.db")

def get_con(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Open SQLite connection (Streamlit-safe)."""
    return sqlite3.connect(str(db_path), check_same_thread=False)

def init_db(db_path: Path = DB_PATH) -> None:
    """Create tables if not exists."""
    con = get_con(db_path)
    cur = con.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS weekly_kpis (
            ds TEXT PRIMARY KEY,
            weekly_revenue REAL,
            weekly_orders INTEGER
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS transactions_raw (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            InvoiceNo TEXT,
            InvoiceDate TEXT,
            Quantity REAL,
            UnitPrice REAL,
            Country TEXT,
            revenue REAL
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_raw_date ON transactions_raw(InvoiceDate);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_raw_invoice ON transactions_raw(InvoiceNo);")

    cur.execute("""
        CREATE TABLE IF NOT EXISTS model_runs (
            run_id INTEGER PRIMARY KEY AUTOINCREMENT,
            target TEXT,
            params_json TEXT,
            train_start TEXT,
            train_end TEXT,
            test_weeks INTEGER,
            horizon_weeks INTEGER,
            week_freq TEXT,
            created_at TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER,
            metric_name TEXT,
            metric_value REAL,
            created_at TEXT,
            FOREIGN KEY(run_id) REFERENCES model_runs(run_id)
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS forecasts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER,
            ds TEXT,
            yhat REAL,
            yhat_lower REAL,
            yhat_upper REAL,
            FOREIGN KEY(run_id) REFERENCES model_runs(run_id)
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_fc_run_ds ON forecasts(run_id, ds);")

    con.commit()
    con.close()

def read_sql(query: str, params: tuple = (), db_path: Path = DB_PATH) -> pd.DataFrame:
    con = get_con(db_path)
    try:
        return pd.read_sql_query(query, con, params=params)
    finally:
        con.close()

def _load_online_retail_excel(excel_source) -> pd.DataFrame:
    """Loads Online Retail II Excel (2 sheets) and returns ONE dataframe."""
    sheets = pd.read_excel(excel_source, sheet_name=None, engine="openpyxl")
    df = pd.concat(sheets.values(), ignore_index=True)

    rename_map = {}
    if "Invoice" in df.columns and "InvoiceNo" not in df.columns:
        rename_map["Invoice"] = "InvoiceNo"
    if "Price" in df.columns and "UnitPrice" not in df.columns:
        rename_map["Price"] = "UnitPrice"
    df = df.rename(columns=rename_map)

    required = ["InvoiceNo", "InvoiceDate", "Quantity", "UnitPrice"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Fehlende Spalten im Excel: {missing}. Gefunden: {list(df.columns)}")

    return df

def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Minimaler Raw-Layer (bereinigt)"""
    x = df.copy()
    x["InvoiceDate"] = pd.to_datetime(x["InvoiceDate"], errors="coerce")
    x = x.dropna(subset=["InvoiceDate"])

    x["InvoiceNo"] = x["InvoiceNo"].astype(str)
    x = x[~x["InvoiceNo"].str.startswith("C", na=False)]

    x["Quantity"] = pd.to_numeric(x["Quantity"], errors="coerce")
    x["UnitPrice"] = pd.to_numeric(x["UnitPrice"], errors="coerce")
    x = x.dropna(subset=["Quantity", "UnitPrice"])
    x = x[(x["Quantity"] > 0) & (x["UnitPrice"] > 0)]

    if "Country" not in x.columns:
        x["Country"] = None

    x["revenue"] = x["Quantity"] * x["UnitPrice"]

    x = x[["InvoiceNo", "InvoiceDate", "Quantity", "UnitPrice", "Country", "revenue"]].copy()
    x["InvoiceDate"] = x["InvoiceDate"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return x

def build_weekly_kpis(transactions: pd.DataFrame, week_freq: str = "W-SUN") -> pd.DataFrame:
    """Aggregation: weekly_revenue + weekly_orders"""
    x = transactions.copy()
    x["InvoiceDate"] = pd.to_datetime(x["InvoiceDate"], errors="coerce")
    x = x.dropna(subset=["InvoiceDate"])

    weekly = (
        x.assign(ds=lambda d: d["InvoiceDate"].dt.to_period("W").dt.end_time.dt.normalize())
         .groupby("ds", as_index=False)
         .agg(
             weekly_revenue=("revenue", "sum"),
             weekly_orders=("InvoiceNo", pd.Series.nunique),
         )
         .sort_values("ds")
    )

    weekly = weekly.set_index("ds").asfreq(week_freq).fillna(0).reset_index()
    weekly["weekly_orders"] = weekly["weekly_orders"].astype(int)
    weekly["ds"] = pd.to_datetime(weekly["ds"]).dt.strftime("%Y-%m-%d")
    return weekly

def import_excel_to_db(
    excel_source,
    mode: str = "replace",
    store_raw: bool = True,
    week_freq: str = "W-SUN",
    db_path: Path = DB_PATH,
) -> Dict[str, Any]:
    """Excel -> transactions_raw (optional) -> weekly_kpis"""
    init_db(db_path)

    raw_df = _load_online_retail_excel(excel_source)
    tx = clean_transactions(raw_df)
    weekly = build_weekly_kpis(tx, week_freq=week_freq)

    con = get_con(db_path)
    cur = con.cursor()

    if mode == "replace":
        cur.execute("DELETE FROM weekly_kpis;")
        if store_raw:
            cur.execute("DELETE FROM transactions_raw;")
        con.commit()

    if store_raw:
        tx.to_sql("transactions_raw", con, if_exists="append", index=False, chunksize=10000)

    rows = list(weekly.itertuples(index=False, name=None))
    cur.executemany(
        "INSERT OR REPLACE INTO weekly_kpis (ds, weekly_revenue, weekly_orders) VALUES (?, ?, ?)",
        rows,
    )

    con.commit()
    con.close()

    return {
        "raw_rows_after_clean": int(len(tx)),
        "weekly_rows": int(len(weekly)),
        "store_raw": bool(store_raw),
        "mode": mode,
        "db_path": str(db_path),
    }

def get_data(db_path: Path = DB_PATH) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns: weekly_kpis, model_runs, metrics, forecasts"""
    weekly_kpis = read_sql("SELECT * FROM weekly_kpis ORDER BY ds", db_path=db_path)
    model_runs  = read_sql("SELECT * FROM model_runs ORDER BY run_id DESC", db_path=db_path)
    metrics     = read_sql("SELECT * FROM metrics ORDER BY id DESC", db_path=db_path)
    forecasts   = read_sql("SELECT * FROM forecasts ORDER BY id DESC", db_path=db_path)
    return weekly_kpis, model_runs, metrics, forecasts
