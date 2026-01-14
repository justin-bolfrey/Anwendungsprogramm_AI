from __future__ import annotations

import os
import sys
import subprocess
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import streamlit as st



# PFAD-FIX 

# Dass Python Projekt-Module findet 
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from Part_justin.src import db_manager as db
from Part_tobi import eda_analysis as eda
from Part_Krisztian.model_prophet import backtest_holdout as prophet_backtest



# Streamlit Config

st.set_page_config(
    page_title="Online Retail II – Analytics & Forecast Dashboard",
    layout="wide",
    page_icon="📈",
)

PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DIR = PROJECT_ROOT / "Part_justin" / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "Part_justin" / "data" / "processed"
DB_FILE = Path(db.DB_PATH)

PIPELINE_EXCEL = PROJECT_ROOT / "Part_justin" / "src" / "01_excel_to_csv.py"
PIPELINE_DB = PROJECT_ROOT / "Part_justin" / "src" / "02_init_db.py"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

WEEK_FREQ = "W-SUN"
DEFAULT_HORIZON_WEEKS = 26
DEFAULT_TEST_WEEKS = 12



# Navigation (Tabs statt Sidebar)

st.write("# 📈 Online Retail II – Analytics & Forecast Dashboard")

tab_import, tab_eda, tab_kpis, tab_forecast, tab_db = st.tabs(
    ["1) Import", "2) EDA", "3) Business KPIs", "4) Prophet Forecast", "5) DB Overview"]
)



# Helpers 

def db_ready() -> bool:
    """True, wenn die SQLite DB-Datei existiert."""
    return DB_FILE.exists()

def safe_run(cmd: list[str]) -> tuple[bool, str]:
    """Startet ein Script per subprocess und gibt (ok, output) zurück."""
    try:
        res = subprocess.run(cmd, check=True, capture_output=True, text=True)
        out = (res.stdout or "") + ("\n" + res.stderr if res.stderr else "")
        return True, out.strip()
    except subprocess.CalledProcessError as e:
        out = (e.stdout or "") + ("\n" + e.stderr if e.stderr else "")
        return False, out.strip()


@st.cache_data(ttl=300)
def cached_db_status() -> dict:
    """Sehr schneller DB-Status ohne große Joins."""
    if not db_ready():
        return {"sales_rows": 0, "min_date": None, "max_date": None}

    conn = db.get_connection()
    df = pd.read_sql(
        """
        SELECT COUNT(*) AS sales_rows,
               MIN(invoice_date) AS min_date,
               MAX(invoice_date) AS max_date
        FROM sales
        """,
        conn,
    )
    conn.close()

    if df.empty:
        return {"sales_rows": 0, "min_date": None, "max_date": None}

    return {
        "sales_rows": int(df.loc[0, "sales_rows"]),
        "min_date": df.loc[0, "min_date"],
        "max_date": df.loc[0, "max_date"],
    }


@st.cache_data(ttl=300)
def cached_countries() -> list[str]:
    """Liste aller Länder (für Filter)."""
    if not db_ready():
        return []
    return db.get_all_countries()


@st.cache_data(ttl=300)
def cached_dashboard_kpis(start_date=None, end_date=None, country=None) -> dict:
    """Kompakte KPI-Aggregationen aus der DB-Schicht."""
    if not db_ready():
        return {"revenue": 0, "orders": 0, "customers": 0, "aov": 0}
    return db.get_dashboard_kpis(start_date=start_date, end_date=end_date, country=country)


@st.cache_data(ttl=300)
def cached_monthly_revenue(year=None, country=None) -> pd.DataFrame:
    if not db_ready():
        return pd.DataFrame()
    return db.get_monthly_revenue(year=year, country=country)


@st.cache_data(ttl=300)
def cached_top_products(limit=10, country=None) -> pd.DataFrame:
    if not db_ready():
        return pd.DataFrame()
    return db.get_top_products(limit=limit, country=country)


@st.cache_data(ttl=300)
def cached_top_countries(limit=10) -> pd.DataFrame:
    if not db_ready():
        return pd.DataFrame()
    return db.get_top_countries_by_revenue(top_n=limit)


@st.cache_data(ttl=300)
def cached_hourly_activity(country=None) -> pd.DataFrame:
    if not db_ready():
        return pd.DataFrame()
    return db.get_hourly_activity(country=country)


@st.cache_data(ttl=300)
def cached_weekly_revenue_sql(start_date=None, end_date=None, country=None) -> pd.DataFrame:
    if not db_ready():
        return pd.DataFrame(columns=["ds", "y"])

    conn = db.get_connection()
    query = """
    SELECT
        date(s.invoice_date, 'weekday 0') AS ds,
        SUM(s.quantity * s.price) AS y
    FROM sales s
    JOIN customers c ON s.customer_id = c.customer_id
    WHERE s.quantity > 0
    """
    params: list = []

    if start_date:
        query += " AND s.invoice_date >= ?"
        params.append(start_date)
    if end_date:
        query += " AND s.invoice_date <= ?"
        params.append(end_date)
    if country:
        query += " AND c.country = ?"
        params.append(country)

    query += " GROUP BY date(s.invoice_date, 'weekday 0') ORDER BY ds"
    df = pd.read_sql(query, conn, params=params)
    conn.close()

    if not df.empty:
        df["ds"] = pd.to_datetime(df["ds"])
        df["y"] = df["y"].astype(float)
    return df



# Pages

with tab_import:
    st.write("## Import: Excel → CSV → SQLite")

    col_a, col_b = st.columns([2, 1], vertical_alignment="top")

    with col_a:
        st.write("### Status")
        if db_ready():
            st.success("DB gefunden ✅")
        else:
            st.warning("DB nicht gefunden ❌ (Pipeline noch nicht ausgeführt)")

        if db_ready():
            status = cached_db_status()
            if status["sales_rows"] > 0 and status["min_date"] and status["max_date"]:
                st.info(
                    f"Sales-Zeilen: {status['sales_rows']:,} | "
                    f"{pd.to_datetime(status['min_date']).date()} bis {pd.to_datetime(status['max_date']).date()}"
                )

        st.write("### Pipeline ausführen")
        st.caption("Ablauf: 01_excel_to_csv.py → 02_init_db.py")

        if st.button("🚀 Pipeline starten", key="btn_pipeline"):
            st.cache_data.clear()
            st.info("Starte Schritt 1: Excel → CSV …")
            ok1, out1 = safe_run(["python", str(PIPELINE_EXCEL)])
            st.code(out1 or "(keine Ausgabe)")
            if not ok1:
                st.error("Schritt 1 fehlgeschlagen.")
            else:
                st.info("Starte Schritt 2: CSV → SQLite DB …")
                ok2, out2 = safe_run(["python", str(PIPELINE_DB)])
                st.code(out2 or "(keine Ausgabe)")
                if ok2 and db_ready():
                    st.success("✅ Import abgeschlossen! DB ist bereit.")
                else:
                    st.error("Schritt 2 fehlgeschlagen oder DB wurde nicht erstellt.")



with tab_eda:
    st.write("## EDA - Explorative Datenalyse")

    if not db_ready():
        st.warning("Keine DB. Erst Import ausführen.")
    else:
        with st.form("eda_form"):
            countries = cached_countries()
            country_choice = st.selectbox(
                "Land (optional)",
                ["(alle)"] + countries,
                key="eda_country",
            )
            country = None if country_choice == "(alle)" else country_choice

            roll_window = st.slider(
                "Rolling Window (Wochen)",
                2, 12, 4,
                key="eda_roll_window",
            )

            run_eda = st.form_submit_button("EDA berechnen")

        if run_eda:
            st.divider()

            # Wochenumsatz
            st.write("### Wochenumsatz (Zeitreihe)")
            ts = cached_weekly_revenue_sql(country=country)
            if ts.empty:
                st.warning("Keine Daten für die Auswahl.")
            else:
                st.line_chart(ts.set_index("ds")["y"])
                st.write("Deskriptive Statistik (Wochenumsatz)")
                st.dataframe(ts[["y"]].describe().T, width="stretch")

                # Rolling Average 
                ts_roll = eda.rolling_average(ts[["ds", "y"]], window=roll_window)
                st.write("### Glättung (Rolling Average)")
                st.line_chart(ts_roll.set_index("ds")[["y", "y_roll"]])

            st.divider()

            # Monatsumsatz
            st.write("### Umsatz pro Monat (mit Rolling Average)")
            monthly = cached_monthly_revenue(country=country)

            if monthly.empty:
                st.info("Keine Monatsdaten verfügbar.")
            else:
                monthly = monthly.copy()
                monthly["monat"] = pd.to_datetime(monthly["monat"] + "-01")
                monthly = monthly.sort_values("monat")

                monthly["umsatz_roll"] = (
                    monthly.set_index("monat")["umsatz"]
                    .rolling(window=3)
                    .mean()
                    .values
                )

                st.line_chart(monthly.set_index("monat")[["umsatz", "umsatz_roll"]])
                st.write("Deskriptive Statistik (Monatsumsatz)")
                st.dataframe(monthly[["umsatz"]].describe().T, width="stretch")

            st.divider()

            # Umsatz nach Wochentag
            st.write("### Umsatz nach Wochentag")
            weekday = eda.weekday_profile(country=country)
            if weekday.empty:
                st.info("Keine Daten für Wochentage.")
            else:
                mapping = {
                    "Monday": "Mo", "Tuesday": "Di", "Wednesday": "Mi",
                    "Thursday": "Do", "Friday": "Fr", "Saturday": "Sa", "Sunday": "So",
                }
                w = weekday.copy()
                w["weekday"] = w["weekday"].astype(str).map(lambda x: mapping.get(x, x))
                order = ["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"]
                w["weekday"] = pd.Categorical(w["weekday"], categories=order, ordered=True)
                w = w.sort_values("weekday")

                st.bar_chart(w.set_index("weekday")["revenue"])
                st.dataframe(weekday, width="stretch")


            with st.expander("Retouren/Stornos – Preview"):
                ret = eda.returns_summary(limit_preview=50)
                st.metric("Anzahl Retouren (Preview)", ret["count_returns"])
                st.metric("Erstattungsbetrag (Preview, absolut)", f"{ret['refund_total']:,.2f}")
                st.dataframe(ret["preview_df"], width="stretch")

        else:
            st.info("Wähle Einstellungen und klicke **EDA berechnen**.")


with tab_kpis:
    st.write("## Business KPIs")

    if not db_ready():
        st.warning("Keine DB. Erst Import ausführen.")
    else:
        with st.form("kpi_form"):
            countries = cached_countries()
            country_choice = st.selectbox(
                "Land (optional)",
                ["(alle)"] + countries,
                key="kpi_country",
            )
            country = None if country_choice == "(alle)" else country_choice

            top_n_products = st.slider("Top-Produkte", 5, 50, 10, key="kpi_top_products")
            top_n_countries = st.slider("Top-Länder", 5, 30, 10, key="kpi_top_countries")

            run_kpis = st.form_submit_button("KPIs berechnen")

        if run_kpis:
            kpis = cached_dashboard_kpis(country=country)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Gesamtumsatz", f"{kpis['revenue']:,.0f}")
            c2.metric("Bestellungen", f"{kpis['orders']:,.0f}")
            c3.metric("Aktive Kunden", f"{kpis['customers']:,.0f}")
            c4.metric("Ø Wert (AOV)", f"{kpis['aov']:,.2f}")

            st.divider()

            st.write("### Top Produkte (nach Umsatz)")
            st.dataframe(
                cached_top_products(limit=top_n_products, country=country),
                width="stretch",
            )

            st.write("### Top Länder (nach Umsatz)")
            st.dataframe(
                cached_top_countries(limit=top_n_countries),
                width="stretch",
            )

            st.write("### Bestellungen nach Uhrzeit")
            hourly = cached_hourly_activity(country=country)
            if not hourly.empty:
                h = hourly.copy()
                h["stunde"] = h["stunde"].astype(int)
                h = h.sort_values("stunde")
                st.bar_chart(h.set_index("stunde")["anzahl_bestellungen"])
            else:
                st.info("Keine Hourly-Daten verfügbar.")

        else:
            st.info("Wähle Einstellungen und klicke **KPIs berechnen**.")


with tab_forecast:
    st.write("## 🔮 Prophet Forecast")

    if not db_ready():
        st.warning("Keine DB. Erst Import ausführen.")
    else:

        try:
            df_sales = db.get_cleaned_prophet_data()
        except AttributeError:

            df_sales = cached_weekly_revenue_sql()

        if df_sales.empty:
            st.warning("⚠️ Keine Daten verfügbar.")
        else:

            df_sales = df_sales.iloc[:-1].copy()

            col_info, col_conf = st.columns([1, 2], vertical_alignment="center")
            with col_info:
                st.info(f"**Datenbasis:** {len(df_sales)} Wochen.")

            with col_conf:
                horizon_weeks = st.slider(
                    "Forecast Horizont (Wochen)",
                    4, 104, DEFAULT_HORIZON_WEEKS,
                    key="fc_horizon",
                )
                test_weeks = st.slider(
                    "Backtest (Holdout)",
                    4, 52, DEFAULT_TEST_WEEKS,
                    help="Wie viele Wochen werden zum Testen vom Ende abgeschnitten?",
                    key="fc_test",
                )

            with st.expander("⚙️ Modell-Parameter (Experten)"):
                c1, c2 = st.columns(2)
                with c1:

                    season_sel = st.selectbox(
                        "Saisonalität",
                        ["Multiplikativ (Empfohlen)", "Additiv"],
                        index=0,
                        key="fc_seasonality",
                    )

                    seasonality_mode = "multiplicative" if "Multiplikativ" in season_sel else "additive"
                
                with c2:

                    cps = st.slider(
                        "Trend-Flexibilität",
                        0.01, 0.5, 0.15, 
                        help="Höher = Modell reagiert schneller auf Trendbrüche.",
                        key="fc_cps",
                    )

            st.divider()

            if st.button("🚀 Forecast berechnen", type="primary", key="btn_forecast"):
                with st.spinner(f"Modell rechnet ({seasonality_mode})..."):
                    try:
                        res = prophet_backtest(
                            df=df_sales,
                            test_weeks=int(test_weeks),
                            horizon_weeks=int(horizon_weeks),
                            week_freq=WEEK_FREQ,
  
                            seasonality_mode=seasonality_mode, 
                            cps=float(cps),
                            sps=10.0, 
                            yearly=True,
                        )

                        st.success("Berechnung erfolgreich!")

                        kpi1, kpi2, kpi3 = st.columns(3)
                        kpi1.metric("MAPE", f"{res.metrics['MAPE_%']:.1f} %")
                        kpi2.metric("MAE", f"{res.metrics['MAE']:.0f} €")
                        kpi3.metric("RMSE", f"{res.metrics['RMSE']:.0f}")

                        fig, ax = plt.subplots(figsize=(10, 3))
                        
               
                        ax.plot(res.train_df["ds"], res.train_df["y"], label="Historie", alpha=0.4, color="#333333")
                        ax.plot(res.test_df["ds"], res.test_df["y"], label="Realität", linestyle="--", linewidth=2, color="#d62728")
                        ax.plot(res.forecast["ds"], res.forecast["yhat"], label=f"Forecast ({seasonality_mode})", linewidth=2, color="#1f77b4")
                        
                        ax.fill_between(
                            res.forecast["ds"],
                            res.forecast["yhat_lower"],
                            res.forecast["yhat_upper"],
                            alpha=0.15, label="Konfidenz", color="#1f77b4"
                        )
                        
                        ax.set_title(f"Sales Forecast - Modus: {seasonality_mode}", fontsize=10)
                        ax.legend(loc="upper left", fontsize="small")
                        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=10))
                        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
                        ax.grid(True, alpha=0.2)
                        
                        st.pyplot(fig)

                    except Exception as e:
                        st.error(f"Fehler bei der Berechnung: {e}")


with tab_db:
    st.write("## DB Overview")

    if not db_ready():
        st.warning("Keine DB. Erst Import ausführen.")
    else:
        st.success("DB gefunden ✅")

        conn = db.get_connection()
        tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;", conn)
        conn.close()

        st.write("### Tabellen")
        st.dataframe(tables, width="stretch")

        # Preview pro Tabelle (max 50 Zeilen)
        for t in tables["name"].tolist():
            with st.expander(f"Preview: {t}"):
                conn = db.get_connection()
                df = pd.read_sql(f"SELECT * FROM {t} LIMIT 50;", conn)
                conn.close()
                st.dataframe(df, width="stretch")
