"""
app_updated_fast.py
-------------------


Wichtig:
- Keine großen Row-Level-Loads ohne Filter
- Aggregationen direkt in SQL (kleine Resultsets)
- Caching (st.cache_data) für DB-Abfragen
- Forms + "Berechnen"-Buttons, damit Slider/Selectbox nicht ständig alles neu rechnen
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import streamlit as st
import sys
import os

# --- PFAD-FIX (WICHTIG) ---
# Das hier sorgt dafür, dass Python die Ordner findet, auch wenn VS Code sie weiß anzeigt
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from Part_justin.src import db_manager as db
from Part_tobi import eda_analysis as eda
from Part_Krisztian.model_prophet import backtest_holdout as prophet_backtest


# -----------------------------
# Streamlit Config
# -----------------------------
st.set_page_config(page_title="Online Retail II – Forecast Cockpit ", layout="wide", page_icon="🛒")

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


# -----------------------------
# Navigation
# -----------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Bereich wählen",
    ["1) Import", "2) EDA", "3) Business KPIs", "4) Forecast + Backtest", "5) DB Overview"],
    index=0,
)

st.write("# 🛒 Online Retail II – Forecast Cockpit ")
st.caption(f"DB-Datei: {DB_FILE}")


# -----------------------------
# Helpers (fast + cached)
# -----------------------------
def db_ready() -> bool:
    return DB_FILE.exists()

def ensure_uploaded_excel(uploaded_file) -> Path:
    """Speichert Upload unter dem erwarteten Namen."""
    target = RAW_DIR / "online_retail_II.xlsx"
    with open(target, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return target

def safe_run(cmd: list[str]) -> tuple[bool, str]:
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
    if not db_ready():
        return []
    return db.get_all_countries()

@st.cache_data(ttl=300)
def cached_dashboard_kpis(start_date=None, end_date=None, country=None) -> dict:
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
    """Wochenumsatz direkt in SQL aggregieren (sehr schnell)."""
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
    params = []
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

@st.cache_data(ttl=300)
def cached_revenue_sample_sql(country=None, limit=50000) -> pd.Series:
    """Umsatz-Stichprobe für Histogramm/Outlier-Checks (statt alles zu laden)."""
    if not db_ready():
        return pd.Series(dtype=float)

    conn = db.get_connection()
    query = """
    SELECT (s.quantity * s.price) AS umsatz
    FROM sales s
    JOIN customers c ON s.customer_id = c.customer_id
    WHERE s.quantity > 0
    """
    params = []
    if country:
        query += " AND c.country = ?"
        params.append(country)
    query += " LIMIT ?"
    params.append(int(limit))

    df = pd.read_sql(query, conn, params=params)
    conn.close()
    if df.empty:
        return pd.Series(dtype=float)
    return df["umsatz"].astype(float)

def iqr_outliers(values: pd.Series, k: float = 1.5) -> dict:
    values = values.dropna()
    if values.empty:
        return {"q1": None, "q3": None, "iqr": None, "lower": None, "upper": None, "outlier_count": 0}
    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    outlier_count = int(((values < lower) | (values > upper)).sum())
    return {"q1": float(q1), "q3": float(q3), "iqr": float(iqr), "lower": float(lower), "upper": float(upper), "outlier_count": outlier_count}


# -----------------------------
# Pages
# -----------------------------
if page == "1) Import":
    st.write("## Step 1: Import Excel → CSV → SQLite ")

    col_a, col_b = st.columns([2, 1], vertical_alignment="top")

    with col_a:
        up = st.file_uploader("online_retail_II.xlsx hochladen", type=["xlsx"])

        st.write("### Status")
        if db_ready():
            st.success("DB gefunden ✅")
        else:
            st.warning("DB nicht gefunden ❌ (Pipeline noch nicht ausgeführt)")

        if db_ready():
            status = cached_db_status()
            if status["sales_rows"] > 0:
                st.info(
                    f"Sales-Zeilen: {status['sales_rows']:,} | "
                    f"{pd.to_datetime(status['min_date']).date()} bis {pd.to_datetime(status['max_date']).date()}"
                )

        st.write("### Pipeline ausführen")
        st.caption("Ablauf: Upload → raw/online_retail_II.xlsx → 01_excel_to_csv.py → 02_init_db.py")

        if up is not None:
            saved = ensure_uploaded_excel(up)
            st.success(f"Upload gespeichert: {saved}")

        if st.button("🚀 Pipeline starten", disabled=(up is None)):
            st.cache_data.clear()  # nach Neuaufbau Cache leeren
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

    with col_b:
        st.markdown(
            """**Performance-Hinweis**
- Keine vollständigen Join-Loads mehr automatisch.
- EDA/KPIs nutzen SQL-Aggregationen + Caching + Buttons.
"""
        )

elif page == "2) EDA":
    st.write("## Step 2: EDA")

    if not db_ready():
        st.warning("Keine DB. Erst Import ausführen.")
    else:
        with st.form("eda_form"):
            countries = cached_countries()
            country_choice = st.selectbox("Land (optional)", ["(alle)"] + countries)
            country = None if country_choice == "(alle)" else country_choice

            
            roll_window = st.slider("Rolling Window (Wochen)", 2, 12, 4)

            run_eda = st.form_submit_button("EDA berechnen")

        if run_eda:
            kpis = cached_dashboard_kpis(country=country)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Umsatz", f"{kpis['revenue']:,.0f}")
            c2.metric("Bestellungen", f"{kpis['orders']:,.0f}")
            c3.metric("Kunden", f"{kpis['customers']:,.0f}")
            c4.metric("Ø Wert", f"{kpis['aov']:,.2f}")

            st.divider()

            st.write("### Wochenumsatz (Zeitreihe)")
            ts = cached_weekly_revenue_sql(country=country)
            if ts.empty:
                st.warning("Keine Daten für die Auswahl.")
            else:
                st.line_chart(ts.set_index("ds")["y"])
                st.write("Deskriptive Statistik (Wochenumsatz)")
                st.dataframe(ts[["y"]].describe().T, use_container_width=True)

                ts_roll = eda.rolling_average(ts.rename(columns={"ds": "ds", "y": "y"}), window=roll_window)
                st.write("### Glättung (Rolling Average)")
                st.line_chart(ts_roll.set_index("ds")[["y", "y_roll"]])

            st.divider()
    
            st.write("### Umsatz pro Monat (mit Rolling Average)")

            monthly = cached_monthly_revenue(country=country)

            if monthly.empty:
                st.info("Keine Monatsdaten verfügbar.")
            else:
            # Monat in echtes Datum umwandeln
                monthly["monat"] = pd.to_datetime(monthly["monat"] + "-01")


            # Rolling Average berechnen
                monthly = monthly.sort_values("monat")
                monthly["umsatz_roll"] = (
                    monthly
                    .set_index("monat")["umsatz"]
                    .rolling(window=roll_window)
                    .mean()
                    .values
                    )
                 # Slider für Rolling Window (EDA-Feature)
                roll_window = st.slider(
                    "Rolling Window (Monate)",
                    min_value=2,
                    max_value=12,
                    value=3
                    )
                # Plot
                st.line_chart(
                    monthly.set_index("monat")[["umsatz", "umsatz_roll"]]
                    )
                st.write("### Deskriptive Statistik (Monatsumsatz)")
                st.dataframe(monthly[["umsatz"]].describe().T, use_container_width=True)
            st.divider()
            st.write("### Umsatz nach Wochentag")

            weekday = eda.weekday_profile(country=country)
            if weekday.empty:
                st.info("Keine Daten für Wochentage.")
            else:
                # optional: deutsche Labels
                mapping = {
                    "Monday": "Mo", "Tuesday": "Di", "Wednesday": "Mi",
                    "Thursday": "Do", "Friday": "Fr", "Saturday": "Sa", "Sunday": "So"
                }
                w = weekday.copy()
                w["weekday"] = w["weekday"].astype(str).map(lambda x: mapping.get(x, x))
                order = ["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"]
                w["weekday"] = pd.Categorical(w["weekday"], categories=order, ordered=True)
                w = w.sort_values("weekday")
                st.bar_chart(w.set_index("weekday")["revenue"])
                st.dataframe(weekday, use_container_width=True)


        else:
            st.info("Wähle Einstellungen und klicke **EDA berechnen**.")

elif page == "3) Business KPIs":
    st.write("## Step 3: Business KPIs ")

    if not db_ready():
        st.warning("Keine DB. Erst Import ausführen.")
    else:
        with st.form("kpi_form"):
            countries = cached_countries()
            country_choice = st.selectbox("Land (optional)", ["(alle)"] + countries)
            country = None if country_choice == "(alle)" else country_choice

            top_n_products = st.slider("Top-Produkte", 5, 50, 10)
            top_n_countries = st.slider("Top-Länder", 5, 30, 10)

            run_kpis = st.form_submit_button("KPIs berechnen")

        if run_kpis:
            kpis = cached_dashboard_kpis(country=country)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Gesamtumsatz", f"{kpis['revenue']:,.0f}")
            c2.metric("Bestellungen", f"{kpis['orders']:,.0f}")
            c3.metric("Aktive Kunden", f"{kpis['customers']:,.0f}")
            c4.metric("Ø Wert (AOV)", f"{kpis['aov']:,.2f}")

            st.write("---")

            st.write("### Top Produkte (nach Umsatz)")
            st.dataframe(cached_top_products(limit=top_n_products, country=country), use_container_width=True)

            st.write("### Top Länder (nach Umsatz)")
            st.dataframe(cached_top_countries(limit=top_n_countries), use_container_width=True)

            st.write("### Bestellungen nach Uhrzeit")
            hourly = cached_hourly_activity(country=country)
            if not hourly.empty:
                h = hourly.copy()
                h["stunde"] = h["stunde"].astype(int)
                h = h.sort_values("stunde")
                st.bar_chart(h.set_index("stunde")["anzahl_bestellungen"])
            else:
                st.info("Keine Hourly-Daten verfügbar.")

            with st.expander("Retouren/Stornos – Preview"):
                ret = eda.returns_summary(limit_preview=50)
                st.metric("Anzahl Retouren (Preview)", ret["count_returns"])
                st.metric("Erstattungsbetrag (Preview, absolut)", f"{ret['refund_total']:,.2f}")
                st.dataframe(ret["preview_df"], use_container_width=True)           
        else:
            st.info("Wähle Einstellungen und klicke **KPIs berechnen**.")

elif page == "4) Forecast + Backtest": # Achte darauf, dass der Name exakt dem in der Sidebar entspricht!
    st.write("## 🔮 AI Forecast + Backtest (Prophet)")

    # Lokale Konstanten (damit es nicht crasht, falls Tobi sie nicht hat)
    WEEK_FREQ = "W-SUN"
    YEARLY = True

    # 1. DATEN LADEN & AGGREGIEREN
    try:
        # Rohdaten holen (Jede Zeile = 1 Transaktion)
        df_raw = db.get_sales_data()
        
        if df_raw.empty:
            st.warning("⚠️ Keine Daten in der Datenbank gefunden.")
        else:
            # Manuelle Aggregation: Von Rohdaten zu Wochen-Summen
            # Das ist notwendig, weil Prophet eine Zeitreihe braucht (ds, y)
            df_proc = df_raw.copy()
            df_proc['invoice_date'] = pd.to_datetime(df_proc['invoice_date'])
            
            # Resample auf Wochen (Sonntag) und Umsatz summieren
            df_sales = df_proc.set_index('invoice_date').resample(WEEK_FREQ)['umsatz'].sum().reset_index()
            df_sales.columns = ['ds', 'y'] # Prophet-Konvention

            # 2. GUI: EINSTELLUNGEN
            col_info, col_conf = st.columns([1, 2])
            with col_info:
                st.info(f"**Datenbasis:**\n{len(df_sales)} Wochen aggregiert.")
            
            with col_conf:
                horizon_weeks = st.slider("Forecast Horizont (Wochen)", 4, 104, 26)
                test_weeks = st.slider("Backtest (Holdout)", 4, 52, 12, help="Wie viele Wochen sollen zum Testen abgeschnitten werden?")

            # Erweiterte Parameter (im Expander versteckt für Clean Look)
            with st.expander("⚙️ Modell-Parameter (Experten)"):
                c1, c2 = st.columns(2)
                with c1:
                    season_sel = st.selectbox("Saisonalität", ["Multiplikativ (Prozentual)", "Additiv (Absolut)"], index=0)
                    seasonality_mode = "multiplicative" if "Multiplikativ" in season_sel else "additive"
                with c2:
                    cps = st.slider("Trend-Flexibilität", 0.01, 0.5, 0.05, help="Höher = Modell reagiert schneller auf Trendbrüche.")

            # 3. BERECHNUNG STARTEN
            st.markdown("---")
            if st.button("🚀 Forecast berechnen", type="primary"):
                with st.spinner("KI trainiert Modell & validiert..."):
                    try:
                        # Aufruf deiner Funktion aus model_prophet.py
                        res = prophet_backtest(
                            df=df_sales,
                            test_weeks=int(test_weeks),
                            horizon_weeks=int(horizon_weeks),
                            week_freq=WEEK_FREQ,
                            seasonality_mode=seasonality_mode,
                            cps=float(cps),
                            sps=3.0,
                            yearly=YEARLY
                        )

                        # 4. ERGEBNISSE ANZEIGEN
                        st.success("Berechnung erfolgreich abgeschlossen!")

                        # Metriken (KPIs)
                        kpi1, kpi2, kpi3 = st.columns(3)
                        kpi1.metric("MAPE (Fehler)", f"{res.metrics['MAPE_%']:.1f} %", help="Mittlerer absoluter prozentualer Fehler")
                        kpi2.metric("MAE (Absolut)", f"{res.metrics['MAE']:.0f} €", help="Mittlerer absoluter Fehler in Euro")
                        kpi3.metric("RMSE", f"{res.metrics['RMSE']:.0f}")

                        # Plotting (Matplotlib)
                        fig, ax = plt.subplots(figsize=(10, 5))
                        
                        # A) Historie (Training) - Grau/Schwarz
                        ax.plot(res.train_df["ds"], res.train_df["y"], label="Historie (Training)", color="#333333", alpha=0.4)
                        
                        # B) Realität (Holdout) - Rot gestrichelt
                        ax.plot(res.test_df["ds"], res.test_df["y"], label="Realität (Holdout)", color="#d62728", linestyle="--", linewidth=2)
                        
                        # C) Forecast - Blau
                        ax.plot(res.forecast["ds"], res.forecast["yhat"], label="KI Forecast", color="#1f77b4", linewidth=2)
                        
                        # D) Unsicherheit (Konfidenzintervall) - Blauer Schatten
                        ax.fill_between(
                            res.forecast["ds"], 
                            res.forecast["yhat_lower"], 
                            res.forecast["yhat_upper"], 
                            alpha=0.15, color="#1f77b4", label="80% Konfidenz"
                        )

                        # E) Styling
                        ax.set_title(f"Sales Forecast ({seasonality_mode})", fontsize=12)
                        ax.set_ylabel("Umsatz (€)")
                        ax.legend(loc="upper left")
                        ax.grid(True, alpha=0.2)
                        
                        # Datumsformatierung der X-Achse
                        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=10))
                        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
                        
                        st.pyplot(fig)

                    except Exception as e:
                        st.error(f"Fehler bei der Modell-Berechnung: {e}")

    except Exception as e:
        st.error(f"Fehler bei der Datenverarbeitung: {e}")

elif page == "5) DB Overview":
    st.write("## Step 5: DB Overview ")

    if not db_ready():
        st.warning("Keine DB. Erst Import ausführen.")
    else:
        st.success("DB gefunden ✅")
        conn = db.get_connection()
        tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;", conn)
        conn.close()

        st.write("### Tabellen")
        st.dataframe(tables, use_container_width=True)

        for t in tables["name"].tolist():
            with st.expander(f"Preview: {t}"):
                conn = db.get_connection()
                df = pd.read_sql(f"SELECT * FROM {t} LIMIT 50;", conn)
                conn.close()
                st.dataframe(df, use_container_width=True)
