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

from Part_justin.src import db_manager as db
from Part_tobi import eda_analysis as eda


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

elif page == "4) Forecast + Backtest":
    st.write("## Step 4: Forecast + Backtest ")

    if not db_ready():
        st.warning("Keine DB. Erst Import ausführen.")
    else:
        with st.form("forecast_form"):
            model_label = st.selectbox("Forecast-Modell", ["Prophet", "SARIMAX"], index=0)
            countries = cached_countries()
            country_choice = st.selectbox("Land (optional)", ["(alle)"] + countries)
            country = None if country_choice == "(alle)" else country_choice

            horizon_weeks = st.slider("Prognose-Horizont (Wochen)", 4, 104, DEFAULT_HORIZON_WEEKS)
            test_weeks = st.slider("Backtest (Holdout): letzte Wochen als Test", 4, 52, DEFAULT_TEST_WEEKS)

            run_fc = st.form_submit_button("Forecast berechnen")

        if run_fc:
            ts = cached_weekly_revenue_sql(country=country)
            if ts.empty or len(ts) < (test_weeks + 10):
                st.warning("Zu wenige Daten für Forecast/Backtest. Bitte Filter lockern.")
                st.stop()

            ts = ts.sort_values("ds")
            full_idx = pd.date_range(ts["ds"].min(), ts["ds"].max(), freq=WEEK_FREQ)
            ts = ts.set_index("ds").reindex(full_idx)
            ts.index.name = "ds"
            ts["y"] = ts["y"].fillna(0.0)
            ts = ts.reset_index()

            train = ts.iloc[:-test_weeks].copy()
            test = ts.iloc[-test_weeks:].copy()

            st.write("### Historische Zeitreihe (Wochenumsatz)")
            st.line_chart(ts.set_index("ds")["y"])

            model_name = "prophet" if model_label.lower().startswith("prophet") else "sarimax"

            if model_name == "prophet":
                try:
                    from prophet import Prophet
                except Exception:
                    Prophet = None

                if Prophet is None:
                    st.error("Prophet ist nicht installiert. Bitte `prophet` in requirements aufnehmen.")
                else:
                    seasonality_mode_val = "multiplicative" if st.checkbox("Multiplicative Seasonality", value=True) else "additive"
                    cp_prior = st.slider("Changepoint Prior Scale", 0.001, 0.5, 0.05)
                    seas_prior = st.slider("Seasonality Prior Scale", 0.01, 20.0, 10.0)

                    df_train = train[["ds", "y"]].copy()
                    m = Prophet(
                        seasonality_mode=seasonality_mode_val,
                        changepoint_prior_scale=cp_prior,
                        seasonality_prior_scale=seas_prior,
                        yearly_seasonality=True,
                        weekly_seasonality=False,
                        daily_seasonality=False,
                    )
                    m.fit(df_train)

                    future = m.make_future_dataframe(periods=horizon_weeks, freq=WEEK_FREQ)
                    fc = m.predict(future)

                    fc_test = fc[fc["ds"].isin(test["ds"])][["ds", "yhat"]].merge(test, on="ds", how="inner")
                    if not fc_test.empty:
                        mae = (fc_test["y"] - fc_test["yhat"]).abs().mean()
                        denom = fc_test["y"].replace(0, pd.NA)
                        mape = ((fc_test["y"] - fc_test["yhat"]).abs() / denom).dropna().mean() * 100
                        c1, c2 = st.columns(2)
                        c1.metric("MAE (Holdout)", f"{mae:,.2f}")
                        c2.metric("MAPE (Holdout)", f"{mape:,.2f}%")

                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.plot(ts["ds"], ts["y"], label="History")
                    ax.plot(fc["ds"], fc["yhat"], label="Forecast")
                    ax.fill_between(fc["ds"], fc["yhat_lower"], fc["yhat_upper"], alpha=0.2, label="Interval")
                    ax.axvline(train["ds"].max(), linestyle="--", alpha=0.6, label="Train End")
                    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=10))
                    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
                    ax.tick_params(axis="x", labelrotation=45, labelsize=9)
                    ax.tick_params(axis="y", labelsize=9)
                    ax.set_title("Wochenumsatz – Prophet Forecast", fontsize=14)
                    ax.legend()
                    st.pyplot(fig)

            else:
                try:
                    from statsmodels.tsa.statespace.sarimax import SARIMAX
                except Exception:
                    SARIMAX = None

                if SARIMAX is None:
                    st.error("statsmodels ist nicht installiert. Bitte `statsmodels` in requirements aufnehmen.")
                else:
                    order = st.text_input("SARIMAX order (p,d,q)", value="(1,1,1)")
                    seasonal_order = st.text_input("SARIMAX seasonal_order (P,D,Q,s)", value="(0,1,1,52)")

                    def parse_tuple(s: str):
                        return tuple(int(x.strip()) for x in s.strip().strip("()").split(",") if x.strip() != "")

                    try:
                        order_t = parse_tuple(order)
                        seas_t = parse_tuple(seasonal_order)
                    except Exception:
                        st.error("Bitte order/seasonal_order im Format (1,1,1) bzw. (0,1,1,52) eingeben.")
                        st.stop()

                    y_train = train.set_index("ds")["y"]
                    y_test = test.set_index("ds")["y"]

                    model = SARIMAX(y_train, order=order_t, seasonal_order=seas_t, enforce_stationarity=False, enforce_invertibility=False)
                    res = model.fit(disp=False)

                    pred_test = res.get_forecast(steps=len(y_test)).predicted_mean
                    mae = (y_test - pred_test).abs().mean()
                    denom = y_test.replace(0, pd.NA)
                    mape = ((y_test - pred_test).abs() / denom).dropna().mean() * 100
                    c1, c2 = st.columns(2)
                    c1.metric("MAE (Holdout)", f"{mae:,.2f}")
                    c2.metric("MAPE (Holdout)", f"{mape:,.2f}%")

                    future_steps = horizon_weeks
                    fc_mean = res.get_forecast(steps=future_steps).predicted_mean
                    fc_index = pd.date_range(train["ds"].max() + pd.tseries.frequencies.to_offset(WEEK_FREQ), periods=future_steps, freq=WEEK_FREQ)

                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.plot(ts["ds"], ts["y"], label="History")
                    ax.plot(y_test.index, pred_test.values, label="Backtest Forecast")
                    ax.plot(fc_index, fc_mean.values, label="Future Forecast")
                    ax.axvline(train["ds"].max(), linestyle="--", alpha=0.6, label="Train End")
                    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=10))
                    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
                    ax.tick_params(axis="x", labelrotation=45, labelsize=9)
                    ax.tick_params(axis="y", labelsize=9)
                    ax.set_title("Wochenumsatz – SARIMAX Forecast", fontsize=14)
                    ax.legend()
                    st.pyplot(fig)
        else:
            st.info("Wähle Einstellungen und klicke **Forecast berechnen**.")

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
