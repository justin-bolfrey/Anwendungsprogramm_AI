"""app.py
------
Vorlesungsstil (Step 1/2/3/4) – ohne Anomalie-Erkennung.

Neu:
- Zweites Modell: SARIMAX (statsmodels) als klassische Baseline/Alternative zu Prophet
- Modellwahl im Forecast-Tab (Prophet vs. SARIMAX)

Hinweis:
- SARIMAX benötigt lückenlosen Weekly-Index -> wir reindexen fehlende Wochen auf 0.
"""

import json
import sqlite3
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import streamlit as st

from create_db_retail import DB_PATH, init_db, read_sql, import_excel_to_db, get_data
from model_prophet import backtest_holdout as prophet_backtest
from model_sarimax import backtest_holdout_sarimax

# -----------------------------
# Basic config
# -----------------------------
st.set_page_config(page_title="Online Retail II – Forecast Cockpit", layout="wide", page_icon="🛒")
init_db()

WEEK_FREQ = "W-SUN"
YEARLY = True

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Bereich wählen",
    ["1) Import", "2) EDA", "3) Business KPIs", "4) Forecast + Backtest", "5) DB Overview"],
    index=0,
)

st.write("# 🛒 Online Retail II – Weekly Forecast Cockpit (Prophet/SARIMAX + SQLite)")
st.caption(f"DB-Datei: {DB_PATH.resolve()}")

# -----------------------------
# Helpers
# -----------------------------
def load_weekly() -> pd.DataFrame:
    df = read_sql("SELECT * FROM weekly_kpis ORDER BY ds")
    if df.empty:
        return df
    df["ds"] = pd.to_datetime(df["ds"])
    return df

# -----------------------------
# Pages
# -----------------------------
if page == "1) Import":
    st.write("## Step 1: Import Excel → SQLite")

    col_a, col_b = st.columns([2, 1], vertical_alignment="top")

    with col_a:
        up = st.file_uploader("online_retail_II.xlsx hochladen", type=["xlsx"])
        path = st.text_input("Oder lokaler Pfad (optional)", value="online_retail_II.xlsx")

        mode_label = st.selectbox("Import-Modus", ["replace (empfohlen)", "append"], index=0)
        mode = "replace" if mode_label.startswith("replace") else "append"

        store_raw = st.checkbox(
            "Raw-Transaktionen zusätzlich in SQLite speichern (für DBeaver/SQL-Analysen)",
            value=True,
            help="Erzeugt Tabelle 'transactions_raw' mit minimalen Spalten. Kann DB größer machen.",
        )

        if st.button("Import starten"):
            try:
                stats = import_excel_to_db(
                    up if up is not None else path,
                    mode=mode,
                    store_raw=store_raw,
                    week_freq=WEEK_FREQ,
                )
                st.success(
                    f"Import OK – Wochen: {stats['weekly_rows']:,} | "
                    f"Raw-Zeilen (bereinigt): {stats['raw_rows_after_clean']:,}"
                )
            except Exception as e:
                st.error(f"Import fehlgeschlagen: {e}")

    with col_b:
        st.markdown(
            """**Was macht der Import?**
- Sheets zusammenführen
- Stornos entfernen (InvoiceNo beginnt mit 'C')
- Quantity/UnitPrice > 0
- Umsatz = Quantity × UnitPrice
- Wochenaggregation (W-SUN)
- Speicherung: weekly_kpis (+ optional transactions_raw)
"""
        )

    st.write("### Aktueller Stand in DB")
    dfw = load_weekly()
    if dfw.empty:
        st.warning("Noch keine weekly_kpis in der DB.")
    else:
        st.info(f"{len(dfw):,} Wochen | {dfw['ds'].min().date()} bis {dfw['ds'].max().date()}")
        st.dataframe(dfw.head(20), use_container_width=True)

elif page == "2) EDA":
    st.write("## Step 2: EDA (Explorative Datenanalyse)")

    dfw = load_weekly()
    if dfw.empty:
        st.warning("Keine Daten. Erst importieren.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Start", str(dfw["ds"].min().date()))
        c2.metric("Ende", str(dfw["ds"].max().date()))
        c3.metric("Wochen", f"{len(dfw):,}")

        series_label = st.selectbox("Zeitreihe", ["Umsatz pro Woche", "Bestellungen pro Woche"])
        series = "weekly_revenue" if series_label.startswith("Umsatz") else "weekly_orders"

        st.line_chart(dfw.set_index("ds")[series])
        st.write("Deskriptive Statistik")
        st.dataframe(dfw[[series]].describe().T, use_container_width=True)

        with st.expander("Raw-Transaktionen (Preview) – aus transactions_raw"):
            raw = read_sql(
                "SELECT InvoiceDate, InvoiceNo, Quantity, UnitPrice, Country, revenue "
                "FROM transactions_raw ORDER BY id DESC LIMIT 50"
            )
            st.dataframe(raw, use_container_width=True)

elif page == "3) Business KPIs":
    st.write("## Business KPIs (Retail Reporting)")

    dfw = load_weekly()
    if dfw.empty:
        st.warning("Keine Daten. Erst importieren.")
    else:
        total_rev = float(dfw["weekly_revenue"].sum())
        total_orders = float(dfw["weekly_orders"].sum())
        avg_rev = float(dfw["weekly_revenue"].mean())
        avg_orders = float(dfw["weekly_orders"].mean())
        rev_per_order = total_rev / total_orders if total_orders else 0.0

        df = dfw.sort_values("ds").copy()
        df["rev_wow_pct"] = df["weekly_revenue"].pct_change() * 100.0
        df["ord_wow_pct"] = df["weekly_orders"].pct_change() * 100.0
        last_rev_growth = float(df["rev_wow_pct"].dropna().iloc[-1]) if df["rev_wow_pct"].notna().any() else 0.0
        last_ord_growth = float(df["ord_wow_pct"].dropna().iloc[-1]) if df["ord_wow_pct"].notna().any() else 0.0

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Gesamtumsatz", f"{total_rev:,.0f}")
        k2.metric("Gesamtbestellungen", f"{total_orders:,.0f}")
        k3.metric("Ø Umsatz/Woche", f"{avg_rev:,.0f}")
        k4.metric("Ø Bestellungen/Woche", f"{avg_orders:,.0f}")

        k5, k6, k7, k8 = st.columns(4)
        k5.metric("Ø Umsatz pro Bestellung", f"{rev_per_order:,.2f}")
        k6.metric("Umsatz WoW (letzte Woche)", f"{last_rev_growth:,.2f} %")
        k7.metric("Orders WoW (letzte Woche)", f"{last_ord_growth:,.2f} %")
        k8.metric("Wochen mit 0 Umsatz / 0 Orders", f"{int((df['weekly_revenue']==0).sum())} / {int((df['weekly_orders']==0).sum())}")

        c1, c2 = st.columns(2)
        with c1:
            st.write("Umsatz pro Woche")
            st.line_chart(df.set_index("ds")["weekly_revenue"])
        with c2:
            st.write("Bestellungen pro Woche")
            st.line_chart(df.set_index("ds")["weekly_orders"])

        st.write("Top 10 Umsatz-Wochen")
        top_rev = df.sort_values("weekly_revenue", ascending=False).head(10)[["ds", "weekly_revenue"]].copy()
        top_rev["ds"] = top_rev["ds"].dt.date.astype(str)
        st.dataframe(top_rev, use_container_width=True)

elif page == "4) Forecast + Backtest":
    st.write("## Step 3/4: Model + Plot (Forecast + Holdout Backtest)")

    dfw = load_weekly()
    if dfw.empty:
        st.warning("Keine Daten. Erst importieren.")
    else:
        # --- Modellwahl ---
        model_label = st.selectbox("Modell", ["Prophet (Data & AI)", "SARIMAX (klassische Baseline)"], index=0)
        model_name = "prophet" if model_label.startswith("Prophet") else "sarimax"

        c_left, c_right = st.columns([2, 3], vertical_alignment="center")
        with c_left:
            target_label = st.selectbox("Zielvariable", ["Umsatz pro Woche", "Bestellungen pro Woche"])
        with c_right:
            st.write("Umsatz = Sum(Quantity × UnitPrice), Bestellungen = Anzahl eindeutiger Invoices pro Woche.")

        target = "weekly_revenue" if target_label.startswith("Umsatz") else "weekly_orders"

        c_left, c_right = st.columns([2, 3], vertical_alignment="center")
        with c_left:
            horizon_weeks = st.slider("Prognose-Horizont (Wochen)", 4, 104, 26)
        with c_right:
            st.write("Für Demo/Prüfung sind 13–26 Wochen oft ideal.")

        c_left, c_right = st.columns([2, 3], vertical_alignment="center")
        with c_left:
            test_weeks = st.slider("Backtest (Holdout): letzte Wochen als Test", 4, 52, 12)
        with c_right:
            st.write("Holdout misst die Prognosegüte auf wirklich 'unbekannten' Wochen.")

        # --- Prophet Parameter ---
        if model_name == "prophet":
            default_mode = "Multiplikativ (Prozentual)" if target == "weekly_revenue" else "Additiv (fester Betrag)"

            c_left, c_right = st.columns([2, 3], vertical_alignment="center")
            with c_left:
                seasonality_label = st.selectbox(
                    "Saisonalitätstyp (Prophet)",
                    ["Additiv (fester Betrag)", "Multiplikativ (Prozentual)"],
                    index=0 if default_mode.startswith("Additiv") else 1,
                )
            with c_right:
                st.write("Optimierung: Wenn Ausschläge bei hohen Werten größer sind → multiplikativ.")

            seasonality_mode = "additive" if seasonality_label.startswith("Additiv") else "multiplicative"

            c_left, c_right = st.columns([2, 3], vertical_alignment="center")
            with c_left:
                cps = st.slider("Trend-Flexibilität (changepoint_prior_scale)", 0.01, 1.0, 0.05)
            with c_right:
                st.write("Overfitting → runter (0.02–0.08). Underfitting → rauf (0.1–0.3).")

            c_left, c_right = st.columns([2, 3], vertical_alignment="center")
            with c_left:
                sps = st.slider("Saison-Stärke (seasonality_prior_scale)", 0.01, 10.0, 5.0 if target=="weekly_revenue" else 3.0)
            with c_right:
                st.write("Saison zu stark → runter (0.5–3). Saison zu flach → rauf (3–8).")

        # --- SARIMAX Parameter ---
        else:
            st.info("SARIMAX ist eine klassische Zeitreihen-Baseline. Bei ~100 Wochen oft stabiler als NN-Modelle.")
            c1, c2, c3 = st.columns(3)
            with c1:
                p = st.slider("AR (p)", 0, 3, 1)
            with c2:
                d = st.slider("Differenzierung (d)", 0, 2, 1)
            with c3:
                q = st.slider("MA (q)", 0, 3, 1)

            use_seasonal = st.checkbox("Jahres-Saisonalität (52 Wochen) aktivieren", value=True)

            if use_seasonal:
                c1, c2, c3 = st.columns(3)
                with c1:
                    P = st.slider("Seasonal AR (P)", 0, 2, 0)
                with c2:
                    D = st.slider("Seasonal Diff (D)", 0, 1, 1)
                with c3:
                    Q = st.slider("Seasonal MA (Q)", 0, 2, 1)
                seasonal_order = (int(P), int(D), int(Q), 52)
            else:
                seasonal_order = (0, 0, 0, 0)

        # --- Run ---
        if st.button("Train + Backtest starten"):
            df = dfw.sort_values("ds")[["ds", target]].rename(columns={target: "y"}).copy()
            df["ds"] = pd.to_datetime(df["ds"])
            df["y"] = df["y"].astype(float)

            try:
                if model_name == "prophet":
                    res = prophet_backtest(
                        df=df,
                        test_weeks=int(test_weeks),
                        horizon_weeks=int(horizon_weeks),
                        week_freq=WEEK_FREQ,
                        seasonality_mode=seasonality_mode,
                        cps=float(cps),
                        sps=float(sps),
                        yearly=YEARLY,
                    )
                    params = {
                        "model": "prophet",
                        "seasonality_mode": seasonality_mode,
                        "changepoint_prior_scale": float(cps),
                        "seasonality_prior_scale": float(sps),
                        "yearly_seasonality": True,
                        "week_freq": WEEK_FREQ,
                    }
                else:
                    res = backtest_holdout_sarimax(
                        df=df,
                        test_weeks=int(test_weeks),
                        horizon_weeks=int(horizon_weeks),
                        week_freq=WEEK_FREQ,
                        order=(int(p), int(d), int(q)),
                        seasonal_order=seasonal_order,
                    )
                    params = {
                        "model": "sarimax",
                        "order": [int(p), int(d), int(q)],
                        "seasonal_order": list(seasonal_order),
                        "week_freq": WEEK_FREQ,
                    }

                # --- Metrics ---
                m1, m2, m3 = st.columns(3)
                m1.metric("MAE", f"{res.metrics['MAE']:.2f}")
                m2.metric("MAPE", f"{res.metrics['MAPE_%']:.2f} %")
                m3.metric("RMSE", f"{res.metrics['RMSE']:.2f}")

                # --- Plot ---
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(df["ds"], df["y"], label="Actual")
                ax.plot(res.forecast["ds"], res.forecast["yhat"], label="Forecast")
                ax.fill_between(
                    res.forecast["ds"],
                    res.forecast["yhat_lower"],
                    res.forecast["yhat_upper"],
                    alpha=0.2,
                    label="Interval",
                )

                ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=10))
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
                ax.tick_params(axis="x", labelrotation=45, labelsize=9)
                ax.tick_params(axis="y", labelsize=9)
                ax.set_title(f"{target_label} – {model_label}", fontsize=14)
                ax.legend()
                fig.tight_layout()
                st.pyplot(fig, clear_figure=True)

                # --- Save run to DB ---
                con = sqlite3.connect(str(DB_PATH), check_same_thread=False)
                cur = con.cursor()

                cur.execute(
                    "INSERT INTO model_runs (target, params_json, train_start, train_end, test_weeks, horizon_weeks, week_freq, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        target,
                        json.dumps(params),
                        str(pd.to_datetime(res.train_df['ds']).min().date()),
                        str(pd.to_datetime(res.train_df['ds']).max().date()),
                        int(test_weeks),
                        int(horizon_weeks),
                        WEEK_FREQ,
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    ),
                )
                run_id = cur.lastrowid

                for k, v in res.metrics.items():
                    cur.execute(
                        "INSERT INTO metrics (run_id, metric_name, metric_value, created_at) VALUES (?, ?, ?, ?)",
                        (run_id, k, float(v), datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                    )

                fc = res.forecast.copy()
                fc["ds"] = pd.to_datetime(fc["ds"]).dt.strftime("%Y-%m-%d")
                rows = list(fc[["ds", "yhat", "yhat_lower", "yhat_upper"]].itertuples(index=False, name=None))
                cur.executemany(
                    "INSERT INTO forecasts (run_id, ds, yhat, yhat_lower, yhat_upper) VALUES (?, ?, ?, ?, ?)",
                    [(int(run_id), *r) for r in rows],
                )

                con.commit()
                con.close()

                st.success(f"Run gespeichert: run_id={run_id}")

            except Exception as e:
                st.error(str(e))

elif page == "5) DB Overview":
    st.write("## DB Overview (SQLite)")
    weekly_kpis, model_runs, metrics, forecasts = get_data()

    st.write("### weekly_kpis (Top 50)")
    st.dataframe(weekly_kpis.head(50), use_container_width=True)

    with st.expander("transactions_raw (Top 50)"):
        raw = read_sql(
            "SELECT InvoiceDate, InvoiceNo, Quantity, UnitPrice, Country, revenue "
            "FROM transactions_raw ORDER BY id DESC LIMIT 50"
        )
        st.dataframe(raw, use_container_width=True)

    st.write("### model_runs (params_json enthält Modelltyp)")
    st.dataframe(model_runs, use_container_width=True)

    st.write("### metrics")
    st.dataframe(metrics.head(200), use_container_width=True)

    st.write("### forecasts (letzte 200)")
    st.dataframe(forecasts.head(200), use_container_width=True)

else:
    st.warning("Unbekannte Seite in Navigation.")
