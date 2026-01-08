# Retail Forecast Cockpit (Vorlesungsstil) – ohne Anomalien

Dateien:
- create_db_retail.py  → DB anlegen + Import + get_data()
- model_prophet.py     → Prophet Forecast + Backtest
- app.py               → Streamlit UI im Step-Format

## Setup (Windows)

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install streamlit pandas numpy matplotlib prophet openpyxl
```

## Start

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## DBeaver
Öffne `retail_prophet.db` (SQLite):
- transactions_raw
- weekly_kpis
- model_runs, metrics, forecasts


## Dependencies

- streamlit
- pandas
- numpy
- matplotlib
- prophet
- statsmodels
- openpyxl
