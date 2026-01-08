# Retail Forecast Cockpit (Vorlesungsstil) – ohne Anomalien

Dateien:
- create_db_retail.py  → DB anlegen + Import + get_data()
- model_prophet.py     → Prophet Forecast + Backtest
- app.py               → Streamlit UI im Step-Format

## Part_justin – ELT-Pipeline (Data Warehouse Stil)

Ziel: Rohdaten möglichst verlustarm laden (ELT). Missing Keys werden auf Dummy -1 gesetzt, Stornos/Retouren bleiben im Raw-Layer erhalten; Filterung passiert im View/Abfrage-Layer.

### Verzeichnisstruktur
- `Part_justin/data/raw/online_retail_II.xlsx` → Eingabe (Excel, 2 Sheets)
- `Part_justin/data/processed/` → erzeugte CSVs (`sales_2009_2010.csv`, `sales_2010_2011.csv`)
- `Part_justin/database/enterprise_data.db` → SQLite-DB mit Star-Schema
- `Part_justin/src/`
  - `01_excel_to_csv.py` → Excel → CSV (zwei Sheets)
  - `02_init_db.py`      → CSV → SQLite (customers, products, sales; Indizes)
  - `db_manager.py`      → Abfrage-Helpers/Views (z. B. Filter `quantity > 0` für Umsatz)
- `Part_justin/notebooks/00_data_exploration.ipynb` → Explorative Checks/Plots

### Design-Entscheidungen
- Missing Customer IDs: werden auf `-1` gesetzt, damit Umsatz erhalten bleibt; -1 steht für „unknown/gast“.
- Stornos/Retouren: werden gespeichert (Quantity < 0 oder Invoice „C…“). Standard-Umsatz-Views filtern später mit `quantity > 0`.
- Indizes: auf `sales(invoice_date)`, `sales(customer_id)`, `sales(stock_code)` für schnellere Queries.

### Pipeline ausführen
```bash
# 1) Excel → CSV
python Part_justin/src/01_excel_to_csv.py

# 2) CSV → SQLite (Star-Schema)
python Part_justin/src/02_init_db.py
```

Ergebnis: `Part_justin/database/enterprise_data.db` mit Tabellen `customers`, `products`, `sales` (inkl. Dummy-Kunde -1, inkl. Stornos). Für Reports nutzen Teammitglieder `db_manager.py`, um z. B. Stornos herauszufiltern oder Umsatz zu berechnen.

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
