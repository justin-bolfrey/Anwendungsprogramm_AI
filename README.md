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


## Part_justin – ELT-Pipeline (Data Warehouse Stil)

Ziel: Rohdaten verlustarm laden (ELT). Unknown-Keys werden auf Dummy -1 gesetzt, Stornos bleiben im Raw-Layer und werden erst in den Views gefiltert.

### Verzeichnisstruktur
- `Part_justin/data/raw/online_retail_II.xlsx` → Eingabe (Excel, 2 Sheets)
- `Part_justin/data/processed/` → erzeugte CSVs (`sales_2009_2010.csv`, `sales_2010_2011.csv`)
- `Part_justin/database/enterprise_data.db` → SQLite-DB (Star-Schema)
- `Part_justin/src/`
  - `01_excel_to_csv.py` → Excel → CSV (Extraktion)
  - `02_init_db.py`      → CSV → SQLite (customers, products, sales; Indizes)
  - `db_manager.py`      → Abfrage-Layer/API (z. B. `quantity > 0` für Umsatz)
- `Part_justin/notebooks/00_data_exploration.ipynb` → Explorative Checks/Plots

### Architektur (3 Layer)
- Ingestion (Layer 1): `01_excel_to_csv.py` extrahiert beide Excel-Sheets in CSV (beschleunigt spätere Läufe).
- Transformation/Load (Layer 2): `02_init_db.py` lädt CSVs, säubert Spalten, konvertiert `InvoiceDate` zu datetime, füllt fehlende Customer IDs mit `-1`, baut Star-Schema und Indizes (`invoice_date`, `customer_id`, `stock_code`), ersetzt Tabellen bei jedem Lauf (`if_exists='replace'`).
- Access (Layer 3): `db_manager.py` kapselt SQL-Queries/Views. Umsatz wird on-the-fly als `quantity * price` berechnet; Standard-Queries blenden Stornos via `s.quantity > 0` aus.

### Datenmodell (Star-Schema)
- Fakt `sales`: `invoice`, `stock_code`, `quantity`, `invoice_date`, `price`, `customer_id` (keine persistierte revenue-Spalte; Umsatz wird berechnet).
- Dimension `customers`: `customer_id`, `country` (Dummy -1 = unbekannt/Gast).
- Dimension `products`: `stock_code`, `description` (letzte bekannte Beschreibung je StockCode).
- Stornos/Retouren: verbleiben in `sales` (quantity < 0 / Invoice „C…“). Für Standard-Umsatzreports filtert der View-Layer, für Retouren-Analysen bleiben sie nutzbar.

### Pipeline ausführen
```bash
# 1) Excel → CSV
python Part_justin/src/01_excel_to_csv.py

# 2) CSV → SQLite (Star-Schema)
python Part_justin/src/02_init_db.py
```

Ergebnis: `Part_justin/database/enterprise_data.db` mit `customers`, `products`, `sales` (Dummy-Kunde -1, Stornos enthalten). Für Reports/Modelle den Abfrage-Layer in `db_manager.py` nutzen.
