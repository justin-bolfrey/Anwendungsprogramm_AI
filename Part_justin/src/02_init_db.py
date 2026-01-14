import pandas as pd
import sqlite3
import os

# --- KONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_DIR = os.path.join(BASE_DIR, '..', 'data', 'processed')
DB_PATH = os.path.join(BASE_DIR, '..', 'database', 'enterprise_data.db')

CSV_FILES = ['sales_2009_2010.csv', 'sales_2010_2011.csv']

def init_database():
    print(" START: Datenbank-Pipeline (aus CSV)...")
    
    # 1. Daten laden
    dfs = []
    for f in CSV_FILES:
        path = os.path.join(CSV_DIR, f)
        print(f"   Lese CSV: {f}...")
        df = pd.read_csv(path, low_memory=False)
        dfs.append(df)
        
    df_all = pd.concat(dfs, ignore_index=True)
    print(f"   --> {len(df_all)} Zeilen geladen.")

    # 2. Bereinigung (Cleaning)
    print("    Bereinige Daten...")
    
    # Spaltennamen säubern
    df_all.columns = df_all.columns.str.strip()
    
    # Fehlende IDs behandeln (-1 für unbekannte Kunden)
    df_all['Customer ID'] = df_all['Customer ID'].fillna(-1).astype(int)
    
    # Datum konvertieren (Essentiell für Prophet!)
    df_all['InvoiceDate'] = pd.to_datetime(df_all['InvoiceDate'])
    
    # Duplikate entfernen 
    # df_all = df_all.drop_duplicates() 

    # 3. Modellierung
    
    # --- Dimension: Customers ---
    print("    Baue Tabelle: customers")
    df_customers = df_all[['Customer ID', 'Country']].drop_duplicates()
    df_customers = df_customers.drop_duplicates(subset=['Customer ID'], keep='last')
    df_customers.columns = ['customer_id', 'country']
    
    # --- Dimension: Products ---
    print("    Baue Tabelle: products")
    df_products = df_all[['StockCode', 'Description']].drop_duplicates()
    df_products = df_products.drop_duplicates(subset=['StockCode'], keep='last')
    df_products.columns = ['stock_code', 'description']
    
    # --- Fakten: Sales ---
    print("    Baue Tabelle: sales")
    df_sales = df_all[['Invoice', 'StockCode', 'Quantity', 'InvoiceDate', 'Price', 'Customer ID']]
    df_sales.columns = ['invoice', 'stock_code', 'quantity', 'invoice_date', 'price', 'customer_id']

    # 4. Speichern (SQLite)
    print(f"    Speichere in DB: {DB_PATH}")
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    
    df_customers.to_sql('customers', conn, if_exists='replace', index=False)
    df_products.to_sql('products', conn, if_exists='replace', index=False)
    df_sales.to_sql('sales', conn, if_exists='replace', index=False)
    
    # Indizes für Performance 
    conn.execute("CREATE INDEX IF NOT EXISTS idx_sales_date ON sales(invoice_date)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_sales_cust ON sales(customer_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_sales_prod ON sales(stock_code)")
    
    conn.close()
    print(" FERTIG! Datenbank steht bereit.")

if __name__ == "__main__":
    init_database()