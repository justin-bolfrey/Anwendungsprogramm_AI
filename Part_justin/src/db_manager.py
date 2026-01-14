import sqlite3
import pandas as pd
import os

# Pfad zur Datenbank
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, '..', 'database', 'enterprise_data.db')

def get_connection():
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"Datenbank nicht gefunden unter: {DB_PATH}. Bitte erst Pipeline ausführen!")
    return sqlite3.connect(DB_PATH)

def get_sales_data(start_date=None, end_date=None, country=None):
    conn = get_connection()
    
    # Basis-Query mit JOINs 
    query = """
    SELECT 
        s.invoice_date,
        p.description as produkt,
        c.country as land,
        (s.quantity * s.price) as umsatz
    FROM sales s
    JOIN products p ON s.stock_code = p.stock_code
    JOIN customers c ON s.customer_id = c.customer_id
    WHERE s.quantity > 0  -- Wir filtern Stornos für die Umsatzanzeige raus!
    """
    
    params = []
    
    # Dynamische Filter hinzufügen
    if start_date:
        query += " AND s.invoice_date >= ?"
        params.append(start_date)
    
    if end_date:
        query += " AND s.invoice_date <= ?"
        params.append(end_date)
        
    if country:
        query += " AND c.country = ?"
        params.append(country)
        
    df = pd.read_sql(query, conn, params=params)
    conn.close()
    
    # Datum in datetime umwandeln 
    if not df.empty:
        df['invoice_date'] = pd.to_datetime(df['invoice_date'])
        
    return df

def get_data_for_prophet(product_name):
    conn = get_connection()
    
    query = """
    SELECT 
        DATE(s.invoice_date) as ds, 
        SUM(s.quantity * s.price) as y
    FROM sales s
    JOIN products p ON s.stock_code = p.stock_code
    WHERE p.description = ? 
      AND s.quantity > 0
    GROUP BY DATE(s.invoice_date)
    ORDER BY ds
    """
    
    df = pd.read_sql(query, conn, params=(product_name,))
    conn.close()
    

    df['ds'] = pd.to_datetime(df['ds'])
    
    return df
# All Products
def get_all_products():
    conn = get_connection()
    df = pd.read_sql("SELECT DISTINCT description FROM products ORDER BY description", conn)
    conn.close()
    return df['description'].tolist()
# All Countries
def get_all_countries():
    conn = get_connection()
    df = pd.read_sql("SELECT DISTINCT country FROM customers ORDER BY country", conn)
    conn.close()
    return df['country'].tolist()

# All Customers
def get_all_customers():
    conn = get_connection()
    df = pd.read_sql("SELECT DISTINCT customer_id FROM customers ORDER BY customer_id", conn)
    conn.close()
    return df['customer_id'].tolist()

# All Invoices
def get_all_invoices():
    conn = get_connection()
    df = pd.read_sql("SELECT DISTINCT invoice FROM sales ORDER BY invoice", conn)
    conn.close()
    return df['invoice'].tolist()



# 1. KPIs für die Scorecard oben im Dashboard
def get_dashboard_kpis(start_date=None, end_date=None, country=None):
    conn = get_connection()
    
    # Basis-Query
    query = """
    SELECT 
        SUM(s.quantity * s.price) as total_revenue,
        COUNT(DISTINCT s.invoice) as total_orders,
        COUNT(DISTINCT s.customer_id) as active_customers,
        AVG(s.quantity * s.price) as avg_order_value
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
        
    cursor = conn.cursor()
    cursor.execute(query, params)
    row = cursor.fetchone()
    conn.close()
    
    return {
        "revenue": row[0] or 0,
        "orders": row[1] or 0,
        "customers": row[2] or 0,
        "aov": row[3] or 0
    }

# 2. Zeitreihe für Line-Charts (Umsatz pro Monat)
def get_monthly_revenue(year=None, country=None):
    conn = get_connection()

    query = """
    SELECT 
        strftime('%Y-%m', s.invoice_date) AS monat,
        SUM(s.quantity * s.price) AS umsatz
    FROM sales s
    JOIN customers c ON s.customer_id = c.customer_id
    WHERE s.quantity > 0
    """

    params = []

    if year:
        query += " AND strftime('%Y', s.invoice_date) = ?"
        params.append(str(year))

    if country:
        query += " AND c.country = ?"
        params.append(country)

    query += " GROUP BY monat ORDER BY monat"

    df = pd.read_sql(query, conn, params=params)
    conn.close()
    return df


# 3. Top Produkte 
def get_top_products(limit=10, country=None):
    conn = get_connection()

    query = """
    SELECT 
        p.description,
        SUM(s.quantity * s.price) AS umsatz,
        SUM(s.quantity) AS verkaufte_menge
    FROM sales s
    JOIN products p ON s.stock_code = p.stock_code
    JOIN customers c ON s.customer_id = c.customer_id
    WHERE s.quantity > 0
      AND p.description IS NOT NULL
      AND TRIM(p.description) <> ''
      AND LENGTH(TRIM(p.description)) >= 5
      AND LOWER(TRIM(p.description)) NOT IN (
          'damages','damaged','missing','manual','dotcom','check','?','none', 'found', 'smashed', '???missing'
      )
    """

    params = []

    if country:
        query += " AND c.country = ?"
        params.append(country)

    query += """
    GROUP BY p.description
    ORDER BY umsatz DESC
    LIMIT ?
    """

    params.append(limit)

    df = pd.read_sql(query, conn, params=params)
    conn.close()
    return df



# 4. Hourly Sales 
def get_hourly_activity(country=None):
    conn = get_connection()

    query = """
    SELECT 
        strftime('%H', s.invoice_date) AS stunde,
        COUNT(DISTINCT s.invoice) AS anzahl_bestellungen
    FROM sales s
    JOIN customers c ON s.customer_id = c.customer_id
    WHERE s.quantity > 0
    """

    params = []

    if country:
        query += " AND c.country = ?"
        params.append(country)

    query += " GROUP BY stunde ORDER BY stunde"

    df = pd.read_sql(query, conn, params=params)
    conn.close()
    return df

def get_cancellations(limit=100, country=None):
    conn = get_connection()

    query = """
    SELECT 
        s.invoice_date, 
        p.description, 
        s.quantity, 
        c.country,
        (s.quantity * s.price) AS erstattung
    FROM sales s
    JOIN products p ON s.stock_code = p.stock_code
    JOIN customers c ON s.customer_id = c.customer_id
    WHERE s.quantity < 0
    """

    params = []

    if country:
        query += " AND c.country = ?"
        params.append(country)

    query += """
    ORDER BY s.invoice_date DESC
    LIMIT ?
    """

    params.append(limit)

    df = pd.read_sql(query, conn, params=params)
    conn.close()
    return df


def get_top_countries_by_revenue(top_n=10):
    conn = get_connection()
    query = """
    SELECT 
        c.country, 
        SUM(s.quantity * s.price) as umsatz
    FROM sales s 
    JOIN customers c ON s.customer_id = c.customer_id
    WHERE s.quantity > 0 
    GROUP BY c.country 
    ORDER BY umsatz DESC 
    LIMIT ?
    """
    df = pd.read_sql(query, conn, params=(top_n,))
    conn.close()
    return df

def get_db_status():
    conn = get_connection()
    df = pd.read_sql(""" 
        SELECT
            COUNT(*) AS sales_rows,
            MIN(invoice_date) AS min_date,
            MAX(invoice_date) AS max_date
        FROM sales
    """, conn)
    conn.close()
    if df.empty:
        return {"sales_rows": 0, "min_date": None, "max_date": None}
    return {
        "sales_rows": int(df.loc[0, "sales_rows"]),
        "min_date": df.loc[0, "min_date"],
        "max_date": df.loc[0, "max_date"],
    }

   

def get_weekly_revenue(start_date=None, end_date=None, country=None):
    conn = get_connection()
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

# --- NEUE FUNKTION FÜR PROPHET ---
def get_cleaned_prophet_data():
    conn = get_connection()
    
    query = """
    SELECT
        date(s.invoice_date, 'weekday 0') AS ds,
        SUM(s.quantity * s.price) AS y
    FROM sales s
    JOIN products p ON s.stock_code = p.stock_code
    WHERE s.quantity > 0 
      AND s.price > 0
      AND s.stock_code NOT IN ('POST', 'DOT', 'M', 'BANK CHARGES', 'D', 'CRUK')
      AND LENGTH(p.description) > 3
    GROUP BY date(s.invoice_date, 'weekday 0')
    ORDER BY ds
    """
    
    df = pd.read_sql(query, conn)
    conn.close()
    
    if not df.empty:
        df["ds"] = pd.to_datetime(df["ds"])
        df["y"] = df["y"].astype(float)
        
        df = df[df['y'] > 10]
        
    return df




# SYSTEM-CHECK 

if __name__ == "__main__":
    # Einstellungen für schönere Ausgabe im Terminal
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    
    print("\n STARTE SYSTEM-CHECK (Deine Version)...\n")

    # --- BLOCK 1: Die Basis-Funktionen ---
    print("--- 1. Basis-Listen (Deine Funktionen) ---")
    
    try:
        # Test: get_all_products
        prods = get_all_products()
        print(f"✅ Produkte: {len(prods)} gefunden. (Bsp: {prods[:2]})")
    except Exception as e: print(f"❌ Fehler bei get_all_products: {e}")

    try:
        # Test: get_all_countries
        countries = get_all_countries()
        print(f"✅ Länder:   {len(countries)} gefunden. (Bsp: {countries[:2]})")
    except Exception as e: print(f"❌ Fehler bei get_all_countries: {e}")

    try:
        # Test: get_all_invoices 
        invoices = get_all_invoices()
        count = len(invoices) if isinstance(invoices, list) else len(invoices)
        print(f"✅ Invoices: {count} Rechnungen gefunden.")
    except NameError:
        print("⚠️ Funktion 'get_all_invoices' nicht gefunden (evtl. Tippfehler?).")
    except Exception as e: print(f"❌ Fehler bei get_all_invoices: {e}")

    try:
        # Test: get_all_customers 
        custs = get_all_customers()
        count_c = len(custs) if isinstance(custs, list) else len(custs)
        print(f"✅ Customers: {count_c} Kunden gefunden.")
    except NameError:
        print("⚠️ Funktion 'get_all_customers' nicht gefunden.")
    except Exception as e: print(f"❌ Fehler bei get_all_customers: {e}")


    # --- BLOCK 2: Die neuen Analyse-Funktionen ---
    print("\n--- 2. Analyse & Charts (Die Ergänzungen) ---")

    try:
        print("Test: get_top_countries_by_revenue...")
        print(get_top_countries_by_revenue(top_n=3))
    except Exception as e: print(f"❌ Fehler: {e}")

    try:
        print("\nTest: get_cancellations (Stornos)...")
        print(get_cancellations(limit=3)[['description', 'quantity', 'erstattung']])
    except Exception as e: print(f"❌ Fehler: {e}")

    try:
        print("\nTest: get_dashboard_kpis...")
        print(get_dashboard_kpis())
    except Exception as e: print(f"❌ Fehler: {e}")


    # --- BLOCK 3: KI Vorbereitung ---
    print("\n--- 3. KI / Prophet ---")
    try:
        test_prod = prods[0] if 'prods' in locals() and len(prods) > 0 else "WHITE HANGING HEART T-LIGHT HOLDER"
        print(f"Daten für '{test_prod}':")
        print(get_data_for_prophet(test_prod).head(2))
    except Exception as e: print(f"❌ Fehler: {e}")

    print("\n✅ CHECK ABGESCHLOSSEN.")