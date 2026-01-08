import sqlite3
import pandas as pd
import os

# Pfad zur Datenbank (dynamisch, damit es bei jedem im Team funktioniert)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, '..', 'database', 'enterprise_data.db')

def get_connection():
    """Hilfsfunktion: Erstellt Verbindung zur DB"""
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"Datenbank nicht gefunden unter: {DB_PATH}. Bitte erst Pipeline ausführen!")
    return sqlite3.connect(DB_PATH)

def get_sales_data(start_date=None, end_date=None, country=None):
    """
    Holt Verkaufsdaten gefiltert nach Datum und Land.
    Perfekt für das Dashboard (Teammitglied 2).
    """
    conn = get_connection()
    
    # Basis-Query mit JOINs (Das Stern-Schema zahlt sich aus!)
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
    
    # Datum in datetime umwandeln (sicher ist sicher)
    if not df.empty:
        df['invoice_date'] = pd.to_datetime(df['invoice_date'])
        
    return df

def get_data_for_prophet(product_name):
    """
    Holt Daten speziell für die KI-Vorhersage (Teammitglied 3).
    Format: Datum (ds) und Umsatz (y)
    """
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
    
    # Wichtig für Prophet: Datentypen
    df['ds'] = pd.to_datetime(df['ds'])
    
    return df

def get_all_products():
    """Für die Dropdown-Liste im Dashboard"""
    conn = get_connection()
    df = pd.read_sql("SELECT DISTINCT description FROM products ORDER BY description", conn)
    conn.close()
    return df['description'].tolist()

def get_all_countries():
    """Für die Dropdown-Liste im Dashboard"""
    conn = get_connection()
    df = pd.read_sql("SELECT DISTINCT country FROM customers ORDER BY country", conn)
    conn.close()
    return df['country'].tolist()

# Kleiner Test, wenn man die Datei direkt ausführt
if __name__ == "__main__":
    print("Teste Datenbank-Verbindung...")
    products = get_all_products()
    print(f"Gefunden: {len(products)} Produkte.")
    print("Beispiel-Produkte:", products[:5])