import pandas as pd
import os

# --- KONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, '..', 'data', 'raw')
PROCESSED_DIR = os.path.join(BASE_DIR, '..', 'data', 'processed')

EXCEL_FILE = "online_retail_II.xlsx"

def convert_excel():
    print("START: Extrahiere Excel Sheets...")
    
    excel_path = os.path.join(RAW_DIR, EXCEL_FILE)
    
    # Sicherstellen, dass Ausgabeordner existiert
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    # Mapping: Excel-Sheet-Name -> Gewünschter CSV-Name
    sheets = {
        'Year 2009-2010': 'sales_2009_2010.csv',
        'Year 2010-2011': 'sales_2010_2011.csv'
    }
    
    for sheet_name, csv_name in sheets.items():
        print(f"   Lese Sheet '{sheet_name}'...")
        try:
            # engine='openpyxl' ist Standard für xlsx
            df = pd.read_excel(excel_path, sheet_name=sheet_name, engine='openpyxl')
            
            output_path = os.path.join(PROCESSED_DIR, csv_name)
            
            # Speichern als CSV
            df.to_csv(output_path, index=False)
            print(f"  Gespeichert als: {csv_name} ({len(df)} Zeilen)")
            
        except Exception as e:
            print(f"  FEHLER bei {sheet_name}: {e}")

    print("FERTIG! Die CSV-Dateien liegen bereit.")

if __name__ == "__main__":
    convert_excel()