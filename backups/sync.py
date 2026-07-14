"""
sync.py - This module is responsible for syncing the local CSV files to the Supabase database.
Use this to add bulk data from the csv located in the data/ folder to the Supabase tables. 
This is a one-time utility to populate the database with existing data.
"""

import pandas as pd
import requests
import os
from dotenv import load_dotenv  
load_dotenv()

TABLE_NAME = os.getenv("TABLE_NAME")
SUPABASE_URL = os.getenv("SUPA_URL")
SUPABASE_KEY = os.getenv("SUPA_KEY")

def sync_api_csv_to_supabase() -> None:
    df = pd.read_csv('./data/api_tok.csv')
    print("Synced api_tok CSV to Supabase")

    records = df.to_dict(orient='records')

    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "resolution=merge-duplicates"
    }
        
    endpoint = f"{SUPABASE_URL}/rest/v1/{TABLE_NAME}"
    res = requests.post(endpoint, headers=headers, json=records)

    if res.status_code in [200, 201]:
        print("Successfully synced data!")
    else:
        print(f"Error: {res.status_code} - {res.text}")
        
if __name__ == "__main__":
    sync_api_csv_to_supabase()