"""
backup.py - Backup client for the table 'api_tok'
"""

import os
import requests
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

TABLE_NAME = os.getenv("TABLE_NAME")
SUPABASE_URL = os.getenv("SUPA_URL")
SUPABASE_KEY = os.getenv("SUPA_KEY")

def backup_data() -> None:    
    curr = os.path.dirname(__file__)
    data_dir = os.path.abspath(os.path.join(curr, "..", "backups/data"))
    
    RAW_CSV_PATH = os.path.join(data_dir, "api_tok.csv")
    
    API_TABLE_ENDPOINT = f"{SUPABASE_URL}/rest/v1/{TABLE_NAME}"
    
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
    }
    
    query = {
    "select": "*",
    "order": "user_id.asc" 
    }
    
    api_res = requests.get(API_TABLE_ENDPOINT, headers=headers, params=query)
    
    if api_res.status_code == 200:
        data = api_res.json()
        pd.DataFrame(data).to_csv(RAW_CSV_PATH, index=False, mode='w')
        print(f"Backed up api_tok TABLE to {RAW_CSV_PATH}")
    else :
        print(f"Failed to backup : {api_res.status_code} - {api_res.text}")

if __name__ == "__main__":
    backup_data()