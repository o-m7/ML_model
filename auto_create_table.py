#!/usr/bin/env python3
"""
Auto-create Supabase trading_signals table
"""
import os
import sys
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

def create_table():
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    
    if not supabase_url or not supabase_key:
        print("ERROR: Missing SUPABASE_URL or SUPABASE_KEY")
        return False
    
    client = create_client(supabase_url, supabase_key)
    
    # Read SQL file
    with open('create_supabase_schema.sql', 'r') as f:
        sql = f.read()
    
    print("Attempting to create trading_signals table...")
    print("="*80)
    
    try:
        # Try to execute SQL (this may fail if permissions are insufficient)
        result = client.postgrest.rpc('exec', {'sql': sql}).execute()
        print("✅ Table created successfully!")
        return True
    except Exception as e:
        print(f"⚠️  Auto-creation failed: {e}")
        print("\nPlease manually create the table:")
        print(f"1. Go to: {supabase_url.replace('https://', 'https://app.')}/project/_/sql")
        print("2. Copy the SQL from: create_supabase_schema.sql")
        print("3. Paste and run in SQL Editor")
        print("\nSQL Preview:")
        print("-"*80)
        print(sql[:500] + "...")
        return False

if __name__ == "__main__":
    create_table()
