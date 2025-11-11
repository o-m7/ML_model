#!/usr/bin/env python3
"""
Download production models from Supabase on Railway startup
"""

import os
import sys
from pathlib import Path
import pickle
from supabase import create_client

# Try to load .env if it exists (local development)
try:
    from dotenv import load_dotenv
    load_dotenv()
except:
    pass  # Railway injects env vars directly

SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')

if not all([SUPABASE_URL, SUPABASE_KEY]):
    print("❌ Missing SUPABASE_URL or SUPABASE_KEY")
    print(f"SUPABASE_URL present: {bool(SUPABASE_URL)}")
    print(f"SUPABASE_KEY present: {bool(SUPABASE_KEY)}")
    sys.exit(0)  # Don't fail deployment, just skip model download

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Models to download
MODELS = [
    ('XAUUSD', '5T'), ('XAUUSD', '15T'), ('XAUUSD', '30T'),
    ('XAGUSD', '5T'), ('XAGUSD', '15T'), ('XAGUSD', '30T'),
    ('EURUSD', '5T'), ('EURUSD', '30T'),
    ('GBPUSD', '5T'), ('GBPUSD', '15T'), ('GBPUSD', '30T'),
    ('AUDUSD', '5T'), ('AUDUSD', '15T'), ('AUDUSD', '30T'),
    ('NZDUSD', '5T'), ('NZDUSD', '15T'), ('NZDUSD', '30T'),
]

def download_models():
    """Download all models from Supabase storage."""
    print("\n" + "="*80)
    print("📦 DOWNLOADING MODELS FROM SUPABASE")
    print("="*80 + "\n")
    
    models_dir = Path('models_production')
    models_dir.mkdir(exist_ok=True)
    
    success_count = 0
    fail_count = 0
    
    for symbol, timeframe in MODELS:
        try:
            # Create symbol directory
            symbol_dir = models_dir / symbol
            symbol_dir.mkdir(exist_ok=True)
            
            # Model filename
            model_filename = f"{symbol}_{timeframe}_PRODUCTION_READY.pkl"
            storage_path = f"{symbol}/{model_filename}"
            local_path = symbol_dir / model_filename
            
            # Skip if already exists
            if local_path.exists():
                print(f"  ✓ {symbol} {timeframe} (cached)")
                success_count += 1
                continue
            
            # Download from Supabase storage
            try:
                file_data = supabase.storage.from_('ml_models').download(storage_path)
                
                # Save locally
                with open(local_path, 'wb') as f:
                    f.write(file_data)
                
                print(f"  ✅ {symbol} {timeframe}")
                success_count += 1
                
            except Exception as e:
                if 'not found' in str(e).lower() or '404' in str(e):
                    print(f"  ⚠️  {symbol} {timeframe} (not in storage)")
                else:
                    print(f"  ❌ {symbol} {timeframe}: {e}")
                fail_count += 1
                
        except Exception as e:
            print(f"  ❌ {symbol} {timeframe}: {e}")
            fail_count += 1
    
    print("\n" + "="*80)
    print(f"✅ Downloaded: {success_count} models")
    if fail_count > 0:
        print(f"⚠️  Failed: {fail_count} models")
    print("="*80 + "\n")
    
    return success_count

if __name__ == "__main__":
    try:
        count = download_models()
        if count == 0:
            print("⚠️  WARNING: No models downloaded!")
            print("System will use fallback predictions.")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error downloading models: {e}")
        sys.exit(0)  # Don't fail deployment, just warn

