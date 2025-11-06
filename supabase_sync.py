#!/usr/bin/env python3
"""
Sync ONNX models and metadata to Supabase
"""

import json
from pathlib import Path
from supabase import create_client
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Supabase credentials from environment
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Please set SUPABASE_URL and SUPABASE_KEY environment variables in .env file")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


def upload_model_to_storage(onnx_path: Path, metadata_path: Path):
    """Upload ONNX model file to Supabase Storage."""
    
    symbol = onnx_path.parent.name
    timeframe = onnx_path.stem.split('_')[1]
    
    # Read ONNX model
    with open(onnx_path, 'rb') as f:
        onnx_bytes = f.read()
    
    # Upload to Supabase Storage
    storage_path = f"models/{symbol}/{symbol}_{timeframe}.onnx"
    
    try:
        response = supabase.storage.from_('ml_models').upload(
            storage_path,
            onnx_bytes,
            file_options={"content-type": "application/octet-stream", "upsert": "true"}
        )
        print(f"  ✅ Uploaded {storage_path}")
        return storage_path
    except Exception as e:
        # Try updating if already exists
        try:
            response = supabase.storage.from_('ml_models').update(
                storage_path,
                onnx_bytes,
                file_options={"content-type": "application/octet-stream"}
            )
            print(f"  ♻️  Updated {storage_path}")
            return storage_path
        except Exception as e2:
            print(f"  ❌ Upload failed: {e2}")
            raise


def insert_model_metadata(metadata_path: Path, storage_path: str):
    """Insert model metadata into Supabase database."""
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Prepare record for database
    record = {
        'symbol': metadata['symbol'],
        'timeframe': metadata['timeframe'],
        'model_path': storage_path,
        'features': metadata['features'],
        'num_features': metadata['num_features'],
        'parameters': metadata['params'],
        'backtest_results': metadata['backtest_results'],
        'win_rate': metadata['backtest_results']['win_rate'],
        'profit_factor': metadata['backtest_results']['profit_factor'],
        'sharpe_ratio': metadata['backtest_results']['sharpe_ratio'],
        'max_drawdown': metadata['backtest_results']['max_drawdown_pct'],
        'total_trades': metadata['backtest_results']['total_trades'],
        'status': 'production_ready',
        'updated_at': datetime.utcnow().isoformat()
    }
    
    try:
        # Check if exists
        existing = supabase.table('ml_models').select('id').eq('symbol', record['symbol']).eq('timeframe', record['timeframe']).execute()
        
        if existing.data:
            # Update
            response = supabase.table('ml_models').update(record).eq('symbol', record['symbol']).eq('timeframe', record['timeframe']).execute()
            print(f"  ♻️  Updated {record['symbol']} {record['timeframe']} in database")
        else:
            # Insert
            response = supabase.table('ml_models').insert(record).execute()
            print(f"  ✅ Inserted {record['symbol']} {record['timeframe']} into database")
    except Exception as e:
        print(f"  ❌ Database error: {e}")
        raise


def sync_all_models():
    """Sync all ONNX models to Supabase."""
    
    onnx_dir = Path('models_onnx')
    
    if not onnx_dir.exists():
        print("❌ models_onnx directory not found. Run convert_models_to_onnx.py first!")
        return
    
    print("\n" + "="*80)
    print("SYNCING MODELS TO SUPABASE")
    print("="*80 + "\n")
    
    synced = []
    failed = []
    
    for symbol_dir in sorted(onnx_dir.iterdir()):
        if not symbol_dir.is_dir():
            continue
        
        print(f"\nProcessing {symbol_dir.name}...")
        
        for onnx_file in sorted(symbol_dir.glob('*.onnx')):
            metadata_file = onnx_file.with_suffix('.json')
            
            if not metadata_file.exists():
                print(f"  ⚠️  No metadata for {onnx_file.name}")
                continue
            
            try:
                # Upload model file
                storage_path = upload_model_to_storage(onnx_file, metadata_file)
                
                # Insert/update metadata
                insert_model_metadata(metadata_file, storage_path)
                
                synced.append((symbol_dir.name, onnx_file.stem.split('_')[1]))
            except Exception as e:
                failed.append((symbol_dir.name, onnx_file.stem.split('_')[1], str(e)))
    
    print("\n" + "="*80)
    print(f"SYNC COMPLETE: {len(synced)} models synced")
    if failed:
        print(f"FAILED: {len(failed)} models")
    print("="*80 + "\n")
    
    if synced:
        print("✅ SYNCED:")
        for symbol, tf in synced:
            print(f"   {symbol} {tf}")
    
    if failed:
        print("\n❌ FAILED:")
        for symbol, tf, error in failed:
            print(f"   {symbol} {tf} - {error}")


if __name__ == '__main__':
    sync_all_models()

