#!/usr/bin/env python3
"""
Create Supabase features table programmatically.

Usage:
    python setup_supabase_table.py
"""

import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

# SQL to create features table
CREATE_TABLE_SQL = """
-- Create features table for storing real-time calculated features
CREATE TABLE IF NOT EXISTS public.features (
    id BIGSERIAL PRIMARY KEY,
    
    -- Identifiers
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    
    -- OHLCV Data
    open DOUBLE PRECISION,
    high DOUBLE PRECISION,
    low DOUBLE PRECISION,
    close DOUBLE PRECISION,
    volume DOUBLE PRECISION,
    
    -- Features (stored as JSONB for flexibility)
    features JSONB NOT NULL,
    
    -- Metadata
    feature_count INTEGER,
    data_source TEXT DEFAULT 'Polygon API',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Constraints
    UNIQUE(symbol, timeframe, timestamp)
);

-- Create indexes for fast queries
CREATE INDEX IF NOT EXISTS idx_features_symbol_timeframe 
    ON public.features(symbol, timeframe);

CREATE INDEX IF NOT EXISTS idx_features_timestamp 
    ON public.features(timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_features_symbol_timeframe_timestamp 
    ON public.features(symbol, timeframe, timestamp DESC);

-- Create index on features JSONB for specific feature lookups
CREATE INDEX IF NOT EXISTS idx_features_jsonb 
    ON public.features USING GIN (features);
"""

def main():
    """Create the features table."""
    print("🔧 Setting up Supabase features table...")
    
    # Get credentials
    supabase_url = os.getenv('SUPABASE_URL')
    supabase_key = os.getenv('SUPABASE_SERVICE_KEY')
    
    if not supabase_url or not supabase_key:
        print("❌ Error: SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in .env")
        return 1
    
    print(f"✓ Supabase URL: {supabase_url}")
    
    # Connect to Supabase
    supabase = create_client(supabase_url, supabase_key)
    
    print("\n📝 Creating table and indexes...")
    print("   (This requires SQL execution via Supabase Dashboard)")
    print("\n" + "="*70)
    print("INSTRUCTIONS:")
    print("="*70)
    print("\n1. Go to your Supabase project:")
    print(f"   {supabase_url}")
    print("\n2. Navigate to: SQL Editor")
    print("\n3. Copy and paste this SQL:\n")
    print(CREATE_TABLE_SQL)
    print("\n4. Click 'Run' to execute")
    print("\n5. Come back and run the feature uploader!")
    print("\n" + "="*70)
    
    # Save SQL to file for easy access
    with open('supabase_schema_features.sql', 'w') as f:
        f.write(CREATE_TABLE_SQL)
    
    print("\n✓ SQL saved to: supabase_schema_features.sql")
    print("\nAlternatively, you can run:")
    print("  psql <connection_string> < supabase_schema_features.sql")
    
    return 0

if __name__ == '__main__':
    exit(main())

