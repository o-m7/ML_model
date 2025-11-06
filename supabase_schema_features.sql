-- ========================================
-- SUPABASE SCHEMA FOR REAL-TIME FEATURES
-- ========================================

-- Drop existing table if needed (be careful in production!)
-- DROP TABLE IF EXISTS public.features CASCADE;

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

-- Enable Row Level Security (optional)
ALTER TABLE public.features ENABLE ROW LEVEL SECURITY;

-- Create policy to allow authenticated users to read
CREATE POLICY "Allow authenticated read access" 
    ON public.features FOR SELECT 
    USING (auth.role() = 'authenticated' OR auth.role() = 'anon');

-- Create policy to allow service role to insert/update
CREATE POLICY "Allow service role full access" 
    ON public.features FOR ALL 
    USING (auth.role() = 'service_role');

-- Create function to automatically update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger for updated_at
DROP TRIGGER IF EXISTS update_features_updated_at ON public.features;
CREATE TRIGGER update_features_updated_at
    BEFORE UPDATE ON public.features
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Create a view for latest features per symbol/timeframe
CREATE OR REPLACE VIEW public.latest_features AS
SELECT DISTINCT ON (symbol, timeframe)
    id,
    symbol,
    timeframe,
    timestamp,
    open,
    high,
    low,
    close,
    volume,
    features,
    feature_count,
    data_source,
    created_at
FROM public.features
ORDER BY symbol, timeframe, timestamp DESC;

-- Grant permissions
GRANT SELECT ON public.latest_features TO anon, authenticated;
GRANT ALL ON public.features TO service_role;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO service_role;

-- Comment on table
COMMENT ON TABLE public.features IS 'Real-time calculated features for trading signals';
COMMENT ON COLUMN public.features.features IS 'JSONB object containing all calculated technical indicators and features';
COMMENT ON VIEW public.latest_features IS 'Latest feature data for each symbol/timeframe combination';

