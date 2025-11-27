-- Trading Signals Table Schema
-- Run this in Supabase SQL Editor

CREATE TABLE IF NOT EXISTS trading_signals (
    id BIGSERIAL PRIMARY KEY,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    model_name TEXT NOT NULL,
    signal_type TEXT NOT NULL,
    confidence NUMERIC NOT NULL,
    entry_market NUMERIC NOT NULL,
    entry_limit NUMERIC NOT NULL,
    take_profit NUMERIC NOT NULL,
    stop_loss NUMERIC NOT NULL,
    order_type TEXT NOT NULL,
    atr NUMERIC NOT NULL,
    spread NUMERIC NOT NULL,
    risk NUMERIC NOT NULL,
    reward NUMERIC NOT NULL,
    rr_ratio NUMERIC NOT NULL,
    current_bid NUMERIC NOT NULL,
    current_ask NUMERIC NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_trading_signals_timestamp ON trading_signals(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_trading_signals_symbol ON trading_signals(symbol);
CREATE INDEX IF NOT EXISTS idx_trading_signals_timeframe ON trading_signals(timeframe);
CREATE INDEX IF NOT EXISTS idx_trading_signals_model ON trading_signals(model_name);

-- Enable Row Level Security (optional)
ALTER TABLE trading_signals ENABLE ROW LEVEL SECURITY;

-- Create policy to allow inserts (adjust as needed)
CREATE POLICY "Allow all inserts" ON trading_signals
    FOR INSERT
    TO public
    WITH CHECK (true);

-- Create policy to allow reads (adjust as needed)
CREATE POLICY "Allow all reads" ON trading_signals
    FOR SELECT
    TO public
    USING (true);
