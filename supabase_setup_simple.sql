-- ============================================================================
-- SIMPLIFIED SUPABASE SETUP - Run this in SQL Editor
-- ============================================================================

-- ML Models table
CREATE TABLE IF NOT EXISTS ml_models (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    model_path TEXT NOT NULL,
    features JSONB NOT NULL,
    num_features INTEGER NOT NULL,
    parameters JSONB NOT NULL,
    backtest_results JSONB NOT NULL,
    win_rate DECIMAL(5,2),
    profit_factor DECIMAL(5,2),
    sharpe_ratio DECIMAL(5,2),
    max_drawdown DECIMAL(5,2),
    total_trades INTEGER,
    status TEXT DEFAULT 'production_ready',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(symbol, timeframe)
);

-- Live signals table
CREATE TABLE IF NOT EXISTS live_signals (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    model_id UUID REFERENCES ml_models(id) ON DELETE CASCADE,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    signal_type TEXT NOT NULL,
    confidence DECIMAL(5,4),
    edge DECIMAL(5,4),
    entry_price DECIMAL(12,5),
    stop_loss DECIMAL(12,5),
    take_profit DECIMAL(12,5),
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    status TEXT DEFAULT 'active',
    expires_at TIMESTAMPTZ
);

-- Trades table
CREATE TABLE IF NOT EXISTS trades (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    signal_id UUID REFERENCES live_signals(id) ON DELETE SET NULL,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    direction TEXT NOT NULL,
    entry_time TIMESTAMPTZ,
    entry_price DECIMAL(12,5),
    exit_time TIMESTAMPTZ,
    exit_price DECIMAL(12,5),
    stop_loss DECIMAL(12,5),
    take_profit DECIMAL(12,5),
    pnl DECIMAL(12,2),
    pnl_pct DECIMAL(8,4),
    r_multiple DECIMAL(6,2),
    exit_reason TEXT,
    status TEXT DEFAULT 'open',
    bars_held INTEGER,
    confidence DECIMAL(5,4),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Performance metrics table
CREATE TABLE IF NOT EXISTS performance_metrics (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    period TEXT NOT NULL,
    total_trades INTEGER DEFAULT 0,
    winning_trades INTEGER DEFAULT 0,
    losing_trades INTEGER DEFAULT 0,
    win_rate DECIMAL(5,2),
    profit_factor DECIMAL(5,2),
    total_pnl DECIMAL(12,2),
    avg_win DECIMAL(10,2),
    avg_loss DECIMAL(10,2),
    max_win DECIMAL(10,2),
    max_loss DECIMAL(10,2),
    avg_r_multiple DECIMAL(6,2),
    sharpe_ratio DECIMAL(5,2),
    max_drawdown DECIMAL(5,2),
    period_start TIMESTAMPTZ,
    period_end TIMESTAMPTZ,
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(symbol, timeframe, period, period_start)
);

-- Ensemble metadata table (Phase 4.1)
CREATE TABLE IF NOT EXISTS ensemble_metadata (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    signal_id UUID REFERENCES live_signals(id) ON DELETE CASCADE,
    symbol TEXT NOT NULL,
    timeframe TEXT,
    strategy TEXT NOT NULL,
    num_models INTEGER NOT NULL,
    agreeing_models INTEGER NOT NULL,
    votes JSONB,
    model_predictions JSONB NOT NULL,
    timestamp TIMESTAMPTZ DEFAULT NOW()
);

-- Economic events table (Phase 4.2)
CREATE TABLE IF NOT EXISTS economic_events (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    event_time TIMESTAMPTZ NOT NULL,
    currency TEXT NOT NULL,
    event_name TEXT NOT NULL,
    impact TEXT NOT NULL,
    blackout_start TIMESTAMPTZ NOT NULL,
    blackout_end TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(event_time, currency, event_name)
);

-- Sentiment data table (Phase 4.3)
CREATE TABLE IF NOT EXISTS sentiment_data (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    symbol TEXT NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    news_sentiment FLOAT,
    reddit_sentiment FLOAT,
    twitter_sentiment FLOAT,
    aggregate_sentiment FLOAT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_ml_models_symbol_tf ON ml_models(symbol, timeframe);
CREATE INDEX IF NOT EXISTS idx_ml_models_status ON ml_models(status);
CREATE INDEX IF NOT EXISTS idx_live_signals_symbol ON live_signals(symbol);
CREATE INDEX IF NOT EXISTS idx_live_signals_status ON live_signals(status);
CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status);
CREATE INDEX IF NOT EXISTS idx_ensemble_metadata_signal ON ensemble_metadata(signal_id);
CREATE INDEX IF NOT EXISTS idx_ensemble_metadata_symbol ON ensemble_metadata(symbol);
CREATE INDEX IF NOT EXISTS idx_economic_events_time ON economic_events(event_time);
CREATE INDEX IF NOT EXISTS idx_economic_events_currency ON economic_events(currency);
CREATE INDEX IF NOT EXISTS idx_sentiment_data_symbol ON sentiment_data(symbol);
CREATE INDEX IF NOT EXISTS idx_sentiment_data_timestamp ON sentiment_data(timestamp);

-- Enable RLS (Row Level Security)
ALTER TABLE ml_models ENABLE ROW LEVEL SECURITY;
ALTER TABLE live_signals ENABLE ROW LEVEL SECURITY;
ALTER TABLE trades ENABLE ROW LEVEL SECURITY;
ALTER TABLE performance_metrics ENABLE ROW LEVEL SECURITY;
ALTER TABLE ensemble_metadata ENABLE ROW LEVEL SECURITY;
ALTER TABLE economic_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE sentiment_data ENABLE ROW LEVEL SECURITY;

-- Allow ALL operations (read, insert, update, delete)
CREATE POLICY "Enable all access to ml_models" ON ml_models FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "Enable all access to live_signals" ON live_signals FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "Enable all access to trades" ON trades FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "Enable all access to performance_metrics" ON performance_metrics FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "Enable all access to ensemble_metadata" ON ensemble_metadata FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "Enable all access to economic_events" ON economic_events FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "Enable all access to sentiment_data" ON sentiment_data FOR ALL USING (true) WITH CHECK (true);

-- ============================================================================
-- DONE! Tables created with full access permissions
-- ============================================================================

