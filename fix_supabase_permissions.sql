-- Fix Supabase RLS Permissions for Model Upload
-- Run this in your Supabase SQL Editor

-- Drop existing restrictive policies
DROP POLICY IF EXISTS "Allow read access to ml_models" ON ml_models;
DROP POLICY IF EXISTS "Allow read access to live_signals" ON live_signals;
DROP POLICY IF EXISTS "Allow read access to trades" ON trades;
DROP POLICY IF EXISTS "Allow read access to performance_metrics" ON performance_metrics;

-- Create permissive policies that allow inserts/updates
CREATE POLICY "Enable all access to ml_models" ON ml_models FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "Enable all access to live_signals" ON live_signals FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "Enable all access to trades" ON trades FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "Enable all access to performance_metrics" ON performance_metrics FOR ALL USING (true) WITH CHECK (true);

-- Alternative: Disable RLS entirely (not recommended for production)
-- ALTER TABLE ml_models DISABLE ROW LEVEL SECURITY;
-- ALTER TABLE live_signals DISABLE ROW LEVEL SECURITY;
-- ALTER TABLE trades DISABLE ROW LEVEL SECURITY;
-- ALTER TABLE performance_metrics DISABLE ROW LEVEL SECURITY;

