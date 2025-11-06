// Supabase Client Configuration
import { createClient } from '@supabase/supabase-js';

const supabaseUrl = import.meta.env.VITE_SUPABASE_URL || 'https://ifetofkhyblyijghuwzs.supabase.co';
const supabaseAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY || '';

export const supabase = createClient(supabaseUrl, supabaseAnonKey);

// Types for TypeScript
export interface MLModel {
  id: string;
  symbol: string;
  timeframe: string;
  win_rate: number;
  profit_factor: number;
  sharpe_ratio: number;
  max_drawdown: number;
  total_trades: number;
  status: string;
  features: string[];
  parameters: any;
  backtest_results: any;
}

export interface LiveSignal {
  id: string;
  symbol: string;
  timeframe: string;
  signal_type: 'long' | 'short' | 'flat';
  confidence: number;
  edge: number;
  entry_price: number;
  timestamp: string;
  status: 'active' | 'closed' | 'expired';
}

export interface Trade {
  id: string;
  symbol: string;
  timeframe: string;
  direction: 'long' | 'short';
  entry_time: string;
  entry_price: number;
  exit_time?: string;
  exit_price?: number;
  pnl?: number;
  pnl_pct?: number;
  status: 'open' | 'closed';
}

