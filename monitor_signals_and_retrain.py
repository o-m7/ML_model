#!/usr/bin/env python3
"""
AUTOMATIC SIGNAL MONITORING AND RETRAINING
============================================
Monitors live signals, checks if they hit TP/SL, logs trades, and triggers retraining.
No manual trading required - learns from signal outcomes automatically.
"""

import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client
import requests

sys.path.insert(0, str(Path(__file__).parent))

load_dotenv()

POLYGON_API_KEY = os.getenv('POLYGON_API_KEY')
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

TICKER_MAP = {
    'XAUUSD': 'C:XAUUSD',
    'XAGUSD': 'C:XAGUSD',
    'EURUSD': 'C:EURUSD',
    'GBPUSD': 'C:GBPUSD',
    'AUDUSD': 'C:AUDUSD',
    'NZDUSD': 'C:NZDUSD',
}


def get_current_price(symbol: str) -> float:
    """Get current price from Polygon snapshot API."""
    ticker = TICKER_MAP.get(symbol, f'C:{symbol}')
    
    try:
        url = f'https://api.polygon.io/v2/snapshot/locale/global/markets/forex/tickers'
        response = requests.get(url, params={'tickers': ticker, 'apiKey': POLYGON_API_KEY}, timeout=10)
        data = response.json()
        
        if data.get('status') == 'OK' and data.get('tickers'):
            ticker_data = data['tickers'][0]
            # Try to get last quote price, fallback to day close
            price = ticker_data.get('lastQuote', {}).get('p') or ticker_data.get('day', {}).get('c')
            if price:
                return float(price)
    except Exception as e:
        print(f"  ⚠️  Error fetching price for {symbol}: {e}")
    
    return None


def check_signal_outcome(signal: dict) -> dict:
    """
    Check if a signal hit TP or SL.
    
    Returns:
        dict with 'status', 'exit_price', 'pnl', 'reason', 'exit_time'
    """
    symbol = signal['symbol']
    direction = signal['signal_type']
    entry_price = signal['entry_price']
    tp_price = signal['take_profit']
    sl_price = signal['stop_loss']
    entry_time = pd.to_datetime(signal['timestamp'])
    
    # Get current price
    current_price = get_current_price(symbol)
    if current_price is None:
        return None
    
    # Check if TP or SL hit
    if direction == 'long':
        if current_price >= tp_price:
            pnl = tp_price - entry_price
            return {
                'status': 'closed',
                'exit_price': tp_price,
                'pnl': pnl,
                'reason': 'take_profit',
                'exit_time': datetime.now(timezone.utc).isoformat()
            }
        elif current_price <= sl_price:
            pnl = sl_price - entry_price
            return {
                'status': 'closed',
                'exit_price': sl_price,
                'pnl': pnl,
                'reason': 'stop_loss',
                'exit_time': datetime.now(timezone.utc).isoformat()
            }
    else:  # short
        if current_price <= tp_price:
            pnl = entry_price - tp_price
            return {
                'status': 'closed',
                'exit_price': tp_price,
                'pnl': pnl,
                'reason': 'take_profit',
                'exit_time': datetime.now(timezone.utc).isoformat()
            }
        elif current_price >= sl_price:
            pnl = entry_price - sl_price
            return {
                'status': 'closed',
                'exit_price': sl_price,
                'pnl': pnl,
                'reason': 'stop_loss',
                'exit_time': datetime.now(timezone.utc).isoformat()
            }
    
    # Still open
    return {'status': 'open'}


def monitor_active_signals():
    """Monitor active signals and log trades when they close."""
    print(f"\n{'='*80}")
    print(f"SIGNAL MONITOR - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")
    
    # Get active signals from last 24 hours
    cutoff_time = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
    
    try:
        response = supabase.table('live_signals').select('*').eq('status', 'active').gte('timestamp', cutoff_time).execute()
        
        if not response.data:
            print("📭 No active signals to monitor")
            return 0
        
        signals = response.data
        print(f"📊 Monitoring {len(signals)} active signals...\n")
        
        closed_count = 0
        
        for signal in signals:
            signal_id = signal['id']
            symbol = signal['symbol']
            timeframe = signal['timeframe']
            direction = signal['signal_type']
            
            # Check outcome
            outcome = check_signal_outcome(signal)
            
            if outcome is None:
                continue
            
            if outcome['status'] == 'closed':
                # Log trade
                trade = {
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'direction': direction,
                    'entry_price': signal['entry_price'],
                    'exit_price': outcome['exit_price'],
                    'pnl': outcome['pnl'],
                    'entry_time': signal['timestamp'],
                    'exit_time': outcome['exit_time'],
                    'status': 'closed',
                    'reason': outcome['reason'],
                    'confidence': signal.get('confidence'),
                    'edge': signal.get('edge')
                }
                
                # Insert into trades table
                supabase.table('trades').insert(trade).execute()
                
                # Update signal status
                supabase.table('live_signals').update({'status': 'closed'}).eq('id', signal_id).execute()
                
                pnl_display = f"+{outcome['pnl']:.5f}" if outcome['pnl'] > 0 else f"{outcome['pnl']:.5f}"
                print(f"✅ {symbol} {timeframe} {direction.upper()}: {outcome['reason'].upper()} | P&L: {pnl_display}")
                
                closed_count += 1
        
        if closed_count > 0:
            print(f"\n{'='*80}")
            print(f"✅ Logged {closed_count} closed trades")
            print(f"{'='*80}\n")
        else:
            print("⏳ All signals still open\n")
        
        return closed_count
        
    except Exception as e:
        print(f"❌ Error monitoring signals: {e}")
        return 0


def check_if_retraining_needed():
    """Check if we have enough new trades to trigger retraining."""
    try:
        # Check when last retraining happened
        response = supabase.table('trades').select('exit_time').order('exit_time', desc=True).limit(1).execute()
        
        if not response.data:
            print("ℹ️  No trades yet - waiting for first closed trade")
            return False
        
        # Count trades in last 24 hours
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
        response = supabase.table('trades').select('id', count='exact').gte('exit_time', cutoff).execute()
        
        trade_count = response.count if hasattr(response, 'count') else len(response.data)
        
        print(f"📊 Trades closed in last 24h: {trade_count}")
        
        # Trigger retraining if we have 10+ new trades
        if trade_count >= 10:
            print("🔄 Enough trades accumulated - retraining recommended!")
            return True
        else:
            print(f"⏳ Need {10 - trade_count} more trades before retraining")
            return False
            
    except Exception as e:
        print(f"⚠️  Error checking retraining status: {e}")
        return False


def trigger_retraining():
    """Trigger the retraining process."""
    print(f"\n{'='*80}")
    print(f"🔧 TRIGGERING RETRAINING")
    print(f"{'='*80}\n")
    
    try:
        # Import and run trade collector
        from trade_collector import main as collect_trades
        print("📊 Step 1: Collecting and analyzing trades...")
        collect_trades()
        
        print("\n" + "="*80)
        print("🧠 Step 2: Retraining models from live trades...")
        print("="*80 + "\n")
        
        # Import and run retraining
        from retrain_from_live_trades import main as retrain_models
        retrain_models()
        
        print("\n" + "="*80)
        print("✅ RETRAINING COMPLETE")
        print("="*80 + "\n")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during retraining: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main monitoring loop."""
    print("\n" + "="*80)
    print("🤖 AUTOMATIC SIGNAL MONITORING & RETRAINING SYSTEM")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")
    
    # Monitor signals
    closed_count = monitor_active_signals()
    
    # Check if retraining needed
    if closed_count > 0 or datetime.now().hour == 0:  # Also check at midnight
        should_retrain = check_if_retraining_needed()
        
        if should_retrain:
            trigger_retraining()
    
    print("\n✅ Monitoring cycle complete\n")


if __name__ == "__main__":
    main()

