#!/usr/bin/env python3
"""
CONTINUOUS LEARNING SYSTEM
===========================
Monitors trades after each session and learns from winners/losers automatically.
More aggressive than the standard system - learns from every session!
"""

import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client

sys.path.insert(0, str(Path(__file__).parent))

load_dotenv()

SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')

if not all([SUPABASE_URL, SUPABASE_KEY]):
    print("❌ Missing environment variables")
    sys.exit(1)

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Learning thresholds (more aggressive)
MIN_TRADES_PER_SESSION = 3      # Learn from just 3 trades!
LEARNING_WINDOW_HOURS = 4       # Check every 4 hours
MIN_LOSER_COUNT = 1             # Learn from even 1 loser

# STRICT ELITE BENCHMARKS
MIN_PROFIT_FACTOR = 1.6         # Only elite models
MAX_DRAWDOWN_PCT = 6.0          # Tight risk control
MIN_SHARPE = 1.0                # Excellent risk-adjusted returns
MIN_WIN_RATE = 45.0             # High win rate required


def analyze_recent_trades(hours_back: int = 4):
    """
    Analyze trades from recent session.
    
    Returns:
        dict with trade statistics
    """
    print(f"\n{'='*80}")
    print(f"📊 ANALYZING TRADES FROM LAST {hours_back} HOURS")
    print(f"{'='*80}\n")
    
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours_back)).isoformat()
    
    try:
        # Fetch recent trades
        response = supabase.table('trades').select('*').gte('exit_time', cutoff).execute()
        
        if not response.data:
            print("📭 No trades in recent session")
            return None
        
        trades = pd.DataFrame(response.data)
        
        # Calculate stats
        total_trades = len(trades)
        winners = trades[trades['reason'] == 'take_profit']
        losers = trades[trades['reason'] == 'stop_loss']
        
        win_count = len(winners)
        loss_count = len(losers)
        win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0
        
        total_pnl = trades['pnl'].sum() if 'pnl' in trades.columns else 0
        
        print(f"📈 Total Trades: {total_trades}")
        print(f"   ✅ Winners: {win_count}")
        print(f"   ❌ Losers: {loss_count}")
        print(f"   📊 Win Rate: {win_rate:.1f}%")
        print(f"   💰 Total P&L: {total_pnl:.5f}\n")
        
        # Analyze by symbol/timeframe
        if total_trades > 0:
            print("📊 Performance by Symbol/Timeframe:")
            for (symbol, tf), group in trades.groupby(['symbol', 'timeframe']):
                sym_winners = len(group[group['reason'] == 'take_profit'])
                sym_total = len(group)
                sym_wr = (sym_winners / sym_total * 100) if sym_total > 0 else 0
                
                status = "✅" if sym_wr >= 50 else "⚠️" if sym_wr >= 40 else "❌"
                print(f"   {status} {symbol} {tf}: {sym_winners}/{sym_total} ({sym_wr:.0f}%)")
        
        # Identify problem areas
        print(f"\n🔍 PROBLEM IDENTIFICATION:")
        problems = []
        
        # Low win rate symbols
        if total_trades >= 3:
            for (symbol, tf), group in trades.groupby(['symbol', 'timeframe']):
                if len(group) >= 2:
                    sym_wr = (len(group[group['reason'] == 'take_profit']) / len(group) * 100)
                    if sym_wr < 40:
                        problems.append(f"❌ {symbol} {tf}: Low win rate ({sym_wr:.0f}%)")
        
        # High confidence losers (bad signals)
        if 'confidence' in trades.columns and loss_count > 0:
            high_conf_losers = losers[losers['confidence'] > 0.5]
            if len(high_conf_losers) > 0:
                problems.append(f"⚠️  {len(high_conf_losers)} high-confidence losers (model overconfident)")
        
        # Directional bias issues
        if total_trades >= 3:
            long_trades = trades[trades['direction'] == 'long']
            short_trades = trades[trades['direction'] == 'short']
            
            if len(long_trades) > 0:
                long_wr = (len(long_trades[long_trades['reason'] == 'take_profit']) / len(long_trades) * 100)
                if long_wr < 35:
                    problems.append(f"❌ Long trades struggling ({long_wr:.0f}% WR)")
            
            if len(short_trades) > 0:
                short_wr = (len(short_trades[short_trades['reason'] == 'take_profit']) / len(short_trades) * 100)
                if short_wr < 35:
                    problems.append(f"❌ Short trades struggling ({short_wr:.0f}% WR)")
        
        if problems:
            for problem in problems:
                print(f"   {problem}")
        else:
            print(f"   ✅ No critical issues detected")
        
        return {
            'total_trades': total_trades,
            'win_count': win_count,
            'loss_count': loss_count,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'problems': problems,
            'should_retrain': loss_count >= MIN_LOSER_COUNT and total_trades >= MIN_TRADES_PER_SESSION
        }
        
    except Exception as e:
        print(f"❌ Error analyzing trades: {e}")
        return None


def extract_learning_data():
    """
    Extract detailed learning data from recent trades.
    Saves losers and winners separately for focused learning.
    """
    print(f"\n{'='*80}")
    print(f"🔬 EXTRACTING LEARNING DATA")
    print(f"{'='*80}\n")
    
    try:
        # Fetch all trades (not just recent)
        response = supabase.table('trades').select('*').execute()
        
        if not response.data:
            print("⚠️  No trade data available")
            return False
        
        trades = pd.DataFrame(response.data)
        
        # Separate winners and losers
        winners = trades[trades['reason'] == 'take_profit']
        losers = trades[trades['reason'] == 'stop_loss']
        
        # Save to CSV for analysis
        output_dir = Path('live_trades')
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if len(winners) > 0:
            winners_file = output_dir / f'winners_{timestamp}.csv'
            winners.to_csv(winners_file, index=False)
            print(f"✅ Saved {len(winners)} winners → {winners_file}")
        
        if len(losers) > 0:
            losers_file = output_dir / f'losers_{timestamp}.csv'
            losers.to_csv(losers_file, index=False)
            print(f"✅ Saved {len(losers)} losers → {losers_file}")
        
        # All trades
        all_trades_file = output_dir / f'all_trades_{timestamp}.csv'
        trades.to_csv(all_trades_file, index=False)
        print(f"✅ Saved {len(trades)} total trades → {all_trades_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error extracting data: {e}")
        return False


def trigger_focused_retraining():
    """
    Trigger retraining with focus on losers.
    """
    print(f"\n{'='*80}")
    print(f"🧠 TRIGGERING FOCUSED RETRAINING")
    print(f"{'='*80}\n")
    
    try:
        # Extract learning data
        if not extract_learning_data():
            print("⚠️  No data to learn from")
            return False
        
        print("\n" + "="*80)
        print("🔧 Step 1: Analyzing losing patterns...")
        print("="*80 + "\n")
        
        # Import and run trade collector
        try:
            from trade_collector import main as collect_trades
            collect_trades()
        except Exception as e:
            print(f"⚠️  Trade collector error: {e}")
        
        print("\n" + "="*80)
        print("🧠 Step 2: Retraining models with emphasis on losers...")
        print("="*80 + "\n")
        
        # Import and run retraining
        try:
            from retrain_from_live_trades import main as retrain_models
            retrain_models()
        except Exception as e:
            print(f"⚠️  Retraining error: {e}")
            return False
        
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
    """Main continuous learning loop."""
    print("\n" + "="*80)
    print("🤖 CONTINUOUS LEARNING SYSTEM")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Learning Window: {LEARNING_WINDOW_HOURS} hours")
    print(f"Min Trades: {MIN_TRADES_PER_SESSION}")
    print(f"Min Losers: {MIN_LOSER_COUNT}")
    print("="*80 + "\n")
    
    # Analyze recent trades
    stats = analyze_recent_trades(hours_back=LEARNING_WINDOW_HOURS)
    
    if stats is None:
        print("\n⏳ No trades to analyze yet. Waiting for trading activity...")
        return
    
    # Check if we should retrain
    if stats['should_retrain']:
        print(f"\n🔄 CONDITIONS MET FOR RETRAINING:")
        print(f"   ✅ {stats['total_trades']} trades (>= {MIN_TRADES_PER_SESSION})")
        print(f"   ✅ {stats['loss_count']} losers (>= {MIN_LOSER_COUNT})")
        print(f"\n🚀 Initiating learning cycle...\n")
        
        success = trigger_focused_retraining()
        
        if success:
            print("\n✅ Learning cycle completed successfully!")
        else:
            print("\n⚠️  Learning cycle completed with some errors")
    else:
        print(f"\n⏳ NOT ENOUGH DATA FOR RETRAINING:")
        
        if stats['total_trades'] < MIN_TRADES_PER_SESSION:
            print(f"   Need {MIN_TRADES_PER_SESSION - stats['total_trades']} more trades")
        
        if stats['loss_count'] < MIN_LOSER_COUNT:
            print(f"   Need {MIN_LOSER_COUNT - stats['loss_count']} more losers")
        
        print("\n   System will check again in next cycle.")
    
    print("\n" + "="*80)
    print("✅ Continuous learning check complete")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

