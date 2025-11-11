#!/usr/bin/env python3
"""
Backtest All Current Models Against Strict Benchmarks
======================================================
Tests all symbol/timeframe combinations to see which pass elite standards.
"""

import sys
from pathlib import Path
import subprocess
import pandas as pd
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

# All combinations to test
SYMBOLS = ['EURUSD', 'GBPUSD', 'AUDUSD', 'NZDUSD', 'XAUUSD', 'XAGUSD']
TIMEFRAMES = ['5T', '15T', '30T', '1H', '4H']

# Strict benchmarks
MIN_PF = 1.6
MAX_DD = 6.0
MIN_SHARPE = 1.0
MIN_WR = 45.0

def parse_result(output: str):
    """Parse training output to extract metrics."""
    try:
        # Look for the results section
        lines = output.split('\n')
        
        metrics = {
            'trades': 0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'sharpe': 0.0,
            'max_dd': 0.0,
            'status': 'UNKNOWN'
        }
        
        for i, line in enumerate(lines):
            if 'Trades:' in line and 'Win Rate:' in lines[i+1]:
                # Found results section
                trades_line = line
                wr_line = lines[i+1]
                pf_line = lines[i+2]
                sharpe_line = lines[i+3]
                dd_line = lines[i+4]
                
                # Parse
                metrics['trades'] = int(trades_line.split(':')[1].split('(')[0].strip())
                metrics['win_rate'] = float(wr_line.split(':')[1].replace('%', '').strip())
                metrics['profit_factor'] = float(pf_line.split(':')[1].strip())
                metrics['sharpe'] = float(sharpe_line.split(':')[1].strip())
                metrics['max_dd'] = float(dd_line.split(':')[1].replace('%', '').strip())
                
                # Status
                if 'FAILED' in output:
                    metrics['status'] = 'FAIL'
                elif 'PASSED' in output or 'Saved model' in output:
                    metrics['status'] = 'PASS'
                
                break
        
        return metrics
    except Exception as e:
        print(f"  ⚠️  Error parsing: {e}")
        return None


def test_model(symbol: str, timeframe: str):
    """Test a single symbol/timeframe combination."""
    print(f"\n{'='*80}")
    print(f"Testing {symbol} {timeframe}...")
    print(f"{'='*80}")
    
    try:
        result = subprocess.run(
            ['python3', 'production_final_system.py', '--symbol', symbol, '--tf', timeframe],
            capture_output=True,
            text=True,
            timeout=600  # 10 min timeout
        )
        
        metrics = parse_result(result.stdout + result.stderr)
        
        if metrics:
            # Check benchmarks
            passes_pf = metrics['profit_factor'] >= MIN_PF
            passes_dd = metrics['max_dd'] <= MAX_DD
            passes_sharpe = metrics['sharpe'] >= MIN_SHARPE
            passes_wr = metrics['win_rate'] >= MIN_WR
            
            passes_all = passes_pf and passes_dd and passes_sharpe and passes_wr
            
            status = "✅ PASS" if passes_all else "❌ FAIL"
            
            print(f"\n{status}")
            print(f"  Trades: {metrics['trades']}")
            print(f"  Win Rate: {metrics['win_rate']:.1f}% {'✅' if passes_wr else '❌'}")
            print(f"  Profit Factor: {metrics['profit_factor']:.2f} {'✅' if passes_pf else '❌'}")
            print(f"  Sharpe: {metrics['sharpe']:.2f} {'✅' if passes_sharpe else '❌'}")
            print(f"  Max DD: {metrics['max_dd']:.1f}% {'✅' if passes_dd else '❌'}")
            
            return {
                'symbol': symbol,
                'timeframe': timeframe,
                **metrics,
                'passes_all': passes_all
            }
        else:
            print(f"  ⚠️  Could not parse results")
            return None
            
    except subprocess.TimeoutExpired:
        print(f"  ⏰ Timeout (10 min)")
        return None
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return None


def main():
    print("\n" + "="*80)
    print("COMPREHENSIVE BACKTEST - ALL MODELS")
    print("="*80)
    print(f"\nSTRICT ELITE BENCHMARKS:")
    print(f"  Profit Factor ≥ {MIN_PF}")
    print(f"  Max Drawdown ≤ {MAX_DD}%")
    print(f"  Sharpe Ratio ≥ {MIN_SHARPE}")
    print(f"  Win Rate ≥ {MIN_WR}%")
    print(f"\nTesting {len(SYMBOLS)} symbols × {len(TIMEFRAMES)} timeframes = {len(SYMBOLS) * len(TIMEFRAMES)} combinations")
    print("="*80)
    
    all_results = []
    
    for symbol in SYMBOLS:
        for timeframe in TIMEFRAMES:
            result = test_model(symbol, timeframe)
            if result:
                all_results.append(result)
    
    # Create summary
    print("\n\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    
    df = pd.DataFrame(all_results)
    
    if len(df) > 0:
        passing = df[df['passes_all'] == True]
        failing = df[df['passes_all'] == False]
        
        print(f"\n✅ PASSING (Elite Models): {len(passing)}/{len(df)}")
        print(f"❌ FAILING: {len(failing)}/{len(df)}")
        
        if len(passing) > 0:
            print(f"\n{'='*80}")
            print("✅ ELITE MODELS:")
            print(f"{'='*80}")
            passing_sorted = passing.sort_values('sharpe', ascending=False)
            for _, row in passing_sorted.iterrows():
                print(f"  {row['symbol']:8} {row['timeframe']:4} | PF:{row['profit_factor']:5.2f} | DD:{row['max_dd']:4.1f}% | Sharpe:{row['sharpe']:5.2f} | WR:{row['win_rate']:5.1f}% | Trades:{row['trades']}")
        
        if len(failing) > 0:
            print(f"\n{'='*80}")
            print("❌ MODELS NEEDING RETRAINING:")
            print(f"{'='*80}")
            failing_sorted = failing.sort_values('sharpe', ascending=False)
            for _, row in failing_sorted.iterrows():
                issues = []
                if row['profit_factor'] < MIN_PF:
                    issues.append(f"PF:{row['profit_factor']:.2f}")
                if row['max_dd'] > MAX_DD:
                    issues.append(f"DD:{row['max_dd']:.1f}%")
                if row['sharpe'] < MIN_SHARPE:
                    issues.append(f"Sharpe:{row['sharpe']:.2f}")
                if row['win_rate'] < MIN_WR:
                    issues.append(f"WR:{row['win_rate']:.1f}%")
                
                print(f"  {row['symbol']:8} {row['timeframe']:4} | {' | '.join(issues)}")
        
        # Save to CSV
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_file = f'backtest_results_{timestamp}.csv'
        df.to_csv(csv_file, index=False)
        print(f"\n📊 Results saved to: {csv_file}")
    
    else:
        print("\n⚠️  No results collected")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()

