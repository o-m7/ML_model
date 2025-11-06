#!/usr/bin/env python3
"""
AUTOMATED WEEKLY INCREMENTAL RETRAINING
========================================
Fetches only the LAST WEEK of data, appends to existing historical data,
and retrains using the EXACT SAME methodology as production_final_system.py
"""

import os
import sys
import pickle
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import requests
from supabase import create_client

# Import the EXACT training system
sys.path.insert(0, str(Path(__file__).parent))
from production_final_system import (
    CONFIG,
    add_features,
    create_balanced_labels,
    select_features,
    BalancedModel,
    ProductionBacktest
)

load_dotenv()

# Configuration
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

TIMEFRAME_MINUTES = {
    '5T': 5,
    '15T': 15,
    '30T': 30,
    '1H': 60,
    '4H': 240
}


def calculate_ta_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate base TA indicators needed by the system."""
    df = df.copy()
    
    # EMA
    df['ema20'] = df['close'].ewm(span=20).mean()
    df['ema50'] = df['close'].ewm(span=50).mean()
    
    # ATR
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df['atr14'] = true_range.rolling(14).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi14'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands %
    sma20 = df['close'].rolling(20).mean()
    std20 = df['close'].rolling(20).std()
    df['bb_upper'] = sma20 + (std20 * 2)
    df['bb_lower'] = sma20 - (std20 * 2)
    df['bb_pct'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    # ADX
    plus_dm = df['high'].diff()
    minus_dm = -df['low'].diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    tr14 = true_range.rolling(14).mean()
    plus_di = 100 * (plus_dm.rolling(14).mean() / tr14)
    minus_di = 100 * (minus_dm.rolling(14).mean() / tr14)
    
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    df['adx'] = dx.rolling(14).mean()
    
    return df


def fetch_weekly_data(symbol: str, timeframe: str, days_back: int = 7):
    """Fetch ONLY the last week of data from Polygon."""
    print(f"  📥 Fetching last {days_back} days of {symbol} {timeframe}...")
    
    ticker = TICKER_MAP.get(symbol, symbol)
    minutes = TIMEFRAME_MINUTES[timeframe]
    
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=days_back)
    
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/{minutes}/minute/{int(start_time.timestamp()*1000)}/{int(end_time.timestamp()*1000)}"
    
    params = {
        'adjusted': 'true',
        'sort': 'asc',
        'limit': 50000,
        'apiKey': POLYGON_API_KEY
    }
    
    try:
        response = requests.get(url, params=params, timeout=60)
        data = response.json()
        
        if 'results' not in data or not data['results']:
            print(f"    ⚠️  No new data available")
            return None
        
        df = pd.DataFrame(data['results'])
        df['timestamp'] = pd.to_datetime(df['t'], unit='ms', utc=True)
        df = df.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'})
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        print(f"    ✅ Got {len(df)} new bars ({df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]})")
        return df
        
    except Exception as e:
        print(f"    ❌ Error: {e}")
        return None


def load_and_update_data(symbol: str, timeframe: str):
    """Load existing data, fetch new week, append, and save."""
    
    data_file = CONFIG.FEATURE_STORE / symbol / f"{symbol}_{timeframe}.parquet"
    
    # Check if we have existing data
    if data_file.exists():
        # INCREMENTAL UPDATE: We have historical data
        old_df = pd.read_parquet(data_file)
        if 'timestamp' not in old_df.columns:
            old_df = old_df.reset_index()
        old_df['timestamp'] = pd.to_datetime(old_df['timestamp'], utc=True)
        print(f"  📂 Loaded {len(old_df):,} existing bars (up to {old_df['timestamp'].iloc[-1]})")
        
        # Fetch new week
        new_df = fetch_weekly_data(symbol, timeframe, days_back=7)
        if new_df is None:
            print(f"    Using existing data only")
            return old_df
        
        # Remove any overlapping timestamps
        last_timestamp = old_df['timestamp'].iloc[-1]
        new_df = new_df[new_df['timestamp'] > last_timestamp]
        
        if len(new_df) == 0:
            print(f"    No new data after {last_timestamp}")
            return old_df
        
        # Calculate TA indicators for new data
        print(f"    🔧 Calculating TA indicators for {len(new_df)} new bars...")
        new_df_with_ta = calculate_ta_indicators(new_df)
        
        # Combine
        combined = pd.concat([old_df, new_df_with_ta], ignore_index=True)
        combined = combined.sort_values('timestamp').reset_index(drop=True)
        
        # Recalculate TA indicators for the entire combined dataset to ensure consistency
        print(f"    🔧 Recalculating TA indicators for all {len(combined):,} bars...")
        combined = calculate_ta_indicators(combined)
        
        print(f"    ✅ Updated: {len(old_df):,} old + {len(new_df):,} new = {len(combined):,} total")
        
    else:
        # FIRST RUN: No existing data, fetch full historical dataset
        print(f"  ⚠️  No existing data - fetching full historical dataset...")
        
        # Fetch 180 days of historical data (enough for training)
        combined = fetch_weekly_data(symbol, timeframe, days_back=180)
        
        if combined is None or len(combined) < 500:
            print(f"    ❌ Failed to fetch historical data")
            return None
        
        # Calculate TA indicators for all data
        print(f"    🔧 Calculating TA indicators for {len(combined):,} bars...")
        combined = calculate_ta_indicators(combined)
        
        print(f"    ✅ Fetched {len(combined):,} bars")
    
    # Save
    data_path = CONFIG.FEATURE_STORE / symbol
    data_path.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(data_file)
    print(f"    💾 Saved to {data_file}")
    
    return combined


def retrain_model(symbol: str, timeframe: str):
    """
    Retrain model using EXACT SAME methodology as train_production()
    """
    print(f"\n{'='*80}")
    print(f"RETRAINING: {symbol} {timeframe}")
    print(f"{'='*80}\n")
    
    try:
        # 1. Load and update data
        print("[1/5] Loading and updating data...")
        df = load_and_update_data(symbol, timeframe)
        
        if df is None or len(df) < 500:
            print(f"  ❌ Insufficient data ({len(df) if df is not None else 0} bars)")
            return None
        
        # 2. Get parameters (same as production)
        params = CONFIG.SYMBOL_PARAMS.get(symbol, {}).get(timeframe, {
            'tp': 1.6, 'sl': 1.0, 'min_conf': 0.38, 'min_edge': 0.10, 'pos_size': 0.7
        })
        print(f"\nParameters: TP={params['tp']}, Conf={params['min_conf']}, Pos={params['pos_size']*100:.0f}%")
        
        # 3. Features & Labels (EXACT SAME as production)
        print("\n[2/5] Features & Labels...")
        df = add_features(df)
        df = create_balanced_labels(df, symbol, timeframe, params['tp'], params['sl'])
        features = select_features(df)
        print(f"Using {len(features)} features")
        
        # 4. Split (use last 12 months for testing, same as production)
        print("\n[3/5] Training...")
        oos_start = pd.to_datetime(CONFIG.TRAIN_END, utc=True) - timedelta(days=CONFIG.OOS_MONTHS * 30)
        train_df = df[df['timestamp'] < oos_start].copy()
        test_df = df[df['timestamp'] >= oos_start].copy()
        
        print(f"Train: {len(train_df):,} bars | Test: {len(test_df):,} bars")
        
        if len(test_df) < 100:
            print(f"  ⚠️  Insufficient test data ({len(test_df)} bars)")
            return None
        
        # 5. Train model (EXACT SAME as production)
        X_train = train_df[features].fillna(0).values
        y_train = train_df['target'].values
        
        model = BalancedModel()
        model.fit(X_train, y_train)
        
        # 6. Backtest (EXACT SAME as production)
        print("\n[4/5] Backtesting...")
        X_test = test_df[features].fillna(0).values
        probs = model.predict_proba(X_test)
        
        flat_probs = pd.Series(probs[:, 0], index=test_df.index)
        long_probs = pd.Series(probs[:, 1], index=test_df.index)
        short_probs = pd.Series(probs[:, 2], index=test_df.index)
        
        signals_long = pd.Series(False, index=test_df.index)
        signals_short = pd.Series(False, index=test_df.index)
        
        for pos in range(len(test_df)):
            probs_i = [flat_probs.iloc[pos], long_probs.iloc[pos], short_probs.iloc[pos]]
            max_prob = max(probs_i)
            sorted_probs = sorted(probs_i, reverse=True)
            edge = sorted_probs[0] - sorted_probs[1]
            
            if long_probs.iloc[pos] == max_prob and long_probs.iloc[pos] >= params['min_conf'] and edge >= params['min_edge']:
                signals_long.iloc[pos] = True
            elif short_probs.iloc[pos] == max_prob and short_probs.iloc[pos] >= params['min_conf'] and edge >= params['min_edge']:
                signals_short.iloc[pos] = True
        
        engine = ProductionBacktest(test_df, symbol, params['pos_size'])
        results = engine.run(signals_long, signals_short, long_probs, short_probs, params['tp'], params['sl'])
        
        # 7. Check benchmarks (same as production)
        print("\n[5/5] Checking benchmarks...")
        min_trades = CONFIG.MIN_TRADES_BY_TF.get(timeframe, 60)
        
        failures = []
        if results['profit_factor'] < CONFIG.MIN_PROFIT_FACTOR:
            failures.append(f"PF {results['profit_factor']:.2f} < {CONFIG.MIN_PROFIT_FACTOR}")
        if results['max_drawdown_pct'] > CONFIG.MAX_DRAWDOWN_PCT:
            failures.append(f"DD {results['max_drawdown_pct']:.1f}% > {CONFIG.MAX_DRAWDOWN_PCT}%")
        if results['sharpe_ratio'] < CONFIG.MIN_SHARPE:
            failures.append(f"Sharpe {results['sharpe_ratio']:.2f} < {CONFIG.MIN_SHARPE}")
        if results['win_rate'] < CONFIG.MIN_WIN_RATE:
            failures.append(f"WR {results['win_rate']:.1f}% < {CONFIG.MIN_WIN_RATE}%")
        if results['total_trades'] < min_trades:
            failures.append(f"Trades {results['total_trades']} < {min_trades}")
        
        passed = len(failures) == 0
        
        # 8. Display results
        print(f"\n{'='*80}")
        print(f"Trades: {results['total_trades']} (L:{results['long_trades']}, S:{results['short_trades']})")
        print(f"Win Rate: {results['win_rate']:.1f}%")
        print(f"Profit Factor: {results['profit_factor']:.2f}")
        print(f"Sharpe: {results['sharpe_ratio']:.2f}")
        print(f"Max DD: {results['max_drawdown_pct']:.1f}%")
        print(f"Return: {results['total_return_pct']:.1f}%")
        print(f"\n{'✅ PRODUCTION READY' if passed else '❌ FAILED: ' + ', '.join(failures)}")
        print(f"{'='*80}\n")
        
        return {
            'model': model,
            'features': features,
            'results': results,
            'params': params,
            'passed': passed,
            'data_points': len(df)
        }
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


def should_deploy_model(new_model: dict, old_model: dict = None):
    """Decide if new model should be deployed."""
    
    # Always deploy if it passes benchmarks and there's no old model
    if old_model is None:
        if new_model['passed']:
            print(f"  ✅ Deploying: No existing model, new model passes benchmarks")
            return True
        else:
            print(f"  ⏭️  Skipping: New model doesn't pass benchmarks")
            return False
    
    # Compare scores
    def calc_score(m):
        r = m.get('results', m)
        return (
            r.get('profit_factor', 0) * 0.4 +
            r.get('win_rate', 0) / 100 * 0.3 +
            r.get('sharpe_ratio', 0) * 0.2 -
            r.get('max_drawdown_pct', 100) / 100 * 0.1
        )
    
    new_score = calc_score(new_model)
    old_score = calc_score(old_model)
    
    improvement = ((new_score - old_score) / old_score * 100) if old_score > 0 else 100
    
    print(f"\n  ⚖️  MODEL COMPARISON:")
    print(f"    Old Score: {old_score:.3f}")
    print(f"    New Score: {new_score:.3f}")
    print(f"    Change:    {improvement:+.1f}%")
    
    # Deploy if improvement >= 5% OR if new model passes and old didn't
    if improvement >= 5:
        print(f"    ✅ Deploying: {improvement:+.1f}% improvement")
        return True
    elif new_model['passed'] and not old_model.get('passed', False):
        print(f"    ✅ Deploying: New model passes benchmarks")
        return True
    elif improvement >= 0 and new_model['passed']:
        print(f"    ✅ Deploying: No worse and passes benchmarks")
        return True
    else:
        print(f"    ⏭️  Skipping: No significant improvement")
        return False


def deploy_model(symbol: str, timeframe: str, model_data: dict):
    """Deploy model to production (same format as production_final_system)."""
    save_dir = CONFIG.MODEL_STORE / symbol
    save_dir.mkdir(parents=True, exist_ok=True)
    
    status = "PRODUCTION_READY" if model_data['passed'] else "FAILED"
    save_path = save_dir / f"{symbol}_{timeframe}_{status}.pkl"
    
    with open(save_path, 'wb') as f:
        pickle.dump({
            'model': model_data['model'],
            'features': model_data['features'],
            'results': model_data['results'],
            'params': model_data['params']
        }, f)
    
    print(f"  💾 Deployed to {save_path}")
    
    # Update Supabase
    try:
        supabase.table('ml_models').upsert({
            'symbol': symbol,
            'timeframe': timeframe,
            'model_path': str(save_path),
            'features': json.dumps(model_data['features']),
            'num_features': len(model_data['features']),
            'parameters': json.dumps(model_data['params']),
            'backtest_results': json.dumps(model_data['results']),
            'win_rate': model_data['results']['win_rate'],
            'profit_factor': model_data['results']['profit_factor'],
            'sharpe_ratio': model_data['results']['sharpe_ratio'],
            'max_drawdown': model_data['results']['max_drawdown_pct'],
            'total_trades': model_data['results']['total_trades'],
            'data_points': model_data['data_points'],
            'status': 'production_ready' if model_data['passed'] else 'failed',
            'updated_at': datetime.now(timezone.utc).isoformat()
        }, on_conflict='symbol,timeframe').execute()
        
        print(f"  ✅ Updated Supabase")
    except Exception as e:
        print(f"  ⚠️  Supabase update failed: {e}")


def main():
    """Main incremental retraining pipeline."""
    print("="*80)
    print(f"AUTOMATED WEEKLY INCREMENTAL RETRAINING")
    print(f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("="*80)
    print(f"\n📌 Strategy: Fetch LAST WEEK → Append → Retrain with EXACT production methodology")
    print("")
    
    results = {
        'total': 0,
        'retrained': 0,
        'deployed': 0,
        'skipped': 0,
        'failed': 0
    }
    
    for symbol in CONFIG.SYMBOLS:
        for timeframe in CONFIG.TIMEFRAMES:
            results['total'] += 1
            
            try:
                # Retrain model
                new_model = retrain_model(symbol, timeframe)
                
                if new_model:
                    results['retrained'] += 1
                    
                    # Load old model if exists
                    old_model_path = CONFIG.MODEL_STORE / symbol / f"{symbol}_{timeframe}_PRODUCTION_READY.pkl"
                    old_model = None
                    
                    if old_model_path.exists():
                        try:
                            with open(old_model_path, 'rb') as f:
                                old_model = pickle.load(f)
                        except:
                            pass
                    
                    # Decide deployment
                    if should_deploy_model(new_model, old_model):
                        deploy_model(symbol, timeframe, new_model)
                        results['deployed'] += 1
                    else:
                        results['skipped'] += 1
                else:
                    results['failed'] += 1
                
            except Exception as e:
                print(f"\n❌ Error processing {symbol} {timeframe}: {e}")
                results['failed'] += 1
    
    # Summary
    print("\n" + "="*80)
    print("RETRAINING SUMMARY")
    print("="*80)
    print(f"Total Models:    {results['total']}")
    print(f"✅ Retrained:    {results['retrained']}")
    print(f"🚀 Deployed:     {results['deployed']}")
    print(f"⏭️  Skipped:      {results['skipped']}")
    print(f"❌ Failed:       {results['failed']}")
    if results['retrained'] > 0:
        print(f"\nSuccess Rate:    {results['retrained']/results['total']*100:.1f}%")
        print(f"Deploy Rate:     {results['deployed']/results['retrained']*100:.1f}%")
    print("="*80)


if __name__ == "__main__":
    main()
