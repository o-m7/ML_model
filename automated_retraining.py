#!/usr/bin/env python3
"""
AUTOMATED WEEKLY RETRAINING PIPELINE
=====================================
Fetches latest data, recalculates features, retrains models, and deploys if better.
"""

import os
import sys
import pickle
import json
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import requests
from supabase import create_client

# Import your training system
sys.path.insert(0, str(Path(__file__).parent))
from production_final_system import (
    ProductionConfig,
    create_labels,
    select_features,
    BalancedModel,
    backtest_strategy
)

load_dotenv()

# Configuration
POLYGON_API_KEY = os.getenv('POLYGON_API_KEY')
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')
DATA_DIR = Path("feature_store")
MODEL_DIR = Path("models_production")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Symbols and timeframes to retrain
SYMBOLS = ["XAUUSD", "XAGUSD", "EURUSD", "GBPUSD", "AUDUSD", "NZDUSD"]
TIMEFRAMES = ["5T", "15T", "30T", "1H", "4H"]

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


def fetch_historical_data(symbol: str, timeframe: str, days_back: int = 365):
    """Fetch historical data from Polygon."""
    print(f"\n📊 Fetching {symbol} {timeframe} data...")
    
    ticker = TICKER_MAP.get(symbol, symbol)
    minutes = TIMEFRAME_MINUTES[timeframe]
    
    end_time = datetime.now()
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
            print(f"  ❌ No data returned for {symbol}")
            return None
        
        df = pd.DataFrame(data['results'])
        df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
        df = df.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'})
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].set_index('timestamp')
        df = df.sort_index()
        
        print(f"  ✅ Fetched {len(df)} bars ({df.index[0]} to {df.index[-1]})")
        return df
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return None


def save_raw_data(symbol: str, timeframe: str, df: pd.DataFrame):
    """Save raw OHLCV data."""
    data_path = DATA_DIR / symbol
    data_path.mkdir(parents=True, exist_ok=True)
    
    filename = data_path / f"{symbol}_{timeframe}_raw.parquet"
    df.to_parquet(filename)
    print(f"  💾 Saved to {filename}")


def load_raw_data(symbol: str, timeframe: str):
    """Load raw OHLCV data."""
    filename = DATA_DIR / symbol / f"{symbol}_{timeframe}_raw.parquet"
    if filename.exists():
        return pd.read_parquet(filename)
    return None


def calculate_features(df: pd.DataFrame):
    """Calculate technical features (simplified version)."""
    print(f"  🔧 Calculating features...")
    
    # Price-based features
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    
    # Volatility
    df['volatility'] = df['returns'].rolling(20).std()
    df['atr'] = (df['high'] - df['low']).rolling(14).mean()
    
    # Moving averages
    for period in [10, 20, 50, 100, 200]:
        df[f'sma_{period}'] = df['close'].rolling(period).mean()
        df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema12 = df['close'].ewm(span=12).mean()
    ema26 = df['close'].ewm(span=26).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    
    # Bollinger Bands
    sma20 = df['close'].rolling(20).mean()
    std20 = df['close'].rolling(20).std()
    df['bb_upper'] = sma20 + (std20 * 2)
    df['bb_lower'] = sma20 - (std20 * 2)
    df['bb_width'] = df['bb_upper'] - df['bb_lower']
    
    # Momentum
    for period in [5, 10, 20]:
        df[f'momentum_{period}'] = df['close'].pct_change(period)
    
    # Volume features
    df['volume_sma'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma']
    
    df = df.dropna()
    print(f"  ✅ Calculated {len(df.columns)} features, {len(df)} valid bars")
    
    return df


def retrain_model(symbol: str, timeframe: str, df: pd.DataFrame):
    """Retrain model for symbol/timeframe."""
    print(f"\n🤖 Retraining {symbol} {timeframe}...")
    
    # Get symbol-specific parameters
    params = ProductionConfig.SYMBOL_PARAMS.get(symbol, {}).get(timeframe, {})
    if not params:
        print(f"  ⚠️  No parameters found for {symbol} {timeframe}")
        return None
    
    tp_mult = params.get('tp', 1.5)
    sl_mult = params.get('sl', 1.0)
    min_conf = params.get('min_conf', 0.35)
    min_edge = params.get('min_edge', 0.08)
    
    # Create labels
    df = create_labels(df, tp_mult, sl_mult, ProductionConfig.FORECAST_HORIZON)
    
    if 'target' not in df.columns:
        print(f"  ❌ Failed to create labels")
        return None
    
    # Select features
    feature_cols = [col for col in df.columns if col not in ['target', 'open', 'high', 'low', 'close', 'volume']]
    selected_features = select_features(df[feature_cols], df['target'], max_features=50)
    
    if not selected_features:
        print(f"  ❌ No features selected")
        return None
    
    # Split data
    train_size = int(len(df) * 0.85)
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]
    
    # Train model
    X_train = train_df[selected_features]
    y_train = train_df['target']
    X_test = test_df[selected_features]
    y_test = test_df['target']
    
    model = BalancedModel()
    model.fit(X_train, y_train)
    
    # Backtest
    metrics = backtest_strategy(
        model, test_df, selected_features,
        symbol, timeframe, min_conf, min_edge
    )
    
    if metrics:
        print(f"  📊 Performance:")
        print(f"     Win Rate: {metrics['win_rate']:.1f}%")
        print(f"     Profit Factor: {metrics['profit_factor']:.2f}")
        print(f"     Sharpe: {metrics['sharpe']:.2f}")
        print(f"     Max DD: {metrics['max_drawdown']:.1f}%")
        print(f"     Total Trades: {metrics['total_trades']}")
        
        return {
            'model': model,
            'features': selected_features,
            'metrics': metrics,
            'params': params
        }
    
    return None


def should_deploy_model(new_metrics: dict, old_metrics: dict = None):
    """Decide if new model is better than old model."""
    if old_metrics is None:
        return True  # No old model, deploy new one
    
    # Compare key metrics
    new_score = (
        new_metrics['profit_factor'] * 0.4 +
        new_metrics['win_rate'] * 0.3 +
        new_metrics['sharpe'] * 0.2 -
        new_metrics['max_drawdown'] * 0.1
    )
    
    old_score = (
        old_metrics.get('profit_factor', 0) * 0.4 +
        old_metrics.get('win_rate', 0) * 0.3 +
        old_metrics.get('sharpe', 0) * 0.2 -
        old_metrics.get('max_drawdown', 100) * 0.1
    )
    
    improvement = ((new_score - old_score) / old_score * 100) if old_score > 0 else 100
    
    print(f"\n📊 Model Comparison:")
    print(f"   Old Score: {old_score:.2f}")
    print(f"   New Score: {new_score:.2f}")
    print(f"   Improvement: {improvement:+.1f}%")
    
    # Deploy if at least 5% improvement
    return improvement >= 5


def deploy_model(symbol: str, timeframe: str, model_data: dict):
    """Save model to production."""
    model_path = MODEL_DIR / symbol
    model_path.mkdir(parents=True, exist_ok=True)
    
    filename = model_path / f"{symbol}_{timeframe}_PRODUCTION_READY.pkl"
    
    with open(filename, 'wb') as f:
        pickle.dump({
            'model': model_data['model'],
            'features': model_data['features'],
            'metadata': {
                'symbol': symbol,
                'timeframe': timeframe,
                'trained_date': datetime.now().isoformat(),
                'metrics': model_data['metrics'],
                'params': model_data['params']
            }
        }, f)
    
    print(f"  ✅ Deployed to {filename}")
    
    # Update Supabase
    try:
        supabase.table('ml_models').upsert({
            'symbol': symbol,
            'timeframe': timeframe,
            'model_path': str(filename),
            'features': json.dumps(model_data['features']),
            'num_features': len(model_data['features']),
            'parameters': json.dumps(model_data['params']),
            'backtest_results': json.dumps(model_data['metrics']),
            'win_rate': model_data['metrics']['win_rate'],
            'profit_factor': model_data['metrics']['profit_factor'],
            'sharpe_ratio': model_data['metrics']['sharpe'],
            'max_drawdown': model_data['metrics']['max_drawdown'],
            'total_trades': model_data['metrics']['total_trades'],
            'status': 'production_ready',
            'updated_at': datetime.now().isoformat()
        }, on_conflict='symbol,timeframe').execute()
        
        print(f"  ✅ Updated Supabase metadata")
    except Exception as e:
        print(f"  ⚠️  Supabase update failed: {e}")


def main():
    """Main retraining pipeline."""
    print("="*80)
    print(f"AUTOMATED WEEKLY RETRAINING - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    results = {
        'total': 0,
        'retrained': 0,
        'deployed': 0,
        'skipped': 0,
        'failed': 0
    }
    
    for symbol in SYMBOLS:
        for timeframe in TIMEFRAMES:
            results['total'] += 1
            
            try:
                # Fetch latest data
                df = fetch_historical_data(symbol, timeframe, days_back=365)
                if df is None:
                    results['failed'] += 1
                    continue
                
                # Save raw data
                save_raw_data(symbol, timeframe, df)
                
                # Calculate features
                df = calculate_features(df)
                
                # Retrain model
                new_model = retrain_model(symbol, timeframe, df)
                
                if new_model:
                    results['retrained'] += 1
                    
                    # Load old model metrics if exists
                    old_model_path = MODEL_DIR / symbol / f"{symbol}_{timeframe}_PRODUCTION_READY.pkl"
                    old_metrics = None
                    
                    if old_model_path.exists():
                        with open(old_model_path, 'rb') as f:
                            old_data = pickle.load(f)
                            old_metrics = old_data.get('metadata', {}).get('metrics')
                    
                    # Decide whether to deploy
                    if should_deploy_model(new_model['metrics'], old_metrics):
                        deploy_model(symbol, timeframe, new_model)
                        results['deployed'] += 1
                    else:
                        print(f"  ⏭️  Skipping deployment - no significant improvement")
                        results['skipped'] += 1
                else:
                    results['failed'] += 1
                
            except Exception as e:
                print(f"\n❌ Error processing {symbol} {timeframe}: {e}")
                results['failed'] += 1
            
            print("")  # Blank line between models
    
    # Summary
    print("="*80)
    print("RETRAINING SUMMARY")
    print("="*80)
    print(f"Total Models: {results['total']}")
    print(f"✅ Retrained: {results['retrained']}")
    print(f"🚀 Deployed: {results['deployed']}")
    print(f"⏭️  Skipped: {results['skipped']}")
    print(f"❌ Failed: {results['failed']}")
    print("="*80)


if __name__ == "__main__":
    main()

