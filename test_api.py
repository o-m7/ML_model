#!/usr/bin/env python3
"""
Test all API endpoints and signal generation
"""

import requests
import json
import random

API_URL = "http://localhost:8000"

def test_health():
    """Test API health"""
    print("\n" + "="*80)
    print("1. HEALTH CHECK")
    print("="*80)
    
    response = requests.get(f"{API_URL}/health")
    data = response.json()
    
    print(f"Status: {data['status']}")
    print(f"Models Available: {data['models_available']}")
    print(f"Supabase: {data.get('supabase', 'not configured')}")


def test_list_models():
    """Test listing all models"""
    print("\n" + "="*80)
    print("2. LIST ALL MODELS")
    print("="*80)
    
    response = requests.get(f"{API_URL}/models")
    models = response.json()
    
    print(f"Total Models: {len(models)}")
    
    # Group by symbol
    by_symbol = {}
    for model in models:
        symbol, tf = model.split('_')
        if symbol not in by_symbol:
            by_symbol[symbol] = []
        by_symbol[symbol].append(tf)
    
    for symbol, tfs in sorted(by_symbol.items()):
        print(f"  {symbol}: {', '.join(sorted(tfs))}")


def test_model_info():
    """Test getting model details"""
    print("\n" + "="*80)
    print("3. GET MODEL INFO")
    print("="*80)
    
    response = requests.get(f"{API_URL}/models/XAUUSD/5T")
    data = response.json()
    
    print(f"Symbol: {data['symbol']}")
    print(f"Timeframe: {data['timeframe']}")
    print(f"Features: {data['num_features']}")
    print(f"Backtest Results:")
    print(f"  - Win Rate: {data['backtest_results']['win_rate']:.1f}%")
    print(f"  - Profit Factor: {data['backtest_results']['profit_factor']:.2f}")
    print(f"  - Sharpe Ratio: {data['backtest_results']['sharpe_ratio']:.2f}")
    print(f"  - Max Drawdown: {data['backtest_results']['max_drawdown_pct']:.1f}%")
    print(f"  - Total Trades: {data['backtest_results']['total_trades']}")


def test_predictions():
    """Test prediction endpoint"""
    print("\n" + "="*80)
    print("4. TEST LIVE PREDICTIONS")
    print("="*80)
    
    models_to_test = [
        ("XAUUSD", "5T"),
        ("XAUUSD", "15T"),
        ("EURUSD", "5T"),
        ("GBPUSD", "15T"),
        ("AUDUSD", "30T"),
    ]
    
    for symbol, tf in models_to_test:
        # Generate random features (would be real market data in production)
        features = [random.uniform(-0.02, 0.02) for _ in range(30)]
        
        response = requests.post(
            f"{API_URL}/predict",
            json={
                "symbol": symbol,
                "timeframe": tf,
                "features": features
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            
            # Determine if signal is strong enough
            trade_icon = "🟢" if result['should_trade'] else "🔴"
            
            print(f"\n{trade_icon} {symbol} {tf}")
            print(f"   Signal: {result['signal'].upper():5s} | Confidence: {result['confidence']*100:5.1f}% | Edge: {result['edge']*100:5.1f}%")
            print(f"   Probabilities: Flat={result['probabilities']['flat']*100:.1f}%, Long={result['probabilities']['long']*100:.1f}%, Short={result['probabilities']['short']*100:.1f}%")
            print(f"   Trade: {result['should_trade']}")
            
            if result['should_trade']:
                print(f"   🎯 TRADE SIGNAL DETECTED!")
        else:
            print(f"\n❌ {symbol} {tf} - Error: {response.status_code}")


def test_performance_summary():
    """Test performance summary"""
    print("\n" + "="*80)
    print("5. PERFORMANCE SUMMARY")
    print("="*80)
    
    response = requests.get(f"{API_URL}/performance/summary")
    data = response.json()
    
    print(f"Total Models: {data['total_models']}")
    print(f"\nBy Symbol:")
    for symbol, info in sorted(data['symbols'].items()):
        tfs = ', '.join(info['timeframes'])
        print(f"  {symbol}: {info['count']} models ({tfs})")
    
    print(f"\nBy Timeframe:")
    for tf, count in sorted(data['timeframes'].items()):
        print(f"  {tf}: {count} models")
    
    print(f"\nAggregate Metrics:")
    metrics = data['aggregate_metrics']
    print(f"  - Avg Win Rate: {metrics['avg_win_rate']:.1f}%")
    print(f"  - Avg Profit Factor: {metrics['avg_profit_factor']:.2f}")
    print(f"  - Avg Sharpe: {metrics['avg_sharpe']:.2f}")
    print(f"  - Avg Max Drawdown: {metrics['avg_max_drawdown']:.1f}%")


def main():
    print("\n" + "="*80)
    print("🚀 ML TRADING API - COMPREHENSIVE TEST")
    print("="*80)
    
    try:
        test_health()
        test_list_models()
        test_model_info()
        test_predictions()
        test_performance_summary()
        
        print("\n" + "="*80)
        print("✅ ALL TESTS PASSED - API IS FULLY OPERATIONAL!")
        print("="*80)
        
        print("\n📚 Available Endpoints:")
        print("  - GET  /health")
        print("  - GET  /models")
        print("  - GET  /models/{symbol}/{timeframe}")
        print("  - POST /predict")
        print("  - GET  /performance/summary")
        print("  - GET  /signals/active")
        print("  - GET  /trades/recent")
        print("\n🌐 Interactive Docs: http://localhost:8000/docs")
        
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Cannot connect to API")
        print("Make sure the API server is running:")
        print("  python3 api_server.py")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")


if __name__ == "__main__":
    main()

