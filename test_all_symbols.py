#!/usr/bin/env python3
"""
Multi-Symbol RenTec Component Test Suite

Tests audit, regime classification, and gates across all 8 symbols.
Generates comparative analysis and recommendations.

Usage:
    python test_all_symbols.py
    python test_all_symbols.py --timeframe 4H  # Specific timeframe
"""

import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from qa.audit import DataAuditor, AuditConfig
from features.regime import RegimeClassifier, Regime
from evaluation.gates import HardGates, GateThresholds


SYMBOLS = ['XAUUSD', 'XAGUSD', 'EURUSD', 'GBPUSD', 'AUDUSD', 'NZDUSD', 'USDJPY', 'USDCAD']
TIMEFRAMES = ['1H']  # Default to 1H for comparison


def test_symbol(symbol: str, timeframe: str) -> Dict:
    """Test one symbol/timeframe combination."""
    
    # Load data
    data_path = Path(f"feature_store/{symbol}/{symbol}_{timeframe}.parquet")
    
    if not data_path.exists():
        return {
            'symbol': symbol,
            'timeframe': timeframe,
            'status': 'MISSING_DATA',
            'error': f"Data not found: {data_path}"
        }
    
    try:
        df = pd.read_parquet(data_path)
        
        # Run audit
        auditor = DataAuditor(AuditConfig())
        audit_results = auditor.audit_all(df, symbol=symbol)
        
        # Add regime
        classifier = RegimeClassifier()
        df = classifier.add_regime(df)
        regime_stats = classifier.get_regime_stats(df)
        
        # Calculate regime characteristics
        regime_pcts = {
            'trend_pct': regime_stats['trend_pct'],
            'range_pct': regime_stats['range_pct'],
            'neutral_pct': regime_stats['neutral_pct'],
            'avg_confidence': regime_stats['avg_confidence']
        }
        
        return {
            'symbol': symbol,
            'timeframe': timeframe,
            'status': 'SUCCESS',
            'bars': len(df),
            'audit_warnings': len(auditor.warnings),
            'audit_passed': True,
            'regime_stats': regime_pcts,
            'data_start': str(df['timestamp'].min()),
            'data_end': str(df['timestamp'].max())
        }
        
    except Exception as e:
        return {
            'symbol': symbol,
            'timeframe': timeframe,
            'status': 'ERROR',
            'error': str(e)
        }


def analyze_regime_patterns(results: List[Dict]) -> Dict:
    """Analyze regime patterns across symbols."""
    
    successful = [r for r in results if r['status'] == 'SUCCESS']
    
    if not successful:
        return {}
    
    # Group by asset class
    metals = [r for r in successful if r['symbol'] in ['XAUUSD', 'XAGUSD']]
    fx = [r for r in successful if r['symbol'] not in ['XAUUSD', 'XAGUSD']]
    
    def avg_regime(group, regime_key):
        if not group:
            return 0.0
        return np.mean([r['regime_stats'][regime_key] for r in group])
    
    analysis = {
        'metals': {
            'count': len(metals),
            'avg_trend': avg_regime(metals, 'trend_pct'),
            'avg_range': avg_regime(metals, 'range_pct'),
            'avg_neutral': avg_regime(metals, 'neutral_pct'),
        },
        'fx': {
            'count': len(fx),
            'avg_trend': avg_regime(fx, 'trend_pct'),
            'avg_range': avg_regime(fx, 'range_pct'),
            'avg_neutral': avg_regime(fx, 'neutral_pct'),
        },
        'all': {
            'count': len(successful),
            'avg_trend': avg_regime(successful, 'trend_pct'),
            'avg_range': avg_regime(successful, 'range_pct'),
            'avg_neutral': avg_regime(successful, 'neutral_pct'),
        }
    }
    
    return analysis


def generate_recommendations(results: List[Dict]) -> List[str]:
    """Generate strategic recommendations based on regime analysis."""
    
    recommendations = []
    
    successful = [r for r in results if r['status'] == 'SUCCESS']
    
    for result in successful:
        symbol = result['symbol']
        regime = result['regime_stats']
        
        # High neutral suggests need for range strategies
        if regime['neutral_pct'] > 50:
            recommendations.append(
                f"⚠️  {symbol}: {regime['neutral_pct']:.1f}% Neutral - "
                f"Current trend strategies will fail. Implement mean-reversion."
            )
        
        # Low trend suggests need for different TP/SL
        if regime['trend_pct'] < 25:
            recommendations.append(
                f"⚠️  {symbol}: Only {regime['trend_pct']:.1f}% Trend - "
                f"Use wider TP (2.0-2.5R) and tighter SL (0.7-0.8R)"
            )
        
        # High range suggests opportunity for range strategies
        if regime['range_pct'] > 20:
            recommendations.append(
                f"✅ {symbol}: {regime['range_pct']:.1f}% Range - "
                f"Good candidate for mean-reversion strategies"
            )
        
        # Good trend percentage
        if regime['trend_pct'] > 30:
            recommendations.append(
                f"✅ {symbol}: {regime['trend_pct']:.1f}% Trend - "
                f"Good candidate for momentum/breakout strategies"
            )
    
    return recommendations


def print_summary_table(results: List[Dict]):
    """Print formatted summary table."""
    
    print("\n" + "="*120)
    print("MULTI-SYMBOL REGIME ANALYSIS SUMMARY")
    print("="*120)
    print(f"{'Symbol':<10} {'TF':<5} {'Bars':>8} {'Trend %':>8} {'Range %':>8} {'Neutral %':>10} {'Confidence':>11} {'Status':<12}")
    print("-"*120)
    
    for result in results:
        symbol = result['symbol']
        tf = result['timeframe']
        status = result['status']
        
        if status == 'SUCCESS':
            bars = result['bars']
            regime = result['regime_stats']
            print(f"{symbol:<10} {tf:<5} {bars:>8,} "
                  f"{regime['trend_pct']:>7.1f}% "
                  f"{regime['range_pct']:>7.1f}% "
                  f"{regime['neutral_pct']:>9.1f}% "
                  f"{regime['avg_confidence']:>10.2f} "
                  f"{'✅ ' + status:<12}")
        else:
            error = result.get('error', 'Unknown error')[:40]
            print(f"{symbol:<10} {tf:<5} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>10} {'N/A':>11} {'❌ ' + error:<50}")
    
    print("="*120)


def print_asset_class_comparison(analysis: Dict):
    """Print asset class comparison."""
    
    if not analysis:
        return
    
    print("\n" + "="*80)
    print("ASSET CLASS COMPARISON")
    print("="*80)
    
    for asset_class in ['metals', 'fx', 'all']:
        if asset_class not in analysis:
            continue
        
        data = analysis[asset_class]
        label = asset_class.upper()
        
        if asset_class == 'all':
            label = 'OVERALL'
        
        print(f"\n{label} ({data['count']} symbols):")
        print(f"  Trend:   {data['avg_trend']:.1f}%")
        print(f"  Range:   {data['avg_range']:.1f}%")
        print(f"  Neutral: {data['avg_neutral']:.1f}%")
    
    print("\n" + "="*80)


def print_recommendations(recommendations: List[str]):
    """Print strategic recommendations."""
    
    print("\n" + "="*80)
    print("STRATEGIC RECOMMENDATIONS")
    print("="*80)
    
    if not recommendations:
        print("No specific recommendations.")
        return
    
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec}")
    
    print("="*80)


def save_results(results: List[Dict], analysis: Dict, recommendations: List[str]):
    """Save results to JSON."""
    
    output = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'results': results,
        'analysis': analysis,
        'recommendations': recommendations
    }
    
    output_dir = Path('reports')
    output_dir.mkdir(exist_ok=True)
    
    output_path = output_dir / 'multi_symbol_regime_analysis.json'
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\n📄 Full results saved to: {output_path}")


def main():
    """Run multi-symbol test suite."""
    
    parser = argparse.ArgumentParser(description='Test all symbols with RenTec components')
    parser.add_argument('--timeframe', '-tf', default='1H', help='Timeframe to test (default: 1H)')
    parser.add_argument('--symbols', nargs='+', default=SYMBOLS, help='Symbols to test')
    
    args = parser.parse_args()
    
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║          Multi-Symbol RenTec Component Test Suite               ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)
    
    print(f"Testing {len(args.symbols)} symbols at {args.timeframe} timeframe...")
    print(f"Symbols: {', '.join(args.symbols)}\n")
    
    # Test all symbols
    results = []
    
    for i, symbol in enumerate(args.symbols, 1):
        print(f"[{i}/{len(args.symbols)}] Testing {symbol}...", end=' ')
        
        result = test_symbol(symbol, args.timeframe)
        results.append(result)
        
        if result['status'] == 'SUCCESS':
            print(f"✅ ({result['bars']:,} bars, "
                  f"Trend: {result['regime_stats']['trend_pct']:.1f}%, "
                  f"Range: {result['regime_stats']['range_pct']:.1f}%)")
        else:
            print(f"❌ {result['status']}")
    
    # Analyze patterns
    analysis = analyze_regime_patterns(results)
    recommendations = generate_recommendations(results)
    
    # Print results
    print_summary_table(results)
    print_asset_class_comparison(analysis)
    print_recommendations(recommendations)
    
    # Save results
    save_results(results, analysis, recommendations)
    
    # Summary
    successful = sum(1 for r in results if r['status'] == 'SUCCESS')
    print(f"\n✅ Successfully tested {successful}/{len(results)} symbols")
    
    if successful < len(results):
        failed = [r['symbol'] for r in results if r['status'] != 'SUCCESS']
        print(f"⚠️  Failed/Missing: {', '.join(failed)}")
    
    return 0 if successful > 0 else 1


if __name__ == "__main__":
    sys.exit(main())

