#!/usr/bin/env python3
"""
MODEL STATUS DASHBOARD
======================

Simple dashboard showing current model deployment status and expected performance.
No external dependencies - reads from local ONNX metadata files.

Usage:
    python model_status_dashboard.py
    python model_status_dashboard.py --symbol XAGUSD
"""

import argparse
import json
from pathlib import Path
from datetime import datetime


def load_model_status():
    """Load model status from ONNX metadata."""
    status = {}

    for symbol in ['XAUUSD', 'XAGUSD']:
        status[symbol] = {}

        onnx_dir = Path(f'models_onnx/{symbol}')
        if not onnx_dir.exists():
            continue

        for json_file in sorted(onnx_dir.glob('*.json')):
            with open(json_file) as f:
                data = json.load(f)

            tf = data['timeframe']
            params = data.get('params', {})
            backtest = data.get('backtest_results', {})

            status[symbol][tf] = {
                'num_features': data.get('num_features', 0),
                'tp': params.get('tp', 0),
                'sl': params.get('sl', 0),
                'min_conf': params.get('min_conf', 0),
                'min_edge': params.get('min_edge', 0),
                'pos_size': params.get('pos_size', 0),
                'total_trades': backtest.get('total_trades', 0),
                'win_rate': backtest.get('win_rate', 0),
                'profit_factor': backtest.get('profit_factor', 0),
                'total_return_pct': backtest.get('total_return_pct', 0),
                'max_drawdown_pct': backtest.get('max_drawdown_pct', 0),
                'sharpe_ratio': backtest.get('sharpe_ratio', 0),
            }

    return status


def print_dashboard(symbol_filter=None):
    """Print model status dashboard."""

    status = load_model_status()

    print('=' * 120)
    print('MODEL DEPLOYMENT STATUS DASHBOARD')
    print('=' * 120)
    print(f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print('=' * 120)

    symbols_to_show = [symbol_filter] if symbol_filter else ['XAUUSD', 'XAGUSD']

    for symbol in symbols_to_show:
        if symbol not in status:
            continue

        print(f'\n{symbol}:')
        print('-' * 120)
        print('{:<8} {:<10} {:<8} {:<8} {:<12} {:<12} {:<12} {:<10} {:<10} {:<10} {:<10}'.format(
            'TF', 'Features', 'TP', 'Conf', 'pos_size', 'Trades', 'Win Rate', 'PF', 'Return', 'DD', 'Sharpe'
        ))
        print('-' * 120)

        for tf in ['5T', '15T', '30T', '1H', '4H']:
            if tf not in status[symbol]:
                continue

            data = status[symbol][tf]

            # Status indicator
            wr = data['win_rate']
            pf = data['profit_factor']
            ret = data['total_return_pct']

            if ret > 50 and wr > 60 and pf > 2.0:
                status_icon = '✅'
            elif ret > 20 and wr > 50 and pf > 1.5:
                status_icon = '🟢'
            elif ret > 10 and wr > 45 and pf > 1.3:
                status_icon = '🟡'
            else:
                status_icon = '🔴'

            print('{} {:<6} {:<10} {:<8.1f} {:<8.2f} {:<12.2f} {:<12} {:<12.1f}% {:<10.2f} {:<10.1f}% {:<10.1f}% {:<10.2f}'.format(
                status_icon,
                tf,
                data['num_features'],
                data['tp'],
                data['min_conf'],
                data['pos_size'],
                data['total_trades'],
                data['win_rate'],
                data['profit_factor'],
                data['total_return_pct'],
                data['max_drawdown_pct'],
                data['sharpe_ratio']
            ))

    print('\n' + '=' * 120)
    print('PERFORMANCE RATINGS:')
    print('  ✅ Excellent:  Return >50%, WR >60%, PF >2.0')
    print('  🟢 Very Good:  Return >20%, WR >50%, PF >1.5')
    print('  🟡 Good:       Return >10%, WR >45%, PF >1.3')
    print('  🔴 Marginal:   Below good thresholds')
    print('=' * 120)

    # Summary
    print('\nDEPLOYMENT SUMMARY:')
    total_models = sum(len(status[s]) for s in status)
    excellent = 0
    good = 0
    marginal = 0

    for symbol in status:
        for tf in status[symbol]:
            data = status[symbol][tf]
            ret = data['total_return_pct']
            wr = data['win_rate']
            pf = data['profit_factor']

            if ret > 50 and wr > 60 and pf > 2.0:
                excellent += 1
            elif ret > 20 and wr > 50 and pf > 1.5:
                good += 1
            elif ret > 10 and wr > 45 and pf > 1.3:
                good += 1
            else:
                marginal += 1

    print(f'  Total Models Deployed: {total_models}')
    print(f'  ✅ Excellent: {excellent}')
    print(f'  🟢 Good: {good}')
    print(f'  🔴 Marginal: {marginal}')
    print('=' * 120)


def main():
    parser = argparse.ArgumentParser(description='Model deployment status dashboard')
    parser.add_argument('--symbol', type=str, help='Filter by symbol (XAUUSD, XAGUSD)')

    args = parser.parse_args()

    print_dashboard(args.symbol)


if __name__ == '__main__':
    main()
