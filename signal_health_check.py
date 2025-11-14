#!/usr/bin/env python3
"""
SIGNAL HEALTH CHECK ENDPOINT
============================
Quick health check for signal generation pipeline.
Returns JSON status that can be monitored by external systems.

Usage:
  python3 signal_health_check.py

  Returns exit code 0 if healthy, 1 if unhealthy
  Prints JSON to stdout with health metrics
"""

import os
import sys
import json
from datetime import datetime, timezone
from supabase import create_client

# Configuration
MAX_SIGNAL_AGE_MINUTES = 5  # Signals should be < 5 minutes old
MIN_SYMBOLS_EXPECTED = 4    # Expect at least 4 symbols generating signals

def check_signal_health():
    """
    Check signal generation health.
    Returns dict with health status and metrics.
    """
    try:
        # Connect to Supabase
        supabase_url = os.getenv('SUPABASE_URL')
        supabase_key = os.getenv('SUPABASE_KEY')

        if not supabase_url or not supabase_key:
            return {
                'status': 'error',
                'healthy': False,
                'error': 'Missing SUPABASE credentials',
                'timestamp': datetime.now(timezone.utc).isoformat()
            }

        supabase = create_client(supabase_url, supabase_key)
        now = datetime.now(timezone.utc)

        # Get recent signals (last 10 minutes)
        response = supabase.table('live_signals').select(
            'symbol,timeframe,timestamp,confidence,edge,signal_type'
        ).order('timestamp', desc=True).limit(100).execute()

        if not response.data:
            return {
                'status': 'unhealthy',
                'healthy': False,
                'reason': 'No signals found in database',
                'signal_count': 0,
                'timestamp': now.isoformat()
            }

        # Analyze signals
        signals = response.data
        latest_signal = signals[0]
        latest_time = datetime.fromisoformat(latest_signal['timestamp'].replace('Z', '+00:00'))
        staleness_minutes = (now - latest_time).total_seconds() / 60

        # Count unique symbols in last 5 minutes
        recent_signals = [
            s for s in signals
            if (now - datetime.fromisoformat(s['timestamp'].replace('Z', '+00:00'))).total_seconds() < 300
        ]
        unique_symbols = len(set(s['symbol'] for s in recent_signals))

        # Calculate metrics
        avg_confidence = sum(s['confidence'] for s in recent_signals) / len(recent_signals) if recent_signals else 0
        avg_edge = sum(s['edge'] for s in recent_signals) / len(recent_signals) if recent_signals else 0

        # Health determination
        is_healthy = (
            staleness_minutes <= MAX_SIGNAL_AGE_MINUTES and
            unique_symbols >= MIN_SYMBOLS_EXPECTED and
            avg_confidence > 0.4 and
            avg_edge > 0.02
        )

        # Build response
        result = {
            'status': 'healthy' if is_healthy else 'degraded',
            'healthy': is_healthy,
            'metrics': {
                'latest_signal_age_minutes': round(staleness_minutes, 2),
                'signal_count_last_5min': len(recent_signals),
                'unique_symbols': unique_symbols,
                'avg_confidence': round(avg_confidence, 3),
                'avg_edge': round(avg_edge, 4),
                'latest_symbol': latest_signal['symbol'],
                'latest_timeframe': latest_signal['timeframe'],
            },
            'checks': {
                'staleness_ok': staleness_minutes <= MAX_SIGNAL_AGE_MINUTES,
                'symbol_count_ok': unique_symbols >= MIN_SYMBOLS_EXPECTED,
                'confidence_ok': avg_confidence > 0.4,
                'edge_ok': avg_edge > 0.02,
            },
            'timestamp': now.isoformat()
        }

        # Add warnings
        warnings = []
        if staleness_minutes > MAX_SIGNAL_AGE_MINUTES:
            warnings.append(f"Signals are stale ({staleness_minutes:.1f} minutes old)")
        if unique_symbols < MIN_SYMBOLS_EXPECTED:
            warnings.append(f"Too few symbols ({unique_symbols} < {MIN_SYMBOLS_EXPECTED})")
        if avg_confidence <= 0.4:
            warnings.append(f"Low average confidence ({avg_confidence:.2f})")
        if avg_edge <= 0.02:
            warnings.append(f"Low average edge ({avg_edge:.3f})")

        if warnings:
            result['warnings'] = warnings

        return result

    except Exception as e:
        return {
            'status': 'error',
            'healthy': False,
            'error': str(e),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }

def main():
    """Run health check and print JSON result."""
    result = check_signal_health()

    # Print JSON
    print(json.dumps(result, indent=2))

    # Exit with appropriate code
    if result['healthy']:
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == '__main__':
    main()
