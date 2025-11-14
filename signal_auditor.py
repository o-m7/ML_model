#!/usr/bin/env python3
"""
SIGNAL AUDIT & ANALYSIS SYSTEM
================================

Analyzes historical signals to identify patterns and improve models.

Features:
- Audits which signals won/lost and why
- Identifies patterns in winning vs losing trades
- Suggests parameter adjustments
- Detects model drift and performance degradation
- Recommends retraining triggers

Usage:
    python signal_auditor.py --days 7 --symbol XAGUSD
    python signal_auditor.py --audit-all --days 30
    python signal_auditor.py --recommend-retraining
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    from supabase import create_client
    SUPABASE_AVAILABLE = True
except (ImportError, Exception):
    print("⚠️  Supabase not available - using local CSV files")
    SUPABASE_AVAILABLE = False
    create_client = None


@dataclass
class SignalAnalysis:
    """Analysis of signal performance patterns."""
    symbol: str
    timeframe: str
    total_signals: int
    wins: int
    losses: int
    win_rate: float
    avg_win_return: float
    avg_loss_return: float
    profit_factor: float

    # Pattern analysis
    best_session: str
    worst_session: str
    best_confidence_range: Tuple[float, float]
    worst_confidence_range: Tuple[float, float]

    # Drift detection
    recent_win_rate: float  # Last 20% of signals
    overall_win_rate: float
    drift_detected: bool
    drift_severity: float  # Percentage difference

    # Recommendations
    should_retrain: bool
    suggested_min_conf: float
    suggested_pos_size: float


class SignalAuditor:
    """Audits trading signals and recommends improvements."""

    def __init__(self, supabase_url: Optional[str] = None, supabase_key: Optional[str] = None):
        """Initialize auditor."""
        self.supabase_url = supabase_url or os.getenv('SUPABASE_URL')
        self.supabase_key = supabase_key or os.getenv('SUPABASE_KEY')

        if SUPABASE_AVAILABLE and self.supabase_url and self.supabase_key:
            self.supabase = create_client(self.supabase_url, self.supabase_key)
            self.online = True
        else:
            self.supabase = None
            self.online = False

    def fetch_signals(self, symbol: Optional[str] = None, timeframe: Optional[str] = None,
                     days: int = 30) -> pd.DataFrame:
        """Fetch signals for analysis."""
        if not self.online:
            # Try loading from local CSV
            csv_path = Path('signal_history.csv')
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                df['created_at'] = pd.to_datetime(df['created_at'])
                return df
            else:
                print("❌ No data available - please provide signal_history.csv or configure Supabase")
                return pd.DataFrame()

        try:
            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(days=days)

            query = self.supabase.table('trading_signals').select('*')

            if symbol:
                query = query.eq('symbol', symbol)
            if timeframe:
                query = query.eq('timeframe', timeframe)

            query = query.gte('created_at', start_date.isoformat())
            query = query.order('created_at', desc=False)

            response = query.execute()

            if not response.data:
                return pd.DataFrame()

            df = pd.DataFrame(response.data)
            df['created_at'] = pd.to_datetime(df['created_at'])

            return df

        except Exception as e:
            print(f"❌ Error fetching signals: {e}")
            return pd.DataFrame()

    def analyze_patterns(self, df: pd.DataFrame, symbol: str, timeframe: str) -> SignalAnalysis:
        """Analyze signal patterns to find improvements."""

        if df.empty:
            return SignalAnalysis(
                symbol=symbol, timeframe=timeframe, total_signals=0,
                wins=0, losses=0, win_rate=0, avg_win_return=0, avg_loss_return=0,
                profit_factor=0, best_session='N/A', worst_session='N/A',
                best_confidence_range=(0, 0), worst_confidence_range=(0, 0),
                recent_win_rate=0, overall_win_rate=0, drift_detected=False,
                drift_severity=0, should_retrain=False, suggested_min_conf=0,
                suggested_pos_size=0
            )

        # Filter closed trades
        closed = df[df['status'].isin(['win', 'loss'])].copy()

        if closed.empty:
            return SignalAnalysis(
                symbol=symbol, timeframe=timeframe, total_signals=len(df),
                wins=0, losses=0, win_rate=0, avg_win_return=0, avg_loss_return=0,
                profit_factor=0, best_session='N/A', worst_session='N/A',
                best_confidence_range=(0, 0), worst_confidence_range=(0, 0),
                recent_win_rate=0, overall_win_rate=0, drift_detected=False,
                drift_severity=0, should_retrain=False, suggested_min_conf=0,
                suggested_pos_size=0
            )

        # Basic metrics
        wins = closed[closed['status'] == 'win']
        losses = closed[closed['status'] == 'loss']

        win_rate = len(wins) / len(closed) * 100 if len(closed) > 0 else 0
        avg_win = wins['return_pct'].mean() if len(wins) > 0 else 0
        avg_loss = abs(losses['return_pct'].mean()) if len(losses) > 0 else 0

        total_wins = wins['pnl'].sum() if len(wins) > 0 else 0
        total_losses = abs(losses['pnl'].sum()) if len(losses) > 0 else 0
        profit_factor = total_wins / total_losses if total_losses > 0 else 0

        # Session analysis (if timestamp available)
        if 'created_at' in closed.columns:
            closed['hour'] = pd.to_datetime(closed['created_at']).dt.hour
            closed['session'] = closed['hour'].apply(self._classify_session)

            session_wr = closed.groupby('session')['status'].apply(
                lambda x: (x == 'win').sum() / len(x) * 100
            )

            best_session = session_wr.idxmax() if len(session_wr) > 0 else 'N/A'
            worst_session = session_wr.idxmin() if len(session_wr) > 0 else 'N/A'
        else:
            best_session = 'N/A'
            worst_session = 'N/A'

        # Confidence analysis (if available)
        if 'confidence' in closed.columns:
            # Divide into quartiles
            closed['conf_quartile'] = pd.qcut(closed['confidence'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
            conf_wr = closed.groupby('conf_quartile')['status'].apply(
                lambda x: (x == 'win').sum() / len(x) * 100
            )

            best_q = conf_wr.idxmax() if len(conf_wr) > 0 else 'Q4'
            worst_q = conf_wr.idxmin() if len(conf_wr) > 0 else 'Q1'

            best_conf_range = (
                closed[closed['conf_quartile'] == best_q]['confidence'].min(),
                closed[closed['conf_quartile'] == best_q]['confidence'].max()
            )
            worst_conf_range = (
                closed[closed['conf_quartile'] == worst_q]['confidence'].min(),
                closed[closed['conf_quartile'] == worst_q]['confidence'].max()
            )

            # Recommend higher min_conf based on best quartile
            suggested_min_conf = best_conf_range[0]
        else:
            best_conf_range = (0, 0)
            worst_conf_range = (0, 0)
            suggested_min_conf = 0.4

        # Drift detection - compare recent vs overall performance
        split_idx = int(len(closed) * 0.8)  # Last 20% as "recent"
        recent = closed.iloc[split_idx:]

        recent_win_rate = len(recent[recent['status'] == 'win']) / len(recent) * 100 if len(recent) > 0 else win_rate
        drift_severity = abs(recent_win_rate - win_rate)
        drift_detected = drift_severity > 10  # >10% change indicates drift

        # Retraining recommendation
        should_retrain = (
            drift_detected or
            win_rate < 45 or
            profit_factor < 1.2 or
            len(closed) > 100  # Enough data for retraining
        )

        # Position sizing recommendation
        if profit_factor > 2.0 and win_rate > 60:
            suggested_pos_size = 0.5  # Increase size for strong performers
        elif profit_factor > 1.5 and win_rate > 50:
            suggested_pos_size = 0.3  # Moderate size
        else:
            suggested_pos_size = 0.2  # Conservative size

        return SignalAnalysis(
            symbol=symbol,
            timeframe=timeframe,
            total_signals=len(df),
            wins=len(wins),
            losses=len(losses),
            win_rate=win_rate,
            avg_win_return=avg_win,
            avg_loss_return=avg_loss,
            profit_factor=profit_factor,
            best_session=best_session,
            worst_session=worst_session,
            best_confidence_range=best_conf_range,
            worst_confidence_range=worst_conf_range,
            recent_win_rate=recent_win_rate,
            overall_win_rate=win_rate,
            drift_detected=drift_detected,
            drift_severity=drift_severity,
            should_retrain=should_retrain,
            suggested_min_conf=suggested_min_conf,
            suggested_pos_size=suggested_pos_size
        )

    def _classify_session(self, hour: int) -> str:
        """Classify trading session by hour (UTC)."""
        if 22 <= hour or hour < 2:
            return 'Asian'
        elif 7 <= hour < 16:
            return 'London'
        elif 13 <= hour < 21:
            return 'NY'
        else:
            return 'Off-hours'

    def generate_audit_report(self, symbol: Optional[str] = None,
                            timeframe: Optional[str] = None, days: int = 30):
        """Generate comprehensive audit report."""

        print('=' * 100)
        print(f'SIGNAL AUDIT REPORT - Last {days} Days')
        print('=' * 100)
        print(f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        print('=' * 100)

        symbols_to_check = [symbol] if symbol else ['XAUUSD', 'XAGUSD']

        all_analyses = []

        for sym in symbols_to_check:
            timeframes_to_check = [timeframe] if timeframe else ['5T', '15T', '30T', '1H']

            print(f'\n{sym}:')
            print('-' * 100)

            for tf in timeframes_to_check:
                df = self.fetch_signals(sym, tf, days)
                analysis = self.analyze_patterns(df, sym, tf)
                all_analyses.append(analysis)

                # Status indicator
                if analysis.should_retrain:
                    status = '🔴 RETRAIN'
                elif analysis.drift_detected:
                    status = '🟡 DRIFT'
                elif analysis.win_rate > 50:
                    status = '✅ GOOD'
                else:
                    status = '⚠️  WATCH'

                print(f'\n  {tf} {status}')
                print(f'    Signals: {analysis.total_signals} | Closed: {analysis.wins + analysis.losses}')
                print(f'    Win Rate: {analysis.win_rate:.1f}% (Recent: {analysis.recent_win_rate:.1f}%)')
                print(f'    Profit Factor: {analysis.profit_factor:.2f}')
                print(f'    Avg Win: +{analysis.avg_win_return:.2f}% | Avg Loss: -{analysis.avg_loss_return:.2f}%')

                if analysis.drift_detected:
                    print(f'    ⚠️  DRIFT DETECTED: {analysis.drift_severity:.1f}% performance change')

                if analysis.best_session != 'N/A':
                    print(f'    Best Session: {analysis.best_session} | Worst: {analysis.worst_session}')

                if analysis.best_confidence_range[0] > 0:
                    print(f'    Best Confidence Range: {analysis.best_confidence_range[0]:.2f}-{analysis.best_confidence_range[1]:.2f}')
                    print(f'    Suggested min_conf: {analysis.suggested_min_conf:.2f}')

                print(f'    Suggested pos_size: {analysis.suggested_pos_size:.2f}')

        print('\n' + '=' * 100)
        print('RETRAINING RECOMMENDATIONS:')
        print('=' * 100)

        needs_retraining = [a for a in all_analyses if a.should_retrain]

        if needs_retraining:
            print(f'\n🔴 {len(needs_retraining)} models need retraining:\n')
            for a in needs_retraining:
                reasons = []
                if a.drift_detected:
                    reasons.append(f'drift ({a.drift_severity:.1f}%)')
                if a.win_rate < 45:
                    reasons.append(f'low WR ({a.win_rate:.1f}%)')
                if a.profit_factor < 1.2:
                    reasons.append(f'low PF ({a.profit_factor:.2f})')
                if a.wins + a.losses > 100:
                    reasons.append('enough data')

                print(f'  • {a.symbol} {a.timeframe}: {", ".join(reasons)}')

            print('\nRetrain command:')
            for a in needs_retraining:
                print(f'  python adaptive_retraining.py --symbol {a.symbol} --tf {a.timeframe}')
        else:
            print('\n✅ All models performing well - no retraining needed')

        print('\n' + '=' * 100)

    def export_recommendations(self, output_file: str = 'retraining_recommendations.json'):
        """Export recommendations to JSON for automated processing."""

        symbols = ['XAUUSD', 'XAGUSD']
        timeframes = ['5T', '15T', '30T', '1H', '4H']

        recommendations = {
            'generated_at': datetime.now().isoformat(),
            'models': []
        }

        for symbol in symbols:
            for tf in timeframes:
                df = self.fetch_signals(symbol, tf, days=30)
                analysis = self.analyze_patterns(df, symbol, tf)

                recommendations['models'].append({
                    'symbol': symbol,
                    'timeframe': tf,
                    'should_retrain': analysis.should_retrain,
                    'drift_detected': analysis.drift_detected,
                    'drift_severity': analysis.drift_severity,
                    'win_rate': analysis.win_rate,
                    'profit_factor': analysis.profit_factor,
                    'suggested_min_conf': analysis.suggested_min_conf,
                    'suggested_pos_size': analysis.suggested_pos_size,
                    'closed_trades': analysis.wins + analysis.losses
                })

        with open(output_file, 'w') as f:
            json.dump(recommendations, f, indent=2)

        print(f'✅ Recommendations exported to {output_file}')


def main():
    parser = argparse.ArgumentParser(description='Audit trading signals and recommend improvements')
    parser.add_argument('--symbol', type=str, help='Symbol (XAUUSD, XAGUSD)')
    parser.add_argument('--tf', type=str, help='Timeframe (5T, 15T, 30T, 1H)')
    parser.add_argument('--days', type=int, default=30, help='Days to analyze (default: 30)')
    parser.add_argument('--audit-all', action='store_true', help='Audit all symbols and timeframes')
    parser.add_argument('--recommend-retraining', action='store_true', help='Export retraining recommendations')

    args = parser.parse_args()

    auditor = SignalAuditor()

    if args.recommend_retraining:
        auditor.export_recommendations()
    elif args.audit_all:
        auditor.generate_audit_report(days=args.days)
    else:
        auditor.generate_audit_report(symbol=args.symbol, timeframe=args.tf, days=args.days)


if __name__ == '__main__':
    main()
