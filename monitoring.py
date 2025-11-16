"""
Model monitoring and retraining module.

Implements performance monitoring, drift detection, and automated retraining logic
aligned with institutional practices.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

from config import Config
from features import build_features
from model_training import TradingModel
from backtest import Backtest
from metrics import calculate_metrics


class PerformanceMonitor:
    """
    Monitor model performance and trigger retraining when needed.

    Tracks:
    - Rolling performance metrics
    - Performance degradation
    - Model drift indicators
    """

    def __init__(self, config: Config):
        """
        Initialize monitor.

        Args:
            config: Configuration object
        """
        self.config = config
        self.performance_log_path = config.monitoring.performance_log_path
        self.performance_log_path.parent.mkdir(parents=True, exist_ok=True)

        # Load existing log or create new
        self.performance_log = self._load_log()

    def _load_log(self) -> List[Dict[str, Any]]:
        """Load performance log from disk."""
        if self.performance_log_path.exists():
            with open(self.performance_log_path, 'r') as f:
                return json.load(f)
        else:
            return []

    def _save_log(self):
        """Save performance log to disk."""
        with open(self.performance_log_path, 'w') as f:
            json.dump(self.performance_log, f, indent=2)

    def log_performance(
        self,
        symbol: str,
        timeframe: int,
        model_version: str,
        metrics: Dict[str, float],
        period_start: str,
        period_end: str
    ):
        """
        Log performance for a period.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe in minutes
            model_version: Model version/timestamp
            metrics: Performance metrics dict
            period_start: Start date of performance period
            period_end: End date of performance period
        """
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'timeframe': timeframe,
            'model_version': model_version,
            'period_start': period_start,
            'period_end': period_end,
            **metrics
        }

        self.performance_log.append(log_entry)
        self._save_log()

        print(f"Logged performance: {symbol} {timeframe}m, "
              f"WR={metrics.get('win_rate', 0)*100:.1f}%, "
              f"PF={metrics.get('profit_factor', 0):.2f}")

    def get_recent_performance(
        self,
        symbol: str,
        timeframe: int,
        n_periods: int = 5
    ) -> pd.DataFrame:
        """
        Get recent performance history.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            n_periods: Number of recent periods

        Returns:
            DataFrame with performance history
        """
        # Filter log for this symbol/timeframe
        filtered = [
            entry for entry in self.performance_log
            if entry['symbol'] == symbol and entry['timeframe'] == timeframe
        ]

        if len(filtered) == 0:
            return pd.DataFrame()

        # Convert to DataFrame and get recent
        df = pd.DataFrame(filtered)
        df = df.sort_values('timestamp').tail(n_periods)

        return df

    def check_performance_degradation(
        self,
        symbol: str,
        timeframe: int
    ) -> Dict[str, bool]:
        """
        Check if performance has degraded below thresholds.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe

        Returns:
            Dict with degradation flags
        """
        recent_perf = self.get_recent_performance(symbol, timeframe, n_periods=3)

        if len(recent_perf) == 0:
            return {
                'degraded': False,
                'reason': 'No performance history'
            }

        # Get average of recent periods
        avg_sharpe = recent_perf['sharpe_ratio'].mean()
        avg_pf = recent_perf['profit_factor'].mean()
        max_dd = recent_perf['max_dd_pct'].min()  # Most negative

        # Check thresholds
        degradation_flags = {
            'sharpe_below_threshold': avg_sharpe < self.config.monitoring.min_rolling_sharpe,
            'pf_below_threshold': avg_pf < self.config.monitoring.min_rolling_pf,
            'dd_above_threshold': abs(max_dd) > self.config.monitoring.max_rolling_dd
        }

        degraded = any(degradation_flags.values())

        result = {
            'degraded': degraded,
            'avg_sharpe': avg_sharpe,
            'avg_pf': avg_pf,
            'max_dd': max_dd,
            **degradation_flags
        }

        if degraded:
            reasons = []
            if degradation_flags['sharpe_below_threshold']:
                reasons.append(f"Sharpe {avg_sharpe:.2f} < {self.config.monitoring.min_rolling_sharpe}")
            if degradation_flags['pf_below_threshold']:
                reasons.append(f"PF {avg_pf:.2f} < {self.config.monitoring.min_rolling_pf}")
            if degradation_flags['dd_above_threshold']:
                reasons.append(f"DD {max_dd:.1f}% > {self.config.monitoring.max_rolling_dd}%")

            result['reason'] = '; '.join(reasons)

        return result

    def should_retrain(
        self,
        symbol: str,
        timeframe: int,
        last_training_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Determine if model should be retrained.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            last_training_date: Date of last training (ISO format)

        Returns:
            Dict with retraining recommendation
        """
        reasons = []

        # Check time since last training
        time_based_retrain = False
        if last_training_date:
            last_train = datetime.fromisoformat(last_training_date)
            days_since = (datetime.now() - last_train).days

            if days_since >= self.config.monitoring.retrain_frequency_days:
                time_based_retrain = True
                reasons.append(f"Time-based: {days_since} days since last training")

        # Check performance degradation
        degradation = self.check_performance_degradation(symbol, timeframe)
        performance_based_retrain = degradation['degraded']

        if performance_based_retrain:
            reasons.append(f"Performance degradation: {degradation.get('reason', 'Unknown')}")

        should_retrain = time_based_retrain or performance_based_retrain

        return {
            'should_retrain': should_retrain,
            'time_based': time_based_retrain,
            'performance_based': performance_based_retrain,
            'reasons': reasons,
            'degradation_details': degradation
        }


def schedule_retraining(config: Config):
    """
    Scheduled retraining routine.

    This function would be called periodically (e.g., weekly) to:
    1. Check if retraining is needed
    2. Retrain models if required
    3. Validate new models
    4. Deploy if performance is acceptable

    In production, this would be triggered by a cron job or scheduler.
    """
    print("="*60)
    print("SCHEDULED RETRAINING CHECK")
    print(f"Timestamp: {datetime.now()}")
    print("="*60)

    monitor = PerformanceMonitor(config)

    # Check each symbol/timeframe
    for symbol in config.data.symbols:
        for timeframe in config.data.timeframes:
            print(f"\nChecking {symbol} {timeframe}m...")

            # Load model metadata to get last training date
            model_dir = config.monitoring.model_dir
            model_files = list(model_dir.glob(f"{symbol}_{timeframe}_*.pkl"))

            if len(model_files) == 0:
                print(f"  No existing model found. Training required.")
                continue

            # Get most recent model
            latest_model = max(model_files, key=lambda p: p.stat().st_mtime)
            model_timestamp = latest_model.stem.split('_')[-1]

            # Convert to datetime (assuming format YYYYMMDD_HHMMSS)
            try:
                last_train_dt = datetime.strptime(model_timestamp, '%Y%m%d_%H%M%S')
                last_train_str = last_train_dt.isoformat()
            except:
                last_train_str = None

            # Check if retraining needed
            retrain_decision = monitor.should_retrain(symbol, timeframe, last_train_str)

            if retrain_decision['should_retrain']:
                print(f"  ✓ Retraining recommended:")
                for reason in retrain_decision['reasons']:
                    print(f"    - {reason}")

                # In production, trigger retraining here
                # For now, just log the recommendation
                print(f"  → Would trigger retraining (not implemented in this example)")
            else:
                print(f"  ✗ No retraining needed")

                # Log current performance
                recent_perf = monitor.get_recent_performance(symbol, timeframe, n_periods=1)
                if len(recent_perf) > 0:
                    latest = recent_perf.iloc[-1]
                    print(f"    Recent: WR={latest.get('win_rate', 0)*100:.1f}%, "
                          f"PF={latest.get('profit_factor', 0):.2f}, "
                          f"Sharpe={latest.get('sharpe_ratio', 0):.2f}")

    print("\n" + "="*60)
    print("Retraining check complete")
    print("="*60)


def backtest_on_new_data(
    model: TradingModel,
    new_data: pd.DataFrame,
    config: Config,
    symbol: str,
    timeframe: int
) -> Dict[str, Any]:
    """
    Backtest existing model on new data for monitoring.

    Args:
        model: Trained model
        new_data: New OHLCV data (not used in training)
        config: Configuration
        symbol: Symbol
        timeframe: Timeframe

    Returns:
        Performance metrics dict
    """
    from features import build_features, get_feature_columns

    # Build features on new data
    df_features = build_features(new_data, config)

    # Generate predictions
    feature_cols = get_feature_columns(df_features)
    df_features['ml_proba'] = model.predict_proba(df_features[feature_cols])

    # Run backtest
    bt = Backtest(df_features, model, config, symbol, timeframe)
    results = bt.run()

    trade_log = bt.get_trade_log()
    equity_curve = results['equity']

    # Calculate metrics
    metrics = calculate_metrics(equity_curve, trade_log, config.risk.initial_capital)

    return {
        'win_rate': metrics.win_rate,
        'profit_factor': metrics.profit_factor,
        'max_dd_pct': metrics.max_drawdown_pct,
        'sharpe_ratio': metrics.sharpe_ratio,
        'total_trades': metrics.total_trades,
        'avg_r': metrics.avg_r
    }


if __name__ == "__main__":
    from config import get_default_config

    config = get_default_config()

    # Create monitor
    monitor = PerformanceMonitor(config)

    # Simulate logging some performance
    for i in range(5):
        period_start = (datetime.now() - timedelta(days=30*(i+1))).isoformat()
        period_end = (datetime.now() - timedelta(days=30*i)).isoformat()

        metrics = {
            'win_rate': 0.60 + np.random.randn() * 0.05,
            'profit_factor': 1.5 + np.random.randn() * 0.2,
            'max_dd_pct': -5.0 + np.random.randn() * 1.0,
            'sharpe_ratio': 1.2 + np.random.randn() * 0.3,
            'total_trades': int(50 + np.random.randn() * 10)
        }

        monitor.log_performance(
            "XAUUSD",
            5,
            f"model_v{i}",
            metrics,
            period_start,
            period_end
        )

    # Check degradation
    print("\nChecking performance degradation...")
    degradation = monitor.check_performance_degradation("XAUUSD", 5)
    print(f"Degraded: {degradation['degraded']}")
    print(f"Details: {degradation}")

    # Check if should retrain
    print("\nChecking retraining decision...")
    decision = monitor.should_retrain("XAUUSD", 5, datetime.now().isoformat())
    print(f"Should retrain: {decision['should_retrain']}")
    print(f"Reasons: {decision['reasons']}")

    # Run scheduled check
    print("\n")
    schedule_retraining(config)
