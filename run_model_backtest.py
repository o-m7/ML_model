#!/usr/bin/env python3
"""
INTEGRATED MODEL BACKTEST RUNNER
==================================
Runs prop-firm backtests using your actual trained ML models.

Integrates:
- Your trained models from models_rentec/
- Feature engineering from live_feature_utils.py
- Prop-firm backtesting engine from backtest_model.py

Usage:
    python run_model_backtest.py --symbol XAUUSD --timeframe 15T
    python run_model_backtest.py --symbol XAGUSD --timeframe 30T --balance 50000
"""

import argparse
import pickle
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# Import your modules
from live_feature_utils import build_feature_frame
from backtest_model import PropFirmBacktester, BacktestResults

# Import custom model classes for unpickling
try:
    from balanced_model import BalancedModel
except ImportError:
    BalancedModel = None


def load_model(symbol: str, timeframe: str) -> tuple:
    """
    Load trained model for symbol/timeframe.

    Returns:
        (model, feature_columns)
    """
    model_path = Path(f"models_rentec/{symbol}/{symbol}_{timeframe}.pkl")

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    with open(model_path, "rb") as f:
        model_data = pickle.load(f)

    # Model structure varies, handle both formats
    if isinstance(model_data, dict):
        model = model_data.get("model")
        features = model_data.get("features", [])
    else:
        # Direct model object
        model = model_data
        features = []

    return model, features


def load_historical_data(symbol: str, timeframe: str, lookback_days: int = 365) -> pd.DataFrame:
    """
    Load historical OHLCV data for backtesting.

    In production, replace this with your actual data source:
    - Polygon API
    - Local database
    - Parquet files
    """
    # Check feature store first
    feature_file = Path(f"feature_store/{symbol}/{symbol}_{timeframe}.parquet")

    if feature_file.exists():
        print(f"Loading data from feature store: {feature_file}")
        df = pd.read_parquet(feature_file)

        # Feature store already has processed features
        # Need to get raw OHLCV for backtesting
        if all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume']):
            df['symbol'] = symbol
            return df

    # If no feature store, would need to fetch from Polygon
    # For now, raise error
    raise FileNotFoundError(
        f"No data found for {symbol} {timeframe}. "
        f"Please download data first using your data pipeline."
    )


def prepare_features(price_df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Build features for all bars in price_df.

    This uses your live_feature_utils.build_feature_frame function.
    """
    print("Building features...")

    # Extract OHLCV only
    ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
    ohlcv_df = price_df[ohlcv_cols].copy()

    # Build features using your function
    features_df = build_feature_frame(ohlcv_df)

    # Add symbol column
    features_df['symbol'] = symbol

    # Reorder columns to put symbol first
    cols = ['symbol'] + [c for c in features_df.columns if c != 'symbol']
    features_df = features_df[cols]

    print(f"Built {len(features_df.columns)} features for {len(features_df)} bars")

    return features_df


class IntegratedModelPredictor:
    """
    Predictor that builds features on-the-fly during backtest.

    Handles direct model prediction for binary classification models.
    """

    def __init__(
        self,
        symbol: str,
        timeframe: str,
        full_price_df: pd.DataFrame,
        features_df: pd.DataFrame
    ):
        """
        Initialize with pre-computed features.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe (e.g., '15T')
            full_price_df: Full OHLCV dataframe
            features_df: Pre-computed features for all bars
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.full_price_df = full_price_df
        self.features_df = features_df

        # Load the model for this specific timeframe
        self.model, self.required_features = self._load_model()

        # Create lookup dict for fast access
        self.features_lookup = {
            idx: row for idx, row in features_df.iterrows()
        }

        print(f"   Loaded model with {len(self.required_features)} features")

    def _load_model(self):
        """Load the specific model for this timeframe."""
        model_path = Path(f"models_rentec/{self.symbol}/{self.symbol}_{self.timeframe}.pkl")

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        with open(model_path, "rb") as f:
            model_data = pickle.load(f)

        # Handle different pickle formats
        if isinstance(model_data, dict):
            model = model_data.get("model")
            features = model_data.get("features", [])
        else:
            # Direct model object
            model = model_data
            features = []

        return model, features

    def __call__(self, row: pd.Series) -> int:
        """
        Predict signal for a bar (called by backtester).

        Args:
            row: Current OHLCV bar

        Returns:
            +1 for long, -1 for short, 0 for flat
        """
        # Get timestamp
        timestamp = row.name

        # Look up features for this timestamp
        if timestamp not in self.features_lookup:
            # No features available (likely early bars)
            return 0

        features_row = self.features_lookup[timestamp]

        # Get prediction
        try:
            # Check we have required features
            if not self.required_features:
                # No feature list - can't predict
                return 0

            missing = set(self.required_features) - set(features_row.index)
            if missing:
                # Missing required features
                return 0

            # Prepare features in correct order
            X = features_row[self.required_features].fillna(0).infer_objects(copy=False).values.reshape(1, -1)

            # Get prediction probabilities
            probs = self.model.predict_proba(X)

            if probs.shape[1] == 3:
                # 3-class model: [flat_prob, long_prob, short_prob]
                flat_prob = float(probs[0, 0])
                long_prob = float(probs[0, 1])
                short_prob = float(probs[0, 2])

                # Return direction with highest probability if above threshold
                max_prob = max(flat_prob, long_prob, short_prob)

                if long_prob == max_prob and long_prob >= 0.40:
                    return 1  # Long
                elif short_prob == max_prob and short_prob >= 0.40:
                    return -1  # Short
                else:
                    return 0  # Flat

            elif probs.shape[1] == 2:
                # Binary classification: [down_prob, up_prob]
                down_prob = float(probs[0, 0])
                up_prob = float(probs[0, 1])

                # Determine signal with confidence threshold
                if up_prob > down_prob and up_prob >= 0.50:
                    return 1  # Long
                elif down_prob > up_prob and down_prob >= 0.50:
                    return -1  # Short
                else:
                    return 0  # Flat
            else:
                # Unexpected format
                return 0

        except Exception as e:
            # On error, return flat (silent to avoid spam)
            return 0


def run_integrated_backtest(
    symbol: str,
    timeframe: str,
    initial_balance: float = 25_000,
    profit_target_pct: float = 0.06,
    max_drawdown_pct: float = 0.04,
    risk_per_trade_pct: float = 0.005,
    lookback_days: int = 365,
    verbose: bool = True
) -> BacktestResults:
    """
    Run complete backtest with your ML models.

    Args:
        symbol: Trading symbol (XAUUSD, XAGUSD)
        timeframe: Timeframe (5T, 15T, 30T, 1H)
        initial_balance: Starting balance
        profit_target_pct: Profit target as decimal
        max_drawdown_pct: Max drawdown as decimal
        risk_per_trade_pct: Risk per trade as decimal
        lookback_days: Days of historical data
        verbose: Print progress

    Returns:
        BacktestResults
    """
    print("=" * 80)
    print(f"INTEGRATED MODEL BACKTEST: {symbol} {timeframe}")
    print("=" * 80)
    print()

    # 1. Load historical data
    print(f"Loading historical data...")
    try:
        price_df = load_historical_data(symbol, timeframe, lookback_days)
        print(f"✅ Loaded {len(price_df):,} bars")
        print(f"   Date range: {price_df.index[0]} to {price_df.index[-1]}")
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        sys.exit(1)

    # 2. Build features
    print(f"\nBuilding features...")
    try:
        features_df = prepare_features(price_df, symbol)
        print(f"✅ Built features: {len(features_df.columns)} columns")

        # Align price_df with features_df (features may have fewer rows due to dropna)
        aligned_price_df = price_df.loc[features_df.index]
        print(f"   Aligned data: {len(aligned_price_df)} bars")

    except Exception as e:
        print(f"❌ Failed to build features: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 3. Create predictor
    print(f"\nInitializing model predictor for {symbol} {timeframe}...")
    try:
        predict_func = IntegratedModelPredictor(
            symbol=symbol,
            timeframe=timeframe,
            full_price_df=aligned_price_df,
            features_df=features_df
        )
    except Exception as e:
        print(f"❌ Failed to initialize predictor: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 4. Run backtest
    print(f"\nRunning backtest...\n")
    backtester = PropFirmBacktester(
        price_df=aligned_price_df,
        predict_function=predict_func,
        initial_balance=initial_balance,
        profit_target_pct=profit_target_pct,
        max_drawdown_pct=max_drawdown_pct,
        risk_per_trade_pct=risk_per_trade_pct,
        verbose=verbose,
        seed=42
    )

    results = backtester.run()

    # 5. Export results
    print("\n📊 Exporting results...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("backtest_results")
    output_dir.mkdir(exist_ok=True)

    # Export trades
    if results.trades:
        trades_df = pd.DataFrame([t.to_dict() for t in results.trades])
        trades_file = output_dir / f"{symbol}_{timeframe}_trades_{timestamp}.csv"
        trades_df.to_csv(trades_file, index=False)
        print(f"✅ Trades: {trades_file}")

    # Export equity curve
    equity_file = output_dir / f"{symbol}_{timeframe}_equity_{timestamp}.csv"
    results.equity_curve.to_csv(equity_file, index=False)
    print(f"✅ Equity curve: {equity_file}")

    # Export summary
    summary = {
        "symbol": symbol,
        "timeframe": timeframe,
        "status": "PASSED" if results.passed else "FAILED",
        "initial_balance": initial_balance,
        "final_equity": results.final_equity,
        "total_pnl": results.total_pnl,
        "pnl_pct": results.total_pnl / initial_balance * 100,
        "max_drawdown_usd": results.max_drawdown_usd,
        "max_drawdown_pct": results.max_drawdown_pct,
        "num_trades": results.num_trades,
        "num_wins": results.num_wins,
        "num_losses": results.num_losses,
        "win_rate": results.win_rate,
        "avg_win": results.avg_win,
        "avg_loss": results.avg_loss,
        "profit_factor": results.profit_factor,
        "total_costs": results.total_costs,
        "failure_reason": results.failure_reason or "N/A"
    }

    summary_df = pd.DataFrame([summary])
    summary_file = output_dir / f"{symbol}_{timeframe}_summary_{timestamp}.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"✅ Summary: {summary_file}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run ML Model Backtest with Prop-Firm Rules")
    parser.add_argument("--symbol", required=True, choices=["XAUUSD", "XAGUSD"],
                       help="Trading symbol")
    parser.add_argument("--timeframe", required=True, choices=["5T", "15T", "30T", "1H"],
                       help="Timeframe")
    parser.add_argument("--balance", type=float, default=25_000,
                       help="Initial balance (default: $25,000)")
    parser.add_argument("--profit-target", type=float, default=0.06,
                       help="Profit target %% (default: 6%%)")
    parser.add_argument("--max-drawdown", type=float, default=0.04,
                       help="Max drawdown %% (default: 4%%)")
    parser.add_argument("--risk-per-trade", type=float, default=0.005,
                       help="Risk per trade %% (default: 0.5%%)")
    parser.add_argument("--lookback-days", type=int, default=365,
                       help="Days of historical data (default: 365)")
    parser.add_argument("--quiet", action="store_true",
                       help="Suppress verbose output")

    args = parser.parse_args()

    # Run backtest
    results = run_integrated_backtest(
        symbol=args.symbol,
        timeframe=args.timeframe,
        initial_balance=args.balance,
        profit_target_pct=args.profit_target,
        max_drawdown_pct=args.max_drawdown,
        risk_per_trade_pct=args.risk_per_trade,
        lookback_days=args.lookback_days,
        verbose=not args.quiet
    )

    # Exit code
    sys.exit(0 if results.passed else 1)


if __name__ == "__main__":
    main()
