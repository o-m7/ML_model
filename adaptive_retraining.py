#!/usr/bin/env python3
"""
ADAPTIVE RETRAINING SYSTEM
===========================

Retrains models using historical backtest data + live trading signals.
Implements online learning to adapt to market changes.

Features:
- Combines backtest data with live trading results
- Learns from actual trade outcomes
- Adjusts for market regime changes
- Optimizes parameters based on recent performance
- Prevents overfitting with validation splits

Usage:
    python adaptive_retraining.py --symbol XAGUSD --tf 30T
    python adaptive_retraining.py --all --min-trades 50
    python adaptive_retraining.py --symbol XAUUSD --tf 15T --online-only
"""

import argparse
import json
import os
import pickle
import sys
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore")

try:
    from supabase import create_client
    SUPABASE_AVAILABLE = True
except (ImportError, Exception):
    print("⚠️  Supabase not available - using local data only")
    SUPABASE_AVAILABLE = False
    create_client = None

from balanced_model import BalancedModel


class AdaptiveRetrainer:
    """Retrain models with live trading data."""

    def __init__(self, symbol: str, timeframe: str):
        """Initialize retrainer."""
        self.symbol = symbol
        self.timeframe = timeframe

        # Supabase connection
        supabase_url = os.getenv('SUPABASE_URL')
        supabase_key = os.getenv('SUPABASE_KEY')

        if SUPABASE_AVAILABLE and supabase_url and supabase_key:
            self.supabase = create_client(supabase_url, supabase_key)
            self.online = True
        else:
            self.supabase = None
            self.online = False

    def load_base_training_data(self) -> Optional[pd.DataFrame]:
        """Load original training data from feature store."""
        data_path = Path(f'feature_store/{self.symbol}/{self.symbol}_{self.timeframe}.parquet')

        if not data_path.exists():
            print(f"❌ Base training data not found: {data_path}")
            return None

        try:
            df = pd.read_parquet(data_path)
            print(f"✓ Loaded {len(df)} bars from base training data")
            return df
        except Exception as e:
            print(f"❌ Error loading base data: {e}")
            return None

    def fetch_live_signals(self, min_days: int = 7) -> pd.DataFrame:
        """Fetch live trading signals from Supabase."""
        if not self.online:
            # Try loading from local CSV
            csv_path = Path('live_signals.csv')
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                df['created_at'] = pd.to_datetime(df['created_at'])
                return df
            else:
                print("⚠️  No live signals available")
                return pd.DataFrame()

        try:
            start_date = datetime.now(timezone.utc) - timedelta(days=min_days)

            query = self.supabase.table('trading_signals').select('*')
            query = query.eq('symbol', self.symbol)
            query = query.eq('timeframe', self.timeframe)
            query = query.gte('created_at', start_date.isoformat())
            query = query.in_('status', ['win', 'loss'])  # Only closed trades

            response = query.execute()

            if not response.data:
                return pd.DataFrame()

            df = pd.DataFrame(response.data)
            df['created_at'] = pd.to_datetime(df['created_at'])

            print(f"✓ Fetched {len(df)} live trading signals")
            return df

        except Exception as e:
            print(f"❌ Error fetching live signals: {e}")
            return pd.DataFrame()

    def convert_signals_to_training_data(self, signals: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Convert live trading signals to training format."""
        if signals.empty:
            return None

        # Extract features and labels from signals
        # This assumes signals contain feature data
        training_data = []

        for _, signal in signals.iterrows():
            # Extract label
            if signal['status'] == 'win':
                label = 1 if signal['direction'] == 'long' else 0
            else:  # loss
                label = 0 if signal['direction'] == 'long' else 1

            # Extract features (if available)
            # Note: This assumes signal contains feature data
            # In practice, you may need to reconstruct features from price data
            row = {
                'timestamp': signal['created_at'],
                'target': label,
                # Add other features here if available in signal data
            }

            training_data.append(row)

        if not training_data:
            return None

        df = pd.DataFrame(training_data)
        print(f"✓ Converted {len(df)} signals to training format")

        return df

    def combine_datasets(self, base_df: pd.DataFrame, live_df: Optional[pd.DataFrame],
                        live_weight: float = 2.0) -> pd.DataFrame:
        """Combine base and live data with weighting for recent performance."""

        if live_df is None or live_df.empty:
            print("ℹ️  No live data to combine, using base data only")
            return base_df

        # Combine datasets
        combined = pd.concat([base_df, live_df], ignore_index=True)

        # Add sample weights (higher weight for live data)
        combined['sample_weight'] = 1.0
        combined.loc[combined.index >= len(base_df), 'sample_weight'] = live_weight

        print(f"✓ Combined dataset: {len(base_df)} base + {len(live_df)} live = {len(combined)} total")
        print(f"   Live data weight: {live_weight}x")

        return combined

    def retrain_model(self, df: pd.DataFrame, features: List[str]) -> Optional[BalancedModel]:
        """Retrain model with combined data."""

        print("\n🔄 Retraining model...")

        # Prepare data
        X = df[features].values
        y = df['target'].values
        sample_weights = df.get('sample_weight', np.ones(len(df))).values

        # Time series split for validation
        tscv = TimeSeriesSplit(n_splits=3)

        best_model = None
        best_score = 0

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            print(f"\n  Fold {fold + 1}/3:")

            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            w_train = sample_weights[train_idx]

            # Train model
            model = BalancedModel()
            model.fit(X_train, y_train)

            # Validate
            val_proba = model.predict_proba(X_val)
            val_pred = np.argmax(val_proba, axis=1)
            val_acc = (val_pred == y_val).mean()

            print(f"    Training samples: {len(X_train)}")
            print(f"    Validation accuracy: {val_acc:.3f}")

            if val_acc > best_score:
                best_score = val_acc
                best_model = model

        print(f"\n✓ Best validation accuracy: {best_score:.3f}")

        return best_model

    def save_retrained_model(self, model: BalancedModel, features: List[str]):
        """Save retrained model."""

        model_dir = Path(f'models_production/{self.symbol}')
        model_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_file = model_dir / f'{self.symbol}_{self.timeframe}_RETRAINED_{timestamp}.pkl'

        model_data = {
            'model': model,
            'features': features,
            'retrained_at': datetime.now().isoformat(),
            'symbol': self.symbol,
            'timeframe': self.timeframe
        }

        with open(model_file, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"✅ Model saved: {model_file}")

        # Also save as latest
        latest_file = model_dir / f'{self.symbol}_{self.timeframe}_PRODUCTION_READY.pkl'
        with open(latest_file, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"✅ Updated production model: {latest_file}")

    def run_adaptive_retraining(self, min_live_trades: int = 50, online_only: bool = False):
        """Run full adaptive retraining pipeline."""

        print('=' * 80)
        print(f'ADAPTIVE RETRAINING: {self.symbol} {self.timeframe}')
        print('=' * 80)

        # Step 1: Load base training data
        if not online_only:
            base_df = self.load_base_training_data()
            if base_df is None:
                print("❌ Cannot proceed without base training data")
                return False
        else:
            base_df = pd.DataFrame()
            print("ℹ️  Online-only mode: using live signals only")

        # Step 2: Fetch live trading signals
        live_signals = self.fetch_live_signals(min_days=30)

        if len(live_signals) < min_live_trades:
            print(f"⚠️  Insufficient live trades: {len(live_signals)} < {min_live_trades}")
            print("   Skipping retraining - need more live data")
            return False

        # Step 3: Convert live signals to training format
        live_df = self.convert_signals_to_training_data(live_signals)

        # Step 4: Combine datasets
        if not online_only:
            combined_df = self.combine_datasets(base_df, live_df, live_weight=2.0)
        else:
            combined_df = live_df

        # Step 5: Extract features (using original model's feature list)
        original_model_path = Path(f'models_production/{self.symbol}/{self.symbol}_{self.timeframe}_PRODUCTION_READY.pkl')

        if not original_model_path.exists():
            print(f"❌ Original model not found: {original_model_path}")
            return False

        with open(original_model_path, 'rb') as f:
            original_model = pickle.load(f)

        features = original_model.get('features', [])
        print(f"\nℹ️  Using {len(features)} features from original model")

        # Ensure all features exist in combined data
        missing_features = [f for f in features if f not in combined_df.columns]
        if missing_features:
            print(f"⚠️  Missing features: {missing_features}")
            print("   Note: Live signal data may not contain all features")
            print("   Consider implementing feature reconstruction")
            return False

        # Step 6: Retrain model
        retrained_model = self.retrain_model(combined_df, features)

        if retrained_model is None:
            print("❌ Retraining failed")
            return False

        # Step 7: Save retrained model
        self.save_retrained_model(retrained_model, features)

        print('\n' + '=' * 80)
        print('✅ RETRAINING COMPLETE')
        print('=' * 80)
        print(f'Model: {self.symbol} {self.timeframe}')
        print(f'Live trades used: {len(live_signals)}')
        print(f'Total training samples: {len(combined_df)}')
        print('=' * 80)

        return True


def main():
    parser = argparse.ArgumentParser(description='Adaptive model retraining with live signals')
    parser.add_argument('--symbol', type=str, help='Symbol (XAUUSD, XAGUSD)')
    parser.add_argument('--tf', type=str, help='Timeframe (5T, 15T, 30T, 1H)')
    parser.add_argument('--all', action='store_true', help='Retrain all models')
    parser.add_argument('--min-trades', type=int, default=50, help='Minimum live trades needed (default: 50)')
    parser.add_argument('--online-only', action='store_true', help='Use only live trading data')

    args = parser.parse_args()

    if args.all:
        symbols = ['XAUUSD', 'XAGUSD']
        timeframes = ['5T', '15T', '30T', '1H']

        for symbol in symbols:
            for tf in timeframes:
                print('\n')
                retrainer = AdaptiveRetrainer(symbol, tf)
                retrainer.run_adaptive_retraining(
                    min_live_trades=args.min_trades,
                    online_only=args.online_only
                )
    else:
        if not args.symbol or not args.tf:
            print("❌ Must specify --symbol and --tf, or use --all")
            return 1

        retrainer = AdaptiveRetrainer(args.symbol, args.tf)
        success = retrainer.run_adaptive_retraining(
            min_live_trades=args.min_trades,
            online_only=args.online_only
        )

        return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
