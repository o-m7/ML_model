#!/usr/bin/env python3
"""
Retrain From Live Trades - Continuous learning from actual trading results
Focuses on learning from mistakes and improving model accuracy
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import accuracy_score, classification_report
import lightgbm as lgb

# Import from production system
sys.path.insert(0, str(Path(__file__).parent))
from production_final_system import BalancedModel, SYMBOL_PARAMS, ProductionConfig, BacktestEngine
from benchmark_validator import BenchmarkValidator


class LiveTradeRetrainer:
    """Retrain models using feedback from live trading"""
    
    def __init__(self):
        self.trades_dir = Path("live_trades")
        self.models_dir = Path("models_production")
        self.feature_store = Path("feature_store")
        
    def load_live_trades(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Load live trades for a symbol/timeframe"""
        
        trade_file = self.trades_dir / f"{symbol}_{timeframe}_live_trades.csv"
        
        if not trade_file.exists():
            print(f"  ⚠️  No live trades found: {trade_file}")
            return pd.DataFrame()
        
        df = pd.read_csv(trade_file)
        print(f"  📊 Loaded {len(df)} live trades from {trade_file.name}")
        
        return df
    
    def identify_error_patterns(self, live_trades: pd.DataFrame) -> dict:
        """
        Identify specific patterns where the model is wrong
        
        Returns:
            Dictionary with error patterns and adjustments needed
        """
        
        if live_trades.empty:
            return {}
        
        losers = live_trades[live_trades['pnl'] < 0].copy()
        winners = live_trades[live_trades['pnl'] > 0].copy()
        
        patterns = {
            'total_trades': len(live_trades),
            'losers': len(losers),
            'winners': len(winners),
            'live_win_rate': len(winners) / len(live_trades) if len(live_trades) > 0 else 0,
            'adjustments': []
        }
        
        # Pattern 1: Confidence calibration
        if not losers.empty and 'confidence' in losers.columns:
            avg_loser_conf = losers['confidence'].mean()
            avg_winner_conf = winners['confidence'].mean() if not winners.empty else 0
            
            if avg_loser_conf > avg_winner_conf - 0.05:
                patterns['adjustments'].append({
                    'type': 'confidence_threshold',
                    'issue': f"Losers have similar confidence ({avg_loser_conf:.3f}) to winners ({avg_winner_conf:.3f})",
                    'action': 'Increase MIN_CONFIDENCE by 0.05',
                    'new_min_conf': avg_winner_conf + 0.05
                })
        
        # Pattern 2: Directional bias
        if not losers.empty:
            loser_directions = losers['direction'].value_counts()
            winner_directions = winners['direction'].value_counts() if not winners.empty else pd.Series()
            
            for direction in ['long', 'short']:
                loser_count = loser_directions.get(direction, 0)
                winner_count = winner_directions.get(direction, 0)
                
                if loser_count > winner_count * 1.5:
                    patterns['adjustments'].append({
                        'type': 'directional_bias',
                        'issue': f"{direction.upper()} trades losing more ({loser_count}L vs {winner_count}W)",
                        'action': f"Increase {direction} class weight in retraining"
                    })
        
        # Pattern 3: Exit reason analysis
        if 'exit_reason' in losers.columns:
            exit_reasons = losers['exit_reason'].value_counts()
            
            if 'stop_loss' in exit_reasons and exit_reasons['stop_loss'] > len(losers) * 0.7:
                patterns['adjustments'].append({
                    'type': 'stop_loss',
                    'issue': f"{exit_reasons['stop_loss']} trades hit stop loss ({exit_reasons['stop_loss']/len(losers)*100:.1f}%)",
                    'action': "Widen stop loss OR tighten entry criteria"
                })
        
        return patterns
    
    def boost_losing_patterns_in_training(self, X_train: np.ndarray, y_train: np.ndarray, 
                                          patterns: dict) -> np.ndarray:
        """
        Adjust sample weights to focus on avoiding losing patterns
        
        Returns:
            Sample weights array
        """
        
        sample_weights = np.ones(len(y_train))
        
        # If live trading shows poor win rate, boost similar patterns in training
        if patterns.get('live_win_rate', 0.5) < 0.45:
            # Boost weight of losing examples (class 0 and 2 for Flat/Down)
            losing_mask = (y_train != 1)  # Not Up
            sample_weights[losing_mask] *= 1.5
            print(f"  ⚙️  Boosting {losing_mask.sum()} losing pattern samples by 1.5x")
        
        return sample_weights
    
    def retrain_model(self, symbol: str, timeframe: str, patterns: dict) -> bool:
        """
        Retrain a single model with live trade feedback
        
        Returns:
            True if retraining successful and model improved
        """
        
        print(f"\n{'='*80}")
        print(f"🔄 RETRAINING: {symbol} {timeframe}")
        print(f"{'='*80}\n")
        
        # Load existing model
        model_file = self.models_dir / symbol / f"{symbol}_{timeframe}_PRODUCTION_READY.pkl"
        
        if not model_file.exists():
            print(f"  ❌ Model not found: {model_file}")
            return False
        
        with open(model_file, 'rb') as f:
            old_model_data = pickle.load(f)
        
        old_model = old_model_data['model']
        features = old_model_data['features']
        backtest_results = old_model_data['backtest_results']
        
        print(f"  📂 Loaded existing model")
        print(f"  📊 Old performance: WR={backtest_results.get('win_rate', 0):.1f}%, PF={backtest_results.get('profit_factor', 0):.2f}")
        
        # Load historical training data
        data_file = self.feature_store / symbol / f"{symbol}_{timeframe}.parquet"
        
        if not data_file.exists():
            print(f"  ❌ Training data not found: {data_file}")
            return False
        
        df = pd.read_parquet(data_file)
        print(f"  📊 Loaded {len(df)} bars of training data")
        
        # Prepare features and labels
        if 'label' not in df.columns:
            print(f"  ❌ No labels in training data")
            return False
        
        # Use same features as original model
        missing_features = [f for f in features if f not in df.columns]
        if missing_features:
            print(f"  ❌ Missing features: {missing_features[:5]}")
            return False
        
        X = df[features].values
        y = df['label'].values
        
        # Split into train/test (use last 20% as OOS)
        split_idx = int(len(df) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print(f"  📊 Train: {len(X_train)} | Test: {len(X_test)}")
        
        # Adjust sample weights based on live trade patterns
        sample_weights = self.boost_losing_patterns_in_training(X_train, y_train, patterns)
        
        # Adjust class weights based on directional bias
        class_weights = {0: 1.0, 1: 1.0, 2: 1.0}  # Flat, Up, Down
        
        for adjustment in patterns.get('adjustments', []):
            if adjustment['type'] == 'directional_bias':
                if 'long' in adjustment['issue']:
                    class_weights[1] *= 1.3  # Boost Up class
                    print(f"  ⚙️  Boosting UP class weight to {class_weights[1]:.2f}")
                elif 'short' in adjustment['issue']:
                    class_weights[2] *= 1.3  # Boost Down class
                    print(f"  ⚙️  Boosting DOWN class weight to {class_weights[2]:.2f}")
        
        # Train new model
        print(f"  🔄 Training new model with live trade feedback...")
        
        new_model = BalancedModel(
            n_estimators=150,
            learning_rate=0.03,
            max_depth=5,
            class_weight=class_weights
        )
        
        new_model.fit(X_train, y_train, sample_weight=sample_weights)
        
        # Evaluate on test set
        y_pred_old = old_model.predict(X_test)
        y_pred_new = new_model.predict(X_test)
        
        old_acc = accuracy_score(y_test, y_pred_old)
        new_acc = accuracy_score(y_test, y_pred_new)
        
        print(f"\n  📊 COMPARISON:")
        print(f"    Old model accuracy: {old_acc:.3f}")
        print(f"    New model accuracy: {new_acc:.3f}")
        print(f"    Improvement: {(new_acc - old_acc):.3f}")
        
        # Check if accuracy improved
        if new_acc < old_acc - 0.02:  # Significant degradation
            print(f"  ❌ New model accuracy degraded significantly. Keeping old model.")
            return False
        
        # Run full backtest on new model to validate benchmarks
        print(f"\n  🔄 Running full backtest to validate benchmarks...")
        
        # Get symbol params
        params = SYMBOL_PARAMS.get(symbol, {}).get(timeframe, {})
        
        # Create backtester
        backtester = BacktestEngine(
            initial_capital=10000,
            position_pct=params.get('position_pct', 0.005),
            commission_pct=0.00004,
            slippage_pct=0.0001,
            tp_atr_mult=params.get('tp_mult', 1.5),
            sl_atr_mult=params.get('sl_mult', 1.0),
            max_drawdown_circuit_breaker=0.07
        )
        
        # Generate predictions on OOS data
        predictions = new_model.predict(X_test)
        probabilities = new_model.predict_proba(X_test)
        
        # Prepare OOS dataframe (need price data for backtesting)
        oos_df = df.iloc[split_idx:].copy()
        oos_df['prediction'] = predictions
        oos_df['confidence'] = probabilities.max(axis=1)
        
        # Run backtest
        try:
            new_backtest_results = backtester.backtest(
                oos_df,
                predictions,
                probabilities,
                min_confidence=params.get('min_conf', 0.40),
                min_edge=params.get('min_edge', 0.05)
            )
        except Exception as e:
            print(f"  ❌ Backtest failed: {e}")
            return False
        
        # Validate against benchmarks
        print(f"\n  🔍 Validating against production benchmarks...")
        passes, failures = BenchmarkValidator.validate(new_backtest_results, timeframe, strict=True)
        
        BenchmarkValidator.print_validation(symbol, timeframe, new_backtest_results, passes, failures)
        
        # Deploy only if passes benchmarks
        if passes:
            print(f"  ✅ New model PASSES all benchmarks! Deploying...")
            
            # Update model file
            new_model_data = {
                'model': new_model,
                'features': features,
                'symbol': symbol,
                'timeframe': timeframe,
                'backtest_results': new_backtest_results,  # Use NEW backtest results
                'live_retrained_at': datetime.now(timezone.utc).isoformat(),
                'live_trade_feedback': patterns,
                'old_backtest_results': backtest_results  # Keep old for comparison
            }
            
            # Backup old model
            backup_file = model_file.parent / f"{symbol}_{timeframe}_BACKUP_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            with open(backup_file, 'wb') as f:
                pickle.dump(old_model_data, f)
            print(f"  💾 Backed up old model: {backup_file.name}")
            
            # Save new model
            with open(model_file, 'wb') as f:
                pickle.dump(new_model_data, f)
            print(f"  💾 Deployed new model: {model_file.name}")
            
            return True
        else:
            print(f"  ❌ New model FAILS benchmarks. Keeping old model.")
            print(f"  📊 Benchmark failures:")
            for failure in failures:
                print(f"     • {failure}")
            return False
    
    def retrain_all_from_live_trades(self) -> None:
        """Retrain all models that have live trade data"""
        
        print(f"\n{'='*80}")
        print(f"🧠 CONTINUOUS LEARNING FROM LIVE TRADES")
        print(f"{'='*80}\n")
        
        if not self.trades_dir.exists():
            print(f"❌ No live trades directory found: {self.trades_dir}")
            return
        
        trade_files = list(self.trades_dir.glob("*_live_trades.csv"))
        
        if not trade_files:
            print(f"⚠️  No live trade files found in {self.trades_dir}")
            return
        
        print(f"📊 Found {len(trade_files)} symbol/timeframe combinations with live trades\n")
        
        improved = 0
        failed = 0
        
        for trade_file in trade_files:
            # Parse symbol and timeframe from filename
            # Format: SYMBOL_TIMEFRAME_live_trades.csv
            parts = trade_file.stem.replace('_live_trades', '').rsplit('_', 1)
            if len(parts) != 2:
                continue
            
            symbol, timeframe = parts
            
            # Load live trades
            live_trades = self.load_live_trades(symbol, timeframe)
            
            if live_trades.empty:
                continue
            
            # Identify error patterns
            patterns = self.identify_error_patterns(live_trades)
            
            print(f"\n  🔍 Error Patterns for {symbol} {timeframe}:")
            print(f"    Live Win Rate: {patterns['live_win_rate']*100:.1f}%")
            print(f"    Winners: {patterns['winners']}, Losers: {patterns['losers']}")
            
            if patterns['adjustments']:
                print(f"    Adjustments needed:")
                for adj in patterns['adjustments']:
                    print(f"      - {adj['type']}: {adj['action']}")
            
            # Retrain model
            if self.retrain_model(symbol, timeframe, patterns):
                improved += 1
            else:
                failed += 1
        
        print(f"\n{'='*80}")
        print(f"📊 RETRAINING SUMMARY")
        print(f"{'='*80}")
        print(f"  Total processed: {len(trade_files)}")
        print(f"  ✅ Improved: {improved}")
        print(f"  ❌ Failed: {failed}")
        print(f"\n{'='*80}\n")


def main():
    """Main execution"""
    
    retrainer = LiveTradeRetrainer()
    retrainer.retrain_all_from_live_trades()
    
    print("✅ Live trade retraining complete!")
    print("\nNext steps:")
    print("1. Convert updated models to ONNX: python3 convert_models_to_onnx.py")
    print("2. Sync to Supabase: python3 supabase_sync.py")
    print("3. Monitor improved performance in live trading\n")


if __name__ == "__main__":
    main()

