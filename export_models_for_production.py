"""
Export Models for Production (Fixed for Joblib)

Converts trained Citadel V5 models to production-ready format.
Uses joblib.load() to match the training script's joblib.dump().

Usage:
    python export_models_for_production.py --symbol XAUUSD
"""

import pickle
import json
import joblib  # CRITICAL: Use joblib.load() not pickle.load()
from pathlib import Path
from datetime import datetime
import argparse

# Import ML libraries for proper object reconstruction
try:
    import lightgbm as lgb
except ImportError:
    print("⚠ Warning: LightGBM not installed")
    lgb = None

try:
    import xgboost as xgb
except ImportError:
    print("⚠ Warning: XGBoost not installed")
    xgb = None

try:
    import catboost as cb
except ImportError:
    print("⚠ Warning: CatBoost not installed")
    cb = None

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
except ImportError:
    print("⚠ Warning: Scikit-learn not installed")

import numpy as np
import pandas as pd


class ModelExporter:
    """Export models from training to production format."""
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.source_dir = Path("ML_model/ML_model/models") / symbol
        self.target_dir = Path("production_models")
        self.target_dir.mkdir(exist_ok=True)
    
    def load_model_file(self, model_file: Path):
        """
        Load a model file using joblib (matches training script).
        Returns the loaded data or raises an exception.
        """
        try:
            # Use joblib.load() - this is what citadel_v5.py uses
            data = joblib.load(model_file)
            return data
        except Exception as e:
            raise Exception(f"Failed to load {model_file.name}: {str(e)}")
    
    def export_all_models(self):
        """Export all models for the symbol."""
        print(f"\n{'='*80}")
        print(f"EXPORTING MODELS TO PRODUCTION")
        print(f"Symbol: {self.symbol}")
        print(f"Source: {self.source_dir}")
        print(f"Target: {self.target_dir}")
        print(f"{'='*80}\n")
        
        if not self.source_dir.exists():
            print(f"❌ ERROR: Source directory not found: {self.source_dir}")
            print(f"   Have you trained models yet?")
            print(f"   Run: python citadel_v5.py --symbol {self.symbol} --all-timeframes")
            return
        
        # Find all model files
        model_files = list(self.source_dir.glob(f"{self.symbol}_*.pkl"))
        
        if not model_files:
            print(f"❌ No models found in {self.source_dir}")
            return
        
        print(f"Found {len(model_files)} models to export\n")
        
        exported_count = 0
        failed_count = 0
        failed_files = []
        
        for model_file in model_files:
            try:
                self.export_single_model(model_file)
                exported_count += 1
            except Exception as e:
                print(f"   ❌ Failed to export {model_file.name}")
                print(f"      Error: {str(e)[:150]}")
                failed_count += 1
                failed_files.append((model_file.name, str(e)))
        
        print(f"\n{'='*80}")
        print(f"EXPORT SUMMARY")
        print(f"{'='*80}")
        print(f"✅ Successfully exported: {exported_count}")
        print(f"❌ Failed: {failed_count}")
        
        if failed_files:
            print(f"\nFailed files (showing first 5):")
            for fname, error in failed_files[:5]:
                print(f"   {fname}")
                print(f"      {error[:100]}")
        
        if exported_count > 0:
            print(f"\n✅ Production models ready in: {self.target_dir.absolute()}")
            print(f"\nNext steps:")
            print(f"   1. Review exported models in {self.target_dir}")
            print(f"   2. Test loading: python test_production_models.py")
            print(f"   3. Deploy to production environment")
        print(f"{'='*80}\n")
    
    def export_single_model(self, model_file: Path):
        """Export a single model to production format."""
        # Load trained model (using joblib to match training script)
        model_data = self.load_model_file(model_file)
        
        # Load metadata
        meta_file = model_file.parent / f"{model_file.stem}_meta.json"
        if not meta_file.exists():
            # Try without _meta suffix
            alt_meta = model_file.parent / f"{model_file.stem}.json"
            if alt_meta.exists():
                meta_file = alt_meta
            else:
                raise FileNotFoundError(f"No metadata file found for {model_file.name}")
        
        with open(meta_file, 'r') as f:
            metadata = json.load(f)
        
        # Extract info with safe defaults
        strategy_name = metadata.get('strategy', {}).get('name', 'unknown_strategy')
        timeframe = metadata.get('timeframe', 'unknown')
        strategy_cfg = metadata.get('strategy', {})
        metrics = metadata.get('metrics', {})
        
        print(f"   Exporting: {timeframe} / {strategy_name}")
        
        # Validate required fields
        if 'model' not in model_data:
            raise ValueError(f"Model data missing 'model' key. Available keys: {list(model_data.keys())}")
        if 'feature_cols' not in model_data:
            raise ValueError(f"Model data missing 'feature_cols' key. Available keys: {list(model_data.keys())}")
        
        # Create production model package
        production_model = {
            # Core model components
            'model': model_data['model'],
            'scaler': model_data.get('scaler'),  # Optional
            'feature_cols': model_data['feature_cols'],
            
            # Model identification
            'model_name': strategy_name,
            'symbol': self.symbol,
            'timeframe': timeframe,
            
            # Trading parameters
            'threshold': strategy_cfg.get('prediction_threshold', 0.5),
            'tp_mult': strategy_cfg.get('tp_mult', 2.0),
            'sl_mult': strategy_cfg.get('sl_mult', 1.0),
            'max_loss_r': strategy_cfg.get('max_loss_r', 3.0),
            
            # Filters
            'session_filter': strategy_cfg.get('session_filter'),
            'regime_filter': strategy_cfg.get('regime_filter'),
            
            # Performance metrics (for production monitoring)
            'performance': {
                'win_rate': metrics.get('win_rate', 0),
                'profit_factor': metrics.get('profit_factor', 0),
                'sharpe': metrics.get('sharpe', 0),
                'drawdown_pct': metrics.get('drawdown_pct', 0),
                'trades': metrics.get('trades', 0),
                'avg_win': metrics.get('avg_win', 0),
                'avg_loss': metrics.get('avg_loss', 0),
                'worst_loss': metrics.get('worst_loss', 0)
            },
            
            # Metadata
            'exported_at': datetime.now().isoformat(),
            'model_type': type(model_data['model']).__name__
        }
        
        # Save to production directory (using standard pickle for portability)
        output_file = self.target_dir / f"{self.symbol}_{timeframe}_{strategy_name}.pkl"
        with open(output_file, 'wb') as f:
            pickle.dump(production_model, f, protocol=4)
        
        print(f"      ✅ Saved to: {output_file.name}")
        print(f"         Model type: {production_model['model_type']}")
        print(f"         Threshold: {production_model['threshold']:.2f}")
        print(f"         Features: {len(production_model['feature_cols'])}")
        
        wr = metrics.get('win_rate', 0)
        pf = metrics.get('profit_factor', 0)
        dd = metrics.get('drawdown_pct', 0)
        print(f"         Performance: WR={wr:.1%}, PF={pf:.2f}, DD={dd:.1f}%")


def main():
    parser = argparse.ArgumentParser(description='Export models for production')
    parser.add_argument('--symbol', type=str, default='XAUUSD', help='Symbol to export')
    
    args = parser.parse_args()
    
    exporter = ModelExporter(args.symbol)
    exporter.export_all_models()


if __name__ == '__main__':
    main()