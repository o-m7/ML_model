"""Live prediction runner."""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Optional
import yaml

from ..models.base import BaseModel
from ..features.builders import FeatureBuilder
from ..features.regime import RegimeFeatures
from ..labels.horizons import get_horizon_config

# Import TA-Lib enricher from parent directory
import sys
from pathlib import Path as PathLib
sys.path.insert(0, str(PathLib(__file__).parent.parent.parent))
from add_talib_features import TALibEnricher


def predict(
    symbol: str,
    timeframe: str,
    latest_bars: pd.DataFrame,
    models_dir: str = "models_intraday",
    config_path: str = "intraday_system/config/settings.yaml"
) -> Dict:
    """
    Generate live trading signal.
    
    Args:
        symbol: Trading symbol
        timeframe: Timeframe
        latest_bars: Recent OHLCV data (at least 200 bars)
        models_dir: Directory containing trained models
        config_path: Path to configuration
        
    Returns:
        Signal dictionary with entry/exit levels
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load trained model
    model_path = Path(models_dir) / symbol / timeframe / "model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"No trained model found for {symbol} {timeframe}")
    
    model = BaseModel.load(model_path)
    
    # Load features list
    features_path = Path(models_dir) / symbol / timeframe / "features.json"
    import json
    with open(features_path, 'r') as f:
        feature_names = json.load(f)
    
    # Build features
    df = latest_bars.copy()
    
    # Ensure OHLCV columns are float64 for TA-Lib
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            df[col] = df[col].astype(np.float64)
    
    # Add base features
    builder = FeatureBuilder()
    df = builder.build_all(df)
    
    # Add comprehensive TA-Lib features (includes all 180+ indicators)
    print("  Adding TA-Lib features...")
    df = TALibEnricher.add_all_indicators(df)
    
    # Add regime features
    regime = RegimeFeatures()
    df = regime.add_all(df)
    
    # Add strategy-specific features
    strategy_name = config['strategy_mapping'].get(timeframe)
    if strategy_name:
        strategy_module = _load_strategy(strategy_name)
        # Get the strategy class (assumes class name follows pattern S{N}_{timeframe}{name})
        strategy_class_name = [name for name in dir(strategy_module) if name.startswith('S') and not name.startswith('__')]
        if strategy_class_name:
            strategy_class = getattr(strategy_module, strategy_class_name[0])
            # Pass minimal config needed by strategy
            strategy_config = {'timeframe': timeframe, 'symbol': symbol}
            strategy_instance = strategy_class(strategy_config)
            df = strategy_instance.build_features(df)
    
    # Get latest bar features
    latest_row = df.iloc[-1]
    
    # Handle missing features by filling with 0
    feature_values = []
    missing_features = []
    for feat in feature_names:
        if feat in latest_row.index:
            feature_values.append(latest_row[feat] if not pd.isna(latest_row[feat]) else 0.0)
        else:
            feature_values.append(0.0)  # Default value for missing features
            missing_features.append(feat)
    
    if missing_features:
        print(f"  ⚠️  Warning: {len(missing_features)} features missing, using default values")
        if len(missing_features) <= 10:
            print(f"     Missing: {missing_features[:10]}")
    
    X = np.array([feature_values])
    
    # Predict
    proba = model.predict_proba(X)[0]
    predicted_class = np.argmax(proba)
    confidence = proba[predicted_class]
    
    # Get label config
    horizon_config = get_horizon_config(timeframe)
    
    # Calculate entry levels
    latest_close = latest_row['close']
    atr = latest_row.get('atr14', latest_close * 0.02)
    
    tp_atr_mult = horizon_config['tp_atr_mult']
    sl_atr_mult = horizon_config['sl_atr_mult']
    horizon_bars = horizon_config['horizon_bars']
    
    # Determine signal
    if predicted_class == 1:  # Up
        signal = 'BUY'
        stop_loss = latest_close - (atr * sl_atr_mult)
        take_profit = latest_close + (atr * tp_atr_mult)
        expected_R = tp_atr_mult / sl_atr_mult
    elif predicted_class == 2:  # Down
        signal = 'SELL'
        stop_loss = latest_close + (atr * sl_atr_mult)
        take_profit = latest_close - (atr * tp_atr_mult)
        expected_R = tp_atr_mult / sl_atr_mult
    else:  # Flat
        signal = 'HOLD'
        stop_loss = latest_close
        take_profit = latest_close
        expected_R = 0.0
    
    # Build response
    current_bar_index = len(df) - 1
    
    response = {
        'signal': signal,
        'confidence': float(confidence),
        'entry_ref': float(latest_close),
        'stop_loss': float(stop_loss),
        'take_profit': float(take_profit),
        'expected_R': float(expected_R),
        'horizon_bars': int(horizon_bars),
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'expiry_bar_index': current_bar_index + horizon_bars,
        'atr': float(atr),
        'probabilities': {
            'flat': float(proba[0]),
            'up': float(proba[1]),
            'down': float(proba[2]) if len(proba) > 2 else 0.0
        }
    }
    
    return response


def _load_strategy(strategy_name: str):
    """Load strategy module dynamically."""
    strategy_map = {
        'S1': 'intraday_system.strategies.s1_5m_momo_breakout',
        'S2': 'intraday_system.strategies.s2_15m_meanrevert_vwap',
        'S3': 'intraday_system.strategies.s3_30m_pullback_trend',
        'S4': 'intraday_system.strategies.s4_1h_breakout_retest',
        'S5': 'intraday_system.strategies.s5_2h_momo_adx_atr',
        'S6': 'intraday_system.strategies.s6_4h_mtf_alignment'
    }
    
    import importlib
    module_path = strategy_map.get(strategy_name)
    if module_path:
        return importlib.import_module(module_path)
    return None

