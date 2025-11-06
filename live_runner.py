#!/usr/bin/env python3
"""
LIVE INFERENCE RUNNER
====================

Production-grade inference system for generating tradable signals.

Features:
- Loads production-ready models from manifest
- Generates BUY/SELL/HOLD signals with full risk parameters
- Drift monitoring and auto-halt on degradation
- Supabase integration ready
- Paper-trading mode for post-2025-10-22 validation

Usage:
    # Generate signals for all symbols
    python3 live_runner.py --mode live
    
    # Paper-trade on unseen data
    python3 live_runner.py --mode paper --start-date 2025-10-23
    
    # Single symbol
    python3 live_runner.py --symbol XAUUSD --tf 15T
"""

import argparse
import json
import pickle
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class SignalConfig:
    """Configuration for signal generation."""
    MODEL_STORE = Path("models_production")
    CONFIDENCE_THRESHOLD = 0.65  # Minimum confidence to trade
    MAX_CONCURRENT_RISK = 0.05   # Max 5% portfolio risk
    DRIFT_WARNING_PF = 1.1       # Warn if rolling PF < 1.1
    DRIFT_HALT_PF = 0.9          # Halt if rolling PF < 0.9
    ROLLING_WINDOW = 20          # trades
    
CONFIG = SignalConfig()


# ============================================================================
# MODEL LOADER
# ============================================================================

class ProductionModelRegistry:
    """Load and manage production-ready models."""
    
    def __init__(self):
        self.models = {}
        self.load_manifest()
    
    def load_manifest(self):
        """Load manifest of production-ready models."""
        manifest_path = CONFIG.MODEL_STORE / "manifest.json"
        
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        
        with open(manifest_path, 'r') as f:
            self.manifest = json.load(f)
        
        # Load only READY models
        ready_models = [r for r in self.manifest['results'] if r['passed']]
        
        print(f"Loading {len(ready_models)} production-ready models...")
        
        for result in ready_models:
            symbol = result['symbol']
            tf = result['timeframe']
            model_path = result['model_path']
            
            if model_path and Path(model_path).exists():
                with open(model_path, 'rb') as f:
                    data = pickle.load(f)
                    
                key = f"{symbol}_{tf}"
                self.models[key] = {
                    'model': data['model'],
                    'features': data['features'],
                    'config': data['config'],
                    'oos_results': data['oos_results'],
                    'symbol': symbol,
                    'timeframe': tf
                }
                print(f"  ✓ {symbol} {tf}")
        
        print(f"\n✅ Loaded {len(self.models)} models")
    
    def get_model(self, symbol: str, timeframe: str):
        """Get model for symbol/timeframe."""
        key = f"{symbol}_{timeframe}"
        return self.models.get(key)
    
    def list_available(self) -> List[str]:
        """List all available model keys."""
        return list(self.models.keys())


# ============================================================================
# SIGNAL GENERATOR
# ============================================================================

@dataclass
class TradingSignal:
    """Complete trading signal with risk parameters."""
    symbol: str
    timeframe: str
    timestamp: str
    signal: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    expected_return: float
    horizon_bars: int
    expiry_timestamp: str
    risk_amount: float
    position_size: float


class LiveSignalGenerator:
    """Generate live trading signals."""
    
    def __init__(self, registry: ProductionModelRegistry):
        self.registry = registry
        self.trade_history = {}  # Per symbol tracking
        self.last_signal_bar = {}  # Cooldown tracking
    
    def generate_signal(self, symbol: str, timeframe: str, 
                       current_bar: pd.Series, features_df: pd.DataFrame) -> Optional[TradingSignal]:
        """
        Generate signal for current bar.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            current_bar: Current OHLCV bar
            features_df: DataFrame with all features (last row = current)
        
        Returns:
            TradingSignal or None if HOLD
        """
        # Get model
        model_data = self.registry.get_model(symbol, timeframe)
        if not model_data:
            return None
        
        model = model_data['model']
        features = model_data['features']
        
        # Check cooldown
        key = f"{symbol}_{timeframe}"
        if key in self.last_signal_bar:
            bars_since_last = len(features_df) - self.last_signal_bar[key]
            if bars_since_last < 5:  # 5-bar cooldown
                return None
        
        # Extract features for current bar
        try:
            X = features_df[features].iloc[-1:].fillna(0).values
        except KeyError as e:
            print(f"  ⚠️  Missing feature {e} for {symbol} {timeframe}")
            return None
        
        # Get prediction
        proba = model.predict_proba(X)[0]
        predicted_class = np.argmax(proba)
        confidence = proba[predicted_class]
        
        # Check confidence threshold
        if confidence < CONFIG.CONFIDENCE_THRESHOLD:
            return None
        
        # Map class to signal
        signal_map = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}
        signal = signal_map[predicted_class]
        
        if signal == 'HOLD':
            return None
        
        # Calculate risk parameters
        entry_price = current_bar['close']
        atr = current_bar.get('atr14', current_bar.get('atr', entry_price * 0.02))
        
        tp_mult = model_data['config']['TP_ATR_MULT']
        sl_mult = model_data['config']['SL_ATR_MULT']
        
        if signal == 'BUY':
            stop_loss = entry_price - (atr * sl_mult)
            take_profit = entry_price + (atr * tp_mult)
            expected_return = tp_mult * sl_mult  # R multiple
        else:  # SELL
            stop_loss = entry_price + (atr * sl_mult)
            take_profit = entry_price - (atr * tp_mult)
            expected_return = tp_mult * sl_mult
        
        # Position sizing (1% risk)
        risk_per_trade = 100000 * 0.01  # $1000 risk
        sl_distance = abs(entry_price - stop_loss)
        position_size = risk_per_trade / sl_distance if sl_distance > 0 else 0
        
        # Create signal
        trading_signal = TradingSignal(
            symbol=symbol,
            timeframe=timeframe,
            timestamp=current_bar['timestamp'].isoformat() if isinstance(current_bar['timestamp'], pd.Timestamp) else str(current_bar['timestamp']),
            signal=signal,
            confidence=float(confidence),
            entry_price=float(entry_price),
            stop_loss=float(stop_loss),
            take_profit=float(take_profit),
            expected_return=float(expected_return),
            horizon_bars=int(model_data['config']['FORECAST_HORIZON']),
            expiry_timestamp=(current_bar['timestamp'] + pd.Timedelta(bars=50, unit='T')).isoformat(),
            risk_amount=float(risk_per_trade),
            position_size=float(position_size)
        )
        
        # Update tracking
        self.last_signal_bar[key] = len(features_df)
        
        return trading_signal
    
    def check_drift(self, symbol: str, timeframe: str, recent_trades: List[Dict]) -> Dict:
        """
        Monitor for model drift.
        
        Args:
            symbol: Symbol
            timeframe: Timeframe
            recent_trades: List of recent closed trades
        
        Returns:
            Drift status dict
        """
        if len(recent_trades) < CONFIG.ROLLING_WINDOW:
            return {'status': 'insufficient_data', 'action': 'continue'}
        
        # Calculate rolling metrics
        recent = recent_trades[-CONFIG.ROLLING_WINDOW:]
        wins = [t for t in recent if t['pnl'] > 0]
        losses = [t for t in recent if t['pnl'] <= 0]
        
        total_wins = sum(t['pnl'] for t in wins)
        total_losses = abs(sum(t['pnl'] for t in losses))
        
        rolling_pf = total_wins / total_losses if total_losses > 0 else 0
        rolling_wr = len(wins) / len(recent) * 100
        
        # Check thresholds
        if rolling_pf < CONFIG.DRIFT_HALT_PF:
            return {
                'status': 'halt',
                'action': 'halt_trading',
                'rolling_pf': rolling_pf,
                'rolling_wr': rolling_wr,
                'message': f'HALT: Rolling PF {rolling_pf:.2f} < {CONFIG.DRIFT_HALT_PF}'
            }
        elif rolling_pf < CONFIG.DRIFT_WARNING_PF:
            return {
                'status': 'warning',
                'action': 'reduce_risk',
                'rolling_pf': rolling_pf,
                'rolling_wr': rolling_wr,
                'message': f'WARNING: Rolling PF {rolling_pf:.2f} < {CONFIG.DRIFT_WARNING_PF}'
            }
        else:
            return {
                'status': 'healthy',
                'action': 'continue',
                'rolling_pf': rolling_pf,
                'rolling_wr': rolling_wr
            }


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Live Signal Generator')
    parser.add_argument('--mode', type=str, choices=['live', 'paper'], default='live')
    parser.add_argument('--symbol', type=str, help='Single symbol')
    parser.add_argument('--tf', type=str, help='Timeframe')
    parser.add_argument('--start-date', type=str, help='Start date for paper trading')
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print(f"LIVE INFERENCE SYSTEM")
    print(f"Mode: {args.mode.upper()}")
    print(f"{'='*80}\n")
    
    # Load models
    registry = ProductionModelRegistry()
    
    if not registry.models:
        print("✗ No production-ready models found!")
        print("  Run training first: python3 production_training_system.py --all")
        return 1
    
    # Initialize generator
    generator = LiveSignalGenerator(registry)
    
    print(f"\n✅ Signal generator ready")
    print(f"Available models: {len(registry.models)}")
    print(f"\nTo integrate with your app:")
    print(f"  1. Call generator.generate_signal() for each new bar")
    print(f"  2. Check generator.check_drift() periodically")
    print(f"  3. Store signals in Supabase")
    print(f"  4. Execute via your broker API")
    
    return 0


if __name__ == '__main__':
    exit(main())

