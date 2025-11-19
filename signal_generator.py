"""
═══════════════════════════════════════════════════════════════════════════════
PRODUCTION SIGNAL GENERATOR V2.2
═══════════════════════════════════════════════════════════════════════════════

Loads trained models and generates real-time trading signals for Supabase.

Features:
- Multi-timeframe support (5T, 15T, 30T)
- Confidence filtering (thresholds from training)
- Regime filtering (avoids bad regimes)
- Deduplication (no repeated signals)
- Error handling and logging
- Supabase integration

Usage:
    # Generate signals once
    python signal_generator_production.py --symbol XAUUSD --generate
    
    # Continuous monitoring (checks every 5 minutes)
    python signal_generator_production.py --symbol XAUUSD --monitor --interval 300
    
    # Specific timeframe only
    python signal_generator_production.py --symbol XAUUSD --timeframe 15T --generate
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import argparse
import time
import warnings

warnings.filterwarnings('ignore')

# Supabase
try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    print("⚠️  Install supabase: pip install supabase")
    SUPABASE_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

class SignalConfig:
    """Configuration for signal generation."""
    
    # Paths
    MODELS_DIR = Path("production_models")
    FEATURE_STORE = Path("ML_model/ML_model/feature_store")
    
    # Supabase (set your credentials)
    SUPABASE_URL = "YOUR_SUPABASE_URL"  # Replace with your URL
    SUPABASE_KEY = "YOUR_SUPABASE_KEY"  # Replace with your key
    SIGNALS_TABLE = "trading_signals"   # Your table name
    
    # Timeframes to monitor
    ACTIVE_TIMEFRAMES = ['5T', '15T', '30T']
    
    # Signal validity (how long a signal is "fresh")
    SIGNAL_VALIDITY_MINUTES = {
        '5T': 15,   # 5-min signals valid for 15 minutes
        '15T': 45,  # 15-min signals valid for 45 minutes
        '30T': 90   # 30-min signals valid for 90 minutes
    }
    
    # Lookback for feature calculation (bars)
    LOOKBACK_BARS = 500  # Enough for 200 EMA + buffer


CONFIG = SignalConfig()


# ═══════════════════════════════════════════════════════════════════════════
# MODEL LOADER
# ═══════════════════════════════════════════════════════════════════════════

class ModelLoader:
    """Load and manage production models."""
    
    def __init__(self, models_dir: Path):
        self.models_dir = models_dir
        self.models = {}
        self.metadata = {}
    
    def load_all_models(self, symbol: str) -> Dict:
        """Load all available models for a symbol."""
        print(f"\n{'='*80}")
        print(f"LOADING PRODUCTION MODELS")
        print(f"{'='*80}")
        
        for timeframe in CONFIG.ACTIVE_TIMEFRAMES:
            try:
                self.load_model(symbol, timeframe)
            except Exception as e:
                print(f"⚠️  Failed to load {timeframe}: {e}")
        
        print(f"\n✅ Loaded {len(self.models)} models: {list(self.models.keys())}\n")
        return self.models
    
    def load_model(self, symbol: str, timeframe: str):
        """Load a specific model."""
        # Find model file (could be any model type)
        model_files = list(self.models_dir.glob(f"{symbol}_{timeframe}_*.pkl"))
        
        if not model_files:
            raise FileNotFoundError(f"No model found for {symbol} {timeframe}")
        
        model_path = model_files[0]  # Take first match
        metadata_path = model_path.with_name(model_path.stem + "_meta.json")
        
        # Load model
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        # Load metadata
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {}
        
        key = f"{symbol}_{timeframe}"
        self.models[key] = model_data
        self.metadata[key] = metadata
        
        print(f"✅ Loaded: {timeframe} - {model_data['model_name']} "
              f"(Threshold: {model_data['threshold']:.2f}, "
              f"PF: {metadata.get('performance', {}).get('profit_factor', 'N/A')})")
    
    def get_model(self, symbol: str, timeframe: str) -> Optional[Dict]:
        """Get a loaded model."""
        key = f"{symbol}_{timeframe}"
        return self.models.get(key)


# ═══════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING (SAME AS TRAINING)
# ═══════════════════════════════════════════════════════════════════════════

class FeatureEngineer:
    """Apply same feature engineering as training."""
    
    @staticmethod
    def add_regime_features(df: pd.DataFrame) -> pd.DataFrame:
        """Regime features."""
        features = df.copy()
        
        if 'atr' in features.columns:
            features['regime_vol_percentile'] = features['atr'].rolling(100).apply(
                lambda x: (x.iloc[-1] > x).sum() / len(x) if len(x) > 0 else 0.5
            )
            features['regime_vol'] = pd.cut(
                features['regime_vol_percentile'],
                bins=[0, 0.33, 0.67, 1.0],
                labels=[0, 1, 2]
            ).astype(float)
        
        if 'ema_20' in features.columns and 'ema_50' in features.columns:
            features['regime_trend_20_50'] = ((features['ema_20'] > features['ema_50']).astype(int) * 2 - 1)
        
        if 'ema_50' in features.columns and 'ema_200' in features.columns:
            features['regime_trend_50_200'] = ((features['ema_50'] > features['ema_200']).astype(int) * 2 - 1)
        
        if 'hour' in features.columns:
            features['regime_session_asian'] = ((features['hour'] >= 0) & (features['hour'] < 8)).astype(int)
            features['regime_session_london'] = ((features['hour'] >= 8) & (features['hour'] < 16)).astype(int)
            features['regime_session_ny'] = ((features['hour'] >= 13) & (features['hour'] < 21)).astype(int)
        
        if 'close' in features.columns:
            high_20 = features['high'].rolling(20).max()
            low_20 = features['low'].rolling(20).min()
            range_20 = high_20 - low_20
            features['regime_range_position'] = (features['close'] - low_20) / (range_20 + 1e-8)
        
        return features
    
    @staticmethod
    def add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
        """Momentum features."""
        features = df.copy()
        
        for period in [5, 10, 20]:
            features[f'momentum_roc_{period}'] = features['close'].pct_change(period)
        
        if 'atr' in features.columns:
            features['momentum_strength_5'] = features['close'].diff(5) / (features['atr'] + 1e-8)
            features['momentum_strength_10'] = features['close'].diff(10) / (features['atr'] + 1e-8)
        
        if 'momentum_roc_5' in features.columns and 'momentum_roc_10' in features.columns:
            features['momentum_accel'] = features['momentum_roc_5'] - features['momentum_roc_10']
        
        return features
    
    @staticmethod
    def add_mean_reversion_features(df: pd.DataFrame) -> pd.DataFrame:
        """Mean reversion features."""
        features = df.copy()
        
        for ma in [20, 50, 100]:
            if f'sma_{ma}' in features.columns:
                features[f'mr_distance_sma_{ma}'] = (
                    (features['close'] - features[f'sma_{ma}']) / (features[f'sma_{ma}'] + 1e-8)
                )
        
        if 'bb_upper' in features.columns and 'bb_lower' in features.columns:
            features['mr_bb_position'] = (
                (features['close'] - features['bb_lower']) / 
                (features['bb_upper'] - features['bb_lower'] + 1e-8)
            )
            features['mr_bb_extreme'] = (
                (features['mr_bb_position'] > 0.95) | (features['mr_bb_position'] < 0.05)
            ).astype(int)
        
        if 'rsi' in features.columns:
            features['mr_rsi_oversold'] = (features['rsi'] < 30).astype(int)
            features['mr_rsi_overbought'] = (features['rsi'] > 70).astype(int)
            features['mr_rsi_neutral'] = ((features['rsi'] >= 40) & (features['rsi'] <= 60)).astype(int)
        
        return features
    
    @staticmethod
    def add_microstructure_features(df: pd.DataFrame) -> pd.DataFrame:
        """Microstructure features."""
        features = df.copy()
        
        features['micro_body'] = abs(features['close'] - features['open'])
        features['micro_upper_wick'] = features['high'] - np.maximum(features['open'], features['close'])
        features['micro_lower_wick'] = np.minimum(features['open'], features['close']) - features['low']
        features['micro_total_range'] = features['high'] - features['low']
        
        features['micro_body_ratio'] = features['micro_body'] / (features['micro_total_range'] + 1e-8)
        features['micro_wick_ratio'] = (
            (features['micro_upper_wick'] + features['micro_lower_wick']) / 
            (features['micro_total_range'] + 1e-8)
        )
        
        if 'volume' in features.columns:
            vol_ma = features['volume'].rolling(20).mean()
            features['micro_volume_surge'] = features['volume'] / (vol_ma + 1)
            features['micro_volume_anomaly'] = (features['micro_volume_surge'] > 2.0).astype(int)
        
        features['micro_gap'] = features['open'] - features['close'].shift(1)
        features['micro_gap_pct'] = features['micro_gap'] / (features['close'].shift(1) + 1e-8)
        
        return features
    
    @staticmethod
    def engineer_all_features(df: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature engineering."""
        df = FeatureEngineer.add_regime_features(df)
        df = FeatureEngineer.add_momentum_features(df)
        df = FeatureEngineer.add_mean_reversion_features(df)
        df = FeatureEngineer.add_microstructure_features(df)
        return df


# ═══════════════════════════════════════════════════════════════════════════
# SIGNAL GENERATOR
# ═══════════════════════════════════════════════════════════════════════════

class SignalGenerator:
    """Generate trading signals from models."""
    
    def __init__(self, model_loader: ModelLoader):
        self.model_loader = model_loader
        self.last_signals = {}  # Deduplication
    
    def load_latest_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Load latest data from feature store."""
        file_path = CONFIG.FEATURE_STORE / symbol / f"{symbol}_{timeframe}.parquet"
        
        if not file_path.exists():
            raise FileNotFoundError(f"Feature store not found: {file_path}")
        
        # Load data
        df = pd.read_parquet(file_path)
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Take only recent bars (for performance)
        df = df.tail(CONFIG.LOOKBACK_BARS).copy()
        
        return df
    
    def generate_signal(self, symbol: str, timeframe: str) -> Optional[Dict]:
        """Generate signal for a specific symbol/timeframe."""
        
        try:
            # Load model
            model_data = self.model_loader.get_model(symbol, timeframe)
            if model_data is None:
                print(f"⚠️  No model for {symbol} {timeframe}")
                return None
            
            # Load latest data
            df = self.load_latest_data(symbol, timeframe)
            
            # Engineer features
            df = FeatureEngineer.engineer_all_features(df)
            df = df.dropna()
            
            if len(df) == 0:
                print(f"⚠️  No valid data after feature engineering")
                return None
            
            # Get latest bar
            latest_idx = df.index[-1]
            latest_row = df.iloc[-1]
            
            # Extract features (same order as training)
            feature_cols = model_data['feature_cols']
            missing = [c for c in feature_cols if c not in df.columns]
            if missing:
                print(f"⚠️  Missing features: {missing[:5]}...")
                return None
            
            X = df.loc[[latest_idx], feature_cols].values
            
            # Make prediction
            model = model_data['model']
            scaler = model_data['scaler']
            
            X_scaled = scaler.transform(X)
            confidence = model.predict_proba(X_scaled)[0, 1]
            
            # Apply confidence threshold
            threshold = model_data['threshold']
            signal_direction = 'BUY' if confidence >= threshold else 'NEUTRAL'
            
            # Check regime filter
            bad_regimes = model_data.get('bad_regimes', [])
            current_regime = self._get_regime(latest_row)
            
            if current_regime in bad_regimes:
                print(f"🚫 {timeframe}: Bad regime detected ({current_regime}) - No signal")
                return None
            
            # Only return signal if it's actionable (BUY)
            if signal_direction == 'NEUTRAL':
                return None
            
            # Build signal
            signal = {
                'symbol': symbol,
                'timeframe': timeframe,
                'direction': signal_direction,
                'confidence': float(confidence),
                'threshold': float(threshold),
                'entry_price': float(latest_row['close']),
                'timestamp': latest_row['timestamp'].isoformat() if hasattr(latest_row['timestamp'], 'isoformat') else str(latest_row['timestamp']),
                'model_name': model_data['model_name'],
                'tp_multiplier': float(model_data['tp_mult']),
                'sl_multiplier': float(model_data['sl_mult']),
                'atr': float(latest_row['atr']),
                'regime': current_regime,
                'metadata': {
                    'profit_factor': model_data.get('performance', {}).get('profit_factor'),
                    'win_rate': model_data.get('performance', {}).get('win_rate'),
                    'max_drawdown': model_data.get('performance', {}).get('max_drawdown_pct')
                }
            }
            
            # Calculate TP/SL prices
            atr = latest_row['atr']
            signal['tp_price'] = float(signal['entry_price'] + (signal['tp_multiplier'] * atr))
            signal['sl_price'] = float(signal['entry_price'] - (signal['sl_multiplier'] * atr))
            
            # Add signal ID for deduplication
            signal['signal_id'] = f"{symbol}_{timeframe}_{signal['timestamp']}"
            
            return signal
            
        except Exception as e:
            print(f"❌ Error generating signal for {symbol} {timeframe}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _get_regime(self, row: pd.Series) -> str:
        """Extract regime from row."""
        if 'regime_vol' in row.index:
            regime_map = {0: 'Low Vol', 1: 'Med Vol', 2: 'High Vol'}
            return regime_map.get(row['regime_vol'], 'Unknown')
        return 'Unknown'
    
    def is_duplicate(self, signal: Dict) -> bool:
        """Check if signal is duplicate of recent signal."""
        key = f"{signal['symbol']}_{signal['timeframe']}"
        
        if key in self.last_signals:
            last_signal = self.last_signals[key]
            
            # Check if within validity window
            last_time = datetime.fromisoformat(last_signal['timestamp'])
            current_time = datetime.fromisoformat(signal['timestamp'])
            validity = CONFIG.SIGNAL_VALIDITY_MINUTES.get(signal['timeframe'], 30)
            
            if (current_time - last_time).total_seconds() < validity * 60:
                return True  # Too recent
        
        # Update last signal
        self.last_signals[key] = signal
        return False


# ═══════════════════════════════════════════════════════════════════════════
# SUPABASE INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════

class SupabaseClient:
    """Send signals to Supabase."""
    
    def __init__(self):
        if not SUPABASE_AVAILABLE:
            raise ImportError("Supabase not installed")
        
        self.client: Client = create_client(
            CONFIG.SUPABASE_URL,
            CONFIG.SUPABASE_KEY
        )
    
    def send_signal(self, signal: Dict) -> bool:
        """Send signal to Supabase."""
        try:
            # Format signal for Supabase
            data = {
                'signal_id': signal['signal_id'],
                'symbol': signal['symbol'],
                'timeframe': signal['timeframe'],
                'direction': signal['direction'],
                'confidence': signal['confidence'],
                'entry_price': signal['entry_price'],
                'tp_price': signal['tp_price'],
                'sl_price': signal['sl_price'],
                'timestamp': signal['timestamp'],
                'model_name': signal['model_name'],
                'atr': signal['atr'],
                'regime': signal['regime'],
                'metadata': signal['metadata'],
                'status': 'ACTIVE',
                'created_at': datetime.utcnow().isoformat()
            }
            
            # Insert into Supabase
            response = self.client.table(CONFIG.SIGNALS_TABLE).insert(data).execute()
            
            print(f"✅ Signal sent to Supabase: {signal['signal_id']}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to send signal to Supabase: {e}")
            return False
    
    def get_active_signals(self, symbol: str = None) -> List[Dict]:
        """Get active signals from Supabase."""
        try:
            query = self.client.table(CONFIG.SIGNALS_TABLE).select("*").eq("status", "ACTIVE")
            
            if symbol:
                query = query.eq("symbol", symbol)
            
            response = query.execute()
            return response.data
            
        except Exception as e:
            print(f"❌ Failed to fetch signals: {e}")
            return []


# ═══════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ═══════════════════════════════════════════════════════════════════════════

class SignalGeneratorApp:
    """Main application."""
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.model_loader = ModelLoader(CONFIG.MODELS_DIR)
        self.signal_generator = SignalGenerator(self.model_loader)
        
        # Initialize Supabase if available
        self.supabase = None
        if SUPABASE_AVAILABLE and CONFIG.SUPABASE_URL != "YOUR_SUPABASE_URL":
            self.supabase = SupabaseClient()
    
    def initialize(self):
        """Load models."""
        print(f"\n{'#'*80}")
        print(f"# PRODUCTION SIGNAL GENERATOR V2.2")
        print(f"# Symbol: {self.symbol}")
        print(f"# Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'#'*80}")
        
        self.model_loader.load_all_models(self.symbol)
    
    def generate_signals(self, timeframe: Optional[str] = None):
        """Generate signals for all or specific timeframe."""
        print(f"\n{'='*80}")
        print(f"GENERATING SIGNALS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}\n")
        
        timeframes = [timeframe] if timeframe else CONFIG.ACTIVE_TIMEFRAMES
        signals_generated = []
        
        for tf in timeframes:
            print(f"🔍 Checking {self.symbol} {tf}...")
            
            signal = self.signal_generator.generate_signal(self.symbol, tf)
            
            if signal is None:
                print(f"   No signal\n")
                continue
            
            # Check duplicate
            if self.signal_generator.is_duplicate(signal):
                print(f"   ⚠️  Duplicate signal (too recent)\n")
                continue
            
            # Print signal
            print(f"   🎯 SIGNAL GENERATED!")
            print(f"   Direction: {signal['direction']}")
            print(f"   Confidence: {signal['confidence']:.1%} (threshold: {signal['threshold']:.1%})")
            print(f"   Entry: ${signal['entry_price']:.2f}")
            print(f"   TP: ${signal['tp_price']:.2f} (+{signal['tp_multiplier']:.1f}R)")
            print(f"   SL: ${signal['sl_price']:.2f} (-{signal['sl_multiplier']:.1f}R)")
            print(f"   Regime: {signal['regime']}\n")
            
            signals_generated.append(signal)
            
            # Send to Supabase
            if self.supabase:
                self.supabase.send_signal(signal)
            else:
                print(f"   ⚠️  Supabase not configured - signal not sent\n")
        
        print(f"\n{'='*80}")
        print(f"SUMMARY: {len(signals_generated)} signals generated")
        print(f"{'='*80}\n")
        
        return signals_generated
    
    def monitor(self, interval_seconds: int = 300):
        """Continuously monitor and generate signals."""
        print(f"\n🔄 MONITORING MODE")
        print(f"   Checking every {interval_seconds} seconds ({interval_seconds/60:.0f} minutes)")
        print(f"   Press Ctrl+C to stop\n")
        
        try:
            while True:
                self.generate_signals()
                print(f"😴 Sleeping for {interval_seconds} seconds...")
                print(f"   Next check at: {(datetime.now() + timedelta(seconds=interval_seconds)).strftime('%H:%M:%S')}\n")
                time.sleep(interval_seconds)
                
        except KeyboardInterrupt:
            print(f"\n\n⏹️  Monitoring stopped by user")


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Production Signal Generator V2.2',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--symbol', type=str, default='XAUUSD', help='Trading symbol')
    parser.add_argument('--timeframe', type=str, help='Specific timeframe (5T, 15T, 30T)')
    parser.add_argument('--generate', action='store_true', help='Generate signals once')
    parser.add_argument('--monitor', action='store_true', help='Continuous monitoring')
    parser.add_argument('--interval', type=int, default=300, help='Monitor interval (seconds)')
    
    args = parser.parse_args()
    
    if not args.generate and not args.monitor:
        parser.print_help()
        return
    
    # Initialize app
    app = SignalGeneratorApp(args.symbol)
    app.initialize()
    
    # Generate or monitor
    if args.monitor:
        app.monitor(args.interval)
    else:
        app.generate_signals(args.timeframe)


if __name__ == '__main__':
    main()
