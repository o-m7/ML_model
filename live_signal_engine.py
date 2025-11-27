"""
Live Signal Generation Engine
=============================

Real-time signal generation from Polygon.io using ONNX models.
- No look-ahead bias
- Feature computation on closed bars only
- Multiple timeframes (1T, 5T, 15T, 30T)
- Supabase integration for frontend
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from collections import defaultdict
import logging
from pathlib import Path

import onnx
import onnxruntime as ort
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# 1. DATA STRUCTURES
# ============================================================================

@dataclass
class AggBar:
    """Normalized OHLCV bar from Polygon."""
    symbol: str
    o: float
    h: float
    l: float
    c: float
    v: float
    start_ts: pd.Timestamp
    end_ts: pd.Timestamp
    
    def to_dict(self):
        return asdict(self)


@dataclass
class Quote:
    """Normalized quote (bid/ask) from Polygon."""
    symbol: str
    bid: float
    ask: float
    ts: pd.Timestamp
    
    def to_dict(self):
        return asdict(self)


@dataclass
class Signal:
    """Signal output (no trade execution)."""
    symbol: str
    timeframe: str
    timestamp: pd.Timestamp
    signal: int  # -1 (short), 0 (neutral), 1 (long)
    confidence: float  # 0.0-1.0
    model_output: Optional[Dict] = None
    features: Optional[Dict] = None
    
    def to_dict(self):
        return {
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'timestamp': self.timestamp.isoformat(),
            'signal': self.signal,
            'confidence': self.confidence,
            'model_output': self.model_output,
            'features': self.features,
        }


# ============================================================================
# 2. CONFIGURATION
# ============================================================================

class Config:
    """Signal generation configuration."""
    
    # Symbol & market
    SYMBOL = "XAUUSD"  # Polygon ticker for gold
    
    # Timeframes
    TIMEFRAMES = {
        "1T": 60,
        "5T": 300,
        "15T": 900,
        "30T": 1800,
    }
    
    # Feature windows
    RSI_WINDOW = 14
    ATR_WINDOW = 14
    MA_FAST = 10
    MA_SLOW = 50
    VOL_WINDOW = 20
    
    # ONNX models
    ONNX_MODELS_DIR = "artifacts/onnx_models"
    
    # API
    POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "")
    POLYGON_WS_URL = "wss://delayed.polygon.io/forex"
    
    # Supabase
    SUPABASE_URL = os.getenv("SUPABASE_URL", "")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")
    SUPABASE_TABLE = "signals"
    
    # Validation
    MIN_BARS_FOR_FEATURES = 50  # Need at least this many bars before generating signals
    SIGNAL_CONFIDENCE_THRESHOLD = 0.5  # Only emit signals above this confidence
    
    @classmethod
    def validate(cls):
        """Validate required config."""
        if not cls.POLYGON_API_KEY:
            raise ValueError("POLYGON_API_KEY not in environment")
        if not cls.SUPABASE_URL or not cls.SUPABASE_KEY:
            raise ValueError("SUPABASE_URL/KEY not in environment")
        logger.info("✓ Configuration validated")


# ============================================================================
# 3. FEATURE ENGINE
# ============================================================================

class FeatureEngine:
    """Computes trading features from market data (no look-ahead)."""
    
    def __init__(self, symbol: str):
        """
        Args:
            symbol: Trading symbol (e.g., 'XAUUSD')
        """
        self.symbol = symbol
        
        # History per timeframe: timeframe -> pd.DataFrame(o, h, l, c, v, ts)
        self.bar_history: Dict[str, pd.DataFrame] = {
            tf: pd.DataFrame(columns=['o', 'h', 'l', 'c', 'v', 'end_ts'])
            for tf in Config.TIMEFRAMES.keys()
        }
        
        # Quote history for spread-based features
        self.quote_history: pd.DataFrame = pd.DataFrame(
            columns=['bid', 'ask', 'ts']
        )
    
    def add_bar(self, timeframe: str, bar: AggBar) -> None:
        """Add a closed bar to history."""
        df = self.bar_history[timeframe]
        new_row = pd.DataFrame({
            'o': [bar.o],
            'h': [bar.h],
            'l': [bar.l],
            'c': [bar.c],
            'v': [bar.v],
            'end_ts': [bar.end_ts],
        })
        self.bar_history[timeframe] = pd.concat([df, new_row], ignore_index=True)
    
    def add_quote(self, quote: Quote) -> None:
        """Add a quote to history (for spread features)."""
        new_row = pd.DataFrame({
            'bid': [quote.bid],
            'ask': [quote.ask],
            'ts': [quote.ts],
        })
        self.quote_history = pd.concat([self.quote_history, new_row], ignore_index=True)
        
        # Keep only last 1000 quotes to avoid memory bloat
        if len(self.quote_history) > 1000:
            self.quote_history = self.quote_history.iloc[-1000:].reset_index(drop=True)
    
    def compute_features(self, timeframe: str) -> Optional[pd.Series]:
        """
        Compute features for current closed bar.
        Returns None if not enough data.
        """
        df = self.bar_history[timeframe].copy()
        
        if len(df) < Config.MIN_BARS_FOR_FEATURES:
            logger.debug(f"Not enough bars for {timeframe}: {len(df)} < {Config.MIN_BARS_FOR_FEATURES}")
            return None
        
        # ===== PRICE & RETURNS =====
        # Safe log returns: handle division by zero and NaN
        close_ratio = df['c'] / df['c'].shift(1)
        close_ratio = close_ratio.replace([np.inf, -np.inf], np.nan)
        df['ret'] = np.log(close_ratio)
        df['ret'] = df['ret'].fillna(0.0)  # First row will be NaN
        
        df['ret_mean_20'] = df['ret'].rolling(20).mean()
        df['ret_std_20'] = df['ret'].rolling(20).std()
        df['ret_zscore_20'] = (df['ret'] - df['ret_mean_20']) / (df['ret_std_20'] + 1e-9)
        df['ret_zscore_20'] = df['ret_zscore_20'].fillna(0.0)
        
        # ===== VOLATILITY & ATR =====
        prev_close = df['c'].shift(1)
        tr1 = df['h'] - df['l']
        tr2 = (df['h'] - prev_close).abs()
        tr3 = (df['l'] - prev_close).abs()
        df['tr'] = np.maximum(tr1, np.maximum(tr2, tr3))
        
        df['atr'] = df['tr'].rolling(Config.ATR_WINDOW).mean()
        df['atr_pct'] = df['atr'] / df['c']
        df['atr_pct'] = df['atr_pct'].fillna(0.0)
        df['vol_20'] = df['ret'].rolling(20).std()
        df['vol_20'] = df['vol_20'].fillna(0.0)
        
        # ===== MOMENTUM & RSI =====
        delta = df['c'].diff().fillna(0.0)
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        
        avg_gain = gain.rolling(Config.RSI_WINDOW).mean()
        avg_loss = loss.rolling(Config.RSI_WINDOW).mean()
        
        rs = avg_gain / (avg_loss + 1e-9)
        df['rsi_14'] = 100 - (100 / (1 + rs))
        df['rsi_14'] = df['rsi_14'].fillna(50.0)  # Neutral RSI for warmup
        
        # ===== MOVING AVERAGES =====
        df['ma_fast_10'] = df['c'].rolling(Config.MA_FAST).mean()
        df['ma_slow_50'] = df['c'].rolling(Config.MA_SLOW).mean()
        df['ma_diff'] = df['ma_fast_10'] - df['ma_slow_50']
        df['ma_ratio'] = df['ma_fast_10'] / (df['ma_slow_50'] + 1e-9)
        df['ma_ratio'] = df['ma_ratio'].fillna(1.0)
        df['trend_up'] = (df['ma_diff'] > 0).astype(int)
        df['trend_down'] = (df['ma_diff'] < 0).astype(int)
        df['ma_diff'] = df['ma_diff'].fillna(0.0)
        
        # ===== VOLUME FEATURES =====
        df['vol_mean_20'] = df['v'].rolling(20).mean()
        df['vol_std_20'] = df['v'].rolling(20).std()
        df['vol_zscore_20'] = (df['v'] - df['vol_mean_20']) / (df['vol_std_20'] + 1e-9)
        df['vol_spike'] = (df['vol_zscore_20'] > 2.0).astype(int)
        df['vol_zscore_20'] = df['vol_zscore_20'].fillna(0.0)
        df['vol_ratio'] = df['v'] / (df['vol_mean_20'] + 1e-9)
        df['vol_ratio'] = df['vol_ratio'].fillna(1.0)
        
        # ===== QUOTE-BASED FEATURES (if available) =====
        df['spread_last'] = 0.0
        df['spread_pct_last'] = 0.0
        df['mid_last'] = df['c']  # Fallback to close price
        
        if len(self.quote_history) > 0:
            latest_quote = self.quote_history.iloc[-1]
            df.loc[df.index[-1], 'spread_last'] = latest_quote['ask'] - latest_quote['bid']
            mid = (latest_quote['bid'] + latest_quote['ask']) / 2
            df.loc[df.index[-1], 'mid_last'] = mid
            df.loc[df.index[-1], 'spread_pct_last'] = \
                df.loc[df.index[-1], 'spread_last'] / (mid + 1e-9)
        
        # ===== EXTRACT CURRENT ROW =====
        feature_cols = [
            'c', 'ret', 'vol_20', 'ret_zscore_20',
            'atr', 'atr_pct',
            'rsi_14',
            'ma_diff', 'ma_ratio', 'trend_up', 'trend_down',
            'v', 'vol_zscore_20', 'vol_spike', 'vol_ratio',
            'spread_last', 'spread_pct_last', 'mid_last',
        ]
        
        current_features = df.iloc[-1][feature_cols].fillna(0).astype(np.float64)
        
        # Store for debugging
        self.last_features_df = df.copy()
        
        return current_features
    
    def get_bar_count(self, timeframe: str) -> int:
        """Get number of bars in history for timeframe."""
        return len(self.bar_history[timeframe])


# ============================================================================
# 4. BAR BUILDER
# ============================================================================

class BarBuilder:
    """Builds aggregated OHLCV bars from minute aggregates."""
    
    def __init__(self, symbol: str):
        """
        Args:
            symbol: Trading symbol
        """
        self.symbol = symbol
        self.timeframes = Config.TIMEFRAMES  # e.g., {"1T": 60, "5T": 300, ...}
        
        # Track current forming bar per timeframe
        # timeframe -> {'start_ts': ..., 'o': ..., 'h': ..., 'l': ..., 'c': ..., 'v': ...}
        self.forming_bars: Dict[str, Optional[Dict]] = {tf: None for tf in self.timeframes}
        
        # Last closed bar timestamp per timeframe
        self.last_closed_ts: Dict[str, Optional[pd.Timestamp]] = {tf: None for tf in self.timeframes}
    
    def process_minute_agg(self, agg: AggBar) -> Dict[str, Optional[AggBar]]:
        """
        Process a minute aggregate and return closed bars per timeframe.
        
        Args:
            agg: Minute aggregate from Polygon
            
        Returns:
            Dict[timeframe] -> closed bar or None
        """
        closed_bars = {}
        
        for timeframe, duration_sec in self.timeframes.items():
            # Determine which bar interval this minute belongs to
            bar_start_ts = self._get_bar_start(agg.end_ts, duration_sec)
            bar_end_ts = bar_start_ts + timedelta(seconds=duration_sec)
            
            # If no forming bar, start one
            if self.forming_bars[timeframe] is None:
                self.forming_bars[timeframe] = {
                    'start_ts': bar_start_ts,
                    'end_ts': bar_end_ts,
                    'o': agg.o,
                    'h': agg.h,
                    'l': agg.l,
                    'c': agg.c,
                    'v': agg.v,
                }
                closed_bars[timeframe] = None
            else:
                forming = self.forming_bars[timeframe]
                
                # If minute agg is for a new bar interval, emit the current one
                if agg.end_ts >= forming['end_ts']:
                    # Close and emit current bar
                    closed_bar = AggBar(
                        symbol=self.symbol,
                        o=forming['o'],
                        h=forming['h'],
                        l=forming['l'],
                        c=forming['c'],
                        v=forming['v'],
                        start_ts=forming['start_ts'],
                        end_ts=forming['end_ts'],
                    )
                    closed_bars[timeframe] = closed_bar
                    self.last_closed_ts[timeframe] = forming['end_ts']
                    
                    # Start new bar
                    self.forming_bars[timeframe] = {
                        'start_ts': bar_start_ts,
                        'end_ts': bar_end_ts,
                        'o': agg.o,
                        'h': agg.h,
                        'l': agg.l,
                        'c': agg.c,
                        'v': agg.v,
                    }
                else:
                    # Update forming bar
                    forming['h'] = max(forming['h'], agg.h)
                    forming['l'] = min(forming['l'], agg.l)
                    forming['c'] = agg.c
                    forming['v'] += agg.v
                    closed_bars[timeframe] = None
        
        return closed_bars
    
    @staticmethod
    def _get_bar_start(ts: pd.Timestamp, duration_sec: int) -> pd.Timestamp:
        """Get the start timestamp of the bar containing ts."""
        epoch = pd.Timestamp('1970-01-01', tz='UTC')
        seconds_since_epoch = int((ts - epoch).total_seconds())
        bar_index = seconds_since_epoch // duration_sec
        bar_start_sec = bar_index * duration_sec
        return epoch + timedelta(seconds=bar_start_sec)


# ============================================================================
# 5. MODEL INTERFACE
# ============================================================================

class ONNXModelInterface:
    """Wraps XGBoost models for signal generation (native inference, no ONNX)."""
    
    def __init__(self, models_dir: str = "artifacts"):
        """
        Args:
            models_dir: Directory containing XGBoost model files (.pkl)
        """
        self.models_dir = models_dir
        self.models: Dict[Tuple[str, str], Any] = {}
        self._load_models()
    
    def _load_models(self) -> None:
        """Load all XGBoost models from directory."""
        models_path = Path(self.models_dir)
        if not models_path.exists():
            logger.warning(f"Models directory not found: {self.models_dir}")
            return
        
        # Load XGBoost models (native, not ONNX)
        for model_file in models_path.glob("*_xgb.pkl"):
            try:
                # Parse filename: ohlcv_model_1T_xgb.pkl or quote_model_5T_xgb.pkl
                parts = model_file.stem.split('_')
                if len(parts) >= 3:
                    model_type = parts[0]  # "ohlcv" or "quote"
                    timeframe = parts[-2]  # "1T", "5T", etc.
                    
                    import joblib
                    model = joblib.load(str(model_file))
                    self.models[(timeframe, model_type)] = model
                    logger.info(f"✓ Loaded {model_file.name}")
            except Exception as e:
                logger.error(f"Failed to load {model_file.name}: {e}")
    
    def predict(
        self,
        timeframe: str,
        model_type: str,
        features: np.ndarray
    ) -> Optional[Dict]:
        """
        Generate signal from features using XGBoost model (native inference).
        
        Args:
            timeframe: "1T", "5T", etc.
            model_type: "ohlcv" or "quote"
            features: Feature vector (1D numpy array)
            
        Returns:
            Dict with signal info or None if model not available
        """
        key = (timeframe, model_type)
        if key not in self.models:
            return None
        
        try:
            model = self.models[key]
            
            # Reshape features for model input
            features_input = features.values.reshape(1, -1) if hasattr(features, 'values') else features.reshape(1, -1)
            features_input = features_input.astype(np.float32)
            
            # Run inference using native XGBoost
            # predict_proba returns shape (n_samples, n_classes)
            probs = model.predict_proba(features_input)[0]
            
            # Get predicted class
            predicted_class = np.argmax(probs)
            confidence = float(probs[predicted_class])
            
            # Map to signal: 0=short, 1=neutral, 2=long (adjust as needed)
            signal_map = {0: -1, 1: 0, 2: 1}
            signal = signal_map.get(predicted_class, 0)
            
            return {
                'signal': signal,
                'confidence': confidence,
                'probabilities': probs.tolist(),
            }
        except Exception as e:
            logger.error(f"Model prediction error ({timeframe}, {model_type}): {e}")
            return None


# ============================================================================
# 6. SIGNAL GENERATOR
# ============================================================================

class SignalGenerator:
    """Orchestrates signal generation from market data to output."""
    
    def __init__(self):
        """Initialize signal generator."""
        Config.validate()
        
        self.symbol = Config.SYMBOL
        self.bar_builder = BarBuilder(self.symbol)
        self.feature_engine = FeatureEngine(self.symbol)
        self.model_interface = ONNXModelInterface(Config.ONNX_MODELS_DIR)
        
        # Signal history
        self.signals_history: List[Signal] = []
        
        logger.info("✓ Signal generator initialized")
    
    def process_polygon_agg(self, agg: AggBar) -> List[Signal]:
        """
        Process a Polygon minute aggregate and generate signals for closed bars.
        
        Args:
            agg: Minute aggregate from Polygon
            
        Returns:
            List of signals for newly closed bars
        """
        signals = []
        
        # Build bars
        closed_bars = self.bar_builder.process_minute_agg(agg)
        
        # Process each closed bar
        for timeframe, bar in closed_bars.items():
            if bar is None:
                continue
            
            # Add to feature engine
            self.feature_engine.add_bar(timeframe, bar)
            
            # Compute features
            features = self.feature_engine.compute_features(timeframe)
            if features is None:
                logger.debug(f"Skipping signal: not enough bars for {timeframe}")
                continue
            
            # Generate signal from models
            signal = self._generate_signal(timeframe, bar.end_ts, features)
            if signal is not None:
                signals.append(signal)
                self.signals_history.append(signal)
                logger.info(f"✓ Signal: {signal.symbol} {signal.timeframe} {signal.signal:+d} conf={signal.confidence:.2f}")
        
        return signals
    
    def process_quote(self, quote: Quote) -> None:
        """Process a quote for spread-based features."""
        self.feature_engine.add_quote(quote)
    
    def _generate_signal(
        self,
        timeframe: str,
        bar_end_ts: pd.Timestamp,
        features: pd.Series
    ) -> Optional[Signal]:
        """
        Generate signal from features using ensemble of models.
        
        Args:
            timeframe: "1T", "5T", etc.
            bar_end_ts: Bar close timestamp
            features: Feature vector
            
        Returns:
            Signal object or None if confidence too low
        """
        features_np = features.values.astype(np.float32)
        
        # Query both model types
        ohlcv_pred = self.model_interface.predict(timeframe, 'ohlcv', features_np)
        quote_pred = self.model_interface.predict(timeframe, 'quote', features_np)
        
        # Ensemble voting
        signals = []
        weights = []
        
        if ohlcv_pred is not None:
            signals.append(ohlcv_pred['signal'])
            weights.append(ohlcv_pred['confidence'])
        
        if quote_pred is not None:
            signals.append(quote_pred['signal'])
            weights.append(quote_pred['confidence'])
        
        if not signals:
            logger.warning(f"No models available for {timeframe}")
            return None
        
        # Weighted ensemble
        avg_signal = np.average(signals, weights=weights)
        avg_confidence = np.mean(weights)
        
        # Thresholding
        if avg_confidence < Config.SIGNAL_CONFIDENCE_THRESHOLD:
            logger.debug(f"Signal below confidence threshold: {avg_confidence:.2f}")
            return None
        
        # Convert to discrete signal
        final_signal = 1 if avg_signal > 0.3 else (-1 if avg_signal < -0.3 else 0)
        
        signal = Signal(
            symbol=self.symbol,
            timeframe=timeframe,
            timestamp=bar_end_ts,
            signal=final_signal,
            confidence=avg_confidence,
            model_output={
                'ohlcv': ohlcv_pred,
                'quote': quote_pred,
                'ensemble_avg': float(avg_signal),
            },
            features=features.to_dict(),
        )
        
        return signal
    
    def get_latest_signal(self, timeframe: str) -> Optional[Signal]:
        """Get latest signal for a timeframe."""
        for signal in reversed(self.signals_history):
            if signal.timeframe == timeframe:
                return signal
        return None
    
    def get_signals_json(self) -> str:
        """Export signals as JSON."""
        return json.dumps([s.to_dict() for s in self.signals_history], default=str, indent=2)


if __name__ == "__main__":
    logger.info("Live Signal Engine - Ready for Polygon.io integration")
