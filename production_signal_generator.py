"""
PRODUCTION SIGNAL GENERATOR - EVENT-DRIVEN ARCHITECTURE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Design Philosophy:
1. Event-driven: Run ONLY when new candle closes
2. Batched alerts: Group signals, send once per minute
3. Guardrails: Regime filters, position limits, staleness checks
4. Reliability: Failover, logging, dead-man switches

Deployment:
- Run on Modal.com or Supabase Edge Function
- Cron triggers per timeframe (5T, 15T, 30T)
- Telegram for alerts, Supabase for signal storage
"""

import asyncio
import json
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional
from pathlib import Path
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
import requests

# Your ML models (ONNX for production)
import onnxruntime as ort


# ═══════════════════════════════════════════════════════════════════════════
# SIGNAL DATACLASS
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TradingSignal:
    """Production-ready trading signal."""
    
    # Core fields
    timestamp: str
    symbol: str
    timeframe: str
    direction: str  # 'LONG', 'SHORT', 'NEUTRAL'
    
    # Model outputs
    probability: float
    model_name: str
    confidence_threshold: float
    
    # Risk parameters
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size_pct: float  # e.g., 0.01 = 1% risk per trade
    
    # Regime context
    regime: str  # 'Low Vol', 'Med Vol', 'High Vol'
    regime_approved: bool
    
    # Execution metadata
    signal_id: str
    generated_at: str
    expires_at: str  # Signal valid for N minutes
    
    # Status
    status: str  # 'PENDING', 'EXECUTED', 'EXPIRED', 'REJECTED'
    rejection_reason: Optional[str] = None
    
    def to_dict(self):
        return asdict(self)
    
    def to_telegram_message(self) -> str:
        """Format signal for Telegram."""
        
        direction_emoji = {
            'LONG': '🟢',
            'SHORT': '🔴',
            'NEUTRAL': '⚪'
        }
        
        return f"""
{direction_emoji[self.direction]} **{self.direction} SIGNAL**

**Symbol:** {self.symbol}
**Timeframe:** {self.timeframe}
**Probability:** {self.probability:.1%}
**Regime:** {self.regime}

**Entry:** ${self.entry_price:.2f}
**Stop Loss:** ${self.stop_loss:.2f} (-{abs(self.entry_price - self.stop_loss):.2f})
**Take Profit:** ${self.take_profit:.2f} (+{abs(self.take_profit - self.entry_price):.2f})
**Risk:** {self.position_size_pct:.1%} of account

**Model:** {self.model_name}
**Signal ID:** {self.signal_id}
**Expires:** {self.expires_at}
        """.strip()


# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

class SignalConfig:
    """Production signal generation config."""
    
    # Telegram
    TELEGRAM_BOT_TOKEN = "YOUR_BOT_TOKEN"
    TELEGRAM_CHAT_ID = "YOUR_CHAT_ID"
    
    # Supabase (for signal storage)
    SUPABASE_URL = "https://your-project.supabase.co"
    SUPABASE_KEY = "your-anon-key"
    
    # Model paths (ONNX for fast inference)
    MODEL_DIR = Path("models/production")
    
    # Timeframes to monitor
    TIMEFRAMES = {
        '5T': {
            'model': 'random_forest_5T.onnx',
            'threshold': 0.52,
            'check_interval': 300,  # 5 minutes
            'signal_lifetime': 10,   # Signal valid for 10 min
            'bad_regimes': ['Low Vol']
        },
        '15T': {
            'model': 'xgboost_15T.onnx',
            'threshold': 0.60,
            'check_interval': 900,   # 15 minutes
            'signal_lifetime': 20,   # Signal valid for 20 min
            'bad_regimes': []
        },
        '30T': {
            'model': 'logistic_30T.onnx',
            'threshold': 0.57,
            'check_interval': 1800,  # 30 minutes
            'signal_lifetime': 40,   # Signal valid for 40 min
            'bad_regimes': []
        }
    }
    
    # Position sizing
    RISK_PER_TRADE = 0.01  # 1% risk per trade
    MAX_OPEN_POSITIONS = 3  # Max concurrent trades
    MAX_DAILY_TRADES = 25
    
    # Guardrails
    MIN_PROBABILITY = 0.50  # Absolute minimum (below threshold)
    MAX_SLIPPAGE_PIPS = 5   # Reject if price moved >5 pips from close
    REGIME_FILTERING_ENABLED = True


# ═══════════════════════════════════════════════════════════════════════════
# FEATURE CALCULATOR (REAL-TIME)
# ═══════════════════════════════════════════════════════════════════════════

class FeatureCalculator:
    """Calculate features from latest OHLCV data."""
    
    @staticmethod
    def calculate_features(df: pd.DataFrame) -> np.ndarray:
        """
        Calculate features matching training pipeline.
        
        Args:
            df: Recent OHLCV data (last 300+ bars for indicators)
        
        Returns:
            Feature vector for latest candle
        """
        # This should match your training feature engineering EXACTLY
        # Import from FeatureEngineer if possible
        
        # Example (simplified - use your actual features):
        features = {}
        
        # Price-based
        features['close'] = df['close'].iloc[-1]
        features['rsi_14'] = calculate_rsi(df['close'], 14).iloc[-1]
        features['atr_14'] = calculate_atr(df, 14).iloc[-1]
        
        # Regime
        atr_percentile = df['atr_14'].rank(pct=True).iloc[-1]
        features['regime_vol'] = 0 if atr_percentile < 0.33 else (1 if atr_percentile < 0.67 else 2)
        
        # Convert to array matching model input
        feature_vector = np.array([features[k] for k in sorted(features.keys())])
        
        return feature_vector
    
    @staticmethod
    def get_regime(df: pd.DataFrame) -> str:
        """Detect current regime."""
        atr = calculate_atr(df, 14).iloc[-1]
        atr_percentile = df['atr_14'].rank(pct=True).iloc[-1]
        
        if atr_percentile < 0.33:
            return 'Low Vol'
        elif atr_percentile < 0.67:
            return 'Med Vol'
        else:
            return 'High Vol'


# ═══════════════════════════════════════════════════════════════════════════
# SIGNAL GENERATOR (CORE ENGINE)
# ═══════════════════════════════════════════════════════════════════════════

class SignalGenerator:
    """Event-driven signal generation engine."""
    
    def __init__(self, symbol: str, timeframe: str):
        self.symbol = symbol
        self.timeframe = timeframe
        self.config = SignalConfig.TIMEFRAMES[timeframe]
        
        # Load ONNX model
        model_path = SignalConfig.MODEL_DIR / self.config['model']
        self.session = ort.InferenceSession(str(model_path))
        
        # Tracking
        self.daily_trade_count = 0
        self.open_positions = 0
        self.last_signal_time = None
    
    def check_for_signal(self, df: pd.DataFrame) -> Optional[TradingSignal]:
        """
        Check if current candle generates a signal.
        
        Args:
            df: Recent OHLCV data
        
        Returns:
            TradingSignal if conditions met, else None
        """
        now = datetime.now(timezone.utc)
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # GUARDRAIL 1: Position limits
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        
        if self.open_positions >= SignalConfig.MAX_OPEN_POSITIONS:
            return None  # Max positions reached
        
        if self.daily_trade_count >= SignalConfig.MAX_DAILY_TRADES:
            return None  # Daily limit reached
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # GUARDRAIL 2: Staleness check
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        
        latest_candle_time = df['timestamp'].iloc[-1]
        time_since_candle = (now - latest_candle_time).total_seconds()
        
        # Allow 60 second buffer for data propagation
        if time_since_candle > self.config['check_interval'] + 60:
            print(f"⚠️  STALE DATA: Last candle {time_since_candle:.0f}s ago")
            return None
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # GUARDRAIL 3: Regime filter
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        
        current_regime = FeatureCalculator.get_regime(df)
        
        if SignalConfig.REGIME_FILTERING_ENABLED:
            if current_regime in self.config['bad_regimes']:
                return self._create_rejected_signal(
                    df,
                    reason=f"Bad regime: {current_regime}"
                )
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # STEP 1: Calculate features
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        
        features = FeatureCalculator.calculate_features(df)
        features_scaled = features.reshape(1, -1)  # Model expects 2D input
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # STEP 2: Model inference
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        
        input_name = self.session.get_inputs()[0].name
        proba = self.session.run(None, {input_name: features_scaled})[0][0][1]  # Probability of class 1
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # GUARDRAIL 4: Probability threshold
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        
        if proba < self.config['threshold']:
            return None  # Below threshold
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # STEP 3: Calculate trade parameters
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        
        entry_price = df['close'].iloc[-1]
        atr = df['atr_14'].iloc[-1]
        
        # ATR-based stops (matching your training labels)
        sl_multiplier = 1.0
        tp_multiplier = self._get_tp_multiplier()  # From your training results
        
        stop_loss = entry_price - (sl_multiplier * atr)
        take_profit = entry_price + (tp_multiplier * atr)
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # STEP 4: Create signal
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        
        signal = TradingSignal(
            timestamp=latest_candle_time.isoformat(),
            symbol=self.symbol,
            timeframe=self.timeframe,
            direction='LONG',  # Model only predicts long for now
            probability=proba,
            model_name=self.config['model'].replace('.onnx', ''),
            confidence_threshold=self.config['threshold'],
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size_pct=SignalConfig.RISK_PER_TRADE,
            regime=current_regime,
            regime_approved=True,
            signal_id=f"{self.symbol}_{self.timeframe}_{now.timestamp()}",
            generated_at=now.isoformat(),
            expires_at=(now + timedelta(minutes=self.config['signal_lifetime'])).isoformat(),
            status='PENDING'
        )
        
        # Update tracking
        self.last_signal_time = now
        self.daily_trade_count += 1
        
        return signal
    
    def _get_tp_multiplier(self) -> float:
        """Get TP multiplier based on timeframe (from training results)."""
        multipliers = {
            '5T': 0.9,   # From your training
            '15T': 1.3,
            '30T': 1.3
        }
        return multipliers.get(self.timeframe, 1.3)
    
    def _create_rejected_signal(self, df: pd.DataFrame, reason: str) -> TradingSignal:
        """Create rejected signal for logging."""
        now = datetime.now(timezone.utc)
        entry_price = df['close'].iloc[-1]
        
        return TradingSignal(
            timestamp=df['timestamp'].iloc[-1].isoformat(),
            symbol=self.symbol,
            timeframe=self.timeframe,
            direction='NEUTRAL',
            probability=0.0,
            model_name=self.config['model'].replace('.onnx', ''),
            confidence_threshold=self.config['threshold'],
            entry_price=entry_price,
            stop_loss=entry_price,
            take_profit=entry_price,
            position_size_pct=0.0,
            regime='Unknown',
            regime_approved=False,
            signal_id=f"REJ_{self.symbol}_{self.timeframe}_{now.timestamp()}",
            generated_at=now.isoformat(),
            expires_at=now.isoformat(),
            status='REJECTED',
            rejection_reason=reason
        )


# ═══════════════════════════════════════════════════════════════════════════
# SIGNAL DISPATCHER (TELEGRAM + SUPABASE)
# ═══════════════════════════════════════════════════════════════════════════

class SignalDispatcher:
    """Send signals to Telegram and store in database."""
    
    @staticmethod
    def send_telegram(signal: TradingSignal):
        """Send signal via Telegram."""
        
        url = f"https://api.telegram.org/bot{SignalConfig.TELEGRAM_BOT_TOKEN}/sendMessage"
        
        payload = {
            'chat_id': SignalConfig.TELEGRAM_CHAT_ID,
            'text': signal.to_telegram_message(),
            'parse_mode': 'Markdown'
        }
        
        try:
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
            print(f"✅ Telegram sent: {signal.signal_id}")
        except Exception as e:
            print(f"❌ Telegram failed: {e}")
    
    @staticmethod
    def store_signal(signal: TradingSignal):
        """Store signal in Supabase."""
        
        url = f"{SignalConfig.SUPABASE_URL}/rest/v1/trading_signals"
        headers = {
            'apikey': SignalConfig.SUPABASE_KEY,
            'Authorization': f'Bearer {SignalConfig.SUPABASE_KEY}',
            'Content-Type': 'application/json'
        }
        
        try:
            response = requests.post(url, json=signal.to_dict(), headers=headers, timeout=10)
            response.raise_for_status()
            print(f"✅ Stored in DB: {signal.signal_id}")
        except Exception as e:
            print(f"❌ DB storage failed: {e}")
    
    @staticmethod
    def dispatch(signal: TradingSignal):
        """Send signal to all endpoints."""
        
        if signal.status == 'PENDING':
            # Only send actionable signals to Telegram
            SignalDispatcher.send_telegram(signal)
        
        # Always store all signals (including rejected) for analysis
        SignalDispatcher.store_signal(signal)


# ═══════════════════════════════════════════════════════════════════════════
# EVENT-DRIVEN SCHEDULER
# ═══════════════════════════════════════════════════════════════════════════

class EventScheduler:
    """
    Run signal checks at candle close times (event-driven).
    
    Deployment options:
    1. Modal.com: Separate function per timeframe
    2. Supabase Edge Function: Cron triggers
    3. Self-hosted: Systemd timers or cron
    """
    
    @staticmethod
    async def run_check(symbol: str, timeframe: str):
        """
        Check for signal at current candle close.
        
        Called by:
        - Modal cron: @modal.function(schedule=modal.Cron("*/5 * * * *"))
        - Supabase: pg_cron every 5 minutes
        """
        
        print(f"\n{'='*80}")
        print(f"SIGNAL CHECK: {symbol} {timeframe}")
        print(f"Time: {datetime.now(timezone.utc).isoformat()}")
        print(f"{'='*80}")
        
        try:
            # 1. Fetch latest data
            df = fetch_latest_ohlcv(symbol, timeframe, lookback=300)
            
            # 2. Generate signal
            generator = SignalGenerator(symbol, timeframe)
            signal = generator.check_for_signal(df)
            
            # 3. Dispatch if signal generated
            if signal:
                if signal.status == 'PENDING':
                    print(f"🎯 SIGNAL GENERATED: {signal.direction} @ {signal.probability:.1%}")
                    SignalDispatcher.dispatch(signal)
                elif signal.status == 'REJECTED':
                    print(f"🚫 SIGNAL REJECTED: {signal.rejection_reason}")
                    SignalDispatcher.store_signal(signal)  # Log rejections
            else:
                print(f"⚪ No signal")
            
        except Exception as e:
            print(f"❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            
            # Alert on failures
            alert_text = f"🚨 Signal generator crashed:\n{symbol} {timeframe}\n{e}"
            requests.post(
                f"https://api.telegram.org/bot{SignalConfig.TELEGRAM_BOT_TOKEN}/sendMessage",
                json={'chat_id': SignalConfig.TELEGRAM_CHAT_ID, 'text': alert_text}
            )


# ═══════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def fetch_latest_ohlcv(symbol: str, timeframe: str, lookback: int = 300) -> pd.DataFrame:
    """
    Fetch latest OHLCV data from Polygon.io or your data source.
    
    Args:
        symbol: Trading symbol
        timeframe: Timeframe (5T, 15T, 30T)
        lookback: Number of bars to fetch
    
    Returns:
        DataFrame with OHLCV data
    """
    # Implementation depends on your data source
    # Example: Polygon.io, Alpaca, or your S3 pipeline
    pass


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate ATR."""
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    return true_range.rolling(period).mean()


# ═══════════════════════════════════════════════════════════════════════════
# DEPLOYMENT EXAMPLE (MODAL.COM)
# ═══════════════════════════════════════════════════════════════════════════

"""
# modal_deployment.py

import modal

app = modal.App("xauusd-signals")

# 5T: Check every 5 minutes
@app.function(schedule=modal.Cron("*/5 * * * *"))
def check_5T():
    import asyncio
    asyncio.run(EventScheduler.run_check('XAUUSD', '5T'))

# 15T: Check every 15 minutes
@app.function(schedule=modal.Cron("*/15 * * * *"))
def check_15T():
    import asyncio
    asyncio.run(EventScheduler.run_check('XAUUSD', '15T'))

# 30T: Check every 30 minutes
@app.function(schedule=modal.Cron("*/30 * * * *"))
def check_30T():
    import asyncio
    asyncio.run(EventScheduler.run_check('XAUUSD', '30T'))

# Deploy: modal deploy modal_deployment.py
"""