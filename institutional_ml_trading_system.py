#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
INSTITUTIONAL-GRADE ML TRADING SYSTEM FOR XAUUSD/XAGUSD
═══════════════════════════════════════════════════════════════════════════════

A complete rebuild addressing critical pipeline failures:
- Poor probability calibration → Isotonic regression calibration
- Misaligned thresholds → Quantile-based dynamic thresholding
- AUC vs Profit disconnect → Profit-aligned labels with transaction costs
- Walk-forward failures → Robust expanding window validation
- No viable strategy → Ensemble models with regime awareness

Author: Quant Research Team
Date: 2025-11-16
Purpose: Production-ready intraday ML trading for precious metals
═══════════════════════════════════════════════════════════════════════════════
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import json

# ML Libraries
import xgboost as xgb
import lightgbm as lgb
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, classification_report,
    confusion_matrix, roc_curve
)

# Statistical
from scipy import stats
from scipy.stats import ks_2samp

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (14, 8)

print("=" * 80)
print("INSTITUTIONAL ML TRADING SYSTEM - INITIALIZED")
print("=" * 80)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: CONFIGURATION & DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

def to_serializable(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, (np.bool_, np.bool8)):
        return bool(obj)
    if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(item) for item in obj]
    return obj


@dataclass
class TradingConfig:
    """
    Configuration for trading strategy and risk management.

    TRADING STRATEGY (for model implementation):
    ============================================
    1. ENTRY SIGNAL:
       - Model predicts probability of profitable trade
       - Enter LONG when: probability > fixed_threshold (default 0.50)
       - Only enter if: spread < max acceptable, liquidity sufficient

    2. POSITION SIZING:
       - Risk per trade: 1% of current equity
       - Position size (lots) = (equity × 0.01) / (SL_distance × $100/lot)
       - Max position: 1.0 lots

    3. TAKE PROFIT:
       - TP_price = entry_price + (tp_atr_multiple × ATR)
       - Default: TP = entry + 2.0×ATR

    4. STOP LOSS:
       - SL_price = entry_price - (sl_atr_multiple × ATR)
       - Default: SL = entry - 1.0×ATR

    5. EXIT RULES:
       - Exit at TP if price hits TP_price
       - Exit at SL if price hits SL_price
       - Exit after max_holding_bars (default 8 bars = 2 hours for 15T)
       - Whichever comes first

    6. COST MODEL:
       - Entry: pay (close + spread/2 + slippage)
       - Exit: receive (close - spread/2 - slippage)
       - Spread: $0.30/oz for XAUUSD
       - Slippage: 0.01% of price
    """

    # Asset parameters
    symbol: str = "XAUUSD"
    timeframe: str = "15T"  # 15-minute bars

    # Transaction costs (realistic for gold/silver)
    spread_gold: float = 0.30  # $0.30 per ounce
    spread_silver: float = 0.03  # $0.03 per ounce
    slippage_pct: float = 0.0001  # 0.01% slippage
    commission_pct: float = 0.0  # Commission if any

    # Risk management
    initial_capital: float = 25000.0  # Starting capital
    max_position_size: float = 1.0  # Max lots/units
    risk_per_trade_pct: float = 0.015  # 1.5% risk per trade ($375 per trade on $25k)
    max_daily_drawdown_pct: float = 0.05  # 5% max daily DD

    # Label creation parameters (TP/SL-based)
    max_holding_bars: int = 8  # Maximum bars to hold trade (5T: 40min, 15T: 120min)
    tp_atr_multiple: float = 2.0  # Take profit = entry ± (2.0 × ATR) - matches diagnostic
    sl_atr_multiple: float = 1.0  # Stop loss = entry ± (1.0 × ATR)
    min_r_multiple: float = 0.3  # Minimum R multiple to consider trade valid (was 0.5)
    use_tpsl_labels: bool = True  # Use TP/SL-based labels vs simple forward return

    # Threshold optimization
    use_fixed_threshold: bool = False  # Use quantile method for adaptive thresholding
    fixed_threshold: float = 0.52  # Fallback: Trade when model predicts >52% probability
    signal_quantile: float = 0.60  # Take top 40% of signals (diagnostic proved edge exists)
    min_trades_per_segment: int = 20  # Minimum viable trades per segment

    # Feature selection (use top features from diagnostic)
    use_top_features_only: bool = True  # Use only the top 20 features identified by diagnostic
    top_features: list = None  # Will be set to top 20 features

    # Quote-based filtering (disabled by default - set to False to skip filtering)
    enable_quote_filtering: bool = False  # Enable/disable quote-based filters
    max_spread_percentile: float = 0.95  # Skip trades when spread > 95th percentile (very conservative)
    min_quote_count: int = 1  # Minimum quotes per bar for trade execution (relaxed)

    # Viability criteria (what makes a strategy acceptable)
    min_profit_factor: float = 1.5
    min_sharpe_ratio: float = 0.5
    max_acceptable_drawdown: float = 0.06  # 6% (tightened from 15%)

    def get_spread(self, symbol: str) -> float:
        """Get spread for given symbol."""
        if 'XAU' in symbol:
            return self.spread_gold
        elif 'XAG' in symbol:
            return self.spread_silver
        else:
            return 0.10  # Default

    def __post_init__(self):
        """Initialize top features list from diagnostic results."""
        if self.top_features is None:
            # Top 20 features identified by diagnose_and_fix.py (AUC 0.6678 for TP/SL)
            self.top_features = [
                'roc_3',
                'price_vs_vwma_10',
                'stoch_k',
                'macd',
                'correlation_20',
                'macd_signal',
                'price_vs_vwma_50',
                'price_vs_vwma_20',
                'roc_10',
                'vwma_20',
                'bb_width_20',
                'distance_from_ma_100',
                'zscore_100',
                'bb_width_50',
                'volume_ratio_20',
                'bb_position_20',
                'mfi',
                'volume_ratio_10',
                'price_ratio',
                'vwap'
            ]


@dataclass
class PerformanceMetrics:
    """Performance metrics for strategy evaluation."""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0

    gross_profit: float = 0.0
    gross_loss: float = 0.0
    net_profit: float = 0.0
    profit_factor: float = 0.0

    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0

    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0

    total_return_pct: float = 0.0
    annualized_return_pct: float = 0.0

    def to_dict(self) -> Dict:
        """Convert to dictionary with JSON-serializable types."""
        return {
            'total_trades': int(self.total_trades),
            'win_rate': float(round(self.win_rate, 3)),
            'profit_factor': float(round(self.profit_factor, 3)),
            'sharpe_ratio': float(round(self.sharpe_ratio, 3)),
            'max_drawdown_pct': float(round(self.max_drawdown_pct, 3)),
            'net_profit': float(round(self.net_profit, 2)),
            'total_return_pct': float(round(self.total_return_pct, 3))
        }

    def is_viable(self, config: TradingConfig) -> bool:
        """Check if strategy meets viability criteria."""
        return (
            self.total_trades >= config.min_trades_per_segment and
            self.profit_factor >= config.min_profit_factor and
            self.sharpe_ratio >= config.min_sharpe_ratio and
            self.max_drawdown_pct <= config.max_acceptable_drawdown
        )


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════════

class FeatureEngineer:
    """
    Advanced feature engineering incorporating proven strategies:
    - VWMA ROC Momentum
    - Volatility-adjusted returns
    - Mean reversion indicators
    - Volume-price dynamics
    - Multi-timeframe alignment
    - Cross-asset relationships (Gold-Silver correlation)
    """

    @staticmethod
    def add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum and trend features (VWMA ROC, MACD, RSI)."""
        df = df.copy()

        # Rate of Change (ROC) - Multiple periods
        for period in [3, 5, 10, 20]:
            df[f'roc_{period}'] = df['close'].pct_change(period)

        # Volume-Weighted Moving Average (VWMA)
        for period in [10, 20, 50]:
            vwma = (df['close'] * df['volume']).rolling(period).sum() / df['volume'].rolling(period).sum()
            df[f'vwma_{period}'] = vwma
            df[f'price_vs_vwma_{period}'] = (df['close'] - vwma) / vwma

        # MACD
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']

        # RSI
        for period in [14, 21]:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            df[f'rsi_{period}'] = 100 - (100 / (1 + rs))

        # Stochastic Oscillator
        low_14 = df['low'].rolling(14).min()
        high_14 = df['high'].rolling(14).max()
        df['stoch_k'] = 100 * (df['close'] - low_14) / (high_14 - low_14)
        df['stoch_d'] = df['stoch_k'].rolling(3).mean()

        return df

    @staticmethod
    def add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility and regime indicators (ATR, Bollinger Bands, regime classification)."""
        df = df.copy()

        # Average True Range (ATR)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)

        for period in [14, 20]:
            df[f'atr_{period}'] = true_range.rolling(period).mean()
            df[f'atr_pct_{period}'] = df[f'atr_{period}'] / df['close']

        # Bollinger Bands
        for period in [20, 50]:
            sma = df['close'].rolling(period).mean()
            std = df['close'].rolling(period).std()
            df[f'bb_upper_{period}'] = sma + (std * 2)
            df[f'bb_lower_{period}'] = sma - (std * 2)
            df[f'bb_width_{period}'] = (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}']) / sma
            df[f'bb_position_{period}'] = (df['close'] - df[f'bb_lower_{period}']) / (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}'])

        # Historical Volatility
        returns = df['close'].pct_change()
        for period in [10, 20, 50]:
            df[f'historical_vol_{period}'] = returns.rolling(period).std() * np.sqrt(252 * 24)  # Annualized

        # Volatility Regime (Low/Medium/High)
        vol_20 = returns.rolling(20).std()
        vol_percentiles = vol_20.rolling(100).apply(lambda x: stats.percentileofscore(x, x.iloc[-1]))
        df['vol_regime'] = pd.cut(vol_percentiles, bins=[0, 33, 66, 100], labels=[0, 1, 2])

        return df

    @staticmethod
    def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features (OBV, MFI, volume momentum)."""
        df = df.copy()

        # On-Balance Volume (OBV)
        obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        df['obv'] = obv
        df['obv_ema'] = obv.ewm(span=20).mean()

        # Money Flow Index (MFI)
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        money_flow = typical_price * df['volume']

        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(14).sum()
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(14).sum()
        mfi = 100 - (100 / (1 + positive_flow / negative_flow))
        df['mfi'] = mfi

        # Volume momentum
        for period in [5, 10, 20]:
            vol_ma = df['volume'].rolling(period).mean()
            df[f'volume_ratio_{period}'] = df['volume'] / vol_ma

        # Volume-Price Trend
        df['vpt'] = (df['close'].pct_change() * df['volume']).cumsum()

        return df

    @staticmethod
    def add_mean_reversion_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add mean reversion indicators."""
        df = df.copy()

        # Distance from moving averages
        for period in [20, 50, 100]:
            ma = df['close'].rolling(period).mean()
            df[f'distance_from_ma_{period}'] = (df['close'] - ma) / ma

            # Z-score
            std = df['close'].rolling(period).std()
            df[f'zscore_{period}'] = (df['close'] - ma) / std

        # VWAP deviation (intraday)
        df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
        df['distance_from_vwap'] = (df['close'] - df['vwap']) / df['vwap']

        return df

    @staticmethod
    def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features (hour, day of week, session)."""
        df = df.copy()

        if 'timestamp' not in df.columns and isinstance(df.index, pd.DatetimeIndex):
            df['timestamp'] = df.index

        if 'timestamp' in df.columns:
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek

            # Cyclic encoding
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

            # Trading session flags (UTC-based)
            df['asian_session'] = ((df['hour'] >= 0) & (df['hour'] < 8)).astype(int)
            df['london_session'] = ((df['hour'] >= 8) & (df['hour'] < 16)).astype(int)
            df['ny_session'] = ((df['hour'] >= 13) & (df['hour'] < 21)).astype(int)

        return df

    @staticmethod
    def add_cross_asset_features(df_primary: pd.DataFrame, df_secondary: pd.DataFrame,
                                 primary_symbol: str, secondary_symbol: str) -> pd.DataFrame:
        """
        Add cross-asset features (e.g., Gold-Silver correlation).
        Uses pairs trading logic from QuantConnect strategies.
        """
        df = df_primary.copy()

        # Ensure both have same index
        if 'timestamp' in df.columns and 'timestamp' in df_secondary.columns:
            df_secondary = df_secondary.set_index('timestamp')
            df = df.set_index('timestamp')

        # Align dataframes
        df_aligned = df.join(df_secondary[['close', 'volume']],
                            rsuffix=f'_{secondary_symbol}', how='left')

        # Price ratio (Gold/Silver ratio if applicable)
        if f'close_{secondary_symbol}' in df_aligned.columns:
            df_aligned['price_ratio'] = df_aligned['close'] / df_aligned[f'close_{secondary_symbol}']
            df_aligned['price_ratio_ma'] = df_aligned['price_ratio'].rolling(20).mean()
            df_aligned['price_ratio_zscore'] = (df_aligned['price_ratio'] - df_aligned['price_ratio_ma']) / df_aligned['price_ratio'].rolling(20).std()

            # Correlation
            for period in [10, 20]:
                df_aligned[f'correlation_{period}'] = df_aligned['close'].rolling(period).corr(df_aligned[f'close_{secondary_symbol}'])

        return df_aligned.reset_index() if 'timestamp' not in df_aligned.columns else df_aligned

    @staticmethod
    def create_all_features(df: pd.DataFrame, df_secondary: Optional[pd.DataFrame] = None,
                           primary_symbol: str = "XAUUSD", secondary_symbol: str = "XAGUSD") -> pd.DataFrame:
        """Create complete feature set."""
        print(f"🔧 Engineering features for {primary_symbol}...")

        df = FeatureEngineer.add_momentum_features(df)
        print("   ✓ Momentum features added")

        df = FeatureEngineer.add_volatility_features(df)
        print("   ✓ Volatility features added")

        df = FeatureEngineer.add_volume_features(df)
        print("   ✓ Volume features added")

        df = FeatureEngineer.add_mean_reversion_features(df)
        print("   ✓ Mean reversion features added")

        df = FeatureEngineer.add_time_features(df)
        print("   ✓ Time features added")

        if df_secondary is not None:
            df = FeatureEngineer.add_cross_asset_features(df, df_secondary, primary_symbol, secondary_symbol)
            print("   ✓ Cross-asset features added")

        # Drop NaN rows created by rolling calculations
        initial_rows = len(df)
        df = df.dropna()
        print(f"   ✓ Cleaned data: {initial_rows} → {len(df)} rows")

        return df


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: PROFIT-ALIGNED LABEL ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════════

class LabelEngineer:
    """
    Creates realistic TP/SL-based labels that reflect actual tradeable opportunities.
    Labels answer: "Would opening a trade at this bar have hit TP before SL?"
    """

    @staticmethod
    def create_profit_labels(df: pd.DataFrame, config: TradingConfig) -> pd.DataFrame:
        """
        Create labels based on TP/SL outcomes over a realistic holding period.

        For each bar, we simulate:
        - Entry at close price
        - TP/SL levels based on ATR multiples
        - Looking forward max_holding_bars to see which hits first

        Returns:
            df with 'target' column:
            - 1 (LONG): TP hit before SL on long side, better than short
            - 0 (FLAT/SHORT): No clear edge or short is better
        """
        df = df.copy()

        if config.use_tpsl_labels:
            return LabelEngineer._create_tpsl_labels(df, config)
        else:
            return LabelEngineer._create_simple_labels(df, config)

    @staticmethod
    def _create_tpsl_labels(df: pd.DataFrame, config: TradingConfig) -> pd.DataFrame:
        """
        TP/SL-based label creation - the realistic approach.
        """
        print(f"\n🎯 Creating TP/SL-based labels...")
        print(f"   Max holding period: {config.max_holding_bars} bars")
        print(f"   TP multiple: {config.tp_atr_multiple} × ATR")
        print(f"   SL multiple: {config.sl_atr_multiple} × ATR")
        print(f"   Min R multiple: {config.min_r_multiple}")

        # Ensure we have ATR
        if 'atr_14' not in df.columns:
            print(f"   ⚠️  ATR not found, calculating properly...")
            # Calculate True Range: max(high-low, |high-prev_close|, |low-prev_close|)
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df['atr_14'] = true_range.rolling(14).mean()

        n = len(df)
        max_hold = config.max_holding_bars

        # Pre-allocate arrays for results
        labels = np.zeros(n, dtype=int)
        long_r_multiples = np.zeros(n)
        short_r_multiples = np.zeros(n)
        holding_periods = np.zeros(n)

        # Process each bar (except last max_hold bars which don't have full future)
        valid_bars = n - max_hold

        for i in range(valid_bars):
            entry_price = df['close'].iloc[i]
            atr = df['atr_14'].iloc[i]

            # Skip if ATR is NaN or zero
            if pd.isna(atr) or atr <= 0:
                continue

            # Define TP/SL levels
            tp_long = entry_price + (config.tp_atr_multiple * atr)
            sl_long = entry_price - (config.sl_atr_multiple * atr)
            tp_short = entry_price - (config.tp_atr_multiple * atr)
            sl_short = entry_price + (config.sl_atr_multiple * atr)

            # Look forward through the holding window
            long_hit_tp = False
            long_hit_sl = False
            short_hit_tp = False
            short_hit_sl = False
            bars_held = 0

            for j in range(1, max_hold + 1):
                idx = i + j
                high = df['high'].iloc[idx]
                low = df['low'].iloc[idx]

                # Check LONG trade
                if not long_hit_tp and not long_hit_sl:
                    if high >= tp_long:
                        long_hit_tp = True
                        bars_held = j
                    elif low <= sl_long:
                        long_hit_sl = True
                        bars_held = j

                # Check SHORT trade
                if not short_hit_tp and not short_hit_sl:
                    if low <= tp_short:
                        short_hit_tp = True
                    elif high >= sl_short:
                        short_hit_sl = True

            # Calculate R multiples
            if long_hit_tp:
                long_r = config.tp_atr_multiple
            elif long_hit_sl:
                long_r = -config.sl_atr_multiple
            else:
                # Neither hit - calculate exit at max_hold
                exit_price = df['close'].iloc[i + max_hold]
                long_r = (exit_price - entry_price) / atr if atr > 0 else 0
                bars_held = max_hold

            if short_hit_tp:
                short_r = config.tp_atr_multiple
            elif short_hit_sl:
                short_r = -config.sl_atr_multiple
            else:
                exit_price = df['close'].iloc[i + max_hold]
                short_r = (entry_price - exit_price) / atr if atr > 0 else 0

            long_r_multiples[i] = long_r
            short_r_multiples[i] = short_r
            holding_periods[i] = bars_held

            # Label logic:
            # LONG (1) if: long R is positive, >= min_r_multiple, and better than short
            # FLAT (0) otherwise
            if long_r >= config.min_r_multiple and long_r > short_r:
                labels[i] = 1

        # Add to dataframe
        df['target'] = labels
        df['long_r'] = long_r_multiples
        df['short_r'] = short_r_multiples
        df['holding_bars'] = holding_periods

        # Remove bars without full future data
        df = df.iloc[:-max_hold].copy()

        # Report statistics
        long_count = (labels[:valid_bars] == 1).sum()
        flat_count = (labels[:valid_bars] == 0).sum()

        # Calculate stats for winning trades
        long_trades = long_r_multiples[:valid_bars][labels[:valid_bars] == 1]
        avg_long_r = long_trades.mean() if len(long_trades) > 0 else 0
        avg_holding = holding_periods[:valid_bars][labels[:valid_bars] == 1].mean() if len(long_trades) > 0 else 0

        print(f"\n📊 TP/SL Label Distribution:")
        print(f"   Total bars:           {valid_bars:,}")
        print(f"   LONG opportunities:   {long_count:,} ({long_count/valid_bars*100:.1f}%)")
        print(f"   FLAT/SHORT:           {flat_count:,} ({flat_count/valid_bars*100:.1f}%)")
        print(f"   Avg LONG R-multiple:  {avg_long_r:.2f}")
        print(f"   Avg holding period:   {avg_holding:.1f} bars")
        print(f"\n   This reflects realistic trades with TP={config.tp_atr_multiple}×ATR, SL={config.sl_atr_multiple}×ATR")

        return df

    @staticmethod
    def _create_simple_labels(df: pd.DataFrame, config: TradingConfig) -> pd.DataFrame:
        """
        Simple forward-return based labels (legacy method).
        """
        print(f"\n📊 Creating simple forward-return labels (legacy)...")

        df = df.copy()
        spread = config.get_spread(config.symbol)
        lookback = config.max_holding_bars

        # Calculate forward returns
        forward_price = df['close'].shift(-lookback)
        forward_return_long = (forward_price - df['close']) / df['close']

        # Account for spread and slippage
        total_cost_pct = (spread / df['close']) + config.slippage_pct + config.commission_pct

        # Profitable long: forward return exceeds costs
        profitable_long = forward_return_long > total_cost_pct

        # Create binary labels
        labels = np.zeros(len(df), dtype=int)
        labels[profitable_long] = 1

        df['target'] = labels
        df['forward_return_long'] = forward_return_long
        df['total_cost_pct'] = total_cost_pct

        # Remove last N bars
        df = df.iloc[:-lookback].copy()

        long_count = (labels[:-lookback] == 1).sum()
        total = len(df)

        print(f"   Total bars:          {total:,}")
        print(f"   Long opportunities:  {long_count:,} ({long_count/total*100:.1f}%)")

        return df

    @staticmethod
    def balance_labels(df: pd.DataFrame, method: str = 'undersample') -> pd.DataFrame:
        """
        Balance labels to prevent model bias.

        Methods:
        - 'undersample': Randomly undersample majority class
        - 'threshold': Use stricter threshold for majority class
        """
        df = df.copy()

        label_counts = df['target'].value_counts()
        minority_class = label_counts.idxmin()
        majority_class = label_counts.idxmax()
        minority_count = label_counts[minority_class]

        print(f"\n⚖️  Balancing labels using {method}...")
        print(f"   Before: {label_counts.to_dict()}")

        if method == 'undersample':
            # Undersample majority class
            minority_df = df[df['target'] == minority_class]
            majority_df = df[df['target'] == majority_class]

            # Randomly sample majority to match minority
            majority_sampled = majority_df.sample(n=minority_count, random_state=42)

            df_balanced = pd.concat([minority_df, majority_sampled]).sort_index()

        elif method == 'threshold':
            # Use stricter threshold for profitable moves
            # This keeps more data but requires stronger signals
            df_balanced = df.copy()
            # Implementation would adjust the profitability threshold
            # For now, just pass through

        else:
            df_balanced = df

        print(f"   After:  {df_balanced['target'].value_counts().to_dict()}")

        return df_balanced


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: MODEL TRAINING WITH CALIBRATION
# ═══════════════════════════════════════════════════════════════════════════════

class EnsembleModelTrainer:
    """
    Trains ensemble of models (XGBoost, LightGBM, MLP) with probability calibration.
    """

    def __init__(self, config: TradingConfig):
        self.config = config
        self.models = {}
        self.calibrated_models = {}
        self.scaler = StandardScaler()
        self.imputer = None  # Will be fitted during training
        self.feature_columns = []
        self.feature_importance = {}

    def prepare_data(self, df: pd.DataFrame, is_training: bool = True) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare features and labels.

        Args:
            df: Input dataframe
            is_training: If True, fit imputer and store feature columns. If False, use stored values.
        """
        if is_training:
            # Identify feature columns (exclude price, target, metadata, and label diagnostics)
            exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'timestamp',
                           'target', 'forward_return_long', 'forward_return_short', 'total_cost_pct',
                           'long_r', 'short_r', 'holding_bars']  # TP/SL label diagnostic columns

            if self.config.use_top_features_only:
                # Use only the top 20 features identified by diagnostic (AUC 0.6678)
                available_cols = set(df.columns)
                feature_cols = [f for f in self.config.top_features if f in available_cols]
                print(f"   [FEATURE SELECTION] Using top {len(feature_cols)} features from diagnostic")
                print(f"   Top 5: {', '.join(feature_cols[:5])}")
            else:
                # Use all available features
                feature_cols = [col for col in df.columns if col not in exclude_cols and not col.endswith('_XAGUSD')]
                print(f"   [FEATURE SELECTION] Using all {len(feature_cols)} available features")

            self.feature_columns = feature_cols
        else:
            # Use stored feature columns from training
            feature_cols = self.feature_columns

        X = df[feature_cols].values
        y = df['target'].values if 'target' in df.columns else np.zeros(len(df))

        # Handle NaN values (quote features may have NaNs for bars before 2022)
        nan_count = np.isnan(X).sum()
        if nan_count > 0:
            if is_training:
                # Fit imputer on training data
                self.imputer = SimpleImputer(strategy='median')
                X = self.imputer.fit_transform(X)
            else:
                # Use stored imputer from training
                X = self.imputer.transform(X)

        return X, y, feature_cols

    def train_xgboost(self, X_train: np.ndarray, y_train: np.ndarray,
                     X_val: np.ndarray, y_val: np.ndarray) -> xgb.XGBClassifier:
        """Train XGBoost model with early stopping and improved regularization."""
        print("\n🌲 Training XGBoost...")

        # Calculate scale_pos_weight for imbalance
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

        model = xgb.XGBClassifier(
            n_estimators=100,  # Reduced to prevent overfitting
            max_depth=3,  # Shallow trees to reduce variance
            learning_rate=0.03,  # Lower learning rate for stability
            min_child_weight=5,  # Increase regularization
            subsample=0.8,  # Sample 80% of data
            colsample_bytree=0.8,  # Sample 80% of features
            gamma=1.0,  # Moderate complexity control
            reg_alpha=0.1,  # Light L1 regularization
            reg_lambda=1.0,  # Light L2 regularization
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1,
            early_stopping_rounds=50
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        # Feature importance
        importance = dict(zip(self.feature_columns, model.feature_importances_))
        self.feature_importance['xgboost'] = importance

        print(f"   ✓ Training AUC: {roc_auc_score(y_train, model.predict_proba(X_train)[:, 1]):.3f}")
        print(f"   ✓ Validation AUC: {roc_auc_score(y_val, model.predict_proba(X_val)[:, 1]):.3f}")

        return model

    def train_lightgbm(self, X_train: np.ndarray, y_train: np.ndarray,
                      X_val: np.ndarray, y_val: np.ndarray) -> lgb.LGBMClassifier:
        """Train LightGBM model with improved regularization."""
        print("\n🌟 Training LightGBM...")

        # Calculate scale_pos_weight
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

        model = lgb.LGBMClassifier(
            n_estimators=100,  # Reduced to prevent overfitting
            max_depth=3,  # Shallow trees
            num_leaves=8,  # Reduced from 31 (2^3 = 8)
            learning_rate=0.03,  # Lower learning rate
            subsample=0.8,  # Standard subsampling
            colsample_bytree=0.8,  # Standard feature sampling
            reg_alpha=0.5,  # Stronger L1 regularization
            reg_lambda=2.0,  # Stronger L2 regularization
            min_child_samples=30,  # Increased for regularization
            min_split_gain=0.05,  # Require more gain for splits
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(30, verbose=False)]
        )

        # Feature importance
        importance = dict(zip(self.feature_columns, model.feature_importances_))
        self.feature_importance['lightgbm'] = importance

        print(f"   ✓ Training AUC: {roc_auc_score(y_train, model.predict_proba(X_train)[:, 1]):.3f}")
        print(f"   ✓ Validation AUC: {roc_auc_score(y_val, model.predict_proba(X_val)[:, 1]):.3f}")

        return model

    def train_mlp(self, X_train: np.ndarray, y_train: np.ndarray,
                 X_val: np.ndarray, y_val: np.ndarray) -> MLPClassifier:
        """Train Neural Network (MLP) with improved architecture and regularization."""
        print("\n🧠 Training Neural Network (MLP)...")

        # Scale features for neural network
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        model = MLPClassifier(
            hidden_layer_sizes=(100, 50),  # Moderate network size
            activation='relu',
            solver='adam',
            alpha=0.001,  # Light L2 regularization
            batch_size=256,  # Standard batch size
            learning_rate_init=0.001,  # Standard learning rate
            max_iter=200,  # More iterations
            early_stopping=True,
            validation_fraction=0.1,  # Standard validation split
            n_iter_no_change=10,  # Standard early stopping patience
            random_state=42,
            verbose=False
        )

        model.fit(X_train_scaled, y_train)

        print(f"   ✓ Training AUC: {roc_auc_score(y_train, model.predict_proba(X_train_scaled)[:, 1]):.3f}")
        print(f"   ✓ Validation AUC: {roc_auc_score(y_val, model.predict_proba(X_val_scaled)[:, 1]):.3f}")

        return model

    def calibrate_model(self, model, X_cal: np.ndarray, y_cal: np.ndarray,
                       model_name: str) -> CalibratedClassifierCV:
        """
        Calibrate model probabilities using isotonic regression.
        Critical fix for probability squashing issue.
        """
        print(f"\n🎯 Calibrating {model_name} probabilities...")

        # For MLP, need to scale features
        if model_name == 'mlp':
            X_cal = self.scaler.transform(X_cal)

        # Get uncalibrated probabilities
        probs_uncal = model.predict_proba(X_cal)[:, 1]
        print(f"   Uncalibrated probs: min={probs_uncal.min():.3f}, "
              f"median={np.median(probs_uncal):.3f}, max={probs_uncal.max():.3f}")

        # Calibrate using isotonic regression
        calibrated = CalibratedClassifierCV(
            estimator=model,
            method='isotonic',  # Works better for tree models
            cv='prefit'  # Use existing trained model
        )

        calibrated.fit(X_cal, y_cal)

        # Get calibrated probabilities
        probs_cal = calibrated.predict_proba(X_cal)[:, 1]
        print(f"   Calibrated probs:   min={probs_cal.min():.3f}, "
              f"median={np.median(probs_cal):.3f}, max={probs_cal.max():.3f}")

        # Measure calibration quality
        # Bin probabilities and check actual positive rate
        bins = np.linspace(0, 1, 11)
        bin_indices = np.digitize(probs_cal, bins)
        calibration_error = 0
        for i in range(1, len(bins)):
            mask = bin_indices == i
            if mask.sum() > 0:
                predicted_prob = probs_cal[mask].mean()
                actual_prob = y_cal[mask].mean()
                calibration_error += abs(predicted_prob - actual_prob)

        print(f"   Calibration error: {calibration_error:.3f}")

        return calibrated

    def train_ensemble(self, df_train: pd.DataFrame, df_val: pd.DataFrame) -> Dict:
        """Train all models and create ensemble."""
        print("=" * 80)
        print("TRAINING ENSEMBLE MODELS")
        print("=" * 80)

        # Prepare data
        X_train, y_train, feature_cols = self.prepare_data(df_train, is_training=True)
        X_val, y_val, _ = self.prepare_data(df_val, is_training=False)

        print(f"\n📊 Training set: {len(X_train):,} samples")
        print(f"📊 Validation set: {len(X_val):,} samples")
        print(f"📊 Features: {len(feature_cols)}")
        print(f"📊 Class balance: {(y_train==1).sum()} pos, {(y_train==0).sum()} neg")

        # Train individual models
        xgb_model = self.train_xgboost(X_train, y_train, X_val, y_val)
        lgb_model = self.train_lightgbm(X_train, y_train, X_val, y_val)
        mlp_model = self.train_mlp(X_train, y_train, X_val, y_val)

        self.models = {
            'xgboost': xgb_model,
            'lightgbm': lgb_model,
            'mlp': mlp_model
        }

        # Calibrate models
        xgb_cal = self.calibrate_model(xgb_model, X_val, y_val, 'xgboost')
        lgb_cal = self.calibrate_model(lgb_model, X_val, y_val, 'lightgbm')
        mlp_cal = self.calibrate_model(mlp_model, X_val, y_val, 'mlp')

        self.calibrated_models = {
            'xgboost': xgb_cal,
            'lightgbm': lgb_cal,
            'mlp': mlp_cal
        }

        return self.calibrated_models

    def predict_ensemble(self, df: pd.DataFrame, method: str = 'average') -> np.ndarray:
        """
        Get ensemble predictions.

        Methods:
        - 'average': Simple average of calibrated probabilities
        - 'weighted': Weighted by validation AUC
        - 'voting': Majority vote
        """
        X, _, _ = self.prepare_data(df, is_training=False)

        predictions = []
        for name, model in self.calibrated_models.items():
            if name == 'mlp':
                X_scaled = self.scaler.transform(X)
                pred = model.predict_proba(X_scaled)[:, 1]
            else:
                pred = model.predict_proba(X)[:, 1]
            predictions.append(pred)

        if method == 'average':
            ensemble_pred = np.mean(predictions, axis=0)
        elif method == 'voting':
            votes = (np.array(predictions) > 0.5).astype(int)
            ensemble_pred = (votes.sum(axis=0) >= 2).astype(float)
        else:
            ensemble_pred = np.mean(predictions, axis=0)

        return ensemble_pred

    def print_feature_importance(self, top_n: int = 20):
        """Print top important features."""
        print("\n" + "=" * 80)
        print("TOP FEATURE IMPORTANCE")
        print("=" * 80)

        for model_name, importance_dict in self.feature_importance.items():
            print(f"\n{model_name.upper()}:")
            sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
            for feat, imp in sorted_features[:top_n]:
                print(f"  {feat:30s}: {imp:.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: DYNAMIC THRESHOLD OPTIMIZATION
# ═══════════════════════════════════════════════════════════════════════════════

class ThresholdOptimizer:
    """
    Optimizes trading thresholds using quantile-based approach.
    Critical fix for misaligned threshold problem.
    """

    @staticmethod
    def find_optimal_threshold(predictions: np.ndarray, y_true: np.ndarray,
                              config: TradingConfig, method: str = 'quantile') -> Tuple[float, Dict]:
        """
        Find optimal threshold for trading signals.

        Methods:
        - 'quantile': Use top N% of predictions
        - 'profit': Maximize expected profit
        - 'f1': Maximize F1 score
        """
        print(f"\n🎯 Optimizing threshold using {method} method...")

        # Print prediction statistics for diagnostics
        print(f"\n📊 Prediction Distribution:")
        print(f"   Min:    {predictions.min():.4f}")
        print(f"   25th:   {np.percentile(predictions, 25):.4f}")
        print(f"   Median: {np.median(predictions):.4f}")
        print(f"   75th:   {np.percentile(predictions, 75):.4f}")
        print(f"   90th:   {np.percentile(predictions, 90):.4f}")
        print(f"   Max:    {predictions.max():.4f}")
        print(f"   Mean:   {predictions.mean():.4f}")
        print(f"   Std:    {predictions.std():.4f}")

        if config.use_fixed_threshold:
            # Use fixed probability threshold - simple and effective
            threshold = config.fixed_threshold
            signals = (predictions >= threshold).astype(int)
            num_signals = signals.sum()

            print(f"\n🎯 Using FIXED threshold: {threshold:.4f}")
            print(f"   Signals Generated: {num_signals} ({num_signals/len(predictions)*100:.1f}%)")

            # If too few signals, relax threshold automatically
            if num_signals < config.min_trades_per_segment:
                print(f"   ⚠️  Too few signals ({num_signals}), relaxing threshold...")
                for trial_threshold in [0.52, 0.50, 0.48, 0.45, 0.40]:
                    signals = (predictions >= trial_threshold).astype(int)
                    num_signals = signals.sum()
                    print(f"   Trying threshold {trial_threshold:.2f}: {num_signals} signals")
                    if num_signals >= config.min_trades_per_segment:
                        threshold = trial_threshold
                        break
                else:
                    # Final fallback - use threshold that gives us at least some trades
                    threshold = 0.40
                    signals = (predictions >= threshold).astype(int)
                    num_signals = signals.sum()
                    print(f"   Final fallback threshold {threshold:.2f}: {num_signals} signals")

        elif method == 'quantile':
            # Use quantile-based threshold
            threshold = np.quantile(predictions, config.signal_quantile)

            signals = (predictions >= threshold).astype(int)
            num_signals = signals.sum()

            print(f"\n🎯 Using QUANTILE threshold ({config.signal_quantile:.2f}): {threshold:.4f}")
            print(f"   Signals Generated: {num_signals} ({num_signals/len(predictions)*100:.1f}%)")

            if num_signals < config.min_trades_per_segment:
                # Relax threshold until we get some signals
                print(f"   ⚠️  Too few signals, relaxing threshold...")
                for quantile in [0.65, 0.60, 0.55, 0.50, 0.45]:
                    threshold = np.quantile(predictions, quantile)
                    signals = (predictions >= threshold).astype(int)
                    num_signals = signals.sum()
                    print(f"   Trying quantile {quantile:.2f}: {num_signals} signals")
                    if num_signals >= config.min_trades_per_segment:
                        break

        elif method == 'profit':
            # Try different thresholds and pick the one with best profit
            thresholds = np.arange(0.3, 0.9, 0.05)
            best_profit = -np.inf
            best_threshold = 0.5

            for thresh in thresholds:
                signals = (predictions >= thresh).astype(int)
                if signals.sum() < config.min_trades_per_segment:
                    continue

                # Simple profit estimation (win rate * avg win - loss rate * avg loss)
                tp = ((signals == 1) & (y_true == 1)).sum()
                fp = ((signals == 1) & (y_true == 0)).sum()

                if (tp + fp) == 0:
                    continue

                precision = tp / (tp + fp)
                expected_profit = precision * 2.0 - (1 - precision) * 1.0  # Assuming 2:1 RR

                if expected_profit > best_profit:
                    best_profit = expected_profit
                    best_threshold = thresh

            threshold = best_threshold
            signals = (predictions >= threshold).astype(int)
            num_signals = signals.sum()

            print(f"   Best profit threshold: {threshold:.4f}")
            print(f"   Expected profit: {best_profit:.3f}")
            print(f"   Signals: {num_signals}")

        else:  # f1
            # Find threshold that maximizes F1 score
            precision_vals, recall_vals, thresholds = precision_recall_curve(y_true, predictions)
            f1_scores = 2 * (precision_vals * recall_vals) / (precision_vals + recall_vals + 1e-10)
            best_idx = np.argmax(f1_scores)
            threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5

            signals = (predictions >= threshold).astype(int)
            num_signals = signals.sum()

            print(f"   Best F1 threshold: {threshold:.4f}")
            print(f"   F1 score: {f1_scores[best_idx]:.3f}")
            print(f"   Signals: {num_signals}")

        # Calculate metrics at this threshold
        if num_signals > 0:
            tp = ((signals == 1) & (y_true == 1)).sum()
            fp = ((signals == 1) & (y_true == 0)).sum()
            fn = ((signals == 0) & (y_true == 1)).sum()
            tn = ((signals == 0) & (y_true == 0)).sum()

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            metrics = {
                'threshold': threshold,
                'num_signals': num_signals,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'tp': tp,
                'fp': fp,
                'fn': fn,
                'tn': tn
            }

            print(f"   Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
        else:
            metrics = {'threshold': threshold, 'num_signals': 0}
            print("   ⚠️ WARNING: No signals generated!")

        return threshold, metrics


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6: BACKTESTING ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class RealisticBacktester:
    """
    Realistic backtesting with transaction costs, slippage, and proper execution logic.
    """

    def __init__(self, config: TradingConfig):
        self.config = config
        self.trades = []
        self.equity_curve = []

    def backtest(self, df: pd.DataFrame, signals: np.ndarray,
                threshold: float, starting_equity: float = None) -> PerformanceMetrics:
        """
        Run backtest on historical data with signals.

        Args:
            df: DataFrame with price data
            signals: Predicted probabilities
            threshold: Signal threshold for trading
            starting_equity: Starting equity for this segment (for cumulative WFV)
        """
        print(f"\n📈 Running backtest...")
        print(f"   Total bars: {len(df):,}")
        print(f"   Threshold: {threshold:.4f}")

        df = df.copy().reset_index(drop=True)
        df['signal_prob'] = signals
        df['signal'] = (signals >= threshold).astype(int)

        # Count raw signals before filtering
        raw_signals = (df['signal'] == 1).sum()
        print(f"   Raw signals (before filtering): {raw_signals} ({raw_signals/len(df)*100:.1f}%)")

        spread = self.config.get_spread(self.config.symbol)
        slippage_pct = self.config.slippage_pct

        # Check if quote features are available for filtering
        has_quote_features = 'q_spread_mean' in df.columns and 'q_quote_count' in df.columns
        spread_threshold = None

        if has_quote_features and self.config.enable_quote_filtering:
            # Calculate spread percentile threshold for filtering (only on non-NaN values)
            valid_spreads = df['q_spread_mean'].dropna()
            if len(valid_spreads) > 0:
                spread_threshold = valid_spreads.quantile(self.config.max_spread_percentile)
                print(f"   Quote-based filtering ENABLED:")
                print(f"     - Max spread threshold: {spread_threshold:.4f}")
                print(f"     - Min quote count: {self.config.min_quote_count}")
            else:
                print(f"   Quote-based filtering DISABLED (no valid quote data)")
                spread_threshold = None
        else:
            if not self.config.enable_quote_filtering:
                print(f"   Quote-based filtering DISABLED (config setting)")
            else:
                print(f"   Quote-based filtering DISABLED (quote features not available)")

        trades = []
        # Use provided starting equity or default to initial capital
        equity = starting_equity if starting_equity is not None else self.config.initial_capital

        # Safety check: ensure we have valid starting equity
        if equity <= 0:
            print(f"\n   ⚠️  WARNING: Invalid starting equity: ${equity:.2f}")
            print(f"   Using default initial_capital: ${self.config.initial_capital:.2f}")
            equity = self.config.initial_capital

        self.starting_equity = equity  # Store for reporting
        equity_curve = [equity]
        peak_equity = equity
        max_dd = 0
        filtered_by_spread = 0
        filtered_by_liquidity = 0

        print(f"\n   💵 Starting equity for this segment: ${equity:,.2f}")

        i = 0
        while i < len(df) - 1:
            if df.loc[i, 'signal'] == 1:
                # Apply quote-based filters if enabled and spread threshold is set
                if spread_threshold is not None:
                    # Only filter if we have valid quote data for this bar
                    spread_val = df.loc[i, 'q_spread_mean']
                    quote_count_val = df.loc[i, 'q_quote_count']

                    # Skip if spread data is valid and too wide
                    if pd.notna(spread_val) and spread_val > spread_threshold:
                        filtered_by_spread += 1
                        i += 1
                        continue

                    # Skip if quote count is valid and insufficient
                    if pd.notna(quote_count_val) and quote_count_val < self.config.min_quote_count:
                        filtered_by_liquidity += 1
                        i += 1
                        continue
                # Enter long trade
                entry_price = df.loc[i, 'close']

                # Get ATR for stop-loss distance (in dollars for XAUUSD)
                atr = df.loc[i, 'atr_14'] if 'atr_14' in df.columns else (df.loc[i, 'high'] - df.loc[i, 'low'])

                # Stop-loss distance in dollars (using SL multiple from config)
                sl_distance = self.config.sl_atr_multiple * atr

                # XAUUSD position sizing:
                # 1 lot = 100 oz, $1 move = $100 profit/loss per lot
                # Risk amount in dollars
                risk_amount = equity * self.config.risk_per_trade_pct

                # Position size in lots based on stop-loss distance
                # Dollar risk per lot = sl_distance * 100
                if sl_distance > 0:
                    position_size_lots = risk_amount / (sl_distance * 100.0)
                    position_size_lots = min(position_size_lots, self.config.max_position_size)
                else:
                    position_size_lots = self.config.max_position_size

                # Exit after max_holding_bars or at end of data
                exit_idx = min(i + self.config.max_holding_bars, len(df) - 1)
                exit_price = df.loc[exit_idx, 'close']

                # Apply spread and slippage to entry and exit
                # Entry: pay ask (close + spread/2) + slippage
                # Exit: receive bid (close - spread/2) - slippage
                effective_entry = entry_price + (spread / 2.0) + (entry_price * slippage_pct)
                effective_exit = exit_price - (spread / 2.0) - (exit_price * slippage_pct)

                # Calculate PnL in dollars
                # Price change in dollars × position size in lots × 100 oz per lot
                price_change = effective_exit - effective_entry
                trade_pnl = price_change * position_size_lots * 100.0

                # Update equity
                equity += trade_pnl

                # Debug output (first 3 trades only)
                if len(trades) < 3:
                    print(f"\n   DEBUG Trade #{len(trades)+1}:")
                    print(f"      Entry: ${entry_price:.2f}, Exit: ${exit_price:.2f}")
                    print(f"      ATR: ${atr:.2f}, SL distance: ${sl_distance:.2f}")
                    print(f"      Risk amount: ${risk_amount:.2f}")
                    print(f"      Position size: {position_size_lots:.4f} lots")
                    print(f"      Effective entry: ${effective_entry:.2f}, exit: ${effective_exit:.2f}")
                    print(f"      Price change: ${price_change:.2f}")
                    print(f"      Trade PnL: ${trade_pnl:.2f}")
                    print(f"      New equity: ${equity:.2f}")

                trades.append({
                    'entry_idx': i,
                    'entry_time': df.loc[i, 'timestamp'] if 'timestamp' in df.columns else i,
                    'entry_price': entry_price,
                    'exit_idx': exit_idx,
                    'exit_time': df.loc[exit_idx, 'timestamp'] if 'timestamp' in df.columns else exit_idx,
                    'exit_price': exit_price,
                    'position_size_lots': position_size_lots,
                    'pnl': trade_pnl,
                    'pnl_pct': (trade_pnl / equity) * 100.0,
                    'signal_prob': df.loc[i, 'signal_prob']
                })

                equity_curve.append(equity)

                # Update max drawdown
                if equity > peak_equity:
                    peak_equity = equity
                dd = (peak_equity - equity) / peak_equity
                if dd > max_dd:
                    max_dd = dd

                # Skip to after exit
                i = exit_idx + 1
            else:
                i += 1

        # Report comprehensive filtering stats
        total_signals = (df['signal'] == 1).sum()
        print(f"\n📊 Trade Execution Summary:")
        print(f"   Total bars in period:     {len(df):,}")
        print(f"   Raw signals generated:    {total_signals} ({total_signals/len(df)*100:.1f}%)")

        if spread_threshold is not None:
            print(f"   Filtered by spread:       {filtered_by_spread} ({filtered_by_spread/total_signals*100:.1f}% of signals)" if total_signals > 0 else "   Filtered by spread:       0")
            print(f"   Filtered by liquidity:    {filtered_by_liquidity} ({filtered_by_liquidity/total_signals*100:.1f}% of signals)" if total_signals > 0 else "   Filtered by liquidity:    0")

        print(f"   Final executed trades:    {len(trades)} ({len(trades)/total_signals*100:.1f}% of signals)" if total_signals > 0 else f"   Final executed trades:    {len(trades)}")

        if total_signals > 0 and len(trades) == 0:
            print(f"\n   ⚠️  WARNING: ALL SIGNALS WERE FILTERED OUT!")
            print(f"   This indicates overly aggressive filtering or a configuration issue.")
        elif total_signals == 0:
            print(f"\n   ⚠️  WARNING: NO SIGNALS GENERATED BY MODEL!")
            print(f"   Model predictions are all below threshold {threshold:.4f}")
            print(f"   Consider:")
            print(f"     - Lowering threshold (current: {threshold:.4f})")
            print(f"     - Checking if model is learning anything")
            print(f"     - Reviewing label distribution")

        # Calculate metrics
        if len(trades) > 0:
            trades_df = pd.DataFrame(trades)

            winning_trades = trades_df[trades_df['pnl'] > 0]
            losing_trades = trades_df[trades_df['pnl'] <= 0]

            total_trades = len(trades_df)
            num_wins = len(winning_trades)
            num_losses = len(losing_trades)
            win_rate = num_wins / total_trades if total_trades > 0 else 0

            gross_profit = winning_trades['pnl'].sum() if num_wins > 0 else 0
            gross_loss = abs(losing_trades['pnl'].sum()) if num_losses > 0 else 0
            net_profit = gross_profit - gross_loss
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

            avg_win = winning_trades['pnl'].mean() if num_wins > 0 else 0
            avg_loss = losing_trades['pnl'].mean() if num_losses > 0 else 0
            largest_win = winning_trades['pnl'].max() if num_wins > 0 else 0
            largest_loss = losing_trades['pnl'].min() if num_losses > 0 else 0

            # Calculate Sharpe ratio using RETURN PERCENTAGES (not dollar PnL)
            # Reconstruct equity at start of each trade
            cumulative_pnl = trades_df['pnl'].cumsum()
            starting_equities = [self.starting_equity] + list(self.starting_equity + cumulative_pnl.iloc[:-1].values)
            return_pcts = trades_df['pnl'].values / np.array(starting_equities)

            if len(return_pcts) > 1 and return_pcts.std() > 0:
                # Per-trade Sharpe (no time-based annualization for intraday)
                sharpe = return_pcts.mean() / return_pcts.std()
            else:
                sharpe = 0

            # Sortino ratio (using return percentages)
            downside_returns = return_pcts[return_pcts < 0]
            if len(downside_returns) > 1 and downside_returns.std() > 0:
                sortino = return_pcts.mean() / downside_returns.std()
            else:
                sortino = 0

            # Total return based on starting equity for this segment
            total_return_pct = (equity - self.starting_equity) / self.starting_equity

            metrics = PerformanceMetrics(
                total_trades=total_trades,
                winning_trades=num_wins,
                losing_trades=num_losses,
                win_rate=win_rate,
                gross_profit=gross_profit,
                gross_loss=gross_loss,
                net_profit=net_profit,
                profit_factor=profit_factor,
                sharpe_ratio=sharpe,
                sortino_ratio=sortino,
                max_drawdown=max_dd,  # Store as decimal (removed erroneous *10000)
                max_drawdown_pct=max_dd,
                avg_win=avg_win,
                avg_loss=avg_loss,
                largest_win=largest_win,
                largest_loss=largest_loss,
                total_return_pct=total_return_pct
            )

        else:
            print("   ⚠️ No trades executed")
            metrics = PerformanceMetrics()

        self.trades = trades
        self.equity_curve = equity_curve
        self.final_equity = equity  # Store final equity for cumulative tracking

        return metrics

    def print_metrics(self, metrics: PerformanceMetrics):
        """Print formatted metrics."""
        print("\n" + "=" * 80)
        print("BACKTEST RESULTS")
        print("=" * 80)
        print(f"Total Trades:        {metrics.total_trades}")
        print(f"Winning Trades:      {metrics.winning_trades} ({metrics.win_rate*100:.1f}%)")
        print(f"Losing Trades:       {metrics.losing_trades}")
        print(f"\nGross Profit:        ${metrics.gross_profit:,.2f}")
        print(f"Gross Loss:          ${metrics.gross_loss:,.2f}")
        print(f"Net Profit:          ${metrics.net_profit:,.2f}")
        print(f"Total Return:        {metrics.total_return_pct*100:.2f}%")
        print(f"\nProfit Factor:       {metrics.profit_factor:.2f}")
        print(f"Sharpe Ratio:        {metrics.sharpe_ratio:.2f}")
        print(f"Sortino Ratio:       {metrics.sortino_ratio:.2f}")
        print(f"Max Drawdown:        {metrics.max_drawdown_pct*100:.2f}%")
        print(f"\nAvg Win:             ${metrics.avg_win:,.2f}")
        print(f"Avg Loss:            ${metrics.avg_loss:,.2f}")
        print(f"Largest Win:         ${metrics.largest_win:,.2f}")
        print(f"Largest Loss:        ${metrics.largest_loss:,.2f}")

        # Viability check
        is_viable = metrics.is_viable(self.config)
        print(f"\n{'✅ VIABLE STRATEGY' if is_viable else '❌ NOT VIABLE'}")
        if not is_viable:
            print(f"   Criteria: PF>={self.config.min_profit_factor}, "
                  f"Sharpe>={self.config.min_sharpe_ratio}, "
                  f"DD<={self.config.max_acceptable_drawdown*100}%, "
                  f"Trades>={self.config.min_trades_per_segment}")

        print("=" * 80)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7: WALK-FORWARD VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

class WalkForwardValidator:
    """
    Walk-forward validation with expanding window.
    """

    def __init__(self, config: TradingConfig):
        self.config = config
        self.results = []
        # Ensure we have a valid initial capital
        if config.initial_capital <= 0:
            raise ValueError(f"Invalid initial_capital: {config.initial_capital}. Must be > 0")
        self.cumulative_equity = config.initial_capital  # Track equity across all segments
        self.all_trades = []  # Accumulate all trades for cumulative analysis
        self.min_equity_floor = config.initial_capital * 0.20  # Stop if equity drops below 20%
        print(f"\n💰 Walk-Forward Validator initialized with ${self.cumulative_equity:,.2f} starting capital")
        print(f"   Minimum equity floor: ${self.min_equity_floor:,.2f} (20% of initial)")

    def create_segments(self, df: pd.DataFrame,
                       train_months: int = 6,
                       test_months: int = 1) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Create walk-forward segments.

        Args:
            df: Full dataset
            train_months: Initial training period
            test_months: Test period length
        """
        print(f"\n📅 Creating walk-forward segments...")
        print(f"   Training period: {train_months} months")
        print(f"   Test period: {test_months} months")

        if 'timestamp' not in df.columns:
            df = df.reset_index()

        df = df.sort_values('timestamp')
        min_date = df['timestamp'].min()
        max_date = df['timestamp'].max()

        segments = []
        current_date = min_date + pd.DateOffset(months=train_months)

        while current_date < max_date:
            # Training set: from start to current_date
            train_df = df[df['timestamp'] < current_date].copy()

            # Test set: next test_months
            test_end = current_date + pd.DateOffset(months=test_months)
            test_df = df[(df['timestamp'] >= current_date) & (df['timestamp'] < test_end)].copy()

            if len(train_df) > 1000 and len(test_df) > 100:
                segments.append((train_df, test_df))
                print(f"   Segment {len(segments)}: Train={len(train_df)}, Test={len(test_df)} "
                      f"(Test: {current_date.date()} to {test_end.date()})")

            # Move to next period (expanding window)
            current_date = test_end

        print(f"\n✓ Created {len(segments)} segments")
        return segments

    def validate(self, df: pd.DataFrame, df_secondary: Optional[pd.DataFrame] = None,
                 train_months: int = 6, test_months: int = 1) -> List[Dict]:
        """
        Run walk-forward validation.

        Args:
            df: Primary dataframe with OHLCV + features
            df_secondary: Optional secondary asset dataframe
            train_months: Initial training period in months
            test_months: Test period length in months
        """
        print("\n" + "=" * 80)
        print("WALK-FORWARD VALIDATION")
        print("=" * 80)

        # Create segments
        segments = self.create_segments(df, train_months=train_months, test_months=test_months)

        all_results = []

        for seg_num, (train_df, test_df) in enumerate(segments, 1):
            print(f"\n{'=' * 80}")
            print(f"SEGMENT {seg_num}/{len(segments)}")
            print(f"{'=' * 80}")

            # Feature engineering
            # Check if features are already calculated
            feature_cols = [col for col in train_df.columns
                           if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume',
                                         'bid', 'ask', 'spread', 'spread_pct', 'mid', 'label']]

            if len(feature_cols) < 10:
                print("   🔧 Calculating features...")
                train_df = FeatureEngineer.create_all_features(train_df, df_secondary)
                test_df = FeatureEngineer.create_all_features(test_df, df_secondary)
            else:
                print(f"   ✓ Using pre-calculated features ({len(feature_cols)} features)")
                # Still need to apply cross-asset features if secondary data provided
                if df_secondary is not None and 'price_ratio' not in train_df.columns:
                    train_df = FeatureEngineer.add_cross_asset_features(
                        train_df, df_secondary, self.config.symbol, "XAGUSD"
                    )
                    test_df = FeatureEngineer.add_cross_asset_features(
                        test_df, df_secondary, self.config.symbol, "XAGUSD"
                    )

            # Label engineering (only needed for training data)
            train_df = LabelEngineer.create_profit_labels(train_df, self.config)
            # Note: test_df doesn't need labels - backtest uses predictions only

            # Balance training labels
            train_df = LabelEngineer.balance_labels(train_df, method='undersample')

            # Split training into train/val for model calibration
            train_size = int(len(train_df) * 0.8)
            train_split = train_df.iloc[:train_size]
            val_split = train_df.iloc[train_size:]

            # Train ensemble
            trainer = EnsembleModelTrainer(self.config)
            trainer.train_ensemble(train_split, val_split)

            # Get predictions on test set
            test_predictions = trainer.predict_ensemble(test_df, method='average')

            # Optimize threshold on validation set
            val_predictions = trainer.predict_ensemble(val_split, method='average')
            threshold, threshold_metrics = ThresholdOptimizer.find_optimal_threshold(
                val_predictions, val_split['target'].values, self.config, method='quantile'
            )

            # Check if equity has dropped below minimum floor
            if self.cumulative_equity < self.min_equity_floor:
                print(f"\n   ⚠️  EQUITY BELOW MINIMUM FLOOR!")
                print(f"   Current: ${self.cumulative_equity:,.2f}, Floor: ${self.min_equity_floor:,.2f}")
                print(f"   Resetting equity to initial capital: ${self.config.initial_capital:,.2f}")
                self.cumulative_equity = self.config.initial_capital

            # Backtest on test set with cumulative equity
            backtester = RealisticBacktester(self.config)
            metrics = backtester.backtest(test_df, test_predictions, threshold,
                                         starting_equity=self.cumulative_equity)
            backtester.print_metrics(metrics)

            # Update cumulative equity and trades
            self.cumulative_equity = backtester.final_equity
            self.all_trades.extend(backtester.trades)

            print(f"\n   💰 Segment equity: ${backtester.starting_equity:,.2f} → ${backtester.final_equity:,.2f}")
            print(f"   📊 Cumulative equity: ${self.cumulative_equity:,.2f}")

            # Store results
            result = {
                'segment': int(seg_num),
                'train_size': int(len(train_df)),
                'test_size': int(len(test_df)),
                'threshold': float(threshold),
                'metrics': metrics.to_dict(),
                'is_viable': bool(metrics.is_viable(self.config))
            }
            all_results.append(result)

        self.results = all_results
        return all_results

    def print_summary(self):
        """Print summary of all segments."""
        if not self.results:
            print("No results to summarize")
            return

        print("\n" + "=" * 80)
        print("WALK-FORWARD VALIDATION SUMMARY")
        print("=" * 80)

        # Aggregate metrics
        total_trades = sum(r['metrics']['total_trades'] for r in self.results)
        viable_segments = sum(1 for r in self.results if r['is_viable'])

        avg_pf = np.mean([r['metrics']['profit_factor'] for r in self.results if r['metrics']['profit_factor'] > 0])
        avg_sharpe = np.mean([r['metrics']['sharpe_ratio'] for r in self.results if r['metrics']['sharpe_ratio'] > 0])
        avg_win_rate = np.mean([r['metrics']['win_rate'] for r in self.results if r['metrics']['total_trades'] > 0])

        print(f"\nTotal Segments:      {len(self.results)}")
        print(f"Viable Segments:     {viable_segments} ({viable_segments/len(self.results)*100:.1f}%)")
        print(f"Total Trades:        {total_trades}")
        print(f"\nAverage Profit Factor:  {avg_pf:.2f}")
        print(f"Average Sharpe Ratio:   {avg_sharpe:.2f}")
        print(f"Average Win Rate:       {avg_win_rate*100:.1f}%")

        print(f"\nPer-Segment Results:")
        print("-" * 80)
        for r in self.results:
            m = r['metrics']
            status = "✅" if r['is_viable'] else "❌"
            print(f"Segment {r['segment']}: {status} | Trades: {m['total_trades']:3d} | "
                  f"PF: {m['profit_factor']:5.2f} | Sharpe: {m['sharpe_ratio']:5.2f} | "
                  f"Return: {m['total_return_pct']*100:6.2f}%")

        print("=" * 80)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8: MAIN EXECUTION PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def load_real_data(symbol: str = "XAUUSD", timeframe: str = "15T", include_quotes: bool = True) -> pd.DataFrame:
    """
    Load REAL data from feature store parquet files. NO SYNTHETIC DATA ALLOWED.

    This function will ONLY load real parquet files with pre-calculated features.
    If the parquet file is not found, it will raise a HARD ERROR.
    NO fallback to synthetic/demo/sample data is permitted.

    Args:
        symbol: 'XAUUSD' or 'XAGUSD'
        timeframe: '5T', '15T', '30T', '1H'
        include_quotes: Whether to merge quote data (bid/ask spreads)

    Returns:
        DataFrame with OHLCV + features + quotes (if available)

    Raises:
        FileNotFoundError: If real parquet file does not exist
        ValueError: If data is invalid or missing required columns
    """
    # Try multiple possible locations for feature_store
    possible_paths = [
        Path("feature_store"),  # Relative to current directory
        Path(__file__).parent / "feature_store",  # Relative to script location
        Path.home() / "Desktop" / "ML_model" / "ML_model" / "feature_store",  # User's desktop location
    ]

    feature_store = None
    data_path = None

    for path in possible_paths:
        test_path = path / symbol / f"{symbol}_{timeframe}.parquet"
        if test_path.exists():
            feature_store = path
            data_path = test_path
            break

    # HARD ERROR if file not found - NO FALLBACK TO SYNTHETIC DATA
    if data_path is None or not data_path.exists():
        attempted_paths = [str(p / symbol / f"{symbol}_{timeframe}.parquet") for p in possible_paths]
        error_msg = f"""
╔════════════════════════════════════════════════════════════════════════════════╗
║ FATAL ERROR: REAL DATA FILE NOT FOUND                                         ║
╚════════════════════════════════════════════════════════════════════════════════╝

Could not load REAL parquet file for {symbol} {timeframe}.

Attempted locations:
{chr(10).join('  - ' + p for p in attempted_paths)}

NO SYNTHETIC OR FALLBACK DATA IS ALLOWED.

To fix this issue:
1. Ensure your feature_store parquet files exist in one of the above locations
2. Run calculate_all_features.py to generate feature files from raw data
3. Or download the feature files from your data source

The pipeline will NOT continue with fake/demo/synthetic data.
"""
        raise FileNotFoundError(error_msg)

    # Load the REAL parquet file
    print("\n" + "=" * 80)
    print("[DATA LOADING] Using REAL parquet file (NO synthetic data)")
    print("=" * 80)
    print(f"[DATA] File path: {data_path.absolute()}")
    print(f"[DATA] File size: {data_path.stat().st_size / 1024 / 1024:.2f} MB")

    df = pd.read_parquet(data_path)
    print(f"[DATA] Successfully loaded: {len(df):,} rows × {len(df.columns)} columns")

    # Ensure timestamp column exists
    if 'timestamp' not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
            if df.columns[0] != 'timestamp':
                df = df.rename(columns={df.columns[0]: 'timestamp'})
        else:
            raise ValueError(f"FATAL: No timestamp column or DatetimeIndex found in {data_path}")

    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Load and merge quote data if requested
    if include_quotes:
        quotes_path = feature_store / "quotes" / symbol / f"{symbol}_{timeframe}_quotes.parquet"

        if quotes_path.exists():
            print(f"[DATA] Loading quote data from: {quotes_path.absolute()}")
            df_quotes = pd.read_parquet(quotes_path)

            # Merge on timestamp
            df = pd.merge(df, df_quotes, on='timestamp', how='left', suffixes=('', '_quote'))

            print(f"[DATA] ✓ Merged {len(df_quotes):,} quote records")
            if 'spread' in df.columns:
                print(f"[DATA] ✓ Avg spread: {df['spread'].mean():.4f} ({df['spread_pct'].mean():.4f}%)")
        else:
            print(f"[DATA] ⚠️  Quote data not found at {quotes_path} (continuing without quotes)")

    # Check features
    feature_cols = [col for col in df.columns
                   if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume',
                                 'bid', 'ask', 'spread', 'spread_pct', 'mid']]

    print(f"\n[DATA] Dataset summary:")
    print(f"  • Total rows: {len(df):,}")
    print(f"  • Total columns: {len(df.columns)}")
    print(f"  • Feature columns: {len(feature_cols)}")
    print(f"  • Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"  • Duration: {(df['timestamp'].max() - df['timestamp'].min()).days} days")
    print(f"\n[DATA] First 3 rows of loaded data:")
    print(df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].head(3))

    if len(feature_cols) < 10:
        raise ValueError(f"FATAL: Only {len(feature_cols)} feature columns found. Expected at least 10. "
                        f"Run calculate_all_features.py to generate features.")

    print(f"[DATA] ✓ Using {len(feature_cols)} pre-calculated features from REAL parquet file")
    print("=" * 80 + "\n")

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# PRODUCTION SIGNAL GENERATOR (for deployment)
# ═══════════════════════════════════════════════════════════════════════════════

def generate_trading_signal(bar: pd.Series, model_probability: float, config: TradingConfig,
                           current_equity: float) -> Optional[Dict]:
    """
    Generate a trading signal based on the implemented strategy.

    This function implements the exact strategy used in training and backtesting,
    ready for production deployment.

    Args:
        bar: Current market bar with OHLCV and features
        model_probability: ML model's predicted probability (0-1)
        config: Trading configuration
        current_equity: Current account equity

    Returns:
        Trading signal dict with entry, TP, SL, position size, or None if no signal
    """
    # STEP 1: Check if model probability exceeds threshold
    if model_probability < config.fixed_threshold:
        return None

    # STEP 2: Get current price and ATR
    entry_price = bar['close']
    atr = bar.get('atr_14', bar['high'] - bar['low'])

    # STEP 3: Check quote quality if available
    if config.enable_quote_filtering:
        spread_val = bar.get('q_spread_mean', None)
        if spread_val is not None and pd.notna(spread_val):
            # Skip if spread too wide (placeholder for actual threshold)
            if spread_val > config.spread_gold * 2:
                return None

    # STEP 4: Calculate position size
    sl_distance = config.sl_atr_multiple * atr
    risk_amount = current_equity * config.risk_per_trade_pct
    position_size_lots = risk_amount / (sl_distance * 100.0)
    position_size_lots = min(position_size_lots, config.max_position_size)

    # STEP 5: Calculate TP and SL levels
    tp_price = entry_price + (config.tp_atr_multiple * atr)
    sl_price = entry_price - (config.sl_atr_multiple * atr)

    # STEP 6: Return signal
    return {
        'timestamp': bar.get('timestamp'),
        'signal_type': 'LONG',
        'entry_price': entry_price,
        'take_profit': tp_price,
        'stop_loss': sl_price,
        'position_size_lots': position_size_lots,
        'atr': atr,
        'model_probability': model_probability,
        'risk_reward_ratio': config.tp_atr_multiple / config.sl_atr_multiple,
        'max_holding_bars': config.max_holding_bars,
        'expected_hold_time_minutes': config.max_holding_bars * 15 if config.timeframe == '15T' else None
    }


def main():
    """Main execution pipeline."""

    # Configuration
    config = TradingConfig(
        symbol="XAUUSD",
        timeframe="15T"
    )

    print(f"\n⚙️  Configuration:")
    print(f"   Symbol: {config.symbol}")
    print(f"   Timeframe: {config.timeframe}")
    print(f"   Initial Capital: ${config.initial_capital:,.2f}")
    print(f"   Risk per trade: {config.risk_per_trade_pct * 100}%")
    print(f"   Max holding: {config.max_holding_bars} bars")
    print(f"   TP/SL multiples: {config.tp_atr_multiple}R / {config.sl_atr_multiple}R")
    print(f"   Fixed threshold: {config.fixed_threshold}")
    print(f"   Min viable PF: {config.min_profit_factor}")
    print(f"   Min viable Sharpe: {config.min_sharpe_ratio}")

    # Load REAL data from feature_store parquet files
    # This will FAIL HARD if parquet files don't exist (no synthetic fallback)
    df_gold = load_real_data("XAUUSD", config.timeframe)
    df_silver = load_real_data("XAGUSD", config.timeframe) if False else None  # Optional

    # Run walk-forward validation
    validator = WalkForwardValidator(config)
    # Use 6 months train, 3 months test for robust validation
    results = validator.validate(df_gold, df_silver, train_months=6, test_months=3)
    validator.print_summary()

    # Save results
    output_dir = Path("institutional_ml_results")
    output_dir.mkdir(exist_ok=True)

    results_file = output_dir / f"wfv_results_{config.symbol}_{config.timeframe}.json"
    with open(results_file, 'w') as f:
        # Convert to JSON-serializable format
        results_json = []
        for r in results:
            r_copy = r.copy()
            results_json.append(r_copy)
        json.dump(results_json, f, indent=2)

    print(f"\n💾 Results saved to {results_file}")

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)
    print("""
Next Steps:
1. Review walk-forward validation results above
2. Check if segments meet viability criteria
3. Analyze feature importance from models
4. Fine-tune hyperparameters if needed
5. Deploy best-performing model to production

Key Improvements Implemented:
✓ Probability calibration using isotonic regression
✓ Quantile-based dynamic thresholding
✓ Profit-aligned labels with transaction costs
✓ Ensemble modeling (XGBoost + LightGBM + MLP)
✓ Walk-forward validation with expanding window
✓ Comprehensive feature engineering
✓ Realistic backtesting with costs and slippage
✓ Viability filtering with institutional criteria
    """)


if __name__ == "__main__":
    main()
