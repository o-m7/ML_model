"""
═══════════════════════════════════════════════════════════════════════════════
ML BACKTESTING ENGINE
═══════════════════════════════════════════════════════════════════════════════

Comprehensive backtesting system with:
- Bar-by-bar trade simulation
- ATR-based SL/TP
- Position sizing
- Risk metrics (WR, PF, Sharpe, DD)
- Trade-level statistics
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import joblib
import json
import os

from trading_strategies import BaseStrategy

# Debug mode flag
DEBUG_MODE = os.getenv('DEBUG_MODE', 'false').lower() == 'true'


# ═══════════════════════════════════════════════════════════════════════════
# RESULT CONTAINERS
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class BacktestResult:
    """Container for backtest results."""
    strategy_name: str
    symbol: str
    timeframe: str
    metrics: Dict[str, float]
    trades: Optional[pd.DataFrame] = None
    equity_curve: Optional[np.ndarray] = None
    
    def __str__(self):
        return (
            f"{self.strategy_name} | {self.symbol} {self.timeframe}\n"
            f"  WR: {self.metrics.get('win_rate', 0):.1%} | "
            f"PF: {self.metrics.get('profit_factor', 0):.2f} | "
            f"Sharpe: {self.metrics.get('sharpe', 0):.2f} | "
            f"DD: {self.metrics.get('max_drawdown_pct', 0):.1f}% | "
            f"Trades: {self.metrics.get('total_trades', 0):,}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# EQUITY & DRAWDOWN UTILITIES (from training system)
# ═══════════════════════════════════════════════════════════════════════════

def compute_equity_and_dd(r_multiples: np.ndarray, risk_per_trade: float = 0.01) -> Tuple[np.ndarray, float, float]:
    """
    Compute equity curve and drawdown metrics correctly.
    
    Args:
        r_multiples: per-trade R results (1.0 = 1R profit, -1.0 = 1R loss)
        risk_per_trade: fraction of equity risked per trade (default 1%)
    
    Returns:
        equity_curve: array of equity values
        max_dd_pct: maximum drawdown as percentage
        max_dd_r: maximum drawdown in R units
    """
    if len(r_multiples) == 0:
        return np.array([1.0]), 0.0, 0.0
    
    # Build equity curve starting at 1.0 (100%)
    equity = np.zeros(len(r_multiples) + 1)
    equity[0] = 1.0
    
    for i, r in enumerate(r_multiples):
        # PnL as fraction of current equity
        pnl_fraction = r * risk_per_trade
        equity[i + 1] = equity[i] * (1.0 + pnl_fraction)
    
    # Compute running peak
    peaks = np.maximum.accumulate(equity)
    
    # Drawdown at each point
    dd_absolute = peaks - equity
    dd_pct = dd_absolute / peaks
    
    # Maximum drawdown
    max_dd_pct = dd_pct.max() * 100.0  # Convert to percentage
    
    # Max DD in R units (approximate)
    max_dd_r = dd_absolute.max() / risk_per_trade
    
    return equity, max_dd_pct, max_dd_r


class RiskMetrics:
    """Calculate trading risk metrics from r_multiples."""
    
    @staticmethod
    def calculate_profit_factor(r_multiples: np.ndarray) -> float:
        """Calculate Profit Factor: sum(R+) / sum(|R-|)"""
        winners = r_multiples[r_multiples > 0]
        losers = r_multiples[r_multiples < 0]
        
        gross_profit = winners.sum() if len(winners) > 0 else 0
        gross_loss = abs(losers.sum()) if len(losers) > 0 else 0
        
        if gross_loss == 0:
            return np.inf if gross_profit > 0 else 0
        
        return gross_profit / gross_loss
    
    @staticmethod
    def calculate_sharpe(r_multiples: np.ndarray) -> float:
        """Calculate Sharpe ratio: mean(R) / std(R) - per-trade"""
        if len(r_multiples) < 2:
            return 0.0
        
        mean_r = r_multiples.mean()
        std_r = r_multiples.std()
        
        if std_r == 0 or np.isnan(std_r):
            return 0.0
        
        return mean_r / std_r
    
    @staticmethod
    def calculate_max_drawdown(r_multiples: np.ndarray, risk_per_trade: float = 0.01) -> Tuple[float, float]:
        """Calculate max drawdown."""
        _, max_dd_pct, max_dd_r = compute_equity_and_dd(r_multiples, risk_per_trade)
        return max_dd_pct, max_dd_r
    
    @staticmethod
    def calculate_all_metrics(r_multiples: np.ndarray, risk_per_trade: float = 0.01) -> Dict:
        """Calculate all risk metrics."""
        if len(r_multiples) == 0:
            return {
                'total_trades': 0,
                'wins': 0,
                'losses': 0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'sharpe': 0.0,
                'max_drawdown_pct': 0.0,
                'max_drawdown_r': 0.0,
                'mean_r': 0.0,
                'median_r': 0.0,
                'total_r': 0.0
            }
        
        wins = (r_multiples > 0).sum()
        losses = (r_multiples < 0).sum()
        win_rate = wins / len(r_multiples) if len(r_multiples) > 0 else 0
        
        pf = RiskMetrics.calculate_profit_factor(r_multiples)
        sharpe = RiskMetrics.calculate_sharpe(r_multiples)
        max_dd_pct, max_dd_r = RiskMetrics.calculate_max_drawdown(r_multiples, risk_per_trade)
        
        return {
            'total_trades': len(r_multiples),
            'wins': int(wins),
            'losses': int(losses),
            'win_rate': win_rate,
            'profit_factor': pf,
            'sharpe': sharpe,
            'max_drawdown_pct': max_dd_pct,
            'max_drawdown_r': max_dd_r,
            'mean_r': r_multiples.mean(),
            'median_r': np.median(r_multiples),
            'total_r': r_multiples.sum()
        }


# ═══════════════════════════════════════════════════════════════════════════
# BACKTESTER CLASS
# ═══════════════════════════════════════════════════════════════════════════

class Backtester:
    """
    Bar-by-bar backtesting engine with ATR-based SL/TP.
    
    Supports:
    - Long and short trades
    - ATR-based stop loss and take profit
    - Time-based exit (max bars in trade)
    - Position sizing
    - Detailed trade logging
    """
    
    def __init__(self, df: pd.DataFrame, strategy: BaseStrategy, 
                 initial_equity: float = 100000.0,
                 risk_per_trade: float = 0.01,
                 spread_cost_r: float = 0.05):
        """
        Initialize backtester.
        
        Args:
            df: DataFrame with OHLCV, signals, ATR
            strategy: Strategy instance
            initial_equity: Starting capital
            risk_per_trade: Fraction of equity risked per trade
            spread_cost_r: Spread cost in R units (e.g., 0.05 = 0.05R per trade)
        """
        self.df = df.copy()
        self.strategy = strategy
        self.initial_equity = initial_equity
        self.risk_per_trade = risk_per_trade
        self.spread_cost_r = spread_cost_r
        
        # Validate required columns
        required = ['open', 'high', 'low', 'close', 'atr', 'signal']
        missing = [col for col in required if col not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
    
    def run(self) -> BacktestResult:
        """
        Execute backtest bar-by-bar.
        
        Returns:
            BacktestResult with metrics and trade details
        """
        trades = []
        r_multiples = []
        
        in_trade = False
        entry_bar = None
        entry_price = None
        sl_price = None
        tp_price = None
        trade_direction = None
        position_scale = 1.0
        bars_in_trade = 0
        
        # Get strategy parameters
        tp_mult = self.strategy.params.get('tp_atr_mult', 2.0)
        sl_mult = self.strategy.params.get('sl_atr_mult', 1.0)
        max_bars = self.strategy.params.get('max_bars_in_trade', 60)
        
        for i in range(len(self.df)):
            row = self.df.iloc[i]
            
            # Check if in trade
            if in_trade:
                bars_in_trade += 1
                
                # Check exits
                exit_type = None
                exit_price = None
                
                # TP hit
                if trade_direction == 1:  # Long
                    if row['high'] >= tp_price:
                        exit_type = 'TP'
                        exit_price = tp_price
                elif trade_direction == -1:  # Short
                    if row['low'] <= tp_price:
                        exit_type = 'TP'
                        exit_price = tp_price
                
                # SL hit
                if exit_type is None:
                    if trade_direction == 1:  # Long
                        if row['low'] <= sl_price:
                            exit_type = 'SL'
                            exit_price = sl_price
                    elif trade_direction == -1:  # Short
                        if row['high'] >= sl_price:
                            exit_type = 'SL'
                            exit_price = sl_price
                
                # Time exit
                if exit_type is None and bars_in_trade >= max_bars:
                    exit_type = 'TIME'
                    exit_price = row['close']
                
                # Exit trade if triggered
                if exit_type is not None:
                    # Calculate R-multiple
                    if trade_direction == 1:  # Long
                        r_pre = (exit_price - entry_price) / (entry_price - sl_price)
                    else:  # Short
                        r_pre = (entry_price - exit_price) / (sl_price - entry_price)
                    
                    # Apply spread cost
                    r_post = r_pre - self.spread_cost_r
                    
                    # Apply position scale (for confidence strategies)
                    r_final = r_post * position_scale
                    
                    # Record trade
                    trades.append({
                        'entry_bar': entry_bar,
                        'exit_bar': i,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'direction': 'LONG' if trade_direction == 1 else 'SHORT',
                        'exit_type': exit_type,
                        'bars_held': bars_in_trade,
                        'r_pre': r_pre,
                        'r_post': r_post,
                        'r_final': r_final,
                        'position_scale': position_scale
                    })
                    
                    r_multiples.append(r_final)
                    
                    # Reset
                    in_trade = False
                    bars_in_trade = 0
            
            # Check for new entries
            if not in_trade and row['signal'] != 0:
                # Enter trade
                entry_bar = i
                entry_price = row['close']  # Enter at close of signal bar
                trade_direction = int(row['signal'])
                atr = row['atr']
                
                # Get position scale if available (Strategy 5)
                position_scale = row.get('position_scale', 1.0)
                if pd.isna(position_scale) or position_scale == 0:
                    position_scale = 1.0
                
                # Calculate SL/TP
                if trade_direction == 1:  # Long
                    sl_price = entry_price - (sl_mult * atr)
                    tp_price = entry_price + (tp_mult * atr)
                else:  # Short
                    sl_price = entry_price + (sl_mult * atr)
                    tp_price = entry_price - (tp_mult * atr)
                
                in_trade = True
                bars_in_trade = 0
        
        # Close any open trade at end
        if in_trade:
            row = self.df.iloc[-1]
            exit_price = row['close']
            
            if trade_direction == 1:
                r_pre = (exit_price - entry_price) / (entry_price - sl_price)
            else:
                r_pre = (entry_price - exit_price) / (sl_price - entry_price)
            
            r_post = r_pre - self.spread_cost_r
            r_final = r_post * position_scale
            
            trades.append({
                'entry_bar': entry_bar,
                'exit_bar': len(self.df) - 1,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'direction': 'LONG' if trade_direction == 1 else 'SHORT',
                'exit_type': 'END',
                'bars_held': bars_in_trade,
                'r_pre': r_pre,
                'r_post': r_post,
                'r_final': r_final,
                'position_scale': position_scale
            })
            
            r_multiples.append(r_final)
        
        # Calculate metrics
        r_array = np.array(r_multiples) if r_multiples else np.array([])
        metrics = RiskMetrics.calculate_all_metrics(r_array, self.risk_per_trade)
        
        # Build equity curve
        equity_curve = None
        if len(r_array) > 0:
            equity_curve, _, _ = compute_equity_and_dd(r_array, self.risk_per_trade)
        
        # Return result
        return BacktestResult(
            strategy_name=self.strategy.name,
            symbol="SYMBOL",  # Will be set by caller
            timeframe=self.strategy.timeframe,
            metrics=metrics,
            trades=pd.DataFrame(trades) if trades else None,
            equity_curve=equity_curve
        )


# ═══════════════════════════════════════════════════════════════════════════
# ML MODEL INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════

def add_model_predictions(df: pd.DataFrame, model_path: Path, 
                         feature_cols: List[str]) -> pd.DataFrame:
    """
    Add ML model predictions to DataFrame without lookahead bias.
    
    Args:
        df: DataFrame with features (must be sorted by timestamp)
        model_path: Path to saved model (joblib format)
        feature_cols: List of feature column names used in training
    
    Returns:
        DataFrame with added p_up and p_down columns
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    # Load model
    model_dict = joblib.load(model_path)
    model = model_dict['model']
    scaler = model_dict['scaler']
    
    # Verify feature columns exist and fill missing ones
    missing_features = [col for col in feature_cols if col not in df.columns]
    if missing_features:
        print(f"   ⚠️  Missing {len(missing_features)} feature columns. Filling with defaults...")
        for col in missing_features:
            # Fill with 0 or appropriate default
            df[col] = 0.0
            if DEBUG_MODE:
                print(f"      Filled {col} with 0")
    
    # CRITICAL: Fill any NaN values in feature columns before prediction
    # ML models don't accept NaN values
    for col in feature_cols:
        if col in df.columns:
            nan_count = df[col].isna().sum()
            if nan_count > 0:
                if DEBUG_MODE:
                    print(f"      ⚠️  Found {nan_count} NaN values in {col}, filling with 0")
                df[col] = df[col].fillna(0.0)
                # Also fill any inf/-inf values with 0
                df[col] = df[col].replace([np.inf, -np.inf], 0.0)
    
    # Extract features
    X = df[feature_cols].values
    
    # Final check: ensure no NaN or inf in X before scaling
    if np.isnan(X).any() or np.isinf(X).any():
        if DEBUG_MODE:
            print(f"      ⚠️  Found NaN/Inf in X before scaling. Filling with 0.")
        # Fill NaN and inf with 0
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    # Get predictions (probabilities)
    y_proba = model.predict_proba(X_scaled)
    
    # Add to dataframe
    df = df.copy()
    df['p_down'] = y_proba[:, 0]  # Probability of class 0 (down/loss)
    df['p_up'] = y_proba[:, 1]    # Probability of class 1 (up/win)
    
    return df


def load_model_and_features(symbol: str, timeframe: str, 
                            base_path: Path = Path("ML_model/ML_model"),
                            df: Optional[pd.DataFrame] = None) -> Tuple[Path, List[str]]:
    """
    Load model path and feature columns for a symbol/timeframe.
    
    Args:
        symbol: Trading symbol (e.g., 'XAUUSD')
        timeframe: Timeframe (e.g., '5T', '15T', '30T')
        base_path: Base path to ML_model directory
        df: Optional DataFrame to auto-generate feature columns from if file is missing
    
    Returns:
        Tuple of (model_path, feature_columns)
    """
    # Model path (check multiple locations following training system convention)
    model_path_candidates = [
        base_path / "models" / symbol / f"{symbol}_{timeframe}_best_model.pkl",
        base_path / "models" / "production" / symbol / f"{symbol}_{timeframe}_best_model.pkl",
        base_path / "models" / "production" / symbol / f"{symbol}_{timeframe}_random_forest.pkl",
        base_path / "models" / "production" / symbol / f"{symbol}_{timeframe}_logistic.pkl",
    ]
    
    model_path = None
    for candidate in model_path_candidates:
        if candidate.exists():
            model_path = candidate
            break
    
    # If no model found, use the first candidate as default (will fail later if not found)
    if model_path is None:
        model_path = model_path_candidates[0]
    
    # Feature columns path (check multiple locations)
    features_path_candidates = [
        base_path / "models" / symbol / f"{symbol}_{timeframe}_feature_cols.json",
        base_path / "models" / "production" / symbol / f"{symbol}_{timeframe}_feature_cols.json",
    ]
    
    features_path = None
    for candidate in features_path_candidates:
        if candidate.exists():
            features_path = candidate
            break
    
    # If no features file found, use the first candidate as default
    if features_path is None:
        features_path = features_path_candidates[0]
    
    # Check if feature columns file exists
    if features_path.exists():
        # Load feature columns from file
        with open(features_path, 'r') as f:
            feature_cols = json.load(f)
    elif df is not None:
        # Auto-generate feature columns from DataFrame
        print(f"   ⚠️  Feature columns file not found, auto-generating from DataFrame...")
        
        # Required columns that are not features
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 
                        'atr', 'hour', 'p_up', 'p_down', 'ml_prediction']
        
        # Get all columns that are not in required list (these are features)
        feature_cols = [c for c in df.columns if c not in required_cols]
        
        # Filter out non-numeric columns
        feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
        
        print(f"   ✅ Auto-generated {len(feature_cols)} feature columns")
        
        # Optionally save the auto-generated feature columns for future use
        features_path.parent.mkdir(parents=True, exist_ok=True)
        with open(features_path, 'w') as f:
            json.dump(feature_cols, f, indent=2)
        print(f"   💾 Saved auto-generated feature columns to: {features_path}")
    else:
        raise FileNotFoundError(
            f"Feature columns file not found: {features_path}\n"
            f"Please ensure the training pipeline has been run for {symbol} {timeframe}"
        )
    
    # Verify model exists
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            f"Please ensure the training pipeline has been run for {symbol} {timeframe}"
        )
    
    return model_path, feature_cols