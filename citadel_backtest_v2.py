"""
═══════════════════════════════════════════════════════════════════════════════
CITADEL-GRADE ML BACKTESTING SYSTEM V2.4 - REALISTIC EXECUTION
═══════════════════════════════════════════════════════════════════════════════

FEATURES:
━━━━━━━━━━━━━━━━━━━━━━━━━
✓ REALISTIC spread costs (timeframe-dependent)
✓ REALISTIC slippage modeling (market impact)
✓ PROPER position sizing (fixed fractional risk)
✓ DETAILED trade logging (CSV export)
✓ COMPREHENSIVE performance metrics
✓ EQUITY CURVE visualization
✓ DRAWDOWN analysis with underwater plot
✓ MONTE CARLO simulation for robustness
✓ TRADE-BY-TRADE analysis

Usage:
    python citadel_backtest_v2.py --symbol XAUUSD --timeframe 5T
    python citadel_backtest_v2.py --symbol XAUUSD --all-timeframes --monte-carlo
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import warnings
import argparse
import json
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# Import FeatureEngineer for feature generation
from citadel_training_system_v2 import FeatureEngineer

warnings.filterwarnings('ignore')


# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class BacktestConfig:
    """Backtesting configuration with REALISTIC costs."""
    
    # Paths
    FEATURE_STORE: Path = Path("ML_model/ML_model/feature_store")
    MODEL_STORE: Path = Path("ML_model/ML_model/models")
    RESULTS_DIR: Path = Path("ML_model/ML_model/backtest_results")
    
    # Account settings
    INITIAL_CAPITAL: float = 100000.0  # $100k starting capital
    RISK_PER_TRADE: float = 0.01  # 1% risk per trade
    MAX_POSITION_SIZE: float = 0.05  # Max 5% of capital per trade
    
    # Execution costs (REALISTIC)
    SPREAD_R_BY_TIMEFRAME: Dict[str, float] = field(default_factory=lambda: {
        "5T": 0.08,   # ~0.08R per trade on 5T
        "15T": 0.06,  # ~0.06R per trade on 15T
        "30T": 0.05,  # ~0.05R per trade on 30T
        "1H": 0.04,   # ~0.04R per trade on 1H
        "4H": 0.03    # ~0.03R per trade on 4H
    })
    
    # Slippage model (as fraction of ATR)
    BASE_SLIPPAGE_R: float = 0.02  # Base slippage: 0.02R
    VOLUME_IMPACT_FACTOR: float = 0.01  # Additional slippage for volume
    
    # Trade management
    USE_TIME_BARRIER: bool = True  # Exit trades at time limit
    ENABLE_TRAILING_STOP: bool = False  # Trailing stop (future feature)
    
    # Risk limits
    MAX_DAILY_LOSS: float = 0.03  # Stop trading if down 3% in a day
    MAX_DRAWDOWN_STOP: float = 0.20  # Stop trading if DD > 20%
    
    # Analysis settings
    MONTE_CARLO_RUNS: int = 1000
    CONFIDENCE_LEVEL: float = 0.95


CONFIG = BacktestConfig()


# ═══════════════════════════════════════════════════════════════════════════
# TRADE CLASS
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Trade:
    """Individual trade record."""
    
    # Entry details
    entry_time: pd.Timestamp
    entry_price: float
    direction: str  # 'LONG' or 'SHORT'
    position_size: float  # Number of units
    
    # Risk management
    stop_loss: float
    take_profit: float
    atr_at_entry: float
    risk_amount: float  # Dollar risk
    
    # Exit details
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None  # 'TP', 'SL', 'TIME', 'DD_STOP'
    
    # Performance
    pnl_gross: float = 0.0  # Before costs
    pnl_net: float = 0.0  # After costs
    r_multiple_gross: float = 0.0
    r_multiple_net: float = 0.0
    
    # Costs
    spread_cost: float = 0.0
    slippage_cost: float = 0.0
    total_cost: float = 0.0
    
    # Trade metadata
    confidence: float = 0.0
    bar_index: int = 0
    bars_held: int = 0
    
    def close_trade(self, exit_time: pd.Timestamp, exit_price: float, 
                    exit_reason: str, spread_r: float, slippage_r: float):
        """Close the trade and calculate PnL."""
        self.exit_time = exit_time
        self.exit_price = exit_price
        self.exit_reason = exit_reason
        self.bars_held = (exit_time - self.entry_time).total_seconds() / 60  # Approximate
        
        # Calculate gross PnL
        if self.direction == 'LONG':
            pnl_points = self.exit_price - self.entry_price
        else:  # SHORT
            pnl_points = self.entry_price - self.exit_price
        
        self.pnl_gross = pnl_points * self.position_size
        
        # Calculate costs
        self.spread_cost = spread_r * self.risk_amount
        self.slippage_cost = slippage_r * self.risk_amount
        self.total_cost = self.spread_cost + self.slippage_cost
        
        # Net PnL
        self.pnl_net = self.pnl_gross - self.total_cost
        
        # R-multiples
        self.r_multiple_gross = self.pnl_gross / self.risk_amount if self.risk_amount > 0 else 0
        self.r_multiple_net = self.pnl_net / self.risk_amount if self.risk_amount > 0 else 0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for logging."""
        return {
            'entry_time': self.entry_time,
            'exit_time': self.exit_time,
            'direction': self.direction,
            'entry_price': self.entry_price,
            'exit_price': self.exit_price,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'position_size': self.position_size,
            'bars_held': self.bars_held,
            'exit_reason': self.exit_reason,
            'pnl_gross': self.pnl_gross,
            'pnl_net': self.pnl_net,
            'r_multiple_gross': self.r_multiple_gross,
            'r_multiple_net': self.r_multiple_net,
            'spread_cost': self.spread_cost,
            'slippage_cost': self.slippage_cost,
            'total_cost': self.total_cost,
            'confidence': self.confidence,
            'atr_at_entry': self.atr_at_entry
        }


# ═══════════════════════════════════════════════════════════════════════════
# POSITION SIZER
# ═══════════════════════════════════════════════════════════════════════════

class PositionSizer:
    """Calculate position sizes based on risk."""
    
    @staticmethod
    def calculate_position_size(account_value: float, risk_per_trade: float,
                               entry_price: float, stop_loss: float,
                               max_position_pct: float = 0.05) -> Tuple[float, float]:
        """
        Calculate position size using fixed fractional risk.
        
        Returns:
            position_size: Number of units to trade
            risk_amount: Dollar amount at risk
        """
        # Dollar amount to risk
        risk_amount = account_value * risk_per_trade
        
        # Price distance to stop loss
        risk_per_unit = abs(entry_price - stop_loss)
        
        if risk_per_unit == 0:
            return 0.0, 0.0
        
        # Position size based on risk
        position_size = risk_amount / risk_per_unit
        
        # Apply maximum position size limit
        max_position_value = account_value * max_position_pct
        max_units = max_position_value / entry_price
        
        position_size = min(position_size, max_units)
        
        # Recalculate actual risk amount
        actual_risk = position_size * risk_per_unit
        
        return position_size, actual_risk


# ═══════════════════════════════════════════════════════════════════════════
# SLIPPAGE MODEL
# ═══════════════════════════════════════════════════════════════════════════

class SlippageModel:
    """Realistic slippage modeling."""
    
    @staticmethod
    def calculate_slippage(atr: float, volume: float, avg_volume: float,
                          base_slippage_r: float = 0.02,
                          volume_impact: float = 0.01) -> float:
        """
        Calculate slippage in R-multiples.
        
        Args:
            atr: Current ATR
            volume: Current bar volume
            avg_volume: Average volume
            base_slippage_r: Base slippage cost
            volume_impact: Additional cost for low volume
        
        Returns:
            slippage_r: Slippage in R-multiples
        """
        # Base slippage
        slippage = base_slippage_r
        
        # Increase slippage during low volume
        if avg_volume > 0:
            volume_ratio = volume / avg_volume
            if volume_ratio < 0.5:  # Low volume
                slippage += volume_impact * (1 - volume_ratio)
        
        return slippage


# ═══════════════════════════════════════════════════════════════════════════
# BACKTEST ENGINE
# ═══════════════════════════════════════════════════════════════════════════

class BacktestEngine:
    """Core backtesting engine with realistic execution."""
    
    def __init__(self, symbol: str, timeframe: str, 
                 initial_capital: float = 100000.0,
                 risk_per_trade: float = 0.01):
        self.symbol = symbol
        self.timeframe = timeframe
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        
        # State
        self.account_value = initial_capital
        self.peak_value = initial_capital
        self.current_drawdown = 0.0
        
        # Trade tracking
        self.trades: List[Trade] = []
        self.open_trade: Optional[Trade] = None
        
        # Equity curve
        self.equity_curve = []
        self.timestamps = []
        
        # Daily tracking
        self.daily_pnl = defaultdict(float)
        self.current_date = None
        
        # Risk limits
        self.trading_stopped = False
        self.stop_reason = None
        
        # Costs
        self.spread_r = CONFIG.SPREAD_R_BY_TIMEFRAME.get(timeframe, 0.05)
        
    def check_risk_limits(self, current_time: pd.Timestamp) -> bool:
        """Check if risk limits are breached."""
        # Check drawdown limit
        self.current_drawdown = (self.peak_value - self.account_value) / self.peak_value
        if self.current_drawdown > CONFIG.MAX_DRAWDOWN_STOP:
            self.trading_stopped = True
            self.stop_reason = f"Max DD exceeded: {self.current_drawdown:.1%}"
            return False
        
        # Check daily loss limit
        current_date = current_time.date()
        if self.daily_pnl[current_date] < -self.initial_capital * CONFIG.MAX_DAILY_LOSS:
            return False  # Stop trading for the day
        
        return True
    
    def update_equity(self, current_time: pd.Timestamp):
        """Update equity curve."""
        self.equity_curve.append(self.account_value)
        self.timestamps.append(current_time)
        
        # Update peak
        if self.account_value > self.peak_value:
            self.peak_value = self.account_value
    
    def open_position(self, current_time: pd.Timestamp, entry_price: float,
                     stop_loss: float, take_profit: float, atr: float,
                     confidence: float, bar_index: int, direction: str = 'LONG'):
        """Open a new position."""
        if self.open_trade is not None:
            return  # Already in a trade
        
        if self.trading_stopped:
            return  # Trading stopped due to risk limits
        
        if not self.check_risk_limits(current_time):
            return  # Risk limits breached
        
        # Calculate position size
        position_size, risk_amount = PositionSizer.calculate_position_size(
            self.account_value, self.risk_per_trade, entry_price, stop_loss,
            CONFIG.MAX_POSITION_SIZE
        )
        
        if position_size == 0:
            return
        
        # Create trade
        self.open_trade = Trade(
            entry_time=current_time,
            entry_price=entry_price,
            direction=direction,
            position_size=position_size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            atr_at_entry=atr,
            risk_amount=risk_amount,
            confidence=confidence,
            bar_index=bar_index
        )
    
    def update_open_trade(self, current_time: pd.Timestamp, high: float, low: float,
                         close: float, volume: float, avg_volume: float,
                         bar_index: int, time_barrier: int):
        """Update open trade and check exit conditions."""
        if self.open_trade is None:
            return
        
        trade = self.open_trade
        exit_price = None
        exit_reason = None
        
        # Check TP/SL
        if trade.direction == 'LONG':
            if high >= trade.take_profit:
                exit_price = trade.take_profit
                exit_reason = 'TP'
            elif low <= trade.stop_loss:
                exit_price = trade.stop_loss
                exit_reason = 'SL'
        else:  # SHORT
            if low <= trade.take_profit:
                exit_price = trade.take_profit
                exit_reason = 'TP'
            elif high >= trade.stop_loss:
                exit_price = trade.stop_loss
                exit_reason = 'SL'
        
        # Check time barrier
        if exit_price is None and CONFIG.USE_TIME_BARRIER:
            bars_elapsed = bar_index - trade.bar_index
            if bars_elapsed >= time_barrier:
                exit_price = close
                exit_reason = 'TIME'
        
        # Execute exit
        if exit_price is not None:
            # Calculate slippage
            slippage_r = SlippageModel.calculate_slippage(
                trade.atr_at_entry, volume, avg_volume,
                CONFIG.BASE_SLIPPAGE_R, CONFIG.VOLUME_IMPACT_FACTOR
            )
            
            # Close trade
            trade.close_trade(current_time, exit_price, exit_reason,
                            self.spread_r, slippage_r)
            
            # Update account
            self.account_value += trade.pnl_net
            
            # Track daily PnL
            current_date = current_time.date()
            self.daily_pnl[current_date] += trade.pnl_net
            
            # Store trade
            self.trades.append(trade)
            self.open_trade = None
    
    def run_backtest(self, df: pd.DataFrame, model_dict: Dict, 
                    feature_cols: List[str], optimal_threshold: float,
                    tp_mult: float, sl_mult: float, time_barrier: int) -> Dict:
        """
        Execute complete backtest.
        
        Args:
            df: DataFrame with OHLCV, features, and ATR
            model_dict: Trained model + scaler
            feature_cols: Feature column names
            optimal_threshold: Confidence threshold
            tp_mult: Take profit multiplier
            sl_mult: Stop loss multiplier
            time_barrier: Time barrier in bars
        
        Returns:
            results: Dict with performance metrics
        """
        print(f"\n🔄 RUNNING BACKTEST: {self.symbol} {self.timeframe}")
        print(f"{'='*80}")
        print(f"   Initial Capital: ${self.initial_capital:,.2f}")
        print(f"   Risk per Trade: {self.risk_per_trade:.1%}")
        print(f"   Spread Cost: {self.spread_r:.3f}R")
        print(f"   Time Barrier: {time_barrier} bars")
        
        # Load model
        model = model_dict['model']
        scaler = model_dict['scaler']
        
        # Prepare features - handle missing columns
        missing_features = [col for col in feature_cols if col not in df.columns]
        if missing_features:
            print(f"   ⚠️  Warning: {len(missing_features)} features missing, filling with 0.0")
            for col in missing_features:
                df[col] = 0.0
        
        # Get available features (in case some were added)
        available_features = [col for col in feature_cols if col in df.columns]
        if len(available_features) != len(feature_cols):
            print(f"   ⚠️  Warning: Only {len(available_features)}/{len(feature_cols)} features available")
        
        # Extract features in the correct order
        X = df[feature_cols].values
        
        # Fill NaN values with 0
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Scale features
        X_scaled = scaler.transform(X)
        
        # Get predictions
        y_proba = model.predict_proba(X_scaled)[:, 1]
        
        # Calculate rolling average volume
        avg_volume = df['volume'].rolling(50).mean()
        
        # Main backtest loop
        for i in range(len(df)):
            if i % 5000 == 0 and i > 0:
                print(f"   Progress: {i:,}/{len(df):,} ({i/len(df)*100:.1f}%)", end='\r', flush=True)
            
            current_time = df['timestamp'].iloc[i]
            current_price = df['close'].iloc[i]
            high = df['high'].iloc[i]
            low = df['low'].iloc[i]
            atr = df['atr'].iloc[i]
            volume = df['volume'].iloc[i]
            avg_vol = avg_volume.iloc[i] if not pd.isna(avg_volume.iloc[i]) else volume
            confidence = y_proba[i]
            
            # Update equity curve
            self.update_equity(current_time)
            
            # Update open trade
            if self.open_trade is not None:
                self.update_open_trade(
                    current_time, high, low, current_price,
                    volume, avg_vol, i, time_barrier
                )
            
            # Check for new signal
            # Note: optimal_threshold may be adjusted for more trades
            if self.open_trade is None and confidence >= optimal_threshold:
                # Entry on next bar's open (to avoid lookahead bias)
                if i + 1 < len(df):
                    entry_price = df['open'].iloc[i + 1]
                    stop_loss = entry_price - (sl_mult * atr)
                    take_profit = entry_price + (tp_mult * atr)
                    
                    # Open position (only LONG for now)
                    self.open_position(
                        df['timestamp'].iloc[i + 1], entry_price,
                        stop_loss, take_profit, atr,
                        confidence, i + 1, direction='LONG'
                    )
        
        print(f"\n")
        
        # Close any remaining open trade
        if self.open_trade is not None:
            last_time = df['timestamp'].iloc[-1]
            last_price = df['close'].iloc[-1]
            last_volume = df['volume'].iloc[-1]
            last_avg_vol = avg_volume.iloc[-1]
            
            slippage_r = SlippageModel.calculate_slippage(
                self.open_trade.atr_at_entry, last_volume, last_avg_vol
            )
            
            self.open_trade.close_trade(
                last_time, last_price, 'END', self.spread_r, slippage_r
            )
            
            self.account_value += self.open_trade.pnl_net
            self.trades.append(self.open_trade)
            self.open_trade = None
        
        # Calculate metrics
        results = self.calculate_metrics(df)
        
        return results
    
    def calculate_metrics(self, df: pd.DataFrame) -> Dict:
        """Calculate comprehensive performance metrics."""
        if len(self.trades) == 0:
            return {'total_trades': 0}
        
        # Convert trades to DataFrame
        trades_df = pd.DataFrame([t.to_dict() for t in self.trades])
        
        # Basic stats
        total_trades = len(self.trades)
        winning_trades = len(trades_df[trades_df['pnl_net'] > 0])
        losing_trades = len(trades_df[trades_df['pnl_net'] < 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # PnL stats
        total_pnl_gross = trades_df['pnl_gross'].sum()
        total_pnl_net = trades_df['pnl_net'].sum()
        total_costs = trades_df['total_cost'].sum()
        
        # R-multiples
        r_multiples = trades_df['r_multiple_net'].values
        mean_r = r_multiples.mean()
        median_r = np.median(r_multiples)
        
        # Profit factor
        gross_profit = trades_df[trades_df['pnl_net'] > 0]['pnl_net'].sum()
        gross_loss = abs(trades_df[trades_df['pnl_net'] < 0]['pnl_net'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
        
        # Sharpe ratio
        sharpe = mean_r / r_multiples.std() if r_multiples.std() > 0 else 0
        
        # Drawdown
        equity_curve = np.array(self.equity_curve)
        peaks = np.maximum.accumulate(equity_curve)
        drawdowns = (peaks - equity_curve) / peaks * 100
        max_drawdown_pct = drawdowns.max()
        
        # Returns
        total_return_pct = (self.account_value - self.initial_capital) / self.initial_capital * 100
        
        # Time-based metrics
        start_date = df['timestamp'].iloc[0]
        end_date = df['timestamp'].iloc[-1]
        days_elapsed = (end_date - start_date).total_seconds() / 86400
        
        trades_per_day = total_trades / days_elapsed if days_elapsed > 0 else 0
        
        # Average holding period
        avg_bars_held = trades_df['bars_held'].mean()
        
        # Consecutive wins/losses
        win_streak = 0
        loss_streak = 0
        max_win_streak = 0
        max_loss_streak = 0
        
        for pnl in trades_df['pnl_net']:
            if pnl > 0:
                win_streak += 1
                loss_streak = 0
                max_win_streak = max(max_win_streak, win_streak)
            else:
                loss_streak += 1
                win_streak = 0
                max_loss_streak = max(max_loss_streak, loss_streak)
        
        return {
            # Trade counts
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            
            # PnL
            'total_pnl_net': total_pnl_net,
            'total_pnl_gross': total_pnl_gross,
            'total_costs': total_costs,
            'final_capital': self.account_value,
            'total_return_pct': total_return_pct,
            
            # R-multiples
            'mean_r': mean_r,
            'median_r': median_r,
            'profit_factor': profit_factor,
            'sharpe': sharpe,
            
            # Risk metrics
            'max_drawdown_pct': max_drawdown_pct,
            
            # Time metrics
            'start_date': start_date,
            'end_date': end_date,
            'days_elapsed': days_elapsed,
            'trades_per_day': trades_per_day,
            'avg_bars_held': avg_bars_held,
            
            # Streaks
            'max_win_streak': max_win_streak,
            'max_loss_streak': max_loss_streak,
            
            # Equity curve
            'equity_curve': self.equity_curve,
            'timestamps': self.timestamps,
            
            # Trades
            'trades_df': trades_df
        }


# ═══════════════════════════════════════════════════════════════════════════
# PERFORMANCE ANALYZER
# ═══════════════════════════════════════════════════════════════════════════

class PerformanceAnalyzer:
    """Analyze backtest results."""
    
    @staticmethod
    def print_summary(results: Dict, symbol: str, timeframe: str):
        """Print performance summary."""
        print(f"\n{'='*80}")
        print(f"BACKTEST RESULTS: {symbol} {timeframe}")
        print(f"{'='*80}")
        
        if results.get('total_trades', 0) == 0:
            print(f"❌ NO TRADES EXECUTED")
            return
        
        print(f"\n📊 TRADE STATISTICS")
        print(f"   Total Trades: {results['total_trades']:,}")
        print(f"   Winning Trades: {results['winning_trades']:,}")
        print(f"   Losing Trades: {results['losing_trades']:,}")
        print(f"   Win Rate: {results['win_rate']:.1%}")
        print(f"   Profit Factor: {results['profit_factor']:.2f}")
        print(f"   Sharpe Ratio: {results['sharpe']:.2f}")
        
        print(f"\n💰 FINANCIAL PERFORMANCE")
        print(f"   Initial Capital: ${results.get('initial_capital', 100000):,.2f}")
        print(f"   Final Capital: ${results['final_capital']:,.2f}")
        print(f"   Total Return: {results['total_return_pct']:.2f}%")
        print(f"   Total PnL (Net): ${results['total_pnl_net']:,.2f}")
        print(f"   Total PnL (Gross): ${results['total_pnl_gross']:,.2f}")
        print(f"   Total Costs: ${results['total_costs']:,.2f}")
        
        print(f"\n📈 R-MULTIPLE ANALYSIS")
        print(f"   Mean R: {results['mean_r']:.2f}")
        print(f"   Median R: {results['median_r']:.2f}")
        
        print(f"\n⚠️  RISK METRICS")
        print(f"   Max Drawdown: {results['max_drawdown_pct']:.2f}%")
        
        print(f"\n⏱️  TIME ANALYSIS")
        start_str = results['start_date'].strftime('%Y-%m-%d %H:%M:%S') if hasattr(results['start_date'], 'strftime') else str(results['start_date'])
        end_str = results['end_date'].strftime('%Y-%m-%d %H:%M:%S') if hasattr(results['end_date'], 'strftime') else str(results['end_date'])
        print(f"   Period: {start_str} to {end_str}")
        print(f"   Days: {results['days_elapsed']:.1f}")
        print(f"   Trades/Day: {results['trades_per_day']:.2f}")
        print(f"   Avg Bars Held: {results['avg_bars_held']:.1f}")
        
        print(f"\n🎯 STREAKS")
        print(f"   Max Win Streak: {results['max_win_streak']}")
        print(f"   Max Loss Streak: {results['max_loss_streak']}")
    
    @staticmethod
    def plot_equity_curve(results: Dict, symbol: str, timeframe: str, save_dir: Path):
        """Plot equity curve."""
        if results.get('total_trades', 0) == 0:
            return
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Equity curve
        equity = np.array(results['equity_curve'])
        timestamps = results['timestamps']
        
        axes[0].plot(timestamps, equity, linewidth=2, color='#2E86AB')
        axes[0].axhline(y=results.get('initial_capital', 100000), 
                       color='gray', linestyle='--', alpha=0.5, label='Initial Capital')
        axes[0].set_title(f'Equity Curve - {symbol} {timeframe}', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Account Value ($)', fontsize=12)
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        # Underwater plot (drawdown)
        peaks = np.maximum.accumulate(equity)
        drawdowns = (peaks - equity) / peaks * 100
        
        axes[1].fill_between(timestamps, 0, -drawdowns, color='#A23B72', alpha=0.7)
        axes[1].set_title('Underwater Plot (Drawdown)', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Drawdown (%)', fontsize=12)
        axes[1].set_xlabel('Date', fontsize=12)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        save_path = save_dir / f"{symbol}_{timeframe}_equity_curve.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   💾 Saved equity curve: {save_path}")
    
    @staticmethod
    def plot_trade_analysis(results: Dict, symbol: str, timeframe: str, save_dir: Path):
        """Plot trade analysis."""
        if results.get('total_trades', 0) == 0:
            return
        
        trades_df = results['trades_df']
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. R-multiple distribution
        axes[0, 0].hist(trades_df['r_multiple_net'], bins=50, 
                       color='#2E86AB', alpha=0.7, edgecolor='black')
        axes[0, 0].axvline(x=0, color='red', linestyle='--', linewidth=2)
        axes[0, 0].axvline(x=trades_df['r_multiple_net'].mean(),
                            color='green', linestyle='--', linewidth=2, label='Mean')
        axes[0, 0].set_title('R-Multiple Distribution', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('R-Multiple')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Exit reason breakdown
        exit_reasons = trades_df['exit_reason'].value_counts()
        colors = {'TP': '#27AE60', 'SL': '#E74C3C', 'TIME': '#F39C12', 'END': '#95A5A6'}
        axes[0, 1].pie(exit_reasons.values, labels=exit_reasons.index, autopct='%1.1f%%',
                      colors=[colors.get(r, '#95A5A6') for r in exit_reasons.index],
                      startangle=90)
        axes[0, 1].set_title('Exit Reason Breakdown', fontsize=12, fontweight='bold')
        
        # 3. Cumulative R-curve
        cumulative_r = trades_df['r_multiple_net'].cumsum()
        axes[1, 0].plot(cumulative_r.values, linewidth=2, color='#2E86AB')
        axes[1, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        axes[1, 0].set_title('Cumulative R-Multiple', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Trade Number')
        axes[1, 0].set_ylabel('Cumulative R')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Trade PnL scatter
        colors_scatter = ['green' if pnl > 0 else 'red' for pnl in trades_df['pnl_net']]
        axes[1, 1].scatter(range(len(trades_df)), trades_df['pnl_net'], 
                          c=colors_scatter, alpha=0.6, edgecolors='black')
        axes[1, 1].axhline(y=0, color='black', linestyle='-', linewidth=1)
        axes[1, 1].set_title('Trade PnL Distribution', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Trade Number')
        axes[1, 1].set_ylabel('PnL ($)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        save_path = save_dir / f"{symbol}_{timeframe}_trade_analysis.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   💾 Saved trade analysis: {save_path}")
    
    @staticmethod
    def export_trades(results: Dict, symbol: str, timeframe: str, save_dir: Path):
        """Export trades to CSV."""
        if results.get('total_trades', 0) == 0:
            return
        
        trades_df = results['trades_df']
        save_path = save_dir / f"{symbol}_{timeframe}_trades.csv"
        trades_df.to_csv(save_path, index=False)
        
        print(f"   💾 Saved trades CSV: {save_path}")
    
    @staticmethod
    def run_simulation(results: Dict, n_runs: int = 1000, 
                      confidence_level: float = 0.95) -> Dict:
        """
        Run Monte Carlo simulation on trade results.
        
        Randomly resamples trade sequence to estimate distribution of outcomes.
        """
        if results.get('total_trades', 0) == 0:
            return {}
        
        print(f"\n🎲 MONTE CARLO SIMULATION ({n_runs} runs)")
        print(f"{'='*80}")
        
        trades_df = results['trades_df']
        r_multiples = trades_df['r_multiple_net'].values
        
        # Run simulations
        final_returns = []
        max_drawdowns = []
        
        for run in range(n_runs):
            # Resample trades with replacement
            resampled_r = np.random.choice(r_multiples, size=len(r_multiples), replace=True)
            
            # Calculate equity curve
            equity = np.zeros(len(resampled_r) + 1)
            equity[0] = 1.0
            
            for i, r in enumerate(resampled_r):
                pnl_fraction = r * 0.01  # Assuming 1% risk per trade
                equity[i + 1] = equity[i] * (1.0 + pnl_fraction)
            
            # Calculate metrics
            final_return = (equity[-1] - 1.0) * 100
            final_returns.append(final_return)
            
            peaks = np.maximum.accumulate(equity)
            dd = (peaks - equity) / peaks
            max_dd = dd.max() * 100
            max_drawdowns.append(max_dd)
        
        # Calculate statistics
        final_returns = np.array(final_returns)
        max_drawdowns = np.array(max_drawdowns)
        
        # Confidence intervals
        ci_lower = (1 - confidence_level) / 2
        ci_upper = 1 - ci_lower
        
        return_ci_lower = np.percentile(final_returns, ci_lower * 100)
        return_ci_upper = np.percentile(final_returns, ci_upper * 100)
        
        dd_ci_lower = np.percentile(max_drawdowns, ci_lower * 100)
        dd_ci_upper = np.percentile(max_drawdowns, ci_upper * 100)
        
        print(f"\n   📊 RETURN DISTRIBUTION")
        print(f"      Mean: {final_returns.mean():.2f}%")
        print(f"      Median: {np.median(final_returns):.2f}%")
        print(f"      Std Dev: {final_returns.std():.2f}%")
        print(f"      {confidence_level:.0%} CI: [{return_ci_lower:.2f}%, {return_ci_upper:.2f}%]")
        
        print(f"\n   ⚠️  DRAWDOWN DISTRIBUTION")
        print(f"      Mean: {max_drawdowns.mean():.2f}%")
        print(f"      Median: {np.median(max_drawdowns):.2f}%")
        print(f"      {confidence_level:.0%} CI: [{dd_ci_lower:.2f}%, {dd_ci_upper:.2f}%]")
        
        # Probability of positive return
        prob_positive = (final_returns > 0).sum() / n_runs
        print(f"\n   ✅ Probability of Positive Return: {prob_positive:.1%}")
        
        return {
            'final_returns': final_returns,
            'max_drawdowns': max_drawdowns,
            'return_ci': (return_ci_lower, return_ci_upper),
            'dd_ci': (dd_ci_lower, dd_ci_upper),
            'prob_positive': prob_positive
        }


# ═══════════════════════════════════════════════════════════════════════════
# BACKTEST RUNNER
# ═══════════════════════════════════════════════════════════════════════════

class BacktestRunner:
    """Main backtest runner."""
    
    def __init__(self, symbol: str, timeframe: str, 
                 enable_monte_carlo: bool = False,
                 more_trades: bool = False,
                 min_trades_per_year: Optional[int] = None):
        self.symbol = symbol
        self.timeframe = timeframe
        self.enable_monte_carlo = enable_monte_carlo
        self.more_trades = more_trades
        self.min_trades_per_year = min_trades_per_year
        
        # Create results directory
        self.results_dir = CONFIG.RESULTS_DIR / self.symbol
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def load_model_and_metadata(self) -> Tuple[Dict, Dict]:
        """Load trained model and metadata."""
        models_dir = CONFIG.MODEL_STORE / self.symbol
        
        # Load model
        model_path = models_dir / f"{self.symbol}_{self.timeframe}_best_model.pkl"
        if not model_path.exists():
            raise FileNotFoundError(
                f"❌ Model not found: {model_path}\n"
                f"   Run training first: python citadel_training_system_v2.py"
            )
        
        model_dict = joblib.load(model_path)
        print(f"✅ Loaded model: {model_path}")
        
        # Load metadata
        metadata_path = models_dir / f"{self.symbol}_{self.timeframe}_metadata.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        print(f"✅ Loaded metadata: {metadata_path}")
        
        # Load feature columns
        features_path = models_dir / f"{self.symbol}_{self.timeframe}_feature_cols.json"
        with open(features_path, 'r') as f:
            feature_cols = json.load(f)
        
        print(f"✅ Loaded feature columns: {len(feature_cols)} features")
        
        return model_dict, {'metadata': metadata, 'feature_cols': feature_cols}
    
    def load_data(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Load feature data and apply feature engineering.
        
        Args:
            start_date: Optional start date filter (YYYY-MM-DD format)
            end_date: Optional end date filter (YYYY-MM-DD format)
        """
        file_path = CONFIG.FEATURE_STORE / self.symbol / f"{self.symbol}_{self.timeframe}.parquet"
        
        if not file_path.exists():
            raise FileNotFoundError(f"❌ Feature file not found: {file_path}")
        
        df = pd.read_parquet(file_path)
        print(f"✅ Loaded data: {len(df):,} rows")
        
        # Handle timestamp in index
        if 'timestamp' not in df.columns:
            if df.index.name == 'timestamp' or isinstance(df.index, pd.DatetimeIndex):
                df = df.reset_index()
                if 'index' in df.columns and df.index.name != 'timestamp':
                    df = df.rename(columns={'index': 'timestamp'})
        
        # Ensure timestamp is datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            # Extract hour for session features
            df['hour'] = df['timestamp'].dt.hour
        elif df.index.name == 'timestamp':
            df.index = pd.to_datetime(df.index)
            df['hour'] = df.index.hour
        
        # Ensure we have OHLCV columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"❌ Missing required columns: {missing_cols}")
        
        # Calculate ATR if missing (required for feature engineering)
        if 'atr' not in df.columns:
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            df['atr'] = true_range.rolling(window=14).mean()
            print(f"   📊 Calculated ATR (missing from data)")
        
        # Apply feature engineering
        print(f"   🔧 Engineering features...")
        df = FeatureEngineer.engineer_all_features(df)
        print(f"   ✅ Feature engineering complete: {len(df.columns)} total columns")
        
        # Apply date filters if provided
        if start_date or end_date:
            original_len = len(df)
            if start_date:
                start_dt = pd.to_datetime(start_date)
                df = df[df['timestamp'] >= start_dt]
                print(f"   📅 Filtered from {start_date}: {len(df):,} rows remaining")
            if end_date:
                end_dt = pd.to_datetime(end_date)
                df = df[df['timestamp'] <= end_dt]
                print(f"   📅 Filtered to {end_date}: {len(df):,} rows remaining")
            if len(df) < original_len:
                print(f"   ✅ Date filter applied: {original_len:,} → {len(df):,} rows")
        
        # Sort by timestamp to ensure chronological order
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Print actual date range
        if len(df) > 0:
            actual_start = df['timestamp'].iloc[0]
            actual_end = df['timestamp'].iloc[-1]
            print(f"   📊 Date range: {actual_start.strftime('%Y-%m-%d %H:%M:%S')} to {actual_end.strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            raise ValueError("❌ No data remaining after date filtering!")
        
        return df
    
    def run(self, start_date: Optional[str] = None, end_date: Optional[str] = None):
        """
        Execute complete backtest.
        
        Args:
            start_date: Optional start date filter (YYYY-MM-DD format)
            end_date: Optional end date filter (YYYY-MM-DD format)
        """
        print(f"\n{'#'*80}")
        print(f"# CITADEL BACKTESTING SYSTEM V2.4")
        print(f"# Symbol: {self.symbol} | Timeframe: {self.timeframe}")
        if start_date or end_date:
            date_range = f"{start_date or 'beginning'} to {end_date or 'end'}"
            print(f"# Date Range: {date_range}")
        print(f"# Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'#'*80}")
        
        # Load model and metadata
        model_dict, config = self.load_model_and_metadata()
        metadata = config['metadata']
        feature_cols = config['feature_cols']
        
        # Load data
        df = self.load_data(start_date, end_date)
        
        # Get training parameters (from metadata or defaults)
        optimal_threshold = metadata.get('optimal_threshold', 0.5)
        tp_mult = metadata.get('tp_mult', 2.0)
        sl_mult = metadata.get('sl_mult', 1.0)
        time_barrier = metadata.get('time_barrier', 50)
        
        # Apply threshold adjustments if requested
        original_threshold = optimal_threshold
        if self.more_trades:
            optimal_threshold = max(0.35, optimal_threshold - 0.15)
            print(f"\n⚠️  MORE TRADES MODE: Lowering confidence threshold")
            print(f"   Original: {original_threshold:.2f} → Adjusted: {optimal_threshold:.2f}")
        
        print(f"\n📋 BACKTEST PARAMETERS:")
        print(f"   Model: {metadata.get('best_model', 'Unknown')}")
        print(f"   Confidence Threshold: {optimal_threshold:.2f}")
        if original_threshold != optimal_threshold:
            print(f"   (Original threshold: {original_threshold:.2f})")
        print(f"   TP Multiplier: {tp_mult:.1f}x ATR")
        print(f"   SL Multiplier: {sl_mult:.1f}x ATR")
        print(f"   Time Barrier: {time_barrier} bars")
        
        # Initialize engine
        engine = BacktestEngine(
            self.symbol, self.timeframe,
            CONFIG.INITIAL_CAPITAL, CONFIG.RISK_PER_TRADE
        )
        
        # Run initial backtest
        results = engine.run_backtest(
            df, model_dict, feature_cols,
            optimal_threshold, tp_mult, sl_mult, time_barrier
        )
        
        # Check if we need to adjust threshold for minimum trades per year
        if self.min_trades_per_year and results.get('total_trades', 0) > 0:
            days = results.get('days_elapsed', 365)  # Default to 365 if not calculated yet
            trading_days_per_year = 252
            trades_per_day = results.get('trades_per_day', 0)
            projected_yearly = trades_per_day * trading_days_per_year if trades_per_day > 0 else results['total_trades']
            
            if projected_yearly < self.min_trades_per_year:
                print(f"\n⚠️  Trade frequency too low: {projected_yearly:.0f} trades/year (target: {self.min_trades_per_year})")
                print(f"   Adjusting confidence threshold to increase trades...")
                
                # Try progressively lower thresholds
                test_thresholds = [
                    max(0.35, optimal_threshold - 0.10),
                    max(0.35, optimal_threshold - 0.15),
                    max(0.35, optimal_threshold - 0.20),
                    0.35  # Minimum threshold
                ]
                
                best_results = results
                best_threshold = optimal_threshold
                
                for test_threshold in test_thresholds:
                    if test_threshold >= best_threshold:
                        continue
                    
                    test_results = engine.run_backtest(
                        df, model_dict, feature_cols,
                        test_threshold, tp_mult, sl_mult, time_barrier
                    )
                    
                    if test_results.get('total_trades', 0) > 0:
                        test_days = test_results.get('days_elapsed', days)
                        test_trades_per_day = test_results.get('trades_per_day', 0)
                        test_projected = test_trades_per_day * trading_days_per_year if test_trades_per_day > 0 else test_results['total_trades']
                        
                        # Check if this meets the target and has reasonable performance
                        if (test_projected >= self.min_trades_per_year and 
                            test_results.get('profit_factor', 0) > 1.0 and
                            test_results.get('win_rate', 0) > 0.40):
                            best_results = test_results
                            best_threshold = test_threshold
                            print(f"   ✅ Threshold {test_threshold:.2f}: {test_projected:.0f} trades/year, PF={test_results.get('profit_factor', 0):.2f}")
                            break
                
                if best_threshold != optimal_threshold:
                    optimal_threshold = best_threshold
                    results = best_results
                    print(f"   📊 Using threshold: {optimal_threshold:.2f} (was {original_threshold:.2f})")
                else:
                    print(f"   ⚠️  Could not reach target trades while maintaining profitability")
        
        results['initial_capital'] = CONFIG.INITIAL_CAPITAL
        
        # Print summary
        PerformanceAnalyzer.print_summary(results, self.symbol, self.timeframe)
        
        # Generate plots
        print(f"\n📊 GENERATING VISUALIZATIONS...")
        PerformanceAnalyzer.plot_equity_curve(results, self.symbol, self.timeframe, self.results_dir)
        PerformanceAnalyzer.plot_trade_analysis(results, self.symbol, self.timeframe, self.results_dir)
        
        # Export trades
        PerformanceAnalyzer.export_trades(results, self.symbol, self.timeframe, self.results_dir)
        
        # Monte Carlo simulation
        if self.enable_monte_carlo and results.get('total_trades', 0) > 0:
            mc_results = PerformanceAnalyzer.run_simulation(
                results, CONFIG.MONTE_CARLO_RUNS, CONFIG.CONFIDENCE_LEVEL
            )
        
        print(f"\n{'#'*80}")
        print(f"# BACKTEST COMPLETE")
        print(f"# Results saved to: {self.results_dir}")
        print(f"# Ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'#'*80}\n")
        
        return results


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Citadel Backtesting System V2.4')
    parser.add_argument('--symbol', type=str, required=True, help='Trading symbol (e.g., XAUUSD)')
    parser.add_argument('--timeframe', type=str, help='Timeframe (e.g., 5T, 15T, 30T, 1H)')
    parser.add_argument('--all-timeframes', action='store_true', help='Run all timeframes')
    parser.add_argument('--monte-carlo', action='store_true', help='Enable Monte Carlo simulation')
    parser.add_argument('--start-date', type=str, help='Start date filter (YYYY-MM-DD format, e.g., 2022-01-01)')
    parser.add_argument('--end-date', type=str, help='End date filter (YYYY-MM-DD format, e.g., 2024-12-31)')
    parser.add_argument('--past-year', action='store_true', 
                       help='Backtest only the past 365 days (last year) - useful for yearly returns analysis')
    parser.add_argument('--more-trades', action='store_true',
                       help='Lower confidence threshold by 0.15 to generate more trades')
    parser.add_argument('--min-trades-per-year', type=int, default=None,
                       help='Minimum trades per year target - automatically adjusts confidence threshold (e.g., 200)')
    
    args = parser.parse_args()
    
    # Handle --past-year flag
    if args.past_year:
        today = datetime.now()
        one_year_ago = today - timedelta(days=365)
        args.start_date = one_year_ago.strftime('%Y-%m-%d')
        args.end_date = today.strftime('%Y-%m-%d')
        print(f"\n📅 PAST YEAR MODE: Backtesting from {args.start_date} to {args.end_date}")
        print(f"   (Last 365 days for yearly returns analysis)\n")
    
    if args.all_timeframes:
        timeframes = ['5T', '15T', '30T', '1H']
    elif args.timeframe:
        timeframes = [args.timeframe]
    else:
        parser.print_help()
        return
    
    all_results = {}
    
    for timeframe in timeframes:
        try:
            runner = BacktestRunner(
                args.symbol, timeframe,
                enable_monte_carlo=args.monte_carlo,
                more_trades=args.more_trades,
                min_trades_per_year=args.min_trades_per_year
            )
            results = runner.run(
                start_date=args.start_date,
                end_date=args.end_date
            )
            all_results[timeframe] = results
            
        except Exception as e:
            print(f"\n❌ ERROR in {timeframe}: {e}")
            import traceback
            traceback.print_exc()
    
    # Multi-timeframe summary
    if len(all_results) > 1:
        print(f"\n{'='*80}")
        print(f"MULTI-TIMEFRAME SUMMARY")
        print(f"{'='*80}")
        
        print(f"\n{'TF':<6} {'Trades':>8} {'WR':>8} {'PF':>8} {'Return':>10} {'MaxDD':>10} {'Sharpe':>8}")
        print("-"*80)
        
        for tf, result in all_results.items():
            if result.get('total_trades', 0) > 0:
                print(f"{tf:<6} {result['total_trades']:>8,} "
                      f"{result['win_rate']:>7.1%} "
                      f"{result['profit_factor']:>8.2f} "
                      f"{result['total_return_pct']:>9.2f}% "
                      f"{result['max_drawdown_pct']:>9.2f}% "
                      f"{result['sharpe']:>8.2f}")
            else:
                print(f"{tf:<6} {'0':>8} {'N/A':>8} {'N/A':>8} {'N/A':>10} {'N/A':>10} {'N/A':>8}")
    
    print(f"\n✅ ALL BACKTESTS COMPLETE\n")


if __name__ == '__main__':
    main()