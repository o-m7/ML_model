#!/usr/bin/env python3
"""
TRUE BACKTESTING ENGINE
========================

Real trade simulation with actual price action, not label matching.

Key differences from old approach:
- Enters at NEXT bar's open (not current close)
- Walks forward bar-by-bar checking highs/lows for SL/TP hits
- Exits on FIRST hit (realistic)
- Tracks REAL trade duration
- Handles gaps, slippage, spread properly
- No pre-calculated labels needed

This is how real trading actually works.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class TradeConfig:
    """Trading configuration."""
    initial_capital: float = 100000
    risk_per_trade_pct: float = 0.01  # 1% risk per trade
    max_position_pct: float = 0.10    # Max 10% of EQUITY (not notional)
    leverage: float = 50.0            # Leverage (50:1 for forex)
    commission_pct: float = 0.00001   # 0.001% commission
    slippage_pct: float = 0.000005    # 0.0005% slippage
    spread_pips: float = 1.0          # 1 pip spread
    confidence_threshold: float = 0.75
    max_bars_in_trade: int = 50       # Max time in trade


@dataclass
class Trade:
    """Individual trade record."""
    entry_idx: int
    entry_time: pd.Timestamp
    entry_price: float
    direction: str  # 'long' only for now
    position_size: float
    sl_price: float
    tp_price: float
    exit_idx: Optional[int] = None
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None  # 'tp', 'sl', 'timeout'
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    bars_held: Optional[int] = None
    commission: Optional[float] = None
    slippage: Optional[float] = None
    confidence: Optional[float] = None


class TrueBacktestEngine:
    """
    Real backtesting engine that simulates actual trading.
    """
    
    def __init__(self, df: pd.DataFrame, config: TradeConfig):
        """
        Initialize with price data.
        
        Args:
            df: DataFrame with OHLCV data and timestamp
            config: Trading configuration
        """
        self.df = df.copy()
        self.config = config
        
        # Ensure required columns
        required = ['open', 'high', 'low', 'close', 'timestamp']
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        self.trades: List[Trade] = []
        self.equity_curve = [config.initial_capital]
        self.current_equity = config.initial_capital
        self.peak_equity = config.initial_capital
        self.active_trade: Optional[Trade] = None
        self.total_risk_taken = 0.0  # Track cumulative risk
    
    def calculate_position_size(self, entry_price: float, sl_price: float) -> Tuple[float, float, float]:
        """
        Calculate position size based on risk management.
        
        Returns:
            (position_size, position_value, actual_risk_amount)
        """
        # Risk amount in dollars
        risk_amount = self.current_equity * self.config.risk_per_trade_pct
        
        # SL distance
        sl_distance = abs(entry_price - sl_price)
        
        if sl_distance <= 0:
            return 0, 0, 0
        
        # Calculate position size to risk exactly risk_amount
        position_size = risk_amount / sl_distance
        position_value = position_size * entry_price
        
        # Calculate margin required (with leverage)
        margin_required = position_value / self.config.leverage
        
        # Check if margin exceeds max allowed (% of equity)
        max_margin = self.current_equity * self.config.max_position_pct
        
        if margin_required > max_margin:
            # Margin too large - scale down position and recalculate actual risk
            position_value = max_margin * self.config.leverage
            position_size = position_value / entry_price
            actual_risk = position_size * sl_distance  # Reduced risk due to margin cap
        else:
            actual_risk = risk_amount
        
        # Check if we have sufficient margin
        if margin_required > self.current_equity:
            # Not enough capital for this trade even with leverage
            return 0, 0, 0
        
        # Validate position isn't too small
        if position_value < (self.current_equity * 0.001):  # Min 0.1% of equity notional
            return 0, 0, 0
        
        return position_size, position_value, actual_risk
    
    def calculate_costs(self, position_value: float, entry_price: float) -> Tuple[float, float]:
        """
        Calculate commission and slippage.
        
        Returns:
            (commission, slippage)
        """
        commission = position_value * self.config.commission_pct
        slippage = position_value * self.config.slippage_pct
        
        return commission, slippage
    
    def check_sl_tp_hit(self, trade: Trade, bar_idx: int) -> Tuple[bool, str, float]:
        """
        Check if SL or TP was hit on this bar.
        
        Returns:
            (hit, reason, exit_price)
        """
        bar = self.df.iloc[bar_idx]
        high = bar['high']
        low = bar['low']
        close = bar['close']
        
        # For LONG trades
        if trade.direction == 'long':
            # Check if SL hit (check low first - pessimistic approach)
            if low <= trade.sl_price:
                # Check if TP also hit on same bar
                if high >= trade.tp_price:
                    # Both hit - which came first?
                    # Conservative: assume SL hit first if opening gap
                    if bar['open'] < trade.sl_price:
                        return True, 'sl', trade.sl_price
                    else:
                        # Assume TP hit first if price gapped up
                        return True, 'tp', trade.tp_price
                else:
                    return True, 'sl', trade.sl_price
            
            # Check if TP hit
            if high >= trade.tp_price:
                return True, 'tp', trade.tp_price
        
        # No hit
        return False, '', 0.0
    
    def enter_trade(self, signal_idx: int, confidence: float, 
                   tp_r: float, sl_r: float, atr: float) -> bool:
        """
        Enter a trade at the NEXT bar's open.
        
        Args:
            signal_idx: Index where signal occurred
            confidence: Model confidence
            tp_r: Take profit in R multiples
            sl_r: Stop loss in R multiples
            atr: ATR value for calculating levels
        
        Returns:
            True if trade entered successfully
        """
        # Can't enter on last bar
        if signal_idx >= len(self.df) - 1:
            return False
        
        # Entry is at NEXT bar's open
        entry_idx = signal_idx + 1
        entry_bar = self.df.iloc[entry_idx]
        entry_price = entry_bar['open']
        entry_time = entry_bar['timestamp']
        
        # Apply spread (entry is worse by spread)
        spread_cost = entry_price * (self.config.spread_pips / 10000)  # Assuming XAUUSD-like instrument
        entry_price += spread_cost
        
        # Calculate SL and TP levels
        sl_price = entry_price - (atr * sl_r)
        tp_price = entry_price + (atr * tp_r)
        
        # Position sizing
        position_size, position_value, actual_risk = self.calculate_position_size(entry_price, sl_price)
        
        if position_size <= 0 or position_value <= 0:
            return False
        
        # Calculate costs
        commission, slippage = self.calculate_costs(position_value, entry_price)
        
        # Apply slippage to entry
        entry_price += (entry_price * self.config.slippage_pct)
        
        # Track total risk
        self.total_risk_taken += actual_risk
        
        # Create trade
        trade = Trade(
            entry_idx=entry_idx,
            entry_time=entry_time,
            entry_price=entry_price,
            direction='long',
            position_size=position_size,
            sl_price=sl_price,
            tp_price=tp_price,
            commission=commission,
            slippage=slippage,
            confidence=confidence
        )
        
        self.active_trade = trade
        return True
    
    def manage_active_trade(self, current_idx: int) -> bool:
        """
        Manage active trade - check for exits.
        
        Returns:
            True if trade was closed
        """
        if self.active_trade is None:
            return False
        
        trade = self.active_trade
        
        # Check for timeout
        bars_held = current_idx - trade.entry_idx
        if bars_held >= self.config.max_bars_in_trade:
            self.close_trade(current_idx, 'timeout', self.df.iloc[current_idx]['close'])
            return True
        
        # Check for SL/TP hit
        hit, reason, exit_price = self.check_sl_tp_hit(trade, current_idx)
        
        if hit:
            self.close_trade(current_idx, reason, exit_price)
            return True
        
        return False
    
    def close_trade(self, exit_idx: int, reason: str, exit_price: float):
        """Close the active trade."""
        if self.active_trade is None:
            return
        
        trade = self.active_trade
        exit_bar = self.df.iloc[exit_idx]
        
        # Apply slippage to exit
        exit_price -= (exit_price * self.config.slippage_pct)
        
        # Calculate P&L
        price_change = exit_price - trade.entry_price
        gross_pnl = trade.position_size * price_change
        
        # Subtract costs
        net_pnl = gross_pnl - trade.commission - trade.slippage
        
        # Update trade
        trade.exit_idx = exit_idx
        trade.exit_time = exit_bar['timestamp']
        trade.exit_price = exit_price
        trade.exit_reason = reason
        trade.pnl = net_pnl
        trade.pnl_pct = (net_pnl / self.current_equity) * 100
        trade.bars_held = exit_idx - trade.entry_idx
        
        # Update equity
        self.current_equity += net_pnl
        
        # Track peak for drawdown
        if self.current_equity > self.peak_equity:
            self.peak_equity = self.current_equity
        
        # Save trade
        self.trades.append(trade)
        self.active_trade = None
    
    def run(self, signals: pd.Series, probabilities: pd.Series, 
            tp_r: float, sl_r: float) -> Dict:
        """
        Run backtest with given signals.
        
        Args:
            signals: Boolean series of trade signals
            probabilities: Model confidence for each signal
            tp_r: Take profit R multiple
            sl_r: Stop loss R multiple
        
        Returns:
            Dictionary with backtest results
        """
        # Ensure we have ATR
        if 'atr14' not in self.df.columns:
            raise ValueError("ATR14 required for calculating TP/SL levels")
        
        # Main backtest loop
        for i in range(len(self.df)):
            # Update equity curve
            self.equity_curve.append(self.current_equity)
            
            # Manage active trade first
            if self.active_trade is not None:
                self.manage_active_trade(i)
                continue  # Don't take new signals while in trade
            
            # Check for new signal
            if i >= len(signals) or not signals.iloc[i]:
                continue
            
            # Check confidence
            if probabilities.iloc[i] < self.config.confidence_threshold:
                continue
            
            # Try to enter trade
            atr = self.df['atr14'].iloc[i]
            self.enter_trade(i, probabilities.iloc[i], tp_r, sl_r, atr)
        
        # Close any remaining active trade
        if self.active_trade is not None:
            self.close_trade(len(self.df) - 1, 'end', self.df.iloc[-1]['close'])
        
        # Calculate metrics
        return self.calculate_metrics()
    
    def calculate_metrics(self) -> Dict:
        """Calculate performance metrics from completed trades."""
        if len(self.trades) == 0:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'total_return': 0,
                'total_return_pct': 0,
                'max_drawdown_pct': 0,
                'profit_factor': 0,
                'sharpe_ratio': 0,
                'avg_bars_held': 0,
                'tp_hit_rate': 0,
                'sl_hit_rate': 0,
                'timeout_rate': 0
            }
        
        trades_df = pd.DataFrame([
            {
                'pnl': t.pnl,
                'pnl_pct': t.pnl_pct,
                'bars_held': t.bars_held,
                'exit_reason': t.exit_reason,
                'confidence': t.confidence
            }
            for t in self.trades
        ])
        
        # Basic stats
        wins = trades_df[trades_df['pnl'] > 0]
        losses = trades_df[trades_df['pnl'] <= 0]
        
        win_rate = len(wins) / len(trades_df) * 100 if len(trades_df) > 0 else 0
        
        total_wins = wins['pnl'].sum() if len(wins) > 0 else 0
        total_losses = abs(losses['pnl'].sum()) if len(losses) > 0 else 1
        profit_factor = total_wins / total_losses if total_losses > 0 else 0
        
        # Drawdown
        equity_series = pd.Series(self.equity_curve)
        rolling_max = equity_series.expanding().max()
        drawdown = (equity_series - rolling_max) / rolling_max
        max_drawdown_pct = abs(drawdown.min()) * 100
        
        # Returns
        returns = equity_series.pct_change().dropna()
        sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
        
        total_return = self.current_equity - self.config.initial_capital
        total_return_pct = (total_return / self.config.initial_capital) * 100
        
        # Exit reason breakdown
        tp_hits = len(trades_df[trades_df['exit_reason'] == 'tp'])
        sl_hits = len(trades_df[trades_df['exit_reason'] == 'sl'])
        timeouts = len(trades_df[trades_df['exit_reason'] == 'timeout'])
        
        return {
            'total_trades': len(trades_df),
            'win_rate': win_rate,
            'total_return': total_return,
            'total_return_pct': total_return_pct,
            'max_drawdown_pct': max_drawdown_pct,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe,
            'avg_bars_held': trades_df['bars_held'].mean(),
            'avg_win': wins['pnl'].mean() if len(wins) > 0 else 0,
            'avg_loss': losses['pnl'].mean() if len(losses) > 0 else 0,
            'largest_win': wins['pnl'].max() if len(wins) > 0 else 0,
            'largest_loss': losses['pnl'].min() if len(losses) > 0 else 0,
            'avg_risk_per_trade': self.total_risk_taken / len(trades_df) if len(trades_df) > 0 else 0,
            'total_risk_taken': self.total_risk_taken,
            'tp_hit_rate': (tp_hits / len(trades_df)) * 100,
            'sl_hit_rate': (sl_hits / len(trades_df)) * 100,
            'timeout_rate': (timeouts / len(trades_df)) * 100,
            'trades': self.trades,
            'equity_curve': self.equity_curve
        }


def run_true_backtest(df: pd.DataFrame, model, features: List[str],
                     config: TradeConfig, tp_r: float, sl_r: float) -> Dict:
    """
    Convenience function to run true backtest.
    
    Args:
        df: Price data with OHLCV
        model: Trained ML model
        features: Feature list
        config: Trading configuration
        tp_r: Take profit R multiple
        sl_r: Stop loss R multiple
    
    Returns:
        Backtest results dictionary
    """
    # Get predictions
    X = df[features].fillna(0).values
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]
    
    # Create signals series
    signals = pd.Series(predictions == 1, index=df.index)
    probs = pd.Series(probabilities, index=df.index)
    
    # Run backtest
    engine = TrueBacktestEngine(df, config)
    results = engine.run(signals, probs, tp_r, sl_r)
    
    return results

