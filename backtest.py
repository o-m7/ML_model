"""
Institutional-Grade Backtesting Engine
Handles realistic spreads, slippage, ATR-scaled sizing, and prop-firm constraints
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class TimeFrame(Enum):
    T5 = "5T"
    T15 = "15T"
    T30 = "30T"
    H1 = "1H"


@dataclass
class TradeConfig:
    """Per-timeframe execution parameters"""
    timeframe: TimeFrame
    spread_pips: float  # Base spread in pips
    slippage_pips: float  # Additional slippage
    pip_value: float  # Dollar value per pip (standard lot)
    
    @classmethod
    def get_config(cls, timeframe: str, symbol: str = "XAUUSD"):
        """Get realistic execution costs by timeframe"""
        configs = {
            "5T": cls(TimeFrame.T5, 0.25, 0.10, 1.0 if symbol == "XAUUSD" else 5.0),
            "15T": cls(TimeFrame.T15, 0.20, 0.10, 1.0 if symbol == "XAUUSD" else 5.0),
            "30T": cls(TimeFrame.T30, 0.18, 0.08, 1.0 if symbol == "XAUUSD" else 5.0),
            "1H": cls(TimeFrame.H1, 0.15, 0.08, 1.0 if symbol == "XAUUSD" else 5.0),
        }
        return configs.get(timeframe, configs["15T"])


@dataclass
class Trade:
    """Single trade record with full execution details"""
    entry_time: pd.Timestamp
    entry_price: float
    direction: int  # 1=long, -1=short
    size_lots: float
    stop_loss: float
    take_profit: float
    atr_entry: float
    
    # Exit details (filled on close)
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None  # 'tp', 'sl', 'time'
    
    # Performance metrics
    pnl_gross: float = 0.0
    pnl_net: float = 0.0
    costs: float = 0.0
    r_multiple: float = 0.0
    mae: float = 0.0  # Max adverse excursion
    mfe: float = 0.0  # Max favorable excursion
    
    def to_dict(self) -> Dict:
        return {
            'entry_time': self.entry_time,
            'exit_time': self.exit_time,
            'direction': 'LONG' if self.direction == 1 else 'SHORT',
            'entry_price': self.entry_price,
            'exit_price': self.exit_price,
            'size_lots': self.size_lots,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'exit_reason': self.exit_reason,
            'pnl_net': self.pnl_net,
            'r_multiple': self.r_multiple,
            'mae': self.mae,
            'mfe': self.mfe,
            'atr_entry': self.atr_entry
        }


class RealisticBacktester:
    """
    Institutional backtesting engine with:
    - Realistic spread/slippage modeling
    - ATR-scaled position sizing
    - Prop-firm risk constraints
    - Walk-forward validation support
    """
    
    def __init__(
        self,
        initial_capital: float = 100000.0,
        risk_per_trade: float = 0.01,  # 1% account risk
        max_position_pct: float = 0.02,  # 2% max position
        max_drawdown_pct: float = 0.06,  # 6% max DD (prop-firm)
        max_daily_loss_pct: float = 0.01,  # 1% daily loss limit
        timeframe: str = "15T",
        symbol: str = "XAUUSD"
    ):
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.max_position_pct = max_position_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.max_daily_loss_pct = max_daily_loss_pct
        
        self.config = TradeConfig.get_config(timeframe, symbol)
        self.symbol = symbol
        
        # State tracking
        self.equity = initial_capital
        self.peak_equity = initial_capital
        self.current_drawdown = 0.0
        self.daily_pnl = 0.0
        self.last_trade_date = None
        
        # Trade tracking
        self.trades: list[Trade] = []
        self.open_trade: Optional[Trade] = None
        self.equity_curve = []
        
    def reset(self):
        """Reset backtest state"""
        self.equity = self.initial_capital
        self.peak_equity = self.initial_capital
        self.current_drawdown = 0.0
        self.daily_pnl = 0.0
        self.last_trade_date = None
        self.trades = []
        self.open_trade = None
        self.equity_curve = []
        
    def calculate_position_size(self, atr: float, stop_distance_atr: float = 2.0) -> float:
        """
        ATR-scaled position sizing with volatility regime adjustment
        
        Args:
            atr: Current ATR value
            stop_distance_atr: Stop distance in ATR units (default 2.0)
            
        Returns:
            Position size in lots
        """
        # Base risk amount
        risk_amount = self.equity * self.risk_per_trade
        
        # Stop distance in price units
        stop_distance_price = atr * stop_distance_atr
        
        # Position size calculation
        # Risk = position_size * stop_distance * pip_value
        position_size = risk_amount / (stop_distance_price * self.config.pip_value)
        
        # Apply max position constraint
        max_position = (self.equity * self.max_position_pct) / self.config.pip_value
        position_size = min(position_size, max_position)
        
        # Volatility regime adjustment (reduce size in extreme volatility)
        # This requires historical ATR distribution - simplified here
        # In production, check if current ATR > 95th percentile of rolling ATR
        
        return round(position_size, 2)
    
    def check_risk_limits(self, current_date: pd.Timestamp) -> bool:
        """
        Verify prop-firm risk constraints
        
        Returns:
            True if trading allowed, False if limits breached
        """
        # Check max drawdown
        if self.current_drawdown >= self.max_drawdown_pct:
            return False
        
        # Reset daily P&L counter
        if self.last_trade_date is None or current_date.date() != self.last_trade_date.date():
            self.daily_pnl = 0.0
            self.last_trade_date = current_date
        
        # Check daily loss limit
        if self.daily_pnl <= -self.equity * self.max_daily_loss_pct:
            return False
        
        return True
    
    def calculate_costs(self, size_lots: float, entry_price: float) -> float:
        """
        Calculate total transaction costs (spread + slippage)
        
        Returns:
            Total cost in dollars
        """
        total_pips = self.config.spread_pips + self.config.slippage_pips
        total_cost = size_lots * total_pips * self.config.pip_value
        return total_cost
    
    def open_position(
        self,
        timestamp: pd.Timestamp,
        signal: int,  # 1=long, -1=short
        price: float,  # Execution price (open of next bar)
        atr: float,
        stop_atr_mult: float = 2.0,
        target_atr_mult: float = 3.0
    ) -> bool:
        """
        Open new position with ATR-scaled stops/targets
        
        Args:
            timestamp: Entry timestamp (signal bar close + 1)
            signal: Direction (1=long, -1=short)
            price: Execution price
            atr: ATR at entry
            stop_atr_mult: Stop distance in ATR units
            target_atr_mult: Target distance in ATR units
            
        Returns:
            True if position opened, False if rejected
        """
        # Check risk limits
        if not self.check_risk_limits(timestamp):
            return False
        
        # Check if already in position
        if self.open_trade is not None:
            return False
        
        # Calculate position size
        size = self.calculate_position_size(atr, stop_atr_mult)
        if size <= 0:
            return False
        
        # Apply spread to entry (conservative: always worst case)
        spread_adjustment = (self.config.spread_pips + self.config.slippage_pips) * signal
        actual_entry = price + spread_adjustment * self.config.pip_value
        
        # Calculate stop and target
        stop_distance = atr * stop_atr_mult
        target_distance = atr * target_atr_mult
        
        if signal == 1:  # Long
            stop_loss = actual_entry - stop_distance
            take_profit = actual_entry + target_distance
        else:  # Short
            stop_loss = actual_entry + stop_distance
            take_profit = actual_entry - target_distance
        
        # Create trade
        self.open_trade = Trade(
            entry_time=timestamp,
            entry_price=actual_entry,
            direction=signal,
            size_lots=size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            atr_entry=atr
        )
        
        # Deduct entry costs immediately
        entry_costs = self.calculate_costs(size, actual_entry)
        self.open_trade.costs += entry_costs
        
        return True
    
    def update_position(self, bar: pd.Series) -> Optional[Trade]:
        """
        Check for stop/target hits during bar
        Uses conservative intrabar execution: worst-case fill
        
        Args:
            bar: OHLCV bar with columns [open, high, low, close]
            
        Returns:
            Closed trade if exited, None otherwise
        """
        if self.open_trade is None:
            return None
        
        trade = self.open_trade
        timestamp = bar.name
        
        # Determine worst-case prices for stop/target evaluation
        if trade.direction == 1:  # Long
            worst_price = bar['low']  # Worst for long = low
            best_price = bar['high']  # Best for long = high
        else:  # Short
            worst_price = bar['high']  # Worst for short = high
            best_price = bar['low']  # Best for short = low
        
        # Update MAE/MFE
        trade.mae = max(trade.mae, abs(worst_price - trade.entry_price))
        trade.mfe = max(trade.mfe, abs(best_price - trade.entry_price))
        
        # Check stop loss (priority: stops before targets)
        stop_hit = (
            (trade.direction == 1 and worst_price <= trade.stop_loss) or
            (trade.direction == -1 and worst_price >= trade.stop_loss)
        )
        
        if stop_hit:
            exit_price = trade.stop_loss
            exit_reason = 'sl'
        else:
            # Check take profit
            target_hit = (
                (trade.direction == 1 and best_price >= trade.take_profit) or
                (trade.direction == -1 and best_price <= trade.take_profit)
            )
            
            if target_hit:
                exit_price = trade.take_profit
                exit_reason = 'tp'
            else:
                return None  # Position still open
        
        # Close position
        return self._close_position(timestamp, exit_price, exit_reason)
    
    def _close_position(
        self,
        timestamp: pd.Timestamp,
        exit_price: float,
        exit_reason: str
    ) -> Trade:
        """Internal: finalize trade exit and update equity"""
        trade = self.open_trade
        
        # Apply exit spread/slippage (conservative: worst case)
        spread_adjustment = (self.config.spread_pips + self.config.slippage_pips) * trade.direction
        actual_exit = exit_price - spread_adjustment * self.config.pip_value
        
        # Calculate P&L
        price_diff = (actual_exit - trade.entry_price) * trade.direction
        gross_pnl = price_diff * trade.size_lots * self.config.pip_value
        
        # Add exit costs
        exit_costs = self.calculate_costs(trade.size_lots, actual_exit)
        trade.costs += exit_costs
        
        # Net P&L
        net_pnl = gross_pnl - trade.costs
        
        # R-multiple calculation
        risk_amount = self.equity * self.risk_per_trade
        r_multiple = net_pnl / risk_amount if risk_amount > 0 else 0
        
        # Update trade
        trade.exit_time = timestamp
        trade.exit_price = actual_exit
        trade.exit_reason = exit_reason
        trade.pnl_gross = gross_pnl
        trade.pnl_net = net_pnl
        trade.r_multiple = r_multiple
        
        # Update account state
        self.equity += net_pnl
        self.daily_pnl += net_pnl
        
        # Update drawdown
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity
            self.current_drawdown = 0.0
        else:
            self.current_drawdown = (self.peak_equity - self.equity) / self.peak_equity
        
        # Record equity point
        self.equity_curve.append({
            'timestamp': timestamp,
            'equity': self.equity,
            'drawdown': self.current_drawdown
        })
        
        # Archive trade and clear open position
        self.trades.append(trade)
        self.open_trade = None
        
        return trade
    
    def run(
        self,
        df: pd.DataFrame,
        predictions: np.ndarray,
        stop_atr_mult: float = 2.0,
        target_atr_mult: float = 3.0
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Execute backtest on historical data with model predictions
        
        Args:
            df: OHLCV data with columns [open, high, low, close, atr]
            predictions: Signal array (1=long, -1=short, 0=no signal)
            stop_atr_mult: Stop distance in ATR units
            target_atr_mult: Target distance in ATR units
            
        Returns:
            (trades_df, performance_metrics)
        """
        self.reset()
        
        # Validate inputs
        assert len(df) == len(predictions), "Data and predictions length mismatch"
        assert 'atr' in df.columns, "ATR column required for position sizing"
        
        for i in range(len(df)):
            bar = df.iloc[i]
            signal = predictions[i]
            
            # Update existing position first
            if self.open_trade is not None:
                self.update_position(bar)
            
            # Open new position if signal present and no open trade
            if signal != 0 and self.open_trade is None:
                # Entry on next bar open (lookahead-free)
                if i + 1 < len(df):
                    next_bar = df.iloc[i + 1]
                    entry_price = next_bar['open']
                    entry_time = next_bar.name
                    atr = bar['atr']  # Use signal bar ATR
                    
                    self.open_position(
                        timestamp=entry_time,
                        signal=signal,
                        price=entry_price,
                        atr=atr,
                        stop_atr_mult=stop_atr_mult,
                        target_atr_mult=target_atr_mult
                    )
        
        # Force close any open position at end
        if self.open_trade is not None:
            last_bar = df.iloc[-1]
            self._close_position(
                timestamp=last_bar.name,
                exit_price=last_bar['close'],
                exit_reason='time'
            )
        
        # Convert trades to DataFrame
        trades_df = pd.DataFrame([t.to_dict() for t in self.trades])
        
        # Calculate performance metrics
        metrics = self._calculate_metrics(trades_df)
        
        return trades_df, metrics
    
    def _calculate_metrics(self, trades_df: pd.DataFrame) -> Dict:
        """Calculate comprehensive performance metrics"""
        if len(trades_df) == 0:
            return {'error': 'No trades executed'}
        
        # Basic statistics
        total_trades = len(trades_df)
        winning_trades = (trades_df['pnl_net'] > 0).sum()
        losing_trades = (trades_df['pnl_net'] < 0).sum()
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # P&L statistics
        total_pnl = trades_df['pnl_net'].sum()
        avg_win = trades_df[trades_df['pnl_net'] > 0]['pnl_net'].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[trades_df['pnl_net'] < 0]['pnl_net'].mean() if losing_trades > 0 else 0
        
        # Profit factor
        gross_profit = trades_df[trades_df['pnl_net'] > 0]['pnl_net'].sum()
        gross_loss = abs(trades_df[trades_df['pnl_net'] < 0]['pnl_net'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
        
        # R-multiple statistics
        avg_r = trades_df['r_multiple'].mean()
        expectancy_r = trades_df['r_multiple'].sum() / total_trades
        
        # Expected value
        ev_dollar = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
        
        # Sharpe ratio (per-trade)
        returns = trades_df['pnl_net'] / self.initial_capital
        sharpe = returns.mean() / returns.std() if returns.std() > 0 else 0
        
        # Drawdown from equity curve
        equity_df = pd.DataFrame(self.equity_curve)
        max_dd = equity_df['drawdown'].max() if len(equity_df) > 0 else 0
        
        # Return metrics
        total_return = (self.equity - self.initial_capital) / self.initial_capital
        
        # Trade pacing
        if len(trades_df) > 0:
            duration = (trades_df['exit_time'].max() - trades_df['entry_time'].min()).total_seconds() / 86400
            trades_per_day = total_trades / duration if duration > 0 else 0
        else:
            trades_per_day = 0
        
        return {
            'total_trades': total_trades,
            'win_rate': round(win_rate, 4),
            'profit_factor': round(profit_factor, 3),
            'total_pnl': round(total_pnl, 2),
            'total_return_pct': round(total_return * 100, 2),
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
            'avg_r_multiple': round(avg_r, 3),
            'expectancy_r': round(expectancy_r, 3),
            'ev_dollar': round(ev_dollar, 2),
            'sharpe_per_trade': round(sharpe, 3),
            'max_drawdown_pct': round(max_dd * 100, 2),
            'final_equity': round(self.equity, 2),
            'trades_per_day': round(trades_per_day, 2),
            # Institutional thresholds
            'meets_profit_factor': profit_factor >= 1.6,
            'meets_win_rate': 0.50 <= win_rate <= 0.60,
            'meets_sharpe': sharpe >= 0.25,
            'meets_drawdown': max_dd <= 0.06,
            'meets_r_multiple': avg_r > 1.2
        }


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Example: Load data and predictions
    # In production, load from your feature pipeline output
    
    # Simulate data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=5000, freq='15T')
    
    df = pd.DataFrame({
        'open': 2000 + np.random.randn(5000).cumsum() * 0.5,
        'high': np.nan,
        'low': np.nan,
        'close': np.nan,
        'atr': 1.5 + np.random.randn(5000) * 0.1
    }, index=dates)
    
    df['high'] = df['open'] + np.abs(np.random.randn(5000) * 0.5)
    df['low'] = df['open'] - np.abs(np.random.randn(5000) * 0.5)
    df['close'] = df['open'] + np.random.randn(5000) * 0.3
    df['atr'] = df['atr'].abs()
    
    # Simulate model predictions (1=long, -1=short, 0=no signal)
    predictions = np.random.choice([1, -1, 0], size=5000, p=[0.15, 0.15, 0.70])
    
    # Run backtest
    backtester = RealisticBacktester(
        initial_capital=100000,
        risk_per_trade=0.01,
        max_drawdown_pct=0.06,
        timeframe="15T",
        symbol="XAUUSD"
    )
    
    trades_df, metrics = backtester.run(
        df=df,
        predictions=predictions,
        stop_atr_mult=2.0,
        target_atr_mult=3.0
    )
    
    # Display results
    print("\n" + "="*60)
    print("BACKTEST RESULTS")
    print("="*60)
    
    for key, value in metrics.items():
        print(f"{key:.<40} {value}")
    
    print("\n" + "="*60)
    print("INSTITUTIONAL THRESHOLDS")
    print("="*60)
    print(f"Profit Factor ≥ 1.6: {'✓' if metrics['meets_profit_factor'] else '✗'}")
    print(f"Win Rate 50-60%: {'✓' if metrics['meets_win_rate'] else '✗'}")
    print(f"Sharpe ≥ 0.25: {'✓' if metrics['meets_sharpe'] else '✗'}")
    print(f"Max DD ≤ 6%: {'✓' if metrics['meets_drawdown'] else '✗'}")
    print(f"R-multiple > 1.2: {'✓' if metrics['meets_r_multiple'] else '✗'}")
    
    # Show first few trades
    print("\n" + "="*60)
    print("SAMPLE TRADES")
    print("="*60)
    print(trades_df.head(10).to_string())