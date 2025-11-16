"""
Backtesting engine with regime-based strategy and ML filter.

Implements vectorized backtesting with realistic cost modeling,
position sizing, and risk management.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


@dataclass
class Trade:
    """Individual trade record."""
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    direction: str  # 'long' or 'short'
    entry_price: float
    exit_price: float
    stop_loss: float
    take_profit: float
    position_size: float
    pnl: float
    pnl_pct: float
    r_multiple: float
    exit_reason: str  # 'tp', 'sl', 'time', 'signal'
    regime: str
    ml_proba: float


class Backtest:
    """
    Regime-based backtest with ML filtering.

    Strategy logic:
    1. Detect regime for each bar
    2. If regime is tradable, generate candidate trade
    3. Query ML model for success probability
    4. If prob > threshold, execute trade
    5. Manage trade with SL/TP and position sizing
    """

    def __init__(
        self,
        df: pd.DataFrame,
        model: Any,
        config,
        symbol: str,
        timeframe: int
    ):
        """
        Initialize backtest.

        Args:
            df: DataFrame with features, regime, and OHLCV
            model: Trained ML model with predict_proba method
            config: Configuration object
            symbol: Trading symbol
            timeframe: Timeframe in minutes
        """
        self.df = df.copy()
        self.model = model
        self.config = config
        self.symbol = symbol
        self.timeframe = timeframe

        self.trades: List[Trade] = []
        self.equity_curve = []
        self.positions = []  # Active positions

        # Extract config parameters
        self.sl_atr_mult = config.strategy.sl_atr_multiplier
        self.tp_atr_mult = config.strategy.tp_atr_multiplier
        self.ml_threshold = config.strategy.ml_prob_threshold
        self.min_expected_r = config.strategy.min_expected_r

        self.risk_per_trade = config.risk.risk_per_trade_pct / 100.0
        self.max_positions = config.risk.max_positions_per_symbol
        self.initial_capital = config.risk.initial_capital

        # Costs
        self.spread = config.costs.spreads.get(symbol, 0.30)
        self.commission_per_lot = config.costs.commission_per_lot
        self.slippage_model = config.costs.slippage_model
        self.slippage_fixed = config.costs.slippage_fixed
        self.slippage_atr_pct = config.costs.slippage_atr_pct
        self.contract_size = config.costs.contract_sizes.get(symbol, 100.0)

    def calculate_position_size(
        self,
        equity: float,
        entry_price: float,
        stop_loss: float
    ) -> float:
        """
        Calculate position size based on fixed risk percentage.

        Args:
            equity: Current account equity
            entry_price: Intended entry price
            stop_loss: Stop loss price

        Returns:
            Position size in lots
        """
        risk_amount = equity * self.risk_per_trade
        risk_per_unit = abs(entry_price - stop_loss)

        if risk_per_unit == 0:
            return 0.0

        # Position size in base currency units
        position_size_units = risk_amount / risk_per_unit

        # Convert to lots
        position_size_lots = position_size_units / self.contract_size

        # Enforce minimum lot size
        min_lot = 0.01
        position_size_lots = max(min_lot, position_size_lots)

        return round(position_size_lots, 2)

    def calculate_transaction_costs(
        self,
        position_size: float,
        atr: float
    ) -> float:
        """
        Calculate total transaction costs (spread + commission + slippage).

        Args:
            position_size: Position size in lots
            atr: Current ATR value

        Returns:
            Total cost in account currency
        """
        # Spread cost
        spread_cost = self.spread * position_size * self.contract_size

        # Commission
        commission = self.commission_per_lot * position_size

        # Slippage
        if self.slippage_model == 'fixed':
            slippage_cost = self.slippage_fixed * position_size * self.contract_size
        else:  # atr_based
            slippage_cost = (atr * self.slippage_atr_pct) * position_size * self.contract_size

        total_cost = spread_cost + commission + slippage_cost

        return total_cost

    def get_ml_probability(self, idx: int) -> float:
        """
        Get ML model's predicted probability for this bar.

        Args:
            idx: Index in dataframe

        Returns:
            Probability of success
        """
        # Check if we already have predictions in df
        if 'ml_proba' in self.df.columns:
            return self.df['ml_proba'].iloc[idx]

        # Otherwise, predict on-the-fly
        from feature_engineering import get_feature_columns

        feature_cols = get_feature_columns(self.df)
        X = self.df[feature_cols].iloc[[idx]]

        proba = self.model.predict_proba(X)[0]

        return proba

    def generate_trade_signal(
        self,
        idx: int
    ) -> Optional[Dict[str, Any]]:
        """
        Generate trade signal based on regime and rules.

        Args:
            idx: Current bar index

        Returns:
            Trade signal dict or None
        """
        row = self.df.iloc[idx]
        regime = row['regime']

        # Skip non-tradable regimes
        if regime in ['event_vol', 'unknown']:
            return None

        # Need ATR for stops
        if 'atr_14' not in self.df.columns or pd.isna(row['atr_14']):
            return None

        atr = row['atr_14']
        close = row['close']

        signal = None

        # TREND UP: Long on pullbacks
        if regime == 'trend_up':
            # Check if we're near pullback level (e.g., 20 EMA)
            if 'ema_20' in self.df.columns and 'dist_to_ema20' in self.df.columns:
                dist_to_ema = row['dist_to_ema20']

                # Entry if within 0.5 ATR of EMA and price bouncing up
                pullback_tolerance = self.config.strategy.pullback_tolerance_atr * atr / close

                if abs(dist_to_ema) <= pullback_tolerance:
                    # Confirm momentum turning up (e.g., MACD positive or RSI > 50)
                    if 'rsi_14' in self.df.columns and row['rsi_14'] > 50:
                        signal = {
                            'direction': 'long',
                            'entry_price': close,
                            'stop_loss': close - self.sl_atr_mult * atr,
                            'take_profit': close + self.tp_atr_mult * atr,
                            'regime': regime
                        }

        # TREND DOWN: Short on pullbacks
        elif regime == 'trend_down':
            if 'ema_20' in self.df.columns and 'dist_to_ema20' in self.df.columns:
                dist_to_ema = row['dist_to_ema20']
                pullback_tolerance = self.config.strategy.pullback_tolerance_atr * atr / close

                if abs(dist_to_ema) <= pullback_tolerance:
                    if 'rsi_14' in self.df.columns and row['rsi_14'] < 50:
                        signal = {
                            'direction': 'short',
                            'entry_price': close,
                            'stop_loss': close + self.sl_atr_mult * atr,
                            'take_profit': close - self.tp_atr_mult * atr,
                            'regime': regime
                        }

        # RANGE: Mean reversion
        elif regime == 'range':
            if 'bb_position' in self.df.columns:
                bb_pos = row['bb_position']

                # Long at lower extreme
                if bb_pos < 0.1:
                    if 'rsi_14' in self.df.columns and row['rsi_14'] < self.config.strategy.rsi_oversold:
                        signal = {
                            'direction': 'long',
                            'entry_price': close,
                            'stop_loss': close - self.sl_atr_mult * atr,
                            'take_profit': close + self.tp_atr_mult * atr,
                            'regime': regime
                        }

                # Short at upper extreme
                elif bb_pos > 0.9:
                    if 'rsi_14' in self.df.columns and row['rsi_14'] > self.config.strategy.rsi_overbought:
                        signal = {
                            'direction': 'short',
                            'entry_price': close,
                            'stop_loss': close + self.sl_atr_mult * atr,
                            'take_profit': close - self.tp_atr_mult * atr,
                            'regime': regime
                        }

        return signal

    def check_ml_filter(self, idx: int, signal: Dict[str, Any]) -> bool:
        """
        Check if trade passes ML filter.

        Args:
            idx: Current bar index
            signal: Trade signal dict

        Returns:
            True if trade should be taken
        """
        ml_proba = self.get_ml_probability(idx)

        # Threshold check
        if ml_proba < self.ml_threshold:
            return False

        # Expected R check
        entry = signal['entry_price']
        sl = signal['stop_loss']
        tp = signal['take_profit']

        risk = abs(entry - sl)
        reward = abs(tp - entry)

        r_win = reward / risk if risk > 0 else 0
        r_loss = 1.0

        # Estimate cost in R units (approximate)
        atr = self.df['atr_14'].iloc[idx]
        est_cost_r = 0.15  # Conservative estimate

        expected_r = ml_proba * r_win - (1 - ml_proba) * r_loss - est_cost_r

        if expected_r < self.min_expected_r:
            return False

        return True

    def manage_position(
        self,
        position: Dict[str, Any],
        idx: int
    ) -> Optional[Trade]:
        """
        Manage open position, check for exits.

        Args:
            position: Position dict
            idx: Current bar index

        Returns:
            Completed Trade object if exited, else None
        """
        row = self.df.iloc[idx]
        high = row['high']
        low = row['low']
        close = row['close']

        direction = position['direction']
        entry_price = position['entry_price']
        sl = position['stop_loss']
        tp = position['take_profit']
        entry_idx = position['entry_idx']

        exit_price = None
        exit_reason = None

        # Check TP and SL
        if direction == 'long':
            # SL hit first (use low of bar)
            if low <= sl:
                exit_price = sl
                exit_reason = 'sl'
            # TP hit (use high of bar)
            elif high >= tp:
                exit_price = tp
                exit_reason = 'tp'

        else:  # short
            # SL hit first (use high of bar)
            if high >= sl:
                exit_price = sl
                exit_reason = 'sl'
            # TP hit (use low of bar)
            elif low <= tp:
                exit_price = tp
                exit_reason = 'tp'

        # Time-based exit (optional)
        if exit_price is None and self.config.strategy.max_bars_in_trade > 0:
            bars_in_trade = idx - entry_idx
            if bars_in_trade >= self.config.strategy.max_bars_in_trade:
                exit_price = close
                exit_reason = 'time'

        # If no exit, position remains open
        if exit_price is None:
            return None

        # Calculate PnL
        position_size = position['position_size']

        if direction == 'long':
            pnl_points = exit_price - entry_price
        else:
            pnl_points = entry_price - exit_price

        pnl_gross = pnl_points * position_size * self.contract_size

        # Deduct transaction costs (entry + exit)
        atr = self.df['atr_14'].iloc[idx]
        costs = 2 * self.calculate_transaction_costs(position_size, atr)  # Entry + exit

        pnl_net = pnl_gross - costs

        # R-multiple
        risk_amount = abs(entry_price - sl) * position_size * self.contract_size
        r_multiple = pnl_net / risk_amount if risk_amount > 0 else 0

        # Create Trade object
        trade = Trade(
            entry_time=self.df.index[entry_idx],
            exit_time=self.df.index[idx],
            direction=direction,
            entry_price=entry_price,
            exit_price=exit_price,
            stop_loss=sl,
            take_profit=tp,
            position_size=position_size,
            pnl=pnl_net,
            pnl_pct=(pnl_net / self.initial_capital) * 100,
            r_multiple=r_multiple,
            exit_reason=exit_reason,
            regime=position['regime'],
            ml_proba=position['ml_proba']
        )

        return trade

    def run(self) -> pd.DataFrame:
        """
        Run backtest.

        Returns:
            DataFrame with equity curve and trade log
        """
        equity = self.initial_capital
        self.equity_curve = [equity]
        self.trades = []
        self.positions = []

        # Iterate through bars
        for idx in range(len(self.df)):
            # Manage open positions
            closed_positions = []

            for pos_idx, position in enumerate(self.positions):
                completed_trade = self.manage_position(position, idx)

                if completed_trade:
                    self.trades.append(completed_trade)
                    equity += completed_trade.pnl
                    closed_positions.append(pos_idx)

            # Remove closed positions
            for pos_idx in reversed(closed_positions):
                self.positions.pop(pos_idx)

            # Check for new entry signals (only if not maxed out on positions)
            if len(self.positions) < self.max_positions:
                signal = self.generate_trade_signal(idx)

                if signal is not None:
                    # ML filter
                    if self.check_ml_filter(idx, signal):
                        # Calculate position size
                        position_size = self.calculate_position_size(
                            equity,
                            signal['entry_price'],
                            signal['stop_loss']
                        )

                        if position_size > 0:
                            # Open position
                            ml_proba = self.get_ml_probability(idx)

                            position = {
                                'entry_idx': idx,
                                'direction': signal['direction'],
                                'entry_price': signal['entry_price'],
                                'stop_loss': signal['stop_loss'],
                                'take_profit': signal['take_profit'],
                                'position_size': position_size,
                                'regime': signal['regime'],
                                'ml_proba': ml_proba
                            }

                            self.positions.append(position)

            # Record equity
            self.equity_curve.append(equity)

        # Close any remaining positions at end
        if len(self.positions) > 0:
            final_idx = len(self.df) - 1
            final_close = self.df['close'].iloc[final_idx]

            for position in self.positions:
                # Force close at final close price
                direction = position['direction']
                entry_price = position['entry_price']
                position_size = position['position_size']

                if direction == 'long':
                    pnl_points = final_close - entry_price
                else:
                    pnl_points = entry_price - final_close

                pnl_gross = pnl_points * position_size * self.contract_size

                atr = self.df['atr_14'].iloc[final_idx]
                costs = 2 * self.calculate_transaction_costs(position_size, atr)

                pnl_net = pnl_gross - costs

                sl = position['stop_loss']
                risk_amount = abs(entry_price - sl) * position_size * self.contract_size
                r_multiple = pnl_net / risk_amount if risk_amount > 0 else 0

                trade = Trade(
                    entry_time=self.df.index[position['entry_idx']],
                    exit_time=self.df.index[final_idx],
                    direction=direction,
                    entry_price=entry_price,
                    exit_price=final_close,
                    stop_loss=position['stop_loss'],
                    take_profit=position['take_profit'],
                    position_size=position_size,
                    pnl=pnl_net,
                    pnl_pct=(pnl_net / self.initial_capital) * 100,
                    r_multiple=r_multiple,
                    exit_reason='end_of_data',
                    regime=position['regime'],
                    ml_proba=position['ml_proba']
                )

                self.trades.append(trade)
                equity += pnl_net

            self.positions = []
            self.equity_curve[-1] = equity

        # Build results dataframe
        results_df = self.build_results_dataframe()

        return results_df

    def build_results_dataframe(self) -> pd.DataFrame:
        """Build DataFrame with backtest results."""
        # Equity curve
        equity_df = pd.DataFrame({
            'equity': self.equity_curve
        }, index=self.df.index.tolist() + [self.df.index[-1]])

        # Add equity to main df
        results = self.df.copy()
        results['equity'] = equity_df['equity'].iloc[:-1].values

        return results

    def get_trade_log(self) -> pd.DataFrame:
        """Get trade log as DataFrame."""
        if len(self.trades) == 0:
            return pd.DataFrame()

        trade_records = []
        for trade in self.trades:
            trade_records.append({
                'entry_time': trade.entry_time,
                'exit_time': trade.exit_time,
                'direction': trade.direction,
                'entry_price': trade.entry_price,
                'exit_price': trade.exit_price,
                'stop_loss': trade.stop_loss,
                'take_profit': trade.take_profit,
                'position_size': trade.position_size,
                'pnl': trade.pnl,
                'pnl_pct': trade.pnl_pct,
                'r_multiple': trade.r_multiple,
                'exit_reason': trade.exit_reason,
                'regime': trade.regime,
                'ml_proba': trade.ml_proba
            })

        trade_log = pd.DataFrame(trade_records)

        return trade_log


if __name__ == "__main__":
    from config import get_default_config
    from data_loader import load_all_data, create_sample_data
    from feature_engineering import build_features
    from model_training import TradingModel

    config = get_default_config()

    # Create and load sample data
    symbol = "XAUUSD"
    timeframe = 5

    file_path = config.data.data_dir / f"{symbol}_{timeframe}.csv"
    if not file_path.exists():
        df = create_sample_data(symbol, timeframe, n_bars=10000, save_path=str(file_path))

    all_data = load_all_data(config)

    # Build features
    key = (symbol, timeframe)
    if key in all_data:
        test_df = all_data[key]['test']
        test_features = build_features(test_df, config)

        print(f"Test data: {len(test_features)} bars")
        print(f"Regimes: {test_features['regime'].value_counts()}")

        # Create a dummy model (random predictions for testing)
        class DummyModel:
            def predict_proba(self, X):
                n = len(X)
                return np.column_stack([np.zeros(n), np.random.rand(n)])

        model = DummyModel()

        # Run backtest
        print("\nRunning backtest...")
        bt = Backtest(test_features, model, config, symbol, timeframe)
        results = bt.run()

        print(f"\nBacktest complete:")
        print(f"  Total trades: {len(bt.trades)}")
        print(f"  Final equity: ${results['equity'].iloc[-1]:,.2f}")

        trade_log = bt.get_trade_log()
        if len(trade_log) > 0:
            print(f"\nSample trades:")
            print(trade_log.head())
