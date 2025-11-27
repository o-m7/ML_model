import numpy as np
import pandas as pd
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class RiskManager:
    """
    Position sizing and execution filters for live trading.
    Implements ATR-scaled sizing, volatility adjustments, and filter rules.
    """
    
    def __init__(self, config: Dict):
        self.method = config['risk_management']['position_sizing']['method']
        self.risk_per_trade = config['risk_management']['position_sizing']['risk_per_trade_pct'] / 100
        self.kelly_cap = config['risk_management']['position_sizing'].get('kelly_fraction_cap', 0.25)
        
        self.min_confidence = config['risk_management']['execution_filters']['min_confidence']
        self.max_spread_atr = config['risk_management']['execution_filters']['max_spread_atr_ratio']
        self.max_vol_zscore = config['risk_management']['execution_filters']['max_volatility_zscore']
        self.blackout_sessions = config['risk_management']['execution_filters'].get('blackout_sessions', [])
    
    def calculate_position_size(self, account_value: float, atr: float, 
                               stop_loss_atr_multiple: float = 1.0,
                               realized_vol: float = None) -> float:
        """
        Calculate position size based on method.
        
        Args:
            account_value: Current account equity
            atr: Current ATR value
            stop_loss_atr_multiple: SL distance in ATR units
            realized_vol: Recent realized volatility (for vol-adjusted sizing)
        
        Returns:
            Position size in lots (or contracts)
        """
        if self.method == 'atr_scaled':
            # Risk fixed % of account
            risk_amount = account_value * self.risk_per_trade
            stop_loss_distance = atr * stop_loss_atr_multiple
            position_size = risk_amount / stop_loss_distance
        
        elif self.method == 'volatility_adjusted':
            # Scale inversely with volatility
            if realized_vol is None or realized_vol == 0:
                realized_vol = atr / 100  # Fallback to ATR
            
            risk_amount = account_value * self.risk_per_trade
            vol_adjustment = 1.0 / (realized_vol / 0.01)  # Normalize to 1% vol
            adjusted_risk = risk_amount * vol_adjustment
            
            stop_loss_distance = atr * stop_loss_atr_multiple
            position_size = adjusted_risk / stop_loss_distance
        
        elif self.method == 'kelly':
            # Kelly criterion (requires win rate and avg win/loss)
            # Placeholder - needs historical stats
            logger.warning("Kelly sizing requires historical win rate and avg R. Using ATR-scaled instead.")
            risk_amount = account_value * self.risk_per_trade
            stop_loss_distance = atr * stop_loss_atr_multiple
            position_size = risk_amount / stop_loss_distance
        
        else:
            raise ValueError(f"Unknown position sizing method: {self.method}")
        
        return position_size
    
    def apply_execution_filters(self, df: pd.DataFrame, 
                               predictions: np.ndarray,
                               prediction_proba: np.ndarray) -> np.ndarray:
        """
        Apply execution filters to predicted signals.
        
        Args:
            df: DataFrame with price/volume data
            predictions: Binary predictions {0, 1}
            prediction_proba: Prediction probabilities [0, 1]
        
        Returns:
            Filtered predictions (may convert some 1s to 0s)
        """
        filtered = predictions.copy()
        
        # Filter 1: Confidence threshold
        low_confidence = prediction_proba < self.min_confidence
        filtered[low_confidence] = 0
        logger.info(f"Confidence filter: removed {low_confidence.sum()} signals")
        
        # Filter 2: Spread filter
        if 'spread' in df.columns and 'ATR' in df.columns:
            spread_ratio = df['spread'] / df['ATR']
            high_spread = spread_ratio > self.max_spread_atr
            filtered[high_spread.values] = 0
            logger.info(f"Spread filter: removed {high_spread.sum()} signals")
        elif 'spread_to_atr' in df.columns:
            # Use pre-calculated ratio if available
            high_spread = df['spread_to_atr'] > self.max_spread_atr
            filtered[high_spread.values] = 0
            logger.info(f"Spread filter: removed {high_spread.sum()} signals")
        else:
            logger.warning("Spread filter skipped: spread or ATR column not found")
        
        # Filter 3: Volatility filter (avoid extreme vol spikes)
        if 'ATR' in df.columns:
            atr_mean = df['ATR'].rolling(100).mean()
            atr_std = df['ATR'].rolling(100).std()
            atr_zscore = (df['ATR'] - atr_mean) / atr_std
            extreme_vol = np.abs(atr_zscore) > self.max_vol_zscore
            filtered[extreme_vol.values] = 0
            logger.info(f"Volatility filter: removed {extreme_vol.sum()} signals")
        
        # Filter 4: Session blackout (if configured)
        if self.blackout_sessions and 'hour' in df.columns:
            for session in self.blackout_sessions:
                if session == 'asian':
                    blackout_mask = (df['hour'] >= 0) & (df['hour'] < 8)
                elif session == 'london':
                    blackout_mask = (df['hour'] >= 8) & (df['hour'] < 16)
                elif session == 'newyork':
                    blackout_mask = (df['hour'] >= 13) & (df['hour'] < 21)
                else:
                    continue
                
                filtered[blackout_mask.values] = 0
                logger.info(f"Session filter ({session}): removed {blackout_mask.sum()} signals")
        
        logger.info(f"Total signals after filters: {filtered.sum()} (from {predictions.sum()})")
        
        return filtered
    
    def calculate_kelly_fraction(self, win_rate: float, avg_win: float, 
                                 avg_loss: float) -> float:
        """
        Calculate Kelly fraction for position sizing.
        
        Kelly % = (WinRate * AvgWin - (1-WinRate) * AvgLoss) / AvgWin
        
        Capped at self.kelly_cap to avoid over-leverage.
        """
        if avg_win == 0:
            return 0
        
        kelly = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
        kelly = max(0, min(kelly, self.kelly_cap))
        
        logger.info(f"Kelly fraction: {kelly:.4f} (capped at {self.kelly_cap})")
        return kelly
    
    def backtest_with_sizing(self, df: pd.DataFrame, predictions: np.ndarray,
                            account_value: float = 10000.0,
                            sl_atr_multiple: float = 1.0) -> Tuple[pd.DataFrame, Dict]:
        """
        Backtest with proper position sizing and risk management.
        
        Returns:
            equity_curve: DataFrame with timestamp and equity
            summary_stats: Dict with final metrics
        """
        equity = account_value
        equity_curve = [equity]
        timestamps = [df.index[0]]
        
        trades = []
        
        for i in range(len(df)):
            if predictions[i] == 1:
                # Calculate position size
                atr = df['ATR'].iloc[i] if 'ATR' in df.columns else 0.01
                realized_vol = df['realized_vol_20'].iloc[i] if 'realized_vol_20' in df.columns else None
                
                position_size = self.calculate_position_size(
                    equity, atr, sl_atr_multiple, realized_vol
                )
                
                # Get trade return
                trade_return = df['return'].iloc[i]
                
                # Apply spread cost
                spread_cost = 0.0002  # 2 pips
                trade_return_net = trade_return - spread_cost
                
                # Calculate P&L
                pnl = position_size * trade_return_net
                equity += pnl
                
                # Record trade
                trades.append({
                    'timestamp': df.index[i],
                    'position_size': position_size,
                    'return': trade_return,
                    'return_net': trade_return_net,
                    'pnl': pnl,
                    'equity': equity
                })
            
            equity_curve.append(equity)
            timestamps.append(df.index[i])
        
        # Create equity curve DataFrame
        equity_df = pd.DataFrame({
            'timestamp': timestamps,
            'equity': equity_curve
        })
        
        # Calculate summary statistics
        trades_df = pd.DataFrame(trades)
        
        if len(trades_df) > 0:
            total_return = (equity - account_value) / account_value
            num_trades = len(trades_df)
            
            # Sharpe ratio (annualized)
            returns = trades_df['pnl'] / account_value
            sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
            
            # Max drawdown
            running_max = equity_df['equity'].cummax()
            drawdown = running_max - equity_df['equity']
            max_dd = drawdown.max()
            max_dd_pct = (max_dd / running_max.max()) * 100
            
            summary = {
                'final_equity': equity,
                'total_return': total_return,
                'total_return_pct': total_return * 100,
                'num_trades': num_trades,
                'sharpe_annualized': sharpe,
                'max_drawdown': max_dd,
                'max_drawdown_pct': max_dd_pct
            }
        else:
            summary = {
                'final_equity': equity,
                'total_return': 0,
                'total_return_pct': 0,
                'num_trades': 0,
                'sharpe_annualized': 0,
                'max_drawdown': 0,
                'max_drawdown_pct': 0
            }
        
        return equity_df, summary