#!/usr/bin/env python3
"""
REAL-TIME EXECUTION SYSTEM
==========================

Production trading system with:
- Sub-second latency predictions
- Real-time feature calculation
- Risk management & position tracking
- Order execution monitoring
- Performance tracking & alerting
- Automatic trading halt on drawdown

Usage:
    python realtime_execution.py --symbol XAUUSD --tf 15T --account-size 100000
"""

import argparse
import json
import pickle
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

class RealTimeExecutionSystem:
    """
    Production real-time trading system.
    """
    
    def __init__(self, symbol: str, timeframe: str, account_size: float):
        self.symbol = symbol
        self.timeframe = timeframe
        self.account_size = account_size
        self.initial_account = account_size
        
        # Load production model
        self.system = self._load_production_model()
        
        # Initialize tracking
        self.positions = {}  # Open positions
        self.trades_history = []  # Closed trades
        self.equity_curve = [account_size]
        self.max_equity = account_size
        
        # Performance monitoring
        self.monitoring_file = Path(f"ML_Trading/monitoring/{symbol}_{timeframe}_performance.json")
        self.monitoring_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing performance data
        if self.monitoring_file.exists():
            with open(self.monitoring_file) as f:
                self.perf_data = json.load(f)
        else:
            self.perf_data = {
                'symbol': symbol,
                'timeframe': timeframe,
                'start_time': datetime.now(timezone.utc).isoformat(),
                'recent_trades': [],
                'expected_win_rate': 0.55,
                'expected_sharpe': 1.0
            }
        
        # Safety limits
        self.max_drawdown_pct = 0.06  # 6% max drawdown
        self.max_daily_loss_pct = 0.02  # 2% max daily loss
        self.trading_halted = False
        
        print(f"\n{'='*80}")
        print(f"REAL-TIME EXECUTION SYSTEM - {symbol} {timeframe}")
        print(f"{'='*80}")
        print(f"Account Size: ${account_size:,.2f}")
        print(f"Max Drawdown: {self.max_drawdown_pct*100:.1f}%")
        print(f"Max Daily Loss: {self.max_daily_loss_pct*100:.1f}%")
        print(f"{'='*80}\n")
    
    def _load_production_model(self) -> Dict:
        """Load production model system."""
        model_path = Path(f"ML_Trading/models/{self.symbol}/{self.symbol}_{self.timeframe}_PRODUCTION.pkl")
        
        if not model_path.exists():
            # Try to find latest model
            model_dir = Path(f"ML_Trading/models/{self.symbol}")
            models = list(model_dir.glob(f"{self.symbol}_{self.timeframe}_production_*.pkl"))
            if not models:
                raise FileNotFoundError(f"No production model found for {self.symbol} {self.timeframe}")
            model_path = max(models, key=lambda p: p.stat().st_mtime)
        
        print(f"Loading model: {model_path}")
        
        with open(model_path, 'rb') as f:
            system = pickle.load(f)
        
        print(f"  Model version: {system.get('version', 'unknown')}")
        print(f"  Trained: {system['results']['trained_at']}")
        print(f"  Backtest Sharpe: {system['results']['backtest']['sharpe']:.2f}")
        
        return system
    
    def calculate_realtime_features(self, current_bar: Dict) -> np.ndarray:
        """
        Calculate features for current bar in real-time.
        This must be FAST (<100ms).
        """
        # In production, this would connect to live data feed
        # For now, we'll assume features are pre-calculated
        # But this is where you'd compute them on-the-fly
        
        # Extract features used by current regime model
        regime = self._detect_current_regime(current_bar)
        features = self.system['regime_detector'].feature_selector.regime_feature_map.get(
            regime.value, {}
        )
        
        # Get feature values
        feature_values = []
        for feat in features:
            value = current_bar.get(feat, 0)
            feature_values.append(value)
        
        return np.array(feature_values).reshape(1, -1)
    
    def _detect_current_regime(self, current_bar: Dict):
        """Detect current market regime."""
        from jpm_production_system import MarketRegime
        
        # Use loaded regime detector
        regime_detector = self.system['regime_detector']
        
        # For real-time, we need recent history
        # In production, maintain sliding window of recent bars
        # For now, use regime from training
        
        # Default to most common regime from training
        regime_dist = self.system['results']['regimes']
        most_common = max(regime_dist.keys(), key=lambda k: regime_dist[k].get('n_samples', 0))
        
        return MarketRegime(most_common)
    
    def generate_signal(self, current_bar: Dict) -> Optional[Dict]:
        """
        Generate trading signal for current bar.
        
        Returns signal dict or None if no trade.
        """
        start_time = time.time()
        
        # Check if trading is halted
        if self.trading_halted:
            return None
        
        # Check if we already have a position
        if self.symbol in self.positions:
            return None  # Already in trade
        
        try:
            # Detect regime
            regime = self._detect_current_regime(current_bar)
            
            # Get features
            from jpm_production_system import StrategySelector
            
            regime_model = self.system['results']['regimes'].get(regime.value)
            if not regime_model:
                return None
            
            features = regime_model['features']
            X = np.array([current_bar.get(f, 0) for f in features]).reshape(1, -1)
            
            # Generate prediction
            ensemble_system = self.system['ensemble_system']
            pred_proba, details = ensemble_system.predict_ensemble(X, regime)
            
            # Get strategy parameters
            strategy = StrategySelector.select_strategy(regime)
            params = StrategySelector.get_strategy_parameters(strategy, regime)
            
            # Check confidence threshold
            if pred_proba < params['min_confidence']:
                return None
            
            # Calculate position size
            atr = current_bar.get('atr14', current_bar['close'] * 0.02)
            risk_amount = self.account_size * 0.01  # 1% risk
            
            sl_distance = atr * params['sl_r']
            position_size = risk_amount / sl_distance
            
            # Apply Kelly fraction
            position_size *= 0.25
            
            # Calculate prices
            entry_price = current_bar['close']
            tp_price = entry_price + (atr * params['tp_r'])
            sl_price = entry_price - (atr * params['sl_r'])
            
            # Execution latency
            latency_ms = (time.time() - start_time) * 1000
            
            signal = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'symbol': self.symbol,
                'regime': regime.value,
                'strategy': strategy.value,
                'signal': 'BUY',
                'confidence': pred_proba * 100,
                'probability': pred_proba,
                'entry_price': entry_price,
                'tp_price': tp_price,
                'sl_price': sl_price,
                'position_size': position_size,
                'risk_amount': risk_amount,
                'risk_reward_ratio': params['tp_r'] / params['sl_r'],
                'latency_ms': latency_ms,
                'model_details': details
            }
            
            print(f"\n{'='*80}")
            print(f"SIGNAL GENERATED - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*80}")
            print(f"Symbol: {self.symbol}")
            print(f"Regime: {regime.value}")
            print(f"Strategy: {strategy.value}")
            print(f"Confidence: {pred_proba*100:.1f}%")
            print(f"Entry: {entry_price:.5f}")
            print(f"TP: {tp_price:.5f} (+{params['tp_r']:.1f}R)")
            print(f"SL: {sl_price:.5f} (-{params['sl_r']:.1f}R)")
            print(f"Size: {position_size:.4f} units")
            print(f"Risk: ${risk_amount:.2f}")
            print(f"Latency: {latency_ms:.2f}ms")
            print(f"{'='*80}\n")
            
            return signal
            
        except Exception as e:
            print(f"Error generating signal: {e}")
            return None
    
    def execute_trade(self, signal: Dict) -> bool:
        """
        Execute trade based on signal.
        In production, this connects to broker API.
        """
        # Validate signal
        if not signal or signal['signal'] != 'BUY':
            return False
        
        # In production, send order to broker
        # For now, simulate order execution
        
        position = {
            'symbol': signal['symbol'],
            'entry_time': datetime.now(timezone.utc),
            'entry_price': signal['entry_price'],
            'position_size': signal['position_size'],
            'tp_price': signal['tp_price'],
            'sl_price': signal['sl_price'],
            'risk_amount': signal['risk_amount'],
            'confidence': signal['confidence'],
            'regime': signal['regime'],
            'strategy': signal['strategy']
        }
        
        self.positions[signal['symbol']] = position
        
        print(f"✓ Trade executed: {signal['symbol']} @ {signal['entry_price']:.5f}")
        
        return True
    
    def update_positions(self, current_bar: Dict) -> List[Dict]:
        """
        Update open positions and check for TP/SL hits.
        Returns list of closed trades.
        """
        closed_trades = []
        
        for symbol, position in list(self.positions.items()):
            # Check TP
            if current_bar['high'] >= position['tp_price']:
                closed_trade = self._close_position(
                    position, current_bar, 'TP',
                    exit_price=position['tp_price']
                )
                closed_trades.append(closed_trade)
                del self.positions[symbol]
            
            # Check SL
            elif current_bar['low'] <= position['sl_price']:
                closed_trade = self._close_position(
                    position, current_bar, 'SL',
                    exit_price=position['sl_price']
                )
                closed_trades.append(closed_trade)
                del self.positions[symbol]
        
        return closed_trades
    
    def _close_position(self, position: Dict, current_bar: Dict,
                       exit_reason: str, exit_price: float) -> Dict:
        """Close position and calculate P&L."""
        entry_price = position['entry_price']
        size = position['position_size']
        
        # Calculate P&L
        price_change = exit_price - entry_price
        gross_pnl = price_change * size
        
        # Apply costs
        commission = exit_price * 0.0001 * size  # 1 bp
        slippage = exit_price * 0.00005 * size   # 0.5 bp
        net_pnl = gross_pnl - commission - slippage
        
        # Update account
        self.account_size += net_pnl
        self.equity_curve.append(self.account_size)
        
        # Update max equity
        if self.account_size > self.max_equity:
            self.max_equity = self.account_size
        
        trade = {
            'symbol': position['symbol'],
            'entry_time': position['entry_time'].isoformat(),
            'exit_time': datetime.now(timezone.utc).isoformat(),
            'entry_price': entry_price,
            'exit_price': exit_price,
            'position_size': size,
            'gross_pnl': gross_pnl,
            'net_pnl': net_pnl,
            'return_pct': (net_pnl / (entry_price * size)) * 100,
            'exit_reason': exit_reason,
            'outcome': 'win' if net_pnl > 0 else 'loss',
            'confidence': position['confidence'],
            'regime': position['regime'],
            'strategy': position['strategy']
        }
        
        self.trades_history.append(trade)
        
        # Update performance monitoring
        self.perf_data['recent_trades'].append(trade)
        self.perf_data['recent_trades'] = self.perf_data['recent_trades'][-100:]  # Keep last 100
        
        # Save to file
        with open(self.monitoring_file, 'w') as f:
            json.dump(self.perf_data, f, indent=2)
        
        # Print trade result
        emoji = "✓" if trade['outcome'] == 'win' else "✗"
        print(f"\n{emoji} TRADE CLOSED - {exit_reason}")
        print(f"  Entry: {entry_price:.5f} @ {position['entry_time'].strftime('%H:%M:%S')}")
        print(f"  Exit: {exit_price:.5f} @ {datetime.now().strftime('%H:%M:%S')}")
        print(f"  P&L: ${net_pnl:+.2f} ({trade['return_pct']:+.2f}%)")
        print(f"  Account: ${self.account_size:,.2f}")
        
        # Check risk limits
        self._check_risk_limits()
        
        return trade
    
    def _check_risk_limits(self):
        """Check if risk limits exceeded and halt trading if needed."""
        # Calculate current drawdown
        drawdown = (self.max_equity - self.account_size) / self.max_equity
        
        if drawdown > self.max_drawdown_pct:
            self.trading_halted = True
            print(f"\n{'='*80}")
            print(f"⚠️  TRADING HALTED - MAX DRAWDOWN EXCEEDED")
            print(f"{'='*80}")
            print(f"Current Drawdown: {drawdown*100:.2f}%")
            print(f"Max Allowed: {self.max_drawdown_pct*100:.2f}%")
            print(f"Account: ${self.account_size:,.2f} (from ${self.max_equity:,.2f})")
            print(f"{'='*80}\n")
            
            # Send alert (in production, send email/SMS)
            self._send_alert("MAX DRAWDOWN EXCEEDED", {
                'drawdown': drawdown,
                'account_size': self.account_size
            })
        
        # Check daily loss
        if len(self.equity_curve) > 1:
            daily_change = (self.account_size - self.equity_curve[0]) / self.equity_curve[0]
            if daily_change < -self.max_daily_loss_pct:
                self.trading_halted = True
                print(f"\n⚠️  TRADING HALTED - MAX DAILY LOSS EXCEEDED")
                print(f"Daily Loss: {daily_change*100:.2f}%\n")
    
    def _send_alert(self, message: str, data: Dict):
        """Send alert (email/SMS in production)."""
        alert_file = Path(f"ML_Trading/monitoring/alerts.json")
        
        alert = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'symbol': self.symbol,
            'message': message,
            'data': data
        }
        
        alerts = []
        if alert_file.exists():
            with open(alert_file) as f:
                alerts = json.load(f)
        
        alerts.append(alert)
        
        with open(alert_file, 'w') as f:
            json.dump(alerts[-100:], f, indent=2)  # Keep last 100 alerts
        
        print(f"Alert sent: {message}")
    
    def get_performance_stats(self) -> Dict:
        """Calculate current performance statistics."""
        if len(self.trades_history) == 0:
            return {}
        
        trades = pd.DataFrame(self.trades_history)
        
        wins = trades[trades['outcome'] == 'win']
        losses = trades[trades['outcome'] == 'loss']
        
        stats = {
            'total_trades': len(trades),
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': len(wins) / len(trades) * 100 if len(trades) > 0 else 0,
            'avg_win': wins['net_pnl'].mean() if len(wins) > 0 else 0,
            'avg_loss': losses['net_pnl'].mean() if len(losses) > 0 else 0,
            'profit_factor': (wins['net_pnl'].sum() / abs(losses['net_pnl'].sum())
                            if len(losses) > 0 and losses['net_pnl'].sum() != 0 else 999),
            'total_pnl': trades['net_pnl'].sum(),
            'total_return_pct': (self.account_size - self.initial_account) / self.initial_account * 100,
            'sharpe_ratio': self._calculate_sharpe(),
            'max_drawdown_pct': self._calculate_max_dd(),
            'current_account': self.account_size
        }
        
        return stats
    
    def _calculate_sharpe(self) -> float:
        """Calculate Sharpe ratio from trade returns."""
        if len(self.trades_history) < 2:
            return 0.0
        
        returns = [t['return_pct'] for t in self.trades_history]
        if np.std(returns) == 0:
            return 0.0
        
        sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252)
        return sharpe
    
    def _calculate_max_dd(self) -> float:
        """Calculate maximum drawdown percentage."""
        if len(self.equity_curve) < 2:
            return 0.0
        
        equity = pd.Series(self.equity_curve)
        running_max = equity.cummax()
        drawdown = (running_max - equity) / running_max * 100
        
        return drawdown.max()
    
    def print_performance_summary(self):
        """Print performance summary."""
        stats = self.get_performance_stats()
        
        if not stats:
            print("No trades yet")
            return
        
        print(f"\n{'='*80}")
        print(f"PERFORMANCE SUMMARY - {self.symbol} {self.timeframe}")
        print(f"{'='*80}")
        print(f"\nTrades: {stats['total_trades']} (W:{stats['wins']} / L:{stats['losses']})")
        print(f"Win Rate: {stats['win_rate']:.1f}%")
        print(f"Profit Factor: {stats['profit_factor']:.2f}")
        print(f"Sharpe Ratio: {stats['sharpe_ratio']:.2f}")
        print(f"\nAvg Win: ${stats['avg_win']:.2f}")
        print(f"Avg Loss: ${stats['avg_loss']:.2f}")
        print(f"\nTotal P&L: ${stats['total_pnl']:+,.2f}")
        print(f"Return: {stats['total_return_pct']:+.2f}%")
        print(f"Max Drawdown: {stats['max_drawdown_pct']:.2f}%")
        print(f"\nCurrent Account: ${stats['current_account']:,.2f}")
        print(f"{'='*80}\n")

def main():
    parser = argparse.ArgumentParser(description='Real-Time Execution System')
    parser.add_argument('--symbol', type=str, required=True, help='Trading symbol')
    parser.add_argument('--tf', type=str, required=True, help='Timeframe')
    parser.add_argument('--account-size', type=float, default=100000,
                       help='Account size (default: 100000)')
    parser.add_argument('--demo', action='store_true',
                       help='Run demo with simulated data')
    
    args = parser.parse_args()
    
    try:
        # Initialize system
        system = RealTimeExecutionSystem(args.symbol, args.tf, args.account_size)
        
        if args.demo:
            print("\nRunning DEMO mode with simulated data...\n")
            # In production, this connects to live data feed
            # For demo, load historical data
            from jpm_production_system import load_data_safe, ProductionConfig
            
            config = ProductionConfig()
            df = load_data_safe(args.symbol, args.tf, config)
            
            # Simulate real-time execution on recent data
            recent_df = df.tail(100)
            
            for idx, row in recent_df.iterrows():
                current_bar = row.to_dict()
                
                # Update open positions
                closed_trades = system.update_positions(current_bar)
                
                # Generate new signal if no position
                if args.symbol not in system.positions:
                    signal = system.generate_signal(current_bar)
                    
                    if signal:
                        system.execute_trade(signal)
                
                # Small delay to simulate real-time
                time.sleep(0.1)
            
            # Print final summary
            system.print_performance_summary()
        
        else:
            print("Live trading mode - Connect to your data feed")
            print("See documentation for integration with broker API")
        
        return 0
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    exit(main())
