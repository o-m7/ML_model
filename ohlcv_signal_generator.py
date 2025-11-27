"""
ohlcv_signal_generator.py - Real-time OHLCV strategy signal generator

Loads saved OHLCV models and generates live trading signals.
No look-ahead bias - only uses data up to current bar.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import logging
from datetime import datetime
from dataclasses import dataclass
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger('ohlcv_signal_gen')


@dataclass
class Signal:
    """Single trading signal output."""
    timestamp: str
    symbol: str
    timeframe: str
    strategy: str
    signal: int  # 1=BUY, -1=SELL, 0=NEUTRAL
    confidence: float  # Model prediction probability
    price: float
    atr: float
    features_used: int


class OHLCVSignalGenerator:
    """Generate signals from loaded OHLCV models."""
    
    def __init__(self, models_dir: str = "OHLCV_models"):
        self.models_dir = Path(models_dir)
        self.models = {}  # {(strategy, symbol, timeframe): model_data}
        self.load_models()
    
    def load_models(self):
        """Load all trained models from OHLCV_models/."""
        if not self.models_dir.exists():
            logger.warning(f"Models directory not found: {self.models_dir}")
            return
        
        for pkl_file in sorted(self.models_dir.glob("*.pkl")):
            try:
                with open(pkl_file, 'rb') as f:
                    model_data = pickle.load(f)
                
                key = (
                    model_data['strategy'],
                    model_data['symbol'],
                    model_data['timeframe']
                )
                self.models[key] = model_data
                
                logger.info(f"✓ Loaded model: {model_data['strategy']} {model_data['symbol']} {model_data['timeframe']}")
                logger.info(f"  TEST WR: {model_data['metrics']['test']['win_rate']:.1%}, "
                           f"PF: {model_data['metrics']['test']['pf']:.2f}")
            
            except Exception as e:
                logger.error(f"Failed to load {pkl_file}: {e}")
        
        logger.info(f"\nLoaded {len(self.models)} models total")
    
    def generate_signal(self, df: pd.DataFrame, strategy: str, symbol: str, 
                       timeframe: str) -> Signal:
        """Generate signal for current bar (last row of df)."""
        if len(df) == 0:
            raise ValueError("Empty dataframe")
        
        key = (strategy, symbol, timeframe)
        if key not in self.models:
            raise ValueError(f"Model not found: {key}")
        
        model_data = self.models[key]
        model = model_data['model']
        scaler = model_data['scaler']
        feature_cols = model_data['feature_cols']
        
        # Get current bar
        current = df.iloc[-1]
        
        # Generate OHLCV signal
        if strategy == 'trend_following':
            ohlcv_signal = self._trend_following_signal(df)
        elif strategy == 'mean_reversion':
            ohlcv_signal = self._mean_reversion_signal(df)
        elif strategy == 'volatility_breakout':
            ohlcv_signal = self._volatility_breakout_signal(df)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        # Prepare features
        X = df[feature_cols].iloc[[-1]].copy()
        X = X.fillna(0).replace([np.inf, -np.inf], 0)
        
        # Get ML prediction
        X_scaled = scaler.transform(X)
        ml_pred = model.predict(X_scaled)[0]
        
        # Get probability (if available)
        try:
            ml_proba = model.predict_proba(X_scaled)[0]
            confidence = ml_proba[int(ml_pred)] if ml_pred in [0, 1] else 0.5
        except:
            confidence = 0.5
        
        # Combine OHLCV signal + ML prediction
        final_signal = ohlcv_signal * ml_pred  # 1 if both agree, 0 if ML disagrees
        
        return Signal(
            timestamp=str(current.get('timestamp', datetime.now())),
            symbol=symbol,
            timeframe=timeframe,
            strategy=strategy,
            signal=int(final_signal),
            confidence=float(confidence),
            price=float(current['close']),
            atr=float(current.get('atr', 0)),
            features_used=len(feature_cols)
        )
    
    @staticmethod
    def _trend_following_signal(df: pd.DataFrame) -> int:
        """Generate trend following signal (close > EMA)."""
        if 'ema_20' not in df.columns:
            df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
        
        return 1 if df['close'].iloc[-1] > df['ema_20'].iloc[-1] else 0
    
    @staticmethod
    def _mean_reversion_signal(df: pd.DataFrame) -> int:
        """Generate mean reversion signal (RSI extremes)."""
        if 'rsi' not in df.columns:
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / (loss + 1e-10)
            df['rsi'] = 100 - (100 / (1 + rs))
        
        rsi = df['rsi'].iloc[-1]
        return 1 if (rsi < 30 or rsi > 70) else 0
    
    @staticmethod
    def _volatility_breakout_signal(df: pd.DataFrame) -> int:
        """Generate volatility breakout signal (price move > ATR)."""
        if 'atr' not in df.columns:
            prev_close = df['close'].shift(1)
            tr = pd.concat([
                df['high'] - df['low'],
                (df['high'] - prev_close).abs(),
                (df['low'] - prev_close).abs()
            ], axis=1).max(axis=1)
            df['atr'] = tr.rolling(14).mean()
        
        price_move = abs(df['close'].iloc[-1] - df['open'].iloc[-1])
        return 1 if price_move > df['atr'].iloc[-1] * 0.5 else 0
    
    def generate_all_signals(self, df: pd.DataFrame, symbol: str, 
                            timeframe: str) -> list:
        """Generate signals for all loaded models on this symbol/timeframe."""
        signals = []
        
        for (strat, sym, tf), _ in self.models.items():
            if sym == symbol and tf == timeframe:
                try:
                    sig = self.generate_signal(df, strat, symbol, timeframe)
                    signals.append(sig)
                except Exception as e:
                    logger.warning(f"Failed to generate signal for {strat}: {e}")
        
        return signals


class LiveBacktester:
    """Run live backtest with real-time position management."""
    
    def __init__(self, signal_gen: OHLCVSignalGenerator, 
                 initial_capital: float = 100000,
                 risk_per_trade: float = 0.02):
        self.signal_gen = signal_gen
        self.capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.positions = {}  # {(symbol, timeframe, strategy): position}
        self.trades = []
        self.equity_curve = []
    
    def run_backtest(self, df: pd.DataFrame, symbol: str, timeframe: str):
        """Run backtest on complete historical data."""
        logger.info("=" * 70)
        logger.info(f"LIVE BACKTEST: {symbol} {timeframe}")
        logger.info("=" * 70)
        
        pnl_cumulative = 0
        
        for i in range(1, len(df)):
            current_slice = df.iloc[:i+1].copy()
            current_bar = df.iloc[i]
            
            # Generate signals
            signals = self.signal_gen.generate_all_signals(
                current_slice, symbol, timeframe
            )
            
            # Process each signal
            for sig in signals:
                pos_key = (sig.symbol, sig.timeframe, sig.strategy)
                
                if sig.signal == 1:  # BUY signal
                    if pos_key not in self.positions:
                        # Enter position
                        risk_amount = self.capital * self.risk_per_trade
                        position_size = risk_amount / max(sig.atr, 0.1)
                        
                        self.positions[pos_key] = {
                            'entry_price': sig.price,
                            'entry_atr': sig.atr,
                            'entry_bar': i,
                            'size': position_size,
                            'strategy': sig.strategy,
                            'confidence': sig.confidence,
                        }
                        
                        if i % 100 == 0:
                            logger.debug(f"[{sig.timestamp}] ENTER {sig.strategy}: "
                                       f"Price={sig.price:.2f}, ATR={sig.atr:.4f}, "
                                       f"Size={position_size:.2f}, Conf={sig.confidence:.2%}")
                
                elif sig.signal == 0:  # EXIT signal
                    if pos_key in self.positions:
                        pos = self.positions[pos_key]
                        trade_pnl = (sig.price - pos['entry_price']) * pos['size']
                        pnl_cumulative += trade_pnl
                        self.capital += trade_pnl
                        
                        self.trades.append({
                            'entry': pos['entry_price'],
                            'exit': sig.price,
                            'pnl': trade_pnl,
                            'strategy': pos['strategy'],
                            'bars_held': i - pos['entry_bar'],
                        })
                        
                        del self.positions[pos_key]
                        
                        if trade_pnl > 0:
                            logger.info(f"✓ WIN: {pos['strategy']} +{trade_pnl:.2f}")
                        else:
                            logger.info(f"✗ LOSS: {pos['strategy']} {trade_pnl:.2f}")
            
            # Record equity
            self.equity_curve.append({
                'bar': i,
                'timestamp': current_bar.get('timestamp', i),
                'capital': self.capital,
                'pnl': pnl_cumulative,
            })
        
        # Summary
        self._print_summary()
    
    def _print_summary(self):
        """Print backtest summary stats."""
        if not self.trades:
            logger.warning("No trades completed")
            return
        
        df_trades = pd.DataFrame(self.trades)
        
        total_trades = len(df_trades)
        winning_trades = (df_trades['pnl'] > 0).sum()
        losing_trades = (df_trades['pnl'] <= 0).sum()
        
        total_pnl = df_trades['pnl'].sum()
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        avg_win = df_trades[df_trades['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = df_trades[df_trades['pnl'] <= 0]['pnl'].mean() if losing_trades > 0 else 0
        
        pf = avg_win * winning_trades / (abs(avg_loss) * losing_trades) if losing_trades > 0 and avg_loss < 0 else 0
        
        logger.info("\n" + "=" * 70)
        logger.info("BACKTEST SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Total Trades: {total_trades}")
        logger.info(f"Win Rate: {win_rate:.1%}")
        logger.info(f"Avg Win: ${avg_win:.2f}")
        logger.info(f"Avg Loss: ${avg_loss:.2f}")
        logger.info(f"Profit Factor: {pf:.2f}")
        logger.info(f"Total PnL: ${total_pnl:.2f}")
        logger.info(f"Final Capital: ${self.capital:.2f}")
        logger.info("=" * 70)
        
        # By strategy
        logger.info("\nBY STRATEGY:")
        for strategy in df_trades['strategy'].unique():
            strat_trades = df_trades[df_trades['strategy'] == strategy]
            strat_wr = (strat_trades['pnl'] > 0).sum() / len(strat_trades)
            strat_pnl = strat_trades['pnl'].sum()
            logger.info(f"  {strategy}: {len(strat_trades)} trades, {strat_wr:.1%} WR, ${strat_pnl:.2f} PnL")


def main():
    import argparse
    
    p = argparse.ArgumentParser()
    p.add_argument('--symbol', default='C:XAU-USD')
    p.add_argument('--timeframe', default='5T')
    p.add_argument('--backtest', action='store_true', help='Run backtest mode')
    args = p.parse_args()
    
    # Initialize generator
    gen = OHLCVSignalGenerator()
    
    if args.backtest:
        # Load historical data
        path = Path("feature_store") / args.symbol / f"{args.symbol}_{args.timeframe}.parquet"
        if not path.exists():
            logger.error(f"Data file not found: {path}")
            sys.exit(1)
        
        df = pd.read_parquet(path)
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Run backtest
        backtester = LiveBacktester(gen)
        backtester.run_backtest(df, args.symbol, args.timeframe)
    else:
        # Live signal mode
        logger.info("OHLCV Signal Generator Ready")
        logger.info(f"Use --backtest to run backtest")


if __name__ == '__main__':
    main()
