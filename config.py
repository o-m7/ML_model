"""
Configuration module for institutional-grade XAUUSD/XAGUSD trading system.

This module contains all configurable parameters for the trading system,
including data paths, trading parameters, risk management, and model settings.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any
from pathlib import Path


@dataclass
class DataConfig:
    """Data loading and preprocessing configuration."""

    # Symbols to trade
    symbols: List[str] = field(default_factory=lambda: ["XAUUSD", "XAGUSD"])

    # Timeframes (in minutes)
    timeframes: List[int] = field(default_factory=lambda: [5, 15, 30, 60])

    # Data paths (override with actual paths)
    data_dir: Path = field(default_factory=lambda: Path("data"))

    # Train/validation/test split ratios (chronological)
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    test_ratio: float = 0.2  # Final out-of-sample hold-out

    # Minimum data points required per symbol/timeframe
    min_data_points: int = 10000


@dataclass
class FeatureConfig:
    """Feature engineering configuration."""

    # Moving averages
    ma_periods: List[int] = field(default_factory=lambda: [10, 20, 50, 100, 200])

    # MACD parameters
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    # RSI periods
    rsi_periods: List[int] = field(default_factory=lambda: [14])

    # ATR periods
    atr_periods: List[int] = field(default_factory=lambda: [14])

    # Bollinger Bands
    bb_period: int = 20
    bb_std: float = 2.0

    # ADX for regime detection
    adx_period: int = 14

    # Volatility percentile lookback
    vol_percentile_window: int = 100

    # Use news/sentiment features (if available)
    use_news_features: bool = False
    news_data_path: Path = field(default_factory=lambda: Path("data/news.csv"))


@dataclass
class RegimeConfig:
    """Regime detection configuration."""

    # ADX thresholds
    adx_trend_threshold: float = 25.0  # Above = trending
    adx_strong_trend: float = 40.0     # Above = strong trend

    # Volatility regime thresholds (ATR percentile)
    vol_high_threshold: float = 80.0   # Above = high volatility / event regime
    vol_low_threshold: float = 20.0    # Below = low volatility

    # MA alignment for trend detection
    ma_alignment_periods: List[int] = field(default_factory=lambda: [20, 50, 100])

    # Range detection
    range_lookback: int = 50  # Bars to check for range behavior


@dataclass
class StrategyConfig:
    """Trading strategy configuration."""

    # Stop-loss and take-profit in ATR units
    sl_atr_multiplier: float = 1.0
    tp_atr_multiplier: float = 2.0

    # ML probability threshold for trade execution
    ml_prob_threshold: float = 0.60

    # Minimum expected R to take trade
    min_expected_r: float = 0.3

    # Pullback entry (for trend regimes)
    pullback_ma_period: int = 20
    pullback_tolerance_atr: float = 0.5

    # Mean reversion (for range regimes)
    bb_extreme_std: float = 2.0  # Entry at ±2 std
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0

    # Time-based exit (max bars in trade, 0 = disabled)
    max_bars_in_trade: int = 0

    # Trailing stop (in ATR units, 0 = disabled)
    trailing_stop_atr: float = 0.0


@dataclass
class RiskConfig:
    """Risk management configuration."""

    # Per-trade risk (% of equity)
    risk_per_trade_pct: float = 0.5

    # Maximum concurrent positions per symbol
    max_positions_per_symbol: int = 1

    # Maximum concurrent positions across all symbols/timeframes
    max_total_positions: int = 4

    # Initial capital
    initial_capital: float = 100000.0

    # Position sizing method: 'fixed_risk' or 'fixed_lots'
    position_sizing_method: str = 'fixed_risk'

    # Fixed lot size (if method = 'fixed_lots')
    fixed_lot_size: float = 1.0


@dataclass
class CostConfig:
    """Transaction cost configuration."""

    # Spread in pips (or dollars for XAUUSD/XAGUSD)
    spreads: Dict[str, float] = field(default_factory=lambda: {
        "XAUUSD": 0.30,  # $0.30 per oz
        "XAGUSD": 0.03   # $0.03 per oz
    })

    # Commission per lot (or per million notional)
    commission_per_lot: float = 7.0  # USD per lot

    # Slippage model: 'fixed' or 'atr_based'
    slippage_model: str = 'fixed'
    slippage_fixed: float = 0.10  # Fixed slippage in price units
    slippage_atr_pct: float = 0.05  # Slippage as % of ATR (if atr_based)

    # Contract specifications
    contract_sizes: Dict[str, float] = field(default_factory=lambda: {
        "XAUUSD": 100.0,  # 100 oz per lot
        "XAGUSD": 5000.0  # 5000 oz per lot
    })


@dataclass
class ModelConfig:
    """Machine learning model configuration."""

    # Model type: 'xgboost', 'lightgbm', 'logistic'
    model_type: str = 'xgboost'

    # Walk-forward parameters
    n_splits: int = 5  # Number of time-series splits

    # XGBoost hyperparameters
    xgb_params: Dict[str, Any] = field(default_factory=lambda: {
        'max_depth': 6,
        'learning_rate': 0.05,
        'n_estimators': 200,
        'min_child_weight': 3,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 0.1,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42,
        'n_jobs': -1,
        'tree_method': 'hist'
    })

    # LightGBM hyperparameters
    lgb_params: Dict[str, Any] = field(default_factory=lambda: {
        'max_depth': 6,
        'learning_rate': 0.05,
        'n_estimators': 200,
        'num_leaves': 31,
        'min_child_samples': 20,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42,
        'n_jobs': -1
    })

    # Feature importance threshold (optional feature selection)
    feature_importance_threshold: float = 0.0

    # Random seed for reproducibility
    random_state: int = 42


@dataclass
class PropEvalConfig:
    """Funded account / prop firm evaluation configuration."""

    # Initial capital for prop evaluation
    initial_capital: float = 100000.0

    # Maximum daily drawdown (%)
    max_daily_dd_pct: float = 5.0

    # Maximum overall drawdown (%)
    max_overall_dd_pct: float = 6.0

    # Minimum trading days
    min_trading_days: int = 10

    # Minimum trades
    min_trades: int = 20

    # Target metrics for acceptance
    target_win_rate: float = 0.60
    target_profit_factor: float = 1.5
    target_max_dd: float = 6.0  # %


@dataclass
class MonitoringConfig:
    """Model monitoring and retraining configuration."""

    # Retraining frequency (days)
    retrain_frequency_days: int = 30

    # Training window (days) - rolling lookback
    training_window_days: int = 730  # 2 years

    # Performance degradation thresholds (trigger retraining or halt)
    min_rolling_sharpe: float = 0.5
    min_rolling_pf: float = 1.2
    max_rolling_dd: float = 8.0  # %

    # Rolling window for performance monitoring (days)
    monitoring_window_days: int = 30

    # Model versioning
    model_dir: Path = field(default_factory=lambda: Path("models"))
    performance_log_path: Path = field(default_factory=lambda: Path("logs/model_performance.json"))


@dataclass
class Config:
    """Master configuration aggregating all sub-configs."""

    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    regime: RegimeConfig = field(default_factory=RegimeConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    costs: CostConfig = field(default_factory=CostConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    prop_eval: PropEvalConfig = field(default_factory=PropEvalConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)

    # Global random seed
    random_seed: int = 42

    # Verbosity
    verbose: bool = True


def get_default_config() -> Config:
    """Return default configuration instance."""
    return Config()


if __name__ == "__main__":
    # Example usage
    config = get_default_config()
    print(f"Symbols: {config.data.symbols}")
    print(f"Timeframes: {config.data.timeframes}")
    print(f"ML Threshold: {config.strategy.ml_prob_threshold}")
    print(f"Risk per trade: {config.risk.risk_per_trade_pct}%")
