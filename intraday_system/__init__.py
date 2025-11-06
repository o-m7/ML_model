"""
INTRADAY TRADING SYSTEM
========================

Production-grade multi-strategy trading system for Forex/Metals.

8 Symbols: XAUUSD, XAGUSD, EURUSD, GBPUSD, AUDUSD, NZDUSD, USDJPY, USDCAD
6 Strategies: S1-S6 across 5m-4h timeframes
Features: Walk-forward CV, ATR-based risk, ensemble ML models
"""

__version__ = "1.0.0"
__author__ = "Quant Dev Team"

from pathlib import Path

PACKAGE_ROOT = Path(__file__).parent
CONFIG_DIR = PACKAGE_ROOT / "config"

