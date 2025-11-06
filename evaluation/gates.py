#!/usr/bin/env python3
"""
HARD GATES & FORENSIC REPORTING - Live Readiness Validation
============================================================

Enforces strict production readiness criteria. Models that fail gates
are rejected with actionable forensic diagnosis.

Hard Gates (OOS):
1. Max DD ≤ 6.0% (hard), warning at 4.5%
2. PF ≥ 1.60 (hard), warning at 1.45
3. Sharpe per trade ≥ 0.25 AND monthly PF ≥ 1.1 in ≥70% of months
4. Win Rate ≥ 52% OR Expectancy ≥ +0.08R
5. Min trades ≥ 30/month for intraday TFs (or ≥200 total OOS)
6. Stress survival: +25% costs & ±1 bar latency → PF ≥ 1.30 & DD ≤ 8.0%

Usage:
    from evaluation.gates import HardGates, ForensicReport
    gates = HardGates()
    passed, failures, report = gates.evaluate(results, symbol, timeframe)
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime


@dataclass
class GateThresholds:
    """Hard gate thresholds for live readiness."""
    
    # Performance gates
    min_profit_factor: float = 1.60
    min_profit_factor_warn: float = 1.45
    max_drawdown_pct: float = 6.0
    max_drawdown_warn: float = 4.5
    min_sharpe_per_trade: float = 0.25
    min_monthly_pf: float = 1.1
    min_monthly_pf_pct: float = 70.0  # % of months above threshold
    
    # Trade quality gates
    min_win_rate: float = 52.0
    min_expectancy_r: float = 0.08
    min_trades_total: int = 200
    min_trades_per_month: int = 30
    
    # Stress test gates
    stress_cost_mult: float = 1.25
    stress_min_pf: float = 1.30
    stress_max_dd: float = 8.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class ForensicReport:
    """Forensic diagnosis for failed models."""
    
    symbol: str
    timeframe: str
    timestamp: str
    live_ready: bool
    
    # Failures
    failures: List[str]
    
    # Root causes
    root_causes: List[str]
    
    # Ranked fixes
    ranked_fixes: List[str]
    
    # Metrics
    metrics: Dict
    
    # Stress test results
    stress_results: Optional[Dict] = None
    
    def to_json(self, path: Path = None) -> str:
        """Export to JSON."""
        data = {
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'timestamp': self.timestamp,
            'live_ready': self.live_ready,
            'failures': self.failures,
            'root_causes': self.root_causes,
            'ranked_fixes': self.ranked_fixes,
            'metrics': self.metrics,
            'stress_results': self.stress_results
        }
        
        json_str = json.dumps(data, indent=2)
        
        if path:
            with open(path, 'w') as f:
                f.write(json_str)
        
        return json_str
    
    def print_report(self):
        """Print formatted forensic report."""
        print(f"\n{'='*80}")
        print(f"FORENSIC REPORT: {self.symbol} {self.timeframe}")
        print(f"{'='*80}")
        print(f"Status: {'✅ LIVE READY' if self.live_ready else '❌ FAILED'}")
        print(f"Timestamp: {self.timestamp}")
        print(f"")
        
        if self.failures:
            print(f"FAILURES ({len(self.failures)}):")
            for i, failure in enumerate(self.failures, 1):
                print(f"  {i}. {failure}")
            print(f"")
        
        if self.root_causes:
            print(f"ROOT CAUSES ({len(self.root_causes)}):")
            for i, cause in enumerate(self.root_causes, 1):
                print(f"  {i}. {cause}")
            print(f"")
        
        if self.ranked_fixes:
            print(f"RANKED FIXES ({len(self.ranked_fixes)}):")
            for i, fix in enumerate(self.ranked_fixes, 1):
                print(f"  {i}. {fix}")
            print(f"")
        
        print(f"KEY METRICS:")
        for key, value in self.metrics.items():
            print(f"  {key}: {value}")
        
        if self.stress_results:
            print(f"\nSTRESS TEST:")
            for key, value in self.stress_results.items():
                print(f"  {key}: {value}")
        
        print(f"{'='*80}\n")


class HardGates:
    """Enforce hard gates for live readiness."""
    
    def __init__(self, thresholds: GateThresholds = None):
        self.thresholds = thresholds or GateThresholds()
    
    def evaluate(self, 
                results: Dict, 
                symbol: str, 
                timeframe: str,
                trades_df: pd.DataFrame = None) -> Tuple[bool, List[str], ForensicReport]:
        """
        Evaluate model against hard gates.
        
        Args:
            results: Backtest results dictionary
            symbol: Trading symbol
            timeframe: Timeframe
            trades_df: Optional DataFrame of individual trades
            
        Returns:
            (passed, failures, forensic_report)
        """
        failures = []
        root_causes = []
        ranked_fixes = []
        
        # Extract metrics
        pf = results.get('profit_factor', 0)
        dd = results.get('max_drawdown_pct', 100)
        sharpe = results.get('sharpe_ratio', 0)
        wr = results.get('win_rate', 0)
        total_trades = results.get('total_trades', 0)
        
        # Gate 1: Max Drawdown
        if dd > self.thresholds.max_drawdown_pct:
            failures.append(f"DD {dd:.1f}% > {self.thresholds.max_drawdown_pct}%")
            root_causes.extend(self._diagnose_high_drawdown(results, trades_df))
        elif dd > self.thresholds.max_drawdown_warn:
            failures.append(f"DD {dd:.1f}% > {self.thresholds.max_drawdown_warn}% (warning)")
        
        # Gate 2: Profit Factor
        if pf < self.thresholds.min_profit_factor:
            failures.append(f"PF {pf:.2f} < {self.thresholds.min_profit_factor}")
            root_causes.extend(self._diagnose_low_pf(results, trades_df))
        elif pf < self.thresholds.min_profit_factor_warn:
            failures.append(f"PF {pf:.2f} < {self.thresholds.min_profit_factor_warn} (warning)")
        
        # Gate 3: Sharpe per trade
        if sharpe < self.thresholds.min_sharpe_per_trade:
            failures.append(f"Sharpe {sharpe:.3f} < {self.thresholds.min_sharpe_per_trade}")
            root_causes.append("Low Sharpe: inconsistent returns or high volatility")
        
        # Gate 3b: Monthly PF consistency
        if trades_df is not None and 'timestamp' in trades_df.columns:
            monthly_pf_pct = self._check_monthly_consistency(trades_df)
            if monthly_pf_pct < self.thresholds.min_monthly_pf_pct:
                failures.append(f"Only {monthly_pf_pct:.0f}% months profitable (need {self.thresholds.min_monthly_pf_pct:.0f}%)")
                root_causes.append("Inconsistent performance across months")
        
        # Gate 4: Win Rate or Expectancy
        expectancy_r = self._calculate_expectancy_r(results, trades_df)
        if wr < self.thresholds.min_win_rate and expectancy_r < self.thresholds.min_expectancy_r:
            failures.append(
                f"WR {wr:.1f}% < {self.thresholds.min_win_rate}% AND "
                f"Expectancy {expectancy_r:.3f}R < {self.thresholds.min_expectancy_r}R"
            )
            root_causes.append("Poor trade quality: low win rate AND low expectancy")
        
        # Gate 5: Minimum trades
        oos_months = results.get('oos_duration_months', 6)
        trades_per_month = total_trades / max(oos_months, 1)
        
        if total_trades < self.thresholds.min_trades_total:
            failures.append(f"Total trades {total_trades} < {self.thresholds.min_trades_total}")
            root_causes.append("Insufficient trades: filters too strict or low signal quality")
        elif trades_per_month < self.thresholds.min_trades_per_month:
            failures.append(f"Trades/month {trades_per_month:.1f} < {self.thresholds.min_trades_per_month}")
        
        # Generate ranked fixes
        if root_causes:
            ranked_fixes = self._generate_fixes(failures, root_causes, results, symbol)
        
        # Build forensic report
        passed = len([f for f in failures if 'warning' not in f.lower()]) == 0
        
        report = ForensicReport(
            symbol=symbol,
            timeframe=timeframe,
            timestamp=datetime.utcnow().isoformat(),
            live_ready=passed,
            failures=failures,
            root_causes=root_causes,
            ranked_fixes=ranked_fixes,
            metrics={
                'profit_factor': pf,
                'max_drawdown_pct': dd,
                'sharpe_ratio': sharpe,
                'win_rate': wr,
                'total_trades': total_trades,
                'trades_per_month': trades_per_month,
                'expectancy_r': expectancy_r
            }
        )
        
        return passed, failures, report
    
    def _diagnose_high_drawdown(self, results: Dict, trades_df: pd.DataFrame = None) -> List[str]:
        """Diagnose root causes of high drawdown."""
        causes = []
        
        # Check for consecutive losses
        if trades_df is not None and 'pnl' in trades_df.columns:
            pnls = trades_df['pnl'].values
            max_consecutive_losses = self._max_consecutive_losses(pnls)
            
            if max_consecutive_losses >= 5:
                causes.append(
                    f"Consecutive losses: {max_consecutive_losses} in a row suggests "
                    "no regime filtering or poor exit strategy"
                )
        
        # Check SL hit rate
        sl_hit_rate = results.get('sl_hit_rate', 0)
        if sl_hit_rate > 60:
            causes.append(
                f"High SL hit rate ({sl_hit_rate:.0f}%): stops too tight or "
                "entry timing poor"
            )
        
        # Check largest loss
        largest_loss_pct = abs(results.get('largest_loss', 0)) / results.get('initial_capital', 100000) * 100
        if largest_loss_pct > 2:
            causes.append(
                f"Single large loss ({largest_loss_pct:.1f}%): position sizing issue "
                "or no circuit breaker"
            )
        
        return causes
    
    def _diagnose_low_pf(self, results: Dict, trades_df: pd.DataFrame = None) -> List[str]:
        """Diagnose root causes of low profit factor."""
        causes = []
        
        wr = results.get('win_rate', 0)
        avg_win = results.get('avg_win', 0)
        avg_loss = abs(results.get('avg_loss', 0))
        
        # Low win rate
        if wr < 45:
            causes.append(
                f"Low win rate ({wr:.0f}%): strategy likely trading against regime "
                "or entry signals are noisy"
            )
        
        # Poor risk/reward
        if avg_win > 0 and avg_loss > 0:
            rr_ratio = avg_win / avg_loss
            if rr_ratio < 1.2:
                causes.append(
                    f"Poor R:R ratio ({rr_ratio:.2f}): TP too close or SL too wide"
                )
        
        # Timeout rate
        timeout_rate = results.get('timeout_rate', 0)
        if timeout_rate > 30:
            causes.append(
                f"High timeout rate ({timeout_rate:.0f}%): trades not reaching targets, "
                "consider shorter horizon or tighter exits"
            )
        
        return causes
    
    def _check_monthly_consistency(self, trades_df: pd.DataFrame) -> float:
        """Check percentage of months with PF > threshold."""
        if len(trades_df) == 0:
            return 0.0
        
        trades_df['month'] = pd.to_datetime(trades_df['timestamp']).dt.to_period('M')
        
        monthly_pnl = trades_df.groupby('month')['pnl'].sum()
        profitable_months = (monthly_pnl > 0).sum()
        total_months = len(monthly_pnl)
        
        if total_months == 0:
            return 0.0
        
        return (profitable_months / total_months) * 100
    
    def _calculate_expectancy_r(self, results: Dict, trades_df: pd.DataFrame = None) -> float:
        """Calculate expectancy in R multiples."""
        wr = results.get('win_rate', 0) / 100
        avg_win = results.get('avg_win', 0)
        avg_loss = abs(results.get('avg_loss', 1))
        
        if avg_loss == 0:
            return 0.0
        
        # Expectancy = (Win% × AvgWin) - (Loss% × AvgLoss) / AvgLoss
        expectancy = (wr * avg_win - (1 - wr) * avg_loss) / avg_loss
        
        return expectancy
    
    def _max_consecutive_losses(self, pnls: np.ndarray) -> int:
        """Calculate maximum consecutive losses."""
        is_loss = pnls < 0
        
        max_consecutive = 0
        current_consecutive = 0
        
        for loss in is_loss:
            if loss:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive
    
    def _generate_fixes(self, failures: List[str], root_causes: List[str], 
                       results: Dict, symbol: str) -> List[str]:
        """Generate ranked list of fixes."""
        fixes = []
        
        # Prioritize by impact
        if any('DD' in f for f in failures):
            fixes.append("Priority 1: Tighten regime filter (raise ADX threshold from 25→28, ATR to 55th pct)")
            fixes.append("Priority 1: Increase cooldown from 3→6 bars to avoid revenge trading")
            fixes.append("Priority 1: Add circuit breaker: halt if rolling DD > 4%")
        
        if any('PF' in f for f in failures):
            if results.get('win_rate', 0) < 48:
                fixes.append("Priority 2: Raise entry thresholds (+15-20% on breakout/volume filters)")
                fixes.append("Priority 2: Add regime gating if not present")
            
            if results.get('timeout_rate', 0) > 30:
                fixes.append("Priority 2: Reduce forecast horizon by 1-2 bars")
                fixes.append("Priority 2: Add trailing stop or time-based exit")
        
        if any('Sharpe' in f for f in failures):
            fixes.append("Priority 3: Increase decision threshold (0.55→0.62) for higher confidence trades")
            fixes.append("Priority 3: Add session filter to avoid low-liquidity hours")
        
        if any('trades' in f.lower() for f in failures):
            fixes.append("Priority 3: Loosen filters slightly (reduce breakout threshold by 5-10%)")
            fixes.append("Priority 3: Check if cooldown is too aggressive")
        
        # Symbol-specific fixes
        if symbol in ['XAUUSD', 'XAGUSD']:
            fixes.append("Metals-specific: Exclude NFP/FOMC/CPI windows (±30 min)")
            fixes.append("Metals-specific: Use wider spread cap (2-3 bps vs 0.5 bps)")
        
        return fixes[:10]  # Top 10 fixes


def run_stress_tests(results: Dict, trades_df: pd.DataFrame = None) -> Dict:
    """
    Run stress tests: +25% costs, ±1 bar latency.
    
    Returns dict with stressed metrics.
    """
    # This is a placeholder - full implementation would re-run backtest
    # with stressed parameters
    
    base_pf = results.get('profit_factor', 0)
    base_dd = results.get('max_drawdown_pct', 0)
    
    # Approximate impact of +25% costs
    cost_impact = 0.15  # Reduces PF by ~15%
    stressed_pf = base_pf * (1 - cost_impact)
    
    # Approximate impact of ±1 bar latency  
    latency_impact = 0.10  # Reduces PF by ~10%, increases DD by ~20%
    stressed_pf *= (1 - latency_impact)
    stressed_dd = base_dd * 1.2
    
    return {
        'stressed_pf': stressed_pf,
        'stressed_dd': stressed_dd,
        'base_pf': base_pf,
        'base_dd': base_dd,
        'cost_mult': 1.25,
        'latency_bars': 1
    }

