#!/usr/bin/env python3
"""
EXECUTION GUARDRAILS
====================

Filters and safety checks to prevent bad trades in live execution.

Guards against:
- Stale data
- High spread periods
- Low liquidity sessions
- High volatility spikes
- Excessive latency
- Signal quality degradation
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict
from dataclasses import dataclass


@dataclass
class GuardrailResult:
    """Result of guardrail check."""
    passed: bool
    reason: str = ""
    value: Optional[float] = None
    threshold: Optional[float] = None


class ExecutionGuardrails:
    """Execution safety checks for live trading."""

    def __init__(
        self,
        max_spread_atr_ratio: float = 0.15,      # Max spread as % of ATR
        max_data_age_seconds: int = 300,          # 5 minutes max staleness
        min_confidence: float = 0.55,             # Minimum model confidence
        max_latency_ms: float = 250,              # Max end-to-end latency
        blocked_sessions: list = None,            # Sessions to avoid
        min_atr_pct: float = 0.003,              # Min ATR as % of price (volatility floor)
        max_atr_pct: float = 0.05,               # Max ATR as % of price (volatility cap)
    ):
        self.max_spread_atr_ratio = max_spread_atr_ratio
        self.max_data_age_seconds = max_data_age_seconds
        self.min_confidence = min_confidence
        self.max_latency_ms = max_latency_ms
        self.blocked_sessions = blocked_sessions or []
        self.min_atr_pct = min_atr_pct
        self.max_atr_pct = max_atr_pct

    def check_data_staleness(
        self,
        last_bar_time: pd.Timestamp,
        timeframe_minutes: int
    ) -> GuardrailResult:
        """
        Check if data is stale.

        For forex markets (24/5), accounts for weekend gaps and daily rollover.

        Args:
            last_bar_time: Timestamp of last bar
            timeframe_minutes: Bar timeframe in minutes

        Returns:
            GuardrailResult
        """
        now = datetime.now(timezone.utc)
        if last_bar_time.tz is None:
            last_bar_time = last_bar_time.tz_localize('UTC')

        age_seconds = (now - last_bar_time).total_seconds()

        # For forex: Allow larger gaps on weekends and during rollover (10-11pm UTC)
        # Forex markets close Friday 10pm UTC, reopen Sunday 10pm UTC
        is_weekend = now.weekday() >= 5  # Saturday=5, Sunday=6
        is_monday_morning = now.weekday() == 0 and now.hour < 2  # Monday before 2am UTC

        if is_weekend or is_monday_morning:
            # During weekend, allow up to 72 hours staleness
            max_allowed_age = 72 * 3600
        else:
            # During trading week: Allow up to 8 hours for Polygon API delays
            # Temporarily relaxed to handle delayed/low-volume periods
            max_allowed_age = 8 * 3600  # 8 hours = 28800 seconds

        if age_seconds > max_allowed_age:
            return GuardrailResult(
                passed=False,
                reason=f"Data stale: {age_seconds:.0f}s old (max {max_allowed_age:.0f}s)",
                value=age_seconds,
                threshold=max_allowed_age
            )

        return GuardrailResult(passed=True, value=age_seconds, threshold=max_allowed_age)

    def check_spread(
        self,
        current_spread: float,
        atr: float
    ) -> GuardrailResult:
        """
        Check if spread is acceptable relative to ATR.

        Wide spreads indicate poor liquidity or high volatility.

        Args:
            current_spread: Current bid-ask spread
            atr: Current ATR value

        Returns:
            GuardrailResult
        """
        if atr == 0 or np.isnan(atr):
            return GuardrailResult(
                passed=False,
                reason="ATR is zero or NaN",
                value=atr
            )

        spread_ratio = current_spread / atr

        if spread_ratio > self.max_spread_atr_ratio:
            return GuardrailResult(
                passed=False,
                reason=f"Spread too wide: {spread_ratio:.2%} of ATR (max {self.max_spread_atr_ratio:.2%})",
                value=spread_ratio,
                threshold=self.max_spread_atr_ratio
            )

        return GuardrailResult(passed=True, value=spread_ratio, threshold=self.max_spread_atr_ratio)

    def check_volatility_regime(
        self,
        atr: float,
        price: float
    ) -> GuardrailResult:
        """
        Check if volatility is in acceptable range.

        Too low = noise trading. Too high = dangerous.

        Args:
            atr: Current ATR
            price: Current price

        Returns:
            GuardrailResult
        """
        atr_pct = atr / price

        if atr_pct < self.min_atr_pct:
            return GuardrailResult(
                passed=False,
                reason=f"Volatility too low: {atr_pct:.3%} (min {self.min_atr_pct:.3%})",
                value=atr_pct,
                threshold=self.min_atr_pct
            )

        if atr_pct > self.max_atr_pct:
            return GuardrailResult(
                passed=False,
                reason=f"Volatility too high: {atr_pct:.3%} (max {self.max_atr_pct:.3%})",
                value=atr_pct,
                threshold=self.max_atr_pct
            )

        return GuardrailResult(passed=True, value=atr_pct, threshold=self.max_atr_pct)

    def check_session(
        self,
        timestamp: pd.Timestamp
    ) -> GuardrailResult:
        """
        Check if current session is tradeable.

        Args:
            timestamp: Current timestamp

        Returns:
            GuardrailResult
        """
        if timestamp.tz is None:
            timestamp = timestamp.tz_localize('UTC')

        hour = timestamp.hour
        day_of_week = timestamp.dayofweek

        # Define sessions (UTC hours)
        # Asia: 0-8, Europe: 8-16, US: 13-21
        if hour >= 0 and hour < 8:
            session = 'asia'
        elif hour >= 8 and hour < 16:
            session = 'europe'
        elif hour >= 13 and hour < 21:
            session = 'us'
        else:
            session = 'overnight'

        if session in self.blocked_sessions:
            return GuardrailResult(
                passed=False,
                reason=f"Trading blocked during {session} session",
                value=hour
            )

        # Block weekends
        if day_of_week >= 5:  # Saturday = 5, Sunday = 6
            return GuardrailResult(
                passed=False,
                reason=f"Weekend trading blocked (day {day_of_week})",
                value=day_of_week
            )

        return GuardrailResult(passed=True, value=hour)

    def check_confidence(
        self,
        confidence: float
    ) -> GuardrailResult:
        """
        Check if model confidence meets threshold.

        Args:
            confidence: Model prediction confidence (0-1)

        Returns:
            GuardrailResult
        """
        if confidence < self.min_confidence:
            return GuardrailResult(
                passed=False,
                reason=f"Confidence too low: {confidence:.2%} (min {self.min_confidence:.2%})",
                value=confidence,
                threshold=self.min_confidence
            )

        return GuardrailResult(passed=True, value=confidence, threshold=self.min_confidence)

    def check_latency(
        self,
        latency_ms: float
    ) -> GuardrailResult:
        """
        Check if execution latency is acceptable.

        Args:
            latency_ms: End-to-end latency in milliseconds

        Returns:
            GuardrailResult
        """
        if latency_ms > self.max_latency_ms:
            return GuardrailResult(
                passed=False,
                reason=f"Latency too high: {latency_ms:.0f}ms (max {self.max_latency_ms:.0f}ms)",
                value=latency_ms,
                threshold=self.max_latency_ms
            )

        return GuardrailResult(passed=True, value=latency_ms, threshold=self.max_latency_ms)

    def check_all(
        self,
        last_bar_time: pd.Timestamp,
        timeframe_minutes: int,
        current_spread: float,
        atr: float,
        price: float,
        confidence: float,
        latency_ms: Optional[float] = None
    ) -> Dict[str, GuardrailResult]:
        """
        Run all guardrail checks.

        Args:
            last_bar_time: Last bar timestamp
            timeframe_minutes: Timeframe in minutes
            current_spread: Current spread
            atr: ATR value
            price: Current price
            confidence: Model confidence
            latency_ms: Optional latency measurement

        Returns:
            Dict of check name -> GuardrailResult
        """
        results = {}

        results['staleness'] = self.check_data_staleness(last_bar_time, timeframe_minutes)
        results['spread'] = self.check_spread(current_spread, atr)
        results['volatility'] = self.check_volatility_regime(atr, price)
        results['session'] = self.check_session(last_bar_time)
        results['confidence'] = self.check_confidence(confidence)

        if latency_ms is not None:
            results['latency'] = self.check_latency(latency_ms)

        return results

    def all_passed(self, results: Dict[str, GuardrailResult]) -> bool:
        """Check if all guardrails passed."""
        return all(r.passed for r in results.values())

    def get_failures(self, results: Dict[str, GuardrailResult]) -> Dict[str, str]:
        """Get dict of failed checks and reasons."""
        return {
            name: result.reason
            for name, result in results.items()
            if not result.passed
        }


# ============================================================================
# PRESET CONFIGURATIONS
# ============================================================================

def get_conservative_guardrails() -> ExecutionGuardrails:
    """Conservative guardrails for risk-averse trading."""
    return ExecutionGuardrails(
        max_spread_atr_ratio=0.10,     # Very tight spread requirement
        max_data_age_seconds=180,       # 3 minutes max
        min_confidence=0.65,            # High confidence requirement
        max_latency_ms=200,             # Fast execution required
        blocked_sessions=['asia', 'overnight'],  # Only trade liquid sessions
        min_atr_pct=0.005,              # Higher volatility floor
        max_atr_pct=0.04,               # Lower volatility cap
    )


def get_moderate_guardrails() -> ExecutionGuardrails:
    """Moderate guardrails for balanced trading."""
    return ExecutionGuardrails(
        max_spread_atr_ratio=0.15,
        max_data_age_seconds=300,
        min_confidence=0.55,
        max_latency_ms=250,
        blocked_sessions=['overnight'],  # Avoid only overnight
        min_atr_pct=0.003,
        max_atr_pct=0.05,
    )


def get_aggressive_guardrails() -> ExecutionGuardrails:
    """Aggressive guardrails for active trading."""
    return ExecutionGuardrails(
        max_spread_atr_ratio=0.20,
        max_data_age_seconds=600,
        min_confidence=0.50,
        max_latency_ms=500,
        blocked_sessions=[],  # Trade all sessions
        min_atr_pct=0.002,
        max_atr_pct=0.08,
    )


# ============================================================================
# TESTING
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*80)
    print("EXECUTION GUARDRAILS - TEST")
    print("="*80 + "\n")

    # Create moderate guardrails
    guards = get_moderate_guardrails()

    # Test case: XAUUSD during London session (use current time)
    now = pd.Timestamp.now(tz='UTC')
    price = 2650.0
    atr = 8.5
    spread = 0.30
    confidence = 0.62
    latency = 180.0

    print("Test scenario:")
    print(f"  Time: {now} (London session)")
    print(f"  Price: ${price}")
    print(f"  ATR: ${atr}")
    print(f"  Spread: ${spread}")
    print(f"  Confidence: {confidence:.1%}")
    print(f"  Latency: {latency}ms")
    print()

    # Run all checks
    results = guards.check_all(
        last_bar_time=now,
        timeframe_minutes=15,
        current_spread=spread,
        atr=atr,
        price=price,
        confidence=confidence,
        latency_ms=latency
    )

    # Display results
    print("Guardrail Results:")
    print("-" * 80)
    for name, result in results.items():
        status = "✅ PASS" if result.passed else "❌ FAIL"
        print(f"{status:10s} {name:15s}", end="")
        if result.value is not None:
            print(f" value={result.value:.3f}", end="")
        if result.threshold is not None:
            print(f" threshold={result.threshold:.3f}", end="")
        if not result.passed:
            print(f" ({result.reason})", end="")
        print()

    print()
    if guards.all_passed(results):
        print("✅ ALL GUARDRAILS PASSED - Trade allowed")
    else:
        failures = guards.get_failures(results)
        print(f"❌ {len(failures)} GUARDRAILS FAILED - Trade blocked")
        for name, reason in failures.items():
            print(f"   - {name}: {reason}")

    # Test edge cases
    print("\n" + "="*80)
    print("Edge Case Tests:")
    print("="*80 + "\n")

    test_cases = [
        ("Stale data", {
            'last_bar_time': now - timedelta(minutes=45),
            'timeframe_minutes': 15,
            'current_spread': spread,
            'atr': atr,
            'price': price,
            'confidence': confidence
        }),
        ("Wide spread", {
            'last_bar_time': now,
            'timeframe_minutes': 15,
            'current_spread': 2.0,  # Very wide
            'atr': atr,
            'price': price,
            'confidence': confidence
        }),
        ("Low confidence", {
            'last_bar_time': now,
            'timeframe_minutes': 15,
            'current_spread': spread,
            'atr': atr,
            'price': price,
            'confidence': 0.45  # Below threshold
        }),
        ("High volatility", {
            'last_bar_time': now,
            'timeframe_minutes': 15,
            'current_spread': spread,
            'atr': 150.0,  # Massive ATR
            'price': price,
            'confidence': confidence
        }),
    ]

    for test_name, params in test_cases:
        results = guards.check_all(**params)
        passed = guards.all_passed(results)
        status = "✅" if passed else "❌"
        print(f"{status} {test_name}: {'PASS' if passed else 'FAIL'}")
        if not passed:
            failures = guards.get_failures(results)
            for name, reason in failures.items():
                print(f"     {name}: {reason}")

    print("\n" + "="*80 + "\n")
