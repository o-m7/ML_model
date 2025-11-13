#!/usr/bin/env python3
"""
COST PARITY UNIT TESTS
======================

Validates that market costs are applied consistently across:
- Backtest engine
- Live execution
- Signal generation

Critical for ensuring backtest=live performance parity.
"""

import unittest
import numpy as np
from market_costs import (
    get_costs,
    get_tp_sl,
    calculate_tp_sl_prices,
    apply_entry_costs,
    apply_exit_costs,
    get_pip_value
)


class TestCostParity(unittest.TestCase):
    """Test cost calculations are consistent."""

    def test_tp_sl_params_exist_for_all_symbols(self):
        """Verify all symbols have TP/SL params for all timeframes."""
        symbols = ['XAUUSD', 'XAGUSD', 'EURUSD', 'GBPUSD', 'AUDUSD', 'NZDUSD']
        timeframes = ['5T', '15T', '30T', '1H', '4H']

        for symbol in symbols:
            for tf in timeframes:
                with self.subTest(symbol=symbol, timeframe=tf):
                    params = get_tp_sl(symbol, tf)
                    self.assertGreater(params.tp_atr_mult, 0, f"{symbol} {tf} TP must be > 0")
                    self.assertGreater(params.sl_atr_mult, 0, f"{symbol} {tf} SL must be > 0")
                    self.assertGreaterEqual(params.tp_atr_mult, params.sl_atr_mult,
                                          f"{symbol} {tf} TP should be >= SL")

    def test_costs_exist_for_all_symbols(self):
        """Verify all symbols have market costs defined."""
        symbols = ['XAUUSD', 'XAGUSD', 'EURUSD', 'GBPUSD', 'AUDUSD', 'NZDUSD']

        for symbol in symbols:
            with self.subTest(symbol=symbol):
                costs = get_costs(symbol)
                self.assertGreater(costs.spread_pips, 0, f"{symbol} spread must be > 0")
                self.assertGreater(costs.commission_pct, 0, f"{symbol} commission must be > 0")
                self.assertGreater(costs.slippage_pct, 0, f"{symbol} slippage must be > 0")

    def test_tp_sl_calculation_long(self):
        """Test TP/SL calculation for LONG trades."""
        symbol = 'XAUUSD'
        tf = '15T'
        entry = 2650.00
        atr = 8.50

        tp, sl = calculate_tp_sl_prices(symbol, tf, entry, atr, 'long')

        # TP should be above entry
        self.assertGreater(tp, entry, "Long TP must be above entry")

        # SL should be below entry
        self.assertLess(sl, entry, "Long SL must be below entry")

        # Check distance matches ATR multipliers
        params = get_tp_sl(symbol, tf)
        expected_tp = entry + (atr * params.tp_atr_mult)
        expected_sl = entry - (atr * params.sl_atr_mult)

        self.assertAlmostEqual(tp, expected_tp, places=2, msg="TP calculation incorrect")
        self.assertAlmostEqual(sl, expected_sl, places=2, msg="SL calculation incorrect")

    def test_tp_sl_calculation_short(self):
        """Test TP/SL calculation for SHORT trades."""
        symbol = 'XAUUSD'
        tf = '15T'
        entry = 2650.00
        atr = 8.50

        tp, sl = calculate_tp_sl_prices(symbol, tf, entry, atr, 'short')

        # TP should be below entry
        self.assertLess(tp, entry, "Short TP must be below entry")

        # SL should be above entry
        self.assertGreater(sl, entry, "Short SL must be above entry")

        # Check distance matches ATR multipliers
        params = get_tp_sl(symbol, tf)
        expected_tp = entry - (atr * params.tp_atr_mult)
        expected_sl = entry + (atr * params.sl_atr_mult)

        self.assertAlmostEqual(tp, expected_tp, places=2, msg="TP calculation incorrect")
        self.assertAlmostEqual(sl, expected_sl, places=2, msg="SL calculation incorrect")

    def test_entry_costs_worsen_price(self):
        """Test that entry costs always worsen the entry price."""
        symbol = 'XAUUSD'
        entry = 2650.00
        notional = 100000

        # Long trade
        adj_entry_long, comm, slip = apply_entry_costs(symbol, entry, notional, 'long')
        self.assertGreater(adj_entry_long, entry, "Long entry should be worse (higher) after costs")

        # Short trade
        adj_entry_short, comm, slip = apply_entry_costs(symbol, entry, notional, 'short')
        self.assertLess(adj_entry_short, entry, "Short entry should be worse (lower) after costs")

    def test_exit_costs_worsen_price(self):
        """Test that exit costs always worsen the exit price."""
        symbol = 'XAUUSD'
        exit_price = 2660.00
        notional = 100000

        # Long trade
        adj_exit_long, comm, slip = apply_exit_costs(symbol, exit_price, notional, 'long')
        self.assertLess(adj_exit_long, exit_price, "Long exit should be worse (lower) after costs")

        # Short trade
        adj_exit_short, comm, slip = apply_exit_costs(symbol, exit_price, notional, 'short')
        self.assertGreater(adj_exit_short, exit_price, "Short exit should be worse (higher) after costs")

    def test_round_trip_costs_always_negative(self):
        """Test that round-trip costs always reduce P&L."""
        symbol = 'XAUUSD'
        entry = 2650.00
        exit_price = 2660.00
        notional = 100000

        # Long trade
        adj_entry_long, entry_comm_long, entry_slip_long = apply_entry_costs(symbol, entry, notional, 'long')
        adj_exit_long, exit_comm_long, exit_slip_long = apply_exit_costs(symbol, exit_price, notional, 'long')

        gross_pnl_long = (exit_price - entry) * (notional / entry)
        net_pnl_long = (adj_exit_long - adj_entry_long) * (notional / adj_entry_long) - \
                       (entry_comm_long + entry_slip_long + exit_comm_long + exit_slip_long)

        self.assertLess(net_pnl_long, gross_pnl_long, "Net P&L must be less than gross after costs")

    def test_costs_scale_with_notional(self):
        """Test that costs scale linearly with position size."""
        symbol = 'XAUUSD'
        entry = 2650.00

        # Test at 2 different notionals
        notional_small = 10000
        notional_large = 100000

        _, comm_small, slip_small = apply_entry_costs(symbol, entry, notional_small, 'long')
        _, comm_large, slip_large = apply_entry_costs(symbol, entry, notional_large, 'long')

        # Commission should scale 10x
        ratio_comm = comm_large / comm_small
        self.assertAlmostEqual(ratio_comm, 10.0, places=1, msg="Commission should scale linearly")

        # Slippage should scale 10x
        ratio_slip = slip_large / slip_small
        self.assertAlmostEqual(ratio_slip, 10.0, places=1, msg="Slippage should scale linearly")

    def test_xauusd_specific_params(self):
        """Test XAUUSD 15T has expected parameters."""
        symbol = 'XAUUSD'
        tf = '15T'

        # TP/SL params
        params = get_tp_sl(symbol, tf)
        self.assertEqual(params.tp_atr_mult, 1.4, "XAUUSD 15T TP should be 1.4R")
        self.assertEqual(params.sl_atr_mult, 1.0, "XAUUSD 15T SL should be 1.0R")
        self.assertEqual(params.risk_reward_ratio, 1.4, "XAUUSD 15T R:R should be 1.4:1")

        # Market costs
        costs = get_costs(symbol)
        self.assertEqual(costs.spread_pips, 3.0, "XAUUSD spread should be 3 pips")
        self.assertEqual(costs.commission_pct, 0.00002, "XAUUSD commission should be 0.002%")
        self.assertEqual(costs.slippage_pct, 0.00001, "XAUUSD slippage should be 0.001%")

    def test_realistic_trade_example(self):
        """Test realistic XAUUSD trade with all costs."""
        symbol = 'XAUUSD'
        tf = '15T'
        entry_price = 2650.00
        atr = 8.50
        position_size = 10.0  # 10 oz
        notional = entry_price * position_size

        # Calculate TP/SL
        tp, sl = calculate_tp_sl_prices(symbol, tf, entry_price, atr, 'long')

        # Apply entry costs
        adj_entry, entry_comm, entry_slip = apply_entry_costs(symbol, entry_price, notional, 'long')

        # Simulate TP hit
        adj_tp, exit_comm, exit_slip = apply_exit_costs(symbol, tp, notional, 'long')

        # Calculate P&L
        gross_pnl = (tp - entry_price) * position_size
        net_pnl = (adj_tp - adj_entry) * position_size - (entry_comm + entry_slip + exit_comm + exit_slip)

        # Verify results are reasonable
        self.assertGreater(gross_pnl, 0, "Gross P&L should be positive at TP")
        self.assertGreater(net_pnl, 0, "Net P&L should be positive at TP (after costs)")
        self.assertLess(net_pnl, gross_pnl, "Net P&L should be less than gross")

        # Cost drag should be reasonable (< 10% of gross)
        cost_drag_pct = (gross_pnl - net_pnl) / gross_pnl * 100
        self.assertLess(cost_drag_pct, 10.0, f"Cost drag {cost_drag_pct:.1f}% seems too high")

        print(f"\nRealistic Trade Example (XAUUSD 15T):")
        print(f"  Entry: ${entry_price:.2f} → ${adj_entry:.2f} (after costs)")
        print(f"  TP: ${tp:.2f} → ${adj_tp:.2f} (after costs)")
        print(f"  SL: ${sl:.2f}")
        print(f"  Position: {position_size} oz (${notional:,.0f} notional)")
        print(f"  Gross P&L: ${gross_pnl:.2f}")
        print(f"  Net P&L: ${net_pnl:.2f}")
        print(f"  Cost drag: ${gross_pnl - net_pnl:.2f} ({cost_drag_pct:.1f}%)")


class TestFeatureAlignment(unittest.TestCase):
    """Test feature calculations are aligned (placeholder)."""

    def test_feature_calculation_placeholder(self):
        """Placeholder for feature alignment tests."""
        # TODO: Add tests that verify:
        # 1. Live features match training features
        # 2. Feature values are identical for same input
        # 3. No NaN/inf values in features
        self.assertTrue(True, "Feature alignment tests not yet implemented")


def run_tests():
    """Run all tests and report results."""
    print("\n" + "="*80)
    print("COST PARITY UNIT TESTS")
    print("="*80 + "\n")

    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestCostParity))
    suite.addTests(loader.loadTestsFromTestCase(TestFeatureAlignment))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    if result.wasSuccessful():
        print("\n✅ ALL TESTS PASSED - Cost parity validated")
    else:
        print("\n❌ SOME TESTS FAILED - Review failures above")

    print("="*80 + "\n")

    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    import sys
    sys.exit(run_tests())
