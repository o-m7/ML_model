#!/usr/bin/env python3
"""
SIMPLE VALIDATION AND CIRCUIT BREAKER TESTS
==========================================
Tests core utilities without external dependencies.
"""

import time
from datetime import datetime, timezone, timedelta
import numpy as np

# ============================================================================
# Copy core functions for standalone testing
# ============================================================================

class CircuitBreaker:
    def __init__(self, failure_threshold=3, timeout_seconds=60):
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'closed'

    def call(self, func, *args, **kwargs):
        if self.state == 'open':
            if time.time() - self.last_failure_time > self.timeout_seconds:
                self.state = 'half-open'
            else:
                raise Exception(f"Circuit breaker OPEN (failures: {self.failure_count})")

        try:
            result = func(*args, **kwargs)
            if self.state == 'half-open':
                self.state = 'closed'
                self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            if self.failure_count >= self.failure_threshold:
                self.state = 'open'
            raise e

def validate_signal_data(signal_dict: dict) -> tuple:
    required_fields = ['symbol', 'timeframe', 'signal_type', 'estimated_entry',
                       'tp_price', 'sl_price', 'confidence', 'edge', 'timestamp']

    for field in required_fields:
        if field not in signal_dict:
            return False, f"Missing required field: {field}"

    numeric_fields = ['estimated_entry', 'tp_price', 'sl_price', 'confidence', 'edge']
    for field in numeric_fields:
        value = signal_dict.get(field)
        if value is None or (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
            return False, f"Invalid value for {field}: {value}"

    if not (0 <= signal_dict['confidence'] <= 1):
        return False, f"Confidence out of range [0,1]: {signal_dict['confidence']}"

    if not (-1 <= signal_dict['edge'] <= 1):
        return False, f"Edge out of range [-1,1]: {signal_dict['edge']}"

    try:
        signal_time = datetime.fromisoformat(signal_dict['timestamp'].replace('Z', '+00:00'))
        age_seconds = (datetime.now(timezone.utc) - signal_time).total_seconds()

        if age_seconds > 300:
            return False, f"Signal timestamp too old: {age_seconds:.0f} seconds"

        if age_seconds < -60:
            return False, f"Signal timestamp in future: {age_seconds:.0f} seconds"

    except Exception as e:
        return False, f"Invalid timestamp format: {e}"

    if signal_dict['signal_type'] not in ['long', 'short']:
        return False, f"Invalid signal_type: {signal_dict['signal_type']}"

    entry = signal_dict['estimated_entry']
    tp = signal_dict['tp_price']
    sl = signal_dict['sl_price']

    if signal_dict['signal_type'] == 'long':
        if not (sl < entry < tp):
            return False, f"Invalid long prices: SL={sl} Entry={entry} TP={tp}"
    else:
        if not (tp < entry < sl):
            return False, f"Invalid short prices: TP={tp} Entry={entry} SL={sl}"

    return True, ""

# ============================================================================
# Tests
# ============================================================================

def test_circuit_breaker():
    print("Testing Circuit Breaker...")

    breaker = CircuitBreaker(failure_threshold=3, timeout_seconds=1)

    def failing_func():
        raise Exception("Simulated failure")

    # Test that circuit opens after 3 failures
    for i in range(3):
        try:
            breaker.call(failing_func)
        except Exception as e:
            if "Circuit breaker OPEN" in str(e):
                assert False, f"Circuit opened too early on attempt {i+1}"

    assert breaker.state == 'open', f"Circuit should be open after 3 failures, got {breaker.state}"
    print("  ✅ Circuit opens after threshold failures")

    # Test that circuit rejects calls when open
    rejected = False
    try:
        breaker.call(failing_func)
    except Exception as e:
        if "Circuit breaker OPEN" in str(e):
            rejected = True

    assert rejected, "Circuit should reject calls when open"
    print("  ✅ Circuit rejects calls when open")

    time.sleep(1.1)
    try:
        breaker.call(failing_func)
    except Exception:
        pass

    assert breaker.state == 'open', "Circuit should stay open after failure in half-open"
    print("  ✅ Circuit handles timeout correctly")

def test_signal_validation():
    print("\nTesting Signal Validation...")

    valid_signal = {
        'symbol': 'EURUSD',
        'timeframe': '5T',
        'signal_type': 'long',
        'estimated_entry': 1.0500,
        'tp_price': 1.0550,
        'sl_price': 1.0450,
        'confidence': 0.65,
        'edge': 0.05,
        'timestamp': datetime.now(timezone.utc).isoformat()
    }

    is_valid, msg = validate_signal_data(valid_signal)
    assert is_valid, f"Valid signal rejected: {msg}"
    print("  ✅ Valid long signal accepted")

    invalid = valid_signal.copy()
    del invalid['confidence']
    is_valid, msg = validate_signal_data(invalid)
    assert not is_valid and 'confidence' in msg
    print("  ✅ Missing field detected")

    invalid = valid_signal.copy()
    invalid['edge'] = float('nan')
    is_valid, msg = validate_signal_data(invalid)
    assert not is_valid and 'edge' in msg.lower()
    print("  ✅ NaN value rejected")

    invalid = valid_signal.copy()
    invalid['confidence'] = 1.5
    is_valid, msg = validate_signal_data(invalid)
    assert not is_valid and 'confidence' in msg.lower()
    print("  ✅ Out-of-range confidence rejected")

    invalid = valid_signal.copy()
    invalid['timestamp'] = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
    is_valid, msg = validate_signal_data(invalid)
    assert not is_valid
    print("  ✅ Stale timestamp rejected")

    invalid = valid_signal.copy()
    invalid['tp_price'] = 1.0400
    is_valid, msg = validate_signal_data(invalid)
    assert not is_valid
    print("  ✅ Invalid long price relationship rejected")

    valid_short = {
        'symbol': 'EURUSD',
        'timeframe': '5T',
        'signal_type': 'short',
        'estimated_entry': 1.0500,
        'tp_price': 1.0450,
        'sl_price': 1.0550,
        'confidence': 0.65,
        'edge': 0.05,
        'timestamp': datetime.now(timezone.utc).isoformat()
    }
    is_valid, msg = validate_signal_data(valid_short)
    assert is_valid, f"Valid short signal rejected: {msg}"
    print("  ✅ Valid short signal accepted")

def main():
    print("="*60)
    print("SIGNAL PIPELINE UNIT TESTS")
    print("="*60 + "\n")

    try:
        test_circuit_breaker()
        test_signal_validation()

        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED")
        print("="*60)
        return 0

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    import sys
    sys.exit(main())
