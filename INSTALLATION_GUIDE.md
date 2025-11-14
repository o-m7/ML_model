# 📦 INSTALLATION GUIDE - Live Trading Fixes

Step-by-step guide to install all dependencies for the live trading fixes.

---

## 🚀 QUICK START (Recommended)

### **Option 1: Install Everything (Recommended)**

```bash
# Navigate to your ML_model directory
cd /path/to/ML_model

# Install all dependencies
pip install -r requirements_fixes.txt

# Verify installation
python -c "import numpy, pandas, matplotlib; print('✅ All dependencies installed')"

# Run tests
python test_cost_parity.py
```

---

### **Option 2: Install Minimal (Just the Fixes)**

```bash
# Install only what's needed for the new modules
pip install -r requirements_minimal.txt

# Verify
python test_cost_parity.py
```

---

## 🐍 PYTHON VERSION REQUIREMENTS

- **Minimum:** Python 3.8+
- **Recommended:** Python 3.10+
- **Tested on:** Python 3.10, 3.11

Check your Python version:
```bash
python --version
# or
python3 --version
```

If you need to install Python:
- **macOS:** `brew install python@3.10`
- **Ubuntu/Debian:** `sudo apt install python3.10`
- **Windows:** Download from [python.org](https://www.python.org/downloads/)

---

## 🌐 VIRTUAL ENVIRONMENT (Highly Recommended)

Using a virtual environment prevents conflicts with system packages.

### **Create Virtual Environment**

```bash
# Create venv
python -m venv venv

# Activate it
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate     # Windows

# Your prompt should now show (venv)
```

### **Install Dependencies in venv**

```bash
# With venv activated:
pip install -r requirements_fixes.txt

# Verify
which python  # Should show path inside venv/
```

### **Deactivate When Done**

```bash
deactivate
```

---

## 📋 MANUAL INSTALLATION (Alternative)

If requirements files don't work, install manually:

### **Core Dependencies (Required)**

```bash
pip install numpy>=1.24.0
pip install pandas>=2.0.0
pip install matplotlib>=3.7.0
pip install python-dateutil
pip install pytz
```

### **ML Dependencies (For Models)**

```bash
pip install scikit-learn>=1.3.0
pip install xgboost>=2.0.0
pip install lightgbm>=4.0.0
```

### **Live Trading Dependencies (For Existing System)**

```bash
pip install pandas-ta>=0.3.14b
pip install requests>=2.31.0
pip install python-dotenv>=1.0.0
pip install supabase>=2.0.0
```

---

## ✅ VERIFY INSTALLATION

After installation, verify everything works:

### **1. Check Core Imports**

```bash
python -c "
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
print('✅ Core dependencies OK')
"
```

### **2. Run Unit Tests**

```bash
python test_cost_parity.py
# Should show: ✅ ALL TESTS PASSED
```

### **3. Test Market Costs Module**

```bash
python market_costs.py
# Should show: ✅ All X symbols validated
```

### **4. Test Execution Guardrails**

```bash
python execution_guardrails.py
# Should show: Guardrail test results
```

---

## 🔧 TROUBLESHOOTING

### **Issue: `pip` not found**

```bash
# Try:
python -m pip install -r requirements_fixes.txt

# Or install pip:
python -m ensurepip --upgrade
```

### **Issue: Permission denied**

```bash
# Don't use sudo! Use virtual environment instead:
python -m venv venv
source venv/bin/activate
pip install -r requirements_fixes.txt
```

### **Issue: Package conflicts**

```bash
# Upgrade pip first:
pip install --upgrade pip

# Then try again:
pip install -r requirements_fixes.txt
```

### **Issue: `numpy` build errors**

```bash
# Install pre-built wheel:
pip install --only-binary :all: numpy

# Or on macOS with M1/M2:
pip install --pre --upgrade numpy
```

### **Issue: `pandas` too old**

```bash
# Upgrade pandas:
pip install --upgrade pandas>=2.0.0
```

### **Issue: `matplotlib` errors on macOS**

```bash
# Install dependencies:
brew install pkg-config
pip install matplotlib
```

---

## 📦 DEPENDENCIES EXPLAINED

### **What Each Module Needs:**

| Module | Dependencies |
|--------|--------------|
| `market_costs.py` | numpy, pandas (for types) |
| `execution_guardrails.py` | numpy, pandas |
| `test_cost_parity.py` | numpy, pandas, unittest (stdlib) |
| `calibrate_thresholds.py` | numpy, pandas, matplotlib, pickle (stdlib) |
| `validate_backtest_with_costs.py` | numpy, pandas, pickle (stdlib) |

### **Why We Need Each Package:**

- **numpy:** Fast numerical operations, array handling
- **pandas:** DataFrames for time-series data
- **matplotlib:** Plotting for calibrate_thresholds.py (optional)
- **scikit-learn:** ML models (XGBoost, LightGBM wrappers)
- **xgboost/lightgbm:** Gradient boosting models
- **pandas-ta:** Technical indicators (existing system)
- **requests:** API calls to Polygon (existing system)
- **supabase:** Database client (existing system)
- **python-dotenv:** Environment variables (existing system)

---

## 🔒 PRODUCTION INSTALLATION

For production deployment, pin exact versions:

```bash
# After testing in dev, freeze versions:
pip freeze > requirements_frozen.txt

# Then in production:
pip install -r requirements_frozen.txt
```

---

## 🐳 DOCKER INSTALLATION (Optional)

If you use Docker:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements_fixes.txt .
RUN pip install --no-cache-dir -r requirements_fixes.txt

COPY . .

CMD ["python", "live_trading_engine_fixed.py"]
```

Build and run:
```bash
docker build -t ml-trading-fixes .
docker run -it ml-trading-fixes
```

---

## 📊 SYSTEM REQUIREMENTS

### **Minimum:**
- Python 3.8+
- 512 MB RAM
- 100 MB disk space

### **Recommended:**
- Python 3.10+
- 2 GB RAM
- 1 GB disk space (for data)
- SSD for faster data loading

---

## 🚀 QUICK VERIFICATION SCRIPT

Create a file `verify_install.py`:

```python
#!/usr/bin/env python3
"""Verify all dependencies are installed correctly."""

import sys

def check_import(module_name, package_name=None):
    """Try importing a module."""
    package_name = package_name or module_name
    try:
        __import__(module_name)
        print(f"✅ {package_name}")
        return True
    except ImportError as e:
        print(f"❌ {package_name} - {e}")
        return False

print("\n" + "="*60)
print("DEPENDENCY CHECK")
print("="*60 + "\n")

checks = [
    ('numpy', 'numpy'),
    ('pandas', 'pandas'),
    ('matplotlib', 'matplotlib'),
    ('sklearn', 'scikit-learn'),
    ('xgboost', 'xgboost'),
    ('lightgbm', 'lightgbm'),
    ('pandas_ta', 'pandas-ta'),
    ('requests', 'requests'),
    ('dotenv', 'python-dotenv'),
]

results = [check_import(mod, pkg) for mod, pkg in checks]

print("\n" + "="*60)
if all(results):
    print("🎉 ALL DEPENDENCIES INSTALLED")
else:
    print("⚠️  SOME DEPENDENCIES MISSING")
    print("\nRun: pip install -r requirements_fixes.txt")
print("="*60 + "\n")

sys.exit(0 if all(results) else 1)
```

Run it:
```bash
python verify_install.py
```

---

## 📞 SUPPORT

If you're still having issues:

1. **Check Python version:** `python --version`
2. **Check pip version:** `pip --version`
3. **Try upgrading pip:** `pip install --upgrade pip`
4. **Use virtual environment:** See section above
5. **Check error messages carefully** - they usually tell you what's missing

---

## ✅ INSTALLATION CHECKLIST

After installation, verify:

- [ ] Python 3.8+ installed
- [ ] Virtual environment created (recommended)
- [ ] `pip install -r requirements_fixes.txt` completed
- [ ] `python test_cost_parity.py` passes all tests
- [ ] `python market_costs.py` runs without errors
- [ ] `python execution_guardrails.py` runs without errors
- [ ] No import errors when running any module

---

**Last Updated:** 2025-11-13
**Python Version:** 3.8 minimum, 3.10+ recommended
**Status:** ✅ Ready for installation
