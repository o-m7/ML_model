# Signal Staleness Root Cause and Fix

## Problem
XAUUSD and XAGUSD signals are 4+ hours stale (last update: 14:45 UTC, current: 19:12 UTC) despite markets being open.

## Root Cause Identified ✅

**GitHub Actions scheduled workflows (cron) only run from the DEFAULT branch (main).**

Current situation:
- All fixes are on feature branch: `claude/fix-xauusd-live-performance-011CV5BJZ1fGqmQxGx59ZHKL`
- Workflow file `.github/workflows/generate_signals.yml` exists on BOTH branches
- **The main branch has the OLD broken version**
- **The feature branch has the FIXED version**
- **Cron schedule (`* * * * *`) does NOT work on feature branches**

This is why:
1. Signals stopped generating automatically at 14:45 UTC
2. All our fixes work perfectly (verified locally)
3. But GitHub Actions isn't running the workflow every minute

## The Fix

**Merge this feature branch to main** to activate the cron schedule with all fixes.

### Option 1: Create Pull Request (Recommended)

```bash
# Create PR to main (requires gh CLI or GitHub web UI)
gh pr create --title "Fix XAUUSD/XAGUSD signal generation" --base main

# Or use GitHub web UI:
# https://github.com/o-m7/ML_model/compare/main...claude/fix-xauusd-live-performance-011CV5BJZ1fGqmQxGx59ZHKL
```

### Option 2: Direct Merge to Main (if you have permissions)

```bash
git checkout main
git merge claude/fix-xauusd-live-performance-011CV5BJZ1fGqmQxGx59ZHKL
git push origin main
```

## Critical Fixes in This Branch

### 1. **Model Loading Failures (XAUUSD/XAGUSD)**
- **Issue**: `Can't get attribute 'BalancedModel'` when unpickling XAGUSD models
- **Fix**: Import BalancedModel FIRST in signal_generator.py and ensemble_predictor.py
- **Files**: `signal_generator.py`, `ensemble_predictor.py`

### 2. **Workflow Concurrency Conflicts**
- **Issue**: "Canceling since a higher priority waiting request exists"
- **Fix**: Added concurrency control with `cancel-in-progress: false`
- **Files**: `.github/workflows/generate_signals.yml`

### 3. **Parameter Mismatches**
- **Issue**: market_costs.py had outdated TP values vs trained models
- **Fix**: Synchronized XAUUSD parameters across all configs
- **Files**: `market_costs.py`

### 4. **Missing Diagnostics**
- **Issue**: Silent failures, no visibility into what's failing
- **Fix**: Created comprehensive diagnostic script
- **Files**: `diagnose_signal_generation.py`, updated workflow

### 5. **XAGUSD Marginal Performance**
- **Issue**: 30T/1H/4H timeframes had poor returns (10.8%, 4.8%, 3.5%)
- **Fix**: Optimized parameters (lower min_conf, higher TP ratios)
- **Files**: `models_onnx/XAGUSD/*.json`, `production_final_system.py`

### 6. **Adaptive Learning System**
- **Issue**: No mechanism to prevent model staleness
- **Fix**: Added weekly retraining, signal auditing, online learning
- **Files**: `signal_auditor.py`, `adaptive_retraining.py`, `weekly_retraining.yml`

## Verification Checklist

After merging to main, verify within 10 minutes:

- [ ] GitHub Actions workflow runs automatically (check Actions tab)
- [ ] Diagnostic step shows all checks passing
- [ ] Signal generation completes successfully
- [ ] Fresh signals appear in Supabase `live_signals` table
- [ ] model_status_dashboard.py shows signals < 5 minutes old
- [ ] No workflow cancellation errors

## Expected Timeline

Once merged to main:
- **T+0 minutes**: Merge completes
- **T+1 minute**: First workflow run triggered by cron
- **T+2-3 minutes**: Workflow completes (install deps, generate signals)
- **T+5 minutes**: Fresh signals visible in Supabase
- **T+10 minutes**: All 22 model signals updated

## Troubleshooting

If signals are still stale after merge:

1. **Check GitHub Actions logs** for the workflow run
2. **Run diagnostic script manually**:
   ```bash
   python3 diagnose_signal_generation.py
   ```
3. **Check for errors in signal generation**:
   ```bash
   python3 signal_generator.py
   ```
4. **Verify Polygon API** is returning fresh data (check API status)
5. **Check Supabase write permissions** for the service account

## Summary

The fix is simple but critical: **merge to main**. All code changes are complete and verified. The only blocker is that GitHub Actions won't run scheduled workflows from feature branches.

---

**All commits ready for merge:**
- `8b09601` Add comprehensive diagnostics and error handling
- `22c8179` Fix XAUUSD parameter consistency with trained models
- `8b6f338` Fix XAUUSD/XAGUSD signal generation and workflow concurrency issues
- `b2c7451` Add adaptive learning system for continuous model improvement
- `1242427` Add monitoring tools and complete XAUUSD/XAGUSD parameter standardization

Branch: `claude/fix-xauusd-live-performance-011CV5BJZ1fGqmQxGx59ZHKL`
Target: `main`
