#!/usr/bin/env python3
"""
QA/AUDIT FRAMEWORK - Pre-Training Data Quality Checks
======================================================

Automated checks for leakage, data integrity, and cost modeling.
Fail fast with actionable errors before any model training begins.

Usage:
    from qa.audit import DataAuditor
    auditor = DataAuditor(config)
    auditor.audit_all(df)  # Raises on any failure
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from pathlib import Path
import yaml
import warnings


@dataclass
class AuditConfig:
    """Configuration for data auditing."""
    
    # Timestamp checks
    require_utc: bool = True
    allow_duplicates: bool = False
    
    # Lookahead guards
    embargo_bars: int = 100
    forecast_horizon: int = 40
    
    # Cost modeling
    spread_pct_metals: float = 0.0002  # 2 bps for metals
    spread_pct_fx: float = 0.00002     # 0.2 bps for FX
    commission_pct: float = 0.00001    # 1 bp
    slippage_pct: float = 0.000005     # 0.5 bp
    
    # Outlier detection
    z_score_threshold: float = 5.0
    spread_percentile_cap: float = 99.0
    
    # Session filters (UTC hours)
    sessions: Dict[str, List[Tuple[int, int]]] = field(default_factory=lambda: {
        'XAUUSD': [(7, 22)],     # London open to NY close
        'XAGUSD': [(7, 22)],
        'EURUSD': [(7, 22)],     # London/NY overlap critical
        'GBPUSD': [(7, 22)],
        'AUDUSD': [(22, 7)],     # Asia/Sydney session
        'NZDUSD': [(22, 7)],
        'USDJPY': [(0, 24)],     # 24/7 but avoid weekends
        'USDCAD': [(12, 22)],    # NA hours
    })
    
    @classmethod
    def from_yaml(cls, path: Path) -> 'AuditConfig':
        """Load config from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data.get('audit', {}))


class AuditError(Exception):
    """Raised when audit check fails."""
    pass


class DataAuditor:
    """Automated data quality auditor."""
    
    def __init__(self, config: AuditConfig = None):
        self.config = config or AuditConfig()
        self.warnings = []
        self.errors = []
    
    def audit_all(self, df: pd.DataFrame, symbol: str = None, 
                  features: List[str] = None) -> Dict:
        """
        Run all audit checks. Raises AuditError on critical failures.
        
        Returns:
            Dict with audit results and warnings
        """
        print(f"\n{'='*80}")
        print(f"AUDIT: {symbol if symbol else 'Data'}")
        print(f"{'='*80}\n")
        
        results = {}
        
        # 1. Timestamp & TZ checks
        print("[1/7] Auditing timestamps...")
        results['timestamps'] = self._audit_timestamps(df)
        
        # 2. Duplicate bars
        print("[2/7] Checking for duplicates...")
        results['duplicates'] = self._audit_duplicates(df)
        
        # 3. Lookahead guards
        print("[3/7] Checking lookahead guards...")
        results['lookahead'] = self._audit_lookahead(df)
        
        # 4. Spreads & costs
        print("[4/7] Validating cost model...")
        results['costs'] = self._audit_costs(df, symbol)
        
        # 5. Session filters
        if symbol:
            print("[5/7] Validating session filters...")
            results['sessions'] = self._audit_sessions(df, symbol)
        
        # 6. Outliers
        print("[6/7] Detecting outliers...")
        results['outliers'] = self._audit_outliers(df, features)
        
        # 7. NA values
        print("[7/7] Checking for missing values...")
        results['missing'] = self._audit_missing(df, features)
        
        # Report
        print(f"\n{'='*80}")
        if self.errors:
            print(f"❌ AUDIT FAILED - {len(self.errors)} errors")
            for error in self.errors:
                print(f"  ❌ {error}")
            raise AuditError(f"Audit failed with {len(self.errors)} errors")
        
        if self.warnings:
            print(f"⚠️  {len(self.warnings)} warnings")
            for warning in self.warnings:
                print(f"  ⚠️  {warning}")
        
        print("✅ AUDIT PASSED")
        print(f"{'='*80}\n")
        
        return results
    
    def _audit_timestamps(self, df: pd.DataFrame) -> Dict:
        """Check timestamp integrity."""
        if 'timestamp' not in df.columns:
            self.errors.append("Missing 'timestamp' column")
            return {'passed': False}
        
        ts = pd.to_datetime(df['timestamp'])
        
        # Check UTC
        if self.config.require_utc:
            if ts.dt.tz is None:
                self.errors.append("Timestamps not timezone-aware")
            elif str(ts.dt.tz) != 'UTC':
                self.errors.append(f"Timestamps not UTC (found: {ts.dt.tz})")
        
        # Check strictly increasing
        if not ts.is_monotonic_increasing:
            diffs = ts.diff()
            negative_count = (diffs < pd.Timedelta(0)).sum()
            self.errors.append(f"Timestamps not strictly increasing ({negative_count} reversals)")
        
        # Check for large gaps (>7 days)
        gaps = ts.diff()
        large_gaps = gaps[gaps > pd.Timedelta(days=7)]
        if len(large_gaps) > 0:
            self.warnings.append(f"Found {len(large_gaps)} gaps >7 days")
        
        return {
            'passed': len(self.errors) == 0,
            'count': len(df),
            'start': ts.min(),
            'end': ts.max(),
            'gaps_7d': len(large_gaps)
        }
    
    def _audit_duplicates(self, df: pd.DataFrame) -> Dict:
        """Check for duplicate timestamps."""
        if 'timestamp' not in df.columns:
            return {'passed': False, 'duplicates': 0}
        
        dupes = df['timestamp'].duplicated().sum()
        
        if dupes > 0 and not self.config.allow_duplicates:
            self.errors.append(f"Found {dupes} duplicate timestamps (run de-duplication)")
        
        return {
            'passed': dupes == 0 or self.config.allow_duplicates,
            'duplicates': dupes
        }
    
    def _audit_lookahead(self, df: pd.DataFrame) -> Dict:
        """Check for lookahead bias in labels."""
        checks = []
        
        # Check if labels exist near end of data
        if 'target' in df.columns:
            horizon = self.config.forecast_horizon
            last_labels = df['target'].iloc[-horizon:]
            non_zero = (last_labels != 0).sum()
            
            if non_zero > horizon * 0.2:  # >20% of last horizon bars labeled
                self.warnings.append(
                    f"Last {horizon} bars have {non_zero} non-zero labels (possible lookahead)"
                )
                checks.append('labels_near_end')
        
        # Check for future-dated features (if timestamp in column names)
        future_cols = [col for col in df.columns 
                      if 'future' in col.lower() or 'forward' in col.lower()]
        if future_cols:
            self.errors.append(f"Found suspicious future-looking features: {future_cols[:5]}")
            checks.append('future_features')
        
        return {
            'passed': 'future_features' not in checks,
            'checks': checks
        }
    
    def _audit_costs(self, df: pd.DataFrame, symbol: str = None) -> Dict:
        """Validate cost model."""
        if not symbol:
            return {'passed': True, 'symbol': None}
        
        # Determine spread based on asset class
        is_metal = symbol in ['XAUUSD', 'XAGUSD']
        expected_spread_pct = (self.config.spread_pct_metals if is_metal 
                              else self.config.spread_pct_fx)
        
        # Check if spread column exists
        has_spread_col = 'spread' in df.columns or 'bid_ask_spread' in df.columns
        
        if not has_spread_col:
            self.warnings.append(f"No spread column found, using default {expected_spread_pct*10000:.1f} bps")
        
        # Estimate spread from bid/ask if available
        if 'bid' in df.columns and 'ask' in df.columns:
            actual_spread_pct = ((df['ask'] - df['bid']) / df['close']).median()
            if abs(actual_spread_pct - expected_spread_pct) > expected_spread_pct * 0.5:
                self.warnings.append(
                    f"Actual spread ({actual_spread_pct*10000:.1f} bps) differs from config "
                    f"({expected_spread_pct*10000:.1f} bps) by >50%"
                )
        
        total_cost_pct = (expected_spread_pct + self.config.commission_pct + 
                         self.config.slippage_pct)
        
        return {
            'passed': True,
            'spread_pct': expected_spread_pct,
            'commission_pct': self.config.commission_pct,
            'slippage_pct': self.config.slippage_pct,
            'total_cost_pct': total_cost_pct,
            'total_cost_bps': total_cost_pct * 10000
        }
    
    def _audit_sessions(self, df: pd.DataFrame, symbol: str) -> Dict:
        """Validate session filters."""
        if 'timestamp' not in df.columns:
            return {'passed': False}
        
        if symbol not in self.config.sessions:
            self.warnings.append(f"No session filter defined for {symbol}")
            return {'passed': True, 'filtered': False}
        
        ts = pd.to_datetime(df['timestamp'])
        hour = ts.dt.hour
        
        # Check if data falls within valid sessions
        sessions = self.config.sessions[symbol]
        in_session = np.zeros(len(df), dtype=bool)
        
        for start_hour, end_hour in sessions:
            if start_hour < end_hour:
                in_session |= (hour >= start_hour) & (hour < end_hour)
            else:  # Wraps midnight
                in_session |= (hour >= start_hour) | (hour < end_hour)
        
        pct_in_session = in_session.mean() * 100
        
        if pct_in_session < 50:
            self.warnings.append(
                f"Only {pct_in_session:.1f}% of bars in valid session for {symbol}"
            )
        
        return {
            'passed': True,
            'sessions': sessions,
            'pct_in_session': pct_in_session,
            'bars_out_of_session': (~in_session).sum()
        }
    
    def _audit_outliers(self, df: pd.DataFrame, features: List[str] = None) -> Dict:
        """Detect and handle outliers."""
        if features is None:
            # Auto-detect numeric features
            features = df.select_dtypes(include=[np.number]).columns.tolist()
            features = [f for f in features if f not in 
                       ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'target']]
        
        if not features:
            return {'passed': True, 'features_checked': 0}
        
        # Sample features to check (don't check all 243 features)
        sample_features = features[:min(50, len(features))]
        
        outlier_counts = {}
        for feat in sample_features:
            if feat not in df.columns:
                continue
            
            values = df[feat].dropna()
            if len(values) == 0:
                continue
            
            # Z-score based outlier detection
            z_scores = np.abs((values - values.mean()) / (values.std() + 1e-10))
            outliers = z_scores > self.config.z_score_threshold
            
            if outliers.sum() > len(values) * 0.01:  # >1% outliers
                outlier_counts[feat] = outliers.sum()
        
        if outlier_counts:
            top_5 = sorted(outlier_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            self.warnings.append(
                f"Found features with >1% outliers: {dict(top_5)}"
            )
        
        return {
            'passed': True,
            'features_checked': len(sample_features),
            'features_with_outliers': len(outlier_counts),
            'top_outlier_features': dict(list(outlier_counts.items())[:5]) if outlier_counts else {}
        }
    
    def _audit_missing(self, df: pd.DataFrame, features: List[str] = None) -> Dict:
        """Check for missing values in model inputs."""
        if features is None:
            # Check all numeric columns
            features = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not features:
            return {'passed': True, 'features_checked': 0}
        
        # Check for NA in features
        missing_counts = {}
        for feat in features:
            if feat not in df.columns:
                continue
            
            na_count = df[feat].isna().sum()
            if na_count > 0:
                missing_counts[feat] = na_count
        
        if missing_counts:
            total_missing = sum(missing_counts.values())
            pct_missing = (total_missing / (len(df) * len(features))) * 100
            
            if pct_missing > 5:
                self.errors.append(
                    f"Found {pct_missing:.2f}% missing values in features"
                )
            else:
                self.warnings.append(
                    f"Found {total_missing:,} missing values ({pct_missing:.2f}%) - will fillna(0)"
                )
        
        return {
            'passed': len(missing_counts) == 0 or all(v < len(df) * 0.05 for v in missing_counts.values()),
            'features_checked': len(features),
            'features_with_missing': len(missing_counts),
            'total_missing_values': sum(missing_counts.values()) if missing_counts else 0
        }


def deduplicate_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate timestamps (keep first occurrence)."""
    if 'timestamp' not in df.columns:
        return df
    
    initial_len = len(df)
    df = df.drop_duplicates(subset='timestamp', keep='first')
    removed = initial_len - len(df)
    
    if removed > 0:
        print(f"  De-duplicated: removed {removed} duplicate timestamps")
    
    return df.reset_index(drop=True)


def winsorize_features(df: pd.DataFrame, features: List[str], 
                       z_threshold: float = 5.0) -> pd.DataFrame:
    """Winsorize feature outliers at z-score threshold."""
    df = df.copy()
    
    for feat in features:
        if feat not in df.columns:
            continue
        
        values = df[feat]
        mean = values.mean()
        std = values.std()
        
        if std < 1e-10:
            continue
        
        # Cap at ±z_threshold standard deviations
        lower = mean - z_threshold * std
        upper = mean + z_threshold * std
        
        df[feat] = values.clip(lower, upper)
    
    return df

