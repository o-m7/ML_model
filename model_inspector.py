"""
Model Feature Inspector - See what features your models need
"""

import pickle
from pathlib import Path

MODELS_DIR = Path("production_models")

print("=" * 80)
print("MODEL FEATURE INSPECTOR")
print("=" * 80)

for model_file in MODELS_DIR.glob("*.pkl"):
    print(f"\n📦 {model_file.name}")
    print("-" * 80)
    
    with open(model_file, 'rb') as f:
        model_data = pickle.load(f)
    
    print(f"Model: {model_data['model_name']}")
    print(f"Threshold: {model_data['threshold']:.2f}")
    print(f"\nTotal Features: {len(model_data['feature_cols'])}")
    
    feature_cols = model_data['feature_cols']
    
    # Categorize features
    htf_features = [f for f in feature_cols if any(tf in f for tf in ['1H_', '4H_', '15T_', '30T_'])]
    base_features = [f for f in feature_cols if f not in htf_features]
    
    print(f"Base features: {len(base_features)}")
    print(f"HTF features: {len(htf_features)}")
    
    if htf_features:
        print(f"\n⚠️  HTF FEATURES REQUIRED:")
        # Group by timeframe
        by_tf = {}
        for feat in htf_features:
            tf = feat.split('_')[0]
            if tf not in by_tf:
                by_tf[tf] = []
            by_tf[tf].append(feat)
        
        for tf, feats in sorted(by_tf.items()):
            print(f"   {tf}: {feats}")
    
    print(f"\nFIRST 10 FEATURES:")
    for i, feat in enumerate(feature_cols[:10], 1):
        print(f"   {i}. {feat}")
    
    if len(feature_cols) > 10:
        print(f"   ... and {len(feature_cols) - 10} more")

print("\n" + "=" * 80)
print("RECOMMENDATION:")
print("=" * 80)
print("If your models have HTF features (1H_, 4H_, etc.), you need to:")
print("1. Retrain models WITHOUT multi-timeframe features, OR")
print("2. Fix the data fetching to include all required timeframes")
print("\nThe signal generator's 8-layer engine handles HTF bias internally,")
print("so you don't need HTF features in your model features.")
print("=" * 80)