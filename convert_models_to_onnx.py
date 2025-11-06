#!/usr/bin/env python3
"""
Convert all LightGBM models to ONNX format for web deployment
"""

import sys
import pickle
from pathlib import Path
import numpy as np
import onnxmltools
from onnxmltools.convert.common.data_types import FloatTensorType
import json

# Import BalancedModel from training script to enable unpickling
sys.path.insert(0, str(Path(__file__).parent))
from production_final_system import BalancedModel

# Try to import onnxruntime for verification (optional on Python 3.14)
try:
    import onnxruntime as rt
    ONNX_RUNTIME_AVAILABLE = True
except ImportError:
    ONNX_RUNTIME_AVAILABLE = False
    print("⚠️  onnxruntime not available - skipping model verification")

def convert_model_to_onnx(pkl_path: Path, onnx_path: Path):
    """Convert a single model to ONNX."""
    
    print(f"Converting {pkl_path.name}...")
    
    # Load model
    with open(pkl_path, 'rb') as f:
        saved = pickle.load(f)
        model = saved['model']
        features = saved['features']
        params = saved['params']
        results = saved['results']
    
    # Get the actual LightGBM model
    lgbm_model = model.model
    
    # Define input shape
    initial_type = [('input', FloatTensorType([None, len(features)]))]
    
    # Convert to ONNX
    onnx_model = onnxmltools.convert_lightgbm(
        lgbm_model,
        initial_types=initial_type,
        target_opset=12
    )
    
    # Save ONNX model
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    onnxmltools.utils.save_model(onnx_model, onnx_path)
    
    # Save metadata
    metadata = {
        'symbol': pkl_path.parent.name,
        'timeframe': pkl_path.stem.split('_')[1],
        'features': features,
        'params': params,
        'backtest_results': results,
        'num_features': len(features),
        'input_shape': [None, len(features)],
    }
    
    metadata_path = onnx_path.with_suffix('.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Verify ONNX model works (if onnxruntime is available)
    if ONNX_RUNTIME_AVAILABLE:
        sess = rt.InferenceSession(str(onnx_path))
        input_name = sess.get_inputs()[0].name
        
        # Test with dummy data
        dummy_input = np.random.randn(1, len(features)).astype(np.float32)
        pred = sess.run(None, {input_name: dummy_input})
        
        print(f"  ✅ Converted and verified: {onnx_path.name}")
    else:
        print(f"  ✅ Converted: {onnx_path.name}")
    
    print(f"     Features: {len(features)}")
    print(f"     Backtest WR: {results['win_rate']:.1f}%")
    print(f"     Backtest PF: {results['profit_factor']:.2f}")
    
    return metadata


def convert_all_models():
    """Convert all production-ready models to ONNX."""
    
    models_dir = Path('models_production')
    onnx_dir = Path('models_onnx')
    
    print("\n" + "="*80)
    print("CONVERTING ALL MODELS TO ONNX")
    print("="*80 + "\n")
    
    converted = []
    failed = []
    
    for symbol_dir in sorted(models_dir.iterdir()):
        if not symbol_dir.is_dir():
            continue
        
        for pkl_file in sorted(symbol_dir.glob('*_PRODUCTION_READY.pkl')):
            # Create ONNX path
            symbol = symbol_dir.name
            timeframe = pkl_file.stem.split('_')[1]
            onnx_path = onnx_dir / symbol / f"{symbol}_{timeframe}.onnx"
            
            try:
                metadata = convert_model_to_onnx(pkl_file, onnx_path)
                converted.append((symbol, timeframe, metadata))
            except Exception as e:
                print(f"  ❌ Failed to convert {pkl_file.name}: {e}")
                failed.append((symbol, timeframe, str(e)))
                continue
    
    print("\n" + "="*80)
    print(f"CONVERSION COMPLETE: {len(converted)} models converted")
    if failed:
        print(f"FAILED: {len(failed)} models")
    print("="*80 + "\n")
    
    # Print summary
    print("✅ SUCCESSFULLY CONVERTED:")
    for symbol, tf, meta in converted:
        print(f"   {symbol} {tf:3s} - {meta['num_features']} features, "
              f"WR: {meta['backtest_results']['win_rate']:.1f}%, "
              f"PF: {meta['backtest_results']['profit_factor']:.2f}")
    
    if failed:
        print("\n❌ FAILED TO CONVERT:")
        for symbol, tf, error in failed:
            print(f"   {symbol} {tf:3s} - {error}")
    
    return converted


if __name__ == '__main__':
    convert_all_models()

