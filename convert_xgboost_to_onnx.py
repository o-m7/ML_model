#!/usr/bin/env python3
"""
Simple XGBoost to ONNX converter - no Supabase dependencies.

Usage:
    python convert_xgboost_to_onnx.py --input models_rentec/XAUUSD/XAUUSD_15T.pkl --output models_onnx/XAUUSD/XAUUSD_15T.onnx
"""

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import onnx
import onnxmltools
from onnxmltools.convert.common.data_types import FloatTensorType

try:
    from onnxruntime import InferenceSession
    ONNXRUNTIME_AVAILABLE = True
except ImportError:
    print("⚠️  onnxruntime not available - skipping verification")
    ONNXRUNTIME_AVAILABLE = False


def load_model(model_path: Path) -> Optional[Dict]:
    """Load pickled XGBoost model."""
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        print(f"✓ Loaded model: {model_path.name}")
        return model_data
    except Exception as e:
        print(f"✗ Failed to load {model_path.name}: {e}")
        return None


def convert_to_onnx(model_data: Dict, output_path: Path) -> bool:
    """Convert XGBoost model to ONNX format."""
    import tempfile
    import xgboost as xgb

    try:
        # Extract model and features
        if isinstance(model_data, dict):
            model = model_data.get('model', model_data)
            features = model_data.get('features', [])
            class_names = model_data.get('class_names', ['Down', 'Up'])
        else:
            model = model_data
            features = []
            class_names = ['Down', 'Up']

        print(f"  Model type: {type(model).__name__}")
        print(f"  Features: {len(features)}")
        print(f"  Classes: {class_names}")

        num_features = len(features) if features else 18  # Default to 18 from shared_features.py

        # Workaround for XGBoost 3.x compatibility issue with onnxmltools
        # Save and reload booster to fix config format issues
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
            temp_path = tmp.name

        try:
            # Save model to temporary file
            model.save_model(temp_path)
            # Load as raw booster
            booster = xgb.Booster()
            booster.load_model(temp_path)

            print(f"  ✓ Loaded booster via temporary file (XGBoost 3.x workaround)")

            # Define input type
            initial_type = [('float_input', FloatTensorType([None, num_features]))]

            # Convert booster to ONNX
            onnx_model = onnxmltools.convert_xgboost(
                booster,
                initial_types=initial_type,
                target_opset=12
            )
        finally:
            # Clean up temp file
            import os
            if os.path.exists(temp_path):
                os.unlink(temp_path)

        # Create output directory
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save ONNX model
        onnx.save_model(onnx_model, str(output_path))
        print(f"✓ Converted to ONNX: {output_path}")

        # Save metadata JSON
        metadata = {
            'symbol': output_path.parent.name,
            'timeframe': output_path.stem.split('_')[1],
            'features': features,
            'class_names': class_names,
            'num_features': num_features,
            'input_shape': [None, num_features],
        }

        # Add backtest results if available
        if 'results' in model_data:
            metadata['backtest_results'] = model_data['results']

        metadata_path = output_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"✓ Saved metadata: {metadata_path}")

        # Verify ONNX model
        if ONNXRUNTIME_AVAILABLE:
            session = InferenceSession(str(output_path))
            dummy_input = np.random.randn(1, num_features).astype(np.float32)
            input_name = session.get_inputs()[0].name
            output = session.run(None, {input_name: dummy_input})
            print(f"✓ ONNX model verified (output shape: {output[0].shape})")

        return True

    except Exception as e:
        print(f"✗ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description='Convert XGBoost model to ONNX')
    parser.add_argument('--input', type=str, required=True, help='Path to input .pkl file')
    parser.add_argument('--output', type=str, required=True, help='Path to output .onnx file')

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"❌ Input file not found: {input_path}")
        return 1

    print("\n" + "="*60)
    print(f"Converting: {input_path.name}")
    print("="*60 + "\n")

    # Load model
    model_data = load_model(input_path)
    if not model_data:
        return 1

    # Convert to ONNX
    if convert_to_onnx(model_data, output_path):
        print("\n" + "="*60)
        print("✅ CONVERSION SUCCESSFUL")
        print("="*60 + "\n")
        return 0
    else:
        print("\n" + "="*60)
        print("❌ CONVERSION FAILED")
        print("="*60 + "\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
