"""
Convert trained XGBoost models to ONNX format for deployment in Supabase Edge Functions.

This script:
1. Loads pickled XGBoost models from ml_capvia/models/
2. Converts them to ONNX format
3. Saves ONNX models to ml_capvia/models/onnx/
4. Uploads ONNX models to Supabase Storage

Requirements:
    pip install onnxmltools skl2onnx onnx xgboost supabase python-dotenv
    pip install onnxruntime  # Optional: for model verification (requires Python <= 3.12)

Usage:
    python convert_model_to_onnx.py --symbol BTCUSD --timeframe 15m
    python convert_model_to_onnx.py --all  # Convert all models in directory
"""

import os
import pickle
import argparse
from pathlib import Path
from typing import Dict, Optional

import onnx
import onnxmltools
from onnxmltools.convert.common.data_types import FloatTensorType
try:
    from onnxruntime import InferenceSession
    ONNXRUNTIME_AVAILABLE = True
except ImportError:
    print("⚠️  onnxruntime not available - model verification will be skipped")
    InferenceSession = None
    ONNXRUNTIME_AVAILABLE = False
import numpy as np
from supabase import create_client, Client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

MODELS_DIR = Path("ml_capvia/models")
ONNX_DIR = MODELS_DIR / "onnx"
ONNX_DIR.mkdir(exist_ok=True, parents=True)

# Supabase credentials
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")


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
    try:
        # Handle different model data structures
        if isinstance(model_data, dict):
            model = model_data.get('model', model_data)
            features = model_data.get('features', [])
        else:
            # If model_data is the model itself
            model = model_data
            features = []
        
        # Debug: print model type
        print(f"  Model type: {type(model).__name__}")
        print(f"  Model class module: {type(model).__module__}")
        
        # Check if model has attributes that might contain the actual models
        if hasattr(model, '__dict__'):
            print(f"  Model attributes: {list(model.__dict__.keys())}")
        
        # Handle EnsembleClassifier - convert individual models
        if type(model).__name__ == 'EnsembleClassifier':
            print(f"\n  🔄 Custom EnsembleClassifier detected!")
            print(f"  Method: {model.method if hasattr(model, 'method') else 'unknown'}")
            
            if hasattr(model, 'models') and isinstance(model.models, dict):
                print(f"  Converting {len(model.models)} individual models to ONNX...")
                
                from skl2onnx import convert_sklearn, update_registered_converter
                from skl2onnx.common.data_types import FloatTensorType as SklearnFloatTensorType
                from onnxmltools.convert.xgboost.operator_converters.XGBoost import convert_xgboost as xgb_onnx_converter
                from onnxmltools.convert.xgboost.shape_calculators.Classifier import calculate_xgboost_classifier_output_shapes
                
                # Register XGBoost converters for sklearn-onnx
                try:
                    from xgboost import XGBClassifier, XGBRegressor
                    update_registered_converter(
                        XGBClassifier, 'XGBoostXGBClassifier',
                        calculate_xgboost_classifier_output_shapes,
                        xgb_onnx_converter,
                        options={'nocl': [True, False], 'zipmap': [True, False, 'columns']}
                    )
                except:
                    pass  # Already registered or not available
                
                num_features = len(features) if features else 50
                converted_models = {}
                
                for model_name, sub_model in model.models.items():
                    model_type = type(sub_model).__name__
                    print(f"\n  Converting {model_name} ({model_type})...")
                    
                    try:
                        if 'XGB' in model_type:
                            # XGBoost model - convert via temp file (workaround for version compatibility)
                            import tempfile
                            import xgboost as xgb
                            
                            initial_type = [('float_input', FloatTensorType([None, num_features]))]
                            
                            # Save and reload booster as workaround
                            with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
                                temp_path = tmp.name
                            
                            try:
                                sub_model.save_model(temp_path)
                                booster = xgb.Booster()
                                booster.load_model(temp_path)
                                sub_onnx = onnxmltools.convert_xgboost(
                                    booster,
                                    initial_types=initial_type,
                                    target_opset=12
                                )
                            finally:
                                import os
                                if os.path.exists(temp_path):
                                    os.unlink(temp_path)
                        elif 'LGBM' in model_type:
                            # LightGBM model - use onnxmltools
                            initial_type = [('float_input', FloatTensorType([None, num_features]))]
                            sub_onnx = onnxmltools.convert_lightgbm(
                                sub_model,
                                initial_types=initial_type,
                                target_opset=12
                            )
                        else:
                            # Sklearn models (Logistic Regression, etc.)
                            initial_type = [('float_input', SklearnFloatTensorType([None, num_features]))]
                            sub_onnx = convert_sklearn(
                                sub_model,
                                initial_types=initial_type,
                                target_opset=12
                            )
                        
                        # Save individual model
                        model_output_path = output_path.parent / f"{output_path.stem}_{model_name}.onnx"
                        onnx.save_model(sub_onnx, str(model_output_path))
                        print(f"  ✓ Saved: {model_output_path.name}")
                        converted_models[model_name] = model_output_path
                        
                    except Exception as e:
                        error_msg = str(e)
                        print(f"  ✗ Failed to convert {model_name}: {error_msg}")
                        
                        # Provide helpful guidance for known issues
                        if 'XGB' in model_type and 'could not convert string to float' in error_msg:
                            print(f"      → Known issue: XGBoost 3.x has compatibility issues with onnxmltools")
                            print(f"      → Workaround: Consider downgrading to XGBoost 2.x or use a different deployment method")
                
                if converted_models:
                    print(f"\n✓ Successfully converted {len(converted_models)}/{len(model.models)} models")
                    # Return success (we've saved the individual models)
                    return True
                else:
                    raise ValueError("Failed to convert any models from the ensemble")
            else:
                raise ValueError(
                    "Custom EnsembleClassifier structure not recognized. "
                    "Expected 'models' attribute as dict."
                )
        else:
            num_features = len(features) if features else 50  # Default fallback
            
            # Define input type - must match your feature count
            initial_type = [('float_input', FloatTensorType([None, num_features]))]
            
            # Convert to ONNX
            onnx_model = onnxmltools.convert_xgboost(
                model,
                initial_types=initial_type,
                target_opset=12
            )
        
        # Save ONNX model
        onnx.save_model(onnx_model, str(output_path))
        print(f"✓ Converted to ONNX: {output_path.name}")
        
        # Verify the model works (if onnxruntime is available)
        if ONNXRUNTIME_AVAILABLE:
            session = InferenceSession(str(output_path))
            dummy_input = np.random.randn(1, num_features).astype(np.float32)
            input_name = session.get_inputs()[0].name
            output = session.run(None, {input_name: dummy_input})
            print(f"✓ ONNX model verified (output shape: {output[0].shape})")
        else:
            print(f"⚠️  Skipping verification (onnxruntime not installed)")
        
        return True
        
    except Exception as e:
        print(f"✗ Conversion failed: {e}")
        return False


def upload_to_supabase(onnx_path: Path, model_data: Dict) -> bool:
    """Upload ONNX model to Supabase Storage and register in ml_model_registry."""
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        print("⚠️  Supabase credentials not found. Skipping upload.")
        return False
    
    try:
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
        
        # Extract model metadata from filename
        # Expected format: SYMBOL_TIMEFRAME_xgboost_vYYYYMMDD.pkl
        filename = onnx_path.stem  # Remove .onnx extension
        parts = filename.replace('.pkl', '').split('_')
        
        if len(parts) < 4:
            print(f"⚠️  Invalid filename format: {filename}")
            return False
        
        symbol = parts[0]
        timeframe = parts[1]
        model_version = parts[3]  # vYYYYMMDD
        
        # Storage path
        storage_path = f"ml_models/{symbol}/{timeframe}/{onnx_path.name}"
        
        # Upload to storage bucket
        with open(onnx_path, 'rb') as f:
            response = supabase.storage.from_('Trade').upload(
                storage_path,
                f,
                file_options={"content-type": "application/octet-stream"}
            )
        
        print(f"✓ Uploaded to Supabase Storage: {storage_path}")
        
        # Register in ml_model_registry table
        oos_metrics = model_data.get('oos_metrics', {})
        features = model_data.get('features', [])
        
        # Deactivate previous models for this symbol/timeframe
        supabase.table('ml_model_registry').update({
            'is_active': False
        }).eq('symbol', symbol).eq('timeframe', timeframe).execute()
        
        # Insert new model record
        supabase.table('ml_model_registry').insert({
            'symbol': symbol,
            'timeframe': timeframe,
            'model_version': model_version,
            'model_path': storage_path,
            'model_type': 'xgboost',
            'model_format': 'onnx',
            'is_active': True,
            'oos_metrics': oos_metrics,
            'features': features
        }).execute()
        
        print(f"✓ Registered in ml_model_registry as active model")
        return True
        
    except Exception as e:
        print(f"✗ Supabase upload failed: {e}")
        return False


def process_model(pkl_path: Path, upload: bool = True, output_dir: Optional[Path] = None) -> bool:
    """Process a single model: load, convert, optionally upload."""
    print(f"\n{'='*60}")
    print(f"Processing: {pkl_path.name}")
    print('='*60)
    
    # Load pickled model
    model_data = load_model(pkl_path)
    if not model_data:
        return False
    
    # Determine output directory
    if output_dir is None:
        output_dir = ONNX_DIR
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Convert to ONNX
    onnx_path = output_dir / pkl_path.with_suffix('.onnx').name
    if not convert_to_onnx(model_data, onnx_path):
        return False
    
    # Upload to Supabase
    if upload:
        upload_to_supabase(onnx_path, model_data)
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Convert XGBoost models to ONNX')
    parser.add_argument('--pkl-path', type=str, help='Direct path to .pkl model file')
    parser.add_argument('--symbol', type=str, help='Symbol to convert (e.g., BTCUSD)')
    parser.add_argument('--timeframe', type=str, help='Timeframe to convert (e.g., 15m)')
    parser.add_argument('--all', action='store_true', help='Convert all models')
    parser.add_argument('--no-upload', action='store_true', help='Skip Supabase upload')
    
    args = parser.parse_args()
    
    # Find models to convert
    output_dir = None
    if args.pkl_path:
        pkl_path = Path(args.pkl_path)
        if not pkl_path.exists():
            print(f"Error: Model file not found: {pkl_path}")
            return
        model_files = [pkl_path]
        # Save ONNX file in the same directory as the input file
        output_dir = pkl_path.parent
    elif args.all:
        model_files = list(MODELS_DIR.glob("*.pkl"))
        if not model_files:
            print("No .pkl model files found in ml_capvia/models/")
            return
        print(f"Found {len(model_files)} model(s) to convert")
    elif args.symbol and args.timeframe:
        # Find model matching symbol and timeframe
        pattern = f"{args.symbol}_{args.timeframe}_*.pkl"
        model_files = list(MODELS_DIR.glob(pattern))
        if not model_files:
            print(f"No model found matching: {pattern}")
            return
        # Use the most recent version
        model_files = [max(model_files, key=lambda p: p.stat().st_mtime)]
    else:
        parser.print_help()
        return
    
    # Process each model
    success_count = 0
    for pkl_path in model_files:
        if process_model(pkl_path, upload=not args.no_upload, output_dir=output_dir):
            success_count += 1
    
    print(f"\n{'='*60}")
    print(f"Conversion complete: {success_count}/{len(model_files)} successful")
    print('='*60)


if __name__ == "__main__":
    main()
