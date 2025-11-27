"""
JSON serialization utilities for numpy/pandas types.
Place in: ML_model/ML_model/json_utils.py
"""

import numpy as np
import pandas as pd
import json
from typing import Any


class NumpyEncoder(json.JSONEncoder):
    """Handle numpy types for JSON serialization."""
    
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif pd.isna(obj):
            return None
        return super().default(obj)


def convert_for_json(obj: Any) -> Any:
    """
    Recursively convert numpy/pandas types to native Python types.
    
    Args:
        obj: Any object that might contain numpy types
        
    Returns:
        Object with all numpy types converted to native Python
    """
    if isinstance(obj, dict):
        return {k: convert_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_for_json(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_for_json(item) for item in obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif pd.isna(obj):
        return None
    else:
        return obj


def safe_json_dump(obj: Any, file_path: str, indent: int = 2) -> bool:
    """
    Safely dump object to JSON with type conversion.
    
    Args:
        obj: Object to serialize
        file_path: Output path
        indent: JSON indentation
        
    Returns:
        True if successful, False otherwise
    """
    try:
        converted = convert_for_json(obj)
        with open(file_path, 'w') as f:
            json.dump(converted, f, indent=indent, cls=NumpyEncoder)
        return True
    except Exception as e:
        print(f"❌ JSON serialization failed: {e}")
        return False