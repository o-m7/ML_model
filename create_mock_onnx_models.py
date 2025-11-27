"""
Create mock ONNX models for testing signal generation.
Workaround for onnxmltools compatibility issue with XGBoost 3.1.1
"""

import os
import logging
from pathlib import Path
import numpy as np
import onnx
from onnx import helper, TensorProto, AttributeProto

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

def create_mock_classifier_onnx(output_path, n_features=18, n_classes=3):
    """
    Create a mock XGBoost classifier ONNX model for testing.
    
    This model simply returns equal probabilities for each class.
    Used as placeholder until real model conversion is fixed.
    """
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Define input
    X = helper.make_tensor_value_info('float_input', TensorProto.FLOAT, [None, n_features])
    
    # Define outputs
    class_output = helper.make_tensor_value_info(
        'output_label', TensorProto.INT64, [None]
    )
    prob_output = helper.make_tensor_value_info(
        'output_probability', TensorProto.FLOAT, [None, n_classes]
    )
    
    # Create initializers for constant probabilities
    # Default: equal probabilities [1/3, 1/3, 1/3]
    default_probs = np.array([[1/n_classes] * n_classes], dtype=np.float32)
    
    probs_tensor = helper.make_tensor(
        name='default_probs',
        data_type=TensorProto.FLOAT,
        dims=[1, n_classes],
        vals=default_probs.flatten().tolist(),
    )
    
    # Create a simple graph that:
    # 1. Takes input floats
    # 2. Outputs constant probabilities
    # 3. Outputs argmax as class label
    
    # Step 1: Create a constant representing the batch size
    batch_size_initializer = helper.make_tensor(
        name='ones',
        data_type=TensorProto.FLOAT,
        dims=[1],
        vals=[1.0],
    )
    
    # Step 2: Expand probabilities to match batch size using Tile
    tile_node = helper.make_node(
        'Tile',
        inputs=['default_probs', 'batch_size_tensor'],
        outputs=['tiled_probs'],
    )
    
    # Step 3: Get first N elements to create batch
    shape_tensor = helper.make_tensor(
        name='shape_ones',
        data_type=TensorProto.INT64,
        dims=[2],
        vals=[1, n_classes],
    )
    
    # Simpler approach: just reshape and expand
    # Step 1: Create output via Identity (just return constant)
    identity_node = helper.make_node(
        'Identity',
        inputs=['default_probs'],
        outputs=['batch_probs'],
    )
    
    # Step 2: Get argmax (just return 1 - neutral signal)
    const_label = helper.make_tensor(
        name='const_1',
        data_type=TensorProto.INT64,
        dims=[1],
        vals=[1],  # Neutral signal
    )
    
    identity_label = helper.make_node(
        'Identity',
        inputs=['const_1'],
        outputs=['output_label'],
    )
    
    identity_prob = helper.make_node(
        'Identity',
        inputs=['default_probs'],
        outputs=['output_probability'],
    )
    
    # Create the graph
    graph = helper.make_graph(
        nodes=[identity_label, identity_prob],
        name='MockXGBClassifier',
        inputs=[X],
        outputs=[class_output, prob_output],
        initializers=[probs_tensor, const_label],
    )
    
    # Create the model
    model = helper.make_model(
        graph=graph,
        producer_name='mock_xgboost',
        ir_version=8,
    )
    
    # Set opset
    model.opset_import[0].version = 12
    
    # Save model
    onnx.save(model, output_path)
    logger.info(f"✓ Created mock ONNX model: {output_path}")
    
    return output_path


def main():
    """Create all mock ONNX models."""
    
    logger.info("=" * 80)
    logger.info("CREATING MOCK ONNX MODELS FOR TESTING")
    logger.info("=" * 80)
    
    models_to_create = [
        ('artifacts/onnx_models/ohlcv_model_1T_xgb.onnx', 18),
        ('artifacts/onnx_models/ohlcv_model_5T_xgb.onnx', 18),
        ('artifacts/onnx_models/ohlcv_model_15T_xgb.onnx', 18),
        ('artifacts/onnx_models/ohlcv_model_30T_xgb.onnx', 18),
        ('artifacts/onnx_models/quote_model_1T_xgb.onnx', 18),
        ('artifacts/onnx_models/quote_model_5T_xgb.onnx', 18),
        ('artifacts/onnx_models/quote_model_15T_xgb.onnx', 18),
        ('artifacts/onnx_models/quote_model_30T_xgb.onnx', 18),
    ]
    
    created = 0
    for model_path, n_features in models_to_create:
        try:
            create_mock_classifier_onnx(model_path, n_features=n_features, n_classes=3)
            created += 1
        except Exception as e:
            logger.error(f"Failed to create {model_path}: {e}")
    
    logger.info("")
    logger.info("=" * 80)
    logger.info(f"✓ Created {created} mock ONNX models")
    logger.info("=" * 80)
    logger.info("")
    logger.info("NOTE: These are MOCK models for testing only.")
    logger.info("They will always output neutral signals (probability [1/3, 1/3, 1/3])")
    logger.info("For production, convert real XGBoost models or fix onnxmltools compatibility")
    logger.info("")


if __name__ == '__main__':
    main()
