"""RenTec-Grade Strategy Implementation Module."""

# IMPORTANT: Load ML strategies FIRST from strategies.py file
# Then optionally load old strategies if available
import sys
import importlib.util
from pathlib import Path

parent_dir = Path(__file__).parent.parent
strategies_file = parent_dir / "strategies.py"

# Load ML strategies from strategies.py file
# CRITICAL: When using importlib, the module needs access to the same Python environment
# We'll use importlib but ensure sys.path includes the current environment
if strategies_file.exists():
    try:
        # Ensure parent directory is in sys.path for relative imports
        if str(parent_dir) not in sys.path:
            sys.path.insert(0, str(parent_dir))
        
        # Use importlib to load the module
        module_name = "ml_strategies_module"
        
        # Remove from sys.modules if already loaded to allow reload
        if module_name in sys.modules:
            del sys.modules[module_name]
        
        # Create module spec
        spec = importlib.util.spec_from_file_location(module_name, strategies_file)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not create module spec for {strategies_file}")
        
        ml_strategies = importlib.util.module_from_spec(spec)
        
        # CRITICAL: Register module in sys.modules BEFORE execution
        # This allows the module's imports to resolve correctly
        sys.modules[module_name] = ml_strategies
        
        # Execute the module - this runs all the code in strategies.py
        # The module should have access to all installed packages (pandas, numpy, etc.)
        # because it's using the same Python interpreter
        spec.loader.exec_module(ml_strategies)
        
        # Verify that required classes exist
        required_attrs = ['BaseStrategy', 'MLTrendStrategy', 'MLMeanReversionStrategy', 
                          'MLBreakoutStrategy', 'MLRegimeSwitchStrategy', 
                          'MLConfidenceVolFilterStrategy', 'get_all_strategies', 
                          'STRATEGY_REGISTRY']
        missing = [attr for attr in required_attrs if not hasattr(ml_strategies, attr)]
        if missing:
            raise ImportError(
                f"Missing required attributes in {strategies_file}: {missing}\n"
                f"Available attributes: {[a for a in dir(ml_strategies) if not a.startswith('_')]}"
            )
        
        # Note: create_strategy is optional (will be created if missing, or should exist in strategies.py)
        
        # Check for create_strategy - create it if it doesn't exist (backwards compatibility)
        if not hasattr(ml_strategies, 'create_strategy'):
            # Create factory function if it doesn't exist
            def create_strategy_func(strategy_name: str, timeframe: str, params=None):
                if strategy_name not in ml_strategies.STRATEGY_REGISTRY:
                    available = list(ml_strategies.STRATEGY_REGISTRY.keys())
                    raise ValueError(f"Unknown strategy: '{strategy_name}'. Available: {available}")
                strategy_class = ml_strategies.STRATEGY_REGISTRY[strategy_name]
                return strategy_class(timeframe, params)
            ml_strategies.create_strategy = create_strategy_func
        
        # Re-export ML strategies (PRIMARY - these are what we need)
        BaseStrategy = ml_strategies.BaseStrategy
        MLTrendStrategy = ml_strategies.MLTrendStrategy
        MLMeanReversionStrategy = ml_strategies.MLMeanReversionStrategy
        MLBreakoutStrategy = ml_strategies.MLBreakoutStrategy
        MLRegimeSwitchStrategy = ml_strategies.MLRegimeSwitchStrategy
        MLConfidenceVolFilterStrategy = ml_strategies.MLConfidenceVolFilterStrategy
        create_strategy = ml_strategies.create_strategy  # Should exist now (either from file or created above)
        get_all_strategies = ml_strategies.get_all_strategies
        STRATEGY_REGISTRY = ml_strategies.STRATEGY_REGISTRY
        
        # Final verification
        if MLTrendStrategy is None:
            raise ImportError(f"MLTrendStrategy is None after import from {strategies_file}")
        if get_all_strategies is None:
            raise ImportError(f"get_all_strategies is None after import from {strategies_file}")
            
    except ImportError:
        # Re-raise ImportError as-is (these are expected and should propagate)
        raise
    except Exception as e:
        # For other exceptions, provide detailed error message
        import traceback
        error_details = traceback.format_exc()
        error_msg = (
            f"❌ Failed to load ML strategies from {strategies_file}\n"
            f"Error type: {type(e).__name__}\n"
            f"Error message: {e}\n"
            f"Full traceback:\n{error_details}\n"
            f"Troubleshooting steps:\n"
            f"  1. Verify pandas and numpy are installed: pip install pandas numpy\n"
            f"  2. Check Python environment: python --version\n"
            f"  3. Verify {strategies_file} exists and is valid Python code\n"
            f"  4. Try importing strategies.py directly: python -c 'import sys; sys.path.insert(0, \"ML_model\"); import strategies'"
        )
        raise ImportError(error_msg) from e
else:
    # Fallback if strategies.py doesn't exist
    raise ImportError(
        f"❌ strategies.py file not found at {strategies_file}\n"
        f"Expected location: {strategies_file}\n"
        f"Current directory: {Path.cwd()}\n"
        f"Parent directory: {parent_dir}\n"
        f"Please ensure strategies.py exists in the ML_model directory."
    )

# Optionally import old strategies from directory modules (if available)
# These are secondary/legacy strategies - failures are non-fatal
try:
    from .base import BaseStrategy as OldBaseStrategy, StrategyConfig
    from .s1_momentum_breakout import S1_MomentumBreakout
    from .s2_meanrevert_vwap import S2_MeanRevertVWAP
    from .s3_pullback_trend import S3_PullbackTrend
    from .s4_breakout_retest import S4_BreakoutRetest
    from .s5_momentum_adx import S5_MomentumADX
    from .s6_mtf_alignment import S6_MultiTFAlignment
    _OLD_STRATEGIES_AVAILABLE = True
except ImportError:
    # Old strategies not available (e.g., missing dependencies) - this is OK
    OldBaseStrategy = None
    StrategyConfig = None
    S1_MomentumBreakout = None
    S2_MeanRevertVWAP = None
    S3_PullbackTrend = None
    S4_BreakoutRetest = None
    S5_MomentumADX = None
    S6_MultiTFAlignment = None
    _OLD_STRATEGIES_AVAILABLE = False

__all__ = [
    # Old strategies
    'OldBaseStrategy',
    'StrategyConfig',
    'S1_MomentumBreakout',
    'S2_MeanRevertVWAP',
    'S3_PullbackTrend',
    'S4_BreakoutRetest',
    'S5_MomentumADX',
    'S6_MultiTFAlignment',
    # ML strategies (from strategies.py)
    'BaseStrategy',
    'MLTrendStrategy',
    'MLMeanReversionStrategy',
    'MLBreakoutStrategy',
    'MLRegimeSwitchStrategy',
    'MLConfidenceVolFilterStrategy',
    'create_strategy',
    'get_all_strategies',
    'STRATEGY_REGISTRY',
]

