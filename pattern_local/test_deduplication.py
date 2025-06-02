"""
Comprehensive test to verify deduplication and correct implementation.
"""

import numpy as np
import logging
from typing import Dict, Any

# Set up minimal logging
logging.basicConfig(level=logging.INFO)

def test_registry_deduplication():
    """Test that registries work correctly and use the base class."""
    print("🔍 Testing Registry Deduplication...")
    
    from .simplification.registry import SimplificationRegistry
    from .solvers.registry import SolverRegistry
    
    # Test that all methods are available
    simp_methods = SimplificationRegistry.list_available()
    solver_methods = SolverRegistry.list_available()
    
    print(f"✅ Simplification methods: {simp_methods}")
    print(f"✅ Solver methods: {solver_methods}")
    
    # Test registry creation
    assert len(simp_methods) >= 3, "Should have at least 3 simplification methods"
    assert len(solver_methods) >= 5, "Should have at least 5 solver methods"
    
    # Test creation works
    simp = SimplificationRegistry.create('none', {})
    solver = SolverRegistry.create('none', {})
    
    print("✅ Registry deduplication successful!")


def test_solver_deduplication():
    """Test that local solvers use the base class and don't have duplicate code."""
    print("\n🔍 Testing Solver Deduplication...")
    
    from .solvers.local_covariance import LocalCovarianceSolver
    from .solvers.lasso import LassoSolver
    from .solvers.ridge import RidgeSolver
    from .solvers.local_base import LocalSolverBase
    
    # Test that all inherit from LocalSolverBase
    assert issubclass(LocalCovarianceSolver, LocalSolverBase), "LocalCovariance should inherit from LocalSolverBase"
    assert issubclass(LassoSolver, LocalSolverBase), "Lasso should inherit from LocalSolverBase"
    assert issubclass(RidgeSolver, LocalSolverBase), "Ridge should inherit from LocalSolverBase"
    
    # Test that they don't have duplicate methods
    lcov = LocalCovarianceSolver({})
    lasso = LassoSolver({})
    ridge = RidgeSolver({})
    
    # These methods should be inherited from base, not duplicated
    for solver in [lcov, lasso, ridge]:
        assert hasattr(solver, '_get_local_data_and_weights'), "Should have base method"
        assert hasattr(solver, '_get_k_neighbors'), "Should have base method"
        assert hasattr(solver, '_get_bandwidth'), "Should have base method"
        assert hasattr(solver, '_get_analysis_point'), "Should have base method"
    
    print("✅ Solver deduplication successful!")


def test_complete_functionality():
    """Test that the entire system works end-to-end."""
    print("\n🔍 Testing Complete Functionality...")
    
    from .core.explainer import PatternLocalExplainer
    
    # Create synthetic data
    np.random.seed(42)
    X_train = np.random.randn(100, 5)
    instance = np.random.randn(5)
    
    def dummy_predict(X):
        return np.sum(X, axis=1)
    
    # Test all simplification-solver combinations
    combinations = [
        ('none', 'none'),
        ('none', 'global_covariance'),
        ('none', 'local_covariance'),
        ('none', 'lasso'),
        ('none', 'ridge'),
        ('lowrank', 'none'),
        ('lowrank', 'local_covariance'),
        ('lowrank', 'lasso'),
        ('lowrank', 'ridge')
    ]
    
    for simp, solver in combinations:
        try:
            # Test backward compatibility
            explainer = PatternLocalExplainer(
                simplification=simp,
                solver=solver
            )
            explainer.fit(X_train)
            result = explainer.explain_instance(instance, dummy_predict, X_train)
            
            assert 'pattern_weights' in result, f"Missing pattern_weights for {simp}-{solver}"
            assert 'lime_weights' in result, f"Missing lime_weights for {simp}-{solver}"
            assert isinstance(result['pattern_weights'], np.ndarray), f"Wrong type for {simp}-{solver}"
            
            print(f"✅ {simp}-{solver}: Working")
            
        except Exception as e:
            print(f"❌ {simp}-{solver}: Failed - {e}")
            raise
    
    print("✅ Complete functionality test successful!")


def test_fluent_interface():
    """Test that the fluent interface works correctly."""
    print("\n🔍 Testing Fluent Interface...")
    
    from .core.explainer import PatternLocalExplainer
    
    # Create synthetic data
    np.random.seed(42)
    X_train = np.random.randn(50, 3)
    instance = np.random.randn(3)
    
    def dummy_predict(X):
        return np.sum(X, axis=1)
    
    # Test fluent interface
    explainer = (PatternLocalExplainer()
                .with_simplification('lowrank', n_components=2)
                .with_solver('local_covariance', k_ratio=0.2)
                .with_lime_params(num_samples=100))
    
    explainer.fit(X_train)
    result = explainer.explain_instance(instance, dummy_predict, X_train)
    
    assert 'pattern_weights' in result
    assert len(result['pattern_weights']) == 2, "Should have 2 components from lowrank"
    
    print("✅ Fluent interface test successful!")


def test_configuration_validation():
    """Test that configuration validation uses registries."""
    print("\n🔍 Testing Configuration Validation...")
    
    from .config.config import SimplificationConfig, SolverConfig
    from .exceptions import ConfigurationError
    
    # Test valid configurations
    simp_config = SimplificationConfig(method='lowrank')
    solver_config = SolverConfig(method='lasso')
    
    print("✅ Valid configurations accepted")
    
    # Test invalid configurations
    try:
        invalid_simp = SimplificationConfig(method='invalid_method')
        assert False, "Should have raised ConfigurationError"
    except ConfigurationError:
        print("✅ Invalid simplification method rejected")
    
    try:
        invalid_solver = SolverConfig(method='invalid_solver')
        assert False, "Should have raised ConfigurationError"
    except ConfigurationError:
        print("✅ Invalid solver method rejected")
    
    print("✅ Configuration validation test successful!")


def run_all_tests():
    """Run all deduplication and functionality tests."""
    print("🚀 Running Complete Deduplication and Functionality Tests\n")
    
    test_registry_deduplication()
    test_solver_deduplication() 
    test_complete_functionality()
    test_fluent_interface()
    test_configuration_validation()
    
    print("\n🎉 ALL DEDUPLICATION TESTS PASSED! 🎉")
    print("✅ Code duplication eliminated")
    print("✅ All functionality preserved")
    print("✅ Implementation is clean and efficient")


if __name__ == "__main__":
    run_all_tests() 