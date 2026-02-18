# PatternLocal Hyperparameter Optimization

This module provides automated hyperparameter optimization capabilities for PatternLocal explainers. It allows you to automatically find the best parameters for your specific use case using various optimization algorithms and evaluation metrics.

## Features

- **Multiple Optimization Backends**: Support for Hyperopt (TPE), Grid Search, and Random Search
- **Comprehensive Search Spaces**: Pre-defined parameter spaces for all simplification methods and solvers
- **Flexible Metrics**: Multiple evaluation metrics including fidelity, stability, and ground truth agreement
- **Caching**: Automatic caching of optimization results to avoid re-computation
- **Parallel Evaluation**: Parallel evaluation of parameter combinations for faster optimization
- **Visualization**: Optimization history plotting and analysis tools

## Quick Start

```python
from patternlocal import OptimizedPatternLocalExplainer

# Create optimized explainer
explainer = OptimizedPatternLocalExplainer(
    cache_dir="optimization_cache",
    n_jobs=-1,  # Use all available cores
    random_state=42
)

# Run optimization
best_params = explainer.optimize_parameters(
    X_val=X_validation,           # Validation instances
    masks_val=ground_truth_masks, # Ground truth importance (optional)
    X_train=X_training,           # Training data
    y_train=y_training,           # Training labels
    model=trained_model,          # Your trained model
    backend="hyperopt",           # Optimization algorithm
    max_evals=100                 # Maximum evaluations
)

# Use optimized explainer
explainer.set_best_params(best_params)
explainer.fit(X_training)
explanation = explainer.explain_instance(instance, model.predict_proba, X_training)
```

## Optimization Backends

### Hyperopt (Recommended)
Uses Tree-structured Parzen Estimator (TPE) for efficient hyperparameter search:

```python
best_params = explainer.optimize_parameters(
    # ... data arguments ...
    backend="hyperopt",
    max_evals=100
)
```

**Requirements**: `pip install hyperopt`

### Grid Search
Exhaustive search over a parameter grid:

```python
best_params = explainer.optimize_parameters(
    # ... data arguments ...
    backend="grid",
    max_evals=50  # Limits grid size if too large
)
```

### Random Search
Random sampling from parameter space:

```python
best_params = explainer.optimize_parameters(
    # ... data arguments ...
    backend="random",
    max_evals=100
)
```

## Evaluation Metrics

### Fidelity Metric (Default)
Measures how well the explanation approximates the model's local behavior:

```python
from patternlocal.optimization.metrics import create_fidelity_metric

metric_fn = create_fidelity_metric(n_samples=1000)
```

### Stability Metric
Measures consistency of explanations across multiple runs:

```python
from patternlocal.optimization.metrics import create_stability_metric

metric_fn = create_stability_metric(n_runs=10)
```

### Ground Truth Agreement
Compares explanations with known ground truth importance:

```python
from patternlocal.optimization.metrics import create_ground_truth_metric

metric_fn = create_ground_truth_metric(method="jaccard")
```

### Combined Metrics
Combine multiple metrics with custom weights:

```python
from patternlocal.optimization.metrics import create_combined_metric

combined_metric = create_combined_metric(
    metrics=[fidelity_metric, stability_metric],
    weights=[0.7, 0.3]  # 70% fidelity, 30% stability
)
```

## Custom Search Spaces

### Default Search Space
```python
from patternlocal.optimization.search_spaces import get_default_search_space

search_space = get_default_search_space(
    simplification="lowrank",
    solver="local_covariance",
    include_lime=True,
    backend="hyperopt"
)
```

### Custom Search Space
```python
from patternlocal.optimization.search_spaces import create_custom_search_space

search_space = create_custom_search_space(
    simplification_methods=["lowrank", "none"],
    solver_methods=["local_covariance", "global_covariance", "lasso"],
    include_lime=True,
    backend="hyperopt"
)
```

## Advanced Usage

### Custom Optimization Configuration
```python
explainer = OptimizedPatternLocalExplainer(
    cache_dir="custom_cache",
    n_jobs=4,
    random_state=42
)

# Custom metric with specific parameters
metric_fn = create_fidelity_metric(n_samples=2000, sigma=0.2)

# Custom search space
search_space = {
    "simplification__n_components": [0.8, 0.9, 0.95, 0.99],
    "solver__k_ratio": [0.05, 0.1, 0.15, 0.2],
    "solver__shrinkage_intensity": [0.0, 0.1, 0.2],
    "lime__num_samples": [500, 1000, 2000]
}

best_params = explainer.optimize_parameters(
    X_val=X_val,
    masks_val=masks_val,
    X_train=X_train,
    y_train=y_train,
    model=model,
    metric_fn=metric_fn,
    search_space=search_space,
    backend="hyperopt",
    max_evals=150,
    use_cache=True,
    save_results=True
)
```

### Optimization Analysis
```python
# Get optimization summary
summary = explainer.get_optimization_summary()
print(f"Best loss: {summary['best_loss']}")
print(f"Execution time: {summary['optimization_stats']['execution_time_seconds']}s")

# Get detailed history
history = explainer.get_optimization_history()
for i, trial in enumerate(history[:5]):
    print(f"Trial {i}: loss={trial['loss']:.6f}, params={trial['params']}")

# Plot optimization progress
explainer.plot_optimization_history(save_path="optimization_plot.png")
```

## Parameter Spaces

### Simplification Methods

**LowRank (PCA)**:
- `n_components`: Variance to retain (0.8, 0.9, 0.95, 0.99)
- `whiten`: Whether to whiten components (True, False)
- `svd_solver`: SVD solver ("auto", "full", "arpack", "randomized")

**Superpixel**:
- `n_segments`: Number of superpixels (50, 100, 200, 300)
- `compactness`: Compactness parameter (0.1 - 1.0)
- `sigma`: Gaussian kernel sigma (0.5 - 2.0)

### Solver Methods

**Local Covariance**:
- `k_ratio`: Ratio of samples for local estimation (0.05 - 0.3)
- `bandwidth`: Kernel bandwidth (None, "auto")
- `kernel`: Kernel function ("gaussian", "exponential", "linear")
- `shrinkage_intensity`: Shrinkage regularization (0.0 - 0.5)
- `distance_metric`: Distance metric ("euclidean", "manhattan", "cosine")

**Global Covariance**:
- `shrinkage_intensity`: Shrinkage regularization (0.0 - 0.5)

**Lasso**:
- `alpha`: Regularization strength (1e-4 - 1.0)
- `max_iter`: Maximum iterations (1000, 2000, 5000)
- `selection`: Coordinate selection ("cyclic", "random")

**Ridge**:
- `alpha`: Regularization strength (1e-4 - 10.0)
- `solver`: Solver method ("auto", "svd", "cholesky", "lsqr")

### LIME Parameters
- `num_samples`: Number of samples (100, 500, 1000, 2000)
- `num_features`: Number of features ("auto", 5, 10, 20)
- `feature_selection`: Selection method ("forward_selection", "lasso_path", "none", "auto")

## Caching and Performance

The optimization system automatically caches results based on experiment configuration:

```python
# Results are cached by default
best_params = explainer.optimize_parameters(
    # ... arguments ...
    use_cache=True,      # Load cached results if available
    save_results=True    # Save results for future use
)

# Check cache information
from patternlocal.optimization.utils import get_cache_info
cache_info = get_cache_info("optimization_cache")
print(f"Cached results: {cache_info['num_cached_results']}")

# Clean up old cache files
from patternlocal.optimization.utils import cleanup_old_cache_files
deleted = cleanup_old_cache_files("optimization_cache", max_age_days=30)
```

## Installation Requirements

### Basic Functionality
```bash
pip install scikit-learn numpy joblib
```

### Hyperopt Backend
```bash
pip install hyperopt
```

### Visualization
```bash
pip install matplotlib
```

## Examples

See `examples/hyperparameter_optimization_example.py` for comprehensive examples including:
- Basic optimization with default settings
- Advanced optimization with custom metrics
- Multiple backend comparisons
- Optimization analysis and visualization

## Tips for Effective Optimization

1. **Start Small**: Begin with a small `max_evals` to test your setup
2. **Use Validation Data**: Ensure your validation set is representative
3. **Cache Results**: Use caching to avoid re-computation
4. **Monitor Progress**: Check optimization history and plots
5. **Parallel Processing**: Use `n_jobs=-1` for faster optimization
6. **Ground Truth**: Provide ground truth masks when available for better evaluation
7. **Combined Metrics**: Use multiple metrics for more robust optimization

## Troubleshooting

### Common Issues

**Memory Errors**: Reduce `batch_size` or `subset_size` in optimization
**Slow Optimization**: Reduce `max_evals` or use fewer parallel jobs
**Import Errors**: Install required dependencies (hyperopt, matplotlib)
**Cache Issues**: Clear cache directory or disable caching

### Performance Optimization

- Use `n_jobs=-1` for parallel evaluation
- Limit validation set size with `subset_size`
- Use grid search for small parameter spaces
- Use hyperopt for larger parameter spaces
- Cache results to avoid re-computation 