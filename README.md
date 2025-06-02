# PatternLocal

PatternLocal is a comprehensive Python package for generating local explanations using PatternLocal method. It provides a clean, modular interface with different data preprocessing techniques for **both tabular and image data**. See the [main paper](https://arxiv.org/abs/2505.11210) for details.

## Features

### 🎯 **Unified API**
- Single interface for all pattern-based explanation methods
- **Supports both tabular and image data** with the same API
- Easy integration with existing ML pipelines
- Consistent parameter handling across methods

### 🔧 **Modular Architecture**
- **Simplification Methods**: Transform data before explanation
  - `NoSimplification`: Work in original feature space
  - `LowRankSimplification`: PCA-based dimensionality reduction
  - `SuperpixelSimplification`: Image segmentation for computer vision

- **PatternLocal Solvers**: Different approaches to compute patterns
  - `NoSolver`: Return LIME weights (baseline)
  - `GlobalCovarianceSolver`: Global covariance baseline
  - `LocalCovarianceSolver`: Local weighted covariance (main pattern method)
  - `LassoSolver`: Local Lasso regression
  - `RidgeSolver`: Local Ridge regression

- **LIME Modes**: Automatic detection and handling
  - `tabular`: For structured/tabular data
  - `image`: For image data with superpixel segmentation

### 🚀 **Easy to Use**

**Tabular Data:**
```python
from patternlocal import PatternLocalExplainer

# Tabular data explainer
explainer = PatternLocalExplainer(
    simplification='lowrank',
    solver='local_covariance'
)
explainer.fit(X_train)
explanation = explainer.explain_instance(instance, predict_fn, X_train)
```

**Image Data:**
```python
# Image data explainer
explainer = PatternLocalExplainer(
    simplification='superpixel',    # Required for images
    solver='local_covariance',
    lime_params={'mode': 'image'},  # Enable image mode
    simplification_params={'image_shape': (28, 28)}
)
explainer.fit(X_train, image_shape=(28, 28))
explanation = explainer.explain_instance(instance, predict_fn, X_train, labels=[1])
```

## Installation

### From Source
```bash
git clone https://github.com/gjoelbye/PatternLocal.git
cd PatternLocal
pip install -e .
```

### Dependencies
- numpy
- scikit-learn
- lime
- scipy
- matplotlib (for demos)

**For image support:**
- scikit-image (install with: `pip install scikit-image`)

## Quick Start

### Tabular Data

```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from patternlocal import PatternLocalExplainer

# Create sample data
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Initialize explainer (automatically uses tabular mode)
explainer = PatternLocalExplainer(
    simplification='none',           # No preprocessing
    solver='local_covariance',       # Pattern method
    random_state=42
)

# Fit explainer
explainer.fit(X)

# Explain an instance
def predict_fn(X):
    return model.predict_proba(X)[:, 1]

explanation = explainer.explain_instance(
    instance=X[0],
    predict_fn=predict_fn,
    X_train=X
)

# Access results
pattern_weights = explanation['pattern_weights']
lime_weights = explanation['lime_weights']
```

### Image Data

```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from patternlocal import PatternLocalExplainer

# Create sample image data (28x28 images)
n_samples, image_shape = 500, (28, 28)
X = np.random.rand(n_samples, image_shape[0] * image_shape[1])
y = np.random.randint(0, 2, n_samples)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

def predict_fn(X):
    return model.predict_proba(X)[:, 1]

# Initialize explainer for image mode
explainer = PatternLocalExplainer(
    simplification='superpixel',    # Required for image mode
    solver='local_covariance',      # Pattern solver
    lime_params={'mode': 'image'},  # Enable image mode
    simplification_params={
        'image_shape': image_shape,  # Image dimensions
        'method': 'slic',           # Segmentation method
        'n_segments': 50            # Number of superpixels
    },
    random_state=42
)

# Fit explainer
explainer.fit(X, image_shape=image_shape)

# Explain an instance
explanation = explainer.explain_instance(
    instance=X[0],
    predict_fn=predict_fn,
    X_train=X,
    labels=[1]                      # Required for image mode
)

# Visualize results
import matplotlib.pyplot as plt
pattern_img = explanation['pattern_weights'].reshape(image_shape)
plt.imshow(pattern_img, cmap='RdBu')
plt.title('Pattern Explanation')
plt.show()
```

## Configuration

### LIME Parameters

**Tabular Mode (default):**
```python
lime_params = {
    'mode': 'tabular',              # Optional (default)
    'num_samples': 5000,            # Number of LIME samples
    'feature_selection': 'auto',    # Feature selection method
    'discretize_continuous': True,  # Discretize continuous features
    'kernel_width': None           # Auto-estimate kernel width
}
```

**Image Mode:**
```python
lime_params = {
    'mode': 'image',               # Required for image mode
    'num_samples': 1000,           # Number of LIME samples
    'kernel_width': 0.25,          # Kernel bandwidth
    'feature_selection': 'none',   # Usually 'none' for images
    'verbose': False              # Verbose output
}
```

## Simplification Methods

### NoSimplification
Identity transformation - works in original feature space.
```python
explainer = PatternLocalExplainer(simplification='none')
```

### LowRankSimplification  
Uses PCA for dimensionality reduction before pattern computation.
```python
# Specify number of components
explainer = PatternLocalExplainer(
    simplification='lowrank',
    simplification_params={'n_components': 10}
)

# Or specify variance to retain
explainer = PatternLocalExplainer(
    simplification='lowrank', 
    simplification_params={'n_components': 0.95}
)
```

### SuperpixelSimplification (Image Data)
Segments images into superpixels for more interpretable explanations.

**SLIC Segmentation:**
```python
explainer = PatternLocalExplainer(
    simplification='superpixel',
    lime_params={'mode': 'image'},
    simplification_params={
        'image_shape': (28, 28),    # Required
        'method': 'slic',
        'n_segments': 100,          # Number of superpixels
        'compactness': 8,           # Balance color vs spatial proximity
        'sigma': 0                  # Gaussian smoothing
    }
)
```

**Grid Segmentation:**
```python
explainer = PatternLocalExplainer(
    simplification='superpixel',
    lime_params={'mode': 'image'},
    simplification_params={
        'image_shape': (28, 28),
        'method': 'grid',
        'grid_rows': 7,             # Number of grid rows
        'grid_cols': 7              # Number of grid columns
    }
)
```

## PatternLocal Solvers

All solvers work with both tabular and image data:

### LocalCovarianceSolver (Recommended)
The main patternlocal method - estimates local covariance matrices.
```python
solver_params = {
    'k_ratio': 0.1,                  # Use 10% of training data
    'bandwidth': None,               # Auto-estimate
    'shrinkage_intensity': 0.0,      # No regularization
    'distance_metric': 'euclidean',  # Distance metric
    'use_projection': True           # Project onto LIME hyperplane
}
```

### LassoSolver
Uses local Lasso regression.
```python
solver_params = {
    'alpha': 1.0,                    # Lasso regularization
    'k_ratio': 0.1                   # Local neighborhood size
}
```

### RidgeSolver
Uses local Ridge regression.
```python
solver_params = {
    'alpha': 1.0,                    # Ridge regularization
    'k_ratio': 0.1                   # Local neighborhood size
}
```

## Complete Examples

### Tabular Data with Different Solvers

```python
from patternlocal import PatternLocalExplainer
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load data
data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

def predict_fn(X):
    return model.predict_proba(X)[:, 1]

# Compare different solvers
solvers = ['none', 'local_covariance', 'lasso', 'ridge']
instance = X_test[0]

for solver_name in solvers:
    explainer = PatternLocalExplainer(
        simplification='lowrank',
        solver=solver_name,
        simplification_params={'n_components': 0.95},
        solver_params={'k_ratio': 0.1} if solver_name != 'none' else {},
        random_state=42
    )
    
    explainer.fit(X_train)
    explanation = explainer.explain_instance(instance, predict_fn, X_train)
    
    print(f"{solver_name}: weights range [{explanation['pattern_weights'].min():.3f}, {explanation['pattern_weights'].max():.3f}]")
```

### Image Data with Visualization

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from patternlocal import PatternLocalExplainer

# Create synthetic image data with spatial patterns
def create_image_data(n_samples=1000, image_shape=(28, 28)):
    n_features = image_shape[0] * image_shape[1]
    X = np.random.rand(n_samples, n_features)
    
    # Create pattern: top half vs bottom half importance
    y = (X[:, :n_features//2].mean(axis=1) > 
         X[:, n_features//2:].mean(axis=1)).astype(int)
    
    return X, y

# Generate data and train model
X, y = create_image_data()
X_train, X_test = X[:800], X[800:]
y_train, y_test = y[:800], y[800:]

model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_train, y_train)

def predict_fn(X):
    return model.predict_proba(X)[:, 1]

# Setup explainer
explainer = PatternLocalExplainer(
    simplification='superpixel',
    solver='local_covariance',
    lime_params={
        'mode': 'image',
        'num_samples': 1000,
        'kernel_width': 0.25
    },
    simplification_params={
        'image_shape': (28, 28),
        'method': 'slic',
        'n_segments': 50,
        'compactness': 8
    },
    solver_params={
        'k_ratio': 0.1,
        'shrinkage_intensity': 0.1
    },
    random_state=42
)

# Fit and explain
explainer.fit(X_train, image_shape=(28, 28))
explanation = explainer.explain_instance(
    instance=X_test[0],
    predict_fn=predict_fn,
    X_train=X_train,
    labels=[1]
)

# Visualize
def visualize_explanation(instance, pattern_weights, lime_weights, image_shape):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(instance.reshape(image_shape), cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # LIME explanation
    lime_img = lime_weights.reshape(image_shape)
    im1 = axes[1].imshow(lime_img, cmap='RdBu', 
                        vmin=-np.abs(lime_img).max(), 
                        vmax=np.abs(lime_img).max())
    axes[1].set_title('LIME Explanation')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1])
    
    # Pattern explanation
    pattern_img = pattern_weights.reshape(image_shape)
    im2 = axes[2].imshow(pattern_img, cmap='RdBu',
                        vmin=-np.abs(pattern_img).max(),
                        vmax=np.abs(pattern_img).max())
    axes[2].set_title('Pattern Explanation')
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2])
    
    plt.tight_layout()
    plt.show()

visualize_explanation(
    X_test[0], 
    explanation['pattern_weights'], 
    explanation['lime_weights'],
    (28, 28)
)
```

## API Reference

### PatternLocalExplainer

Main class for patternlocal explanations supporting both tabular and image data.

**Parameters:**
- `simplification`: str or BaseSimplification instance
  - `'none'`: No preprocessing (tabular data)
  - `'lowrank'`: PCA preprocessing (tabular data)
  - `'superpixel'`: Superpixel segmentation (image data)
- `solver`: str or BaseSolver instance  
- `lime_params`: dict, optional - Parameters for LIME
  - For tabular: `feature_selection`, `discretize_continuous`, `kernel_width`
  - For image: `mode='image'`, `kernel_width`, `verbose`
- `simplification_params`: dict, optional - Parameters for simplification
- `solver_params`: dict, optional - Parameters for solver
- `random_state`: int, optional - Random seed

**Methods:**
- `fit(X_train, **kwargs)`: Fit explainer to training data
  - For image data: include `image_shape=(height, width)`
- `explain_instance(instance, predict_fn, X_train, **kwargs)`: Generate explanation
  - For image data: include `labels=[class_index]`

**Returns (explain_instance):**
Dictionary with keys:
- `'pattern_weights'`: PatternLocal explanation weights
- `'lime_weights'`: Original LIME weights
- `'lime_intercept'`: LIME intercept
- `'local_exp'`: LIME explanation object

## Mode Detection

PatternLocal automatically detects the appropriate mode based on your configuration:

- **Tabular Mode**: Default when `simplification` is `'none'` or `'lowrank'`
- **Image Mode**: Automatically enabled when:
  - `simplification='superpixel'` AND `lime_params={'mode': 'image'}`
  - Image shape is provided in `simplification_params`

## Examples

See the `examples/` directory for comprehensive usage examples:
- ` patternlocal_demo.py`: Tabular data examples
- `image_mode_demo.py`: Image data examples

Run the demos:
```bash
python examples/ patternlocal_demo.py      # Tabular examples
python examples/image_mode_demo.py         # Image examples
```

## Citation

If you use PatternLocal in your research, please cite:

```bibtex
@misc{gjoelbye2025patternlocal,
      title={Minimizing False-Positive Attributions in Explanations of Non-Linear Models}, 
      author={Anders Gjølbye and Stefan Haufe and Lars Kai Hansen},
      year={2025},
      eprint={2505.11210},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2505.11210}, 
}
```

## Contributing

We welcome contributions! Please see our contributing guidelines for details.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built on top of the excellent [LIME](https://github.com/marcotcr/lime) package
