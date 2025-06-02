# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2024-01-XX

### Added
- Initial release of PatternLocal
- Unified API for both tabular and image data
- Automatic mode detection based on simplification type
- Support for multiple simplification methods:
  - NoSimplification for tabular data
  - LowRankSimplification (PCA) for dimensionality reduction
  - SuperpixelSimplification for image data
- Multiple pattern solvers:
  - NoSolver (LIME baseline)
  - GlobalCovarianceSolver
  - LocalCovarianceSolver (main pattern method)
  - LassoSolver
  - RidgeSolver
- Comprehensive examples and documentation
- Full test suite with pytest
- Type hints support
- CI/CD with GitHub Actions

### Features
- **Unified Interface**: Same API works for both tabular and image data
- **Auto-Detection**: Automatically detects appropriate LIME mode
- **Modular Design**: Easy to extend with new simplification methods and solvers
- **Comprehensive Testing**: Full test coverage with multiple test scenarios
- **Professional Packaging**: Modern Python packaging with pyproject.toml

## [Unreleased]

### Planned
- Additional simplification methods
- Performance optimizations
- Enhanced documentation with Sphinx
- More example notebooks
- Integration with popular ML frameworks 