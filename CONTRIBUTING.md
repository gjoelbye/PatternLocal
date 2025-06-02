# Contributing to PatternLocal

We welcome contributions to PatternLocal! This document provides guidelines for contributing to the project.

## Development Setup

### 1. Clone the Repository
```bash
git clone https://github.com/your-repo/PatternXAI.git
cd PatternXAI
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Development Dependencies
```bash
pip install -e .
# Or use requirements-dev.txt
pip install -r requirements-dev.txt
```

### 4. Install Pre-commit Hooks (Optional but Recommended)
```bash
pre-commit install
```

## Development Workflow

### Code Style
We use several tools to maintain code quality:

- **Black** for code formatting
- **isort** for import sorting
- **flake8** for linting

Run all checks:
```bash
black patternlocal examples
isort patternlocal examples
flake8 patternlocal examples
mypy patternlocal
```

### Testing
Run tests with pytest:
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov= patternlocal --cov-report=html

# Run specific test file
pytest patternlocal/tests/test_unified_api.py
```

### Adding New Features

1. **Create a branch** for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Write tests first** (TDD approach):
   - Add tests in ` patternlocal/tests/`
   - Follow existing test patterns
   - Use the fixtures in `conftest.py`

3. **Implement your feature**:
   - Follow the existing code style
   - Add type hints
   - Write comprehensive docstrings

4. **Update documentation**:
   - Update README.md if needed
   - Add examples if applicable
   - Update CHANGELOG.md

5. **Run all checks**:
   ```bash
   pytest
   black --check patternlocal examples
   isort --check-only patternlocal examples
   flake8 patternlocal examples
   ```

## Types of Contributions

### 🐛 Bug Reports
- Use the issue template
- Include minimal reproducible example
- Specify environment details

### ✨ Feature Requests
- Describe the use case clearly
- Explain why the feature would be valuable
- Consider if it fits the project scope

### 📝 Documentation
- Fix typos and improve clarity
- Add examples and tutorials
- Improve docstrings

### 🔧 Code Contributions
- New simplification methods
- New pattern solvers
- Performance improvements
- Bug fixes

## Adding New Components

### New Simplification Method
1. Create a class inheriting from `BaseSimplification`
2. Implement all abstract methods
3. Add comprehensive tests
4. Update the main explainer to recognize it
5. Add documentation and examples

### New Pattern Solver
1. Create a class inheriting from `BaseSolver`
2. Implement the `solve` method
3. Add comprehensive tests
4. Update the main explainer to recognize it
5. Add documentation and examples

## Code Review Process

1. All submissions require code review
2. We use GitHub Pull Requests
3. Tests must pass
4. Code style checks must pass
5. At least one maintainer approval required

## Pull Request Guidelines

1. **Clear title and description**
2. **Link related issues**
3. **Include tests for new functionality**
4. **Update documentation as needed**
5. **Keep PRs focused and small when possible**

### PR Checklist
- [ ] Tests added/updated and passing
- [ ] Code style checks passing
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Type hints added
- [ ] Examples updated if needed

## License

By contributing, you agree that your contributions will be licensed under the MIT License.