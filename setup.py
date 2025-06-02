"""
Setup script for PatternLocal package.
"""

from setuptools import find_packages, setup

# Read README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="patternlocal",
    version="1.0.0",
    author="Anders Gjølbye",
    author_email="agjma@dtu.com",
    description="Unified pattern-based explanations for machine learning models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/gjoelbye/PatternLocal",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.0",
        "scikit-learn>=1.0.0",
        "scipy>=1.7.0",
        "lime>=0.2.0",
    ],
    extras_require={
        "superpixel": ["scikit-image>=0.18.0"],
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.900",
        ],
        "examples": [
            "matplotlib>=3.3.0",
            "jupyter>=1.0.0",
            "seaborn>=0.11.0",
        ],
    },
    keywords="explainable-ai, machine-learning, pattern-analysis, lime, interpretability",
    project_urls={
        "Source": "https://github.com/gjoelbye/PatternLocal",
    },
    entry_points={
        "console_scripts": [
            "pattern-local-demo=pattern_local.examples.demo:main",
        ],
    },
)
