# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

SSRM (Sequential Sample Ratio Mismatch) is a Python library for sequential testing of Sample Ratio Mismatch in A/B testing experiments. It uses Bayesian methods to detect when traffic allocation ratios deviate from expected distributions.

## Development Commands

### Environment Setup
```bash
make install-dev    # Install all dependencies including dev tools
make install        # Install only runtime dependencies
```

The project uses [uv](https://docs.astral.sh/uv/) for dependency management. After `make install-dev`, pre-commit hooks will be automatically installed.

### Testing
```bash
make test           # Run unit tests with coverage
make check          # Run all checks (lint, typecheck, test)
make lint           # Run ruff linter only
make typecheck      # Run pyright type checker
```

To run a single test file:
```bash
uv run pytest ssrm_test/test_ssrm_test.py
```

To run a specific test function:
```bash
uv run pytest ssrm_test/test_ssrm_test.py::test_accumulator
```

### Code Formatting and Linting
```bash
make fmt            # Format and fix linting issues with ruff
make fmt-notebooks  # Format Jupyter notebooks with black_nbconvert
```

Pre-commit hooks automatically run ruff on commit. All code follows ruff's default style (88-character line length).

### Documentation
```bash
make docs           # Build Sphinx documentation (outputs to docs/build/html/)
jupyter lab         # Start Jupyter to work with tutorial notebooks
```

### Build
```bash
make release        # Build distribution artifacts in dist/
make clean          # Remove generated files, caches, and build artifacts
```

## Architecture

### Core Module: ssrm_test.py

The library provides sequential Bayesian hypothesis testing for SRM detection. Key architectural concepts:

**Data Formats**: The library accepts two data formats:
- Unit-level data: One-hot encoded vectors per observation (e.g., `[[1,0,0], [0,1,0], ...]`)
- Time-aggregated data: Count vectors per time period (e.g., `[[20,17,9], [18,21,8], ...]`)

**Sequential Processing**: Uses a functional accumulation pattern via `toolz.accumulate` to compute statistics incrementally as data arrives. The `accumulator()` function is the core binary operator that:
- Updates posterior distributions (Dirichlet for M1, fixed probabilities for M0)
- Accumulates log marginal likelihoods for both hypotheses
- Enables streaming computation without reprocessing historical data

**Hypothesis Testing Framework**:
- **M0 (null)**: Traffic follows expected `null_probabilities` distribution
- **M1 (alternative)**: Traffic follows some other distribution, modeled with a Dirichlet prior
- Bayes factors are computed as the ratio of marginal likelihoods (M1/M0)
- Posterior probabilities transform Bayes factors into interpretable probabilities

**Prior Configuration**:
- `dirichlet_probability`: Mean of the alternative hypothesis prior (defaults to null_probabilities)
- `dirichlet_concentration`: Controls prior tightness (default 10000 = strong prior)

### Public API Functions

Main entry points in order of typical usage:

1. **`srm_test(data, null_probabilities)`**: Simple single-value test returning final posterior probability
2. **`sequential_posterior_probabilities(...)`**: Returns probability of SRM at each data point
3. **`sequential_bayes_factors(...)`**: Returns Bayes factors sequentially
4. **`sequential_p_values(...)`**: Returns calibrated p-values that control Type I error
5. **`sequential_posteriors(...)`**: Low-level function returning full posterior state at each step

### Testing Philosophy

Tests validate:
- Mathematical correctness: sequential accumulation equals batch computation (test_accumulator)
- Numerical stability: handling of extreme Bayes factors without overflow warnings (test_overflow)
- Monotonicity properties: p-values are non-increasing (test_p_values_decreasing_and_in_range)
- Input validation: data must be integer arrays (test_data_validator)

### Code Style Requirements

- **Format**: Ruff formatter (line length 88)
- **Linting**: Ruff with pycodestyle, pyflakes, isort, pyupgrade, bugbear, and comprehensions rules
- **Type checking**: Pyright in basic mode
- **Docstrings**: Numpydoc format (matching pandas conventions)
- **Type hints**: Modern Python 3.11+ syntax (built-in generics like `list[int]`)

### Dependencies

Core: numpy, scipy, toolz
- `scipy.special`: Log-gamma functions for numerical stability
- `toolz.itertoolz.accumulate`: Functional streaming computation pattern

**Note**: Requires Python 3.11+ and NumPy 2.0+

## Working with Notebooks

Tutorial notebook: `notebooks/introduction.ipynb`

To format notebooks after editing:
```bash
make fmt-notebooks
```

To clear notebook outputs:
```bash
make clean-notebooks
```
