# Contributing to FullStackAutoQuant

Thank you for your interest in contributing! This document explains how to get involved effectively.

## Where to Contribute

We welcome contributions in the following areas. Please check the [Issues](https://github.com/Zizhao-HUANG/FullStackAutoQuant/issues) page for tasks labeled `good first issue` or `help wanted`.

### Contribution Areas (by priority)

| Area | Examples | Difficulty |
|------|----------|------------|
| **Documentation** | Architecture walkthroughs, tutorials, bilingual (EN/CN) docs, API reference | Easy to Medium |
| **Tests** | Increase test coverage, edge case testing, integration tests | Easy to Medium |
| **Examples and Notebooks** | Usage walkthroughs, visualization notebooks, strategy analysis demos | Medium |
| **CI/CD and Tooling** | GitHub Actions improvements, Docker Compose setup, dev tooling | Medium |
| **Benchmarking** | Performance benchmarks, model comparison scripts, profiling | Medium to Hard |
| **New Modules** | Signal monitoring, regime detection, analytics (as standalone modules) | Hard |

### Guidelines for Core Code Changes

The core trading logic (`model/`, `trading/`, `backtest/`) is the backbone of a production system. Changes to these areas require:

1. **An approved RFC**: open an issue describing your proposed change before writing code
2. **Comprehensive tests**: all new behavior must be covered
3. **No behavioral regressions**: existing tests must continue to pass
4. **Owner review**: all PRs require approval from [@Zizhao-HUANG](https://github.com/Zizhao-HUANG)

## Development Setup

```bash
# Clone and install
git clone https://github.com/Zizhao-HUANG/FullStackAutoQuant.git
cd FullStackAutoQuant
pip install -e ".[dev]"
```

## Code Quality

Before submitting a pull request:

```bash
# Lint
make lint

# Format
make format

# Tests
make test
```

## Code Style

1. **PEP 8**: We use `ruff` for linting and `black` for formatting (line length: 100).
2. **Type Hints**: Add type annotations to all public functions.
3. **Docstrings**: Include docstrings for modules, classes, and public functions.
4. **Commits**: Use clear, descriptive commit messages. Prefer small, focused commits.

## Pull Request Process

1. **Check Issues first**: look for an existing issue or create one to discuss your idea
2. Fork the repository
3. Create a feature branch (`git checkout -b feature/your_feature`)
4. Make your changes with tests
5. Run `make lint test` to verify
6. Submit a pull request using the PR template
7. Address review feedback promptly

> **Note**: All PRs require review and approval from the repository owner. Please be patient during the review process.

## Reporting Issues

- Use the [issue tracker](https://github.com/Zizhao-HUANG/FullStackAutoQuant/issues)
- Include steps to reproduce, expected vs actual behavior, and environment details
- For feature requests, describe the use case and proposed approach

## License

By contributing, you agree that your contributions will be licensed under the project's [CC BY-NC-SA 4.0](LICENSE) license.
