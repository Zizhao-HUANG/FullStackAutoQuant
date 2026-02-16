# Contributing to FullStackAutoQuant

Thank you for your interest in contributing! Here's how to get started.

## Development Setup

```bash
# Clone and install
git clone https://github.com/Zizhao-HUANG/FullStackAutoQuant.git
cd FullStackAutoQuant
pip install -e ".[dev]"
```

## Code Quality

Before submitting a pull request, ensure:

```bash
# Lint
make lint

# Format
make format

# Tests
make test
```

## Guidelines

1. **Code Style**: Follow PEP 8. We use `ruff` for linting and `black` for formatting.
2. **Type Hints**: Add type annotations to all public functions.
3. **Tests**: Include tests for new functionality.
4. **Documentation**: Update relevant docs for user-facing changes.
5. **Commits**: Use clear, descriptive commit messages.

## Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Make your changes with tests
4. Run `make lint test` to verify
5. Submit a pull request with a clear description

## Reporting Issues

- Use the [issue tracker](https://github.com/Zizhao-HUANG/FullStackAutoQuant/issues)
- Include steps to reproduce, expected vs actual behavior, and environment details

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.
