# Contributing to the Project

We warmly welcome contributions from the community! This guide will help you get started with contributing to our project.

## Development Setup
1. Clone the repository `git clone https://github.com/flixOpt/flixopt.git`
2. Install the development dependencies `pip install -e ".[dev]"`
3. Install pre-commit hooks `pre-commit install` (one-time setup)
4. Run `pytest` to ensure your code passes all tests

## Code Quality
We use [Ruff](https://github.com/astral-sh/ruff) for linting and formatting. After the one-time setup above, **code quality checks run automatically on every commit**.

To run manually:
- `ruff check --fix .` to check and fix linting issues
- `ruff format .` to format code or
- `pre-commit run` or `pre-commit run --all-files` to trigger all checks

## Documentation (Optional)
FlixOpt uses [mkdocs](https://www.mkdocs.org/) to generate documentation.
To work on documentation:
```bash
pip install -e ".[docs]"
mkdocs serve
```
Then navigate to http://127.0.0.1:8000/

## Testing
- `pytest` to run the test suite
- You can also run the provided python script `run_all_test.py`

---
# Best practices

## Coding Guidelines

- Follow PEP 8 style guidelines
- Write clear, commented code
- Include type hints
- Create or update tests for new functionality
- Ensure 100% test coverage for new code

## Branches & Releases
New features should be branched from `main` into `feature/*`
As stated, we follow **Semantic Versioning**. Releases are created manually from the `main` branch.
