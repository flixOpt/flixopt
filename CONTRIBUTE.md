# Contributing to FlixOpt

We warmly welcome contributions from the community! Whether you're fixing bugs, adding features, improving documentation, or sharing examples, your contributions are valuable.

## Ways to Contribute

### 🐛 Report Issues
Found a bug or have a feature request? Please [open an issue](https://github.com/flixOpt/flixopt/issues) on GitHub.

When reporting issues, please include:
- A clear description of the problem
- Steps to reproduce the issue
- Expected vs. actual behavior
- Your environment (OS, Python version, FlixOpt version)
- Minimal code example if applicable

### 💡 Share Examples
Help others learn FlixOpt by contributing examples:
- Real-world use cases
- Tutorial notebooks
- Integration examples with other tools
- Add them to the `examples/` directory

### 📖 Improve Documentation
Documentation improvements are always welcome:
- Fix typos or clarify existing docs
- Add missing documentation
- Translate documentation
- Improve code comments

### 🔧 Submit Code Contributions
Ready to contribute code? Great! See the sections below for setup and guidelines.

---

## Development Setup

### Getting Started
1. Fork and clone the repository:
   ```bash
   git clone https://github.com/flixOpt/flixopt.git
   cd flixopt
   ```

2. Install development dependencies:
   ```bash
   pip install -e ".[full, dev]"
   ```

3. Set up pre-commit hooks (one-time setup):
   ```bash
   pre-commit install
   ```

4. Verify your setup:
   ```bash
   pytest
   ```

### Working with Documentation
FlixOpt uses [mkdocs](https://www.mkdocs.org/) to generate documentation.

To work on documentation:
```bash
pip install -e ".[docs]"
mkdocs serve
```
Then navigate to http://127.0.0.1:8000/

---

## Code Quality Standards

### Automated Checks
We use [Ruff](https://github.com/astral-sh/ruff) for linting and formatting. After the one-time setup above, **code quality checks run automatically on every commit**.

### Manual Checks
To run checks manually:
- `ruff check --fix .` - Check and fix linting issues
- `ruff format .` - Format code
- `pre-commit run --all-files` - Run all pre-commit checks

### Testing
All tests are located in the `tests/` directory with a flat structure:
- `test_component.py` - Component tests
- `test_flow.py` - Flow tests
- `test_storage.py` - Storage tests
- etc.

#### Running Tests
- `pytest` - Run the full test suite (excluding examples by default)
- `pytest tests/test_component.py` - Run a specific test file
- `pytest tests/test_component.py::TestClassName` - Run a specific test class
- `pytest tests/test_component.py::TestClassName::test_method` - Run a specific test
- `pytest -m slow` - Run only slow tests
- `pytest -m examples` - Run example tests (normally skipped)
- `pytest -k "keyword"` - Run tests matching a keyword

#### Common Test Patterns
The `tests/conftest.py` file provides shared fixtures:
- `solver_fixture` - Parameterized solver fixture (HiGHS, Gurobi)
- `highs_solver` - HiGHS solver instance
- Coordinate configuration fixtures for timesteps, periods, scenarios

Use these fixtures by adding them as function parameters:
```python
def test_my_feature(solver_fixture):
    # solver_fixture is automatically provided by pytest
    model = fx.FlowSystem(...)
    model.solve(solver_fixture)
```

#### Testing Guidelines
- Write tests for all new functionality
- Ensure all tests pass before submitting a PR
- Aim for 100% test coverage for new code
- Use descriptive test names that explain what's being tested
- Add the `@pytest.mark.slow` decorator for tests that take >5 seconds

### Coding Guidelines
- Follow [PEP 8](https://pep8.org/) style guidelines
- Write clear, self-documenting code with helpful comments
- Include type hints for function signatures
- Create or update tests for new functionality
- Aim for 100% test coverage for new code

---

## Workflow

### Branches & Pull Requests
1. Create a feature branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and commit them with clear messages

3. Push your branch and open a Pull Request

4. Ensure all CI checks pass

### Branch Naming
- Features: `feature/feature-name`
- Bug fixes: `fix/bug-description`
- Documentation: `docs/what-changed`

### Commit Messages
- Use clear, descriptive commit messages
- Start with a verb (Add, Fix, Update, Remove, etc.)
- Keep the first line under 72 characters

---

## Releases

We follow **Semantic Versioning** (MAJOR.MINOR.PATCH). Releases are created manually from the `main` branch by maintainers.

---

## Questions?

If you have questions or need help, feel free to:
- Open a discussion on GitHub
- Ask in an issue
- Reach out to the maintainers

Thank you for contributing to FlixOpt!
