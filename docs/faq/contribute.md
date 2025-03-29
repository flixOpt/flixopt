# Contributing to the Project

We warmly welcome contributions from the community! This guide will help you get started with contributing to our project.

## Development Setup
1. Clone the repository `git clone https://github.com/flixOpt/flixopt.git`
2. Install the development dependencies `pip install -editable .[dev, docs]`
3. Run `pytest` and `ruff check .` to ensure your code passes all tests

## Documentation
FlixOpt uses [mkdocs](https://www.mkdocs.org/) to generate documentation. To preview the documentation locally, run `mkdocs serve` in the root directory.

## Helpful Commands
- `mkdocs serve` to preview the documentation locally. Navigate to `http://127.0.0.1:8000/` to view the documentation.
- `pytest` to run the test suite (You can also run the provided python script `run_all_test.py`)
- `ruff check .` to run the linter
- `ruff check . --fix` to automatically fix linting issues

---
# Best practices

## Coding Guidelines

- Follow PEP 8 style guidelines
- Write clear, commented code
- Include type hints
- Create or update tests for new functionality
- Ensure 100% test coverage for new code

## Branches
As we start to think FlixOpt in **Releases**, we decided to introduce multiple **dev**-branches instead of only one:
Following the **Semantic Versioning** guidelines, we introduced:
- `next/patch`: This is where all pull requests for the next patch release (1.0.x) go.  
- `next/minor`: This is where all pull requests for the next minor release (1.x.0) go.  
- `next/major`: This is where all pull requests for the next major release (x.0.0) go.

Everything else remains in `feature/...`-branches.

## Pull requests
Every feature or bugfix should be merged into one of the 3 [release branches](#branches), using **Squash and merge** or a regular **single commit**.
At some point, `next/minor` or `next/major` will get merged into `main` using a regular **Merge**  (not squash).
*This ensures that Features are kept separate, and the `next/...`branches stay in synch with ``main`.*

## Releases
As stated, we follow **Semantic Versioning**.
Right after one of the 3 [release branches](#branches) is merged into main, a **Tag** should be added to the merge commit and pushed to the main branch. The tag has the form `v1.2.3`.
With this tag,  a release with **Release Notes** must be created. 

*This is our current best practice*
