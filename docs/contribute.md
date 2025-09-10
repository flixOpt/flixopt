# Contributing to the Project

We warmly welcome contributions from the community! This guide will help you get started with contributing to our project.

## Development Setup
1. Clone the repository `git clone https://github.com/flixOpt/flixopt.git`
2. Install the development dependencies `pip install -editable .[dev, docs]`
3. Run `pytest` and `ruff check .` to ensure your code passes all tests

## Documentation
FlixOpt uses [mkdocs](https://www.mkdocs.org/) to generate documentation. To preview the documentation locally, run `mkdocs serve` in the root directory.

## Tutorials & Examples
We greatly appreciate any contributions regarding examples and tutorials.
We consider examples as python scripts, and tutorials as interactive marimo notebooks.
Regarding marimo Notebooks, they are deployed on HuggingFace, and need to follow a certain structure:
```
tutorials/
├── tutorial-"nr"-"name"/
│   ├── app.py              # Marimo notebook
│   ├── requirements.txt    # From template (usually)
│   ├── README.md           # Change according to content
│   ├── Dockerfile          # From template
│   └── .gitattributes      # From template
```
Further, the tutorial should be linked in our docs and added to the tutorial deployment workflow.


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
- Ensure test coverage for new code

## Releases
As stated, we follow **Semantic Versioning**.
Right after one of the 3 [release branches](#branches) is merged into main, a **Tag** should be added to the merge commit and pushed to the main branch. The tag has the form `v1.2.3`.
With this tag,  a release with **Release Notes** must be created. 

*This is our current best practice*
