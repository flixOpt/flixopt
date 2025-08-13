# Contributing to FlixOpt

Thanks for your interest in contributing to FlixOpt! 🚀

## Quick Start

1. **Fork & Clone**
   ```bash
   git clone https://github.com/yourusername/flixopt.git
   cd flixopt
   ```

2. **Install for Development**
   ```bash
   pip install -e ".[full]"
   ```

3. **Make Changes & Submit PR**
   ```bash
   git checkout -b feature/your-change
   # Make your changes
   git commit -m "Add: description of changes"
   git push origin feature/your-change
   # Create Pull Request on GitHub
   ```

## How to Contribute

### 🐛 **Found a Bug?**
Use our [bug report template](https://github.com/flixOpt/flixopt/issues/new?template=bug_report.yml) with:
- Minimal code example
- FlixOpt version, Python version, solver used
- Expected vs actual behavior

### ✨ **Have a Feature Idea?**
Use our [feature request template](https://github.com/flixOpt/flixopt/issues/new?template=feature_request.yml) with:
- Clear energy system use case
- Specific examples of what you want to model

### ❓ **Need Help?**
- Check the [documentation](https://flixopt.github.io/flixopt/latest/) first
- Search [existing issues](https://github.com/flixOpt/flixopt/issues)
- Start a [discussion](https://github.com/flixOpt/flixopt/discussions)

## Code Guidelines

- **Style**: Follow PEP 8, use descriptive names
- **Documentation**: Add docstrings with units (kW, kWh, etc.) if applicable
- **Energy Focus**: Use energy domain terminology consistently
- **Testing**: Test with different solvers when applicable

### Example
```python
def create_storage(
    label: str,
    capacity_kwh: float,
    charging_power_kw: float
) -> Storage:
    """
    Create a battery storage component.

    Args:
        label: Unique identifier
        capacity_kwh: Storage capacity [kWh]
        charging_power_kw: Maximum charging power [kW]
    """
```

## What We Welcome

- 🔧 New energy components (batteries, heat pumps, etc.)
- 📚 Documentation improvements
- 🐛 Bug fixes
- 🧪 Test cases
- 💡 Energy system examples

## Questions?

- 📖 [Documentation](https://flixopt.github.io/flixopt/latest/)
- 💬 [Discussions](https://github.com/flixOpt/flixopt/discussions)
- 📧 Contact maintainers (see README)

---

**Every contribution helps advance sustainable energy solutions! 🌱⚡**
