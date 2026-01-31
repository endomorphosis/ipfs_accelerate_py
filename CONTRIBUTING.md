# Contributing to IPFS Accelerate Python

Thank you for your interest in contributing to IPFS Accelerate Python! This document provides guidelines and instructions for contributing.

## 🎯 Ways to Contribute

We welcome contributions in many forms:

- 🐛 **Bug Reports** - Help us identify and fix issues
- 💡 **Feature Requests** - Suggest new functionality
- 📚 **Documentation** - Improve guides, examples, and API docs
- 🧪 **Tests** - Add test coverage for edge cases
- 🔧 **Code** - Fix bugs or implement features
- 🌍 **Translations** - Help translate documentation
- 💬 **Community Support** - Answer questions in discussions

## 📋 Before You Start

1. **Search existing issues** - Check if someone already reported the bug or requested the feature
2. **Read the docs** - Familiarize yourself with the project structure and conventions
3. **Discuss major changes** - Open an issue to discuss significant changes before starting work

## 🚀 Getting Started

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/ipfs_accelerate_py.git
cd ipfs_accelerate_py

# Add upstream remote
git remote add upstream https://github.com/endomorphosis/ipfs_accelerate_py.git
```

### 2. Set Up Development Environment

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode with all dependencies
pip install -e ".[dev,test]"

# Install pre-commit hooks (recommended)
pre-commit install
```

### 3. Create a Branch

```bash
# Create a feature branch
git checkout -b feature/your-feature-name

# Or for bug fixes
git checkout -b fix/bug-description
```

## 📝 Development Guidelines

### Code Style

We follow these conventions:

- **Python**: PEP 8 style guide
- **Formatting**: Use `black` for code formatting
- **Imports**: Use `isort` for import sorting
- **Type Hints**: Add type hints to new code
- **Docstrings**: Use Google-style docstrings

```bash
# Format your code before committing
black ipfs_accelerate_py/
isort ipfs_accelerate_py/

# Check linting
pylint ipfs_accelerate_py/
flake8 ipfs_accelerate_py/
```

### Testing

All contributions should include tests:

```bash
# Run all tests
pytest

# Run specific test file
pytest test/test_your_feature.py

# Run with coverage
pytest --cov=ipfs_accelerate_py --cov-report=html

# Run specific test
pytest test/test_your_feature.py::test_specific_function
```

**Test Requirements:**
- ✅ All new features must have tests
- ✅ Bug fixes should include regression tests
- ✅ Maintain or improve code coverage
- ✅ Tests should be fast and isolated
- ✅ Use fixtures for common setup

### Documentation

Update documentation when you:

- Add new features
- Change APIs
- Fix bugs that affect usage
- Add configuration options

**Documentation Files:**
- `README.md` - Project overview
- `docs/` - Comprehensive guides
- `docs/API.md` - API reference
- Code docstrings - Inline documentation

```python
def new_function(param: str) -> bool:
    """
    Brief description of what the function does.
    
    Longer description with more details about the function's
    behavior, edge cases, and usage examples.
    
    Args:
        param: Description of the parameter.
        
    Returns:
        Description of return value.
        
    Raises:
        ValueError: When param is invalid.
        
    Example:
        >>> result = new_function("test")
        >>> print(result)
        True
    """
    # Implementation
    return True
```

## 🔄 Pull Request Process

### 1. Make Your Changes

- Write clear, focused commits
- Follow the coding standards
- Add tests for new functionality
- Update documentation

### 2. Commit Messages

Use clear, descriptive commit messages:

```
Short summary (50 chars or less)

More detailed explanation if needed. Wrap at 72 characters.
Explain what changed and why, not how.

- Bullet points are okay
- Use present tense ("Add feature" not "Added feature")
- Reference issues: Fixes #123
```

**Commit Message Format:**
- `feat: Add new feature`
- `fix: Fix bug description`
- `docs: Update documentation`
- `test: Add test coverage`
- `refactor: Improve code structure`
- `style: Format code`
- `chore: Update dependencies`

### 3. Push and Create Pull Request

```bash
# Ensure your branch is up to date
git fetch upstream
git rebase upstream/main

# Push to your fork
git push origin feature/your-feature-name

# Create pull request on GitHub
```

### 4. Pull Request Template

When creating a PR, include:

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Tests pass locally
- [ ] Added new tests
- [ ] Updated documentation

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-reviewed code
- [ ] Commented complex logic
- [ ] Updated documentation
- [ ] No breaking changes (or documented)
```

### 5. Code Review

- Respond to feedback promptly
- Make requested changes
- Keep discussion professional and constructive
- Update PR based on reviews

## 🐛 Reporting Bugs

### Before Reporting

1. **Check existing issues** - Bug might already be reported
2. **Try latest version** - Bug might already be fixed
3. **Search documentation** - Might be expected behavior

### Bug Report Template

```markdown
**Description**
Clear description of the bug

**To Reproduce**
Steps to reproduce:
1. Step one
2. Step two
3. See error

**Expected Behavior**
What you expected to happen

**Actual Behavior**
What actually happened

**Environment**
- OS: [e.g., Ubuntu 22.04]
- Python version: [e.g., 3.10.0]
- Package version: [e.g., 0.0.45]
- Hardware: [e.g., NVIDIA RTX 3090]

**Additional Context**
- Error messages
- Logs
- Screenshots
- Minimal reproducible example
```

## 💡 Suggesting Features

### Feature Request Template

```markdown
**Problem**
What problem does this feature solve?

**Proposed Solution**
How should it work?

**Alternatives**
Other solutions you considered

**Additional Context**
Use cases, examples, mockups
```

## 🧪 Testing Guidelines

### Test Structure

```python
import pytest
from ipfs_accelerate_py import IPFSAccelerator

class TestFeature:
    """Test suite for new feature."""
    
    @pytest.fixture
    def accelerator(self):
        """Create accelerator instance for tests."""
        return IPFSAccelerator()
    
    def test_basic_functionality(self, accelerator):
        """Test basic feature functionality."""
        result = accelerator.new_feature()
        assert result == expected_value
    
    def test_edge_case(self, accelerator):
        """Test edge case handling."""
        with pytest.raises(ValueError):
            accelerator.new_feature(invalid_input)
```

### Test Best Practices

- ✅ **Isolated** - Tests don't depend on each other
- ✅ **Fast** - Tests run quickly
- ✅ **Descriptive** - Test names clearly describe what they test
- ✅ **Focused** - Each test tests one thing
- ✅ **Deterministic** - Tests produce same results every time

## 📚 Documentation Guidelines

### Writing Style

- **Clear and concise** - Get to the point quickly
- **Use examples** - Show, don't just tell
- **Target audience** - Write for your audience's skill level
- **Active voice** - "The function returns" not "is returned"
- **Present tense** - "The function does" not "will do"

### Documentation Structure

1. **Overview** - What is it?
2. **Installation** - How to install
3. **Quick Start** - Get running in 5 minutes
4. **Usage** - Detailed usage examples
5. **API Reference** - Complete API documentation
6. **Troubleshooting** - Common issues and solutions

## 🔐 Security

### Reporting Security Issues

**DO NOT** open public issues for security vulnerabilities.

Instead:
1. Email security concerns to: starworks5@gmail.com
2. Include "SECURITY" in subject line
3. Provide detailed description
4. Allow time for fix before disclosure

See [SECURITY.md](SECURITY.md) for full policy.

## 💬 Community Guidelines

### Code of Conduct

Be respectful and inclusive:

- ✅ Use welcoming language
- ✅ Respect different viewpoints
- ✅ Accept constructive criticism
- ✅ Focus on what's best for the community
- ❌ No harassment or discrimination
- ❌ No trolling or insulting comments
- ❌ No unwelcome attention

### Getting Help

- 📖 **Documentation**: [docs/](docs/)
- ❓ **FAQ**: [docs/FAQ.md](docs/FAQ.md)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/endomorphosis/ipfs_accelerate_py/discussions)
- 🐛 **Issues**: [GitHub Issues](https://github.com/endomorphosis/ipfs_accelerate_py/issues)

## 🏆 Recognition

Contributors are recognized in:

- Git commit history
- Release notes
- Contributors list on GitHub
- Special mentions in documentation

## 📄 License

By contributing, you agree that your contributions will be licensed under the AGPLv3+ license. See [LICENSE](LICENSE) for details.

## ❓ Questions?

- Ask in [GitHub Discussions](https://github.com/endomorphosis/ipfs_accelerate_py/discussions)
- Check [FAQ](docs/FAQ.md)
- Email: starworks5@gmail.com

---

**Thank you for contributing to IPFS Accelerate Python!** 🚀

Every contribution, no matter how small, helps make this project better for everyone.
