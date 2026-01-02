# Contributing to Hospital Details Analysis

Thank you for your interest in contributing to this project! We welcome contributions from the community.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Pull Request Process](#pull-request-process)
- [Reporting Issues](#reporting-issues)

---

## Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inspiring community for all. Please be respectful and constructive in your interactions.

### Our Standards

✅ **Do:**
- Be respectful and inclusive
- Provide constructive feedback
- Focus on what is best for the community
- Show empathy towards other contributors

❌ **Don't:**
- Use inappropriate language or imagery
- Engage in personal attacks or trolling
- Publish others' private information
- Conduct yourself unprofessionally

---

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/your-username/Hospital_Details_Analysis.git
   cd Hospital_Details_Analysis
   ```
3. **Add upstream remote**:
   ```bash
   git remote add upstream https://github.com/original-owner/Hospital_Details_Analysis.git
   ```

---

## How to Contribute

### Types of Contributions

We welcome many types of contributions:

- 🐛 **Bug fixes**
- ✨ **New features**
- 📝 **Documentation improvements**
- 🎨 **UI/UX enhancements**
- 🧪 **Test coverage improvements**
- 🔧 **Performance optimizations**
- 🌐 **Translations**

### Before You Start

1. **Check existing issues** to see if someone is already working on it
2. **Create an issue** to discuss major changes before implementing
3. **Comment on the issue** to let others know you're working on it

---

## Development Setup

### Backend Development

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate
# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install dev dependencies (optional)
pip install pytest black flake8 mypy

# Set up environment
cp .env.example .env
# Edit .env and add your API keys

# Run backend
python -m uvicorn src.api.main:app --reload
```

### Frontend Development

```bash
cd frontend

# Install dependencies
npm install

# Run development server
npm run dev

# Run linting
npm run lint

# Build for production
npm run build
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_inference.py

# Run with coverage
pytest --cov=src tests/
```

---

## Coding Standards

### Python (Backend)

We follow [PEP 8](https://pep8.org/) with some modifications:

- **Line length**: 100 characters (not 79)
- **Formatting**: Use [Black](https://black.readthedocs.io/)
- **Linting**: Use [Flake8](https://flake8.pycqa.org/)
- **Type hints**: Encouraged but not required
- **Docstrings**: Use Google style docstrings

#### Example:

```python
def extract_features(case_study_text: str) -> dict:
    """
    Extract clinical features from case study text.

    Args:
        case_study_text: Unstructured clinical text

    Returns:
        Dictionary containing extracted features with confidence scores

    Raises:
        ExtractionError: If extraction fails after retries
    """
    # Implementation
    pass
```

#### Code Formatting

```bash
# Format code with Black
black src/

# Check linting
flake8 src/

# Type checking (optional)
mypy src/
```

### TypeScript (Frontend)

- Use **TypeScript** for all new code
- Follow **ESLint** rules
- Use **functional components** with hooks
- Keep components **small and focused**

```bash
# Lint frontend code
cd frontend
npm run lint
```

---

## Testing Guidelines

### Writing Tests

- **Write tests** for all new features
- **Update tests** when modifying existing code
- Aim for **high coverage** but prioritize **meaningful tests**
- Use **descriptive test names**

#### Test Structure

```python
def test_feature_extraction_with_complete_data():
    """Test that extraction works correctly with all fields present."""
    # Arrange
    case_text = "Patient is 62-year-old male..."

    # Act
    result = extractor.extract(case_text)

    # Assert
    assert result["age"]["value"] == 62.0
    assert result["age"]["confidence"] >= 0.9
```

### Test Categories

- **Unit tests**: Test individual functions/methods
- **Integration tests**: Test component interactions
- **End-to-end tests**: Test complete workflows

---

## Pull Request Process

### 1. Create a Feature Branch

```bash
# Update main branch
git checkout main
git pull upstream main

# Create feature branch
git checkout -b feature/your-feature-name
```

### 2. Make Your Changes

- Write clean, documented code
- Follow coding standards
- Add/update tests
- Update documentation if needed

### 3. Commit Your Changes

Use clear, descriptive commit messages:

```bash
git add .
git commit -m "Add multi-language support for extraction prompts

- Added French translation for system prompts
- Updated prompt builder to support language parameter
- Added language detection utility
- Tests for new functionality"
```

#### Commit Message Guidelines

- Use present tense ("Add feature" not "Added feature")
- First line: summary (50 chars or less)
- Blank line, then detailed description if needed
- Reference issues: "Fixes #123" or "Relates to #456"

### 4. Push and Create PR

```bash
# Push to your fork
git push origin feature/your-feature-name

# Go to GitHub and create Pull Request
```

### 5. PR Description

Include in your PR description:

- **What** changes were made
- **Why** the changes were necessary
- **How** to test the changes
- **Screenshots** (for UI changes)
- **Breaking changes** (if any)
- **Related issues** (#123)

#### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
How to test these changes

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex code
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] All tests passing
```

### 6. Code Review

- Address review feedback promptly
- Push new commits to the same branch
- Request re-review when ready

### 7. Merge

Once approved, a maintainer will merge your PR.

---

## Reporting Issues

### Bug Reports

When reporting bugs, include:

- **Clear title** describing the issue
- **Steps to reproduce** the problem
- **Expected behavior** vs **actual behavior**
- **Environment details**:
  - OS and version
  - Python version
  - Node.js version (for frontend)
  - Model being used (gpt-4o, etc.)
- **Error messages** (full stack trace)
- **Screenshots** (if applicable)

#### Bug Report Template

```markdown
**Describe the bug**
A clear description of what the bug is.

**To Reproduce**
1. Go to '...'
2. Click on '...'
3. See error

**Expected behavior**
What you expected to happen.

**Screenshots**
If applicable, add screenshots.

**Environment:**
- OS: [e.g., Windows 11]
- Python: [e.g., 3.10.5]
- Browser: [e.g., Chrome 120]

**Additional context**
Any other relevant information.
```

### Feature Requests

For feature requests, include:

- **Use case**: Why is this needed?
- **Proposed solution**: How should it work?
- **Alternatives considered**: Other approaches you've thought about
- **Additional context**: Examples, mockups, etc.

---

## Development Workflow

### Branching Strategy

- `main`: Production-ready code
- `develop`: Integration branch (if used)
- `feature/*`: New features
- `bugfix/*`: Bug fixes
- `hotfix/*`: Urgent production fixes

### Release Process

1. Update version in relevant files
2. Update CHANGELOG.md
3. Create release branch
4. Test thoroughly
5. Merge to main
6. Tag release
7. Deploy

---

## Areas Needing Help

We especially welcome contributions in these areas:

1. **Documentation**
   - Tutorial videos
   - Use case examples
   - API documentation improvements

2. **Testing**
   - Increasing test coverage
   - End-to-end tests
   - Performance tests

3. **Features**
   - See [GitHub Issues](https://github.com/yourusername/Hospital_Details_Analysis/issues) labeled "help wanted"
   - Check the [Roadmap](README.md#roadmap) in README

4. **Performance**
   - Caching improvements
   - Query optimization
   - Bundle size reduction (frontend)

---

## Questions?

- **General Questions**: [GitHub Discussions](https://github.com/yourusername/Hospital_Details_Analysis/discussions)
- **Bug Reports**: [GitHub Issues](https://github.com/yourusername/Hospital_Details_Analysis/issues)
- **Email**: your-email@example.com

---

## Recognition

Contributors will be recognized in:
- GitHub contributors page
- README.md acknowledgments section
- Release notes

Thank you for contributing! 🎉
