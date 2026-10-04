# Publishing ipfs_accelerate_py to PyPI

This guide outlines the steps required to publish this package to PyPI.

## Prerequisites

Before publishing, ensure you have the following:

1. PyPI account with appropriate permissions
2. GitHub Trusted Publishing setup (preferred approach):
   - GitHub Environment named "pypi" configured
   - PyPI Project configured to allow trusted publishing from this repository
   - No secrets required with trusted publishing

Alternatively, for token-based authentication (legacy approach):
   - GitHub repository secrets setup:
     - `PYPI_API_TOKEN`: Your PyPI API token

## Manual Publishing

If you need to publish manually, follow these steps:

1. Install build tools:
```bash
pip install --upgrade pip
pip install build twine
```

2. Build the package:
```bash
python -m build
```

3. Check the package:
```bash
twine check dist/*
```

4. Upload to TestPyPI first to verify (recommended):
```bash
twine upload --repository-url https://test.pypi.org/legacy/ dist/*
```

5. Install from TestPyPI to verify installation works:
```bash
pip install --index-url https://test.pypi.org/simple/ ipfs_accelerate_py
```

6. Upload to PyPI:
```bash
twine upload dist/*
```

## Automated Publishing via GitHub Actions

To publish automatically via GitHub Actions:

1. Set up GitHub Trusted Publishing (one-time setup):
   - Create a GitHub Environment named "pypi" in repository settings
   - On PyPI, go to Account Settings → Add Publishing
   - Select "GitHub Actions" publishing method
   - Configure the repository and environment name

2. Publishing methods:
   
   a) Automatic triggering with a version bump:
      - Update the version in both `pyproject.toml` and `setup.py`
      - Commit and push to the main branch
      - The GitHub Actions workflow will automatically detect the version change
      - The workflow will build and publish to PyPI when the version number is increased
   
   b) Manual triggering (useful for troubleshooting):
      - Go to GitHub Actions → "Upload Python Package" workflow
      - Click "Run workflow" button in the UI
      - Select "Force publishing regardless of version change" if you want to bypass version check
      - Click "Run workflow" to start the process

3. The publishing uses OpenID Connect (OIDC) trusted publishing for secure authentication.

Note: Manual GitHub release creation is no longer needed - publishing happens automatically whenever the version is bumped or manually triggered.

## Version Management

Always ensure version numbers are consistent:
- `setup.py`: version='X.Y.Z'
- `pyproject.toml`: version = "X.Y.Z"

Increment the version number for each new release according to semantic versioning.

## Post-Publish Verification

After publishing, verify the package can be installed:

```bash
# In a fresh virtual environment
pip install ipfs_accelerate_py
```

Test that the installed package works correctly:

```python
import ipfs_accelerate_py
```