# Publishing Guide

This guide explains how to publish `llm-instrumentation` to PyPI using GitHub Actions.

## GitHub Actions Workflow

The repository includes a GitHub Actions workflow ([.github/workflows/python-publish.yml](../.github/workflows/python-publish.yml)) that automatically builds and publishes the package to PyPI when you create a GitHub release.

### How It Works

1. **Trigger**: The workflow runs when you publish a GitHub release
2. **Build**: Builds the package from the `llm-instrumentation` subdirectory
3. **Upload**: Uploads the built distributions as artifacts
4. **Publish**: Publishes to PyPI using Trusted Publishing (no API tokens needed!)

### Recent Changes

The workflow was updated to:
- Change directory to `llm-instrumentation` before building (fixes the "no pyproject.toml" error)
- Use the correct path `llm-instrumentation/dist/` for artifacts
- Set the PyPI project URL to `https://pypi.org/project/llm-instrumentation/`

## Setting Up PyPI Publishing

### 1. Configure Trusted Publishing on PyPI

Trusted Publishing is the modern, secure way to publish to PyPI without API tokens.

1. Go to [PyPI](https://pypi.org/) and log in (or create an account)
2. For first-time publishing, you'll need to create the project manually or use an API token once
3. Go to your project settings on PyPI
4. Navigate to "Publishing" → "Add a new publisher"
5. Configure the GitHub publisher:
   - **Owner**: `rubenfb23`
   - **Repository**: `STRAP-LLM`
   - **Workflow**: `python-publish.yml`
   - **Environment**: `pypi`

### 2. Create a PyPI Environment in GitHub

1. Go to your GitHub repository settings
2. Navigate to "Environments"
3. Create a new environment named `pypi`
4. (Optional) Add protection rules:
   - Required reviewers
   - Wait timer
   - Restrict to specific branches

## Publishing a New Release

### Step 1: Update Version

Update the version in [pyproject.toml](pyproject.toml):

```toml
[project]
name = "llm-instrumentation"
version = "0.2.0"  # Increment this
```

### Step 2: Commit and Push

```bash
git add llm-instrumentation/pyproject.toml
git commit -m "Bump version to 0.2.0"
git push origin main
```

### Step 3: Create a GitHub Release

1. Go to your repository on GitHub
2. Click "Releases" → "Create a new release"
3. Click "Choose a tag" and create a new tag (e.g., `v0.2.0`)
4. Set the release title (e.g., "v0.2.0")
5. Add release notes describing changes
6. Click "Publish release"

### Step 4: Monitor the Workflow

1. Go to the "Actions" tab in your repository
2. Watch the "Upload Python Package" workflow run
3. The workflow will:
   - Build the package in the `llm-instrumentation` directory
   - Upload artifacts
   - Publish to PyPI (requires approval if you set up protection rules)

### Step 5: Verify Publication

After the workflow completes, verify your package on PyPI:
- https://pypi.org/project/llm-instrumentation/

## Manual Publishing (Alternative)

If you need to publish manually (not recommended):

### Using Twine with API Token

1. Create an API token on PyPI
2. Install twine: `pip install twine`
3. Build the package:
   ```bash
   cd llm-instrumentation
   python -m build
   ```
4. Upload to PyPI:
   ```bash
   python -m twine upload dist/*
   ```
5. Enter your API token when prompted

### Testing on Test PyPI First

Before publishing to the main PyPI, test on Test PyPI:

```bash
python -m twine upload --repository testpypi dist/*
```

Then install and test:
```bash
pip install --index-url https://test.pypi.org/simple/ llm-instrumentation
```

## Version Numbering

Follow [Semantic Versioning](https://semver.org/):

- **Major** (1.0.0): Breaking changes
- **Minor** (0.2.0): New features, backwards compatible
- **Patch** (0.1.1): Bug fixes, backwards compatible

## Troubleshooting

### "Source does not appear to be a Python project"

**Cause**: The workflow is running from the wrong directory.

**Fix**: Ensure the workflow includes `cd llm-instrumentation` before building (already fixed in the current workflow).

### "Trusted publishing is not configured"

**Cause**: PyPI doesn't have the GitHub publisher configured.

**Fix**: Follow the "Configure Trusted Publishing on PyPI" steps above.

### "Environment protection rules not satisfied"

**Cause**: The `pypi` environment has protection rules (like required reviewers).

**Fix**: Either:
1. Remove protection rules from the environment, or
2. Approve the deployment in the Actions tab

### Build Fails

**Cause**: Issues with dependencies or code.

**Fix**: Test the build locally first:
```bash
cd llm-instrumentation
python -m build
```

## Best Practices

1. **Always test locally** before creating a release
2. **Update CHANGELOG.md** with release notes
3. **Tag releases** with semantic versions (v0.1.0, v0.2.0, etc.)
4. **Use branch protection** on main to prevent accidental pushes
5. **Set up environment protection rules** for the `pypi` environment
6. **Keep README.md updated** with installation instructions

## Resources

- [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/)
- [GitHub Actions for Python](https://docs.github.com/en/actions/automating-builds-and-tests/building-and-testing-python)
- [Python Packaging Guide](https://packaging.python.org/)
- [Semantic Versioning](https://semver.org/)
