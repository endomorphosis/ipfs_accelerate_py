# Documentation Maintenance Workflow

This directory contains a GitHub Actions workflow for automated weekly documentation maintenance.

## Overview

The `documentation-maintenance.yml` workflow runs automatically every Monday at 9:00 AM UTC to help maintain high-quality documentation for both users and programming agents.

## What It Does

### 1. **Codebase Analysis** 📊
- Scans all Python files in the repository
- Identifies classes, functions, and modules
- Measures documentation coverage (docstrings)
- Creates a detailed analysis report

### 2. **API Documentation Generation** 📚
- Generates HTML API documentation using `pdoc3`
- Creates comprehensive reference documentation
- Places generated docs in `docs/api/` directory

### 3. **Example Validation** ✅
- Extracts Python code blocks from all markdown files
- Validates syntax of code examples
- Reports any syntax errors found

### 4. **Missing Documentation Detection** 🔍
- Identifies undocumented classes and functions
- Prioritizes files by number of missing docstrings
- Creates actionable TODO list

### 5. **README Coverage Analysis** 💡
- Checks which modules are mentioned in README
- Suggests additions for better coverage
- Generates improvement recommendations

### 6. **Automated Reporting** 📝
- Creates comprehensive documentation report
- Generates TODO list for contributors
- Provides README improvement suggestions

## Schedule

The workflow runs:
- **Automatically**: Every Monday at 9:00 AM UTC
- **Manually**: Can be triggered via GitHub Actions UI (workflow_dispatch)

## Output Files

When the workflow runs, it generates:

1. **`DOCUMENTATION_REPORT.md`**: Main report with analysis summary and recommendations
2. **`DOCUMENTATION_TODO.md`**: Prioritized list of items needing documentation
3. **`README_SUGGESTIONS.md`**: Suggestions for improving the README
4. **`docs/api/`**: Generated API documentation (HTML)
5. **`documentation_analysis.json`**: Raw analysis data

## How It Helps Users

### For End Users
- **Up-to-date Documentation**: API docs are regenerated weekly
- **Clear Entry Points**: Report identifies main modules and usage patterns
- **Working Examples**: Code examples are validated for syntax errors

### For Programming Agents
The workflow provides:
- **Comprehensive API Reference**: Complete function signatures and descriptions
- **Module Structure**: Clear understanding of codebase organization
- **Entry Points**: Identification of main classes and functions
- **Usage Patterns**: Common patterns extracted from examples
- **Documentation Coverage**: Understanding of which areas are well-documented

## Workflow Features

### Permissions
The workflow requires:
- `contents: write` - To commit documentation updates
- `pull-requests: write` - To create PRs with updates
- `issues: write` - To create issues if needed

### Pull Request Creation
If documentation updates are needed, the workflow:
1. Creates a new branch: `docs/maintenance-{run_number}`
2. Commits all documentation changes
3. Opens a Pull Request with detailed summary
4. Labels the PR as `documentation` and `automated`

### Artifacts
All analysis files are uploaded as workflow artifacts and retained for 30 days for review.

## Manual Trigger

To run the workflow manually:
1. Go to **Actions** tab in GitHub
2. Select **"Weekly Documentation Maintenance"**
3. Click **"Run workflow"**
4. Select the branch and click **"Run workflow"**

## Customization

You can customize the workflow by editing `.github/workflows/documentation-maintenance.yml`:

### Change Schedule
```yaml
schedule:
  - cron: '0 9 * * 1'  # Mondays at 9 AM UTC
```

Use [crontab.guru](https://crontab.guru/) to generate different schedules.

### Adjust Analysis Scope
Modify the filter in `analyze_codebase.py` to include/exclude directories:
```python
py_files = [
    f for f in py_files 
    if 'venv' not in str(f)  # Add more filters here
]
```

### Change Documentation Tool
Replace `pdoc3` with alternatives like:
- `sphinx` - For more complex projects
- `mkdocs` - For markdown-based docs
- `pydoc` - Built-in Python documentation

## Integration with Development Workflow

### For Contributors
1. Check `DOCUMENTATION_TODO.md` for items to document
2. Review `README_SUGGESTIONS.md` before updating README
3. Ensure new code includes docstrings to maintain coverage

### For Maintainers
1. Review weekly PRs from the workflow
2. Merge documentation updates regularly
3. Monitor documentation coverage trends
4. Address high-priority items in TODO list

## Troubleshooting

### Workflow Fails
- Check the workflow logs in GitHub Actions
- Ensure all required dependencies are in requirements.txt
- Verify Python syntax in inline scripts

### No PR Created
- Workflow only creates PRs when there are documentation changes
- Check if documentation is already up-to-date
- Review workflow logs for commit status

### API Docs Not Generated
- Ensure `pdoc3` is installed correctly
- Check for syntax errors in Python files
- Some modules may fail silently - check logs

## Benefits

### Continuous Documentation Quality
- Regular analysis prevents documentation debt
- Automated detection of missing docs
- Validated code examples ensure accuracy

### Better Developer Experience
- Always-current API documentation
- Clear entry points for new contributors
- Prioritized list of documentation needs

### AI/Agent Friendly
- Structured documentation format
- Comprehensive API coverage
- Clear module relationships
- Usage pattern documentation

## Technical Details

### Tools Used
- **pdoc3**: API documentation generation
- **ast**: Python code analysis
- **gitpython**: Git operations
- **markdown**: Markdown processing
- **beautifulsoup4**: HTML parsing

### Analysis Metrics
- Module-level docstring coverage
- Class documentation percentage
- Function documentation percentage
- Code example validation rate
- README module coverage

### Performance
- Typical runtime: 5-10 minutes
- Analysis scales with codebase size
- Can handle thousands of Python files

## Future Enhancements

Potential improvements:
- [ ] Integration with ReadTheDocs
- [ ] Automatic issue creation for missing docs
- [ ] Documentation quality scoring
- [ ] Cross-reference validation
- [ ] Changelog generation
- [ ] API breaking change detection

## Contributing

To improve the documentation workflow:
1. Edit `.github/workflows/documentation-maintenance.yml`
2. Test changes with manual workflow trigger
3. Submit PR with description of improvements

## License

This workflow is part of the ipfs_accelerate_py project and follows the same license (AGPLv3+).
