# VSCode Development Configuration

This directory contains VSCode configuration files for the ipfs_accelerate_py project.

## Files

### tasks.json
Pre-configured development tasks accessible via VSCode's Command Palette.

**How to use:**
1. Press `Ctrl+Shift+P` (Windows/Linux) or `Cmd+Shift+P` (Mac)
2. Type "Tasks: Run Task"
3. Select the task you want to run

**Available Task Categories:**

#### Testing Tasks
- **Run All Tests** - Runs complete pytest suite (default build task: `Ctrl+Shift+B`)
- **Run Unit Tests** - Runs only fast unit tests (excludes integration and slow tests)
- **Run Integration Tests** - Runs integration tests
- **Run Benchmarks** - Runs performance benchmarks
- **Run MCP Tests** - Tests MCP components
- **Run Distributed Tests** - Tests distributed features
- **Check Code Coverage** - Generates coverage report

#### Code Quality Tasks
- **Lint Code (Flake8)** - Checks code style and quality
- **Type Check (mypy)** - Validates type hints
- **Format Code (Black)** - Auto-formats code to standards

#### Development Tasks
- **Install Dependencies** - Installs requirements.txt
- **Install Dev Dependencies** - Installs package with dev extras
- **Clean Build Artifacts** - Removes caches and build files

#### Docker Tasks
- **Build Docker Image** - Builds the Docker container
- **Start Docker Compose** - Starts services with docker-compose
- **Stop Docker Compose** - Stops docker-compose services
- **Start Docker CI Services** - Starts CI-specific services
- **Stop Docker CI Services** - Stops CI services

#### IPFS Tasks
- **Test IPFS Connection** - Verifies IPFS daemon connectivity

#### Dev Tools Tasks
- **Dev Tools: Check Python Compilation** - Validates Python syntax
- **Dev Tools: Check Imports** - Validates import statements
- **Dev Tools: Python Code Quality Check** - Comprehensive quality checks
- **Dev Tools: Audit Docstrings** - Checks documentation completeness
- **Dev Tools: Find Documentation** - Discovers all documentation files

### launch.json
Debugging configurations for Python code.

**Available Configurations:**
- **Python: Current File** - Debug the currently open Python file
- **Python: All Tests** - Debug all pytest tests
- **Python: Current Test File** - Debug tests in current file
- **Python: Specific Test** - Debug a specific test (select text first)
- **Python: Debug MCP Tests** - Debug MCP test suite
- **Python: Attach** - Attach debugger to running process (port 5678)

**How to use:**
1. Open the file or test you want to debug
2. Set breakpoints by clicking left of line numbers
3. Press `F5` or select "Run > Start Debugging"
4. Choose the appropriate debug configuration

### settings.json
Project-specific VSCode settings.

**Key Settings:**
- Python interpreter: `.venv/bin/python`
- Testing: pytest enabled
- Linting: flake8 with 120 char line length
- Formatting: black with format-on-save
- Auto-organize imports on save
- Hidden files: `__pycache__`, `.pytest_cache`, etc.

### extensions.json
Recommended VSCode extensions for this project.

**Recommended Extensions:**
- Python (ms-python.python)
- Pylance (ms-python.vscode-pylance)
- Black Formatter (ms-python.black-formatter)
- Flake8 (ms-python.flake8)
- mypy (ms-python.mypy-type-checker)
- Jupyter (ms-toolsai.jupyter)
- YAML (redhat.vscode-yaml)
- GitLens (eamodio.gitlens)
- GitHub Copilot (github.copilot)
- GitHub Pull Requests (github.vscode-pull-request-github)
- Docker (ms-azuretools.vscode-docker)
- Remote Containers (ms-vscode-remote.remote-containers)

## Quick Start

### First Time Setup

1. **Open workspace in VSCode:**
   ```bash
   code /path/to/ipfs_accelerate_py
   ```

2. **Install recommended extensions:**
   - VSCode will prompt you to install recommended extensions
   - Or manually: `Ctrl+Shift+P` → "Extensions: Show Recommended Extensions"

3. **Select Python interpreter:**
   - `Ctrl+Shift+P` → "Python: Select Interpreter"
   - Choose `.venv/bin/python`

4. **Install dependencies:**
   - `Ctrl+Shift+P` → "Tasks: Run Task" → "Install Dependencies"

### Common Workflows

#### Running Tests
```
Ctrl+Shift+B    - Run all tests (default)
Ctrl+Shift+P → Tasks: Run Task → Run Unit Tests
Ctrl+Shift+P → Tasks: Run Task → Run Integration Tests
```

#### Code Quality
```
Ctrl+Shift+P → Tasks: Run Task → Lint Code (Flake8)
Ctrl+Shift+P → Tasks: Run Task → Type Check (mypy)
Ctrl+Shift+P → Tasks: Run Task → Format Code (Black)
```

Or simply save files (format-on-save is enabled).

#### Debugging
```
F5              - Start debugging
F9              - Toggle breakpoint
F10             - Step over
F11             - Step into
Shift+F11       - Step out
Shift+F5        - Stop debugging
```

#### Docker
```
Ctrl+Shift+P → Tasks: Run Task → Build Docker Image
Ctrl+Shift+P → Tasks: Run Task → Start Docker Compose
Ctrl+Shift+P → Tasks: Run Task → Stop Docker Compose
```

## Keyboard Shortcuts Reference

| Shortcut | Action |
|----------|--------|
| `Ctrl+Shift+P` | Command Palette |
| `Ctrl+Shift+B` | Run default build task (Run All Tests) |
| `F5` | Start debugging |
| `Ctrl+` ` | Toggle terminal |
| `Ctrl+K Ctrl+T` | Change color theme |
| `Ctrl+,` | Open settings |

## Troubleshooting

### Tasks Not Showing Up
1. Reload VSCode: `Ctrl+Shift+P` → "Developer: Reload Window"
2. Check JSON validity: `python -m json.tool .vscode/tasks.json`
3. Verify file location: `.vscode/tasks.json` at workspace root

### Python Not Found
1. Check virtual environment exists: `ls .venv/bin/python`
2. Update interpreter: `Ctrl+Shift+P` → "Python: Select Interpreter"
3. Verify settings.json has correct path

### Task Execution Fails
1. Ensure virtual environment is activated
2. Install missing tools: `pip install pytest flake8 mypy black pytest-cov`
3. Check command exists: `which pytest` or `which flake8`

### Linter/Formatter Not Working
1. Install tools: `pip install flake8 black mypy`
2. Reload window: `Ctrl+Shift+P` → "Developer: Reload Window"
3. Check Python extension is installed and enabled

### IPFS Connection Test Fails
1. Ensure IPFS daemon is running: `ipfs daemon`
2. Check IPFS is accessible: `ipfs id`
3. Verify IPFS API port (default: 5001)

## Customization

### Adding New Tasks

Edit `.vscode/tasks.json` and add a new task object to the `tasks` array:

```json
{
    "label": "My Custom Task",
    "type": "shell",
    "command": "your-command",
    "args": ["arg1", "arg2"],
    "group": "test",
    "problemMatcher": []
}
```

### Modifying Settings

Edit `.vscode/settings.json` to customize project settings. Common modifications:

```json
{
    "python.linting.flake8Args": ["--max-line-length=100"],
    "python.formatting.blackArgs": ["--line-length=100"],
    "editor.formatOnSave": false
}
```

### Adding Input Variables

For tasks that need user input, add to the `inputs` array in tasks.json:

```json
{
    "id": "myInput",
    "description": "Enter value",
    "default": "default_value",
    "type": "promptString"
}
```

Then reference in task: `"${input:myInput}"`

## Integration with CI/CD

Many VSCode tasks mirror CI/CD pipeline commands, allowing you to:
- Run the same tests locally as in CI
- Validate code before pushing
- Debug CI failures locally

Match your local environment to CI:
1. Use same Python version
2. Install same dependencies
3. Run same test commands

## Related Documentation

- [VSCode Tasks Documentation](https://code.visualstudio.com/docs/editor/tasks)
- [Python in VSCode](https://code.visualstudio.com/docs/python/python-tutorial)
- [Debugging in VSCode](https://code.visualstudio.com/docs/editor/debugging)
- [AUTOMATION_README.md](../AUTOMATION_README.md) - Complete automation guide

## Support

**Questions or issues?**
- Create an issue with the `vscode` label
- Check [AUTOMATION_README.md](../AUTOMATION_README.md) for more details
- Refer to [QUICKSTART_VSCODE_INTEGRATION.md](../QUICKSTART_VSCODE_INTEGRATION.md)

---

**Last Updated:** January 29, 2026  
**Version:** 1.0  
**Status:** Active ✅
