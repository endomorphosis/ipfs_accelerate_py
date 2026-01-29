# Quick Start: VSCode Tasks Integration

**Phase:** 1 of 4  
**Timeline:** 1-2 days  
**Difficulty:** Easy  
**Value:** Immediate productivity boost

---

## 🎯 Goal

Set up 45+ one-click development tasks in VSCode for the ipfs_accelerate_py project, enabling developers to execute common operations without remembering complex command-line syntax.

---

## 📋 Prerequisites

- [ ] VSCode installed (version 1.60+)
- [ ] Python extension for VSCode installed
- [ ] Git command line tools
- [ ] Python 3.8+ with pip
- [ ] Access to ipfs_accelerate_py repository

---

## 🚀 Step-by-Step Implementation

### Step 1: Create .vscode Directory (5 minutes)

```bash
cd /home/runner/work/ipfs_accelerate_py/ipfs_accelerate_py
mkdir -p .vscode
cd .vscode
```

### Step 2: Download tasks.json Template (10 minutes)

**Option A: Direct Download**
```bash
curl -o tasks.json.template \
  https://raw.githubusercontent.com/endomorphosis/ipfs_datasets_py/main/.vscode/tasks.json
```

**Option B: Manual Copy**
1. Visit: https://github.com/endomorphosis/ipfs_datasets_py/blob/main/.vscode/tasks.json
2. Copy contents
3. Save as `tasks.json.template` in `.vscode/`

### Step 3: Create Adapted tasks.json (30-45 minutes)

Create `.vscode/tasks.json` with ipfs_accelerate_py-specific tasks:

```json
{
	"version": "2.0.0",
	"tasks": [
		{
			"label": "Run All Tests",
			"type": "shell",
			"command": "${workspaceFolder}/.venv/bin/python",
			"windows": {
				"command": "${workspaceFolder}\\.venv\\Scripts\\python.exe"
			},
			"args": [
				"-m",
				"pytest",
				"tests/",
				"-v"
			],
			"group": {
				"kind": "test",
				"isDefault": true
			},
			"problemMatcher": [],
			"options": {
				"cwd": "${workspaceFolder}"
			}
		},
		{
			"label": "Run Unit Tests",
			"type": "shell",
			"command": "${workspaceFolder}/.venv/bin/python",
			"windows": {
				"command": "${workspaceFolder}\\.venv\\Scripts\\python.exe"
			},
			"args": [
				"-m",
				"pytest",
				"tests/",
				"-v",
				"-m",
				"unit"
			],
			"group": "test",
			"problemMatcher": []
		},
		{
			"label": "Run Integration Tests",
			"type": "shell",
			"command": "${workspaceFolder}/.venv/bin/python",
			"windows": {
				"command": "${workspaceFolder}\\.venv\\Scripts\\python.exe"
			},
			"args": [
				"-m",
				"pytest",
				"tests/",
				"-v",
				"-m",
				"integration"
			],
			"group": "test",
			"problemMatcher": []
		},
		{
			"label": "Install Dependencies",
			"type": "shell",
			"command": "${workspaceFolder}/.venv/bin/pip",
			"windows": {
				"command": "${workspaceFolder}\\.venv\\Scripts\\pip.exe"
			},
			"args": [
				"install",
				"-r",
				"requirements.txt"
			],
			"group": "build",
			"problemMatcher": []
		},
		{
			"label": "Lint Code (Flake8)",
			"type": "shell",
			"command": "${workspaceFolder}/.venv/bin/flake8",
			"windows": {
				"command": "${workspaceFolder}\\.venv\\Scripts\\flake8.exe"
			},
			"args": [
				"ipfs_accelerate_py/",
				"--max-line-length=120"
			],
			"group": "test",
			"problemMatcher": []
		},
		{
			"label": "Type Check (mypy)",
			"type": "shell",
			"command": "${workspaceFolder}/.venv/bin/mypy",
			"windows": {
				"command": "${workspaceFolder}\\.venv\\Scripts\\mypy.exe"
			},
			"args": [
				"ipfs_accelerate_py/"
			],
			"group": "test",
			"problemMatcher": []
		},
		{
			"label": "Build Docker Image",
			"type": "shell",
			"command": "docker",
			"args": [
				"build",
				"-t",
				"ipfs-accelerate-py:latest",
				"."
			],
			"group": "build",
			"problemMatcher": []
		},
		{
			"label": "Start Docker Compose",
			"type": "shell",
			"command": "docker-compose",
			"args": [
				"up",
				"-d"
			],
			"group": "build",
			"problemMatcher": []
		},
		{
			"label": "Stop Docker Compose",
			"type": "shell",
			"command": "docker-compose",
			"args": [
				"down"
			],
			"group": "build",
			"problemMatcher": []
		},
		{
			"label": "Run Benchmarks",
			"type": "shell",
			"command": "${workspaceFolder}/.venv/bin/python",
			"windows": {
				"command": "${workspaceFolder}\\.venv\\Scripts\\python.exe"
			},
			"args": [
				"-m",
				"pytest",
				"benchmarks/",
				"-v"
			],
			"group": "test",
			"problemMatcher": []
		},
		{
			"label": "Test IPFS Connection",
			"type": "shell",
			"command": "${workspaceFolder}/.venv/bin/python",
			"windows": {
				"command": "${workspaceFolder}\\.venv\\Scripts\\python.exe"
			},
			"args": [
				"-c",
				"import ipfshttpclient; client = ipfshttpclient.connect(); print(f'Connected to IPFS: {client.id()}')"
			],
			"group": "test",
			"problemMatcher": []
		},
		{
			"label": "Clean Build Artifacts",
			"type": "shell",
			"command": "bash",
			"args": [
				"-c",
				"find . -type d -name '__pycache__' -exec rm -rf {} + ; find . -type f -name '*.pyc' -delete ; rm -rf build/ dist/ *.egg-info"
			],
			"group": "build",
			"problemMatcher": []
		},
		{
			"label": "Generate Documentation",
			"type": "shell",
			"command": "${workspaceFolder}/.venv/bin/sphinx-build",
			"windows": {
				"command": "${workspaceFolder}\\.venv\\Scripts\\sphinx-build.exe"
			},
			"args": [
				"-b",
				"html",
				"docs/source",
				"docs/build/html"
			],
			"group": "build",
			"problemMatcher": []
		},
		{
			"label": "Check Code Coverage",
			"type": "shell",
			"command": "${workspaceFolder}/.venv/bin/pytest",
			"windows": {
				"command": "${workspaceFolder}\\.venv\\Scripts\\pytest.exe"
			},
			"args": [
				"--cov=ipfs_accelerate_py",
				"--cov-report=html",
				"--cov-report=term",
				"tests/"
			],
			"group": "test",
			"problemMatcher": []
		},
		{
			"label": "Format Code (Black)",
			"type": "shell",
			"command": "${workspaceFolder}/.venv/bin/black",
			"windows": {
				"command": "${workspaceFolder}\\.venv\\Scripts\\black.exe"
			},
			"args": [
				"ipfs_accelerate_py/",
				"tests/"
			],
			"group": "build",
			"problemMatcher": []
		}
	]
}
```

### Step 4: Create launch.json for Debugging (15 minutes)

Create `.vscode/launch.json`:

```json
{
	"version": "0.2.0",
	"configurations": [
		{
			"name": "Python: Current File",
			"type": "python",
			"request": "launch",
			"program": "${file}",
			"console": "integratedTerminal",
			"justMyCode": true
		},
		{
			"name": "Python: All Tests",
			"type": "python",
			"request": "launch",
			"module": "pytest",
			"args": [
				"tests/",
				"-v"
			],
			"console": "integratedTerminal",
			"justMyCode": false
		},
		{
			"name": "Python: Specific Test",
			"type": "python",
			"request": "launch",
			"module": "pytest",
			"args": [
				"${file}",
				"-v"
			],
			"console": "integratedTerminal",
			"justMyCode": false
		},
		{
			"name": "Python: Attach",
			"type": "python",
			"request": "attach",
			"connect": {
				"host": "localhost",
				"port": 5678
			},
			"pathMappings": [
				{
					"localRoot": "${workspaceFolder}",
					"remoteRoot": "."
				}
			]
		}
	]
}
```

### Step 5: Create settings.json (10 minutes)

Create `.vscode/settings.json`:

```json
{
	"python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python",
	"python.testing.pytestEnabled": true,
	"python.testing.unittestEnabled": false,
	"python.testing.pytestArgs": [
		"tests"
	],
	"python.linting.enabled": true,
	"python.linting.pylintEnabled": false,
	"python.linting.flake8Enabled": true,
	"python.linting.flake8Args": [
		"--max-line-length=120"
	],
	"python.formatting.provider": "black",
	"python.formatting.blackArgs": [
		"--line-length=120"
	],
	"editor.formatOnSave": true,
	"editor.codeActionsOnSave": {
		"source.organizeImports": true
	},
	"files.exclude": {
		"**/__pycache__": true,
		"**/*.pyc": true,
		"**/.pytest_cache": true,
		"**/.mypy_cache": true
	},
	"files.associations": {
		"*.yml": "yaml",
		"*.yaml": "yaml"
	},
	"terminal.integrated.env.linux": {
		"PYTHONPATH": "${workspaceFolder}"
	},
	"terminal.integrated.env.osx": {
		"PYTHONPATH": "${workspaceFolder}"
	},
	"terminal.integrated.env.windows": {
		"PYTHONPATH": "${workspaceFolder}"
	}
}
```

### Step 6: Create .vscode/extensions.json (5 minutes)

Recommend useful extensions:

```json
{
	"recommendations": [
		"ms-python.python",
		"ms-python.vscode-pylance",
		"ms-python.black-formatter",
		"ms-toolsai.jupyter",
		"redhat.vscode-yaml",
		"eamodio.gitlens",
		"github.copilot"
	]
}
```

### Step 7: Test VSCode Tasks (15 minutes)

1. **Open VSCode**
   ```bash
   code /home/runner/work/ipfs_accelerate_py/ipfs_accelerate_py
   ```

2. **Test Task Runner**
   - Press `Ctrl+Shift+P` (Windows/Linux) or `Cmd+Shift+P` (Mac)
   - Type "Tasks: Run Task"
   - Select "Install Dependencies"
   - Verify it runs successfully

3. **Test Each Task Category**
   - Testing tasks: "Run All Tests"
   - Development tasks: "Lint Code (Flake8)"
   - Docker tasks: "Build Docker Image"
   - Build tasks: "Clean Build Artifacts"

4. **Test Debugging**
   - Open a test file
   - Press F5
   - Select "Python: Specific Test"
   - Verify debugger works

### Step 8: Create Documentation (20 minutes)

Create `.vscode/README.md`:

```markdown
# VSCode Development Tasks

## Available Tasks

### Testing Tasks
- **Run All Tests** (Ctrl+Shift+B) - Runs complete test suite
- **Run Unit Tests** - Runs only unit tests
- **Run Integration Tests** - Runs only integration tests
- **Run Benchmarks** - Runs performance benchmarks
- **Check Code Coverage** - Generates coverage report

### Code Quality Tasks
- **Lint Code (Flake8)** - Checks code style
- **Type Check (mypy)** - Validates type hints
- **Format Code (Black)** - Auto-formats code

### Development Tasks
- **Install Dependencies** - Installs requirements
- **Clean Build Artifacts** - Removes build files
- **Generate Documentation** - Builds Sphinx docs

### Docker Tasks
- **Build Docker Image** - Builds container
- **Start Docker Compose** - Starts services
- **Stop Docker Compose** - Stops services

### IPFS Tasks
- **Test IPFS Connection** - Verifies IPFS connectivity

## How to Use

1. Press `Ctrl+Shift+P` (Windows/Linux) or `Cmd+Shift+P` (Mac)
2. Type "Tasks: Run Task"
3. Select the task you want to run

## Debugging

1. Open the file you want to debug
2. Set breakpoints by clicking left of line numbers
3. Press F5 or select "Run > Start Debugging"
4. Choose appropriate debug configuration

## Keyboard Shortcuts

- `Ctrl+Shift+B` - Run default build task (Run All Tests)
- `F5` - Start debugging
- `Ctrl+Shift+P` - Command palette

## Customization

Edit `.vscode/tasks.json` to add or modify tasks.
```

---

## ✅ Validation Checklist

After implementation, verify:

- [ ] `.vscode/` directory created
- [ ] `tasks.json` created and valid JSON
- [ ] `launch.json` created and valid JSON
- [ ] `settings.json` created and valid JSON
- [ ] `extensions.json` created and valid JSON
- [ ] README.md created
- [ ] VSCode can be opened in workspace
- [ ] Tasks menu populated with all tasks
- [ ] At least 3 tasks tested successfully
- [ ] Debugging configuration works
- [ ] Python extension recognizes settings
- [ ] Team can access and use tasks

---

## 📊 Expected Results

After completing this integration:

✅ **15 tasks** immediately available  
✅ **One-click** test execution  
✅ **Integrated** debugging  
✅ **Consistent** development environment  
✅ **30% reduction** in manual command execution  

---

## 🐛 Troubleshooting

### Tasks Not Showing Up
1. Reload VSCode: `Ctrl+Shift+P` → "Developer: Reload Window"
2. Check JSON validity: `python -m json.tool .vscode/tasks.json`
3. Verify file location: `.vscode/tasks.json` at workspace root

### Python Not Found
1. Check virtual environment: `ls .venv/bin/python`
2. Update interpreter: `Ctrl+Shift+P` → "Python: Select Interpreter"
3. Verify settings.json has correct path

### Task Fails to Execute
1. Check command exists: `which pytest` or `which flake8`
2. Install missing tools: `pip install pytest flake8 mypy black`
3. Check working directory in task configuration

### Docker Commands Fail
1. Verify Docker installed: `docker --version`
2. Check Docker daemon running: `docker ps`
3. Verify permissions: `sudo usermod -aG docker $USER`

---

## 🎓 Training Resources

### For Team Members

**15-Minute Introduction:**
1. Open VSCode in workspace
2. Show Command Palette (`Ctrl+Shift+P`)
3. Demonstrate "Tasks: Run Task"
4. Run 3-4 common tasks
5. Show debugging configuration

**Video Tutorial:**
Record a 5-minute screen recording showing:
- Opening task runner
- Running tests
- Viewing results
- Starting debugging

**Cheat Sheet:**
```
Quick Reference:
- Run tests: Ctrl+Shift+P → "Tasks: Run Task" → "Run All Tests"
- Lint code: Ctrl+Shift+P → "Tasks: Run Task" → "Lint Code"
- Debug: F5 → Select configuration
- Format: Ctrl+Shift+P → "Format Document"
```

---

## 📈 Success Metrics

Track these metrics after 1 week:

| Metric | Target |
|--------|--------|
| Team members using VSCode | 90%+ |
| Tasks executed per day | 20+ |
| Manual CLI commands | 70% reduction |
| Time saved per developer | 2 hours/week |
| Developer satisfaction | 8/10 or higher |

---

## 🚀 Next Steps

After completing VSCode integration:

1. ✅ **Celebrate the win!** You've saved the team significant time
2. 📊 **Track usage** for 1 week
3. 🔄 **Gather feedback** from team
4. ➕ **Add more tasks** based on feedback
5. 📖 **Move to Phase 2:** Issue-to-PR workflow integration

---

## 📚 Additional Resources

- [VSCode Tasks Documentation](https://code.visualstudio.com/docs/editor/tasks)
- [Python in VSCode](https://code.visualstudio.com/docs/python/python-tutorial)
- [Debugging in VSCode](https://code.visualstudio.com/docs/editor/debugging)
- [ipfs_datasets_py tasks.json](https://github.com/endomorphosis/ipfs_datasets_py/blob/main/.vscode/tasks.json)

---

**Questions?** Create an issue with the `vscode` label or contact the development team.

**Ready for more automation?** Check out [AUTOMATION_INTEGRATION_PLAN.md](./AUTOMATION_INTEGRATION_PLAN.md) for the complete roadmap!
