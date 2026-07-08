# Zero-Touch Multi-Platform Installers

Complete automated installation system for ipfs_accelerate_py cache infrastructure supporting all major platforms and architectures.

## Quick Install

### One-Line Install (Unix/Linux/macOS)
```bash
curl -fsSL https://raw.githubusercontent.com/endomorphosis/ipfs_accelerate_py/main/install/install.sh | bash
```

### One-Line Install (Windows)
```powershell
iwr -useb https://raw.githubusercontent.com/endomorphosis/ipfs_accelerate_py/main/install/install.ps1 | iex
```

### Docker (All Platforms)
```bash
docker run -v ~/.cache/ipfs_accelerate_py:/cache endomorphosis/ipfs-accelerate-py-cache:latest
```

## Supported Platforms

| Platform | Architecture | Status | Notes |
|----------|-------------|--------|-------|
| **Linux** | x86_64 | ✅ Tested | Ubuntu 20.04+, Debian 11+, RHEL 8+ |
| **Linux** | ARM64/aarch64 | ✅ Tested | Ubuntu 20.04+, Debian 11+ |
| **Linux** | ARMv7 | ✅ Supported | Raspberry Pi 3/4 |
| **macOS** | x86_64 (Intel) | ✅ Tested | macOS 11+ |
| **macOS** | ARM64 (Apple Silicon) | ✅ Tested | macOS 11+ (M1/M2/M3) |
| **Windows** | x64 | ✅ Tested | Windows 10/11 |
| **Windows** | ARM64 | ✅ Supported | Windows 11 ARM |
| **FreeBSD** | x86_64 | ⚠️ Experimental | FreeBSD 13+ |

## Installation Profiles

Choose the right profile for your needs:

### 1. Minimal (50MB)
Core cache infrastructure only, no ML dependencies.

```bash
./install.sh --profile minimal
```

**Includes:**
- Base cache with CID support
- CID index for fast lookups
- Basic cache adapters

**Use when:**
- You only need caching functionality
- Minimal disk space/bandwidth
- No ML model support needed

### 2. Standard (200MB) - **Recommended**
Cache infrastructure with CLI and API integrations.

```bash
./install.sh --profile standard
```

**Includes:**
- Everything from Minimal
- All 9 CLI integrations (GitHub, Copilot, VSCode, etc.)
- All 12 API wrappers (OpenAI, Claude, Groq, etc.)
- HuggingFace Hub support

**Use when:**
- You want comprehensive caching
- Using LLM APIs or inference engines
- Need CLI tool integrations

### 3. Full (2GB)
Complete installation with ML model support.

```bash
./install.sh --profile full
```

**Includes:**
- Everything from Standard
- PyTorch and transformers
- Inference engine support (vLLM, TGI, TEI, OVMS)
- Model optimization tools (ONNX, OpenVINO)

**Use when:**
- Running local ML models
- Using inference engines
- Need complete functionality

### 4. CLI-only (100MB)
Just CLI integrations without API wrappers.

```bash
./install.sh --profile cli
```

**Includes:**
- Core cache infrastructure
- All CLI integrations
- No API wrappers

**Use when:**
- Only using command-line tools
- Don't need programmatic API access

## Advanced Installation Options

### Silent Installation
No interactive prompts (useful for automation):

```bash
./install.sh --silent --profile standard
```

### Custom Cache Directory
Specify a custom location for cache:

```bash
./install.sh --cache-dir /opt/ipfs_accelerate_cache
```

### Skip CLI Tools
Install Python packages only, skip system CLI tools:

```bash
./install.sh --no-cli-tools
```

### Use Existing Virtual Environment
Skip venv creation (use current environment):

```bash
source .venv/bin/activate
./install.sh --skip-venv
```

### Verify Installation
Check if installation is working:

```bash
./install.sh --verify
```

### Verbose Output
See detailed installation progress:

```bash
./install.sh --verbose
```

## Docker Installation

### Pre-built Multi-Arch Images

Pull the appropriate image for your platform:

```bash
# Automatic platform detection
docker pull endomorphosis/ipfs-accelerate-py-cache:latest

# Explicitly specify platform
docker pull --platform linux/amd64 endomorphosis/ipfs-accelerate-py-cache:latest
docker pull --platform linux/arm64 endomorphosis/ipfs-accelerate-py-cache:latest
docker pull --platform linux/arm/v7 endomorphosis/ipfs-accelerate-py-cache:latest
```

### Run Container

```bash
# Basic usage
docker run -d \
  --name ipfs-cache \
  -v ~/.cache/ipfs_accelerate_py:/cache \
  endomorphosis/ipfs-accelerate-py-cache:latest

# With environment variables
docker run -d \
  --name ipfs-cache \
  -v ~/.cache/ipfs_accelerate_py:/cache \
  -e OPENAI_API_KEY=your-key \
  -e IPFS_ACCELERATE_CACHE_TTL=7200 \
  endomorphosis/ipfs-accelerate-py-cache:latest
```

### Docker Compose

```yaml
version: '3.8'
services:
  cache:
    image: endomorphosis/ipfs-accelerate-py-cache:latest
    volumes:
      - ~/.cache/ipfs_accelerate_py:/cache
    environment:
      - IPFS_ACCELERATE_CACHE_DIR=/cache
      - IPFS_ACCELERATE_CACHE_TTL=3600
    restart: unless-stopped
```

### Build Your Own Multi-Arch Image

```bash
# Enable BuildKit
export DOCKER_BUILDKIT=1

# Build for current platform
docker build -f install/Dockerfile.cache -t my-cache:latest .

# Build for multiple platforms (requires buildx)
docker buildx build \
  --platform linux/amd64,linux/arm64,linux/arm/v7 \
  -f install/Dockerfile.cache \
  -t my-cache:latest \
  --push .
```

## Platform-Specific Notes

### Linux

**Debian/Ubuntu:**
```bash
# Install dependencies
sudo apt-get update
sudo apt-get install -y python3 python3-venv curl git

# Run installer
curl -fsSL https://raw.githubusercontent.com/endomorphosis/ipfs_accelerate_py/main/install/install.sh | bash
```

**RHEL/CentOS/Fedora:**
```bash
# Install dependencies
sudo yum install -y python3 python3-venv curl git

# Run installer
curl -fsSL https://raw.githubusercontent.com/endomorphosis/ipfs_accelerate_py/main/install/install.sh | bash
```

**Arch Linux:**
```bash
# Install dependencies
sudo pacman -S python python-pip curl git

# Run installer
curl -fsSL https://raw.githubusercontent.com/endomorphosis/ipfs_accelerate_py/main/install/install.sh | bash
```

### macOS

**Homebrew required for CLI tools:**
```bash
# Install Homebrew if not present
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Run installer
curl -fsSL https://raw.githubusercontent.com/endomorphosis/ipfs_accelerate_py/main/install/install.sh | bash
```

**Apple Silicon (M1/M2/M3):**
The installer automatically detects Apple Silicon and installs ARM64 versions.

### Windows

**PowerShell 5.1+ required:**
```powershell
# Check PowerShell version
$PSVersionTable.PSVersion

# Run installer
iwr -useb https://raw.githubusercontent.com/endomorphosis/ipfs_accelerate_py/main/install/install.ps1 | iex
```

**Windows Subsystem for Linux (WSL):**
Use the Linux installer inside WSL:
```bash
curl -fsSL https://raw.githubusercontent.com/endomorphosis/ipfs_accelerate_py/main/install/install.sh | bash
```

## Environment Variables

The installer configures these environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `IPFS_ACCELERATE_CACHE_DIR` | `~/.cache/ipfs_accelerate_py` | Cache storage directory |
| `IPFS_ACCELERATE_CACHE_TTL` | `3600` | Default TTL in seconds |

Optional API keys (configure after installation):

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key |
| `ANTHROPIC_API_KEY` | Claude API key |
| `GOOGLE_API_KEY` | Gemini API key |
| `GROQ_API_KEY` | Groq API key |
| `HF_TOKEN` | HuggingFace token |
| `GITHUB_TOKEN` | GitHub token |

## CI/CD Integration

### GitHub Actions

```yaml
- name: Install Cache Infrastructure
  run: |
    curl -fsSL https://raw.githubusercontent.com/endomorphosis/ipfs_accelerate_py/main/install/install.sh | bash -s -- --profile minimal --silent
```

### GitLab CI

```yaml
install_cache:
  script:
    - curl -fsSL https://raw.githubusercontent.com/endomorphosis/ipfs_accelerate_py/main/install/install.sh | bash -s -- --profile minimal --silent
```

### Jenkins

```groovy
sh 'curl -fsSL https://raw.githubusercontent.com/endomorphosis/ipfs_accelerate_py/main/install/install.sh | bash -s -- --profile minimal --silent'
```

### CircleCI

```yaml
- run:
    name: Install Cache Infrastructure
    command: |
      curl -fsSL https://raw.githubusercontent.com/endomorphosis/ipfs_accelerate_py/main/install/install.sh | bash -s -- --profile minimal --silent
```

## Verification

After installation, verify everything is working:

```bash
# Using installer
./install.sh --verify

# Manual verification
python -c "from ipfs_accelerate_py.common.llm_cache import get_global_llm_cache; cache = get_global_llm_cache(); print('✓ Cache infrastructure installed and working')"
```

Expected output:
```
✓ Cache infrastructure installed and working
```

## Uninstallation

### Unix/Linux/macOS
```bash
curl -fsSL https://raw.githubusercontent.com/endomorphosis/ipfs_accelerate_py/main/install/uninstall.sh | bash
```

Or manually:
```bash
pip uninstall ipfs_accelerate_py
rm -rf ~/.cache/ipfs_accelerate_py
```

### Windows
```powershell
iwr -useb https://raw.githubusercontent.com/endomorphosis/ipfs_accelerate_py/main/install/uninstall.ps1 | iex
```

Or manually:
```powershell
pip uninstall ipfs_accelerate_py
Remove-Item -Recurse -Force $env:USERPROFILE\.cache\ipfs_accelerate_py
```

## Troubleshooting

### Permission Denied (Unix)
```bash
chmod +x install/install.sh
./install/install.sh
```

### Python Not Found
The installer looks for Python in this order:
1. `python3.12`, `python3.11`, `python3.10`, `python3.9`
2. `python3`
3. `python`
4. `py` (Windows only)

Install Python 3.8+ from:
- **Linux:** System package manager
- **macOS:** https://www.python.org/downloads/ or `brew install python`
- **Windows:** https://www.python.org/downloads/

### Network Issues

Use offline installation if dependencies are cached:
```bash
./install.sh --offline
```

### Platform Not Detected

Manually specify platform:
```bash
./install.sh --platform linux --arch x86_64
```

### Import Errors

Ensure virtual environment is activated:
```bash
source .venv/bin/activate  # Unix
.venv\Scripts\activate     # Windows
```

### Logs

Check installation logs:
```bash
cat ~/.cache/ipfs_accelerate_py/install.log
```

## Manual Installation

If automatic installation fails:

```bash
# Clone repository
git clone https://github.com/endomorphosis/ipfs_accelerate_py.git
cd ipfs_accelerate_py

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install dependencies
pip install -U pip setuptools wheel
pip install -e ".[cache]"

# Create cache directory
mkdir -p ~/.cache/ipfs_accelerate_py

# Set environment variables (add to ~/.bashrc or ~/.zshrc)
export IPFS_ACCELERATE_CACHE_DIR="$HOME/.cache/ipfs_accelerate_py"
export IPFS_ACCELERATE_CACHE_TTL=3600

# Verify
python -c "from ipfs_accelerate_py.common.llm_cache import get_global_llm_cache; print('OK')"
```

## Support

- **Documentation:** See `CLI_INTEGRATIONS.md` and `API_INTEGRATIONS_COMPLETE.md`
- **Issues:** https://github.com/endomorphosis/ipfs_accelerate_py/issues
- **Logs:** `~/.cache/ipfs_accelerate_py/install.log`

## What Gets Installed

The installer sets up:

1. **Python Packages:**
   - Core cache infrastructure (`base_cache`, `cid_index`)
   - Cache adapters (LLM, HuggingFace, Docker, etc.)
   - CLI integrations (GitHub, Copilot, VSCode, etc.)
   - API wrappers (OpenAI, Claude, Groq, etc.)

2. **CLI Tools (optional):**
   - GitHub CLI (`gh`)
   - HuggingFace CLI (`huggingface-cli`)
   - Vast AI CLI (`vastai`)

3. **Configuration:**
   - Cache directory: `~/.cache/ipfs_accelerate_py`
   - Environment variables in shell config
   - Virtual environment (optional)

4. **Documentation:**
   - Usage guides
   - API references
   - Examples

## Performance

Expected installation times (on modern hardware with good internet):

| Profile | Download | Install | Total |
|---------|----------|---------|-------|
| Minimal | ~30s | ~30s | ~1min |
| Standard | ~2min | ~2min | ~4min |
| Full | ~10min | ~5min | ~15min |
| CLI | ~1min | ~1min | ~2min |

## Security

The installers:
- Use HTTPS for all downloads
- Verify package signatures (pip)
- Create isolated virtual environments
- Run with user permissions (no sudo required)
- Log all operations
- Support offline installation

## License

AGPLv3+ - See LICENSE file for details.
