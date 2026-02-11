# Cross-Platform Testing Guide - Linux & Windows

## Overview

This guide helps you test the cache functionality on both Linux and Windows laptops before deploying to Docker containers. This ensures compatibility across platforms and identifies platform-specific issues early.

**New:** The test now automatically tries alternative backends (Kubo, Storacha, S3) via `ipfs-kit-py` when `libp2p-py` is not available!

## Quick Start

### On Both Platforms

```bash
# 1. Run cross-platform test
python test_cross_platform_cache.py

# 2. Review results
# Look for "Platform Compatibility Report" section
# Check "ALTERNATIVE BACKENDS AVAILABLE" if P2P fails

# 3. If tests pass, proceed to Docker testing
# If libp2p fails, use alternative backends (see below)
```

## Alternative Backends (When libp2p Fails)

If `libp2p-py` doesn't work on your platform (common on Windows), the test automatically checks for alternative backends using `ipfs-kit-py`:

### Install Alternative Backend Support

```bash
# Install ipfs-kit-py for alternative backends
pip install ipfs-kit-py

# Re-run test to check available backends
python test_cross_platform_cache.py
```

### Supported Backends

1. **Kubo (IPFS)** - Local IPFS daemon
   ```bash
   # Install IPFS, then:
   ipfs daemon &
   ```

2. **Storacha** - Cloud storage (web3.storage)
   ```bash
   export WEB3_STORAGE_TOKEN=your_token
   ```

3. **S3** - AWS S3 or compatible
   ```bash
   export AWS_ACCESS_KEY_ID=your_key
   export AWS_SECRET_ACCESS_KEY=your_secret
   export S3_BUCKET=cache
   ```

**See [ALTERNATIVE_BACKENDS_GUIDE.md](./ALTERNATIVE_BACKENDS_GUIDE.md) for detailed setup.**

## Platform-Specific Setup

### Linux Laptop Setup

#### Prerequisites
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python 3.8+
sudo apt install python3 python3-pip python3-venv -y

# Install build tools (for P2P dependencies)
sudo apt install build-essential python3-dev -y

# Install Docker (if not already installed)
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
```

#### Setup Steps
```bash
# 1. Clone repository
cd ~/projects
git clone <your-repo-url>
cd <repo-name>

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip
./install_p2p_cache_deps.sh

# 4. Run cross-platform test
python test_cross_platform_cache.py
```

#### Expected Results (Linux)
```
Platform: Linux
Test Results: 11/11 passed
✅ Linux is fully compatible!

Platform-Specific:
- Native libp2p support
- Full P2P functionality
- Docker integration works
- File paths use /
```

#### Common Linux Issues

**Issue 1: Permission denied on Docker**
```bash
# Add user to docker group
sudo usermod -aG docker $USER
# Log out and back in
```

**Issue 2: libp2p build fails**
```bash
# Install build dependencies
sudo apt install build-essential python3-dev libssl-dev libffi-dev
pip install --upgrade pip setuptools wheel
pip install "libp2p @ git+https://github.com/libp2p/py-libp2p@main"
```

**Issue 3: Port already in use**
```bash
# Check what's using the port
sudo netstat -tulpn | grep 9000
# Kill process or use different port
```

### Windows Laptop Setup

#### Prerequisites

1. **Install Python 3.8+**
   - Download from [python.org](https://www.python.org/downloads/)
   - **Important:** Check "Add Python to PATH" during installation
   - Do NOT use Microsoft Store version

2. **Install Git for Windows**
   - Download from [git-scm.com](https://git-scm.com/download/win)
   - Use default settings

3. **Install Visual Studio Build Tools** (for compiling native extensions)
   - Download from [visualstudio.microsoft.com](https://visualstudio.microsoft.com/downloads/)
   - Install "Desktop development with C++"
   - OR install standalone: `winget install Microsoft.VisualStudio.2022.BuildTools`

4. **Install Docker Desktop**
   - Download from [docker.com](https://www.docker.com/products/docker-desktop)
   - Enable WSL 2 backend (recommended)

#### Setup Steps (PowerShell)
```powershell
# 1. Clone repository
cd C:\Users\<YourUsername>\Projects
git clone <your-repo-url>
cd <repo-name>

# 2. Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# 3. Upgrade pip
python -m pip install --upgrade pip

# 4. Install dependencies
python -m pip install "libp2p @ git+https://github.com/libp2p/py-libp2p@main" pymultihash>=0.8.2 py-multiformats-cid cryptography

# 5. Run cross-platform test
python test_cross_platform_cache.py
```

#### Expected Results (Windows)
```
Platform: Windows
Test Results: 11/11 passed (or 9/11 with warnings)
⚠️  Windows is mostly compatible with some warnings

Platform-Specific:
- libp2p may have limited support
- P2P functionality may be limited
- Docker Desktop required
- File paths use \
```

#### Common Windows Issues

**Issue 1: "python" not recognized**
```powershell
# Add Python to PATH manually
# 1. Find Python installation: where python
# 2. Add to PATH in System Environment Variables
# OR reinstall Python with "Add to PATH" checked
```

**Issue 2: Execution policy error**
```powershell
# Allow script execution
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Issue 3: libp2p installation fails**
```powershell
# Option A: Use WSL (recommended for full P2P support)
wsl --install
wsl
# Then follow Linux instructions inside WSL

# Option B: Skip P2P, test without it
# Set CACHE_ENABLE_P2P=false
$env:CACHE_ENABLE_P2P="false"
python test_cross_platform_cache.py
```

**Issue 4: Docker Desktop not starting**
```powershell
# Check WSL 2 is installed
wsl --list --verbose

# Update WSL
wsl --update

# Restart Docker Desktop
```

**Issue 5: Firewall blocking ports**
```powershell
# Allow Python through firewall
New-NetFirewallRule -DisplayName "Python" -Direction Inbound -Program "C:\...\python.exe" -Action Allow

# OR open specific port
New-NetFirewallRule -DisplayName "Cache P2P" -Direction Inbound -LocalPort 9000 -Protocol TCP -Action Allow
```

## Cross-Platform Test Scenarios

### Scenario 1: Basic Cache (No P2P)

**Test on both platforms:**

```python
# test_basic_cache.py
from ipfs_accelerate_py.github_cli.cache import configure_cache
import tempfile

# Create cache without P2P
cache = configure_cache(
    cache_dir=tempfile.mkdtemp(),
    enable_p2p=False,
    enable_persistence=False
)

# Test operations
cache.put("test", {"data": "value"}, ttl=300)
result = cache.get("test")
print(f"✅ Cache works: {result}")

cache.shutdown()
```

**Expected:** Works on both Linux and Windows

### Scenario 2: Cache with Persistence

**Test on both platforms:**

```python
# test_persistent_cache.py
from ipfs_accelerate_py.github_cli.cache import configure_cache
from pathlib import Path

# Create cache with persistence
cache_dir = Path.home() / ".cache" / "test_cache"
cache_dir.mkdir(parents=True, exist_ok=True)

cache = configure_cache(
    cache_dir=str(cache_dir),
    enable_p2p=False,
    enable_persistence=True
)

# Test operations
cache.put("persistent", {"data": "saved"}, ttl=300)
cache.shutdown()

# Reload
cache2 = configure_cache(
    cache_dir=str(cache_dir),
    enable_p2p=False,
    enable_persistence=True
)
result = cache2.get("persistent")
print(f"✅ Persistence works: {result}")

cache2.shutdown()
```

**Expected:** Works on both platforms

### Scenario 3: P2P Cache (Platform-Dependent)

**Test on Linux (should work):**
```bash
python test_cross_platform_cache.py
# Look for "P2P support available: True"
```

**Test on Windows (may not work):**
```powershell
python test_cross_platform_cache.py
# May show "P2P support available: False"
# This is OK - use cache without P2P on Windows
```

## Testing Between Two Laptops

Use the dedicated two-machine smoke tool:
- [tools/github_p2p_cache_smoke.py](tools/github_p2p_cache_smoke.py)
- Runbook: [GITHUB_P2P_CACHE_TWO_LAPTOP_RUNBOOK.md](GITHUB_P2P_CACHE_TWO_LAPTOP_RUNBOOK.md)

Quickstart (recommended):

1) On both laptops:

```bash
export GITHUB_REPOSITORY=owner/repo
export CACHE_P2P_SHARED_SECRET='replace-with-a-random-shared-secret'
```

2) Laptop B (reader):

```bash
export CACHE_LISTEN_PORT=9101
python tools/github_p2p_cache_smoke.py --read --target octocat/Hello-World --wait-seconds 120 --verbose
```

3) Laptop A (writer):

```bash
export CACHE_LISTEN_PORT=9100
python tools/github_p2p_cache_smoke.py --write --target octocat/Hello-World --verbose
```

Notes:
- If `connected_peers` stays 0 on the writer, inbound firewall/NAT is the usual culprit.
- For same-host multi-process testing, use `IPFS_ACCELERATE_P2P_CACHE_DIR` + unique `RUNNER_NAME` and set `IPFS_ACCELERATE_P2P_FORCE_LOCALHOST=1`.

## Validation Checklist

### On Linux Laptop
- [ ] Python 3.8+ installed
- [ ] Virtual environment created
- [ ] Dependencies installed (including libp2p)
- [ ] Cross-platform test passes (11/11)
- [ ] P2P cache initializes
- [ ] Can bind to network ports
- [ ] Docker installed and user in docker group

### On Windows Laptop
- [ ] Python 3.8+ installed (from python.org)
- [ ] Virtual environment created
- [ ] Dependencies installed (cryptography at minimum)
- [ ] Cross-platform test passes (9/11 or better)
- [ ] Basic cache works (without P2P is OK)
- [ ] File operations work
- [ ] Docker Desktop installed and running

### Both Platforms
- [ ] Can create/read/write files
- [ ] Can bind to network ports
- [ ] Environment variables work
- [ ] AnyIO works
- [ ] Path operations work

## Next Steps

### After Cross-Platform Testing Passes

1. **Test Docker Locally** (on each platform)
   ```bash
   # Linux
   docker run --network host -e CACHE_ENABLE_P2P=true your-image

   # Windows (Docker Desktop)
   docker run --network host -e CACHE_ENABLE_P2P=false your-image
   ```

2. **Test Docker to Host Communication**
   - Linux: Test with `--network host`
   - Windows: Test with bridge network (P2P may not work)

3. **Deploy to CI/CD**
   - Use findings from local tests
   - Configure based on platform constraints

## Troubleshooting

### Test Fails on Both Platforms

**Check:**
1. Python version >= 3.8
2. Virtual environment activated
3. Dependencies installed correctly
4. No firewall blocking

**Solution:**
```bash
# Reinstall everything
rm -rf venv  # On Linux
# OR
Remove-Item -Recurse -Force venv  # On Windows

# Start fresh
python -m venv venv
source venv/bin/activate  # Linux
# OR
.\venv\Scripts\Activate.ps1  # Windows

pip install --upgrade pip
pip install -r requirements.txt
```

### Test Passes on Linux, Fails on Windows

**Most Common Issue:** libp2p compatibility

**Solution:**
- Accept P2P won't work on Windows
- Test without P2P: `$env:CACHE_ENABLE_P2P="false"`
- Use WSL for full compatibility (recommended)

### Both Platforms Pass, Docker Fails

**Check:**
1. Docker daemon running
2. Network mode correct
3. Environment variables passed to container
4. Ports not blocked by firewall

**Solution:** See [DOCKER_CACHE_QUICK_START.md](./DOCKER_CACHE_QUICK_START.md)

## Summary

| Feature | Linux | Windows | WSL | Docker |
|---------|-------|---------|-----|--------|
| Basic Cache | ✅ | ✅ | ✅ | ✅ |
| File Operations | ✅ | ✅ | ✅ | ✅ |
| Network Ops | ✅ | ✅ | ✅ | ✅ |
| P2P Cache | ✅ | ⚠️ Limited | ✅ | ✅ |
| libp2p | ✅ | ❌/⚠️ | ✅ | ✅ |
| Docker Integration | ✅ | ✅ | ✅ | N/A |

**Recommendation:**
- **Linux:** Use natively, full features work
- **Windows:** Use WSL for development, or test without P2P
- **Production:** Deploy with Docker (Linux containers)

## Support

- **Run test:** `python test_cross_platform_cache.py`
- **Check results:** Review "Platform Compatibility Report"
- **Platform issues:** See platform-specific sections above
- **Docker issues:** See [DEPLOYMENT_GUIDE.md](./DEPLOYMENT_GUIDE.md)
