# Zero-Touch Multi-Platform Installers

This directory contains automated, zero-touch installers for the ipfs_accelerate_py cache infrastructure across all platforms and architectures.

## Quick Start

### Linux (x86_64 / ARM64)
```bash
curl -fsSL https://raw.githubusercontent.com/endomorphosis/ipfs_accelerate_py/main/install/install.sh | bash
```

### macOS (Intel / Apple Silicon)
```bash
curl -fsSL https://raw.githubusercontent.com/endomorphosis/ipfs_accelerate_py/main/install/install.sh | bash
```

### Windows (x64 / ARM64)
```powershell
iwr -useb https://raw.githubusercontent.com/endomorphosis/ipfs_accelerate_py/main/install/install.ps1 | iex
```

## What Gets Installed

The installers automatically detect your platform and architecture, then install:

1. **Python dependencies** - All required packages for cache infrastructure
2. **CLI tools** (if needed):
   - GitHub CLI (`gh`)
   - GitHub Copilot CLI
   - VSCode CLI (`code`)
   - HuggingFace CLI (`huggingface-cli`)
   - Vast AI CLI (`vastai`)
3. **Cache infrastructure**:
   - Base cache with CID support
   - CID index for fast lookups
   - All cache adapters (LLM, HuggingFace Hub, Docker, etc.)
   - All CLI integrations with caching
   - All API integrations with caching
4. **Configuration**:
   - Cache directory setup (`~/.cache/ipfs_accelerate_py`)
   - Environment variables
   - Optional API keys setup

## Supported Platforms

| Platform | Architecture | Status |
|----------|-------------|--------|
| **Linux** | x86_64 | ✅ Tested |
| **Linux** | ARM64/aarch64 | ✅ Tested |
| **Linux** | ARMv7 | ✅ Supported |
| **macOS** | x86_64 (Intel) | ✅ Tested |
| **macOS** | ARM64 (Apple Silicon) | ✅ Tested |
| **Windows** | x64 | ✅ Tested |
| **Windows** | ARM64 | ✅ Supported |
| **FreeBSD** | x86_64 | ⚠️ Experimental |

## Installation Profiles

### Minimal (default)
- Core cache infrastructure only
- No ML dependencies
- ~50MB download

### Standard (recommended)
- Cache infrastructure
- CLI integrations
- API wrappers
- ~200MB download

### Full
- Everything from Standard
- ML model support (torch, transformers)
- Inference engine support
- ~2GB download

### CLI-only
- CLI integrations only
- No API wrappers
- ~100MB download

## Support

For issues with the installer, check logs in `~/.cache/ipfs_accelerate_py/install.log` or open an issue at https://github.com/endomorphosis/ipfs_accelerate_py/issues
