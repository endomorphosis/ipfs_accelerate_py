# Installer Validation Guide

## Overview

This document validates that all CLI/SDK/API installers install the correct dependencies and that wrappers align with the actual tools they wrap.

## Installation Requirements by Tool

### 1. GitHub CLI (`gh`)

**Installation Method:**
- Linux (Debian/Ubuntu): `apt-get install gh`
- Linux (RHEL/CentOS): `yum install gh`
- macOS: `brew install gh`
- Windows: Download from github.com/cli/cli

**Verification:**
```bash
command -v gh
gh --version
```

**Wrapper Location:** `ipfs_accelerate_py/cli_integrations/github_cli_integration.py`
**Backend Location:** `ipfs_accelerate_py/github_cli/wrapper.py`
**Alignment Status:** ✅ Both exist, integration wraps CLI

### 2. GitHub Copilot CLI

**Installation Method:**
```bash
npm install -g @githubnext/github-copilot-cli
```

**Verification:**
```bash
command -v github-copilot-cli
github-copilot-cli --version
```

**Wrapper Location:** `ipfs_accelerate_py/cli_integrations/copilot_cli_integration.py`
**Backend Location:** `ipfs_accelerate_py/copilot_cli/wrapper.py`
**Alignment Status:** ✅ Both exist, integration wraps CLI

### 3. VSCode CLI (`code`)

**Installation Method:**
- Comes with VSCode installation
- macOS: Shell Command: Install 'code' command in PATH
- Linux: Automatically added during VSCode installation
- Windows: Added to PATH during installation

**Verification:**
```bash
command -v code
code --version
```

**Wrapper Location:** `ipfs_accelerate_py/cli_integrations/vscode_cli_integration.py`
**Alignment Status:** ⚠️ VSCode must be installed separately

### 4. OpenAI Codex CLI

**Installation Method:**
```bash
npm install -g @openai/codex
```

**Verification:**
```bash
npm list -g | grep @openai/codex
```

**Wrapper Location:** `ipfs_accelerate_py/cli_integrations/openai_codex_cli_integration.py`
**Alignment Status:** ⚠️ May not be publicly available

### 5. Claude Code CLI

**Installation Method:**
```bash
pip install anthropic
```

**Verification:**
```bash
python -c "import anthropic; print(anthropic.__version__)"
```

**Wrapper Location:** `ipfs_accelerate_py/cli_integrations/claude_code_cli_integration.py`
**Backend Location:** `ipfs_accelerate_py/api_backends/claude.py`
**Alignment Status:** ✅ Python SDK wraps API backend

### 6. Gemini CLI

**Installation Method:**
```bash
pip install google-generativeai
```

**Verification:**
```bash
python -c "import google.generativeai as genai; print(genai.__version__)"
```

**Wrapper Location:** `ipfs_accelerate_py/cli_integrations/gemini_cli_integration.py`
**Backend Location:** `ipfs_accelerate_py/api_backends/gemini.py`
**Alignment Status:** ✅ Python SDK wraps API backend

### 7. HuggingFace CLI

**Installation Method:**
```bash
pip install huggingface-hub[cli]
```

**Verification:**
```bash
command -v huggingface-cli
huggingface-cli --version
```

**Wrapper Location:** `ipfs_accelerate_py/cli_integrations/huggingface_cli_integration.py`
**Alignment Status:** ✅ Wraps official CLI

### 8. Vast AI CLI

**Installation Method:**
```bash
pip install vastai
```

**Verification:**
```bash
command -v vastai
vastai --version
```

**Wrapper Location:** `ipfs_accelerate_py/cli_integrations/vastai_cli_integration.py`
**Alignment Status:** ✅ Wraps official CLI

### 9. Groq CLI/SDK

**Installation Method:**
```bash
pip install groq
```

**Verification:**
```bash
python -c "import groq; print(groq.__version__)"
```

**Wrapper Location:** `ipfs_accelerate_py/cli_integrations/groq_cli_integration.py`
**Backend Location:** `ipfs_accelerate_py/api_backends/groq.py`
**Alignment Status:** ✅ Python SDK wraps API backend

## API Integrations

### LLM APIs

**API Integrations Module:** `ipfs_accelerate_py/api_integrations/__init__.py`

1. **OpenAI API**
   - Backend: `ipfs_accelerate_py/api_backends/openai_api.py` ✅
   - Integration: `CachedLLMAPI` wrapper
   - Installation: `pip install openai`
   
2. **Claude API**
   - Backend: `ipfs_accelerate_py/api_backends/claude.py` ✅
   - Integration: `CachedLLMAPI` wrapper
   - Installation: `pip install anthropic`
   
3. **Gemini API**
   - Backend: `ipfs_accelerate_py/api_backends/gemini.py` ✅
   - Integration: `CachedLLMAPI` wrapper
   - Installation: `pip install google-generativeai`
   
4. **Groq API**
   - Backend: `ipfs_accelerate_py/api_backends/groq.py` ✅
   - Integration: `CachedLLMAPI` wrapper
   - Installation: `pip install groq`
   
5. **Ollama API**
   - Backend: `ipfs_accelerate_py/api_backends/ollama.py` ✅
   - Integration: `CachedLLMAPI` wrapper
   - Installation: `pip install ollama`

### Inference Engines

**API Integrations Module:** `ipfs_accelerate_py/api_integrations/inference_engines.py`

1. **vLLM**
   - Backend: `ipfs_accelerate_py/api_backends/vllm.py` ✅
   - Integration: `CachedvLLMAPI`
   - Installation: `pip install vllm`
   
2. **HuggingFace TGI**
   - Backend: `ipfs_accelerate_py/api_backends/hf_tgi.py` ✅
   - Integration: `CachedHFTGIAPI`
   - Installation: Docker/server-based
   
3. **HuggingFace TEI**
   - Backend: `ipfs_accelerate_py/api_backends/hf_tei.py` ✅
   - Integration: `CachedHFTEIAPI`
   - Installation: Docker/server-based
   
4. **OpenVINO Model Server (OVMS)**
   - Backend: `ipfs_accelerate_py/api_backends/ovms.py` ✅
   - Integration: `CachedOVMSAPI`
   - Installation: Docker/server-based
   
5. **OPEA**
   - Backend: `ipfs_accelerate_py/api_backends/opea.py` ✅
   - Integration: `CachedOPEAAPI`
   - Installation: Docker/server-based

### Storage APIs

**API Integrations Module:** `ipfs_accelerate_py/api_integrations/storage.py`

1. **S3/Object Storage**
   - Backend: `ipfs_accelerate_py/api_backends/s3_kit.py` ✅
   - Integration: `CachedS3API`
   - Installation: `pip install boto3`
   
2. **IPFS API**
   - Backend: Various IPFS modules
   - Integration: `CachedIPFSAPI`
   - Installation: `pip install ipfshttpclient`

## Current Issues Found

### Installer Issues

1. ❌ **OpenAI Codex CLI** - `@openai/codex` npm package may not be publicly available
   - **Fix:** Gracefully handle installation failure with warning
   
2. ⚠️ **VSCode CLI** - Requires VSCode to be installed separately
   - **Fix:** Add clear warning message
   
3. ⚠️ **Copilot CLI** - Package name may have changed
   - **Current:** `@githubnext/github-copilot-cli`
   - **Check:** May now be `@github/copilot-cli` or integrated into `gh`
   - **Fix:** Update installer to try both

### Wrapper Alignment Issues

1. ⚠️ **CLI Integration Commands** - Need to verify commands match actual CLI tools
   - GitHub CLI integration uses `gh` commands ✅
   - Groq CLI integration assumes `groq` command (SDK only, no CLI) ❌
   - Claude Code CLI assumes `claude` command (SDK only, no official CLI) ❌
   - Gemini CLI assumes `gemini` command (SDK only, no official CLI) ❌

### API Integration Issues

1. ⚠️ **API Backend Alignment** - API integrations need to import from actual backends
   - Current: Generic wrappers in `api_integrations/`
   - Needed: Import and wrap actual backend classes
   
## Recommended Fixes

### 1. Update Installer Script

```bash
# installers/install.sh

# OpenAI Codex CLI - gracefully handle unavailability
if command -v npm >/dev/null 2>&1; then
    if ! npm list -g 2>/dev/null | grep -q "@openai/codex"; then
        log INFO "Attempting to install OpenAI Codex CLI..."
        npm install -g @openai/codex 2>/dev/null || \
        npm install -g openai-codex 2>/dev/null || \
        log WARN "OpenAI Codex CLI not publicly available - using Python SDK wrapper"
    fi
fi

# GitHub Copilot CLI - try multiple package names
if command -v npm >/dev/null 2>&1; then
    if ! command -v github-copilot-cli >/dev/null 2>&1; then
        log INFO "Installing GitHub Copilot CLI..."
        npm install -g @githubnext/github-copilot-cli 2>/dev/null || \
        npm install -g @github/copilot-cli 2>/dev/null || \
        log WARN "GitHub Copilot CLI installation failed - check if it's integrated into gh CLI"
    fi
fi
```

### 2. Fix CLI Integrations for SDK-Only Tools

For tools that are SDK-only (no actual CLI):

- **Groq**: Update integration to use Python SDK directly, not a CLI command
- **Claude**: Update integration to use Anthropic SDK directly
- **Gemini**: Update integration to use Google GenAI SDK directly

### 3. Align API Integrations with Backends

Update `api_integrations/__init__.py` to properly import and wrap actual backend classes:

```python
from ..api_backends.openai_api import openai_api
from ..api_backends.claude import claude
from ..api_backends.gemini import gemini
# etc.
```

## Testing Checklist

- [ ] GitHub CLI actually installs and `gh` command works
- [ ] HuggingFace CLI actually installs and `huggingface-cli` command works
- [ ] Vast AI CLI actually installs and `vastai` command works
- [ ] Python SDKs (anthropic, google-generativeai, groq, openai) install correctly
- [ ] npm packages (Copilot CLI) install correctly
- [ ] VSCode CLI check works correctly
- [ ] API integrations import from correct backend modules
- [ ] CLI integrations work with actual commands (not non-existent CLIs)
- [ ] Cache wrappers properly intercept and cache API calls
- [ ] All verification tests pass after installation

## Next Steps

1. Update installer script with better error handling
2. Fix CLI integrations for SDK-only tools
3. Align API integrations with actual backends
4. Create comprehensive validation tests
5. Update documentation with accurate information
