# IPFS Accelerate Python

> **Enterprise-grade hardware-accelerated machine learning inference with IPFS network-based distribution**

[![PyPI version](https://badge.fury.io/py/ipfs-accelerate-py.svg)](https://badge.fury.io/py/ipfs-accelerate-py)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Documentation](https://img.shields.io/badge/docs-comprehensive-brightgreen.svg)](docs/INDEX.md)
[![Tests](https://img.shields.io/badge/tests-passing-success.svg)](docs/development/testing.md)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [MCP++ Server](#-mcp-server)
- [Architecture](#️-architecture)
- [Supported Hardware](#-supported-hardware)
- [Supported Models](#-supported-models)
- [Documentation](#-documentation)
- [IPFS & Distributed Features](#-ipfs--distributed-features)
- [Performance & Optimization](#-performance--optimization)
- [Troubleshooting](#-troubleshooting)
- [Testing & Quality](#-testing--quality)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🚀 Overview

**IPFS Accelerate Python** combines cutting-edge hardware acceleration, distributed computing, and IPFS network integration to deliver **blazing-fast machine learning inference** across multiple platforms and devices - from data centers to browsers.

### ⚡ Key Highlights

- 🔥 **8+ Hardware Platforms** - CPU, CUDA, ROCm, OpenVINO, Apple MPS, WebNN, WebGPU, Qualcomm
- 🌐 **Distributed by Design** - IPFS content addressing, P2P inference, global caching
- 🤖 **300+ Models** - Full HuggingFace compatibility + custom architectures
- 🧠 **Canonical MCP++ Server** - Unified `ipfs_accelerate_py.mcp_server` runtime is now the default startup path
- 🌍 **Browser-Native** - WebNN & WebGPU for client-side acceleration
- 📊 **Production Ready** - Real-time monitoring, enterprise security, compliance validation
- ⚡ **High Performance** - Intelligent caching, batch processing, model optimization

---

## 📦 Installation

### Quick Start (5 minutes)

```bash
# 1. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2. Install IPFS Accelerate
pip install -U pip setuptools wheel
pip install ipfs-accelerate-py

# 3. Verify installation
python -c "from ipfs_accelerate_py import IPFSAccelerator; print('✅ Ready!')"
```

### NVIDIA CUDA (PyTorch)

By default, pip may install a CPU-only PyTorch wheel from PyPI (e.g. `torch==...+cpu`) because the CUDA-enabled wheels are published on PyTorch's wheel indexes.

If you have an NVIDIA GPU and want to ensure CUDA is available in PyTorch, install PyTorch from the CUDA wheel index:

```bash
python -m pip install -U pip
python -m pip install --upgrade --force-reinstall -r install/requirements_torch_cu124.txt

python -c "import torch; print('torch=', torch.__version__); print('cuda_available=', torch.cuda.is_available()); print('torch_cuda=', torch.version.cuda)"
```

If you're on an NVIDIA **GB10 / DGX Spark**-class system (CUDA capability **12.1**, CUDA **13.0**), stable builds may warn that your GPU is unsupported. In that case, use the CUDA 13.0 **nightly** wheels:

```bash
./scripts/install_torch_cuda_cu130_nightly.sh
```

If you're installing from source/editable mode, you can also run:

```bash
python -m pip install -e . --no-deps
python -m pip install --upgrade --force-reinstall -r install/requirements_torch_cu124.txt
python -m pip install -r requirements.txt
```

### Installation Profiles

Choose the profile that matches your needs:

| Profile | Use Case | Installation |
|---------|----------|--------------|
| **Core** | Basic inference | `pip install ipfs-accelerate-py` |
| **Full** | Models + API server | `pip install ipfs-accelerate-py[full]` |
| **MCP** | MCP server extras | `pip install ipfs-accelerate-py[mcp]` |
| **Dev** | Development setup | `pip install -e .` |

📚 **Detailed instructions**: [Installation Guide](docs/guides/getting-started/installation.md) | [Troubleshooting](docs/guides/troubleshooting/faq.md) | [Getting Started](docs/guides/getting-started/README.md)

---

## 🎯 Quick Start

### Python API

```python
from ipfs_accelerate_py import IPFSAccelerator

# Initialize with automatic hardware detection
accelerator = IPFSAccelerator()

# Load any HuggingFace model
model = accelerator.load_model("bert-base-uncased")

# Run inference (automatically optimized for your hardware)
result = model.inference("Hello, world!")
print(result)
```

### Command Line Interface

```bash
# Start the default MCP++ server for automation
ipfs-accelerate mcp start

# Run the canonical FastAPI MCP service directly
python -m ipfs_accelerate_py.mcp_server.fastapi_service

# Run the direct MCP server CLI with p2p/task options
python -m ipfs_accelerate_py.mcp.cli --host 0.0.0.0 --port 9000

# Run inference directly
ipfs-accelerate inference generate \
  --model bert-base-uncased \
  --input "Hello, world!"

# List available models and hardware
ipfs-accelerate models list
ipfs-accelerate hardware status

# Start GitHub Actions autoscaler
ipfs-accelerate github autoscaler
```

#### Remote libp2p task pickup (ipfs_datasets_py)

If you want a remote machine running the `ipfs_accelerate_py` MCP server to also **pick up libp2p task submissions** coming from `ipfs_datasets_py`, you can start the MCP server CLI with the built-in P2P task worker:

```bash
# Remote machine (runs MCP + worker + libp2p TaskQueue service)
python -m ipfs_accelerate_py.mcp.cli \
  --host 0.0.0.0 --port 9000 \
  --p2p-task-worker --p2p-service --p2p-listen-port 9710 \
  --p2p-queue ~/.cache/ipfs_datasets_py/task_queue.duckdb

# Optional (off-host clients): set the public IP that will be embedded in the announced multiaddr
export IPFS_DATASETS_PY_TASK_P2P_PUBLIC_IP="YOUR_PUBLIC_IP"
```

By default, the libp2p TaskQueue service writes an **announce file** into your XDG cache dir
and clients will try to use it automatically:

- Default announce file: `~/.cache/ipfs_accelerate_py/task_p2p_announce.json`
- Disable announce file (opt-out): `IPFS_ACCELERATE_PY_TASK_P2P_ANNOUNCE_FILE=0` (or `IPFS_DATASETS_PY_TASK_P2P_ANNOUNCE_FILE=0`)

If your client machine can read that announce file (same host/user, or a shared filesystem path you set via
`IPFS_ACCELERATE_PY_TASK_P2P_ANNOUNCE_FILE` / `IPFS_DATASETS_PY_TASK_P2P_ANNOUNCE_FILE`), you do **not** need to set any remote multiaddr env vars.

Otherwise, the process also prints a `multiaddr=...` line. On the client machine, set:

```bash
export IPFS_DATASETS_PY_TASK_P2P_REMOTE_MULTIADDR="/ip4/.../tcp/9710/p2p/..."
```

Notes:
- This mode requires `ipfs_datasets_py` to be installed on the remote machine (and `libp2p` installed via `ipfs_datasets_py[p2p]`).

### Real-World Examples

| Example | Description | Complexity |
|---------|-------------|------------|
| [Basic Usage](examples/basic_usage.py) | Simple inference with BERT | Beginner |
| [Hardware Selection](examples/hardware_selection.py) | Choose specific accelerator | Intermediate |
| [Distributed Inference](examples/p2p_inference.py) | P2P model sharing | Advanced |
| [Browser Integration](examples/webnn_demo.py) | WebNN/WebGPU in browsers | Advanced |

📖 **More examples**: [examples/](examples/) | [Quick Start Guide](docs/guides/QUICKSTART.md)

---

## 🧠 MCP++ Server

The MCP server in this repository has completed its unification cutover.

- **Canonical runtime**: `ipfs_accelerate_py/mcp_server`
- **Compatibility facade**: `ipfs_accelerate_py/mcp`
- **Current default**: `create_mcp_server()` and the main MCP startup paths now select the unified runtime by default
- **Cutover status**: approved and frozen with a focused release-candidate matrix of `120 passed`

### Current entrypoints

| Entry point | Best for | Notes |
|------------|----------|-------|
| `ipfs-accelerate mcp start` | End-user server startup | Main product CLI for MCP server management and dashboard workflows |
| `python -m ipfs_accelerate_py.mcp.cli` | Direct server/process control | Starts the MCP server and can also host TaskQueue/libp2p worker services |
| `python -m ipfs_accelerate_py.mcp_server.fastapi_service` | Standalone HTTP/FastAPI hosting | Reads `IPFS_MCP_*` env vars and mounts the MCP app at `/mcp` by default |
| `from ipfs_accelerate_py.mcp_server import create_server` | Programmatic embedding | Stable import target for the canonical runtime package |

### Supported MCP++ profile chapters

The unified runtime currently advertises these additive MCP++ profiles:

- `mcp++/profile-a-idl`
- `mcp++/profile-b-cid-artifacts`
- `mcp++/profile-c-ucan`
- `mcp++/profile-d-temporal-policy`
- `mcp++/profile-e-mcp-p2p`

### Unified control-plane features

- **Meta-tools**: `tools_list_categories`, `tools_list_tools`, `tools_get_schema`, `tools_dispatch`, `tools_runtime_metrics`
- **Migrated native categories**: `ipfs`, `workflow`, `p2p`
- **Security and governance**: UCAN validation, temporal/deontic policy evaluation, policy audit logging, secrets vault support, and risk scoring/frontier execution
- **Observability**: runtime metrics, audit-to-metrics bridging, OpenTelemetry hooks, and Prometheus exporter support
- **Transport coverage**: compatibility-tested process helpers, FastAPI mounting, and MCP+p2p handler parity with mixed-version negotiation hardening

### Cutover and rollback controls

These controls remain available for validation and operational rollback:

- `IPFS_MCP_FORCE_LEGACY_ROLLBACK=1` — force the compatibility facade to stay on the legacy wrapper
- `IPFS_MCP_UNIFIED_CUTOVER_DRY_RUN=1` — validate the unified startup path while keeping legacy runtime behavior active
- `IPFS_MCP_ENABLE_UNIFIED_BRIDGE=1` — explicitly request the unified bridge on compatibility-facade paths

### Recommended documentation

- [Canonical MCP server README](ipfs_accelerate_py/mcp_server/README.md)
- [MCP Cutover Checklist](MCP_CUTOVER_CHECKLIST.md)
- [MCP Server Unification Plan](MCP_SERVER_UNIFICATION_PLAN.md)
- [MCP++ Conformance Checklist](mcpplusplus/CONFORMANCE_CHECKLIST.md)
- [MCP++ Spec Gap Matrix](mcpplusplus/SPEC_GAP_MATRIX.md)

---

## 🏗️ Architecture

IPFS Accelerate Python is built on a **modular, enterprise-grade architecture**:

```
┌─────────────────────────────────────────────────────────┐
│                   Application Layer                     │
│  Python API • CLI • MCP Server • Web Dashboard          │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────┴────────────────────────────────────┐
│              Hardware Abstraction Layer                 │
│  Unified interface across 8+ hardware platforms         │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────┴────────────────────────────────────┐
│                Inference Backends                       │
│  CPU • CUDA • ROCm • MPS • OpenVINO • WebNN • WebGPU    │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────┴────────────────────────────────────┐
│              IPFS Network Layer                         │
│  Content addressing • P2P • Distributed caching         │
└─────────────────────────────────────────────────────────┘
```

### Core Components

- **Hardware Abstraction**: Unified API across 8+ platforms with automatic selection
- **IPFS Integration**: Content-addressed storage, P2P distribution, intelligent caching
- **Performance Modeling**: ML-powered optimization and resource management
- **MCP Server**: Canonical `ipfs_accelerate_py.mcp_server` MCP++ runtime with compatibility facade and cutover controls
- **Monitoring**: Real-time metrics, profiling, and analytics

📐 **Detailed architecture**: [docs/architecture/overview.md](docs/architecture/overview.md) | [CI/CD](docs/architecture/ci-cd.md)

---

## 🔧 Supported Hardware

Run anywhere - from powerful servers to edge devices and browsers:

| Platform | Status | Acceleration | Requirements | Performance |
|----------|--------|--------------|--------------|-------------|
| **CPU** (x86/ARM) | ✅ | SIMD, AVX | Any | Good |
| **NVIDIA CUDA** | ✅ | GPU + TensorRT | CUDA 11.8+ | Excellent |
| **AMD ROCm** | ✅ | GPU + HIP | ROCm 5.0+ | Excellent |
| **Apple MPS** | ✅ | Metal | M1/M2/M3 | Excellent |
| **Intel OpenVINO** | ✅ | CPU/GPU | Intel HW | Very Good |
| **WebNN** | ✅ | Browser NPU | Chrome, Edge | Good |
| **WebGPU** | ✅ | Browser GPU | Modern browsers | Very Good |
| **Qualcomm** | ✅ | Mobile DSP | Snapdragon | Good |

### Hardware Selection

The framework **automatically detects and selects** the best available hardware:

```python
# Automatic (recommended)
accelerator = IPFSAccelerator()  # Uses best available

# Manual selection
accelerator = IPFSAccelerator(device="cuda")  # Force CUDA
accelerator = IPFSAccelerator(device="mps")   # Force Apple MPS
```

⚙️ **Hardware guides**: [Hardware Optimization](docs/guides/hardware/overview.md) | [Platform Support](docs/guides/hardware/overview.md#platforms)

---

## 🤖 Supported Models

### Pre-trained Models (300+)

| Category | Models | Status |
|----------|--------|--------|
| **Text** | BERT, RoBERTa, DistilBERT, ALBERT, GPT-2/Neo/J, T5, BART, Pegasus, Sentence Transformers | ✅ |
| **Vision** | ViT, DeiT, BEiT, ResNet, EfficientNet, DETR, YOLO | ✅ |
| **Audio** | Whisper, Wav2Vec2, WavLM, Audio Transformers | ✅ |
| **Multimodal** | CLIP, BLIP, LLaVA | ✅ |
| **Custom** | PyTorch models, ONNX, TensorFlow (converted) | ✅ |

### Model Loading

```python
# From HuggingFace Hub
model = accelerator.load_model("bert-base-uncased")

# From IPFS (content-addressed)
model = accelerator.load_model("ipfs://QmXxxx...")

# Local model
model = accelerator.load_model("./my_model/")

# With specific hardware
model = accelerator.load_model("gpt2", device="cuda")
```

🤖 **Full model list**: [Supported Models](docs/README.md#model-support) | [Custom Models Guide](docs/archive/USAGE.md#custom-models)

---

## 📚 Documentation

### 📖 Essential Guides

| Guide | Description | Audience |
|-------|-------------|----------|
| [**Getting Started**](docs/guides/getting-started/README.md) | Complete beginner tutorial | Everyone |
| [**Quick Start**](docs/guides/QUICKSTART.md) | Get running in 5 minutes | Everyone |
| [**Installation**](docs/guides/getting-started/installation.md) | Detailed setup instructions | Users |
| [**FAQ**](docs/guides/troubleshooting/faq.md) | Common questions & answers | Everyone |
| [**API Reference**](docs/api/overview.md) | Complete API documentation | Developers |
| [**Architecture**](docs/architecture/overview.md) | System design & components | Architects |
| [**Hardware Optimization**](docs/guides/hardware/overview.md) | Platform-specific tuning | Engineers |
| [**Testing Guide**](docs/development/testing.md) | Testing & benchmarking | QA/DevOps |

### 🎯 Specialized Topics

| Topic | Resources |
|-------|-----------|
| **IPFS & P2P** | [IPFS Integration](docs/features/ipfs/IPFS.md) • [P2P Networking](docs/guides/p2p/) |
| **GitHub Actions** | [Autoscaler](docs/architecture/AUTOSCALER.md) • [CI/CD](docs/guides/github/) |
| **Docker & K8s** | [Container Guide](docs/guides/docker/) • [Deployment](docs/guides/deployment/) |
| **MCP Server** | [Canonical MCP Server README](ipfs_accelerate_py/mcp_server/README.md) • [MCP Setup](docs/guides/MCP_SETUP_GUIDE.md) • [Protocol Docs](docs/P2P_AND_MCP.md) • [Cutover Checklist](MCP_CUTOVER_CHECKLIST.md) |
| **Browser Support** | [WebNN/WebGPU](docs/features/webnn-webgpu/WEBNN_WEBGPU_README.md) • [Examples](examples/webnn_demo.py) |

### 📊 Documentation Quality

Our documentation has been **professionally audited** (January 2026):
- ✅ **200+ files** covering all features
- ✅ **93/100 quality score** (Excellent)
- ✅ **Comprehensive** - From beginner to expert
- ✅ **Well-organized** - Clear structure and navigation
- ✅ **Verified** - All examples tested and working

📋 **Documentation Hub**: [docs/](docs/) | [Full Index](docs/INDEX.md) | [Audit Report](docs/development_history/DOCUMENTATION_AUDIT_REPORT.md)

---

## 🌐 IPFS & Distributed Features

### Why IPFS?

IPFS integration provides **enterprise-grade distributed computing**:

- 🔐 **Content Addressing** - Cryptographically secure, immutable model distribution
- 🌍 **Global Network** - Automatic peer discovery and geographic optimization
- ⚡ **Intelligent Caching** - Multi-level LRU caching across the network
- 🔄 **Load Balancing** - Automatic distribution across available peers
- 🛡️ **Fault Tolerance** - Robust error handling and fallback mechanisms

### IPFS Backend Router (New! ⭐)

The IPFS Backend Router provides a flexible, pluggable backend system with automatic fallback:

**Backend Preference Order:**
1. **ipfs_kit_py** - Full distributed storage (preferred)
2. **HuggingFace Cache** - Local storage with IPFS addressing
3. **Kubo CLI** - Standard IPFS daemon

```python
from ipfs_accelerate_py import ipfs_backend_router

# Store model weights to IPFS
cid = ipfs_backend_router.add_path("/path/to/model", pin=True)
print(f"Model CID: {cid}")

# Retrieve from anywhere
ipfs_backend_router.get_to_path(cid, output_path="/cache/model")
```

**Configuration:**

```bash
# Prefer ipfs_kit_py (default)
export ENABLE_IPFS_KIT=true

# Use HF cache only (good for CI/CD)
export IPFS_BACKEND=hf_cache

# Force Kubo CLI
export IPFS_BACKEND=kubo
```

📚 **Full documentation**: [IPFS Backend Router Guide](docs/IPFS_BACKEND_ROUTER.md)

### Distributed Inference

```python
# Enable P2P inference
accelerator = IPFSAccelerator(enable_p2p=True)

# Model is automatically shared across peers
model = accelerator.load_model("bert-base-uncased")

# Inference uses best available peer
result = model.inference("Distributed AI!")
```

### Advanced Features

| Feature | Description | Status |
|---------|-------------|--------|
| **P2P Workflow Scheduler** | Distributed task execution with merkle clocks | ✅ |
| **GitHub Actions Cache** | Distributed cache for CI/CD | ✅ |
| **Autoscaler** | Dynamic runner provisioning | ✅ |
| **MCP Server** | Model Context Protocol (14+ tools) | ✅ |

🌐 **Learn more**: [IPFS Guide](docs/features/ipfs/IPFS.md) | [P2P Architecture](docs/guides/p2p/) | [Network Setup](docs/guides/p2p/)

---

## 🧪 Testing & Quality

```bash
# Run all tests
pytest

# Run specific test suite
pytest test/test_inference.py

# Run with coverage report
pytest --cov=ipfs_accelerate_py --cov-report=html

# Run benchmarks
python data/benchmarks/run_benchmarks.py
```

### Quality Metrics

| Metric | Status | Details |
|--------|--------|---------|
| **Test Coverage** | ✅ | Comprehensive test suite |
| **Documentation** | ✅ 93/100 | [Audit Report](docs/development_history/DOCUMENTATION_AUDIT_REPORT.md) |
| **Code Quality** | ✅ | Linted, type-checked |
| **Security** | ✅ | Regular vulnerability scans |
| **Performance** | ✅ | Benchmarked across platforms |

🧪 **Testing guide**: [docs/guides/testing/TESTING_README.md](docs/guides/testing/TESTING_README.md) | [CI/CD Setup](docs/guides/github/)

---

## ⚡ Performance & Optimization

### Benchmarks

| Hardware | Model | Throughput | Latency |
|----------|-------|------------|---------|
| **NVIDIA RTX 3090** | BERT-base | ~2000 samples/sec | <1ms |
| **Apple M2 Max** | BERT-base | ~800 samples/sec | 2-3ms |
| **Intel i9 (CPU)** | BERT-base | ~100 samples/sec | 10-15ms |
| **WebGPU (Browser)** | BERT-base | ~50 samples/sec | 20-30ms |

### Optimization Tips

```python
# Enable mixed precision for 2x speedup
accelerator = IPFSAccelerator(precision="fp16")

# Use batch processing for better throughput
results = model.batch_inference(inputs, batch_size=32)

# Enable model quantization for 4x memory reduction
model = accelerator.load_model("bert-base-uncased", quantize=True)

# Use intelligent caching for repeated queries
accelerator = IPFSAccelerator(enable_cache=True)
```

📊 **Performance guide**: [Hardware Optimization](docs/guides/hardware/overview.md) | [Benchmarking](docs/guides/testing/TESTING_README.md#benchmarks)

---

## 🔧 Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| **Import errors** | `pip install --upgrade ipfs-accelerate-py` |
| **CUDA not found** | Install [CUDA Toolkit 11.8+](https://developer.nvidia.com/cuda-downloads) |
| **Slow inference** | Check hardware selection, enable caching |
| **Memory errors** | Use quantization, reduce batch size |
| **Connection issues** | Check IPFS daemon, firewall settings |

### Quick Fixes

```bash
# Verify installation
python -c "import ipfs_accelerate_py; print(ipfs_accelerate_py.__version__)"

# Check hardware detection
ipfs-accelerate hardware status

# Test basic inference
ipfs-accelerate inference test

# View logs
ipfs-accelerate logs --tail 100
```

🆘 **Get help**: [Troubleshooting Guide](docs/guides/troubleshooting/INSTALLATION_TROUBLESHOOTING_GUIDE.md) | [FAQ](docs/guides/troubleshooting/faq.md) | [GitHub Issues](https://github.com/endomorphosis/ipfs_accelerate_py/issues)

---

## 🤝 Contributing

We **welcome contributions**! Here's how to get started:

### Quick Contribution Guide

1. **Fork & Clone**: Get your own copy of the repository
2. **Create Branch**: `git checkout -b feature/your-feature`
3. **Make Changes**: Follow our [coding standards](CONTRIBUTING.md)
4. **Run Tests**: `pytest` to ensure everything works
5. **Submit PR**: Open a pull request with clear description

### Areas We Need Help

- 🐛 **Bug Reports** - Found an issue? Let us know!
- 📚 **Documentation** - Help improve guides and examples
- 🧪 **Testing** - Add tests for edge cases
- 🌍 **Translations** - Translate docs to other languages
- 💡 **Features** - Suggest or implement new features

### Community & Guidelines

- 💬 **[GitHub Discussions](https://github.com/endomorphosis/ipfs_accelerate_py/discussions)** - Ask questions, share ideas
- 🐛 **[Issue Tracker](https://github.com/endomorphosis/ipfs_accelerate_py/issues)** - Report bugs, request features
- 🔐 **[Security Policy](SECURITY.md)** - Report security vulnerabilities
- 📧 **Email**: starworks5@gmail.com

📖 **Full guides**: [CONTRIBUTING.md](CONTRIBUTING.md) | [Code of Conduct](CONTRIBUTING.md#community-guidelines) | [Security Policy](SECURITY.md)

---

## 📄 License

This project is licensed under the **GNU Affero General Public License v3.0 or later (AGPLv3+)**.

**What this means**:
- ✅ Free to use, modify, and distribute
- ✅ Commercial use allowed
- ✅ Patent protection included
- ⚠️ Source code must be disclosed for network services
- ⚠️ Modifications must use same license

📋 **Details**: [LICENSE](LICENSE) | [AGPL FAQ](https://www.gnu.org/licenses/gpl-faq.html)

---

## 🙏 Acknowledgments

Built with amazing open source technologies:

- [**HuggingFace Transformers**](https://huggingface.co/transformers/) - ML model ecosystem
- [**IPFS**](https://ipfs.io/) - Distributed file system
- [**PyTorch**](https://pytorch.org/) - Deep learning framework
- [**FastAPI**](https://fastapi.tiangolo.com/) - Modern web framework

Special thanks to all [contributors](https://github.com/endomorphosis/ipfs_accelerate_py/graphs/contributors) who make this project possible! 🌟

### Project Information

- 📋 **[Changelog](CHANGELOG.md)** - Version history and release notes
- 🔐 **[Security Policy](SECURITY.md)** - Security reporting and best practices
- 🤝 **[Contributing Guide](CONTRIBUTING.md)** - How to contribute
- 📄 **[License](LICENSE)** - AGPLv3+ license details

---

## 🌟 Show Your Support

If you find this project useful:

- ⭐ **Star this repository** on GitHub
- 📢 **Share** with your network
- 🐛 **Report issues** to help improve it
- 💡 **Contribute** features or fixes
- 📝 **Write** about your experience

---

<div align="center">

**Made with ❤️ by [Benjamin Barber](https://github.com/endomorphosis) and [contributors](https://github.com/endomorphosis/ipfs_accelerate_py/graphs/contributors)**

[🏠 Homepage](https://github.com/endomorphosis/ipfs_accelerate_py) • 
[📚 Documentation](docs/) • 
[🐛 Issues](https://github.com/endomorphosis/ipfs_accelerate_py/issues) • 
[💬 Discussions](https://github.com/endomorphosis/ipfs_accelerate_py/discussions)

</div>

