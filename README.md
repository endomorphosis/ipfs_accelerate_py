# IPFS Accelerate Python

> **Enterprise-grade hardware-accelerated machine learning inference with IPFS network-based distribution**

[![PyPI version](https://badge.fury.io/py/ipfs-accelerate-py.svg)](https://badge.fury.io/py/ipfs-accelerate-py)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Documentation](https://img.shields.io/badge/docs-comprehensive-brightgreen.svg)](docs/README.md)
[![Tests](https://img.shields.io/badge/tests-passing-success.svg)](docs/TESTING.md)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
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

### Installation Profiles

Choose the profile that matches your needs:

| Profile | Use Case | Installation |
|---------|----------|--------------|
| **Core** | Basic inference | `pip install ipfs-accelerate-py` |
| **Full** | Models + API server | `pip install ipfs-accelerate-py[full]` |
| **MCP** | MCP server extras | `pip install ipfs-accelerate-py[mcp]` |
| **Dev** | Development setup | `pip install -e .` |

📚 **Detailed instructions**: [Installation Guide](docs/guides/INSTALL.md) | [Troubleshooting](docs/INSTALLATION_TROUBLESHOOTING_GUIDE.md) | [Getting Started](docs/GETTING_STARTED.md)

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
# Start the MCP server for automation
ipfs-accelerate mcp start

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

### Real-World Examples

| Example | Description | Complexity |
|---------|-------------|------------|
| [Basic Usage](examples/basic_usage.py) | Simple inference with BERT | Beginner |
| [Hardware Selection](examples/hardware_selection.py) | Choose specific accelerator | Intermediate |
| [Distributed Inference](examples/p2p_inference.py) | P2P model sharing | Advanced |
| [Browser Integration](examples/webnn_demo.py) | WebNN/WebGPU in browsers | Advanced |

📖 **More examples**: [examples/](examples/) | [Quick Start Guide](docs/guides/QUICKSTART.md)

---

## 🏗️ Architecture

IPFS Accelerate Python is built on a **modular, enterprise-grade architecture**:

```
┌─────────────────────────────────────────────────────────┐
│                   Application Layer                      │
│  Python API • CLI • MCP Server • Web Dashboard          │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────┴────────────────────────────────────┐
│              Hardware Abstraction Layer                  │
│  Unified interface across 8+ hardware platforms          │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────┴────────────────────────────────────┐
│                Inference Backends                        │
│  CPU • CUDA • ROCm • MPS • OpenVINO • WebNN • WebGPU    │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────┴────────────────────────────────────┐
│              IPFS Network Layer                          │
│  Content addressing • P2P • Distributed caching          │
└─────────────────────────────────────────────────────────┘
```

### Core Components

- **Hardware Abstraction**: Unified API across 8+ platforms with automatic selection
- **IPFS Integration**: Content-addressed storage, P2P distribution, intelligent caching
- **Performance Modeling**: ML-powered optimization and resource management
- **MCP Server**: Model Context Protocol for standardized automation
- **Monitoring**: Real-time metrics, profiling, and analytics

📐 **Detailed architecture**: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | [System Design](docs/ARCHITECTURE.md#system-design)

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

⚙️ **Hardware guides**: [Hardware Optimization](docs/HARDWARE.md) | [Platform-Specific](docs/HARDWARE.md#platform-guides)

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

🤖 **Full model list**: [Supported Models](docs/README.md#model-support) | [Custom Models Guide](docs/USAGE.md#custom-models)

---

## 📚 Documentation

### 📖 Essential Guides

| Guide | Description | Audience |
|-------|-------------|----------|
| [**Getting Started**](docs/GETTING_STARTED.md) | Complete beginner tutorial | Everyone |
| [**Quick Start**](docs/guides/QUICKSTART.md) | Get running in 5 minutes | Everyone |
| [**Installation**](docs/guides/INSTALL.md) | Detailed setup instructions | Users |
| [**FAQ**](docs/FAQ.md) | Common questions & answers | Everyone |
| [**API Reference**](docs/API.md) | Complete API documentation | Developers |
| [**Architecture**](docs/ARCHITECTURE.md) | System design & components | Architects |
| [**Hardware Optimization**](docs/HARDWARE.md) | Platform-specific tuning | Engineers |
| [**Testing Guide**](docs/TESTING.md) | Testing & benchmarking | QA/DevOps |

### 🎯 Specialized Topics

| Topic | Resources |
|-------|-----------|
| **IPFS & P2P** | [IPFS Integration](docs/IPFS.md) • [P2P Networking](docs/guides/p2p/) |
| **GitHub Actions** | [Autoscaler](docs/architecture/AUTOSCALER.md) • [CI/CD](docs/guides/github/) |
| **Docker & K8s** | [Container Guide](docs/guides/docker/) • [Deployment](docs/guides/deployment/) |
| **MCP Server** | [MCP Setup](docs/guides/MCP_SETUP_GUIDE.md) • [Protocol Docs](docs/P2P_AND_MCP.md) |
| **Browser Support** | [WebNN/WebGPU](docs/WEBNN_WEBGPU_README.md) • [Examples](examples/webnn_demo.py) |

### 📊 Documentation Quality

Our documentation has been **professionally audited** (January 2026):
- ✅ **200+ files** covering all features
- ✅ **93/100 quality score** (Excellent)
- ✅ **Comprehensive** - From beginner to expert
- ✅ **Well-organized** - Clear structure and navigation
- ✅ **Verified** - All examples tested and working

📋 **Documentation Hub**: [docs/](docs/) | [Full Index](docs/INDEX.md) | [Audit Report](docs/DOCUMENTATION_AUDIT_REPORT.md)

---

## 🌐 IPFS & Distributed Features

### Why IPFS?

IPFS integration provides **enterprise-grade distributed computing**:

- 🔐 **Content Addressing** - Cryptographically secure, immutable model distribution
- 🌍 **Global Network** - Automatic peer discovery and geographic optimization
- ⚡ **Intelligent Caching** - Multi-level LRU caching across the network
- 🔄 **Load Balancing** - Automatic distribution across available peers
- 🛡️ **Fault Tolerance** - Robust error handling and fallback mechanisms

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

🌐 **Learn more**: [IPFS Guide](docs/IPFS.md) | [P2P Architecture](docs/P2P_AND_MCP.md) | [Network Setup](docs/guides/p2p/)

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
| **Documentation** | ✅ 93/100 | [Audit Report](docs/DOCUMENTATION_AUDIT_REPORT.md) |
| **Code Quality** | ✅ | Linted, type-checked |
| **Security** | ✅ | Regular vulnerability scans |
| **Performance** | ✅ | Benchmarked across platforms |

🧪 **Testing guide**: [docs/TESTING.md](docs/TESTING.md) | [CI/CD Setup](docs/guides/github/)

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

📊 **Performance guide**: [Hardware Optimization](docs/HARDWARE.md) | [Benchmarking](docs/TESTING.md#benchmarks)

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

🆘 **Get help**: [Troubleshooting Guide](docs/INSTALLATION_TROUBLESHOOTING_GUIDE.md) | [FAQ](docs/FAQ.md) | [GitHub Issues](https://github.com/endomorphosis/ipfs_accelerate_py/issues)

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
