# IPFS Accelerate Python

A comprehensive framework for hardware-accelerated machine learning inference with IPFS network-based distribution.

[![PyPI version](https://badge.fury.io/py/ipfs-accelerate-py.svg)](https://badge.fury.io/py/ipfs-accelerate-py)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## 🚀 Overview

IPFS Accelerate Python is an enterprise-grade platform that combines hardware acceleration, distributed computing, and IPFS network integration to provide blazing-fast machine learning inference across multiple platforms and devices.

### Key Features

- **🔥 Multi-Platform Hardware Acceleration**: CPU, CUDA, ROCm, OpenVINO, Apple MPS, WebNN, WebGPU, Qualcomm
- **🌐 IPFS Network Integration**: Distributed model storage, peer-to-peer inference, and content addressing
- **🤖 300+ Supported Models**: Compatible with HuggingFace Transformers and custom model architectures
- **🌍 Browser Support**: Client-side acceleration using WebNN and WebGPU APIs
- **📊 Real-time Monitoring**: Performance analytics, hardware profiling, and optimization insights
- **🔒 Enterprise Security**: Zero-trust architecture with compliance validation
- **⚡ High Performance**: Optimized inference pipelines with intelligent caching

## 📦 Installation

### Quick Start (Core Install)

```bash
# Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install IPFS Accelerate
pip install -U pip setuptools wheel
pip install ipfs-accelerate-py
```

### Installation Profiles

Choose the installation profile that fits your needs:

```bash
# Minimal runtime (core features only)
pip install ipfs-accelerate-py[minimal]

# Full runtime (includes models, API server, etc.)
pip install ipfs-accelerate-py[full]

# MCP server extras
pip install ipfs-accelerate-py[mcp]

# Development install from source
git clone https://github.com/endomorphosis/ipfs_accelerate_py.git
cd ipfs_accelerate_py
pip install -e .
```

For detailed installation instructions, see [INSTALL.md](INSTALL.md).

## 🎯 Quick Start Guide

### Basic Usage

```python
from ipfs_accelerate_py import IPFSAccelerator

# Initialize the accelerator
accelerator = IPFSAccelerator()

# Load a model from HuggingFace or IPFS
model = accelerator.load_model("bert-base-uncased")

# Run inference
result = model.inference("Hello, world!")
print(result)
```

### CLI Usage

```bash
# Start the MCP server
ipfs-accelerate mcp start

# Run inference via CLI
ipfs-accelerate inference generate --model bert-base-uncased --input "Hello, world!"

# List available models
ipfs-accelerate models list

# Check hardware status
ipfs-accelerate hardware status

# Start the GitHub autoscaler for CI/CD
ipfs-accelerate github autoscaler
```

For more examples, see [QUICKSTART.md](QUICKSTART.md).

## 🏗️ Architecture

IPFS Accelerate Python is built on a modular, enterprise-grade architecture:

- **Hardware Abstraction Layer**: Unified API across 8+ hardware platforms
- **IPFS Integration**: Content-addressed model distribution and caching
- **Performance Modeling**: ML-powered optimization and resource management
- **MCP Server**: Model Context Protocol for standardized inference
- **Dashboard & Monitoring**: Real-time performance tracking and analytics

For detailed architecture documentation, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## 🔧 Supported Hardware

| Platform | Status | Acceleration | Notes |
|----------|--------|--------------|-------|
| CPU (x86/ARM) | ✅ | SIMD, AVX | All platforms |
| NVIDIA CUDA | ✅ | GPU + TensorRT | CUDA 11.8+ |
| AMD ROCm | ✅ | GPU + HIP | ROCm 5.0+ |
| Apple MPS | ✅ | Metal | M1/M2/M3 chips |
| Intel OpenVINO | ✅ | CPU/GPU | Intel hardware |
| WebNN | ✅ | Browser NPU | Chrome, Edge |
| WebGPU | ✅ | Browser GPU | Modern browsers |
| Qualcomm | ✅ | Mobile DSP | Edge devices |

## 🤖 Supported Models

### Text Models
- BERT, RoBERTa, DistilBERT, ALBERT
- GPT-2, GPT-Neo, GPT-J
- T5, BART, Pegasus
- Sentence Transformers

### Vision Models
- ViT, DeiT, BEiT
- ResNet, EfficientNet
- DETR, YOLO
- CLIP (multimodal)

### Audio Models
- Whisper
- Wav2Vec2, WavLM
- Audio Transformers

### Custom Models
- Support for custom PyTorch models
- ONNX model support
- TensorFlow model conversion

For the complete list, see [docs/README.md](docs/README.md#model-support).

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [QUICKSTART.md](QUICKSTART.md) | Quick start guide with examples |
| [INSTALL.md](INSTALL.md) | Detailed installation instructions |
| [docs/README.md](docs/README.md) | Complete documentation index |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | System architecture overview |
| [docs/API.md](docs/API.md) | API reference documentation |
| [docs/HARDWARE.md](docs/HARDWARE.md) | Hardware optimization guide |
| [docs/IPFS.md](docs/IPFS.md) | IPFS integration details |
| [docs/P2P_AND_MCP.md](docs/P2P_AND_MCP.md) | P2P workflow scheduling & MCP server |
| [docs/TESTING.md](docs/TESTING.md) | Testing framework guide |

### Specialized Guides
- [AUTOSCALER.md](AUTOSCALER.md) - GitHub Actions autoscaler
- [P2P_SETUP_GUIDE.md](P2P_SETUP_GUIDE.md) - Peer-to-peer cache setup
- [MCP_SETUP_GUIDE.md](MCP_SETUP_GUIDE.md) - MCP server configuration
- [DOCKER_USAGE.md](DOCKER_USAGE.md) - Docker deployment guide

## 🌐 IPFS Features

### Content-Addressed Storage
- Cryptographically secure model distribution
- Immutable model versioning
- Automatic deduplication

### Distributed Inference
- Peer-to-peer model sharing
- Network-wide caching
- Load balancing across peers

### Provider Discovery
- Automatic peer discovery
- Geographic optimization
- Fallback mechanisms

## 🚀 Advanced Features

### GitHub Integration
- **Autoscaler**: Automatic runner provisioning for CI/CD workflows
- **P2P Cache**: Distributed cache for GitHub Actions
- **MCP Integration**: Model Context Protocol support

### Performance Optimization
- **Intelligent Caching**: Multi-level LRU caching
- **Hardware Profiling**: Real-time performance monitoring
- **Model Optimization**: Quantization, pruning, distillation
- **Batch Processing**: Efficient batch inference

### Browser Integration
- **WebNN API**: Native neural network acceleration
- **WebGPU**: High-performance GPU compute
- **Cross-Browser**: Chrome, Firefox, Edge, Safari support

## 🧪 Testing

```bash
# Run all tests
pytest

# Run specific test suite
pytest test/test_inference.py

# Run with coverage
pytest --cov=ipfs_accelerate_py

# Run benchmarks
python benchmarks/run_benchmarks.py
```

For comprehensive testing guide, see [docs/TESTING.md](docs/TESTING.md).

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details on:
- Code of conduct
- Development setup
- Pull request process
- Coding standards

## 📄 License

This project is licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).  
See [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

Built with:
- [HuggingFace Transformers](https://huggingface.co/transformers/)
- [IPFS](https://ipfs.io/)
- [PyTorch](https://pytorch.org/)
- [FastAPI](https://fastapi.tiangolo.com/)

## 📧 Contact & Support

- **Issues**: [GitHub Issues](https://github.com/endomorphosis/ipfs_accelerate_py/issues)
- **Discussions**: [GitHub Discussions](https://github.com/endomorphosis/ipfs_accelerate_py/discussions)
- **Email**: starworks5@gmail.com

## 🌟 Star History

If you find this project useful, please consider giving it a star! ⭐

---

**Made with ❤️ by Benjamin Barber and contributors**
