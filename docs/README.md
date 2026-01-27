# IPFS Accelerate Python Documentation

Comprehensive documentation for the IPFS Accelerate Python framework - a complete solution for hardware-accelerated machine learning inference with IPFS network-based distribution.

## Quick Links

- **🚀 [Installation & Setup](INSTALLATION.md)** - Get started quickly with comprehensive installation guide
- **📖 [Usage Guide](USAGE.md)** - Learn how to use all framework features
- **🔧 [API Reference](API.md)** - Complete API documentation with examples
- **⚙️ [Hardware Optimization](HARDWARE.md)** - Maximize performance across different hardware
- **🌐 [IPFS Integration](IPFS.md)** - Leverage distributed inference and content addressing
- **🔗 [P2P & MCP Architecture](P2P_AND_MCP.md)** - P2P workflow scheduling and MCP server guide
- **🧪 [Testing Guide](TESTING.md)** - Comprehensive testing framework and best practices

## What is IPFS Accelerate Python?

IPFS Accelerate Python is a comprehensive framework that combines:

- **Hardware Acceleration**: Support for CPU, CUDA, ROCm, OpenVINO, Apple MPS, WebNN, and WebGPU
- **IPFS Integration**: Distributed model storage, caching, and peer-to-peer inference
- **Browser Support**: Client-side acceleration using WebNN and WebGPU
- **300+ Models**: Compatible with HuggingFace Transformers and custom models
- **Cross-Platform**: Works on Linux, macOS, and Windows

## Documentation Structure

### Getting Started
1. **[Installation & Setup](INSTALLATION.md)** - Complete installation guide with hardware setup
2. **[Usage Guide](USAGE.md)** - Basic to advanced usage patterns with examples
3. **[Examples](../examples/README.md)** - Practical examples and demos

### Technical Reference
4. **[API Reference](API.md)** - Complete API documentation with all methods and parameters
5. **[Architecture Overview](ARCHITECTURE.md)** - System design, components, and data flow
6. **[Testing Guide](TESTING.md)** - Testing framework, benchmarks, and quality assurance

### Specialization Guides
7. **[Hardware Optimization](HARDWARE.md)** - Platform-specific optimization strategies
8. **[IPFS Integration](IPFS.md)** - Distributed inference and content addressing
9. **[P2P & MCP Architecture](P2P_AND_MCP.md)** - P2P workflow scheduling and MCP server
10. **[WebNN/WebGPU Integration](../WEBNN_WEBGPU_README.md)** - Browser-based acceleration

### Organized Guides
11. **[GitHub Guides](guides/github/)** - GitHub Actions, autoscaling, authentication, P2P cache
12. **[Docker Guides](guides/docker/)** - Container deployment, caching, security
13. **[P2P Guides](guides/p2p/)** - Distributed computing, libp2p, workflow scheduling
14. **[Deployment Guides](guides/deployment/)** - Production deployment, cross-platform

## Key Features Covered

### 🔧 Hardware Acceleration
- **CPU Optimization**: x86/x64, ARM with SIMD acceleration
- **NVIDIA CUDA**: GPU acceleration with TensorRT optimization  
- **AMD ROCm**: AMD GPU support with HIP/ROCm
- **Intel OpenVINO**: CPU and Intel GPU optimization
- **Apple Silicon**: Metal Performance Shaders (MPS) for M1/M2/M3
- **WebNN/WebGPU**: Browser-based hardware acceleration
- **Qualcomm**: Mobile and edge device acceleration

### 🌐 IPFS Network Features
- **Content Addressing**: Cryptographically secure model storage
- **Distributed Inference**: Peer-to-peer model sharing and processing
- **Intelligent Caching**: Multi-level caching with LRU eviction
- **Provider Discovery**: Automatic network peer discovery and selection
- **Fault Tolerance**: Robust error handling and fallback mechanisms

### 🤖 Model Support
- **Text Models**: BERT, GPT, T5, RoBERTa, DistilBERT, ALBERT, etc.
- **Vision Models**: ViT, ResNet, EfficientNet, CLIP, DETR, etc.  
- **Audio Models**: Whisper, Wav2Vec2, WavLM, etc.
- **Multimodal**: CLIP, BLIP, LLaVA, etc.
- **Custom Models**: Support for custom model architectures

### 🌍 Browser Integration
- **Cross-Browser**: Chrome, Firefox, Edge, Safari support
- **WebNN API**: Native neural network acceleration
- **WebGPU**: High-performance GPU compute in browsers
- **Precision Control**: fp16, fp32, mixed precision support
- **Real-time Performance**: Optimized for interactive applications

## Getting Help

### Documentation Navigation
- Use the **Table of Contents** in each document for quick navigation
- Look for **🔗 Cross-references** between related sections  
- Check **💡 Tips and Examples** throughout the documentation
- Reference **⚠️ Troubleshooting** sections when needed

### Community Support
- **GitHub Issues**: [Report bugs and request features](https://github.com/endomorphosis/ipfs_accelerate_py/issues)
- **Discussions**: [Community questions and sharing](https://github.com/endomorphosis/ipfs_accelerate_py/discussions)
- **Examples**: Browse the [examples directory](../examples/) for inspiration

### Contributing
- **Contributing Guide**: See [CONTRIBUTING.md](../CONTRIBUTING.md)
- **Code of Conduct**: Review our community guidelines
- **Development Setup**: Follow the [Testing Guide](TESTING.md#development-setup)

## Documentation Updates

This documentation was comprehensively updated to reflect the current state of the IPFS Accelerate Python framework, including recent additions such as:

- **P2P Workflow Scheduler**: Distributed task execution with merkle clocks and fibonacci heaps
- **MCP Server**: Model Context Protocol server with 14+ tools
- **CLI Endpoint Adapters**: Direct integration with Claude, OpenAI, Gemini, VSCode CLIs
- **Enhanced Inference**: Multi-backend routing (local, distributed, API, CLI modes)
- **GitHub Integration**: P2P cache, autoscaler, workflow discovery

All examples, APIs, and features have been verified and updated for accuracy.

**Last Updated**: January 2026  
**Framework Version**: 0.0.45+  
**Documentation Coverage**: Comprehensive (Core + Recent Features)

---

Start with the [Installation Guide](INSTALLATION.md) to begin using IPFS Accelerate Python! 🚀

