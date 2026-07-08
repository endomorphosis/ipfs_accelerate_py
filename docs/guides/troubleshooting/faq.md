# Frequently Asked Questions (FAQ)

## Installation & Setup

### Q: What are the system requirements?

**A:** Minimal requirements:
- Python 3.8 or later
- 2GB RAM (4GB+ recommended)
- Any modern CPU (x86/x64/ARM)

For GPU acceleration, see [Hardware Requirements](HARDWARE.md#requirements).

### Q: How do I install without pip?

**A:** Clone and install from source:
```bash
git clone https://github.com/endomorphosis/ipfs_accelerate_py.git
cd ipfs_accelerate_py
pip install -e .
```

### Q: Can I use this offline?

**A:** Yes! Models can be:
- Pre-downloaded from HuggingFace
- Loaded from local files
- Cached from previous downloads

IPFS features require internet connectivity but fall back gracefully.

---

## Hardware & Performance

### Q: Which hardware should I use?

**A:** The framework **automatically selects** the best available hardware. Priority order:
1. CUDA (NVIDIA GPUs) - Best performance
2. ROCm (AMD GPUs) - Excellent performance
3. MPS (Apple Silicon) - Excellent for M1/M2/M3
4. OpenVINO (Intel) - Good CPU/GPU performance
5. CPU - Works everywhere

### Q: Why is inference slow?

**A:** Common causes and fixes:

| Issue | Solution |
|-------|----------|
| First run | Models are being downloaded - subsequent runs are faster |
| CPU only | Install GPU drivers for acceleration |
| Large models | Use quantization: `load_model("model", quantize=True)` |
| No caching | Enable caching: `IPFSAccelerator(enable_cache=True)` |

### Q: How much memory do I need?

**A:** Depends on model size:
- BERT-base: ~500MB
- GPT-2: ~1GB
- Large models: 4GB+

Use quantization to reduce memory by 50-75%.

---

## Models & Inference

### Q: What models are supported?

**A:** Over 300 models including:
- All HuggingFace Transformers models
- Custom PyTorch models
- ONNX models
- TensorFlow (with conversion)

See [Getting Started](../getting-started/README.md) for complete setup and usage guidance.

### Q: Can I use my own custom model?

**A:** Yes! Three ways:

```python
# 1. Load from local directory
model = accelerator.load_model("./my_model/")

# 2. Load PyTorch model directly
model = accelerator.load_pytorch_model(my_torch_model)

# 3. Convert and load
model = accelerator.from_tensorflow(tf_model)
```

See [Custom Models Guide](USAGE.md#custom-models) for details.

### Q: How do I improve inference speed?

**A:** Multiple optimization strategies:

```python
# 1. Use mixed precision (2x faster)
accelerator = IPFSAccelerator(precision="fp16")

# 2. Enable batching
results = model.batch_inference(inputs, batch_size=32)

# 3. Use quantization (4x faster, uses less memory)
model = accelerator.load_model("bert-base", quantize=True)

# 4. Enable caching for repeated queries
accelerator = IPFSAccelerator(enable_cache=True)
```

---

## IPFS & Networking

### Q: Do I need IPFS installed?

**A:** No! The framework includes everything needed. However:
- Optional: Install IPFS daemon for better P2P features
- Automatic fallback to HTTP if IPFS unavailable
- Local caching reduces network dependency

### Q: How does P2P inference work?

**A:** Models are:
1. Content-addressed (unique hash)
2. Automatically shared across network
3. Cached locally and on peers
4. Load-balanced across available nodes

Enable with: `IPFSAccelerator(enable_p2p=True)`

### Q: Is my data private?

**A:** Yes! Options:
- **Local mode**: No network communication
- **Private networks**: Run your own IPFS network
- **Encryption**: All network traffic can be encrypted

See [Security Guide](ARCHITECTURE.md#security) for details.

---

## Browser & WebNN/WebGPU

### Q: Which browsers support WebNN/WebGPU?

**A:** Current support:

| Browser | WebNN | WebGPU | Status |
|---------|-------|--------|--------|
| Chrome 113+ | ✅ | ✅ | Full support |
| Edge 113+ | ✅ | ✅ | Full support |
| Firefox | ⚠️ | ✅ | WebGPU only |
| Safari | ❌ | ✅ | WebGPU only |

### Q: How do I use in the browser?

**A:** Two approaches:

1. **Direct JavaScript** (see [examples/browser/](../examples/browser/))
2. **Python API** generates browser-compatible code

See [WebNN/WebGPU Guide](../../features/webnn-webgpu/WEBNN_WEBGPU_README.md) for complete tutorial.

---

## Development & Integration

### Q: How do I integrate with my application?

**A:** Multiple integration methods:

```python
# 1. Python API (most flexible)
from ipfs_accelerate_py import IPFSAccelerator
accelerator = IPFSAccelerator()

# 2. CLI (for scripting)
# ipfs-accelerate inference generate --model bert-base

# 3. MCP Server (for automation)
# ipfs-accelerate mcp start

# 4. REST API (coming soon)
```

### Q: Is there a REST API?

**A:** Coming soon! Meanwhile, use:
- FastAPI integration (see examples/)
- MCP Server for automation
- Direct Python API

### Q: Can I use with Docker?

**A:** Yes! See [Docker Guide](../docker/DOCKER_CONTAINER_GUIDE.md) for:
- Pre-built images
- Custom Dockerfile examples
- Kubernetes deployment
- GPU passthrough setup

---

## Troubleshooting

### Q: Import error: "No module named 'ipfs_accelerate_py'"

**A:** Fix:
```bash
# Upgrade installation
pip install --upgrade ipfs-accelerate-py

# Or install from source
pip install -e .

# Verify
python -c "import ipfs_accelerate_py; print('OK')"
```

### Q: CUDA not found despite having NVIDIA GPU

**A:** Install CUDA Toolkit:
```bash
# Check current CUDA version
nvidia-smi

# Install CUDA 11.8+ from:
# https://developer.nvidia.com/cuda-downloads

# Verify
python -c "import torch; print(torch.cuda.is_available())"
```

### Q: "Connection refused" errors

**A:** Check:
1. IPFS daemon status
2. Firewall settings
3. Network connectivity

Fallback to local mode:
```python
accelerator = IPFSAccelerator(offline_mode=True)
```

### Q: Out of memory errors

**A:** Solutions:
```python
# 1. Use quantization
model = accelerator.load_model("model", quantize=True)

# 2. Reduce batch size
results = model.batch_inference(inputs, batch_size=8)

# 3. Use smaller model variant
model = accelerator.load_model("distilbert-base")  # vs bert-base
```

---

## GitHub Actions & CI/CD

### Q: How do I use the autoscaler?

**A:** Start the autoscaler:
```bash
ipfs-accelerate github autoscaler --token YOUR_TOKEN
```

It automatically:
- Provisions runners when needed
- Scales down when idle
- Manages P2P cache distribution

See [Autoscaler Guide](../../architecture/AUTOSCALER.md) for full setup.

### Q: How does the GitHub Actions cache work?

**A:** The P2P cache:
1. Shares artifacts across runners
2. Reduces duplicate downloads
3. Automatically synchronizes
4. Falls back to GitHub cache

Setup: [GitHub Actions Guide](../github/GITHUB_CACHE_COMPREHENSIVE.md)

---

## Contributing & Support

### Q: How can I contribute?

**A:** Many ways to help:
- 🐛 Report bugs
- 📚 Improve documentation
- 🧪 Add tests
- 💡 Suggest features
- 🌍 Translate docs

See [Contributing Guide](../../../CONTRIBUTING.md) for details.

### Q: Where can I get help?

**A:** Multiple channels:
- 📖 [Documentation](../../README.md)
- 🐛 [GitHub Issues](https://github.com/endomorphosis/ipfs_accelerate_py/issues)
- 💬 [Discussions](https://github.com/endomorphosis/ipfs_accelerate_py/discussions)
- 📧 Email: starworks5@gmail.com

### Q: Is commercial use allowed?

**A:** Yes! Licensed under AGPLv3+:
- ✅ Commercial use allowed
- ✅ Modification allowed
- ✅ Distribution allowed
- ⚠️ Must disclose source for network services
- ⚠️ Must use same license

See [LICENSE](../LICENSE) for full terms.

---

## Advanced Topics

### Q: How do I optimize for production?

**A:** Production checklist:
1. ✅ Enable monitoring and logging
2. ✅ Set up caching infrastructure
3. ✅ Configure autoscaling
4. ✅ Implement health checks
5. ✅ Use containerization
6. ✅ Set up CI/CD pipeline

See [Production Deployment Guide](../deployment/DEPLOYMENT_GUIDE.md).

### Q: Can I run multiple models simultaneously?

**A:** Yes! Use multiple accelerator instances:

```python
# Option 1: Separate instances
acc1 = IPFSAccelerator()
acc2 = IPFSAccelerator()
model1 = acc1.load_model("bert-base")
model2 = acc2.load_model("gpt2")

# Option 2: Share accelerator (automatic resource management)
acc = IPFSAccelerator()
models = {
    "bert": acc.load_model("bert-base"),
    "gpt2": acc.load_model("gpt2"),
}
```

### Q: What's the performance overhead of IPFS?

**A:** Minimal with caching:
- First load: ~2-5 seconds (download)
- Cached loads: < 100ms (same as local)
- P2P lookup: ~100-500ms
- Fallback to HTTP if needed

Optimize with: `IPFSAccelerator(enable_cache=True)`

---

## Still Have Questions?

- 📖 **Check the docs**: [Complete Documentation](../../README.md)
- 💬 **Ask the community**: [GitHub Discussions](https://github.com/endomorphosis/ipfs_accelerate_py/discussions)
- 🐛 **Report an issue**: [Issue Tracker](https://github.com/endomorphosis/ipfs_accelerate_py/issues)
- 📧 **Email us**: starworks5@gmail.com

---

*Last updated: January 2026*
