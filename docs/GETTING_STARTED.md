# 🚀 Getting Started with IPFS Accelerate Python

**Welcome!** This guide will have you running ML inference in **5 minutes**. Choose your path:

---

## 👤 Choose Your Path

### 🎯 I Want To...

| Goal | Time | Path |
|------|------|------|
| **Try it quickly** | 5 min | → [Quick Start](#quick-start-5-minutes) |
| **Learn by example** | 10 min | → [Hands-On Tutorial](#hands-on-tutorial) |
| **Deploy to production** | 30 min | → [Production Setup](#production-setup) |
| **Integrate with my app** | 15 min | → [Integration Guide](#integration-guide) |

---

## Quick Start (5 minutes)

### Step 1: Install

```bash
# Create virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install IPFS Accelerate
pip install ipfs-accelerate-py

# ✅ Verify installation (should print version)
python -c "import ipfs_accelerate_py; print(ipfs_accelerate_py.__version__)"
```

### Step 2: Run Your First Inference

Create `hello_world.py`:

```python
from ipfs_accelerate_py import IPFSAccelerator

# Initialize (detects your hardware automatically)
print("🚀 Initializing IPFS Accelerate...")
accelerator = IPFSAccelerator()

# Load a model (downloads if needed)
print("📥 Loading BERT model...")
model = accelerator.load_model("bert-base-uncased")

# Run inference
print("🤖 Running inference...")
text = "IPFS Accelerate makes ML inference easy and fast!"
result = model.inference(text)

print("✅ Success! Result:", result)
```

Run it:
```bash
python hello_world.py
```

**Expected output:**
```
🚀 Initializing IPFS Accelerate...
✅ Hardware detected: CUDA (NVIDIA GeForce RTX 3090)
📥 Loading BERT model...
✅ Model loaded successfully
🤖 Running inference...
✅ Success! Result: [embeddings array...]
```

### Step 3: Check Your Hardware

```bash
# See what hardware is available
ipfs-accelerate hardware status
```

**Congratulations! 🎉** You're now running hardware-accelerated ML inference!

---

## Hands-On Tutorial

### Part 1: Understanding the Basics (2 minutes)

The framework has **three main components**:

```python
# 1. Accelerator - Manages hardware and resources
accelerator = IPFSAccelerator()

# 2. Model - Loads and manages ML models
model = accelerator.load_model("bert-base-uncased")

# 3. Inference - Runs predictions
result = model.inference("Your text here")
```

### Part 2: Hardware Selection (2 minutes)

```python
# Automatic (recommended) - picks best available
acc = IPFSAccelerator()  

# Manual selection - force specific hardware
acc_cuda = IPFSAccelerator(device="cuda")    # NVIDIA GPU
acc_mps = IPFSAccelerator(device="mps")      # Apple Silicon
acc_cpu = IPFSAccelerator(device="cpu")      # CPU only

# Check what you're using
print(f"Using: {acc.device}")
```

### Part 3: Different Model Types (3 minutes)

```python
from ipfs_accelerate_py import IPFSAccelerator

accelerator = IPFSAccelerator()

# Text model
bert = accelerator.load_model("bert-base-uncased")
text_result = bert.inference("Hello world")

# Vision model  
vit = accelerator.load_model("google/vit-base-patch16-224")
image_result = vit.inference(image_path="photo.jpg")

# Audio model
whisper = accelerator.load_model("openai/whisper-base")
audio_result = whisper.inference(audio_path="speech.wav")
```

### Part 4: Optimization Tricks (3 minutes)

```python
# 1. Faster with mixed precision (2x speedup)
fast_acc = IPFSAccelerator(precision="fp16")

# 2. Use less memory with quantization (4x less RAM)
model = accelerator.load_model("bert-base", quantize=True)

# 3. Better throughput with batching
texts = ["text 1", "text 2", "text 3"]
results = model.batch_inference(texts, batch_size=32)

# 4. Faster repeated queries with caching
acc_cached = IPFSAccelerator(enable_cache=True)
```

**Next**: Try [examples/](../examples/) for more advanced scenarios!

---

## Production Setup

### Prerequisites

- ✅ Python 3.8+
- ✅ 4GB+ RAM
- ✅ (Optional) GPU with drivers installed
- ✅ (Optional) IPFS daemon for P2P features

### Step 1: Install with Full Features

```bash
# Virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Full installation
pip install ipfs-accelerate-py[full]

# For MCP server
pip install ipfs-accelerate-py[mcp]
```

### Step 2: Configuration

Create `config.yaml`:

```yaml
# Hardware settings
device: cuda  # or 'mps', 'cpu', 'auto'
precision: fp16  # or 'fp32', 'int8'

# Performance
enable_cache: true
batch_size: 32
max_workers: 4

# IPFS/P2P
enable_p2p: true
ipfs_gateway: "https://ipfs.io"

# Monitoring
enable_metrics: true
log_level: INFO
```

Load configuration:

```python
from ipfs_accelerate_py import IPFSAccelerator

# Load from file
accelerator = IPFSAccelerator.from_config("config.yaml")

# Or pass directly
accelerator = IPFSAccelerator(
    device="cuda",
    precision="fp16",
    enable_cache=True
)
```

### Step 3: Docker Deployment

Create `Dockerfile`:

```dockerfile
FROM python:3.10-slim

# Install dependencies
RUN pip install ipfs-accelerate-py[full]

# Copy your application
COPY app.py /app/
WORKDIR /app

# Run
CMD ["python", "app.py"]
```

Build and run:

```bash
docker build -t my-ml-service .
docker run -p 8000:8000 my-ml-service
```

For GPU support, see [Docker GPU Guide](guides/docker/DOCKER_CONTAINER_GUIDE.md).

### Step 4: Monitoring

```python
from ipfs_accelerate_py import IPFSAccelerator

# Enable monitoring
accelerator = IPFSAccelerator(enable_metrics=True)

# Get metrics
metrics = accelerator.get_metrics()
print(f"Total inferences: {metrics['total_inferences']}")
print(f"Average latency: {metrics['avg_latency_ms']}ms")
print(f"Cache hit rate: {metrics['cache_hit_rate']}%")
```

**Production Guides**:
- [Deployment Guide](guides/deployment/DEPLOYMENT_GUIDE.md)
- [Monitoring Setup](guides/infrastructure/README.md)
- [Security Best Practices](ARCHITECTURE.md#security)

---

## Integration Guide

### REST API Server

Create a FastAPI server:

```python
from fastapi import FastAPI
from ipfs_accelerate_py import IPFSAccelerator

app = FastAPI()
accelerator = IPFSAccelerator()
model = accelerator.load_model("bert-base-uncased")

@app.post("/inference")
async def run_inference(text: str):
    result = model.inference(text)
    return {"result": result}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

Run and test:

```bash
# Start server
python api_server.py

# Test (in another terminal)
curl -X POST "http://localhost:8000/inference" \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world"}'
```

### CLI Integration

Use in bash scripts:

```bash
#!/bin/bash

# Run batch inference
for file in *.txt; do
  echo "Processing $file..."
  ipfs-accelerate inference generate \
    --model bert-base-uncased \
    --input "$file" \
    --output "${file}.result"
done

echo "✅ All files processed!"
```

### Python Library

Integrate into your application:

```python
import ipfs_accelerate_py as ia

class MyMLService:
    def __init__(self):
        self.accelerator = ia.IPFSAccelerator()
        self.models = {
            'text': self.accelerator.load_model('bert-base'),
            'vision': self.accelerator.load_model('vit-base'),
        }
    
    def process_text(self, text):
        return self.models['text'].inference(text)
    
    def process_image(self, image):
        return self.models['vision'].inference(image)

# Use in your app
service = MyMLService()
result = service.process_text("Hello!")
```

### MCP Server (Automation)

Start the MCP server for automation tools:

```bash
# Start server
ipfs-accelerate mcp start --port 8080

# The server provides 14+ tools for:
# - Model management
# - Inference
# - Hardware monitoring
# - Cache management
```

---

## 🎓 Learning Resources

### Documentation

| Resource | Description | Time |
|----------|-------------|------|
| [API Reference](API.md) | Complete API docs | Reference |
| [Architecture](ARCHITECTURE.md) | System design | 15 min |
| [Hardware Guide](HARDWARE.md) | Platform optimization | 20 min |
| [IPFS Integration](IPFS.md) | Distributed features | 15 min |

### Examples

| Example | Description | Complexity |
|---------|-------------|------------|
| [basic_usage.py](../examples/basic_usage.py) | Simple inference | Beginner |
| [batch_processing.py](../examples/batch_processing.py) | Process multiple inputs | Beginner |
| [hardware_selection.py](../examples/hardware_selection.py) | Choose hardware | Intermediate |
| [custom_model.py](../examples/custom_model.py) | Use your own model | Intermediate |
| [p2p_inference.py](../examples/p2p_inference.py) | Distributed inference | Advanced |
| [production_deploy.py](../examples/production_deploy.py) | Full production setup | Advanced |

### Video Tutorials (Coming Soon)

- 🎥 Installation and Setup
- 🎥 Your First Inference
- 🎥 Hardware Optimization
- 🎥 Production Deployment

---

## 🆘 Need Help?

### Common Issues

| Problem | Solution |
|---------|----------|
| Installation fails | Try `pip install --upgrade pip setuptools wheel` first |
| Import error | Check virtual environment is activated |
| Slow inference | See [Performance Guide](#part-4-optimization-tricks-3-minutes) |
| CUDA not found | Install [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) |

### Get Support

- 📖 **Documentation**: [docs/](README.md)
- ❓ **FAQ**: [FAQ.md](FAQ.md)
- 🐛 **Issues**: [GitHub Issues](https://github.com/endomorphosis/ipfs_accelerate_py/issues)
- 💬 **Community**: [Discussions](https://github.com/endomorphosis/ipfs_accelerate_py/discussions)
- 📧 **Email**: starworks5@gmail.com

---

## 🎯 What's Next?

Choose your learning path:

1. **Explore Examples** → [examples/](../examples/)
2. **Read Architecture** → [ARCHITECTURE.md](ARCHITECTURE.md)
3. **Optimize Performance** → [HARDWARE.md](HARDWARE.md)
4. **Deploy to Production** → [DEPLOYMENT_GUIDE.md](guides/deployment/DEPLOYMENT_GUIDE.md)
5. **Join Community** → [Discussions](https://github.com/endomorphosis/ipfs_accelerate_py/discussions)

---

**Happy coding! 🚀**

*Last updated: January 2026*
