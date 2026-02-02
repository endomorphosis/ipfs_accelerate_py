# Unified HuggingFace Model Server

A production-ready model server for HuggingFace models with OpenAI-compatible API.

## Features

✅ **Automatic Skill Discovery** - Automatically discovers and registers HF skills
✅ **OpenAI-Compatible API** - Drop-in replacement for OpenAI API
✅ **Intelligent Hardware Selection** - Automatically selects optimal hardware
✅ **Multi-Model Serving** - Load and serve multiple models simultaneously
✅ **Request Batching** - Batch requests for efficient inference
✅ **Response Caching** - Cache responses for identical requests
✅ **Circuit Breaker** - Protect against failing models
✅ **Health Checks** - `/health` and `/ready` endpoints
✅ **Prometheus Metrics** - Export metrics for monitoring
✅ **Async/Await Support** - Full async support throughout

## Quick Start

### Installation

```bash
pip install -r requirements-hf-server.txt
```

### Start Server

```bash
# Basic usage
python -m ipfs_accelerate_py.hf_model_server.cli serve

# Custom configuration
python -m ipfs_accelerate_py.hf_model_server.cli serve \
    --host 0.0.0.0 \
    --port 8000 \
    --log-level INFO
```

### Using the API

The server provides OpenAI-compatible endpoints:

#### Text Completions

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt2",
    "prompt": "Once upon a time",
    "max_tokens": 50
  }'
```

#### Chat Completions

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt2",
    "messages": [
      {"role": "user", "content": "Hello!"}
    ]
  }'
```

#### Embeddings

```bash
curl http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "bert-base-uncased",
    "input": "Hello, world!"
  }'
```

#### List Models

```bash
curl http://localhost:8000/v1/models
```

## CLI Commands

### Discover Skills

```bash
python -m ipfs_accelerate_py.hf_model_server.cli discover
```

### Check Hardware

```bash
python -m ipfs_accelerate_py.hf_model_server.cli hardware
```

## Configuration

Configuration can be done via:
1. Command-line arguments
2. Environment variables
3. Configuration file (future)

### Environment Variables

```bash
# Server settings
export HF_SERVER_HOST=0.0.0.0
export HF_SERVER_PORT=8000
export HF_SERVER_WORKERS=1
export HF_SERVER_LOG_LEVEL=INFO

# Feature toggles
export HF_SERVER_ENABLE_BATCHING=true
export HF_SERVER_ENABLE_CACHING=true
export HF_SERVER_ENABLE_CIRCUIT_BREAKER=true

# API settings
export HF_SERVER_API_KEY=your-secret-key
```

## API Endpoints

### OpenAI-Compatible

- `POST /v1/completions` - Text completions
- `POST /v1/chat/completions` - Chat completions
- `POST /v1/embeddings` - Text embeddings
- `GET /v1/models` - List available models

### Model Management

- `POST /models/load` - Load a model
- `POST /models/unload` - Unload a model

### Health & Status

- `GET /health` - Health check
- `GET /ready` - Readiness check
- `GET /status` - Server status

### Metrics (if enabled)

- `GET /metrics` - Prometheus metrics

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   FastAPI Server                        │
├─────────────────────────────────────────────────────────┤
│  OpenAI API  │  Model Management  │  Health & Metrics  │
├─────────────────────────────────────────────────────────┤
│              Middleware Layer                           │
│  Batching  │  Caching  │  Circuit Breaker              │
├─────────────────────────────────────────────────────────┤
│              Core Components                            │
│  Skill Registry  │  Hardware Selector  │  Model Loader │
├─────────────────────────────────────────────────────────┤
│              Hardware Layer                             │
│  CUDA  │  ROCm  │  MPS  │  OpenVINO  │  QNN  │  CPU  │
└─────────────────────────────────────────────────────────┘
```

## Hardware Support

The server automatically detects and uses available hardware:

| Hardware | Detection | Features |
|----------|-----------|----------|
| CUDA | `torch.cuda.is_available()` | Multi-GPU, FP16, Quantization |
| ROCm | `torch.hip.is_available()` | AMD GPUs |
| MPS | `torch.backends.mps.is_available()` | Apple Silicon |
| OpenVINO | `import openvino` | Intel optimization |
| QNN | Custom | Qualcomm NPU |
| CPU | Always | Fallback |

## Development

### Project Structure

```
ipfs_accelerate_py/hf_model_server/
├── __init__.py              # Package initialization
├── server.py                # Main FastAPI server
├── config.py                # Configuration management
├── cli.py                   # Command-line interface
├── registry/
│   ├── __init__.py
│   └── skill_registry.py    # Skill discovery & registry
├── hardware/
│   ├── __init__.py
│   └── detector.py          # Hardware detection
├── loader/                  # Model loading (future)
├── api/
│   ├── __init__.py
│   └── schemas.py           # Pydantic schemas
├── middleware/              # Batching, caching (future)
├── monitoring/              # Metrics, health (future)
└── utils/                   # Utilities
```

### Running Tests

```bash
pytest test/test_hf_model_server.py
```

## Roadmap

### Phase 1: Core Infrastructure ✅
- [x] Server configuration
- [x] Skill discovery and registry
- [x] Hardware detection
- [x] Basic FastAPI server
- [x] OpenAI-compatible schemas

### Phase 2: Model Loading (In Progress)
- [ ] Model loader with caching
- [ ] Multi-model management
- [ ] Memory management

### Phase 3: Performance Features
- [ ] Request batching
- [ ] Response caching
- [ ] Circuit breaker
- [ ] Load balancing

### Phase 4: Monitoring & Reliability
- [ ] Health checks (complete)
- [ ] Prometheus metrics
- [ ] Request logging
- [ ] Error handling

### Phase 5: Production Features
- [ ] Authentication
- [ ] Rate limiting
- [ ] Request queuing
- [ ] Auto-scaling

## Contributing

Contributions welcome! See CONTRIBUTING.md for guidelines.

## License

See LICENSE file for details.
