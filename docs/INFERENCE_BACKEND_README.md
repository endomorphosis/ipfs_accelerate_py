# Unified Inference Backend System

A comprehensive, production-ready system for managing machine learning inference across multiple backend types and protocols.

## Quick Start

```python
from ipfs_accelerate_py.unified_inference_service import start_unified_service

# Start all services
service = await start_unified_service()

# Access components
backend_manager = service.get_backend_manager()
hf_server = service.get_hf_server()
p2p_node = service.get_p2p_node()
```

## Features

### 🎯 **Unified Backend Management**
- Support for 7 backend types (GPU, API, CLI, P2P, WebSocket, MCP, Hybrid)
- Intelligent load balancing (round-robin, least-loaded, best-performance)
- Automatic health monitoring
- Comprehensive metrics tracking

### 🌐 **Multi-Protocol Support**
- **HTTP/REST**: OpenAI-compatible API endpoints
- **WebSocket**: Real-time streaming with topic subscriptions
- **libp2p**: Distributed P2P inference
- **MCP**: Tool integration for automation

### 📊 **Monitoring & Observability**
- Real-time backend status
- Performance metrics per backend
- Health checks with automatic failover
- Comprehensive status reporting

### ⚡ **High Performance**
- Streaming inference via WebSocket
- Parallel request processing
- Efficient load balancing
- Automatic failover

## Architecture

```
┌─────────────────────────────────────────┐
│    Unified Inference Service            │
│  - Configuration & Orchestration        │
└────────────────┬────────────────────────┘
                 │
    ┌────────────┼────────────┐
    │            │            │
┌───▼──────┐ ┌──▼──────┐ ┌──▼─────────┐
│ Backend  │ │ HF      │ │ P2P        │
│ Manager  │ │ Server  │ │ Network    │
└──────────┘ └─────────┘ └────────────┘
```

## Components

### Backend Manager
`inference_backend_manager.py` - Central coordinator for all backends

**Key Features:**
- Backend registration and discovery
- Health monitoring (configurable intervals)
- Load balancing strategies
- Metrics tracking (requests, latency, queue size)
- Status reporting

**Usage:**
```python
from ipfs_accelerate_py.inference_backend_manager import get_backend_manager

manager = get_backend_manager()

# Select backend for task
backend = manager.select_backend_for_task("text-generation", model="gpt2")

# Get comprehensive status
status = manager.get_backend_status_report()
```

### WebSocket Handler
`hf_model_server/websocket_handler.py` - Real-time bidirectional communication

**Key Features:**
- Multi-client connection management
- Topic-based subscriptions
- Streaming inference responses
- Real-time status updates

**Usage:**
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/my_client');

ws.send(JSON.stringify({
    type: 'inference',
    model: 'gpt2',
    inputs: 'Hello, world!',
    stream: true
}));
```

### libp2p Inference
`libp2p_inference.py` - Distributed P2P inference

**Key Features:**
- Peer discovery (mDNS + bootstrap)
- Capability-based routing
- Network-wide load balancing
- Automatic failover

**Usage:**
```python
from ipfs_accelerate_py.libp2p_inference import LibP2PInferenceNode

node = LibP2PInferenceNode()
await node.start()

# Register capabilities
node.register_capability(PeerCapability.TEXT_GENERATION)
node.register_model("gpt2")

# Submit distributed request
response = await node.submit_inference_request(
    task="text-generation",
    model="gpt2",
    inputs="Hello!"
)
```

### MCP Tools
`mcp/tools/backend_management.py` - Tool integration

**Available Tools:**
- `list_inference_backends()` - List all backends
- `get_backend_status()` - Get status report
- `select_backend_for_inference()` - Select best backend
- `route_inference_request()` - Route requests
- `get_supported_tasks()` - List supported tasks

## Installation

### Basic Installation
```bash
pip install -r requirements.txt
```

### With HF Server Support
```bash
pip install -r requirements-hf-server.txt
```

### With libp2p Support
```bash
pip install "libp2p @ git+https://github.com/libp2p/py-libp2p@main"
pip install pymultihash>=0.8.2
```

## Configuration

### Simple Configuration
```python
from ipfs_accelerate_py.unified_inference_service import InferenceServiceConfig

config = InferenceServiceConfig(
    enable_backend_manager=True,
    enable_hf_server=True,
    enable_websocket=True,
    enable_libp2p=False
)

service = await start_unified_service(config)
```

### Advanced Configuration
```python
config = InferenceServiceConfig(
    # Backend Manager
    enable_backend_manager=True,
    backend_health_checks=True,
    backend_health_check_interval=60,
    load_balancing_strategy="best_performance",
    
    # HF Server
    enable_hf_server=True,
    hf_server_host="0.0.0.0",
    hf_server_port=8000,
    
    # Protocols
    enable_websocket=True,
    enable_libp2p=True,
    
    # Backends
    enable_api_backends=True,
    api_backends=["hf_tgi", "ollama", "openai_api"],
    enable_cli_backends=True
)
```

## API Endpoints

### HTTP/REST

- `GET /health` - Health check
- `GET /status` - Server status
- `GET /v1/models` - List models
- `POST /v1/completions` - Text completions
- `POST /v1/chat/completions` - Chat completions
- `POST /v1/embeddings` - Embeddings
- `POST /models/load` - Load model
- `POST /models/unload` - Unload model

### WebSocket

- `ws://host:port/ws/{client_id}` - Client connection

**Message Types:**
- `subscribe` - Subscribe to topics
- `inference` - Submit inference request
- `status` - Get status
- `ping` / `pong` - Keepalive

## Examples

See `examples/multi_protocol_inference.py` for a complete working example demonstrating:
- HTTP/REST API usage
- WebSocket real-time communication
- MCP tool integration

## Documentation

- **[Architecture Guide](UNIFIED_INFERENCE_ARCHITECTURE.md)** - Complete architecture documentation
- **[Deployment Guide](DEPLOYMENT_GUIDE.md)** - Step-by-step deployment instructions
- **[Implementation Summary](IMPLEMENTATION_SUMMARY.md)** - Comprehensive overview

## Testing

Run integration tests:
```bash
python -m pytest test/test_unified_inference.py -v
```

**Results:** 13/15 tests passing (87% success rate)

## Performance

- **Backend Selection:** O(n) where n = backends for task
- **Health Checks:** Configurable interval (default 60s)
- **WebSocket:** Low-latency bidirectional streaming
- **Load Balancing:** 
  - Round-robin: O(1)
  - Least-loaded: O(n log n)
  - Best-performance: O(n log n)

## Monitoring

### Status Report
```python
manager = get_backend_manager()
status = manager.get_backend_status_report()

print(f"Total Backends: {status['total_backends']}")
print(f"Total Requests: {status['total_requests']}")
print(f"Success Rate: {status['total_successful']/status['total_requests']*100}%")
```

### Backend Metrics
```python
backend = manager.get_backend("backend_id")
print(f"Requests: {backend.metrics.total_requests}")
print(f"Avg Latency: {backend.metrics.average_latency_ms}ms")
print(f"Queue Size: {backend.metrics.current_queue_size}")
```

## Troubleshooting

### Backend Not Responding
1. Check backend status: `manager.get_backend_status_report()`
2. Verify health checks are enabled
3. Check backend-specific logs

### High Latency
1. Review load balancing strategy
2. Check queue sizes
3. Consider adding more backends

### WebSocket Disconnections
1. Implement reconnection logic
2. Use ping/pong for keepalive
3. Check network stability

## Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

## License

GNU Affero General Public License v3 or later (AGPLv3+)

## Support

- **Issues:** https://github.com/endomorphosis/ipfs_accelerate_py/issues
- **Documentation:** See docs/ directory
- **Examples:** See examples/ directory
