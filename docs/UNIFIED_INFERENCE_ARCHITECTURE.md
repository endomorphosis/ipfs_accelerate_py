# Unified Inference Backend Architecture

## Overview

The Unified Inference Backend system provides a comprehensive framework for managing, routing, and load balancing inference requests across multiple backend types, protocols, and deployment scenarios.

## Architecture Components

### 1. Inference Backend Manager (`inference_backend_manager.py`)

The central coordinator for all inference backends.

**Key Features:**
- Backend registration and discovery
- Health monitoring and status reporting
- Intelligent request routing
- Multiple load balancing strategies (round-robin, least-loaded, best-performance)
- Comprehensive metrics tracking

**Backend Types Supported:**
- `GPU`: Local GPU-accelerated inference (CUDA, ROCm, MPS, etc.)
- `API`: Remote API endpoints (OpenAI, Anthropic, HuggingFace, etc.)
- `CLI`: CLI tool integrations (Claude CLI, OpenAI CLI, etc.)
- `P2P`: libp2p distributed inference
- `WEBSOCKET`: WebSocket-enabled real-time backends
- `MCP`: Model Context Protocol backends
- `HYBRID`: Multi-protocol support

**Usage Example:**
```python
from ipfs_accelerate_py.inference_backend_manager import (
    get_backend_manager, BackendType, BackendCapabilities
)

# Get the global manager
manager = get_backend_manager()

# Register a backend
manager.register_backend(
    backend_id="my_gpu_backend",
    backend_type=BackendType.GPU,
    name="Local CUDA Backend",
    instance=my_backend_instance,
    capabilities=BackendCapabilities(
        supported_tasks={"text-generation", "text-embedding"},
        supports_streaming=True,
        supports_batching=True,
        hardware_types={"cuda"},
        protocols={"http", "websocket"}
    ),
    endpoint="http://localhost:8000"
)

# Select best backend for a task
backend = manager.select_backend_for_task(
    task="text-generation",
    model="gpt2",
    preferred_types=[BackendType.GPU, BackendType.API]
)

# Get comprehensive status
status = manager.get_backend_status_report()
```

### 2. WebSocket Handler (`hf_model_server/websocket_handler.py`)

Provides real-time bidirectional communication for inference and monitoring.

**Features:**
- Connection management for multiple clients
- Topic-based subscription system
- Streaming inference responses
- Real-time status updates
- Queue monitoring

**Supported Message Types:**
- `subscribe` / `unsubscribe`: Topic subscriptions
- `inference`: Submit inference request
- `status`: Get backend status
- `ping` / `pong`: Connection keepalive

**WebSocket Protocol:**
```javascript
// Connect to WebSocket
const ws = new WebSocket('ws://localhost:8000/ws/my_client_id');

// Subscribe to topics
ws.send(JSON.stringify({
    type: 'subscribe',
    topics: ['inference', 'status', 'backend']
}));

// Submit inference request
ws.send(JSON.stringify({
    type: 'inference',
    request_id: 'req_123',
    model: 'gpt2',
    task: 'text-generation',
    inputs: 'Hello, world!',
    parameters: {max_length: 50},
    stream: true
}));

// Receive streaming response
ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.type === 'inference_chunk') {
        console.log('Token:', data.chunk);
    }
};
```

### 3. libp2p Inference (`libp2p_inference.py`)

Distributed peer-to-peer inference across the network.

**Features:**
- Automatic peer discovery (mDNS, bootstrap peers)
- Request routing to available peers
- Load balancing across P2P network
- Fault tolerance with automatic failover
- Performance-aware peer selection

**Capabilities:**
```python
from ipfs_accelerate_py.libp2p_inference import LibP2PInferenceNode, PeerCapability

# Create and start P2P node
node = LibP2PInferenceNode({
    'bootstrap_peers': [
        '/ip4/104.131.131.82/tcp/4001/p2p/QmaCpDMGvV2BGHeYERUEnRQAwe3N8SzbUtfsmvsqQLuvuJ'
    ],
    'enable_mdns': True
})
await node.start()

# Register local capabilities
node.register_capability(PeerCapability.TEXT_GENERATION)
node.register_model('gpt2')

# Submit distributed inference request
response = await node.submit_inference_request(
    task='text-generation',
    model='gpt2',
    inputs='Hello, world!',
    parameters={'max_length': 50}
)
```

### 4. Unified Inference Service (`unified_inference_service.py`)

Single entry point that coordinates all components.

**What It Does:**
- Initializes and manages backend manager
- Starts HF model server with WebSocket support
- Connects to P2P network
- Registers all available backends automatically
- Provides comprehensive status reporting

**Quick Start:**
```python
from ipfs_accelerate_py.unified_inference_service import (
    start_unified_service, InferenceServiceConfig
)

# Create configuration
config = InferenceServiceConfig(
    enable_backend_manager=True,
    enable_hf_server=True,
    enable_websocket=True,
    enable_libp2p=True,
    enable_api_backends=True,
    enable_cli_backends=True,
    hf_server_host='0.0.0.0',
    hf_server_port=8000
)

# Start service
service = await start_unified_service(config)

# Access components
backend_manager = service.get_backend_manager()
hf_server = service.get_hf_server()
p2p_node = service.get_p2p_node()
```

## Multi-Protocol Serving

### HTTP/REST API

OpenAI-compatible endpoints:
- `POST /v1/completions` - Text completions
- `POST /v1/chat/completions` - Chat completions
- `POST /v1/embeddings` - Text embeddings
- `GET /v1/models` - List available models

Extended endpoints:
- `POST /models/load` - Load a model
- `POST /models/unload` - Unload a model
- `GET /status` - Server status
- `GET /health` - Health check

### WebSocket

Real-time bidirectional communication:
- `ws://host:port/ws/{client_id}` - Client WebSocket connection

Message types:
- Inference requests with streaming support
- Status updates and monitoring
- Topic-based subscriptions

### libp2p

Distributed P2P inference:
- Automatic peer discovery
- Capability-based routing
- Network-wide load balancing
- Fault-tolerant execution

### MCP Server

Model Context Protocol integration:
- `list_inference_backends()` - List available backends
- `get_backend_status()` - Get status report
- `select_backend_for_inference()` - Backend selection
- `route_inference_request()` - Route inference
- `get_supported_tasks()` - List supported tasks

## Backend Discovery and Reporting

### Automatic Backend Discovery

The system automatically discovers and registers:
1. **API Backends**: HF-TGI, HF-TEI, Ollama, OpenAI API, etc.
2. **CLI Backends**: Claude CLI, OpenAI CLI, Gemini CLI, etc.
3. **Local GPU**: CUDA, ROCm, MPS acceleration
4. **P2P Peers**: libp2p network nodes
5. **WebSocket Endpoints**: Real-time streaming backends

### Health Monitoring

Continuous health checks:
- Periodic backend health verification
- Automatic status updates
- Failed request tracking
- Performance metrics collection

Status levels:
- `HEALTHY`: Backend operational
- `DEGRADED`: Experiencing issues
- `UNHEALTHY`: Not responding correctly
- `OFFLINE`: Not reachable
- `INITIALIZING`: Starting up
- `UNKNOWN`: Status not yet determined

### Metrics Tracking

Per-backend metrics:
- Total requests
- Successful/failed requests
- Average latency
- Current queue size
- Active connections
- Models loaded
- Uptime

## Load Balancing Strategies

### Round Robin
Distributes requests evenly across backends:
```python
manager = get_backend_manager({
    'load_balancing': 'round_robin'
})
```

### Least Loaded
Selects backend with smallest queue:
```python
manager = get_backend_manager({
    'load_balancing': 'least_loaded'
})
```

### Best Performance
Selects backend with best average latency:
```python
manager = get_backend_manager({
    'load_balancing': 'best_performance'
})
```

## Configuration

### Environment Variables

```bash
# Backend Manager
BACKEND_HEALTH_CHECK_INTERVAL=60
LOAD_BALANCING_STRATEGY=round_robin

# HF Model Server
HF_SERVER_HOST=0.0.0.0
HF_SERVER_PORT=8000
HF_AUTO_DISCOVER=true
HF_ENABLE_HARDWARE_DETECTION=true

# libp2p
LIBP2P_BOOTSTRAP_PEERS=/ip4/...
LIBP2P_ENABLE_MDNS=true
LIBP2P_DISCOVERY_INTERVAL=60

# API Backends
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
HF_API_TOKEN=hf_...
```

### Configuration File

```python
# config.py
from ipfs_accelerate_py.unified_inference_service import InferenceServiceConfig

config = InferenceServiceConfig(
    # Backend Manager
    enable_backend_manager=True,
    backend_health_checks=True,
    backend_health_check_interval=60,
    load_balancing_strategy="round_robin",
    
    # HF Server
    enable_hf_server=True,
    hf_server_host="0.0.0.0",
    hf_server_port=8000,
    hf_auto_discover_skills=True,
    hf_enable_hardware_detection=True,
    
    # WebSocket
    enable_websocket=True,
    
    # libp2p
    enable_libp2p=True,
    libp2p_bootstrap_peers=[
        "/ip4/104.131.131.82/tcp/4001/p2p/QmaCpDMGvV2BGHeYERUEnRQAwe3N8SzbUtfsmvsqQLuvuJ"
    ],
    
    # Backends
    enable_api_backends=True,
    api_backends=["hf_tgi", "hf_tei", "ollama", "openai_api"],
    enable_cli_backends=True,
    cli_backends=["claude_cli", "openai_cli"]
)
```

## Deployment Scenarios

### 1. Local Development

Single machine with GPU:
```python
config = InferenceServiceConfig(
    enable_hf_server=True,
    enable_websocket=True,
    enable_libp2p=False,  # Disable P2P for local dev
    enable_api_backends=False,
    enable_cli_backends=False
)
```

### 2. Multi-GPU Server

Multiple GPUs with load balancing:
```python
config = InferenceServiceConfig(
    enable_backend_manager=True,
    load_balancing_strategy="least_loaded",
    enable_hf_server=True,
    hf_enable_hardware_detection=True
)
```

### 3. Distributed P2P Network

Multiple nodes across network:
```python
config = InferenceServiceConfig(
    enable_libp2p=True,
    libp2p_bootstrap_peers=[...],
    enable_backend_manager=True,
    load_balancing_strategy="best_performance"
)
```

### 4. Hybrid Cloud + Local

Mix of local and cloud resources:
```python
config = InferenceServiceConfig(
    enable_hf_server=True,  # Local GPU
    enable_api_backends=True,  # Cloud APIs
    api_backends=["openai_api", "anthropic"],
    load_balancing_strategy="round_robin"
)
```

## Best Practices

1. **Health Monitoring**: Always enable health checks in production
2. **Load Balancing**: Choose strategy based on workload characteristics
3. **Fallback Backends**: Register multiple backends for redundancy
4. **Resource Limits**: Configure appropriate queue sizes and timeouts
5. **Monitoring**: Use metrics tracking to identify bottlenecks
6. **Security**: Use API keys and authentication for production deployments
7. **P2P Network**: Use bootstrap peers for reliable network joining

## Troubleshooting

### Backend Not Responding

1. Check backend status: `manager.get_backend_status_report()`
2. Verify health checks are enabled
3. Check backend-specific logs
4. Ensure network connectivity

### High Latency

1. Review load balancing strategy
2. Check queue sizes: `backend.metrics.current_queue_size`
3. Consider adding more backends
4. Enable batching if supported

### P2P Connection Issues

1. Verify bootstrap peers are reachable
2. Check firewall settings
3. Enable mDNS for local discovery
4. Review libp2p logs

### WebSocket Disconnections

1. Implement reconnection logic in client
2. Use ping/pong for keepalive
3. Check network stability
4. Review timeout settings

## API Reference

See individual module documentation:
- `inference_backend_manager.py` - Backend management
- `websocket_handler.py` - WebSocket protocol
- `libp2p_inference.py` - P2P networking
- `unified_inference_service.py` - Service coordination
- `mcp/tools/backend_management.py` - MCP integration
