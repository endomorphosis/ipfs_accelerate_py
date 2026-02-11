# Multi-Protocol Inference Deployment Guide

This guide provides step-by-step instructions for deploying the unified inference service with support for HTTP, WebSocket, and libp2p protocols.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Deployment Scenarios](#deployment-scenarios)
5. [Protocol-Specific Setup](#protocol-specific-setup)
6. [Monitoring and Management](#monitoring-and-management)
7. [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements

- **Python**: 3.8 or higher
- **Memory**: Minimum 8GB RAM (16GB+ recommended for GPU inference)
- **Storage**: 10GB+ free space for models
- **GPU** (optional): CUDA 11.0+ or ROCm for accelerated inference

### Network Requirements

For P2P (libp2p) deployment:
- **Ports**: TCP/UDP ports for P2P communication (default: 4001)
- **Firewall**: Allow inbound/outbound P2P traffic
- **NAT**: Port forwarding or UPnP for external access

For WebSocket:
- **Ports**: HTTP server port (default: 8000)
- **SSL/TLS**: Recommended for production (WSS protocol)

## Installation

### 1. Install Base Package

```bash
# Clone repository
git clone https://github.com/endomorphosis/ipfs_accelerate_py
cd ipfs_accelerate_py

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install base requirements
pip install -r requirements.txt
```

### 2. Install HF Model Server Requirements

```bash
# Install HF server dependencies
pip install -r requirements-hf-server.txt

# Install additional ML frameworks (choose based on your hardware)
pip install torch torchvision  # For CUDA
# OR
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm5.6  # For ROCm
# OR
pip install torch torchvision  # For CPU/MPS (Apple Silicon)
```

### 3. Install libp2p Dependencies (Optional)

```bash
# Install libp2p for P2P networking
pip install "libp2p @ git+https://github.com/libp2p/py-libp2p@main"
pip install pymultihash>=0.8.2
```

### 4. Install MCP Dependencies (Optional)

```bash
# Install MCP server requirements
pip install -r ipfs_accelerate_py/mcp/requirements.txt
```

## Configuration

### Basic Configuration

Create a configuration file `config.py`:

```python
from ipfs_accelerate_py.unified_inference_service import InferenceServiceConfig

config = InferenceServiceConfig(
    # Core services
    enable_backend_manager=True,
    enable_hf_server=True,
    enable_websocket=True,
    enable_libp2p=False,  # Enable if using P2P
    
    # Server settings
    hf_server_host="0.0.0.0",
    hf_server_port=8000,
    
    # Backend manager
    backend_health_checks=True,
    backend_health_check_interval=60,
    load_balancing_strategy="round_robin",
    
    # Logging
    log_level="INFO"
)
```

### Environment Variables

Create `.env` file:

```bash
# Server Configuration
HF_SERVER_HOST=0.0.0.0
HF_SERVER_PORT=8000

# API Keys (if using external APIs)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
HF_API_TOKEN=hf_...
GROQ_API_KEY=gsk_...

# libp2p Configuration
LIBP2P_BOOTSTRAP_PEERS=/ip4/104.131.131.82/tcp/4001/p2p/QmaCpDMGvV2BGHeYERUEnRQAwe3N8SzbUtfsmvsqQLuvuJ
LIBP2P_ENABLE_MDNS=true

# Backend Configuration
BACKEND_HEALTH_CHECK_INTERVAL=60
LOAD_BALANCING_STRATEGY=round_robin
```

## Deployment Scenarios

### Scenario 1: Local HTTP Server (Simplest)

For basic HTTP inference without P2P or advanced features:

```python
# run_local_server.py
import asyncio
from ipfs_accelerate_py.unified_inference_service import (
    start_unified_service, InferenceServiceConfig
)

async def main():
    config = InferenceServiceConfig(
        enable_backend_manager=True,
        enable_hf_server=True,
        enable_websocket=False,
        enable_libp2p=False,
        enable_api_backends=False,
        enable_cli_backends=False,
        hf_server_host="127.0.0.1",
        hf_server_port=8000
    )
    
    service = await start_unified_service(config)
    
    # Run server
    import uvicorn
    uvicorn.run(
        service.get_hf_server().app,
        host=config.hf_server_host,
        port=config.hf_server_port
    )

if __name__ == "__main__":
    asyncio.run(main())
```

Run:
```bash
python run_local_server.py
```

Test:
```bash
curl http://localhost:8000/health
curl http://localhost:8000/v1/models
```

### Scenario 2: HTTP + WebSocket Server

For real-time inference with streaming support:

```python
# run_websocket_server.py
import asyncio
from ipfs_accelerate_py.unified_inference_service import (
    start_unified_service, InferenceServiceConfig
)

async def main():
    config = InferenceServiceConfig(
        enable_backend_manager=True,
        enable_hf_server=True,
        enable_websocket=True,  # Enable WebSocket
        enable_libp2p=False,
        hf_server_host="0.0.0.0",
        hf_server_port=8000
    )
    
    service = await start_unified_service(config)
    
    import uvicorn
    uvicorn.run(
        service.get_hf_server().app,
        host=config.hf_server_host,
        port=config.hf_server_port
    )

if __name__ == "__main__":
    asyncio.run(main())
```

Test WebSocket:
```bash
# Install wscat for testing
npm install -g wscat

# Connect and test
wscat -c ws://localhost:8000/ws/test_client

# Send ping
{"type": "ping"}

# Subscribe to topics
{"type": "subscribe", "topics": ["inference", "status"]}

# Run inference
{
  "type": "inference",
  "request_id": "req_123",
  "model": "gpt2",
  "task": "text-generation",
  "inputs": "Hello, world!",
  "stream": true
}
```

### Scenario 3: Multi-Backend with API Multiplexing

For production with multiple backends and failover:

```python
# run_multi_backend.py
import asyncio
from ipfs_accelerate_py.unified_inference_service import (
    start_unified_service, InferenceServiceConfig
)

async def main():
    config = InferenceServiceConfig(
        enable_backend_manager=True,
        enable_hf_server=True,
        enable_websocket=True,
        enable_libp2p=False,
        enable_api_backends=True,  # Enable API backends
        api_backends=[
            "hf_tgi",
            "hf_tei",
            "ollama",
            "openai_api"
        ],
        enable_cli_backends=True,  # Enable CLI backends
        cli_backends=[
            "claude_cli",
            "openai_cli"
        ],
        load_balancing_strategy="best_performance",
        backend_health_checks=True
    )
    
    service = await start_unified_service(config)
    
    import uvicorn
    uvicorn.run(
        service.get_hf_server().app,
        host=config.hf_server_host,
        port=config.hf_server_port
    )

if __name__ == "__main__":
    asyncio.run(main())
```

### Scenario 4: Distributed P2P Network

For peer-to-peer distributed inference:

```python
# run_p2p_node.py
import asyncio
from ipfs_accelerate_py.unified_inference_service import (
    start_unified_service, InferenceServiceConfig
)

async def main():
    config = InferenceServiceConfig(
        enable_backend_manager=True,
        enable_hf_server=True,
        enable_websocket=True,
        enable_libp2p=True,  # Enable P2P
        libp2p_bootstrap_peers=[
            "/ip4/104.131.131.82/tcp/4001/p2p/QmaCpDMGvV2BGHeYERUEnRQAwe3N8SzbUtfsmvsqQLuvuJ"
        ],
        libp2p_enable_mdns=True,
        libp2p_discovery_interval=60
    )
    
    service = await start_unified_service(config)
    
    # Register capabilities on P2P node
    p2p_node = service.get_p2p_node()
    if p2p_node:
        from ipfs_accelerate_py.libp2p_inference import PeerCapability
        p2p_node.register_capability(PeerCapability.TEXT_GENERATION)
        p2p_node.register_model("gpt2")
    
    import uvicorn
    uvicorn.run(
        service.get_hf_server().app,
        host=config.hf_server_host,
        port=config.hf_server_port
    )

if __name__ == "__main__":
    asyncio.run(main())
```

## Protocol-Specific Setup

### HTTP/REST API Setup

1. **Configure CORS** (if needed for web clients):
```python
config = ServerConfig()
config.enable_cors = True
config.cors_origins = ["http://localhost:3000", "https://myapp.com"]
```

2. **Add Authentication** (recommended for production):
```python
# In server.py routes
from fastapi import Header, HTTPException

def verify_token(authorization: str = Header(None)):
    if not authorization or authorization != "Bearer YOUR_SECRET_TOKEN":
        raise HTTPException(status_code=401, detail="Invalid token")

@app.get("/v1/models", dependencies=[Depends(verify_token)])
async def list_models():
    # ...
```

3. **Setup HTTPS** (production):
```bash
# Generate SSL certificate
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes

# Run with SSL
uvicorn main:app --host 0.0.0.0 --port 443 --ssl-keyfile key.pem --ssl-certfile cert.pem
```

### WebSocket Setup

1. **Configure Connection Manager**:
```python
from ipfs_accelerate_py.hf_model_server.websocket_handler import get_connection_manager

connection_manager = get_connection_manager()
# Manager is automatically configured
```

2. **Client Implementation** (JavaScript):
```javascript
class InferenceWebSocketClient {
    constructor(url) {
        this.ws = new WebSocket(url);
        this.handlers = {};
        
        this.ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            if (this.handlers[data.type]) {
                this.handlers[data.type](data);
            }
        };
    }
    
    on(type, handler) {
        this.handlers[type] = handler;
    }
    
    send(message) {
        this.ws.send(JSON.stringify(message));
    }
    
    subscribe(topics) {
        this.send({ type: 'subscribe', topics });
    }
    
    inference(model, inputs, options = {}) {
        this.send({
            type: 'inference',
            model,
            inputs,
            task: options.task || 'text-generation',
            parameters: options.parameters || {},
            stream: options.stream || false,
            request_id: options.request_id || `req_${Date.now()}`
        });
    }
}

// Usage
const client = new InferenceWebSocketClient('ws://localhost:8000/ws/my_client');

client.on('inference_result', (data) => {
    console.log('Result:', data.result);
});

client.on('inference_chunk', (data) => {
    console.log('Token:', data.chunk);
});

client.inference('gpt2', 'Hello, world!', { stream: true });
```

3. **Python Client Example**:
```python
import asyncio
import websockets
import json

async def websocket_client():
    uri = "ws://localhost:8000/ws/my_client"
    
    async with websockets.connect(uri) as websocket:
        # Subscribe to topics
        await websocket.send(json.dumps({
            "type": "subscribe",
            "topics": ["inference"]
        }))
        
        # Send inference request
        await websocket.send(json.dumps({
            "type": "inference",
            "request_id": "req_001",
            "model": "gpt2",
            "task": "text-generation",
            "inputs": "Hello, world!",
            "stream": True
        }))
        
        # Receive responses
        async for message in websocket:
            data = json.loads(message)
            print(f"Received: {data}")
            
            if data["type"] == "inference_complete":
                break

asyncio.run(websocket_client())
```

### libp2p P2P Setup

1. **Configure Bootstrap Peers**:
```python
# Use public IPFS bootstrap nodes
bootstrap_peers = [
    "/dnsaddr/bootstrap.libp2p.io/p2p/QmNnooDu7bfjPFoTZYxMNLWUQJyrVwtbZg5gBMjTezGAJN",
    "/dnsaddr/bootstrap.libp2p.io/p2p/QmQCU2EcMqAqQPR2i9bChDtGNJchTbq5TbXJJ16u19uLTa",
    "/ip4/104.131.131.82/tcp/4001/p2p/QmaCpDMGvV2BGHeYERUEnRQAwe3N8SzbUtfsmvsqQLuvuJ"
]

config.libp2p_bootstrap_peers = bootstrap_peers
```

2. **Enable mDNS for Local Discovery**:
```python
config.libp2p_enable_mdns = True
```

3. **Register Capabilities**:
```python
p2p_node = service.get_p2p_node()

# Register what your node can do
from ipfs_accelerate_py.libp2p_inference import PeerCapability

p2p_node.register_capability(PeerCapability.TEXT_GENERATION)
p2p_node.register_capability(PeerCapability.TEXT_EMBEDDING)

# Register available models
p2p_node.register_model("gpt2")
p2p_node.register_model("bert-base-uncased")
```

## Monitoring and Management

### Backend Status Monitoring

```python
# Get comprehensive status
manager = service.get_backend_manager()
status = manager.get_backend_status_report()

print(f"Total Backends: {status['total_backends']}")
print(f"Healthy Backends: {status['backends_by_status']['healthy']}")
print(f"Total Requests: {status['total_requests']}")
```

### Health Check Endpoint

```bash
# Check server health
curl http://localhost:8000/health

# Check readiness
curl http://localhost:8000/ready

# Get detailed status
curl http://localhost:8000/status
```

### MCP Integration for Management

```python
# Use MCP tools for management
from ipfs_accelerate_py.mcp.tools.backend_management import (
    list_inference_backends,
    get_backend_status,
    select_backend_for_inference
)

# List all backends
backends = list_inference_backends()

# Get status
status = get_backend_status()

# Select backend for task
backend = select_backend_for_inference(
    task="text-generation",
    model="gpt2"
)
```

## Troubleshooting

### Common Issues

**1. Port Already in Use**
```bash
# Check what's using the port
lsof -i :8000  # macOS/Linux
netstat -ano | findstr :8000  # Windows

# Use different port
config.hf_server_port = 8001
```

**2. libp2p Connection Failures**
```bash
# Check firewall
sudo ufw allow 4001/tcp  # Linux
# Or configure Windows Firewall

# Test bootstrap peer connectivity
telnet 104.131.131.82 4001
```

**3. WebSocket Connection Drops**
```python
# Increase timeout in client
ws = new WebSocket(url);
ws.onclose = () => {
    // Reconnect after delay
    setTimeout(() => connectWebSocket(), 5000);
};
```

**4. Backend Not Responding**
```python
# Check backend health
manager = get_backend_manager()
backend = manager.get_backend("backend_id")
print(f"Status: {backend.status}")
print(f"Last seen: {backend.last_seen}")

# Force health check
await manager.check_backend_health("backend_id")
```

### Logging

Enable debug logging:
```python
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

View logs for specific components:
```python
logging.getLogger("ipfs_accelerate_py.inference_backend_manager").setLevel(logging.DEBUG)
logging.getLogger("ipfs_accelerate_py.libp2p_inference").setLevel(logging.DEBUG)
logging.getLogger("ipfs_accelerate_py.hf_model_server").setLevel(logging.DEBUG)
```

## Next Steps

- [Unified Inference Architecture](UNIFIED_INFERENCE_ARCHITECTURE.md) - Detailed architecture documentation
- [API Reference](../ipfs_accelerate_py/) - Code documentation
- [Examples](../examples/) - More usage examples
