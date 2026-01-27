# P2P Workflow Scheduling and MCP Server Architecture

## Overview

IPFS Accelerate Python includes advanced peer-to-peer (P2P) workflow scheduling and Model Context Protocol (MCP) server capabilities that enable distributed task execution and standardized AI model interaction.

## Table of Contents

- [P2P Workflow Scheduler](#p2p-workflow-scheduler)
- [MCP Server Architecture](#mcp-server-architecture)
- [MCP Tools Reference](#mcp-tools-reference)
- [CLI Endpoint Adapters](#cli-endpoint-adapters)
- [Integration Guide](#integration-guide)

---

## P2P Workflow Scheduler

The P2P Workflow Scheduler (`ipfs_accelerate_py/p2p_workflow_scheduler.py`) provides distributed task execution across peer-to-peer networks, bypassing centralized APIs for improved scalability and resilience.

### Key Features

- **Merkle Clock Consensus**: Distributed consensus using vector clocks with merkle tree hashing
- **Fibonacci Heap Scheduling**: Efficient O(1) insert and O(log n) delete-min for priority-based task scheduling
- **Peer Selection**: Hamming distance calculation for optimal peer assignment
- **Workflow Tagging**: Flexible tagging system for execution mode classification

### Architecture Components

#### 1. MerkleClock

Implements a hybrid vector clock + merkle tree for distributed consensus.

```python
from ipfs_accelerate_py.p2p_workflow_scheduler import MerkleClock

# Create a merkle clock for this node
clock = MerkleClock(node_id="node-123")

# Increment clock on local event
clock.tick()

# Merge with another node's clock
other_clock = MerkleClock(node_id="node-456")
clock.update(other_clock)

# Get merkle root hash for consensus
merkle_root = clock.get_hash()

# Compare clocks for causality
comparison = clock.compare(other_clock)
# Returns: -1 (before), 0 (concurrent), 1 (after)
```

**Use Cases:**
- Determining task ownership in distributed systems
- Resolving conflicts in P2P task assignment
- Maintaining causal consistency across network nodes

#### 2. FibonacciHeap

Efficient priority queue for workflow scheduling with amortized O(1) insert.

```python
from ipfs_accelerate_py.p2p_workflow_scheduler import FibonacciHeap

heap = FibonacciHeap()

# Insert tasks with priority (lower = higher priority)
heap.insert(key=1, data={"task": "high_priority_job", "workflow_id": "wf-001"})
heap.insert(key=5, data={"task": "low_priority_job", "workflow_id": "wf-002"})
heap.insert(key=3, data={"task": "medium_priority_job", "workflow_id": "wf-003"})

# Extract highest priority task
min_task = heap.extract_min()
# Returns: {"task": "high_priority_job", "workflow_id": "wf-001"}

# Decrease priority of a task (makes it more urgent)
heap.decrease_key(node, new_key=0)
```

**Benefits:**
- O(1) amortized insert time
- O(log n) amortized delete-min time
- O(1) amortized decrease-key operation
- Ideal for dynamic priority workflows

#### 3. WorkflowTag System

Classification system for workflow execution modes:

```python
from ipfs_accelerate_py.p2p_workflow_scheduler import WorkflowTag

class WorkflowTag(Enum):
    GITHUB_API = "github-api"          # Standard GitHub API workflows
    P2P_ELIGIBLE = "p2p-eligible"      # Can execute via P2P network
    P2P_ONLY = "p2p-only"             # Must execute via P2P (bypasses GitHub)
    UNIT_TEST = "unit-test"            # Unit test workflows
    CODE_GENERATION = "code-generation" # Code generation tasks
    WEB_SCRAPING = "web-scraping"      # Web scraping tasks
    DATA_PROCESSING = "data-processing" # Data processing tasks
```

**Usage Example:**

```python
from ipfs_accelerate_py.p2p_workflow_scheduler import P2PWorkflowScheduler

scheduler = P2PWorkflowScheduler(node_id="worker-01")

# Tag workflow for P2P execution
workflow = {
    "id": "wf-001",
    "name": "model-inference-batch",
    "tag": WorkflowTag.P2P_ELIGIBLE,
    "priority": 1,
    "tasks": [...]
}

scheduler.schedule_workflow(workflow)

# Get next task to execute
next_task = scheduler.get_next_task()
```

#### 4. Peer Selection Algorithm

Uses hamming distance to select optimal peers for task distribution:

```python
def calculate_hamming_distance(peer_capabilities: List[str], 
                               task_requirements: List[str]) -> int:
    """
    Calculate hamming distance between peer capabilities and task requirements.
    Lower distance = better match.
    """
    # Convert to binary vectors and compute XOR
    distance = sum(1 for cap, req in zip(peer_capabilities, task_requirements) 
                   if cap != req)
    return distance
```

**Example:**

```python
peer_capabilities = ["cuda", "fp16", "large-memory", "fast-network"]
task_requirements = ["cuda", "fp16", "large-memory", "slow-network"]

distance = calculate_hamming_distance(peer_capabilities, task_requirements)
# Returns: 1 (only network speed differs)
```

### P2P Scheduler Configuration

```python
from ipfs_accelerate_py.p2p_workflow_scheduler import P2PWorkflowScheduler

scheduler = P2PWorkflowScheduler(
    node_id="worker-01",
    config={
        "max_concurrent_tasks": 4,
        "peer_discovery_interval": 30,  # seconds
        "merkle_sync_interval": 60,     # seconds
        "priority_boost_factor": 0.9,   # boost factor for starved tasks
        "enable_task_stealing": True,   # allow tasks to be stolen from busy peers
        "heartbeat_timeout": 120,       # seconds before peer considered dead
    }
)

# Start scheduler
await scheduler.start()

# Submit workflow
workflow_id = await scheduler.submit_workflow({
    "name": "batch-inference",
    "tag": WorkflowTag.P2P_ELIGIBLE,
    "priority": 2,
    "tasks": [
        {"model": "bert-base", "input": "text1"},
        {"model": "bert-base", "input": "text2"},
    ]
})

# Monitor progress
status = await scheduler.get_workflow_status(workflow_id)
print(f"Completed: {status['completed']}/{status['total']} tasks")
```

---

## MCP Server Architecture

The Model Context Protocol (MCP) server (`ipfs_accelerate_py/mcp/server.py`) provides a standardized interface for AI model interaction using the FastMCP framework.

### Server Implementation

#### StandaloneMCP

Basic MCP server without IPFS Accelerate integration:

```python
from mcp.server import create_standalone_mcp_server

# Create standalone server
mcp = create_standalone_mcp_server()

# Run server
if __name__ == "__main__":
    mcp.run()
```

#### IPFSAccelerateMCPServer

Full-featured MCP server with IPFS Accelerate integration:

```python
from mcp.server import create_mcp_server

# Create MCP server with IPFS Accelerate context
mcp = create_mcp_server(
    resources={
        "models": ["bert-base-uncased", "gpt2"],
        "hardware": {
            "preferred": "cuda",
            "fallback": ["mps", "cpu"]
        }
    },
    metadata={
        "project": "my-ml-project",
        "version": "1.0.0"
    }
)

# Run server on custom host/port
if __name__ == "__main__":
    mcp.run(transport="stdio")  # or "sse" for Server-Sent Events
```

### MCP Server Features

1. **Tool Registration**: Automatically registers 14+ MCP tools
2. **Resource Management**: Handles model loading, caching, and lifecycle
3. **Prompt Templates**: Provides reusable prompt templates for common tasks
4. **Context Management**: Maintains conversation context and state
5. **Error Handling**: Comprehensive error reporting and recovery

### Starting the MCP Server

#### Via CLI

```bash
# Start basic MCP server
ipfs-accelerate mcp start

# Start with dashboard
ipfs-accelerate mcp dashboard

# Start with custom config
ipfs-accelerate mcp start --config config.json --port 8080

# Check server status
ipfs-accelerate mcp status
```

#### Via Python API

```python
import anyio
from mcp.server import create_mcp_server

async def main():
    # Create and configure server
    mcp = create_mcp_server(
        resources={},
        metadata={"environment": "production"}
    )
    
    # Start server
    await mcp.serve_stdio()

anyio.run(main)
```

#### Via Docker

```bash
# Build MCP server container
docker build -t ipfs-accelerate-mcp -f Dockerfile.mcp .

# Run MCP server
docker run -p 8080:8080 \
  -v $(pwd)/models:/models \
  -e MCP_LOG_LEVEL=info \
  ipfs-accelerate-mcp
```

---

## MCP Tools Reference

The MCP server provides 14 specialized tools across different categories:

### 1. Inference Tools

#### `enhanced_inference.py` - Multi-Backend Inference Orchestration

Intelligent routing across local, distributed, API, and CLI inference modes.

```python
# Tool: run_enhanced_inference
{
    "model": "bert-base-uncased",
    "input": "Hello, world!",
    "mode": "auto",  # auto, local, distributed, api, cli
    "backend": "auto"  # auto, vllm, ollama, openai, etc.
}
```

**Supported Modes:**
- `local`: Direct local model inference
- `distributed`: P2P distributed inference across network
- `api`: Cloud API endpoints (OpenAI, Anthropic, etc.)
- `cli`: CLI-based inference (ollama, llama.cpp, etc.)
- `auto`: Automatic mode selection based on availability

#### `inference.py` - Basic Inference

Standard inference without advanced routing.

```python
# Tool: run_inference
{
    "model": "gpt2",
    "input": "Once upon a time",
    "max_length": 100
}
```

### 2. Model Management Tools

#### `models.py` - Model Discovery and Management

```python
# Tool: search_models
{
    "query": "bert sentiment analysis",
    "filter": {
        "task": "text-classification",
        "library": "transformers"
    },
    "limit": 10
}

# Tool: get_model_recommendations
{
    "task": "text-generation",
    "hardware": "cuda",
    "constraints": {
        "max_parameters": "7B",
        "max_memory_gb": 16
    }
}

# Tool: check_model_compatibility
{
    "model": "meta-llama/Llama-2-7b-hf",
    "hardware": "cpu",
    "check_performance": true
}
```

### 3. Hardware Tools

#### `hardware.py` - Hardware Detection and Testing

```python
# Tool: detect_hardware
{}  # No parameters, returns full hardware info

# Tool: test_hardware_capability
{
    "hardware_type": "cuda",
    "test_mode": "inference",  # inference, training, memory
    "model_size": "7B"
}

# Tool: get_optimal_configuration
{
    "model": "bert-base-uncased",
    "available_hardware": ["cuda", "cpu"],
    "optimization_target": "latency"  # latency, throughput, memory
}
```

### 4. Workflow Tools

#### `workflows.py` - Workflow Management

```python
# Tool: list_workflows
{
    "status": "running",  # running, completed, failed, all
    "tag": "p2p-eligible"
}

# Tool: get_workflow_status
{
    "workflow_id": "wf-001"
}

# Tool: cancel_workflow
{
    "workflow_id": "wf-002"
}
```

#### `p2p_workflow_tools.py` - P2P Workflow Control

```python
# Tool: submit_p2p_workflow
{
    "name": "distributed-inference",
    "tasks": [
        {"model": "bert-base", "input": "text1"},
        {"model": "bert-base", "input": "text2"}
    ],
    "priority": 1,
    "tag": "p2p-only"
}

# Tool: get_p2p_network_status
{}  # Returns peer count, active tasks, network health

# Tool: rebalance_p2p_tasks
{
    "strategy": "load"  # load, latency, proximity
}
```

### 5. GitHub Integration Tools

#### `github_tools.py` - GitHub API and Cache Management

```python
# Tool: gh_list_repos
{
    "owner": "myorg",
    "type": "all",  # all, public, private
    "limit": 50
}

# Tool: gh_workflow_runs
{
    "repo": "myorg/myrepo",
    "workflow": "ci.yml",
    "status": "completed"
}

# Tool: gh_cache_stats
{}  # Returns cache hit rate, size, entries

# Tool: gh_provision_runner
{
    "repo": "myorg/myrepo",
    "labels": ["self-hosted", "gpu"],
    "count": 2
}
```

### 6. Dashboard Tools

#### `dashboard_data.py` - Metrics Aggregation

```python
# Tool: get_dashboard_metrics
{
    "time_range": "24h",  # 1h, 24h, 7d, 30d
    "metrics": ["inference_count", "latency_p50", "error_rate"]
}

# Tool: get_model_usage_stats
{
    "model": "bert-base-uncased",
    "period": "week"
}
```

### 7. System Tools

#### `status.py` - System Status

```python
# Tool: get_system_status
{}  # Returns overall health, resource usage

# Tool: get_service_health
{
    "services": ["mcp", "p2p", "ipfs"]
}
```

#### `system_logs.py` - Log Management

```python
# Tool: get_recent_logs
{
    "level": "error",  # debug, info, warning, error
    "lines": 100,
    "service": "mcp"
}

# Tool: search_logs
{
    "query": "inference failed",
    "time_range": "1h"
}
```

### 8. Endpoint Tools

#### `endpoints.py` - Endpoint Management

```python
# Tool: list_endpoints
{
    "type": "all"  # all, local, distributed, api
}

# Tool: test_endpoint
{
    "endpoint_id": "ep-001",
    "test_payload": {"input": "test"}
}
```

---

## CLI Endpoint Adapters

The CLI Endpoint Adapters (`ipfs_accelerate_py/mcp/tools/cli_endpoint_adapters.py`) provide direct integration with popular AI CLI tools.

### Supported CLI Tools

#### 1. Claude Desktop CLI

```python
from mcp.tools.cli_endpoint_adapters import ClaudeDesktopAdapter

adapter = ClaudeDesktopAdapter()

# Send prompt to Claude
response = await adapter.send_prompt(
    prompt="Explain quantum computing",
    model="claude-3-opus"
)

# Stream response
async for chunk in adapter.stream_prompt(prompt="Write a poem"):
    print(chunk, end="", flush=True)
```

#### 2. OpenAI CLI

```python
from mcp.tools.cli_endpoint_adapters import OpenAICLIAdapter

adapter = OpenAICLIAdapter()

response = await adapter.chat_completion(
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "What is ML?"}
    ],
    model="gpt-4"
)
```

#### 3. Google Gemini CLI

```python
from mcp.tools.cli_endpoint_adapters import GeminiCLIAdapter

adapter = GeminiCLIAdapter()

response = await adapter.generate_content(
    prompt="Explain transformer architecture",
    model="gemini-pro"
)
```

#### 4. VSCode Copilot

```python
from mcp.tools.cli_endpoint_adapters import VSCodeCopilotAdapter

adapter = VSCodeCopilotAdapter()

# Get code completion
completion = await adapter.get_completion(
    context="def calculate_fibonacci(n):",
    language="python"
)

# Get code explanation
explanation = await adapter.explain_code(
    code="for i in range(10): print(i)",
    language="python"
)
```

### CLI Adapter Configuration

```python
# Configure all adapters
from mcp.tools.cli_endpoint_adapters import configure_cli_adapters

configure_cli_adapters({
    "claude": {
        "api_key": "sk-...",
        "default_model": "claude-3-opus"
    },
    "openai": {
        "api_key": "sk-...",
        "organization": "org-..."
    },
    "gemini": {
        "api_key": "AI...",
        "project": "my-project"
    },
    "vscode": {
        "auth_token": "..."
    }
})
```

---

## Integration Guide

### Integrating P2P Scheduler with MCP Server

```python
from mcp.server import create_mcp_server
from ipfs_accelerate_py.p2p_workflow_scheduler import P2PWorkflowScheduler

# Create P2P scheduler
scheduler = P2PWorkflowScheduler(node_id="node-01")

# Create MCP server with P2P context
mcp = create_mcp_server(
    resources={
        "p2p_scheduler": scheduler,
        "enable_p2p": True
    }
)

# Register P2P tools
from mcp.tools.p2p_workflow_tools import register_p2p_tools
register_p2p_tools(mcp, scheduler)

# Start both services
async def main():
    await scheduler.start()
    await mcp.serve_stdio()

anyio.run(main)
```

### Using MCP Tools from Python

```python
from mcp.server import create_mcp_server

# Create server
mcp = create_mcp_server()

# Call MCP tool directly
result = await mcp.call_tool(
    "run_enhanced_inference",
    {
        "model": "bert-base-uncased",
        "input": "Test input",
        "mode": "local"
    }
)

print(result)
```

### Integration with External Systems

#### FastAPI Integration

```python
from fastapi import FastAPI
from mcp.server import create_mcp_server

app = FastAPI()
mcp = create_mcp_server()

@app.post("/inference")
async def inference_endpoint(request: dict):
    result = await mcp.call_tool(
        "run_enhanced_inference",
        request
    )
    return result

# Run both FastAPI and MCP server
```

#### GitHub Actions Integration

```yaml
name: P2P Distributed Inference

on: [push]

jobs:
  inference:
    runs-on: self-hosted
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup P2P Node
        run: |
          python -m ipfs_accelerate_py.p2p_workflow_scheduler setup \
            --node-id ${{ github.run_id }}
      
      - name: Run Distributed Inference
        run: |
          python -m ipfs_accelerate_py.p2p_workflow_scheduler submit \
            --workflow inference.yml \
            --tag p2p-eligible
```

---

## Performance Considerations

### P2P Scheduler Performance

- **Task Insertion**: O(1) amortized via Fibonacci heap
- **Task Extraction**: O(log n) amortized
- **Peer Selection**: O(n) where n = number of active peers
- **Merkle Sync**: O(m) where m = number of nodes in network

**Optimization Tips:**
1. Use appropriate priority values (0-1000 recommended)
2. Set `max_concurrent_tasks` based on hardware capabilities
3. Enable `task_stealing` for better load balancing
4. Tune `merkle_sync_interval` based on network latency

### MCP Server Performance

- **Cold Start**: ~2-3 seconds (with model loading)
- **Warm Inference**: <100ms for small models
- **Tool Call Overhead**: ~5-10ms per call
- **Concurrent Requests**: Supports up to 100 concurrent connections

**Optimization Tips:**
1. Pre-load frequently used models on server startup
2. Use connection pooling for API backends
3. Enable response caching for repeated queries
4. Set appropriate timeout values

---

## Security Considerations

### P2P Network Security

1. **Peer Authentication**: Use mutual TLS for peer connections
2. **Task Verification**: Verify merkle clock signatures before accepting tasks
3. **Resource Limits**: Set hard limits on task resources (CPU, memory, time)
4. **Network Isolation**: Use private networks for sensitive workloads

### MCP Server Security

1. **Authentication**: Require API keys or OAuth tokens
2. **Rate Limiting**: Implement per-client rate limits
3. **Input Validation**: Sanitize all inputs before processing
4. **Resource Quotas**: Enforce compute and storage quotas per user

---

## Troubleshooting

### P2P Scheduler Issues

**Problem**: Tasks not being assigned
```bash
# Check peer connectivity
python -m ipfs_accelerate_py.p2p_workflow_scheduler diagnose --check-peers

# Verify merkle clock sync
python -m ipfs_accelerate_py.p2p_workflow_scheduler sync-clocks --force
```

**Problem**: High task latency
```bash
# Check peer load distribution
python -m ipfs_accelerate_py.p2p_workflow_scheduler stats --show-load

# Enable task stealing
python -m ipfs_accelerate_py.p2p_workflow_scheduler config \
  --set enable_task_stealing=true
```

### MCP Server Issues

**Problem**: Server not starting
```bash
# Check dependencies
pip install -e ".[mcp]"

# Verify port availability
lsof -i :8080

# Check logs
tail -f ~/.local/share/ipfs_accelerate/mcp.log
```

**Problem**: Tool call failures
```bash
# Test tool directly
python -m mcp.tools.test_tool enhanced_inference

# Enable debug logging
export MCP_LOG_LEVEL=debug
ipfs-accelerate mcp start
```

---

## Further Reading

- [ARCHITECTURE.md](ARCHITECTURE.md) - Overall system architecture
- [API.md](API.md) - Complete API reference
- [P2P_SETUP_GUIDE.md](../P2P_SETUP_GUIDE.md) - P2P network setup
- [MCP_SETUP_GUIDE.md](../MCP_SETUP_GUIDE.md) - MCP server configuration
- [TESTING.md](TESTING.md) - Testing framework and best practices

---

**Last Updated**: January 2026  
**Version**: 0.0.45+  
**Status**: Production Ready ✅
