# Quick Start Guide - IPFS Accelerate Python

Get started quickly with IPFS Accelerate Python for hardware-accelerated ML inference, P2P workflow scheduling, and MCP server integration.

## 📋 Table of Contents

- [Installation](#installation)
- [Basic Inference](#basic-inference)
- [MCP Server](#mcp-server)
- [P2P Workflow Scheduling](#p2p-workflow-scheduling)
- [GitHub Integration](#github-integration)
- [CLI Tools](#cli-tools)
- [Next Steps](#next-steps)

---

## Installation

```bash
# Quick install
pip install ipfs-accelerate-py

# Or install from source
git clone https://github.com/endomorphosis/ipfs_accelerate_py.git
cd ipfs_accelerate_py
pip install -e .
```

For detailed installation options, see [INSTALL.md](INSTALL.md).

---

## Basic Inference

### Python API

```python
from ipfs_accelerate_py import ipfs_accelerate_py

# Initialize
accelerator = ipfs_accelerate_py({}, {})

# Run inference
result = accelerator.process(
    model="bert-base-uncased",
    input_data={"input_ids": [101, 2054, 2003, 2026, 2171, 102]},
    endpoint_type="text_embedding"
)

print(result)
```

### CLI

```bash
# Run inference via CLI
ipfs-accelerate inference generate \
  --model bert-base-uncased \
  --input "Hello, world!"

# List available models
ipfs-accelerate models list

# Detect hardware
ipfs-accelerate hardware detect
```

---

## MCP Server

### Quick Start

```bash
# Start MCP server
ipfs-accelerate mcp start

# Start with dashboard
ipfs-accelerate mcp dashboard

# Check status
ipfs-accelerate mcp status
```

### Using MCP Tools

```python
from mcp.server import create_mcp_server
import anyio

async def main():
    # Create MCP server
    mcp = create_mcp_server()
    
    # Call tool for inference
    result = await mcp.call_tool(
        "run_enhanced_inference",
        {
            "model": "bert-base-uncased",
            "input": "Test input",
            "mode": "auto"
        }
    )
    
    print(result)

anyio.run(main)
```

**Available Tools**: 14+ tools for inference, model management, hardware detection, workflows, and GitHub integration.

See [docs/P2P_AND_MCP.md](docs/P2P_AND_MCP.md#mcp-tools-reference) for complete tool reference.

---

## P2P Workflow Scheduling

### Distributed Task Execution

```python
from ipfs_accelerate_py.p2p_workflow_scheduler import (
    P2PWorkflowScheduler,
    WorkflowTag
)
import anyio

async def main():
    # Create P2P scheduler
    scheduler = P2PWorkflowScheduler(node_id="worker-01")
    
    # Start scheduler
    await scheduler.start()
    
    # Submit distributed workflow
    workflow_id = await scheduler.submit_workflow({
        "name": "batch-inference",
        "tag": WorkflowTag.P2P_ELIGIBLE,
        "priority": 1,
        "tasks": [
            {"model": "bert-base", "input": "text1"},
            {"model": "bert-base", "input": "text2"},
            {"model": "bert-base", "input": "text3"}
        ]
    })
    
    # Monitor progress
    status = await scheduler.get_workflow_status(workflow_id)
    print(f"Completed: {status['completed']}/{status['total']}")

anyio.run(main)
```

### Via CLI

```bash
# Submit P2P workflow
ipfs-accelerate workflow submit \
  --tag p2p-eligible \
  --priority 1 \
  batch-inference.yml

# Check network status
ipfs-accelerate network p2p-status

# Monitor workflow
ipfs-accelerate workflow status <workflow-id>
```

See [docs/P2P_AND_MCP.md](docs/P2P_AND_MCP.md#p2p-workflow-scheduler) for detailed documentation.

---

## GitHub Integration

---

## GitHub Integration

### Auto-Scaling Runner Service

The autoscaler automatically monitors workflows and provisions runners as needed:

```bash
# 1. Authenticate (one time)
gh auth login

# 2. Start the autoscaler (runs continuously)
ipfs-accelerate github autoscaler

# Or with options
ipfs-accelerate github autoscaler --owner myorg --interval 30
```

The autoscaler will:
- ✅ Monitor your repos for workflow activity
- ✅ Detect running and failed workflows automatically
- ✅ Provision self-hosted runners on demand
- ✅ Respect your system's CPU core limit
- ✅ Work completely automatically once started

**See [AUTOSCALER.md](AUTOSCALER.md) for complete autoscaler documentation.**

### GitHub CLI Commands

```bash
# List repositories
ipfs-accelerate github repos --owner myorg

# List workflow runs
ipfs-accelerate github workflows --repo myorg/myrepo

# Provision runner
ipfs-accelerate github provision-runner \
  --repo myorg/myrepo \
  --labels self-hosted,gpu

# Check cache stats
ipfs-accelerate github cache-stats
```

### P2P Cache for GitHub Actions

```bash
# Setup P2P cache
ipfs-accelerate github setup-p2p-cache

# Enable for workflow
# Add to .github/workflows/your-workflow.yml:
jobs:
  build:
    runs-on: self-hosted
    steps:
      - uses: actions/cache@v3
        with:
          path: ~/.cache
          key: p2p-cache-${{ github.sha }}
          restore-keys: p2p-cache-
```

---

## CLI Tools

### Inference

```bash
# Text generation
ipfs-accelerate inference generate \
  --model gpt2 \
  --input "Once upon a time" \
  --max-length 100

# Text embedding
ipfs-accelerate inference embed \
  --model bert-base-uncased \
  --input "Hello, world!"

# Batch inference
ipfs-accelerate inference batch \
  --model bert-base \
  --input-file inputs.txt \
  --output-file results.json
```

### Model Management

```bash
# Search models
ipfs-accelerate models search "sentiment analysis"

# Get model info
ipfs-accelerate models info bert-base-uncased

# Check compatibility
ipfs-accelerate models check-compat \
  --model llama-2-7b \
  --hardware cuda

# Download model
ipfs-accelerate models download bert-base-uncased
```

### Hardware Operations

```bash
# Detect hardware
ipfs-accelerate hardware detect

# Test capabilities
ipfs-accelerate hardware test --type cuda

# Get optimal config
ipfs-accelerate hardware optimize --model bert-base
```

### Network Operations

```bash
# IPFS network status
ipfs-accelerate network status

# Add file to IPFS
ipfs-accelerate files add myfile.txt

# Get file from IPFS
ipfs-accelerate files get QmHash... output.txt
```

---

## Next Steps

### Documentation

- **[USAGE.md](docs/USAGE.md)** - Comprehensive usage guide
- **[API.md](docs/API.md)** - Complete API reference
- **[P2P_AND_MCP.md](docs/P2P_AND_MCP.md)** - P2P & MCP architecture
- **[HARDWARE.md](docs/HARDWARE.md)** - Hardware optimization guide
- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)** - System architecture

### Examples

Explore practical examples in the [examples/](examples/) directory:

- Basic inference examples
- Multi-backend routing
- P2P distributed workflows
- MCP server integration
- GitHub Actions workflows

### Testing

```bash
# Run tests
pytest

# Run with coverage
pytest --cov=ipfs_accelerate_py

# Run benchmarks
python benchmarks/run_benchmarks.py
```

### Community

- **GitHub Issues**: [Report bugs](https://github.com/endomorphosis/ipfs_accelerate_py/issues)
- **Discussions**: [Ask questions](https://github.com/endomorphosis/ipfs_accelerate_py/discussions)
- **Contributing**: See [CONTRIBUTING.md](CONTRIBUTING.md)

---

## Quick Verification

Run the integration test to verify everything works:

```bash
# Test basic functionality
python -c "from ipfs_accelerate_py import ipfs_accelerate_py; print('✓ Import successful')"

# Test hardware detection
ipfs-accelerate hardware detect

# Test inference
ipfs-accelerate inference generate \
  --model bert-base-uncased \
  --input "Test input"

# Test MCP server
ipfs-accelerate mcp status
```

Expected output: No errors, hardware detected, inference completes successfully.

---

**For complete documentation, see [docs/README.md](docs/README.md)**

