"""
Example Prompts for IPFS Accelerate MCP Server

This module provides example prompts for using the MCP server's capabilities.
"""

import logging
from typing import Dict, List, Any

logger = logging.getLogger("ipfs_accelerate_mcp.prompts.examples")

def register_prompts(mcp):
    """Register example prompts with the MCP server"""
    
    @mcp.prompt("hardware_detection")
    def hardware_detection_prompt() -> str:
        """
        Prompt for hardware detection
        
        This prompt provides an example of how to detect available hardware.
        
        Returns:
            Example prompt for hardware detection
        """
        return """
# Hardware Detection with IPFS Accelerate

IPFS Accelerate provides comprehensive hardware detection capabilities to help you identify the best accelerators for your AI workloads.

## Basic Hardware Detection

To detect available hardware accelerators on your system:

```python
# Connect to the MCP server
client = FastMCPClient("http://localhost:8080")

# Test available hardware
hardware = client.use_tool("test_hardware")
print(f"Available accelerators: {hardware['available_accelerators']}")

# Get detailed hardware info
hardware_info = client.use_tool("get_hardware_info")
print(f"CPU: {hardware_info['processor']}")
print(f"Memory: {hardware_info['memory']['total_gb']:.1f} GB")

if "gpus" in hardware_info:
    for i, gpu in enumerate(hardware_info["gpus"]):
        print(f"GPU {i+1}: {gpu['name']} ({gpu['memory_gb']:.1f} GB)")
```

## IPFS Accelerate Hardware Detection

IPFS Accelerate extends standard hardware detection with specialized capabilities:

```python
# Use IPFS-specific hardware detection
ipfs_hw = client.use_tool("test_ipfs_hardware")

# Check for specialized accelerators
if "webgpu_available" in ipfs_hw and ipfs_hw["webgpu_available"]:
    print("WebGPU acceleration is available!")
    
if "webnn_available" in ipfs_hw and ipfs_hw["webnn_available"]:
    print("WebNN acceleration is available!")
```

## Hardware Recommendations

Get tailored hardware recommendations for your specific model size:

```python
# Get recommendations for a 7B parameter model
recommendations = client.use_tool(
    "get_hardware_recommendation",
    model_size=7_000_000_000,
    task_type="text-generation"
)

# Print the best recommendation
best = recommendations["best_recommendation"]
print(f"Best hardware for 7B model: {best['device']}")
print(f"Reason: {best['reason']}")
```

This will give you a complete overview of your system's hardware capabilities and how to best leverage them with IPFS Accelerate.
"""
    
    @mcp.prompt("model_recommendation")
    def model_recommendation_prompt() -> str:
        """
        Prompt for model recommendation
        
        This prompt provides an example of how to get hardware and model recommendations.
        
        Returns:
            Example prompt for model recommendation
        """
        return """
# IPFS Accelerate Model Recommendations

IPFS Accelerate helps you select the optimal model and hardware configuration for your specific AI task.

## Task-Based Model Selection

To get model recommendations based on your task:

```python
# Connect to the MCP server
client = FastMCPClient("http://localhost:8080")

# Get model recommendations for a specific task
recommendations = client.use_tool(
    "get_model_recommendations",
    task="text-summarization",
    max_response_tokens=1024,
    hardware_constraints={"max_memory_gb": 8}
)

# Print top recommendations
for i, model in enumerate(recommendations["models"][:3]):
    print(f"{i+1}. {model['name']}")
    print(f"   Parameters: {model['parameters']:,}")
    print(f"   Hardware: {model['recommended_hardware']}")
    print(f"   Performance score: {model['performance_score']:.2f}/10")
```

## Hardware-Model Matching

To find the best hardware for a specific model:

```python
# Get hardware recommendation for a specific model
hardware = client.use_tool(
    "get_hardware_for_model",
    model_id="mistralai/Mistral-7B-v0.1",
    batch_size=4,
    max_sequence_length=8192
)

# Print recommendation
print(f"Recommended hardware: {hardware['recommended_device']}")
print(f"Required memory: {hardware['required_memory_gb']:.1f}GB")
```

## IPFS Accelerate Enhanced Models

IPFS Accelerate optimizes certain models for distributed inference:

```python
# Get IPFS Accelerate optimized models
optimized = client.use_tool("get_ipfs_optimized_models")

# Filter for a specific task
task_models = [m for m in optimized["models"] if "text-generation" in m["tasks"]]
for model in task_models:
    print(f"{model['name']}: {model['ipfs_optimization_level']}/5 optimization level")
```

This allows you to select the optimal model and hardware configuration for your specific requirements.

```python
# Connect to the MCP server
client = FastMCPClient("http://localhost:8080")

# Define your model size (in parameters)
model_size = 7_000_000_000  # 7B parameters

# Get hardware recommendation
recommendation = client.use_tool(
    "get_hardware_recommendation",
    model_size=model_size,
    task_type="generation"  # Options: "generation", "inference", "embedding", "training"
)

# Check the best recommendation
best = recommendation["best_recommendation"]
print(f"Best device: {best['device']}")
print(f"Suitability score: {best['suitability_score']}/10")
print(f"Suitable: {best['suitable']}")
print(f"Reason: {best['reason']}")

# View all recommendations
for rec in recommendation["recommendations"]:
    print(f"Device: {rec['device']}, Score: {rec['suitability_score']}/10")
```

This will help you choose the most appropriate hardware for your model.
"""
    
    @mcp.prompt("run_inference")
    def run_inference_prompt() -> str:
        """
        Prompt for running inference
        
        This prompt provides an example of how to run inference on a model.
        
        Returns:
            Example prompt for running inference
        """
        return """
To run inference on a machine learning model:

```python
# Connect to the MCP server
client = FastMCPClient("http://localhost:8080")

# For embedding generation
texts = [
    "This is the first example sentence.",
    "Here is another sentence to embed."
]

results = client.use_tool(
    "run_inference",
    model="BAAI/bge-small-en-v1.5",  # An embedding model
    inputs=texts,
    device="cpu"  # Or use "cuda:0", "openvino", etc. based on hardware
)

# Access the embeddings
embeddings = results["embeddings"]
print(f"Generated {len(embeddings)} embeddings of size {results['embedding_size']}")

# For text generation
prompts = ["Explain what IPFS is in a simple way."]

results = client.use_tool(
    "run_inference",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # A text generation model
    inputs=prompts,
    device="cpu",
    max_length=100
)

# Access the generated text
outputs = results["outputs"]
print(f"Generated response: {outputs[0]}")
```

This allows you to leverage the hardware acceleration capabilities for ML inference.
"""
    
    @mcp.prompt("manage_endpoints")
    def manage_endpoints_prompt() -> str:
        """
        Prompt for managing endpoints
        
        This prompt provides an example of how to manage model endpoints.
        
        Returns:
            Example prompt for managing endpoints
        """
        return """
To manage persistent model endpoints:

```python
# Connect to the MCP server
client = FastMCPClient("http://localhost:8080")

# Create a new endpoint
endpoint = client.use_tool(
    "add_endpoint",
    model="BAAI/bge-small-en-v1.5",
    device="cpu",
    max_batch_size=32,
    description="Embedding endpoint for documents"
)

# Get the endpoint ID
endpoint_id = endpoint["id"]

# Use the endpoint for inference
results = client.use_tool(
    "run_inference",
    endpoint_id=endpoint_id,
    inputs=["This is a test sentence."]
)

# Update the endpoint
updated = client.use_tool(
    "update_endpoint",
    endpoint_id=endpoint_id,
    max_batch_size=64,
    description="Updated embedding endpoint"
)

# List all endpoints
endpoints = client.use_tool("get_endpoints")
for ep in endpoints["endpoints"]:
    print(f"ID: {ep['id']}, Model: {ep['model']}, Status: {ep['status']}")

# Remove the endpoint when done
removed = client.use_tool("remove_endpoint", endpoint_id=endpoint_id)
```

Endpoints allow you to keep models loaded for repeated inference operations.
"""
    
    @mcp.prompt("monitor_performance")
    def monitor_performance_prompt() -> str:
        """
        Prompt for monitoring performance
        
        This prompt provides an example of how to monitor server performance.
        
        Returns:
            Example prompt for monitoring performance
        """
        return """
To monitor server performance:

```python
# Connect to the MCP server
client = FastMCPClient("http://localhost:8080")

# Get server status
status = client.use_tool("get_server_status")
print(f"Server version: {status['version']}")
print(f"Uptime: {status['uptime_seconds']:.1f} seconds")

# Get performance metrics
metrics = client.use_tool("get_performance_metrics")
print(f"CPU usage: {metrics['cpu_percent']:.1f}%")
print(f"Memory usage: {metrics['memory_percent']:.1f}%")
print(f"Disk usage: {metrics['disk_percent']:.1f}%")

# Start a monitoring session
session = client.use_tool("start_session", session_name="Performance test")
session_id = session["id"]

# Run operations and log them
for i in range(5):
    # Log an operation
    log = client.use_tool(
        "log_operation",
        session_id=session_id,
        operation_type=f"test_operation_{i}",
        processing_time=0.1 * (i + 1)
    )
    
    # Perform some work here...
    import time
    time.sleep(0.5)

# End the session
end = client.use_tool("end_session", session_id=session_id)
print(f"Session duration: {end['duration_seconds']:.1f} seconds")
print(f"Average processing time: {end['avg_processing_time']:.3f} seconds")
```

This helps you monitor and optimize performance of your ML operations.
"""

    @mcp.prompt("integration_with_ipfs")
    def integration_with_ipfs_prompt() -> str:
        """
        Prompt for IPFS Accelerate integration
        
        This prompt provides an example of how to use the integrated mode with IPFS Accelerate.
        
        Returns:
            Example prompt for IPFS Accelerate integration
        """
        return """
To use the MCP server with IPFS Accelerate integration:

1. Start the server in integrated mode:

```bash
python -m ipfs_accelerate_py.mcp.standalone --integrated
```

2. Access IPFS Accelerate-specific capabilities:

```python
# Connect to the MCP server
client = FastMCPClient("http://localhost:8080")

# Get IPFS Accelerate version
version = client.access_resource("ipfs_accelerate_version")
print(f"IPFS Accelerate version: {version}")

# Get supported acceleration methods
support = client.access_resource("acceleration_support")
print("Supported acceleration methods:")
for method, available in support.items():
    print(f"  {method}: {'Yes' if available else 'No'}")

# Detect hardware using IPFS Accelerate
if "detect_ipfs_hardware" in client.list_tools():
    hardware = client.use_tool("detect_ipfs_hardware")
    print(f"WebNN available: {hardware['webnn_available']}")
    print(f"WebGPU available: {hardware['webgpu_available']}")
```

This allows you to leverage IPFS Accelerate's specific hardware acceleration capabilities.
"""
