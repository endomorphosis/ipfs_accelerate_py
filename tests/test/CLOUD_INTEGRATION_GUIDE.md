# Multi-Node and Cloud Integration Guide

This guide covers the distributed benchmark and cloud integration capabilities of the IPFS Accelerate Python Framework, focusing on the newly implemented multi-node benchmarking, cloud platform integration, and model serving infrastructure.

## Overview

The multi-node and cloud integration system provides:

1. **Distributed Benchmarking** - Run benchmarks across multiple nodes (local and cloud-based)
2. **Cloud Platform Support** - Deploy and run models on AWS, GCP, and Azure
3. **Cost Optimization** - Compare performance and cost across platforms
4. **Model Deployment** - Deploy optimized models to cloud environments

## Key Components

### Distributed Benchmark Coordinator

The `DistributedBenchmarkCoordinator` in `multi_node_cloud_integration.py` is the central component that manages distributed benchmarking and cloud deployment. It provides:

- Node discovery and management across platforms
- Parallel benchmark execution on multiple nodes
- Result aggregation and comparison
- Performance reporting with cost analysis
- Model deployment with compression

## Getting Started

### Prerequisites

To use cloud integration features, you'll need:

- Python 3.8+
- PyTorch 2.0+
- Transformers library
- Cloud provider SDKs (as needed):
  - AWS: `pip install boto3`
  - GCP: `pip install google-cloud-storage google-cloud-compute`
  - Azure: `pip install azure-storage-blob azure-mgmt-compute`

### Basic Usage

#### Listing Available Nodes

```bash
# List all available nodes (local and cloud)
python test/multi_node_cloud_integration.py list-nodes

# Save node list to a file
python test/multi_node_cloud_integration.py list-nodes --output nodes.json
```

#### Running Distributed Benchmarks

```bash
# Run benchmark on default nodes with multiple models
python test/multi_node_cloud_integration.py benchmark --models bert-base-uncased,t5-small --batch-sizes 1,2,4,8

# Specify nodes to use
python test/multi_node_cloud_integration.py benchmark --models bert-base-uncased --nodes local,aws-g4dn.xlarge
```

#### Generating Reports

```bash
# Generate comparison report from benchmark results
python test/multi_node_cloud_integration.py report --results ./distributed_benchmarks/benchmark_results_*.json
```

#### Deploying Models

```bash
# Deploy model with compression optimizations
python test/multi_node_cloud_integration.py deploy --model bert-base-uncased --target local:cpu --optimization balanced

# Deploy to cloud with aggressive optimization
python test/multi_node_cloud_integration.py deploy --model t5-small --target aws:g4dn.xlarge --optimization aggressive
```

#### Model Serving

```bash
# Start cloud-based model serving
python test/multi_node_cloud_integration.py serve --model bert-base-uncased --provider aws --instance g4dn.xlarge
```

## Configuration

The multi-node and cloud integration system can be configured using a JSON configuration file:

```json
{
  "nodes": [
    {"id": "local", "type": "local", "name": "Local Node"},
    {"id": "aws-g4dn-xlarge", "type": "aws", "name": "AWS G4dn Xlarge", "instance_type": "g4dn.xlarge", "region": "us-west-2"}
  ],
  "benchmark_defaults": {
    "repeats": 3,
    "batch_sizes": [1, 2, 4, 8],
    "timeout_seconds": 600
  },
  "cloud_defaults": {
    "aws": {
      "region": "us-west-2",
      "instance_type": "g4dn.xlarge"
    },
    "gcp": {
      "zone": "us-central1-a",
      "machine_type": "n1-standard-4"
    },
    "azure": {
      "location": "eastus",
      "vm_size": "Standard_NC6s_v3"
    }
  }
}
```

Specify the configuration file path when running commands:

```bash
python test/multi_node_cloud_integration.py benchmark --models bert-base-uncased --config my_config.json
```

## Cloud Credentials

Cloud credentials can be provided through environment variables:

### AWS

```bash
export AWS_ACCESS_KEY_ID=your_key_id
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_DEFAULT_REGION=us-west-2
```

### GCP

```bash
export GOOGLE_APPLICATION_CREDENTIALS=path_to_service_account_json
```

### Azure

```bash
export AZURE_STORAGE_CONNECTION_STRING=your_connection_string
```

## Benchmark Reports

The system generates comprehensive benchmark reports in Markdown format, including:

1. **Performance Comparison** - Latency, throughput, and memory usage across nodes
2. **Node Comparison** - Average throughput and success rates by node
3. **Cost Comparison** - Estimated costs across cloud providers
4. **Performance Recommendations** - Optimal node and batch size for each model

Example report sections:

### Latency Comparison (seconds)

| Node | Batch 1 | Batch 2 | Batch 4 | Batch 8 |
| ---- | ------- | ------- | ------- | ------- |
| local | 0.0235 | 0.0320 | 0.0480 | 0.0795 |
| aws-g4dn.xlarge | 0.0182 | 0.0225 | 0.0320 | 0.0510 |
| gcp-n1-standard-8 | 0.0195 | 0.0240 | 0.0350 | 0.0560 |

### Cost Comparison

| Provider | Total Cost | Duration (hours) | Cost per hour |
| -------- | ---------- | ---------------- | ------------- |
| AWS | $0.26 | 0.50 | $0.52 |
| GCP | $0.19 | 0.50 | $0.38 |

## Cloud-Specific Optimizations

The system applies cloud-specific optimizations for each provider:

### AWS Optimizations

- Instance type selection based on model size and type
- Region selection based on cost and latency requirements
- GPU selection (T4, V100, A10G) based on model characteristics
- Cost optimization with spot instances for non-critical workloads

### GCP Optimizations

- Machine type optimization (standard, highmem, highcpu)
- Custom hardware optimizations for specific model types
- Region selection based on latency and cost

### Azure Optimizations

- VM size selection based on model requirements
- Hardware acceleration selection (CPU, GPU, FPGA)
- Region optimization for latency and cost

## Integration with Model Compression

The system integrates with the model compression system for optimized cloud deployment:

```bash
# Deploy with compression optimizations
python test/multi_node_cloud_integration.py deploy --model bert-base-uncased --target aws:g4dn.xlarge --optimization aggressive
```

This combines model compression techniques with cloud deployment for optimal performance and cost:

1. Quantization - INT8/FP16 based on cloud hardware capabilities
2. Pruning - Model size reduction for faster deployment and lower costs
3. Graph optimization - Cloud-specific graph optimizations
4. Format conversion - Optimal format for each cloud provider (ONNX, TensorRT, etc.)

## Performance Comparison Examples

### Text Generation Models (LLMs)

| Model | Platform | Throughput (tokens/sec) | Latency (sec) | Cost ($/hour) |
|-------|----------|-------------------------|---------------|---------------|
| LLaMA (7B) | Local CUDA | 25.2 | 0.55 | N/A |
| LLaMA (7B) | AWS g4dn.xlarge | 21.8 | 0.62 | $0.53 |
| LLaMA (7B) | AWS p3.2xlarge | 38.5 | 0.32 | $3.06 |
| LLaMA (7B) | GCP n1-standard-8 + T4 | 22.5 | 0.60 | $0.69 |
| LLaMA (7B) | Azure NC6s_v3 | 35.2 | 0.35 | $0.75 |

### Embedding Models

| Model | Platform | Throughput (embeddings/sec) | Latency (ms) | Cost ($/hour) |
|-------|----------|----------------------------|--------------|---------------|
| BERT-base | Local CPU | 185 | 5.4 | N/A |
| BERT-base | Local CUDA | 925 | 1.1 | N/A |
| BERT-base | AWS g4dn.xlarge | 880 | 1.2 | $0.53 |
| BERT-base | GCP n1-standard-4 | 210 | 4.8 | $0.19 |
| BERT-base | GCP n1-standard-4 + T4 | 840 | 1.2 | $0.63 |
| BERT-base | Azure Standard_DS4_v3 | 205 | 4.9 | $0.19 |

## Cost Optimization Guidelines

The system provides cost optimization recommendations based on workload:

### Batch Processing Workloads

- Use spot instances or preemptible VMs for non-critical batch jobs
- Select regions with lower costs for jobs without latency requirements
- Use auto-scaling groups to handle variable workloads
- Consider reserved instances for consistent workloads

### Serving Workloads

- Balance instance size and batch processing for optimal cost per prediction
- Use multi-model serving when possible to maximize hardware utilization
- Consider smaller model variants with distillation for cost-sensitive deployments
- Implement caching for frequent requests

### Hybrid Deployment

- Deploy compute-intensive preprocessing on cloud, inference on local hardware
- Use cloud for peak demand handling, local for baseline workloads
- Implement workload-specific routing between cloud providers

## Advanced Features

### Custom Hardware Selection

```bash
# Run benchmark with specific hardware filtering
python test/multi_node_cloud_integration.py benchmark --models t5-base --hardware-filter "GPU:T4+"
```

### Distributed Model Parallelism

For large models that exceed single-node capability:

```bash
# Deploy with model parallelism
python test/multi_node_cloud_integration.py deploy --model llama-70b --target aws:p3.8xlarge --sharded
```

### Cross-Provider Deployment

Deploy the same model across multiple providers for comparison:

```bash
# Deploy to multiple providers
python test/multi_node_cloud_integration.py deploy --model bert-base --targets aws:g4dn.xlarge,gcp:n1-standard-4-t4,azure:Standard_NC6s_v3
```

## Conclusion

The multi-node and cloud integration system provides a comprehensive solution for distributed benchmarking, cloud deployment, and cost optimization. By combining model compression techniques with cloud-specific optimizations, it enables efficient deployment of models across various environments.

For detailed implementation, see `multi_node_cloud_integration.py`.