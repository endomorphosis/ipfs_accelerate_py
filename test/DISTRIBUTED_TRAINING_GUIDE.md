# Distributed Training Configuration Guide

## Overview

The IPFS Accelerate Python Framework provides comprehensive tools for optimizing distributed training configurations across different hardware platforms. This guide explains how to use the hardware selection system and distributed training suite to achieve optimal performance for training large models.

## Table of Contents

1. [Introduction to Distributed Training](#introduction-to-distributed-training)
2. [Hardware Selection System](#hardware-selection-system)
3. [Distributed Training Configuration](#distributed-training-configuration)
4. [Memory Optimization Techniques](#memory-optimization-techniques)
5. [Command-Line Tools](#command-line-tools)
6. [Integration with Training Frameworks](#integration-with-training-frameworks)

## Introduction to Distributed Training

Distributed training enables efficient training of large models across multiple GPUs or nodes. The framework supports several distributed training strategies:

- **DDP (DistributedDataParallel)**: Data parallelism with model replication
- **FSDP (Fully Sharded Data Parallel)**: Shards model parameters, gradients, and optimizer states
- **DeepSpeed ZeRO**: Multi-stage optimization with different levels of sharding

Each strategy has different trade-offs in terms of memory usage, communication overhead, and ease of implementation.

## Hardware Selection System

The `hardware_selector.py` module provides automated hardware selection based on model characteristics and benchmarking data.

### Basic Distributed Selection

```python
from hardware_selector import HardwareSelector

# Create hardware selector
selector = HardwareSelector()

# Select optimal hardware for distributed training
result = selector.select_hardware_for_task(
    model_family="text_generation",
    model_name="facebook/opt-1.3b",
    task_type="training",
    batch_size=8,
    distributed=True,
    gpu_count=4
)

# Access results
print(f"Primary recommendation: {result['primary_recommendation']}")
print(f"Distributed strategy: {result['distributed_strategy']}")
print(f"GPU count: {result['gpu_count']}")
print(f"Fallback options: {result['fallback_options']}")
```

## Distributed Training Configuration

The framework provides tools to generate optimal distributed training configurations based on model characteristics and available hardware.

### Configuration Generation

```python
# Generate detailed distributed training configuration
config = selector.get_distributed_training_config(
    model_family="text_generation",
    model_name="facebook/opt-1.3b",
    gpu_count=8,
    batch_size=4,
    max_memory_gb=24  # Optional GPU memory constraint
)

# Access configuration
print(f"Strategy: {config['distributed_strategy']}")  # DDP, FSDP, or DeepSpeed
print(f"Per-GPU batch size: {config['per_gpu_batch_size']}")
print(f"Global batch size: {config['global_batch_size']}")
print(f"Mixed precision: {config['mixed_precision']}")

# Check memory estimates
memory_info = config['estimated_memory']
print(f"Parameters: {memory_info['parameters_gb']:.2f} GB")
print(f"Activations: {memory_info['activations_gb']:.2f} GB")
print(f"Optimizer states: {memory_info['optimizer_gb']:.2f} GB")
print(f"Total per GPU: {memory_info['per_gpu_gb']:.2f} GB")
```

### Strategy Selection

The framework automatically selects the optimal distributed training strategy based on model size, GPU count, and memory constraints:

- **Small models (< 1B parameters)**:
  - 1-4 GPUs: DDP
  - 4+ GPUs: DDP or FSDP depending on memory requirements

- **Medium models (1-10B parameters)**:
  - 1-2 GPUs: DDP with gradient checkpointing
  - 4-8 GPUs: FSDP
  - 8+ GPUs: FSDP or DeepSpeed ZeRO-3

- **Large models (10B+ parameters)**:
  - 1-4 GPUs: FSDP with CPU offloading
  - 4-8 GPUs: DeepSpeed ZeRO-3
  - 8+ GPUs: DeepSpeed ZeRO-3 with ZeRO-Infinity

## Memory Optimization Techniques

The framework provides several memory optimization techniques for large models:

### Gradient Accumulation

Reduces memory usage by splitting batches and accumulating gradients:

```python
config = selector.get_distributed_training_config(
    model_family="text_generation",
    model_name="facebook/opt-6.7b",
    gpu_count=4,
    batch_size=8,
    max_memory_gb=16
)

grad_accum_steps = config.get("gradient_accumulation_steps", 1)
print(f"Gradient accumulation steps: {grad_accum_steps}")
print(f"Effective batch size: {config['per_gpu_batch_size'] * config['gpu_count'] * grad_accum_steps}")
```

### Gradient Checkpointing

Trades computation for memory by not storing intermediate activations:

```python
config = selector.get_distributed_training_config(
    model_family="text_generation",
    model_name="facebook/opt-6.7b",
    gpu_count=4,
    batch_size=8,
    max_memory_gb=16
)

if "gradient_checkpointing" in config and config["gradient_checkpointing"]:
    print("Gradient checkpointing enabled")
    print("Memory savings: Approximately 60-80% reduction in activation memory")
```

### DeepSpeed ZeRO Optimization

Multi-stage optimization with different levels of sharding:

```python
config = selector.get_distributed_training_config(
    model_family="text_generation",
    model_name="facebook/opt-30b",
    gpu_count=8,
    batch_size=1,
    max_memory_gb=40
)

if "deepspeed_config" in config:
    zero_stage = config["deepspeed_config"].get("zero_stage", 1)
    print(f"DeepSpeed ZeRO Stage: {zero_stage}")
    
    if zero_stage == 2:
        print("Memory optimization: Optimizer states sharded across GPUs")
    elif zero_stage == 3:
        print("Memory optimization: Parameters, gradients, and optimizer states sharded across GPUs")
```

### FSDP Configuration

PyTorch Fully Sharded Data Parallel with CPU offloading:

```python
config = selector.get_distributed_training_config(
    model_family="text_generation",
    model_name="facebook/opt-13b",
    gpu_count=4,
    batch_size=1,
    max_memory_gb=16
)

if "fsdp_config" in config:
    print(f"FSDP sharding strategy: {config['fsdp_config'].get('sharding_strategy', 'FULL_SHARD')}")
    
    if config['fsdp_config'].get('cpu_offload', False):
        print("CPU offloading enabled: Optimizer states stored in CPU memory")
    
    if config['fsdp_config'].get('activation_checkpointing', False):
        print("Activation checkpointing enabled with FSDP")
```

## Command-Line Tools

### Hardware Selector CLI

The `hardware_selector.py` script provides command-line access to hardware selection:

```bash
# Distributed training configuration
python hardware_selector.py --model-family text_generation --model-name llama-7b --batch-size 16 --mode training --distributed --gpu-count 8

# With memory constraints
python hardware_selector.py --model-family text_generation --model-name llama-7b --mode training --distributed --gpu-count 8 --max-memory-gb 24
```

### Training Benchmark Generator

The `run_training_benchmark.py` script generates comprehensive benchmark configurations:

```bash
# List sample models for benchmarking
python run_training_benchmark.py --list-models

# Generate distributed benchmark configuration
python run_training_benchmark.py --model bert-base-uncased --distributed --max-gpus 4 --output bert_benchmark.json

# Specify available hardware
python run_training_benchmark.py --model t5-small --hardware cuda rocm --distributed --max-gpus 8
```

## Integration with Training Frameworks

### PyTorch Integration

```python
import torch
import torch.distributed as dist
from hardware_selector import HardwareSelector

# Initialize selector
selector = HardwareSelector()

# Get model parameters
model_name = "facebook/opt-1.3b"
model_family = "text_generation"
gpu_count = torch.cuda.device_count()
batch_size = 8
max_memory_gb = 24  # Per GPU memory constraint

# Get optimal configuration
config = selector.get_distributed_training_config(
    model_family=model_family,
    model_name=model_name,
    gpu_count=gpu_count,
    batch_size=batch_size,
    max_memory_gb=max_memory_gb
)

# Initialize distributed environment
dist.init_process_group(backend="nccl")
local_rank = dist.get_rank()
torch.cuda.set_device(local_rank)

# Create model
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(model_name)

# Apply configuration
if config["distributed_strategy"] == "DDP":
    model = model.to(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(
        model, 
        device_ids=[local_rank]
    )
    
elif config["distributed_strategy"] == "FSDP":
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        CPUOffload,
        MixedPrecision
    )
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    
    # Apply FSDP configuration
    fsdp_config = config.get("fsdp_config", {})
    
    # Configure CPU offloading if needed
    cpu_offload = None
    if fsdp_config.get("cpu_offload", False):
        cpu_offload = CPUOffload(offload_params=True)
    
    # Configure mixed precision
    mixed_precision = None
    if config.get("mixed_precision", False):
        mixed_precision = MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
            buffer_dtype=torch.float16
        )
    
    # Enable gradient checkpointing if configured
    if config.get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()
    
    # Apply FSDP
    from transformers.models.opt.modeling_opt import OPTDecoderLayer
    model = FSDP(
        model,
        auto_wrap_policy=transformer_auto_wrap_policy(transformer_layer_cls={OPTDecoderLayer}),
        cpu_offload=cpu_offload,
        mixed_precision=mixed_precision
    )

elif config["distributed_strategy"] == "DeepSpeed":
    import deepspeed
    
    # Create DeepSpeed config
    ds_config = {
        "train_batch_size": config["global_batch_size"],
        "gradient_accumulation_steps": config.get("gradient_accumulation_steps", 1),
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": 5e-5,
                "betas": [0.9, 0.999],
                "eps": 1e-8
            }
        },
        "fp16": {
            "enabled": config.get("mixed_precision", False)
        },
        "zero_optimization": {
            "stage": config.get("deepspeed_config", {}).get("zero_stage", 2),
            "offload_optimizer": config.get("deepspeed_config", {}).get("offload_optimizer", False),
            "offload_param": config.get("deepspeed_config", {}).get("offload_param", False)
        }
    }
    
    # Initialize DeepSpeed
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=ds_config
    )
    model = model_engine

# Set up training loop
grad_accum_steps = config.get("gradient_accumulation_steps", 1)
```

### HuggingFace Trainer Integration

```python
from transformers import Trainer, TrainingArguments
from hardware_selector import HardwareSelector

# Initialize selector
selector = HardwareSelector()

# Get model configuration
config = selector.get_distributed_training_config(
    model_family="text_generation",
    model_name="facebook/opt-1.3b",
    gpu_count=torch.cuda.device_count(),
    batch_size=8
)

# Create training arguments
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=config["per_gpu_batch_size"],
    gradient_accumulation_steps=config.get("gradient_accumulation_steps", 1),
    gradient_checkpointing=config.get("gradient_checkpointing", False),
    fp16=config.get("mixed_precision", False),
    local_rank=int(os.environ.get("LOCAL_RANK", -1))
)

# Add DeepSpeed configuration if needed
if config["distributed_strategy"] == "DeepSpeed":
    training_args.deepspeed = {
        "zero_stage": config.get("deepspeed_config", {}).get("zero_stage", 2),
        "offload_optimizer_device": "cpu" if config.get("deepspeed_config", {}).get("offload_optimizer", False) else "none",
        "offload_param_device": "cpu" if config.get("deepspeed_config", {}).get("offload_param", False) else "none"
    }

# Create model and trainer
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("facebook/opt-1.3b")
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b")

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer
)

# Start training
trainer.train()
```

## Conclusion

The distributed training configuration system provides powerful tools for optimizing training performance across different hardware platforms. By leveraging these tools, you can achieve optimal performance for training large models, even with memory-constrained environments.

For more information, refer to:
- [Hardware Benchmarking Guide](./HARDWARE_BENCHMARKING_GUIDE.md)
- [Model Compression Guide](./MODEL_COMPRESSION_GUIDE.md)
- [Performance Optimization Plan](./PERFORMANCE_OPTIMIZATION_PLAN.md)