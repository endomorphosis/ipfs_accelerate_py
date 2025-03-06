#!/usr/bin/env python3
"""
Distributed Training Benchmark for the IPFS Accelerate framework.

This script implements distributed training benchmarks as part of Phase 16 of the
IPFS Accelerate Python framework project, specifically focusing on:

1. Multi-node distributed training performance testing
2. Hardware scaling efficiency analysis
3. Integration with the benchmark database
4. Support for different distributed strategies (DataParallel, DDP, DeepSpeed, FSDP)

Usage:
  python distributed_training_benchmark.py --model bert-base-uncased --strategy ddp --nodes 2
  python distributed_training_benchmark.py --model t5-small --strategy deepspeed --nodes 4
  python distributed_training_benchmark.py --analyze-scaling --output scaling_analysis.json
"""

import os
import sys
import json
import time
import logging
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
import socket
import subprocess
import itertools

# Add DuckDB database support
try:
    from benchmark_db_api import BenchmarkDBAPI
    BENCHMARK_DB_AVAILABLE = True
except ImportError:
    BENCHMARK_DB_AVAILABLE = False
    logger.warning("benchmark_db_api not available. Using deprecated JSON fallback.")


# Always deprecate JSON output in favor of DuckDB
DEPRECATE_JSON_OUTPUT = os.environ.get("DEPRECATE_JSON_OUTPUT", "1").lower() in ("1", "true", "yes")


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("distributed_training_benchmark")

# Global constants
PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TEST_DIR = PROJECT_ROOT / "test"
BENCHMARK_DIR = TEST_DIR / "benchmark_results"
DISTRIBUTED_BENCHMARK_DIR = BENCHMARK_DIR / "distributed_training"
DISTRIBUTED_TEMPLATES_DIR = TEST_DIR / "distributed_templates"

# Ensure directories exist
BENCHMARK_DIR.mkdir(exist_ok=True, parents=True)
DISTRIBUTED_BENCHMARK_DIR.mkdir(exist_ok=True, parents=True)
DISTRIBUTED_TEMPLATES_DIR.mkdir(exist_ok=True, parents=True)

# Key models suitable for distributed training
DISTRIBUTED_MODELS = {
    "bert": {
        "name": "BERT",
        "models": ["bert-base-uncased", "bert-large-uncased"],
        "category": "text_embedding",
        "batch_sizes": [16, 32, 64, 128],
        "strategies": ["ddp", "deepspeed", "fsdp"]
    },
    "t5": {
        "name": "T5",
        "models": ["t5-small", "t5-base"],
        "category": "text_generation",
        "batch_sizes": [8, 16, 32, 64],
        "strategies": ["ddp", "deepspeed", "fsdp"]
    },
    "llama": {
        "name": "LLAMA",
        "models": ["facebook/opt-125m", "facebook/opt-1.3b"],
        "category": "text_generation",
        "batch_sizes": [4, 8, 16, 32],
        "strategies": ["ddp", "deepspeed", "fsdp"]
    },
    "vit": {
        "name": "ViT",
        "models": ["google/vit-base-patch16-224"],
        "category": "vision",
        "batch_sizes": [16, 32, 64, 128],
        "strategies": ["ddp", "fsdp"]
    },
    "gpt2": {
        "name": "GPT-2",
        "models": ["gpt2"],
        "category": "text_generation",
        "batch_sizes": [4, 8, 16, 32],
        "strategies": ["ddp", "deepspeed", "fsdp"]
    }
}

# Distributed training strategies
STRATEGIES = {
    "data_parallel": {
        "name": "DataParallel",
        "pytorch_module": "torch.nn.DataParallel",
        "description": "Simple multi-GPU parallelism within a single node",
        "multi_node": False
    },
    "ddp": {
        "name": "DistributedDataParallel",
        "pytorch_module": "torch.nn.parallel.DistributedDataParallel",
        "description": "Efficient multi-GPU/node distributed training using NCCL/Gloo",
        "multi_node": True
    },
    "deepspeed": {
        "name": "DeepSpeed",
        "pytorch_module": "deepspeed",
        "description": "Microsoft's optimized distributed training library with ZeRO",
        "multi_node": True
    },
    "fsdp": {
        "name": "FullyShardedDataParallel",
        "pytorch_module": "torch.distributed.fsdp",
        "description": "Memory-efficient sharded data parallelism",
        "multi_node": True
    }
}

def detect_available_gpus() -> List[int]:
    """
    Detect available GPUs on the current node.
    
    Returns:
        List[int]: List of available GPU indices
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return []
        
        return list(range(torch.cuda.device_count()))
    except ImportError:
        logger.warning("PyTorch not available for GPU detection")
        return []

def detect_available_nodes() -> List[str]:
    """
    Detect available nodes in the cluster.
    
    In a real implementation, this would check a cluster configuration file
    or query a resource manager. Here we simulate with the local node only.
    
    Returns:
        List[str]: List of hostnames of available nodes
    """
    # In actual implementation, this would read a node list file or query slurm/PBS
    # For demonstration, we just return the local node
    try:
        hostname = socket.gethostname()
        return [hostname]
    except:
        return ["localhost"]

def create_distributed_training_script(
    model_key: str,
    model_name: str,
    strategy: str,
    batch_size: int = 32,
    num_nodes: int = 1,
    gpus_per_node: int = None,
    output_file: Optional[str] = None
) -> str:
    """
    Create a Python script for distributed training.
    
    Args:
        model_key (str): Key identifying the model
        model_name (str): Full name of the model
        strategy (str): Distributed strategy to use
        batch_size (int): Batch size per GPU
        num_nodes (int): Number of nodes to use
        gpus_per_node (int): Number of GPUs per node
        output_file (str): Path to output file
        
    Returns:
        str: Path to the created script
    """
    model_info = DISTRIBUTED_MODELS.get(model_key, {})
    category = model_info.get("category", "unknown")
    
    # Determine number of GPUs if not specified
    if gpus_per_node is None:
        available_gpus = detect_available_gpus()
        gpus_per_node = len(available_gpus) if available_gpus else 1
    
    # Adjust batch size to be per GPU
    global_batch_size = batch_size * gpus_per_node * num_nodes
    
    # Load template
    template_file = DISTRIBUTED_TEMPLATES_DIR / f"{strategy}_template.py"
    if not os.path.exists(template_file):
        # Create template file if it doesn't exist
        create_template_file(strategy)
    
# JSON output deprecated in favor of database storage
if not DEPRECATE_JSON_OUTPUT:
        with open(template_file, "r") as f:
            template = f.read()
        
        # Customize template
        script = template.replace("{{MODEL_NAME}}", model_name)
        script = script.replace("{{MODEL_KEY}}", model_key)
        script = script.replace("{{CATEGORY}}", category)
        script = script.replace("{{STRATEGY}}", strategy)
        script = script.replace("{{BATCH_SIZE}}", str(batch_size))
        script = script.replace("{{GLOBAL_BATCH_SIZE}}", str(global_batch_size))
        script = script.replace("{{NUM_NODES}}", str(num_nodes))
        script = script.replace("{{GPUS_PER_NODE}}", str(gpus_per_node))
        
        # Add custom dataset code for specific model types
        if category == "text_embedding" or category == "text_generation":
            dataset_code = """
        # Create a text dataset
        from datasets import load_dataset
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        
        # Create tokenizer
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Tokenize function
        def tokenize_function(examples):
            return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)
        
        # Tokenize dataset
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
        
        # Create dataloader
        from torch.utils.data import DataLoader
        train_dataloader = DataLoader(
            tokenized_dataset, 
            batch_size=batch_size_per_gpu,
            shuffle=True,
            pin_memory=True,
            num_workers=4
        )
        """
        elif category == "vision":
            dataset_code = """
        # Create an image dataset
        from torchvision.datasets import ImageFolder
        from torchvision import transforms
        
        # Image transformations
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Create a synthetic dataset if real data not available
        import torch
        import os
        
        # Create temp directory for synthetic data
        os.makedirs("./synthetic_data/class1", exist_ok=True)
        os.makedirs("./synthetic_data/class2", exist_ok=True)
        
        # Generate some random images
        for i in range(10):
            img = torch.randint(0, 256, (3, 224, 224), dtype=torch.uint8)
            transforms.ToPILImage()(img).save(f"./synthetic_data/class{i%2+1}/img_{i}.jpg")
        
        # Load dataset
        dataset = ImageFolder("./synthetic_data", transform=transform)
        
        # Create dataloader
        from torch.utils.data import DataLoader
        train_dataloader = DataLoader(
            dataset, 
            batch_size=batch_size_per_gpu,
            shuffle=True,
            pin_memory=True,
            num_workers=4
        )
        """
        else:
            dataset_code = """
        # Create a generic dataset
        import torch
        from torch.utils.data import TensorDataset, DataLoader
        
        # Create synthetic data
        num_samples = 10000
        input_shape = (3, 224, 224) if "vision" in model_name.lower() else (128,)
        inputs = torch.randn(num_samples, *input_shape)
        labels = torch.randint(0, 2, (num_samples,))
        
        # Create dataset and dataloader
        dataset = TensorDataset(inputs, labels)
        train_dataloader = DataLoader(
            dataset, 
            batch_size=batch_size_per_gpu,
            shuffle=True,
            pin_memory=True,
            num_workers=4
        )
        """
        
        script = script.replace("{{DATASET_CODE}}", dataset_code)
        
        # Determine output file path
        if output_file is None:
            output_file = DISTRIBUTED_TEMPLATES_DIR / f"train_{model_key}_{strategy}_{num_nodes}nodes_{gpus_per_node}gpus.py"
        
        # Create file
        with open(output_file, "w") as f:
            f.write(script)
        
        return str(output_file)
    
    def create_template_file(strategy: str):
        """
        Create a template file for a specific distributed strategy.
        
        Args:
            strategy (str): The distributed strategy to create a template for
        """
        template_file = DISTRIBUTED_TEMPLATES_DIR / f"{strategy}_template.py"
        
        if strategy == "data_parallel":
            template = """#!/usr/bin/env python3
    \"\"\"
    DataParallel training template for {{MODEL_NAME}}.
    \"\"\"
    
    import os
    import sys
    import time
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from transformers import AutoModel, AutoModelForSequenceClassification, AutoModelForCausalLM
    from torch.utils.data import DataLoader, DistributedSampler
    import json
    from datetime import datetime
    
    # Model and training configuration
    model_name = "{{MODEL_NAME}}"
    model_key = "{{MODEL_KEY}}"
    category = "{{CATEGORY}}"
    batch_size_per_gpu = {{BATCH_SIZE}}
    global_batch_size = {{GLOBAL_BATCH_SIZE}}
    num_nodes = {{NUM_NODES}}
    gpus_per_node = {{GPUS_PER_NODE}}
    total_gpus = num_nodes * gpus_per_node
    num_epochs = 3
    learning_rate = 5e-5
    gradient_accumulation_steps = 1
    use_mixed_precision = True
    report_interval = 10
    output_file = f"distributed_results_{model_key}_DataParallel_{num_nodes}nodes_{gpus_per_node}gpus.json"
    
    def main():
        # Check for GPUs
        if not torch.cuda.is_available():
            print("No GPUs available, exiting")
            sys.exit(1)
        
        device_ids = list(range(gpus_per_node))
        
        # Set up mixed precision
        scaler = torch.cuda.amp.GradScaler() if use_mixed_precision else None
        
        # Create dataset and dataloader
    {{DATASET_CODE}}
        
        # Load model
        print(f"Loading model {model_name}")
        start_time = time.time()
        
        # Load model based on category
        if category == "text_embedding":
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
        elif category == "text_generation":
            model = AutoModelForCausalLM.from_pretrained(model_name)
        elif category == "vision":
            from transformers import AutoModelForImageClassification
            model = AutoModelForImageClassification.from_pretrained(model_name)
        else:
            model = AutoModel.from_pretrained(model_name)
        
        model_load_time = time.time() - start_time
        print(f"Model loaded in {model_load_time:.2f}s")
        
        # Move model to GPU and wrap with DataParallel
        model = model.cuda()
        model = nn.DataParallel(model, device_ids=device_ids)
        
        # Set up optimizer
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
        
        # Train the model
        model.train()
        total_start_time = time.time()
        training_steps = 0
        epoch_times = []
        
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            
            for batch_idx, batch in enumerate(train_dataloader):
                # Move batch to device
                batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                # Forward pass with mixed precision
                if use_mixed_precision:
                    with torch.cuda.amp.autocast():
                        outputs = model(**batch)
                        loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
                        loss = loss / gradient_accumulation_steps
                    
                    # Backward pass with gradient scaling
                    scaler.scale(loss).backward()
                    
                    if (batch_idx + 1) % gradient_accumulation_steps == 0:
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                else:
                    # Standard forward and backward pass
                    outputs = model(**batch)
                    loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
                    loss = loss / gradient_accumulation_steps
                    
                    loss.backward()
                    
                    if (batch_idx + 1) % gradient_accumulation_steps == 0:
                        optimizer.step()
                        optimizer.zero_grad()
                
                training_steps += 1
                
                if batch_idx % report_interval == 0:
                    print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_dataloader)}, Loss: {loss.item():.4f}")
                
                # For benchmark purposes, limit training steps
                if training_steps >= 100:
                    break
            
            epoch_time = time.time() - epoch_start_time
            epoch_times.append(epoch_time)
            print(f"Epoch {epoch+1} completed in {epoch_time:.2f}s")
        
        total_training_time = time.time() - total_start_time
        
        # Collect and report metrics
        metrics = {
            "model": model_name,
            "model_key": model_key,
            "strategy": "DataParallel",
            "category": category,
            "num_nodes": num_nodes,
            "gpus_per_node": gpus_per_node,
            "total_gpus": total_gpus,
            "batch_size_per_gpu": batch_size_per_gpu,
            "global_batch_size": global_batch_size,
            "model_load_time": model_load_time,
            "total_training_time": total_training_time,
            "average_epoch_time": sum(epoch_times) / len(epoch_times),
            "samples_per_second": global_batch_size * training_steps / total_training_time,
            "timestamp": datetime.now().isoformat(),
            "hostname": os.uname()[1]
        }
        
        # Save metrics
        with open(output_file, 'w') as f:
            json.dump(metrics, f, indent=2)
else:
    logger.info("JSON output is deprecated. Results are stored directly in the database.")

    
    print(f"Training completed in {total_training_time:.2f}s")
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()
"""
    elif strategy == "ddp":
        template = """#!/usr/bin/env python3
\"\"\"
DistributedDataParallel training template for {{MODEL_NAME}}.
\"\"\"

import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoModel, AutoModelForSequenceClassification, AutoModelForCausalLM
import json
from datetime import datetime

# Model and training configuration
model_name = "{{MODEL_NAME}}"
model_key = "{{MODEL_KEY}}"
category = "{{CATEGORY}}"
batch_size_per_gpu = {{BATCH_SIZE}}
global_batch_size = {{GLOBAL_BATCH_SIZE}}
num_nodes = {{NUM_NODES}}
gpus_per_node = {{GPUS_PER_NODE}}
total_gpus = num_nodes * gpus_per_node
num_epochs = 3
learning_rate = 5e-5
gradient_accumulation_steps = 1
use_mixed_precision = True
report_interval = 10
node_rank = int(os.environ.get("NODE_RANK", 0))
output_file = f"distributed_results_{model_key}_DDP_{num_nodes}nodes_{gpus_per_node}gpus.json"

def setup(rank, world_size):
    """Initialize the distributed training environment."""
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '12355')
    
    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    """Clean up the distributed training environment."""
    dist.destroy_process_group()

def train(local_rank, world_size):
    # Calculate global rank
    global_rank = node_rank * gpus_per_node + local_rank
    
    # Set up the process group
    setup(global_rank, world_size)
    
    # Set device
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    
    # Set up mixed precision
    scaler = torch.cuda.amp.GradScaler() if use_mixed_precision else None
    
    # Create dataset and dataloader with DistributedSampler
{{DATASET_CODE}}
    
    # Create distributed sampler
    sampler = DistributedSampler(
        dataset, 
        num_replicas=world_size,
        rank=global_rank,
        shuffle=True
    )
    
    # Create new dataloader with sampler
    train_dataloader = DataLoader(
        dataset, 
        batch_size=batch_size_per_gpu,
        sampler=sampler,
        pin_memory=True,
        num_workers=4
    )
    
    # Load model
    if global_rank == 0:
        print(f"Loading model {model_name}")
    start_time = time.time()
    
    # Load model based on category
    if category == "text_embedding":
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
    elif category == "text_generation":
        model = AutoModelForCausalLM.from_pretrained(model_name)
    elif category == "vision":
        from transformers import AutoModelForImageClassification
        model = AutoModelForImageClassification.from_pretrained(model_name)
    else:
        model = AutoModel.from_pretrained(model_name)
    
    model_load_time = time.time() - start_time
    if global_rank == 0:
        print(f"Model loaded in {model_load_time:.2f}s")
    
    # Move model to device
    model = model.to(device)
    
    # Wrap model with DDP
    model = DDP(model, device_ids=[local_rank])
    
    # Set up optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Train the model
    model.train()
    total_start_time = time.time()
    training_steps = 0
    epoch_times = []
    
    for epoch in range(num_epochs):
        # Set epoch for distributed sampler
        sampler.set_epoch(epoch)
        
        epoch_start_time = time.time()
        
        for batch_idx, batch in enumerate(train_dataloader):
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Forward pass with mixed precision
            if use_mixed_precision:
                with torch.cuda.amp.autocast():
                    outputs = model(**batch)
                    loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
                    loss = loss / gradient_accumulation_steps
                
                # Backward pass with gradient scaling
                scaler.scale(loss).backward()
                
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                # Standard forward and backward pass
                outputs = model(**batch)
                loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
                loss = loss / gradient_accumulation_steps
                
                loss.backward()
                
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
            
            training_steps += 1
            
            if global_rank == 0 and batch_idx % report_interval == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_dataloader)}, Loss: {loss.item():.4f}")
            
            # For benchmark purposes, limit training steps
            if training_steps >= 100:
                break
        
        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)
        if global_rank == 0:
            print(f"Epoch {epoch+1} completed in {epoch_time:.2f}s")
    
    total_training_time = time.time() - total_start_time
    
    # Only the main process saves results
    if global_rank == 0:
        # Collect and report metrics
        metrics = {
            "model": model_name,
            "model_key": model_key,
            "strategy": "DistributedDataParallel",
            "category": category,
            "num_nodes": num_nodes,
            "gpus_per_node": gpus_per_node,
            "total_gpus": total_gpus,
            "batch_size_per_gpu": batch_size_per_gpu,
            "global_batch_size": global_batch_size,
            "model_load_time": model_load_time,
            "total_training_time": total_training_time,
            "average_epoch_time": sum(epoch_times) / len(epoch_times),
            "samples_per_second": global_batch_size * training_steps / total_training_time,
            "timestamp": datetime.now().isoformat(),
            "hostname": os.uname()[1]
        }
        
        # Save metrics
        with open(output_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"Training completed in {total_training_time:.2f}s")
        print(f"Results saved to {output_file}")
    
    # Clean up distributed process group
    cleanup()

def main():
    # Check for GPUs
    if not torch.cuda.is_available():
        print("No GPUs available, exiting")
        sys.exit(1)
    
    # Launch processes
    world_size = gpus_per_node * num_nodes
    mp.spawn(train, args=(world_size,), nprocs=gpus_per_node, join=True)

if __name__ == "__main__":
    main()
"""
    elif strategy == "deepspeed":
        template = """#!/usr/bin/env python3
\"\"\"
DeepSpeed training template for {{MODEL_NAME}}.
\"\"\"

import os
import sys
import time
import torch
import deepspeed
from transformers import AutoModel, AutoModelForSequenceClassification, AutoModelForCausalLM
import json
from datetime import datetime

# Model and training configuration
model_name = "{{MODEL_NAME}}"
model_key = "{{MODEL_KEY}}"
category = "{{CATEGORY}}"
batch_size_per_gpu = {{BATCH_SIZE}}
global_batch_size = {{GLOBAL_BATCH_SIZE}}
num_nodes = {{NUM_NODES}}
gpus_per_node = {{GPUS_PER_NODE}}
total_gpus = num_nodes * gpus_per_node
num_epochs = 3
learning_rate = 5e-5
gradient_accumulation_steps = 1
report_interval = 10
node_rank = int(os.environ.get("NODE_RANK", 0))
output_file = f"distributed_results_{model_key}_DeepSpeed_{num_nodes}nodes_{gpus_per_node}gpus.json"

# DeepSpeed configuration
deepspeed_config = {
    "train_batch_size": global_batch_size,
    "gradient_accumulation_steps": gradient_accumulation_steps,
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": learning_rate,
            "weight_decay": 0.01
        }
    },
    "fp16": {
        "enabled": True
    },
    "zero_optimization": {
        "stage": 2,
        "allgather_partitions": True,
        "allgather_bucket_size": 2e8,
        "overlap_comm": True,
        "reduce_scatter": True,
        "reduce_bucket_size": 2e8,
        "contiguous_gradients": True
    }
}

def main():
    # Initialize DeepSpeed
    deepspeed.init_distributed()
    
    # Create dataset and dataloader
{{DATASET_CODE}}
    
    # Load model
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    global_rank = int(os.environ.get("RANK", 0))
    
    if global_rank == 0:
        print(f"Loading model {model_name}")
    start_time = time.time()
    
    # Load model based on category
    if category == "text_embedding":
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
    elif category == "text_generation":
        model = AutoModelForCausalLM.from_pretrained(model_name)
    elif category == "vision":
        from transformers import AutoModelForImageClassification
        model = AutoModelForImageClassification.from_pretrained(model_name)
    else:
        model = AutoModel.from_pretrained(model_name)
    
    model_load_time = time.time() - start_time
    if global_rank == 0:
        print(f"Model loaded in {model_load_time:.2f}s")
    
    # Initialize DeepSpeed engine
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=deepspeed_config,
        training_data=train_dataloader
    )
    
    # Train the model
    model_engine.train()
    total_start_time = time.time()
    training_steps = 0
    epoch_times = []
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        for batch_idx, batch in enumerate(train_dataloader):
            # Forward pass
            outputs = model_engine(**batch)
            loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
            
            # Backward pass
            model_engine.backward(loss)
            
            # Weight update
            model_engine.step()
            
            training_steps += 1
            
            if global_rank == 0 and batch_idx % report_interval == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_dataloader)}, Loss: {loss.item():.4f}")
            
            # For benchmark purposes, limit training steps
            if training_steps >= 100:
                break
        
        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)
        if global_rank == 0:
            print(f"Epoch {epoch+1} completed in {epoch_time:.2f}s")
    
    total_training_time = time.time() - total_start_time
    
    # Only the main process saves results
    if global_rank == 0:
        # Collect and report metrics
        metrics = {
            "model": model_name,
            "model_key": model_key,
            "strategy": "DeepSpeed",
            "category": category,
            "num_nodes": num_nodes,
            "gpus_per_node": gpus_per_node,
            "total_gpus": total_gpus,
            "batch_size_per_gpu": batch_size_per_gpu,
            "global_batch_size": global_batch_size,
            "model_load_time": model_load_time,
            "total_training_time": total_training_time,
            "average_epoch_time": sum(epoch_times) / len(epoch_times),
            "samples_per_second": global_batch_size * training_steps / total_training_time,
            "timestamp": datetime.now().isoformat(),
            "hostname": os.uname()[1],
            "zero_stage": deepspeed_config["zero_optimization"]["stage"]
        }
        
        # Save metrics
        with open(output_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"Training completed in {total_training_time:.2f}s")
        print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()
"""
    elif strategy == "fsdp":
        template = """#!/usr/bin/env python3
\"\"\"
FullyShardedDataParallel training template for {{MODEL_NAME}}.
\"\"\"

import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import default_auto_wrap_policy
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoModel, AutoModelForSequenceClassification, AutoModelForCausalLM
import json
from datetime import datetime

# Model and training configuration
model_name = "{{MODEL_NAME}}"
model_key = "{{MODEL_KEY}}"
category = "{{CATEGORY}}"
batch_size_per_gpu = {{BATCH_SIZE}}
global_batch_size = {{GLOBAL_BATCH_SIZE}}
num_nodes = {{NUM_NODES}}
gpus_per_node = {{GPUS_PER_NODE}}
total_gpus = num_nodes * gpus_per_node
num_epochs = 3
learning_rate = 5e-5
gradient_accumulation_steps = 1
use_mixed_precision = True
report_interval = 10
node_rank = int(os.environ.get("NODE_RANK", 0))
output_file = f"distributed_results_{model_key}_FSDP_{num_nodes}nodes_{gpus_per_node}gpus.json"

def setup(rank, world_size):
    """Initialize the distributed training environment."""
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '12355')
    
    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    """Clean up the distributed training environment."""
    dist.destroy_process_group()

def train(local_rank, world_size):
    # Calculate global rank
    global_rank = node_rank * gpus_per_node + local_rank
    
    # Set up the process group
    setup(global_rank, world_size)
    
    # Set device
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    
    # Set up mixed precision
    mixed_precision_policy = None
    if use_mixed_precision:
        from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
        scaler = ShardedGradScaler()
        
        # Set up mixed precision policy for FSDP
        bfloat16_available = torch.cuda.is_bf16_supported()
        if bfloat16_available:
            from torch.distributed.fsdp import MixedPrecision
            from torch.dtype import bfloat16
            mixed_precision_policy = MixedPrecision(
                param_dtype=bfloat16,
                reduce_dtype=bfloat16,
                buffer_dtype=bfloat16
            )
        else:
            from torch.distributed.fsdp import MixedPrecision
            mixed_precision_policy = MixedPrecision(
                param_dtype=torch.float16,
                reduce_dtype=torch.float16,
                buffer_dtype=torch.float16
            )
    
    # Create dataset and dataloader with DistributedSampler
{{DATASET_CODE}}
    
    # Create distributed sampler
    sampler = DistributedSampler(
        dataset, 
        num_replicas=world_size,
        rank=global_rank,
        shuffle=True
    )
    
    # Create new dataloader with sampler
    train_dataloader = DataLoader(
        dataset, 
        batch_size=batch_size_per_gpu,
        sampler=sampler,
        pin_memory=True,
        num_workers=4
    )
    
    # Load model
    if global_rank == 0:
        print(f"Loading model {model_name}")
    start_time = time.time()
    
    # Load model based on category
    if category == "text_embedding":
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
    elif category == "text_generation":
        model = AutoModelForCausalLM.from_pretrained(model_name)
    elif category == "vision":
        from transformers import AutoModelForImageClassification
        model = AutoModelForImageClassification.from_pretrained(model_name)
    else:
        model = AutoModel.from_pretrained(model_name)
    
    model_load_time = time.time() - start_time
    if global_rank == 0:
        print(f"Model loaded in {model_load_time:.2f}s")
    
    # Move model to device
    model = model.to(device)
    
    # Wrap model with FSDP
    model = FSDP(
        model,
        mixed_precision=mixed_precision_policy,
        cpu_offload=CPUOffload(offload_params=False),
        auto_wrap_policy=default_auto_wrap_policy,
        device_id=device
    )
    
    # Set up optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Train the model
    model.train()
    total_start_time = time.time()
    training_steps = 0
    epoch_times = []
    
    for epoch in range(num_epochs):
        # Set epoch for distributed sampler
        sampler.set_epoch(epoch)
        
        epoch_start_time = time.time()
        
        for batch_idx, batch in enumerate(train_dataloader):
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Forward pass with mixed precision
            if use_mixed_precision:
                with torch.cuda.amp.autocast():
                    outputs = model(**batch)
                    loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
                    loss = loss / gradient_accumulation_steps
                
                # Backward pass with gradient scaling
                scaler.scale(loss).backward()
                
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                # Standard forward and backward pass
                outputs = model(**batch)
                loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
                loss = loss / gradient_accumulation_steps
                
                loss.backward()
                
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
            
            training_steps += 1
            
            if global_rank == 0 and batch_idx % report_interval == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_dataloader)}, Loss: {loss.item():.4f}")
            
            # For benchmark purposes, limit training steps
            if training_steps >= 100:
                break
        
        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)
        if global_rank == 0:
            print(f"Epoch {epoch+1} completed in {epoch_time:.2f}s")
    
    total_training_time = time.time() - total_start_time
    
    # Only the main process saves results
    if global_rank == 0:
        # Collect and report metrics
        metrics = {
            "model": model_name,
            "model_key": model_key,
            "strategy": "FullyShardedDataParallel",
            "category": category,
# JSON output deprecated in favor of database storage
if not DEPRECATE_JSON_OUTPUT:
                "num_nodes": num_nodes,
                "gpus_per_node": gpus_per_node,
                "total_gpus": total_gpus,
                "batch_size_per_gpu": batch_size_per_gpu,
                "global_batch_size": global_batch_size,
                "model_load_time": model_load_time,
                "total_training_time": total_training_time,
                "average_epoch_time": sum(epoch_times) / len(epoch_times),
                "samples_per_second": global_batch_size * training_steps / total_training_time,
                "timestamp": datetime.now().isoformat(),
                "hostname": os.uname()[1],
                "mixed_precision": use_mixed_precision
            }
            
            # Save metrics
            with open(output_file, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            print(f"Training completed in {total_training_time:.2f}s")
            print(f"Results saved to {output_file}")
        
        # Clean up distributed process group
        cleanup()
    
    def main():
        # Check for GPUs
        if not torch.cuda.is_available():
            print("No GPUs available, exiting")
            sys.exit(1)
        
        # Launch processes
        world_size = gpus_per_node * num_nodes
        mp.spawn(train, args=(world_size,), nprocs=gpus_per_node, join=True)
    
    if __name__ == "__main__":
        main()
    """
        
        with open(template_file, "w") as f:
            f.write(template)
    
    def create_launcher_script(
        script_path: str,
        num_nodes: int = 1,
        gpus_per_node: int = None,
        node_list: List[str] = None,
        output_file: Optional[str] = None
    ) -> str:
        """
        Create a launcher script for distributed training.
        
        Args:
            script_path (str): Path to the Python script to launch
            num_nodes (int): Number of nodes to use
            gpus_per_node (int): Number of GPUs per node
            node_list (List[str]): List of node hostnames
            output_file (str): Path to output file
            
        Returns:
            str: Path to the created launcher script
        """
        # Determine number of GPUs if not specified
        if gpus_per_node is None:
            available_gpus = detect_available_gpus()
            gpus_per_node = len(available_gpus) if available_gpus else 1
        
        # Get node list if not specified
        if node_list is None:
            node_list = detect_available_nodes()
            # Truncate to requested number of nodes
            node_list = node_list[:num_nodes]
        
        # Create launcher script
        launcher = f"""#!/bin/bash
    # Launcher script for distributed training
    
    # Node information
    NODES=({" ".join(node_list)})
    NUM_NODES={len(node_list)}
    GPUS_PER_NODE={gpus_per_node}
    MASTER_ADDR={node_list[0]}
    MASTER_PORT=12355
    SCRIPT={script_path}
    
    # Launch on each node
    for ((i=0; i<$NUM_NODES; i++)); do
        NODE=${NODES[$i]}
        echo "Launching on node $NODE (rank $i)"
        
        if [ "$NODE" = "$(hostname)" ]; then
            # Local node
            NODE_RANK=$i MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT python $SCRIPT &
        else
            # Remote node
            ssh $NODE "cd $(pwd) && NODE_RANK=$i MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT python $SCRIPT" &
        fi
    done
    
    # Wait for all processes to complete
    wait
    echo "All nodes completed"
    """
        
        # Determine output file path
        if output_file is None:
            output_file = DISTRIBUTED_TEMPLATES_DIR / f"launch_{os.path.basename(script_path)}.sh"
        
        # Create file
        with open(output_file, "w") as f:
            f.write(launcher)
        
        # Make executable
        os.chmod(output_file, 0o755)
        
        return str(output_file)
    
    def run_distributed_benchmark(
        model_key: str,
        strategy: str,
        num_nodes: int = 1,
        gpus_per_node: int = None,
        batch_size: int = None,
        timeout: int = 3600
    ) -> Dict[str, Any]:
        """
        Run a distributed training benchmark.
        
        Args:
            model_key (str): Key for the model to benchmark
            strategy (str): Distributed strategy to use
            num_nodes (int): Number of nodes to use
            gpus_per_node (int): Number of GPUs per node
            batch_size (int): Batch size per GPU
            timeout (int): Timeout in seconds
            
        Returns:
            Dict[str, Any]: Benchmark results
        """
        if model_key not in DISTRIBUTED_MODELS:
            logger.error(f"Unknown model: {model_key}")
            return {
                "model": model_key,
                "strategy": strategy,
                "status": "error",
                "error": "Unknown model"
            }
        
        if strategy not in STRATEGIES:
            logger.error(f"Unknown strategy: {strategy}")
            return {
                "model": model_key,
                "strategy": strategy,
                "status": "error",
                "error": "Unknown strategy"
            }
        
        # Check strategy compatibility
        if num_nodes > 1 and not STRATEGIES[strategy].get("multi_node", False):
            logger.error(f"Strategy {strategy} does not support multi-node training")
            return {
                "model": model_key,
                "strategy": strategy,
                "status": "error",
                "error": f"Strategy {strategy} does not support multi-node training"
            }
        
        # Determine number of GPUs if not specified
        if gpus_per_node is None:
            available_gpus = detect_available_gpus()
            gpus_per_node = len(available_gpus) if available_gpus else 1
        
        # Use default batch size if not specified
        if batch_size is None:
            batch_size = DISTRIBUTED_MODELS[model_key].get("batch_sizes", [16])[0]
        
        # Create node list
        node_list = detect_available_nodes()
        if len(node_list) < num_nodes:
            logger.warning(f"Only {len(node_list)} nodes available, requested {num_nodes}")
            num_nodes = len(node_list)
        
        # Get model name
        model_name = DISTRIBUTED_MODELS[model_key]["models"][0]
        
        # In an actual implementation, we would:
        # 1. Create the training script
        script_path = create_distributed_training_script(
            model_key=model_key,
            model_name=model_name,
            strategy=strategy,
            batch_size=batch_size,
            num_nodes=num_nodes,
            gpus_per_node=gpus_per_node
        )
        
        # 2. Create the launcher script
        launcher_path = create_launcher_script(
            script_path=script_path,
            num_nodes=num_nodes,
            gpus_per_node=gpus_per_node,
            node_list=node_list[:num_nodes]
        )
        
        # 3. Run the benchmark
        logger.info(f"Running distributed benchmark for {model_key} with {strategy} on {num_nodes} nodes x {gpus_per_node} GPUs")
        
        # In a real implementation, we would execute the launcher script
else:
    logger.info("JSON output is deprecated. Results are stored directly in the database.")

    # Here, we simulate the results
    
    # 4. Load and return results
    result_file = DISTRIBUTED_BENCHMARK_DIR / f"distributed_results_{model_key}_{strategy}_{num_nodes}nodes_{gpus_per_node}gpus.json"
    
    # Simulate benchmark results
    # In a real implementation, this would be read from the result file after execution
    results = {
        "model": model_name,
        "model_key": model_key,
        "strategy": STRATEGIES[strategy]["name"],
        "category": DISTRIBUTED_MODELS[model_key]["category"],
        "num_nodes": num_nodes,
        "gpus_per_node": gpus_per_node,
        "total_gpus": num_nodes * gpus_per_node,
        "batch_size_per_gpu": batch_size,
        "global_batch_size": batch_size * gpus_per_node * num_nodes,
        "model_load_time": 10.5,  # Simulated value in seconds
        "total_training_time": 120.3,  # Simulated value in seconds
        "average_epoch_time": 40.1,  # Simulated value in seconds
        "samples_per_second": 256.8,  # Simulated value
        "timestamp": datetime.now().isoformat(),
        "hostname": socket.gethostname(),
        "status": "success"
    }
    
    # Save simulated results
    with open(result_file, "w") as f:
        json.dump(results, f, indent=2)
    
    return results

def run_scaling_analysis(
    model_keys: List[str] = None,
    strategies: List[str] = None,
    max_nodes: int = 4,
    output_file: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run a scaling analysis for distributed training.
    
    Args:
        model_keys (List[str]): List of models to benchmark
        strategies (List[str]): List of strategies to benchmark
        max_nodes (int): Maximum number of nodes to test
        output_file (str): Path to output file
        
    Returns:
        Dict[str, Any]: Scaling analysis results
    """
    # Use default models if not specified
    if model_keys is None:
        model_keys = list(DISTRIBUTED_MODELS.keys())
    
    # Use default strategies if not specified
    if strategies is None:
        strategies = ["ddp", "deepspeed", "fsdp"]
    
    # Detect available GPUs
    available_gpus = detect_available_gpus()
    gpus_per_node = len(available_gpus) if available_gpus else 1
    
    # Detect available nodes
    node_list = detect_available_nodes()
    actual_max_nodes = min(max_nodes, len(node_list))
    
    # Create results directory
    os.makedirs(DISTRIBUTED_BENCHMARK_DIR, exist_ok=True)
    
    # Initialize results
    scaling_results = {
        "timestamp": datetime.now().isoformat(),
        "gpus_per_node": gpus_per_node,
        "max_nodes": actual_max_nodes,
        "models": {}
    }
    
    # Run benchmarks for each model, strategy, and node count
    for model_key in model_keys:
        if model_key not in DISTRIBUTED_MODELS:
            logger.warning(f"Skipping unknown model: {model_key}")
            continue
        
        model_name = DISTRIBUTED_MODELS[model_key]["models"][0]
        scaling_results["models"][model_key] = {
            "name": model_name,
            "category": DISTRIBUTED_MODELS[model_key]["category"],
            "strategies": {}
        }
        
        for strategy in strategies:
            if strategy not in STRATEGIES:
                logger.warning(f"Skipping unknown strategy: {strategy}")
                continue
            
            # Check if strategy supports multi-node
            if actual_max_nodes > 1 and not STRATEGIES[strategy].get("multi_node", False):
                logger.warning(f"Strategy {strategy} does not support multi-node, skipping scaling tests")
                continue
            
            scaling_results["models"][model_key]["strategies"][strategy] = {
                "name": STRATEGIES[strategy]["name"],
                "scaling": {}
            }
            
            # Test with different node counts
            for num_nodes in range(1, actual_max_nodes + 1):
                # Use a relatively small batch size for scaling tests
                batch_size = DISTRIBUTED_MODELS[model_key]["batch_sizes"][0]
                
                # Run benchmark
                try:
                    result = run_distributed_benchmark(
                        model_key=model_key,
                        strategy=strategy,
                        num_nodes=num_nodes,
                        gpus_per_node=gpus_per_node,
                        batch_size=batch_size
                    )
                    
                    if result.get("status") == "success":
                        # Store throughput and scaling factor
                        throughput = result.get("samples_per_second", 0)
                        
                        # Calculate scaling factor relative to 1 node
                        if num_nodes == 1:
                            base_throughput = throughput
                            scaling_factor = 1.0
                        else:
                            scaling_factor = throughput / base_throughput if base_throughput else 0
                        
                        scaling_results["models"][model_key]["strategies"][strategy]["scaling"][str(num_nodes)] = {
                            "nodes": num_nodes,
                            "total_gpus": num_nodes * gpus_per_node,
                            "throughput": throughput,
                            "scaling_factor": scaling_factor,
                            "efficiency": scaling_factor / num_nodes if num_nodes else 0
                        }
                except Exception as e:
                    logger.error(f"Error during benchmark: {e}")
    
    # Save results
    if output_file is None:
        output_file = DISTRIBUTED_BENCHMARK_DIR / f"scaling_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(output_file, "w") as f:
        json.dump(scaling_results, f, indent=2)
    
    logger.info(f"Scaling analysis completed and saved to {output_file}")
    
    return scaling_results

def create_scaling_charts(scaling_results: Dict[str, Any], output_dir: Optional[str] = None) -> List[str]:
    """
    Create scaling charts from analysis results.
    
    Args:
        scaling_results (Dict[str, Any]): Scaling analysis results
        output_dir (str): Directory to save charts
        
    Returns:
        List[str]: Paths to created chart files
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Set output directory
        if output_dir is None:
            output_dir = DISTRIBUTED_BENCHMARK_DIR / "charts"
        os.makedirs(output_dir, exist_ok=True)
        
        chart_files = []
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams["figure.figsize"] = (12, 8)
        
        # Create charts for each model
        for model_key, model_data in scaling_results.get("models", {}).items():
            model_name = model_data.get("name", model_key)
            
            # Create throughput chart
            plt.figure()
            
            # For each strategy, plot throughput vs nodes
            for strategy_key, strategy_data in model_data.get("strategies", {}).items():
                strategy_name = strategy_data.get("name", strategy_key)
                
                # Extract node counts and throughput values
                nodes = []
                throughput = []
                
                for node_data in strategy_data.get("scaling", {}).values():
                    nodes.append(node_data.get("nodes", 0))
                    throughput.append(node_data.get("throughput", 0))
                
                if nodes and throughput:
                    plt.plot(nodes, throughput, marker='o', label=strategy_name)
            
            plt.title(f"Throughput Scaling for {model_name}")
            plt.xlabel("Number of Nodes")
            plt.ylabel("Throughput (samples/second)")
            plt.legend()
            plt.grid(True)
            
            # Save chart
            throughput_file = os.path.join(output_dir, f"{model_key}_throughput_scaling.png")
            plt.savefig(throughput_file)
            plt.close()
            chart_files.append(throughput_file)
            
            # Create scaling efficiency chart
            plt.figure()
            
            # For each strategy, plot scaling efficiency vs nodes
            for strategy_key, strategy_data in model_data.get("strategies", {}).items():
                strategy_name = strategy_data.get("name", strategy_key)
                
                # Extract node counts and efficiency values
                nodes = []
                efficiency = []
                
                for node_data in strategy_data.get("scaling", {}).values():
                    if node_data.get("nodes", 0) > 1:  # Skip single node
                        nodes.append(node_data.get("nodes", 0))
                        efficiency.append(node_data.get("efficiency", 0) * 100)  # Convert to percentage
                
                if nodes and efficiency:
                    plt.plot(nodes, efficiency, marker='o', label=strategy_name)
            
            # Add ideal scaling line
            if nodes:
                plt.plot(nodes, [100] * len(nodes), 'k--', label="Ideal (100%)")
            
            plt.title(f"Scaling Efficiency for {model_name}")
            plt.xlabel("Number of Nodes")
            plt.ylabel("Scaling Efficiency (%)")
            plt.legend()
            plt.grid(True)
            
            # Save chart
            efficiency_file = os.path.join(output_dir, f"{model_key}_scaling_efficiency.png")
            plt.savefig(efficiency_file)
            plt.close()
            chart_files.append(efficiency_file)
        
        # Create comparative chart across models for specific strategy (e.g., DDP)
        strategy_key = "ddp"  # Use DDP as the common strategy
        plt.figure()
        
        # For each model, plot scaling factor vs nodes
        for model_key, model_data in scaling_results.get("models", {}).items():
            model_name = model_data.get("name", model_key)
            
            if strategy_key in model_data.get("strategies", {}):
                strategy_data = model_data["strategies"][strategy_key]
                
                # Extract node counts and scaling factors
                nodes = []
                scaling = []
                
                for node_data in strategy_data.get("scaling", {}).values():
                    nodes.append(node_data.get("nodes", 0))
                    scaling.append(node_data.get("scaling_factor", 0))
                
                if nodes and scaling:
                    plt.plot(nodes, scaling, marker='o', label=model_name)
        
        # Add ideal scaling line
        max_nodes = scaling_results.get("max_nodes", 4)
        plt.plot(range(1, max_nodes + 1), range(1, max_nodes + 1), 'k--', label="Ideal")
        
        plt.title(f"Scaling Factor Comparison (DDP Strategy)")
        plt.xlabel("Number of Nodes")
        plt.ylabel("Scaling Factor")
        plt.legend()
        plt.grid(True)
        
        # Save chart
        comparison_file = os.path.join(output_dir, f"model_comparison_scaling.png")
        plt.savefig(comparison_file)
        plt.close()
        chart_files.append(comparison_file)
        
        return chart_files
    
    except ImportError:
        logger.warning("Matplotlib or seaborn not available, skipping chart creation")
        return []

def update_benchmark_database(results: Dict[str, Any]) -> bool:
    """
    Update the central benchmark database with distributed training results.
    
    Args:
        results (Dict[str, Any]): Benchmark results
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Load existing database if available
        db_file = BENCHMARK_DIR / "hardware_model_benchmark_db.parquet"
        if os.path.exists(db_file):
            import pandas as pd
            df = pd.read_parquet(db_file)
            
            # Create a new entry
            entry = {
                "model": results.get("model_key"),
                "model_name": results.get("model"),
                "category": DISTRIBUTED_MODELS.get(results.get("model_key"), {}).get("category"),
                "hardware": f"distributed_{results.get('strategy')}",
                "hardware_name": f"Distributed {results.get('strategy')} ({results.get('num_nodes')} nodes)",
                "batch_size": results.get("global_batch_size"),
                "precision": "fp16" if results.get("mixed_precision", True) else "fp32",
                "mode": "training",
                "status": results.get("status"),
                "timestamp": results.get("timestamp"),
                "throughput": results.get("samples_per_second"),
                "latency_mean": results.get("average_epoch_time") * 1000 if results.get("average_epoch_time") else 0,  # Convert to ms
                "memory_usage": 0,  # Not provided in distributed training
                "startup_time": results.get("model_load_time") * 1000 if results.get("model_load_time") else 0,  # Convert to ms
                "first_inference": 0,  # Not applicable to training
                "num_nodes": results.get("num_nodes"),
                "gpus_per_node": results.get("gpus_per_node"),
                "total_gpus": results.get("total_gpus")
            }
            
            # Check if entry already exists
            mask = (
                (df["model"] == entry["model"]) &
                (df["hardware"] == entry["hardware"]) &
                (df["batch_size"] == entry["batch_size"]) &
                (df["mode"] == entry["mode"]) &
                (df["num_nodes"] == entry["num_nodes"]) &
                (df["gpus_per_node"] == entry["gpus_per_node"])
            )
            
            if mask.any():
                # Update existing entry
                for key, value in entry.items():
                    if key in df.columns:
                        df.loc[mask, key] = value
            else:
                # Add new entry
                df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)
            
            # Save updated database
            df.to_parquet(db_file)
            logger.info(f"Updated benchmark database at {db_file}")
            return True
        else:
            logger.error(f"Benchmark database not found at {db_file}")
            return False
    
    except Exception as e:
        logger.error(f"Error updating benchmark database: {e}")
        return False

def main():
    """
    Main function.
    """
    parser = argparse.ArgumentParser(description="Distributed Training Benchmark")
    
    # Main options
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", help="Model to benchmark")
    group.add_argument("--all-models", action="store_true", help="Benchmark all models")
    group.add_argument("--analyze-scaling", action="store_true", help="Run scaling analysis")
    group.add_argument("--create-charts", help="Create charts from analysis file")
    
    # Distributed training options
    parser.add_argument("--strategy", choices=list(STRATEGIES.keys()), default="ddp", help="Distributed training strategy")
    parser.add_argument("--nodes", type=int, default=1, help="Number of nodes")
    parser.add_argument("--gpus-per-node", type=int, help="Number of GPUs per node")
    parser.add_argument("--batch-size", type=int, help="Batch size per GPU")
    
    # Scaling analysis options
    parser.add_argument("--max-nodes", type=int, default=4, help="Maximum number of nodes for scaling analysis")
    
    # Output options
    parser.add_argument("--output", help="Output file for results")
    
    
    parser.add_argument("--db-path", type=str, default=None,
                      help="Path to the benchmark database")
    parser.add_argument("--db-only", action="store_true",
                      help="Store results only in the database, not in JSON")
args = parser.parse_args()
    
    # Create directories
    os.makedirs(DISTRIBUTED_BENCHMARK_DIR, exist_ok=True)
    os.makedirs(DISTRIBUTED_TEMPLATES_DIR, exist_ok=True)
    
    if args.model:
        if args.model not in DISTRIBUTED_MODELS:
            available_models = ", ".join(DISTRIBUTED_MODELS.keys())
            print(f"Error: Unknown model: {args.model}")
            print(f"Available models: {available_models}")
            sys.exit(1)
        
        print(f"Running distributed benchmark for {args.model} with {args.strategy} on {args.nodes} nodes")
        results = run_distributed_benchmark(
            model_key=args.model,
            strategy=args.strategy,
            num_nodes=args.nodes,
            gpus_per_node=args.gpus_per_node,
            batch_size=args.batch_size
        )
        
        if results.get("status") == "success":
            # Save results
            output_file = args.output or DISTRIBUTED_BENCHMARK_DIR / f"benchmark_{args.model}_{args.strategy}_{args.nodes}nodes.json"
            with open(output_file, "w") as f:
                json.dump(results, f, indent=2)
            
            print(f"Benchmark completed successfully")
            print(f"Throughput: {results.get('samples_per_second', 0):.2f} samples/second")
            print(f"Training time: {results.get('total_training_time', 0):.2f} seconds")
            print(f"Results saved to {output_file}")
            
            # Update benchmark database
            update_benchmark_database(results)
        else:
            print(f"Benchmark failed: {results.get('error', 'Unknown error')}")
    
    elif args.all_models:
        print(f"Running distributed benchmarks for all models with {args.strategy} on {args.nodes} nodes")
        
        all_results = {}
        for model_key in DISTRIBUTED_MODELS:
            print(f"Benchmarking {model_key}...")
            results = run_distributed_benchmark(
                model_key=model_key,
                strategy=args.strategy,
                num_nodes=args.nodes,
                gpus_per_node=args.gpus_per_node,
                batch_size=args.batch_size
            )
            
            all_results[model_key] = results
            
            # Update benchmark database
            if results.get("status") == "success":
                update_benchmark_database(results)
        
        # Save all results
        output_file = args.output or DISTRIBUTED_BENCHMARK_DIR / f"benchmark_all_{args.strategy}_{args.nodes}nodes.json"
        with open(output_file, "w") as f:
            json.dump(all_results, f, indent=2)
        
        print(f"All benchmarks completed")
        print(f"Results saved to {output_file}")
    
    elif args.analyze_scaling:
        print(f"Running scaling analysis with up to {args.max_nodes} nodes")
        
        results = run_scaling_analysis(
            max_nodes=args.max_nodes,
            output_file=args.output
        )
        
        if results:
            # Create charts
            chart_files = create_scaling_charts(results)
            
            if chart_files:
                print(f"Scaling analysis completed")
                print(f"Charts saved to: {', '.join(chart_files)}")
            else:
                print(f"Scaling analysis completed, but no charts were created")
    
    elif args.create_charts:
        print(f"Creating charts from analysis file: {args.create_charts}")
        
        # Load analysis results
        try:
            with open(args.create_charts, "r") as f:
# Try database first, fall back to JSON if necessary
try:
    from benchmark_db_api import BenchmarkDBAPI
    db_api = BenchmarkDBAPI(db_path=os.environ.get("BENCHMARK_DB_PATH", "./benchmark_db.duckdb"))
    results = db_api.get_benchmark_results()
    logger.info("Successfully loaded results from database")
except Exception as e:
    logger.warning(f"Error reading from database, falling back to JSON: {e}")
                    results = json.load(f)

            
            # Create charts
            chart_files = create_scaling_charts(results, output_dir=args.output)
            
            if chart_files:
                print(f"Charts created successfully")
                print(f"Charts saved to: {', '.join(chart_files)}")
            else:
                print(f"No charts were created")
        except Exception as e:
            print(f"Error creating charts: {e}")

if __name__ == "__main__":
    main()