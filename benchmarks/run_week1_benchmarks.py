#!/usr/bin/env python3
"""
Training Benchmark Runner for the IPFS Accelerate Python Framework.

This script generates distributed training benchmark configurations for different models
and hardware platforms, leveraging the hardware selection system to make optimal choices.

Usage:
  python run_training_benchmark.py --model bert-base-uncased --hardware cuda
  python run_training_benchmark.py --model t5-small --distributed --max-gpus 4 --output t5_benchmark_config.json
  python run_training_benchmark.py --list-models
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

# Add DuckDB database support
try:
    from benchmark_db_api import BenchmarkDBAPI
    BENCHMARK_DB_AVAILABLE = True
except ImportError:
    BENCHMARK_DB_AVAILABLE = False
    logger.warning("benchmark_db_api not available. Using deprecated JSON fallback.")


# Always deprecate JSON output in favor of DuckDB
DEPRECATE_JSON_OUTPUT = os.environ.get("DEPRECATE_JSON_OUTPUT", "1").lower() in ("1", "true", "yes")


from hardware_selector import HardwareSelector


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("training_benchmark")

# Define sample models for different model families
SAMPLE_MODELS = {
    "embedding": [
        "bert-base-uncased",
        "distilbert-base-uncased",
        "sentence-transformers/all-MiniLM-L6-v2",
        "roberta-base"
    ],
    "text_generation": [
        "t5-small",
        "facebook/opt-125m",
        "EleutherAI/pythia-70m",
        "google/gemma-2b"
    ],
    "vision": [
        "google/vit-base-patch16-224",
        "facebook/convnext-tiny-224",
        "microsoft/resnet-50",
        "microsoft/dit-base-finetuned-rvlcdip"
    ],
    "audio": [
        "openai/whisper-tiny",
        "facebook/wav2vec2-base-960h",
        "microsoft/wavlm-base",
        "facebook/hubert-base-ls960"
    ],
    "multimodal": [
        "openai/clip-vit-base-patch32",
        "Salesforce/blip-image-captioning-base",
        "microsoft/git-base",
        "facebook/flava-full"
    ]
}


def list_sample_models() -> None:
    """
    Print a list of sample models for each model family.
    """
    print("Sample Models for Benchmarking:")
    print("-------------------------------")
    for family, models in SAMPLE_MODELS.items():
        print(f"\n{family.upper()} MODELS:")
        for model in models:
            print(f"  - {model}")


def run_training_benchmark(
    model_name: str,
    output_dir: str = "./benchmark_results",
    output_file: Optional[str] = None,
    available_hardware: Optional[List[str]] = None,
    include_distributed: bool = False,
    max_gpus: int = 8,
    mode: str = "training"
) -> Dict[str, Any]:
    """
    Run a training benchmark configuration for a specific model.
    
    Args:
        model_name (str): Name of the model to benchmark.
        output_dir (str): Directory to store benchmark results.
        output_file (Optional[str]): Path to save benchmark configuration.
        available_hardware (Optional[List[str]]): List of available hardware platforms.
        include_distributed (bool): Whether to include distributed configurations.
        max_gpus (int): Maximum number of GPUs to consider for distributed training.
        mode (str): "training" or "inference".
        
    Returns:
        Dict[str, Any]: Benchmark configuration
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create hardware selector
    selector = HardwareSelector(database_path=output_dir)
    
    # Generate benchmark configuration
    logger.info(f"Generating benchmark configuration for {model_name}...")
    benchmark_config = selector.select_hardware_for_training_benchmark(
        model_name=model_name,
        mode=mode,
        available_hardware=available_hardware,
        include_distributed=include_distributed,
        max_gpus=max_gpus
    )
    
    # Add metadata
    benchmark_config["timestamp"] = datetime.now().isoformat()
    benchmark_config["parameters"] = {
        "model_name": model_name,
        "available_hardware": available_hardware,
        "include_distributed": include_distributed,
        "max_gpus": max_gpus,
        "mode": mode
    }
    
    # Save configuration if output file is specified
    if output_file:
        output_path = Path(output_file)
# JSON output deprecated in favor of database storage
if not DEPRECATE_JSON_OUTPUT:
            with open(output_path, "w") as f:
                json.dump(benchmark_config, f, indent=2)
else:
    logger.info("JSON output is deprecated. Results are stored directly in the database.")

        logger.info(f"Benchmark configuration saved to {output_path}")
    
    # Print a summary
    print_benchmark_summary(benchmark_config)
    
    return benchmark_config


def print_benchmark_summary(config: Dict[str, Any]) -> None:
    """
    Print a summary of the benchmark configuration.
    
    Args:
        config (Dict[str, Any]): Benchmark configuration.
    """
    print("\n" + "="*70)
    print(f"Benchmark Summary for {config['model_name']} ({config['model_family']})")
    print("="*70)
    
    print(f"\nMode: {config['mode']}")
    
    # Single device configurations
    print("\nSingle Device Configurations:")
    for hw, configs in config.get("single_device", {}).items():
        print(f"  {hw.upper()}: {len(configs)} configurations")
        for i, cfg in enumerate(configs[:3]):  # Show only first 3
            print(f"    - Batch size: {cfg['batch_size']}, Mixed precision: {cfg['mixed_precision']}")
        
        if len(configs) > 3:
            print(f"    ... and {len(configs) - 3} more configurations")
    
    # Distributed configurations
    if "distributed" in config and config["distributed"]:
        print("\nDistributed Configurations:")
        for hw, configs in config["distributed"].items():
            print(f"  {hw.upper()}: {len(configs)} configurations")
            for i, cfg in enumerate(configs[:3]):  # Show only first 3
                print(f"    - {cfg['gpu_count']} GPUs, Strategy: {cfg['strategy']}")
                print(f"      Batch size: {cfg['per_gpu_batch_size']} per GPU (global: {cfg['global_batch_size']})")
                print(f"      Estimated memory: {cfg['estimated_memory_gb']:.2f} GB per GPU")
                
                if "optimizations" in cfg and cfg["optimizations"]:
                    print(f"      Optimizations: {', '.join(cfg['optimizations'])}")
            
            if len(configs) > 3:
                print(f"    ... and {len(configs) - 3} more configurations")
    
    print("\n" + "="*70)


def main():
    """
    Main function.
    """
    parser = argparse.ArgumentParser(description="Training Benchmark Runner")
    
    # Main options
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", help="Model name to benchmark")
    group.add_argument("--list-models", action="store_true", help="List sample models for benchmarking")
    
    # Output options
    parser.add_argument("--output-dir", default="./benchmark_results", help="Directory to store benchmark results")
    parser.add_argument("--output", help="Path to save benchmark configuration")
    
    # Hardware options
    parser.add_argument("--hardware", nargs="+", help="Available hardware platforms")
    
    # Distributed options
    parser.add_argument("--distributed", action="store_true", help="Include distributed configurations")
    parser.add_argument("--max-gpus", type=int, default=8, help="Maximum number of GPUs for distributed training")
    
    # Mode
    parser.add_argument("--mode", choices=["training", "inference"], default="training", help="Mode (training or inference)")
    
    # Debug options
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    
    parser.add_argument("--db-path", type=str, default=None,
                      help="Path to the benchmark database")
    parser.add_argument("--db-only", action="store_true",
                      help="Store results only in the database, not in JSON")
args = parser.parse_args()
    
    # Configure logging
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # List sample models
    if args.list_models:
        list_sample_models()
        return
    
    # Run training benchmark
    run_training_benchmark(
        model_name=args.model,
        output_dir=args.output_dir,
        output_file=args.output,
        available_hardware=args.hardware,
        include_distributed=args.distributed,
        max_gpus=args.max_gpus,
        mode=args.mode
    )


if __name__ == "__main__":
    main()