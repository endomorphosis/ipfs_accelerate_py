#!/usr/bin/env python
"""
Benchmark Model Verification Integration Example

This script demonstrates how to integrate the Model File Verification and Conversion Pipeline
with the benchmark system, ensuring models are properly validated before benchmarking.

Key features:
- Pre-benchmark model file verification
- Automated PyTorch to ONNX conversion when needed
- Robust error handling and retry logic
- Caching of converted models
- Integration with benchmark database

Usage:
    python benchmark_model_verification.py --model bert-base-uncased --file-path model.onnx
    python benchmark_model_verification.py --models bert-base-uncased t5-small --file-path model.onnx
    python benchmark_model_verification.py --model-file models.txt
"""

import os
import sys
import logging
import argparse
import json
import time
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
from pathlib import Path

# Import the model file verification utility
from model_file_verification import ModelFileVerifier, run_verification, batch_verify_models
from model_file_verification import ModelVerificationError, ModelConversionError, ModelFileNotFoundError

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("benchmark_model_verification")

def load_models_from_file(file_path: str) -> List[str]:
    """Load model IDs from a file, one per line."""
    try:
        with open(file_path, 'r') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    except Exception as e:
        logger.error(f"Error loading models from file {file_path}: {e}")
        sys.exit(1)

def setup_benchmark_environment():
    """Setup the benchmark environment with necessary configurations."""
    # Create benchmark results directory if it doesn't exist
    os.makedirs("benchmark_results", exist_ok=True)
    
    # Initialize benchmark configuration
    benchmark_config = {
        "benchmark_time": datetime.now().isoformat(),
        "models_verified": 0,
        "models_converted": 0,
        "benchmark_results": []
    }
    
    return benchmark_config

def run_benchmark(model_path: str, model_config: Dict[str, Any], 
                was_converted: bool, hardware_type: str) -> Dict[str, Any]:
    """
    Run benchmark for a specific model configuration.
    
    Args:
        model_path: Path to the model file
        model_config: Configuration for the model
        was_converted: Whether the model was converted from PyTorch
        hardware_type: Hardware type to run the benchmark on
        
    Returns:
        Dictionary containing benchmark results
    """
    # In a real implementation, this would run the actual benchmark
    # For demonstration purposes, we'll simulate benchmark results
    
    logger.info(f"Running benchmark for {model_config['model_id']} on {hardware_type}")
    logger.info(f"Using {'converted' if was_converted else 'original'} model at {model_path}")
    
    # Simulate benchmarking with different batch sizes
    batch_results = {}
    for batch_size in model_config.get('batch_sizes', [1, 2, 4, 8]):
        # Simulate benchmark execution
        time.sleep(0.2)  # Simulate benchmark execution time
        
        # Simulate benchmark metrics
        throughput = 100.0 / batch_size if not was_converted else 90.0 / batch_size
        latency = 5.0 * batch_size if not was_converted else 5.5 * batch_size
        memory = 500 * batch_size if not was_converted else 550 * batch_size
        
        batch_results[str(batch_size)] = {
            "throughput_items_per_second": throughput,
            "latency_ms": latency,
            "memory_mb": memory
        }
    
    # Create benchmark result
    result = {
        "model_id": model_config['model_id'],
        "model_type": model_config.get('model_type', 'unknown'),
        "hardware_type": hardware_type,
        "was_converted": was_converted,
        "model_path": model_path,
        "timestamp": datetime.now().isoformat(),
        "batch_results": batch_results,
        # Add flags to indicate source for database and reporting
        "model_source": "pytorch_conversion" if was_converted else "huggingface",
        "verification_status": "converted" if was_converted else "original"
    }
    
    logger.info(f"Benchmark completed for {model_config['model_id']}")
    return result

def save_benchmark_results(benchmark_config: Dict[str, Any], output_dir: str):
    """Save benchmark results to a file and database."""
    output_file = os.path.join(
        output_dir, 
        f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    
    with open(output_file, 'w') as f:
        json.dump(benchmark_config, f, indent=2)
    
    logger.info(f"Benchmark results saved to {output_file}")
    
    # Save to database if available
    try:
        # Attempt to use duckdb database if available
        try:
            import duckdb
            db_path = os.environ.get('BENCHMARK_DB_PATH', './benchmark_db.duckdb')
            
            if os.path.exists(db_path):
                logger.info(f"Storing results in DuckDB database: {db_path}")
                
                # Connect to the database
                conn = duckdb.connect(db_path)
                
                # Store results
                for result in benchmark_config['benchmark_results']:
                    # In a real implementation, we would extract data and store it properly
                    model_id = result['model_id']
                    model_type = result['model_type']
                    hardware_type = result['hardware_type']
                    was_converted = result['was_converted']
                    
                    logger.info(f"Storing result for {model_id} on {hardware_type} in database")
                    
                    # Here we would use proper SQL to insert into the database
                    # This is just a simplified example
                    conn.execute("""
                        INSERT INTO benchmark_runs (model_id, model_type, hardware_type, was_converted, timestamp)
                        VALUES (?, ?, ?, ?, ?)
                    """, (model_id, model_type, hardware_type, was_converted, datetime.now().isoformat()))
                
                # Close the connection
                conn.close()
            else:
                logger.warning(f"Database file not found: {db_path}")
        
        # Fallback to file storage if duckdb is not available
        except ImportError:
            logger.info("DuckDB not available, falling back to file storage")
    
    except Exception as e:
        logger.warning(f"Error storing results in database: {e}")

def run_benchmarks_with_verification(models: List[Dict[str, Any]], hardware_type: str, 
                                   output_dir: str = "benchmark_results",
                                   huggingface_token: Optional[str] = None,
                                   cache_dir: Optional[str] = None):
    """
    Run benchmarks with model verification and fallback conversion.
    
    Args:
        models: List of model configurations
        hardware_type: Hardware type to run benchmarks on
        output_dir: Directory to save benchmark results
        huggingface_token: Optional HuggingFace API token
        cache_dir: Optional cache directory for models
    """
    # Setup benchmark environment
    benchmark_config = setup_benchmark_environment()
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a model file verifier
    verifier = ModelFileVerifier(
        cache_dir=cache_dir,
        huggingface_token=huggingface_token
    )
    
    # Run benchmarks for each model
    for model_config in models:
        model_id = model_config['model_id']
        file_path = model_config.get('file_path', 'model.onnx')
        model_type = model_config.get('model_type')
        
        logger.info(f"Preparing benchmark for {model_id}")
        
        try:
            # Verify the model file exists or can be converted
            model_path, was_converted = verifier.verify_model_for_benchmark(
                model_id=model_id,
                file_path=file_path,
                model_type=model_type
            )
            
            # Update verification/conversion counters
            if was_converted:
                benchmark_config['models_converted'] += 1
                logger.info(f"Using converted model for {model_id}")
            else:
                benchmark_config['models_verified'] += 1
                logger.info(f"Using original model for {model_id}")
            
            # Run benchmark
            result = run_benchmark(
                model_path=model_path,
                model_config=model_config,
                was_converted=was_converted,
                hardware_type=hardware_type
            )
            
            # Add result to benchmark results
            benchmark_config['benchmark_results'].append(result)
            
        except ModelVerificationError as e:
            logger.error(f"Model verification failed for {model_id}: {e}")
        except ModelConversionError as e:
            logger.error(f"Model conversion failed for {model_id}: {e}")
        except ModelFileNotFoundError as e:
            logger.error(f"Model file not found for {model_id}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error for {model_id}: {e}")
    
    # Save benchmark results
    save_benchmark_results(benchmark_config, output_dir)
    
    # Print summary
    logger.info(f"Benchmark complete: {len(benchmark_config['benchmark_results'])} models benchmarked")
    logger.info(f"Models verified: {benchmark_config['models_verified']}")
    logger.info(f"Models converted: {benchmark_config['models_converted']}")

def main():
    """Main function for the benchmark model verification."""
    parser = argparse.ArgumentParser(description='Run benchmarks with model verification')
    
    # Model selection
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--model', type=str, help='HuggingFace model ID')
    group.add_argument('--models', type=str, nargs='+', help='Multiple HuggingFace model IDs')
    group.add_argument('--model-file', type=str, help='File containing model IDs (one per line)')
    
    # File path and model type
    parser.add_argument('--file-path', type=str, default='model.onnx', 
                       help='Path to the model file within the repository')
    parser.add_argument('--model-type', type=str, 
                       help='Model type (auto-detected if not provided)')
    
    # Benchmark configuration
    parser.add_argument('--hardware', type=str, default='cpu', 
                       help='Hardware type to run benchmarks on')
    parser.add_argument('--batch-sizes', type=str, default='1,2,4,8', 
                       help='Comma-separated list of batch sizes to benchmark')
    
    # Output configuration
    parser.add_argument('--output-dir', type=str, default='benchmark_results', 
                       help='Directory to save benchmark results')
    
    # Model verification configuration
    parser.add_argument('--token', type=str, 
                       help='HuggingFace API token for private models')
    parser.add_argument('--cache-dir', type=str, 
                       help='Cache directory for models')
    
    # Other options
    parser.add_argument('--verbose', '-v', action='store_true', 
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Set log level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Parse batch sizes
    batch_sizes = [int(size) for size in args.batch_sizes.split(',')]
    
    # Get model configurations
    models = []
    
    if args.model:
        models.append({
            'model_id': args.model,
            'file_path': args.file_path,
            'model_type': args.model_type,
            'batch_sizes': batch_sizes
        })
    elif args.models:
        for model_id in args.models:
            models.append({
                'model_id': model_id,
                'file_path': args.file_path,
                'model_type': args.model_type,
                'batch_sizes': batch_sizes
            })
    else:
        model_ids = load_models_from_file(args.model_file)
        for model_id in model_ids:
            models.append({
                'model_id': model_id,
                'file_path': args.file_path,
                'model_type': args.model_type,
                'batch_sizes': batch_sizes
            })
    
    # Run benchmarks with verification
    run_benchmarks_with_verification(
        models=models,
        hardware_type=args.hardware,
        output_dir=args.output_dir,
        huggingface_token=args.token,
        cache_dir=args.cache_dir
    )

if __name__ == "__main__":
    main()