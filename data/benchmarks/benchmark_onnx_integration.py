"""
Benchmark ONNX Integration Example

This script demonstrates how to integrate the ONNX verification and conversion utility
with the benchmark system, ensuring models are properly validated before benchmarking
and providing fallback to PyTorch conversion when ONNX files are not available on HuggingFace.
"""

import os
import sys
import logging
import argparse
import json
import time
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("benchmark_onnx_integration")

# Import the ONNX verification utility
from onnx_verification import verify_and_get_onnx_model, OnnxVerificationError, OnnxConversionError

def setup_benchmark_environment():
    """Setup the benchmark environment with necessary configurations."""
    # Create benchmark results directory if it doesn't exist
    os.makedirs("benchmark_results", exist_ok=True)
    
    # Initialize any benchmark-specific configurations
    benchmark_config = {
        "benchmark_time": datetime.now().isoformat(),
        "models_verified": 0,
        "models_converted": 0,
        "benchmark_results": []
    }
    
    return benchmark_config

def get_model_configurations() -> List[Dict[str, Any]]:
    """
    Get the list of model configurations to benchmark.
    In a real scenario, this might be loaded from a configuration file.
    """
    return [
        {
            "model_id": "bert-base-uncased",
            "onnx_path": "model.onnx",
            "model_type": "bert",
            "batch_sizes": [1, 2, 4, 8]
        },
        {
            "model_id": "t5-small",
            "onnx_path": "model.onnx",
            "model_type": "t5",
            "batch_sizes": [1, 2, 4]
        },
        {
            "model_id": "openai/whisper-tiny",
            "onnx_path": "model.onnx",
            "model_type": "whisper",
            "batch_sizes": [1]
        }
    ]

def run_benchmark(model_path: str, model_config: Dict[str, Any], 
                 was_converted: bool, hardware_type: str) -> Dict[str, Any]:
    """
    Run benchmark for a specific model configuration.
    
    Args:
        model_path: Path to the ONNX model file
        model_config: Configuration for the model
        was_converted: Whether the model was converted from PyTorch
        hardware_type: Hardware type to run the benchmark on
        
    Returns:
        Dictionary containing benchmark results
    """
    # In a real implementation, this would run the actual benchmark
    # For demonstration purposes, we'll simulate benchmark results
    
    logger.info(f"Running benchmark for {model_config['model_id']} on {hardware_type}")
    logger.info(f"Using {'converted' if was_converted else 'original'} ONNX model at {model_path}")
    
    # Simulate benchmarking with different batch sizes
    batch_results = {}
    for batch_size in model_config['batch_sizes']:
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
        "model_type": model_config['model_type'],
        "hardware_type": hardware_type,
        "was_converted": was_converted,
        "model_path": model_path,
        "timestamp": datetime.now().isoformat(),
        "batch_results": batch_results,
        # Add flags to indicate source for database and reporting
        "onnx_source": "pytorch_conversion" if was_converted else "huggingface",
        "onnx_verification_status": "converted" if was_converted else "original"
    }
    
    logger.info(f"Benchmark completed for {model_config['model_id']}")
    return result

def save_benchmark_results(benchmark_config: Dict[str, Any]):
    """Save benchmark results to a file."""
    output_file = os.path.join(
        "benchmark_results", 
        f"onnx_benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    
    with open(output_file, 'w') as f:
        json.dump(benchmark_config, f, indent=2)
    
    logger.info(f"Benchmark results saved to {output_file}")
    
    # Save to database if available
    try:
        from benchmark_db_api import store_benchmark_in_database
        
        # First check if database has the required ONNX schema fields
        has_onnx_fields = check_database_schema()
        
        for result in benchmark_config['benchmark_results']:
            # Set additional fields for ONNX conversion tracking if database has the schema
            if has_onnx_fields:
                was_converted = result.get('was_converted', False)
                onnx_source = result.get('onnx_source', 'unknown')
                
                # Update result with additional fields for the database
                result.update({
                    "onnx_source": onnx_source,
                    "onnx_conversion_status": "converted" if was_converted else "original",
                    "onnx_conversion_time": datetime.now().isoformat() if was_converted else None,
                    "onnx_local_path": result.get('model_path') if was_converted else None
                })
            
            store_benchmark_in_database(result)
            
        logger.info("Benchmark results stored in database")
    except ImportError:
        logger.warning("benchmark_db_api not available, results not stored in database")
        
def check_database_schema() -> bool:
    """
    Check if the database has the ONNX tracking schema fields.
    
    Returns:
        True if the database has the ONNX tracking schema fields, False otherwise
    """
    try:
        import duckdb
        
        # Get the database path from environment variable or use default
        db_path = os.environ.get('BENCHMARK_DB_PATH', './benchmark_db.duckdb')
        
        if not os.path.exists(db_path):
            logger.warning(f"Database file does not exist: {db_path}")
            return False
        
        # Connect to the database
        conn = duckdb.connect(db_path)
        
        # Check if the performance_results table has the onnx_source column
        result = conn.execute("""
            SELECT count(*) FROM information_schema.columns 
            WHERE table_name = 'performance_results' AND column_name = 'onnx_source'
        """).fetchone()
        
        has_schema = result[0] > 0
        
        # Close the connection
        conn.close()
        
        if not has_schema:
            logger.warning("Database schema does not have ONNX tracking fields. Run onnx_db_schema_update.py to update the schema.")
        
        return has_schema
    except:
        logger.warning("Error checking database schema. Assuming schema does not have ONNX tracking fields.")
        return False

def get_conversion_config(model_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get model-specific conversion configuration.
    
    Args:
        model_config: Model configuration
        
    Returns:
        Dictionary with conversion configuration
    """
    model_type = model_config['model_type']
    
    # Base configuration
    config = {
        "model_type": model_type,
        "opset_version": 12
    }
    
    # Model-specific configurations
    if model_type == 'bert':
        config.update({
            "input_shapes": {
                "batch_size": 1,
                "sequence_length": 128
            },
            "input_names": ["input_ids", "attention_mask"],
            "output_names": ["last_hidden_state", "pooler_output"],
            "dynamic_axes": {
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "attention_mask": {0: "batch_size", 1: "sequence_length"},
                "last_hidden_state": {0: "batch_size", 1: "sequence_length"}
            }
        })
    elif model_type == 't5':
        config.update({
            "input_shapes": {
                "batch_size": 1,
                "sequence_length": 128
            },
            "input_names": ["input_ids", "attention_mask"],
            "output_names": ["last_hidden_state"],
            "dynamic_axes": {
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "attention_mask": {0: "batch_size", 1: "sequence_length"},
                "last_hidden_state": {0: "batch_size", 1: "sequence_length"}
            }
        })
    elif model_type == 'whisper':
        config.update({
            "input_shapes": {
                "batch_size": 1,
                "feature_size": 80,
                "sequence_length": 3000
            },
            "input_names": ["input_features"],
            "output_names": ["last_hidden_state"],
            "dynamic_axes": {
                "input_features": {0: "batch_size", 2: "sequence_length"},
                "last_hidden_state": {0: "batch_size", 1: "sequence_length"}
            }
        })
    
    return config

def run_benchmarks_with_onnx_verification(hardware_type: str, models: Optional[List[str]] = None):
    """
    Run benchmarks with ONNX verification and fallback conversion.
    
    Args:
        hardware_type: Hardware type to run benchmarks on
        models: Optional list of model IDs to benchmark (if None, benchmark all configured models)
    """
    # Setup benchmark environment
    benchmark_config = setup_benchmark_environment()
    
    # Get model configurations
    model_configs = get_model_configurations()
    
    # Filter models if specified
    if models:
        model_configs = [config for config in model_configs if config['model_id'] in models]
    
    if not model_configs:
        logger.warning("No models selected for benchmarking")
        return
    
    # Run benchmarks for each model
    for model_config in model_configs:
        model_id = model_config['model_id']
        onnx_path = model_config['onnx_path']
        
        logger.info(f"Preparing benchmark for {model_id}")
        
        try:
            # Get conversion configuration for this model
            conversion_config = get_conversion_config(model_config)
            
            # Verify ONNX file existence and convert if necessary
            model_path, was_converted = verify_and_get_onnx_model(
                model_id=model_id,
                onnx_path=onnx_path,
                conversion_config=conversion_config
            )
            
            # Update verification/conversion counters
            if was_converted:
                benchmark_config['models_converted'] += 1
                logger.info(f"Using converted ONNX model for {model_id}")
            else:
                benchmark_config['models_verified'] += 1
                logger.info(f"Using original ONNX model for {model_id}")
            
            # Run benchmark
            result = run_benchmark(
                model_path=model_path,
                model_config=model_config,
                was_converted=was_converted,
                hardware_type=hardware_type
            )
            
            # Add result to benchmark results
            benchmark_config['benchmark_results'].append(result)
            
        except OnnxVerificationError as e:
            logger.error(f"ONNX verification failed for {model_id}: {e}")
        except OnnxConversionError as e:
            logger.error(f"ONNX conversion failed for {model_id}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error for {model_id}: {e}")
    
    # Save benchmark results
    save_benchmark_results(benchmark_config)
    
    # Print summary
    logger.info(f"Benchmark complete: {len(benchmark_config['benchmark_results'])} models benchmarked")
    logger.info(f"Models verified: {benchmark_config['models_verified']}")
    logger.info(f"Models converted: {benchmark_config['models_converted']}")

def main():
    """Main function to run the benchmark with ONNX verification."""
    parser = argparse.ArgumentParser(description='Run benchmarks with ONNX verification')
    parser.add_argument('--hardware', type=str, default='cpu', 
                        help='Hardware type to run benchmarks on')
    parser.add_argument('--models', type=str, nargs='+',
                        help='Specific models to benchmark (space-separated list of model IDs)')
    
    args = parser.parse_args()
    
    # Run benchmarks
    run_benchmarks_with_onnx_verification(
        hardware_type=args.hardware,
        models=args.models
    )

if __name__ == "__main__":
    main()