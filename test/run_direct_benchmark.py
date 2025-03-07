#!/usr/bin/env python3
"""
Direct Benchmark Runner

This script performs a simple benchmark of a model on a specific hardware platform.
It directly uses PyTorch to load and benchmark the model, avoiding the complexity
of the full benchmark system.

Usage:
    python run_direct_benchmark.py --model bert-base-uncased --hardware cpu
"""

import os
import sys
import time
import json
import logging
import argparse
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_BATCH_SIZES = [1, 2, 4, 8]
OUTPUT_DIR = "benchmark_results"

# Database configuration
DB_PATH = os.environ.get("BENCHMARK_DB_PATH", "./benchmark_db.duckdb")
# By default, we now prefer DuckDB over JSON files
USE_DB_ONLY = os.environ.get("DEPRECATE_JSON_OUTPUT", "1").lower() in ("1", "true", "yes")

def try_import_model(model_name: str) -> Optional[Any]:
    """Try to import a model from HuggingFace Transformers."""
    try:
        from transformers import AutoModel, AutoTokenizer, AutoFeatureExtractor
        
        # Try to load the model
        logger.info(f"Loading model: {model_name}")
        
        # Determine model type based on name
        if "bert" in model_name.lower():
            model = AutoModel.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            return {"model": model, "tokenizer": tokenizer, "type": "text"}
        elif "t5" in model_name.lower():
            from transformers import T5Model, T5Tokenizer
            model = T5Model.from_pretrained(model_name)
            tokenizer = T5Tokenizer.from_pretrained(model_name)
            return {"model": model, "tokenizer": tokenizer, "type": "text"}
        elif "vit" in model_name.lower():
            from transformers import ViTModel, ViTFeatureExtractor
            model = ViTModel.from_pretrained(model_name)
            processor = ViTFeatureExtractor.from_pretrained(model_name)
            return {"model": model, "processor": processor, "type": "vision"}
        else:
            logger.warning(f"Unknown model type: {model_name}, attempting generic loading")
            model = AutoModel.from_pretrained(model_name)
            tokenizer = None
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
            except:
                try:
                    tokenizer = AutoFeatureExtractor.from_pretrained(model_name)
                except:
                    pass
            return {"model": model, "tokenizer": tokenizer, "type": "unknown"}
    
    except Exception as e:
        logger.error(f"Error loading model {model_name}: {str(e)}")
        return None

def get_device(hardware: str) -> torch.device:
    """Get the appropriate device for the specified hardware."""
    if hardware == "cpu":
        return torch.device("cpu")
    elif hardware == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    elif hardware == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        logger.warning(f"Requested hardware {hardware} not available, falling back to CPU")
        return torch.device("cpu")

def generate_inputs(model_info: Dict[str, Any], batch_size: int = 1) -> Dict[str, torch.Tensor]:
    """Generate appropriate inputs for the model."""
    model_type = model_info.get("type", "unknown")
    model_name = str(model_info.get("model", "")).lower()
    
    if model_type == "text":
        tokenizer = model_info["tokenizer"]
        if tokenizer is None:
            logger.error("No tokenizer available for text model")
            return None
        
        # Generate simple text input
        text = ["Hello, this is a test" for _ in range(batch_size)]
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        
        # Special handling for T5 models
        if "t5" in model_name:
            # T5 requires decoder_input_ids
            decoder_input_ids = torch.ones((batch_size, 1), dtype=torch.long)
            inputs["decoder_input_ids"] = decoder_input_ids
            
        return inputs
    
    elif model_type == "vision":
        processor = model_info.get("processor")
        if processor is None:
            logger.error("No processor available for vision model")
            return None
        
        # Create dummy image tensors
        batch = torch.rand(batch_size, 3, 224, 224)
        if "processor" in model_info:
            try:
                inputs = processor(images=batch, return_tensors="pt")
                return inputs
            except:
                # Fall back to raw tensors
                return {"pixel_values": batch}
        else:
            return {"pixel_values": batch}
    
    else:
        logger.warning(f"Unknown model type {model_type}, using generic input")
        # Generic input: random tensors
        return {"input_ids": torch.randint(0, 1000, (batch_size, 32))}

def benchmark_model(model_info: Dict[str, Any], device: torch.device, batch_size: int = 1, 
                   num_runs: int = 5, warmup: int = 3) -> Dict[str, Any]:
    """Benchmark the model with the given configuration."""
    model = model_info["model"]
    model.to(device)
    model.eval()
    
    # Generate inputs
    inputs = generate_inputs(model_info, batch_size)
    if inputs is None:
        return {"success": False, "error": "Failed to generate inputs"}
    
    # Move inputs to device
    for key in inputs:
        inputs[key] = inputs[key].to(device)
    
    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = model(**inputs)
    
    # Benchmark
    latencies = []
    memory_usage = []
    
    for i in range(num_runs):
        # Clear cache if using CUDA
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        start_time = time.time()
        with torch.no_grad():
            outputs = model(**inputs)
        end_time = time.time()
        
        # Record latency
        latency = (end_time - start_time) * 1000  # Convert to ms
        latencies.append(latency)
        
        # Record memory usage
        if device.type == "cuda":
            memory = torch.cuda.max_memory_allocated() / (1024 * 1024)  # Convert to MB
        else:
            # Use psutil for CPU if available
            try:
                import psutil
                memory = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)  # Convert to MB
            except:
                memory = 0
        memory_usage.append(memory)
    
    # Calculate metrics
    avg_latency = sum(latencies) / len(latencies)
    throughput = 1000 * batch_size / avg_latency  # items per second (1000ms = 1s)
    avg_memory = sum(memory_usage) / len(memory_usage)
    
    return {
        "success": True,
        "batch_size": batch_size,
        "avg_latency_ms": avg_latency,
        "throughput_items_per_second": throughput,
        "memory_mb": avg_memory,
        "latencies_ms": latencies,
        "device": str(device),
        "timestamp": datetime.now().isoformat()
    }

def store_benchmark_in_database(results: Dict[str, Any], db_path: str = DB_PATH) -> bool:
    """
    Store benchmark results in DuckDB database.
    
    Args:
        results: Benchmark results dictionary
        db_path: Path to DuckDB database
        
    Returns:
        bool: True if stored successfully, False otherwise
    """
    try:
        # Try to import DuckDB
        import duckdb
        
        # Connect to database
        conn = duckdb.connect(db_path)
        
        # Extract key metrics
        model_name = results.get("model", "unknown")
        hardware_type = results.get("hardware", "unknown")
        timestamp = results.get("timestamp", datetime.now().isoformat())
        
        # Make sure the necessary tables exist
        conn.execute("""
        CREATE TABLE IF NOT EXISTS models (
            model_id INTEGER PRIMARY KEY,
            model_name VARCHAR UNIQUE,
            model_family VARCHAR,
            model_type VARCHAR,
            metadata JSON
        )
        """)
        
        conn.execute("""
        CREATE TABLE IF NOT EXISTS hardware_platforms (
            hardware_id INTEGER PRIMARY KEY,
            hardware_type VARCHAR UNIQUE,
            is_simulated BOOLEAN DEFAULT FALSE,
            simulation_reason VARCHAR,
            metadata JSON
        )
        """)
        
        conn.execute("""
        CREATE TABLE IF NOT EXISTS test_runs (
            run_id INTEGER PRIMARY KEY,
            test_name VARCHAR,
            test_type VARCHAR,
            start_time TIMESTAMP,
            end_time TIMESTAMP,
            status VARCHAR,
            metadata JSON
        )
        """)
        
        conn.execute("""
        CREATE TABLE IF NOT EXISTS performance_results (
            id INTEGER PRIMARY KEY,
            run_id INTEGER,
            model_id INTEGER,
            hardware_id INTEGER,
            batch_size INTEGER,
            sequence_length INTEGER DEFAULT NULL,
            latency_ms FLOAT,
            throughput_items_per_second FLOAT,
            memory_mb FLOAT,
            is_simulated BOOLEAN DEFAULT FALSE,
            simulation_reason VARCHAR,
            created_at TIMESTAMP,
            metadata JSON,
            FOREIGN KEY (run_id) REFERENCES test_runs(run_id),
            FOREIGN KEY (model_id) REFERENCES models(model_id),
            FOREIGN KEY (hardware_id) REFERENCES hardware_platforms(hardware_id)
        )
        """)
        
        # Create test run
        conn.execute("""
        INSERT INTO test_runs (test_name, test_type, start_time, end_time, status, metadata)
        VALUES (?, ?, ?, ?, ?, ?)
        """, [
            f"benchmark_{model_name}_{hardware_type}",
            "direct_benchmark",
            timestamp,
            timestamp,  # Same as start for simple benchmarks
            "completed",
            json.dumps({"source": "run_direct_benchmark.py"})
        ])
        
        # Get run_id (last inserted row)
        run_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        
        # Insert or get model
        conn.execute("""
        INSERT INTO models (model_name, model_type, metadata)
        VALUES (?, ?, ?)
        ON CONFLICT (model_name) DO UPDATE SET
        metadata = json_patch(metadata, excluded.metadata)
        RETURNING model_id
        """, [
            model_name,
            "unknown",  # We don't have model type info in the results
            json.dumps({"last_benchmark": timestamp})
        ])
        
        # Get model_id
        model_id = conn.execute("SELECT model_id FROM models WHERE model_name = ?", [model_name]).fetchone()[0]
        
        # Insert or get hardware platform
        conn.execute("""
        INSERT INTO hardware_platforms (hardware_type, metadata)
        VALUES (?, ?)
        ON CONFLICT (hardware_type) DO UPDATE SET
        metadata = json_patch(metadata, excluded.metadata)
        RETURNING hardware_id
        """, [
            hardware_type,
            json.dumps({"last_benchmark": timestamp})
        ])
        
        # Get hardware_id
        hardware_id = conn.execute("SELECT hardware_id FROM hardware_platforms WHERE hardware_type = ?", [hardware_type]).fetchone()[0]
        
        # Insert performance results for each batch size
        batch_results = results.get("batch_results", {})
        for batch_size, batch_result in batch_results.items():
            if batch_result.get("success", False):
                conn.execute("""
                INSERT INTO performance_results (
                    run_id, model_id, hardware_id, batch_size, 
                    latency_ms, throughput_items_per_second, memory_mb,
                    created_at, metadata
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    run_id,
                    model_id,
                    hardware_id,
                    int(batch_size),
                    batch_result.get("avg_latency_ms", 0),
                    batch_result.get("throughput_items_per_second", 0),
                    batch_result.get("memory_mb", 0),
                    timestamp,
                    json.dumps(batch_result)
                ])
        
        # Commit changes
        conn.commit()
        conn.close()
        
        logger.info(f"Benchmark results for {model_name} on {hardware_type} stored in database")
        return True
    
    except ImportError:
        logger.warning("DuckDB not available. Cannot store results in database.")
        return False
    
    except Exception as e:
        logger.error(f"Error storing benchmark results in database: {str(e)}")
        return False

def run_benchmarks(model_name: str, hardware: str, batch_sizes: List[int] = None, 
                  output_dir: str = OUTPUT_DIR, verbose: bool = False, 
                  db_path: str = DB_PATH, db_only: bool = USE_DB_ONLY) -> Dict[str, Any]:
    """Run benchmarks for a model on a specific hardware platform."""
    if batch_sizes is None:
        batch_sizes = DEFAULT_BATCH_SIZES
    
    # Create output directory if needed
    if not db_only:
        os.makedirs(output_dir, exist_ok=True)
    elif verbose:
        logger.info("Using database storage only (JSON output deprecated)")
    
    # Load model
    model_info = try_import_model(model_name)
    if model_info is None:
        return {
            "model": model_name,
            "hardware": hardware,
            "success": False,
            "error": "Failed to load model",
            "timestamp": datetime.now().isoformat()
        }
    
    # Get device
    device = get_device(hardware)
    if verbose:
        logger.info(f"Using device: {device}")
    
    # Run benchmarks for each batch size
    results = {
        "model": model_name,
        "hardware": hardware,
        "success": True,
        "batch_results": {},
        "timestamp": datetime.now().isoformat()
    }
    
    for batch_size in batch_sizes:
        if verbose:
            logger.info(f"Benchmarking batch size: {batch_size}")
        
        try:
            batch_result = benchmark_model(model_info, device, batch_size)
            if batch_result["success"]:
                results["batch_results"][str(batch_size)] = batch_result
                
                if verbose:
                    logger.info(f"Batch {batch_size}: Latency {batch_result['avg_latency_ms']:.2f}ms, "
                                f"Throughput {batch_result['throughput_items_per_second']:.2f} items/s, "
                                f"Memory {batch_result['memory_mb']:.2f}MB")
            else:
                logger.warning(f"Benchmark failed for batch size {batch_size}: {batch_result.get('error', 'Unknown error')}")
                results["batch_results"][str(batch_size)] = {
                    "success": False,
                    "error": batch_result.get("error", "Unknown error")
                }
        except Exception as e:
            logger.error(f"Error benchmarking batch size {batch_size}: {str(e)}")
            results["batch_results"][str(batch_size)] = {
                "success": False,
                "error": str(e)
            }
    
    # Store results in database if possible
    db_stored = store_benchmark_in_database(results, db_path)
    
    # Save results to JSON file if needed
    if not db_only:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f"direct_benchmark_{model_name.replace('/', '_')}_{hardware}_{timestamp}.json")
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        
        if verbose:
            logger.info(f"Results saved to {output_path}")
    elif not db_stored and verbose:
        logger.warning("Results could not be stored in database and JSON output is disabled.")
    
    return results

def main():
    """Main function for the direct benchmark script."""
    parser = argparse.ArgumentParser(description="Run direct benchmarks on a model")
    parser.add_argument("--model", required=True, help="Model name (e.g., bert-base-uncased)")
    parser.add_argument("--hardware", choices=["cpu", "cuda", "mps"], default="cpu", help="Hardware to use")
    parser.add_argument("--batch-sizes", help="Comma-separated list of batch sizes to test")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Directory to save results")
    parser.add_argument("--db-path", help="Path to DuckDB database (defaults to BENCHMARK_DB_PATH env var)")
    parser.add_argument("--db-only", action="store_true", help="Store results only in database, not in JSON")
    parser.add_argument("--no-db", action="store_true", help="Don't store results in database")
    parser.add_argument("--verbose", action="store_true", help="Print detailed output")
    
    args = parser.parse_args()
    
    # Process batch sizes
    batch_sizes = None
    if args.batch_sizes:
        try:
            batch_sizes = [int(b.strip()) for b in args.batch_sizes.split(",")]
        except ValueError:
            logger.error("Error: Batch sizes must be comma-separated integers")
            return 1
    
    # Get database path
    db_path = args.db_path or DB_PATH
    
    # Determine whether to use database only (default) or also JSON
    db_only = USE_DB_ONLY
    if args.db_only:
        db_only = True
    elif args.no_db:
        db_only = False
    
    # Run benchmarks
    logger.info(f"Running direct benchmarks for {args.model} on {args.hardware}")
    
    if db_only:
        logger.info("Using database storage only (JSON output deprecated)")
    elif args.no_db:
        logger.info("Using JSON output only (not recommended)")
    
    results = run_benchmarks(
        args.model,
        args.hardware,
        batch_sizes,
        args.output_dir,
        args.verbose,
        db_path,
        db_only
    )
    
    if results["success"]:
        logger.info("Benchmarks completed successfully")
        
        # Print summary
        print("\nBenchmark Summary:")
        print(f"Model: {args.model}")
        print(f"Hardware: {args.hardware}")
        print("\nResults by Batch Size:")
        
        for batch_size, batch_result in results["batch_results"].items():
            if batch_result.get("success", False):
                print(f"Batch {batch_size}: "
                      f"Latency {batch_result['avg_latency_ms']:.2f}ms, "
                      f"Throughput {batch_result['throughput_items_per_second']:.2f} items/s, "
                      f"Memory {batch_result['memory_mb']:.2f}MB")
            else:
                print(f"Batch {batch_size}: Failed - {batch_result.get('error', 'Unknown error')}")
        
        return 0
    else:
        logger.error(f"Benchmarks failed: {results.get('error', 'Unknown error')}")
        return 1

if __name__ == "__main__":
    sys.exit(main())