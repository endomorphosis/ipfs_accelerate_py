"""
Utilities for importing benchmark results from other formats.
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from ..benchmark import BenchmarkResult, BenchmarkResults, BenchmarkConfig

logger = logging.getLogger("benchmark.utils.importers")

def import_from_original_format(file_path: str) -> BenchmarkResults:
    """
    Import benchmark results from the original format.
    
    Args:
        file_path: Path to the JSON file with original benchmark results
        
    Returns:
        BenchmarkResults object
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Extract model ID
    model_id = data.get("model", "unknown-model")
    
    # Extract hardware platforms
    hardware_platforms = data.get("hardware_tested", [])
    
    # Extract results
    results = []
    for hw in hardware_platforms:
        hw_results = data.get("results", {}).get(hw, {})
        for result_data in hw_results.get("results", []):
            try:
                # Extract basic information
                batch_size = result_data.get("batch_size", 1)
                sequence_length = result_data.get("sequence_length", 1)
                
                # Extract metrics
                metrics = {}
                
                # Latency
                inference_time_ms = result_data.get("inference_time_ms")
                if inference_time_ms is not None:
                    metrics["latency_ms"] = inference_time_ms
                
                # Memory
                memory_usage_mb = result_data.get("memory_usage_mb")
                if memory_usage_mb is not None:
                    metrics["memory_peak_mb"] = memory_usage_mb
                    metrics["memory_used_mb"] = memory_usage_mb
                
                # Load time
                load_time_s = result_data.get("load_time_s")
                if load_time_s is not None:
                    metrics["load_time_s"] = load_time_s
                
                # Timestamp
                timestamp = result_data.get("timestamp", datetime.now().isoformat())
                
                # Create BenchmarkResult
                result = BenchmarkResult(
                    model_id=model_id,
                    hardware=hw,
                    batch_size=batch_size,
                    sequence_length=sequence_length,
                    metrics=metrics,
                    input_shape={},
                    output_shape={},
                    timestamp=timestamp
                )
                
                results.append(result)
                
            except Exception as e:
                logger.warning(f"Error importing result: {e}")
    
    # Create BenchmarkConfig
    config = BenchmarkConfig(
        model_id=model_id,
        hardware=hardware_platforms,
        batch_sizes=list(set(r.batch_size for r in results)),
        sequence_lengths=list(set(r.sequence_length for r in results)),
        output_dir=os.path.dirname(file_path)
    )
    
    # Create BenchmarkResults
    return BenchmarkResults(results, config)

def import_from_directory(directory: str, pattern: Optional[str] = None) -> Dict[str, BenchmarkResults]:
    """
    Import all benchmark results from a directory.
    
    Args:
        directory: Directory containing benchmark JSON files
        pattern: Optional filename pattern to match
        
    Returns:
        Dictionary mapping model IDs to BenchmarkResults objects
    """
    results = {}
    
    # List JSON files in directory
    for filename in os.listdir(directory):
        if not filename.endswith(".json"):
            continue
            
        if pattern and pattern not in filename:
            continue
        
        file_path = os.path.join(directory, filename)
        
        try:
            # Import from file
            benchmark_results = import_from_original_format(file_path)
            
            # Add to results dictionary
            results[benchmark_results.config.model_id] = benchmark_results
            
        except Exception as e:
            logger.warning(f"Error importing {file_path}: {e}")
    
    return results