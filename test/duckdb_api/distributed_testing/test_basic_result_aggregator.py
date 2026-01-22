#!/usr/bin/env python3
"""
Basic example of using the ResultAggregatorService.

This script demonstrates a simple usage of the ResultAggregatorService
with mock data and without database dependencies.
"""

import os
import sys
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any
from pathlib import Path

# Add parent directory to path
parent_dir = str(Path(__file__).parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import the ResultAggregatorService
from duckdb_api.distributed_testing.result_aggregator import (
    ResultAggregatorService,
    RESULT_TYPE_PERFORMANCE,
    AGGREGATION_LEVEL_MODEL,
    AGGREGATION_LEVEL_HARDWARE,
    AGGREGATION_LEVEL_MODEL_HARDWARE
)


def generate_sample_performance_results() -> List[Dict[str, Any]]:
    """Generate sample performance test results."""
    results = []
    
    # Create results for 2 models on 2 hardware platforms
    models = ["model1", "model2"]
    hardware = ["hw1", "hw2"]
    
    for model_id in models:
        for hardware_id in hardware:
            # Multiple runs with different performance metrics
            for i in range(10):
                # Base performance values
                base_latency = 100 if model_id == "model1" else 200
                base_throughput = 50 if model_id == "model1" else 25
                
                # Hardware efficiency factor
                hw_factor = 1.0 if hardware_id == "hw1" else 0.8
                
                # Random variation (±10%)
                variation = 0.9 + (i / 45)  # From 0.9 to 1.1
                
                # Calculate metrics
                latency = base_latency * hw_factor * variation
                throughput = base_throughput * hw_factor * (2 - variation)  # Inverse relationship
                memory = (base_latency * hw_factor * 0.5) + (i * 2)
                
                # Create result
                result = {
                    "result_id": f"perf_{model_id}_{hardware_id}_{i}",
                    "run_id": f"run_{i % 3}",  # Group into 3 runs
                    "model_id": model_id,
                    "hardware_id": hardware_id,
                    "model_family": "transformer" if model_id == "model1" else "diffusion",
                    "test_case": "inference",
                    "batch_size": 1 if i % 2 == 0 else 4,
                    "precision": "fp16" if i % 3 == 0 else "fp32",
                    "total_time_seconds": latency * 0.01,  # Convert to seconds
                    "average_latency_ms": latency,
                    "throughput_items_per_second": throughput,
                    "memory_peak_mb": memory,
                    "iterations": 100,
                    "warmup_iterations": 10,
                    "is_simulated": i % 5 == 0,  # Some simulated results
                }
                
                # Add a timestamp from the last 7 days
                days_ago = i % 7
                result["timestamp"] = (datetime.now() - timedelta(days=days_ago)).isoformat()
                
                # Add an anomaly for testing
                if i == 9 and model_id == "model1" and hardware_id == "hw1":
                    result["average_latency_ms"] *= 3  # Major latency spike
                    
                if i == 8 and model_id == "model2" and hardware_id == "hw2":
                    result["throughput_items_per_second"] *= 3  # Major throughput improvement
                    
                results.append(result)
                
    return results


class MockDBManager:
    """Mock database manager that returns sample data."""
    
    def __init__(self, sample_data):
        self.sample_data = sample_data
        
    def get_performance_results(self, **kwargs):
        """Return sample performance results."""
        return self.sample_data
    
    def get_hardware_info(self, hardware_id):
        """Return mock hardware info."""
        return {
            "device_name": f"Mock {hardware_id}",
            "hardware_type": "GPU",
            "platform": "Test Platform",
            "memory_gb": 16
        }
    
    def get_model_info(self, model_id):
        """Return mock model info."""
        return {
            "model_name": f"Mock {model_id}",
            "model_family": "transformer" if "1" in model_id else "diffusion",
            "modality": "text" if "1" in model_id else "image",
            "parameters_million": 100 if "1" in model_id else 200
        }


def main():
    """Run the example."""
    print("========== Basic ResultAggregatorService Example ==========")
    
    # Generate sample data
    sample_data = generate_sample_performance_results()
    print(f"Generated {len(sample_data)} sample performance results")
    
    # Create mock database manager
    db_manager = MockDBManager(sample_data)
    
    # Create result aggregator service
    aggregator = ResultAggregatorService(db_manager=db_manager)
    
    # Configure to disable model family grouping for this example
    aggregator.configure({
        "model_family_grouping": False,
        "cache_ttl_seconds": 600,
        "anomaly_threshold": 2.0
    })
    print("Created and configured ResultAggregatorService")
    
    # Aggregate by model
    print("\n----- Aggregating by Model -----")
    model_results = aggregator.aggregate_results(
        result_type=RESULT_TYPE_PERFORMANCE,
        aggregation_level=AGGREGATION_LEVEL_MODEL
    )
    
    # Print basic statistics for each model
    basic_stats = model_results["results"].get("basic_statistics", {})
    for model_id, stats in basic_stats.items():
        print(f"\nModel: {model_id}")
        print(f"Result count: {stats.get('result_count', 0)}")
        
        # Print latency stats
        if "average_latency_ms" in stats:
            latency_stats = stats["average_latency_ms"]
            print(f"Latency (ms): mean={latency_stats.get('mean', 0):.2f}, "
                  f"min={latency_stats.get('min', 0):.2f}, "
                  f"max={latency_stats.get('max', 0):.2f}")
        
        # Print throughput stats
        if "throughput_items_per_second" in stats:
            throughput_stats = stats["throughput_items_per_second"]
            print(f"Throughput (items/s): mean={throughput_stats.get('mean', 0):.2f}, "
                  f"min={throughput_stats.get('min', 0):.2f}, "
                  f"max={throughput_stats.get('max', 0):.2f}")
    
    # Detect anomalies
    print("\n----- Detecting Anomalies -----")
    anomalies = aggregator.get_result_anomalies(
        result_type=RESULT_TYPE_PERFORMANCE,
        aggregation_level=AGGREGATION_LEVEL_MODEL_HARDWARE
    )
    
    # Print anomalies
    print(f"Found {anomalies.get('anomaly_count', 0)} anomalies")
    for anomaly in anomalies.get("anomalies", []):
        print(f"\nAnomaly in {anomaly.get('group', 'unknown')}, metric: {anomaly.get('metric', 'unknown')}")
        print(f"Value: {anomaly.get('value', 0):.2f}, Z-score: {anomaly.get('z_score', 0):.2f}")
        print(f"Direction: {anomaly.get('direction', 'unknown')}, Severity: {anomaly.get('severity', 'unknown')}")
    
    # Export as JSON
    print("\n----- Exporting as JSON -----")
    json_output = aggregator.export_results(
        result_type=RESULT_TYPE_PERFORMANCE,
        aggregation_level=AGGREGATION_LEVEL_MODEL,
        format="json"
    )
    
    # Create output directory if it doesn't exist
    os.makedirs("output", exist_ok=True)
    
    # Write to file
    output_file = "output/aggregated_results.json"
    with open(output_file, "w") as f:
        f.write(json_output)
    print(f"Exported results to {output_file}")
    
    print("\n========== Example Complete ==========")


if __name__ == "__main__":
    main()