# Predictive Performance System - Integration Quick Reference

**Version:** 1.0  
**Date:** March 10, 2025

This quick reference guide provides concise examples and usage patterns for the integrated Active Learning and Hardware Recommendation system.

## Setup and Initialization

```python
# Import required components
from predictive_performance.active_learning import ActiveLearningSystem
from predictive_performance.hardware_recommender import HardwareRecommender
from predictive_performance.predict import PerformancePredictor

# Initialize components
predictor = PerformancePredictor()
active_learner = ActiveLearningSystem()
hw_recommender = HardwareRecommender(
    predictor=predictor,
    available_hardware=["cpu", "cuda", "rocm", "mps", "openvino", "qnn", "webnn", "webgpu"]
)
```

## Basic Integration Example

```python
# Generate integrated recommendations
results = active_learner.integrate_with_hardware_recommender(
    hardware_recommender=hw_recommender,
    test_budget=10,
    optimize_for="throughput"
)

# Access recommendations
recommendations = results["recommendations"]

# Print top recommendations
for i, config in enumerate(recommendations[:3]):
    print(f"Recommendation #{i+1}: {config['model_name']} on {config['hardware']}")
    print(f"  Recommended hardware: {config['recommended_hardware']}")
    print(f"  Information gain: {config['expected_information_gain']:.4f}")
    print(f"  Combined score: {config['combined_score']:.4f}")
```

## Command-Line Usage

```bash
# Generate integrated recommendations
python example.py integrate --budget 10 --metric throughput --output recommendations.json

# Run with specific parameters
python example.py integrate --budget 20 --metric latency --output latency_recommendations.json
```

## Response Structure

```json
{
  "recommendations": [
    {
      "model_name": "bert-base-uncased",
      "model_type": "text_embedding",
      "hardware": "cuda",
      "batch_size": 8,
      "expected_information_gain": 0.94,
      "uncertainty": 0.89,
      "diversity": 0.65,
      "recommended_hardware": "cuda",
      "hardware_match": true,
      "hardware_score": 0.97,
      "combined_score": 0.95,
      "alternatives": [
        {"hardware": "rocm", "score": 0.85},
        {"hardware": "mps", "score": 0.75}
      ]
    }
  ],
  "total_candidates": 50,
  "enhanced_candidates": 50,
  "final_recommendations": 10,
  "optimization_metric": "throughput",
  "strategy": "integrated_active_learning",
  "timestamp": "2025-03-10T15:30:45.123456"
}
```

## Common Use Cases

### Optimizing for Throughput

```python
# Optimize for maximum throughput
throughput_results = active_learner.integrate_with_hardware_recommender(
    hardware_recommender=hw_recommender,
    test_budget=10,
    optimize_for="throughput"
)
```

### Optimizing for Latency

```python
# Optimize for minimum latency
latency_results = active_learner.integrate_with_hardware_recommender(
    hardware_recommender=hw_recommender,
    test_budget=10,
    optimize_for="latency"
)
```

### Optimizing for Memory Efficiency

```python
# Optimize for memory efficiency
memory_results = active_learner.integrate_with_hardware_recommender(
    hardware_recommender=hw_recommender,
    test_budget=10,
    optimize_for="memory"
)
```

### Finding Hardware Mismatches

```python
# Generate recommendations
results = active_learner.integrate_with_hardware_recommender(
    hardware_recommender=hw_recommender,
    test_budget=20,
    optimize_for="throughput"
)

# Filter for hardware mismatches
mismatches = [
    config for config in results["recommendations"] 
    if not config.get("hardware_match", False)
]

# Print mismatches
for config in mismatches:
    print(f"Mismatch: {config['model_name']} currently on {config['hardware']}")
    print(f"  Recommended hardware: {config['recommended_hardware']}")
    print(f"  Potential improvement: {(config['combined_score']/config['expected_information_gain']-1)*100:.1f}%")
```

### Saving and Loading Results

```python
import json
from datetime import datetime

# Save results to file
def save_results(results, filename=None):
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"integrated_recommendations_{timestamp}.json"
    
    # Convert non-serializable objects to strings
    def prepare_for_json(obj):
        if hasattr(obj, 'isoformat'):
            return obj.isoformat()
        return obj
    
    with open(filename, "w") as f:
        json.dump(results, f, indent=2, default=prepare_for_json)
    
    return filename

# Load results from file
def load_results(filename):
    with open(filename, "r") as f:
        return json.load(f)

# Usage
filename = save_results(results)
loaded_results = load_results(filename)
```

## Using Test Results to Run Benchmarks

```python
# Function to convert recommendations to benchmark commands
def create_benchmark_commands(recommendations, output_dir="benchmark_results"):
    commands = []
    for config in recommendations:
        model = config["model_name"]
        hardware = config["hardware"]
        batch_size = config["batch_size"]
        
        command = (
            f"python run_benchmark.py "
            f"--model {model} "
            f"--hardware {hardware} "
            f"--batch-size {batch_size} "
            f"--output-dir {output_dir}"
        )
        commands.append(command)
    
    return commands

# Generate benchmark commands
commands = create_benchmark_commands(results["recommendations"])

# Print commands
for cmd in commands:
    print(cmd)

# Optionally save to a shell script
with open("run_recommended_benchmarks.sh", "w") as f:
    f.write("#!/bin/bash\n\n")
    for cmd in commands:
        f.write(f"{cmd}\n")
```

## Integration with DuckDB

```python
import duckdb

# Store integrated recommendations in DuckDB
def store_recommendations_in_db(results, db_path="benchmark_db.duckdb"):
    conn = duckdb.connect(db_path)
    
    # Create table if it doesn't exist
    conn.execute("""
    CREATE TABLE IF NOT EXISTS integrated_recommendations (
        id INTEGER PRIMARY KEY,
        timestamp TIMESTAMP,
        model_name VARCHAR,
        model_type VARCHAR,
        hardware VARCHAR,
        batch_size INTEGER,
        recommended_hardware VARCHAR,
        hardware_match BOOLEAN,
        information_gain FLOAT,
        combined_score FLOAT,
        optimization_metric VARCHAR
    )
    """)
    
    # Insert recommendations
    for config in results["recommendations"]:
        conn.execute("""
        INSERT INTO integrated_recommendations 
        (timestamp, model_name, model_type, hardware, batch_size, 
         recommended_hardware, hardware_match, information_gain, 
         combined_score, optimization_metric)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now(),
            config["model_name"],
            config.get("model_type", "unknown"),
            config["hardware"],
            config["batch_size"],
            config.get("recommended_hardware", config["hardware"]),
            config.get("hardware_match", True),
            config.get("expected_information_gain", 0.0),
            config.get("combined_score", 0.0),
            results["optimization_metric"]
        ))
    
    conn.close()

# Usage
store_recommendations_in_db(results)
```

## Common Parameter Combinations

| Use Case | optimize_for | test_budget | Notes |
|----------|--------------|-------------|-------|
| General exploration | "throughput" | 10 | Good starting point |
| Memory optimization | "memory" | 15 | For memory-constrained systems |
| Low-latency systems | "latency" | 10 | For real-time applications |
| Hardware validation | "throughput" | 20 | Focus on hardware mismatches |
| Daily test planning | "throughput" | 5 | Limited testing resources |
| Comprehensive analysis | All metrics | 30 | Run multiple optimizations |

## Further Information

For more detailed documentation, see:
- [PREDICTIVE_PERFORMANCE_GUIDE.md](PREDICTIVE_PERFORMANCE_GUIDE.md)
- [ACTIVE_LEARNING_DESIGN.md](ACTIVE_LEARNING_DESIGN.md)
- [INTEGRATED_ACTIVE_LEARNING_GUIDE.md](INTEGRATED_ACTIVE_LEARNING_GUIDE.md)
- [TESTING_GUIDE.md](TESTING_GUIDE.md)