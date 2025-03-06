# Model Registry Integration System

**Date: March 7, 2025**  
**Status: Initial Implementation**

## Overview

The Model Registry Integration System provides a framework for integrating test results with a model registry, calculating suitability scores for hardware-model pairs, implementing an automatic hardware recommender based on task requirements, and adding versioning support for model-hardware compatibility. This system builds upon the existing benchmark database to provide a comprehensive model management solution.

## Features

- **Test Results Integration**: Link benchmark results to specific model versions
- **Suitability Scoring**: Calculate compatibility and suitability scores for hardware-model pairs
- **Hardware Recommendation**: Automatically recommend optimal hardware for specific models and tasks
- **Version Control**: Track model versions and hardware compatibility over time

## Database Schema

The system extends the existing benchmark database with the following tables:

### Model Registry Versions Table

Tracks model versions in the registry:

```sql
CREATE TABLE IF NOT EXISTS model_registry_versions (
    id INTEGER PRIMARY KEY,
    model_id INTEGER,
    version_tag VARCHAR,
    version_hash VARCHAR,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSON,
    FOREIGN KEY (model_id) REFERENCES models(id)
)
```

### Hardware-Model Compatibility Table

Stores compatibility and suitability scores for hardware-model pairs:

```sql
CREATE TABLE IF NOT EXISTS hardware_model_compatibility (
    id INTEGER PRIMARY KEY,
    model_id INTEGER,
    hardware_id INTEGER,
    compatibility_score FLOAT,
    suitability_score FLOAT,
    recommended_batch_size INTEGER,
    recommended_precision VARCHAR,
    memory_requirement FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    notes TEXT,
    FOREIGN KEY (model_id) REFERENCES models(id),
    FOREIGN KEY (hardware_id) REFERENCES hardware_platforms(id)
)
```

### Task Recommendations Table

Stores task-specific hardware recommendations:

```sql
CREATE TABLE IF NOT EXISTS task_recommendations (
    id INTEGER PRIMARY KEY,
    task_type VARCHAR,
    model_id INTEGER,
    hardware_id INTEGER,
    suitability_score FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (model_id) REFERENCES models(id),
    FOREIGN KEY (hardware_id) REFERENCES hardware_platforms(id)
)
```

### Hardware Compatibility Snapshots Table

Stores snapshots of hardware compatibility for model versions:

```sql
CREATE TABLE IF NOT EXISTS hardware_compatibility_snapshots (
    id INTEGER PRIMARY KEY,
    model_id INTEGER,
    version_id INTEGER,
    hardware_id INTEGER,
    compatibility_score FLOAT,
    suitability_score FLOAT,
    recommended_batch_size INTEGER,
    recommended_precision VARCHAR,
    memory_requirement FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (model_id) REFERENCES models(id),
    FOREIGN KEY (version_id) REFERENCES model_registry_versions(id),
    FOREIGN KEY (hardware_id) REFERENCES hardware_platforms(id)
)
```

## Components

### ModelRegistryIntegration

Integrates test results with the model registry and calculates suitability scores.

**Key Methods:**
- `create_schema_extensions()`: Creates required database tables
- `link_test_results(model_name, model_version, result_ids)`: Links test results to a model version
- `calculate_suitability_scores(model_id, hardware_id, update_db)`: Calculates compatibility and suitability scores

**Suitability Scoring Algorithm:**
1. Collect performance data for a model-hardware pair
2. Calculate throughput score (normalized to maximum throughput)
3. Calculate latency score (normalized to minimum latency)
4. Calculate memory efficiency score (based on memory usage)
5. Combine scores with weighted formula:
   - compatibility_score = (throughput_score * 0.5) + (latency_score * 0.3) + (memory_score * 0.2)
6. Apply confidence factor based on test count:
   - suitability_score = compatibility_score * confidence_factor

### HardwareRecommender

Recommends hardware based on model and task requirements.

**Key Methods:**
- `recommend_hardware(model_name, task_type, batch_size, latency_sensitive, memory_constrained, top_k)`: Recommends optimal hardware
- `update_task_recommendations(task_type)`: Updates task-specific recommendations

**Recommendation Algorithm:**
1. Retrieve hardware compatibility scores for a model
2. Apply task-specific weighting:
   - For inference: prioritize latency and throughput
   - For training: prioritize throughput and memory efficiency
3. Apply requirement-specific weighting:
   - For latency-sensitive tasks: increase weight of latency score
   - For memory-constrained environments: increase weight of memory efficiency
4. Calculate weighted score and rank hardware options
5. Return top-k recommendations

### VersionControlSystem

Manages model versions and hardware compatibility over time.

**Key Methods:**
- `add_model_version(model_name, version_tag, version_hash, metadata)`: Adds a new model version
- `get_version_history(model_name)`: Gets version history for a model
- `create_compatibility_snapshot(model_name, version_tag)`: Creates a snapshot of hardware compatibility
- `compare_compatibility_versions(model_name, version_tag1, version_tag2)`: Compares hardware compatibility between versions

**Version Control Features:**
- Track model versions with metadata (commit hash, description, etc.)
- Store snapshots of hardware compatibility for each version
- Compare compatibility changes between versions
- Identify significant changes in compatibility or performance

## Usage

### Command-Line Interface

The system provides a command-line interface for various operations:

```bash
# Create schema extensions
python scripts/model_registry_integration.py create-schema --db-path ./benchmark_db.duckdb

# Link test results to a model version
python scripts/model_registry_integration.py link-tests --model "bert-base-uncased" --version "v1.0.0" --result-ids "1,2,3,4"

# Calculate suitability scores
python scripts/model_registry_integration.py calculate-scores --model "bert-base-uncased" --hardware "cuda"

# Recommend hardware for a model
python scripts/model_registry_integration.py recommend --model "bert-base-uncased" --task "inference" --latency-sensitive

# Update task recommendations
python scripts/model_registry_integration.py update-task --task "training"

# Add a model version
python scripts/model_registry_integration.py add-version --model "bert-base-uncased" --version "v1.0.0" --hash "abcdef123456" --metadata '{"author": "user", "description": "Initial release"}'

# Get version history
python scripts/model_registry_integration.py version-history --model "bert-base-uncased"

# Create compatibility snapshot
python scripts/model_registry_integration.py create-snapshot --model "bert-base-uncased" --version "v1.0.0"

# Compare compatibility versions
python scripts/model_registry_integration.py compare-versions --model "bert-base-uncased" --version1 "v1.0.0" --version2 "v1.1.0"
```

### Programmatic API

The system can also be used programmatically:

```python
from scripts.model_registry_integration import (
    ModelRegistryIntegration, 
    HardwareRecommender, 
    VersionControlSystem
)

# Create schema
integration = ModelRegistryIntegration()
integration.create_schema_extensions()

# Link test results
integration.link_test_results("bert-base-uncased", "v1.0.0", [1, 2, 3, 4])

# Calculate suitability scores
scores = integration.calculate_suitability_scores(model_id=1)

# Recommend hardware
recommender = HardwareRecommender()
recommendations = recommender.recommend_hardware("bert-base-uncased", task_type="inference", latency_sensitive=True)

# Add model version
version_control = VersionControlSystem()
version_id = version_control.add_model_version("bert-base-uncased", "v1.0.0", "abcdef123456")

# Compare versions
changes = version_control.compare_compatibility_versions("bert-base-uncased", "v1.0.0", "v1.1.0")
```

## Integration with Model Registry Systems

The Model Registry Integration System is designed to work with various model registry systems:

1. **HuggingFace Model Hub**: Link to models in the HuggingFace Model Hub
2. **MLflow Model Registry**: Integrate with MLflow for enterprise model management
3. **Custom Model Registry**: Support for custom, in-house model registries
4. **GitHub Repository Integration**: Link models to specific GitHub repositories and commits

## Applications

### Automated Hardware Selection

Automatically select the optimal hardware for a given model and task:

```python
from scripts.model_registry_integration import HardwareRecommender

# Initialize recommender
recommender = HardwareRecommender()

# Get recommendations for inference
inference_recommendations = recommender.recommend_hardware(
    model_name="llama-7b",
    task_type="inference",
    batch_size=1,
    latency_sensitive=True
)

# Get recommendations for training
training_recommendations = recommender.recommend_hardware(
    model_name="llama-7b",
    task_type="training",
    batch_size=16,
    memory_constrained=True
)
```

### Version Performance Tracking

Track performance changes across model versions:

```python
from scripts.model_registry_integration import VersionControlSystem

# Initialize version control
version_control = VersionControlSystem()

# Create snapshot for current version
version_control.create_compatibility_snapshot("llama-7b", "v1.0.0")

# Compare with previous version
changes = version_control.compare_compatibility_versions("llama-7b", "v0.9.0", "v1.0.0")

# Check for significant changes
for change in changes:
    if abs(change['compatibility_change']) > 10:
        print(f"Significant change detected for {change['hardware_type']}: {change['compatibility_change']}%")
```

## Implementation Status

- ✅ Database Schema Extensions
- ✅ Test Results Integration
- ✅ Suitability Scoring Algorithm
- ✅ Hardware Recommendation System
- ✅ Version Control System
- 🔄 Integration with Model Hub/Registry (In Progress)
- 🔄 Web API for Hardware Recommendations (In Progress)
- ❓ Advanced Analytics Dashboard (Future)

## Future Enhancements

1. **Advanced Scoring Models**: Implement machine learning models for more accurate suitability scoring
2. **Distributed Training Recommendations**: Enhance recommendations for distributed training setups
3. **Cost Optimization**: Include cost considerations in hardware recommendations
4. **A/B Testing Support**: Add support for A/B testing of model versions
5. **Automated Model Deployment**: Integrate with deployment systems for automated model deployment

## References

- [HARDWARE_SELECTION_GUIDE.md](HARDWARE_SELECTION_GUIDE.md)
- [MODEL_FAMILY_GUIDE.md](MODEL_FAMILY_GUIDE.md)
- [BENCHMARK_DATABASE_GUIDE.md](BENCHMARK_DATABASE_GUIDE.md)
- [NEXT_STEPS.md](NEXT_STEPS.md)