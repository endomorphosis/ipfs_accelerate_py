# Intelligent Task Distribution for Distributed Testing Framework

This document provides detailed information about the intelligent task distribution system implemented in Phase 3 of the Distributed Testing Framework. The system enables sophisticated task scheduling based on worker capabilities, task requirements, and historical performance data.

## Overview

The intelligent task distribution system improves workload distribution by:

1. Matching tasks to workers based on hardware capabilities
2. Tracking and utilizing worker specialization for specific tasks
3. Grouping similar tasks together for better resource utilization
4. Using historical performance data to predict execution times
5. Dynamically adapting to changing workloads and worker conditions

## Key Components

### Task Affinity

Task affinity helps group similar tasks together for better cache utilization and performance. This is particularly beneficial for:

- Models from the same family (e.g., BERT, T5, LLAMA)
- Tasks that use similar datasets or frameworks
- Tasks that benefit from shared resources or cached data

Task affinity is implemented through:

- A pre-initialized affinity map for known model relationships
- Dynamic updates based on successful task executions
- Adaptive scoring in the worker selection algorithm

### Worker Specialization

Worker specialization tracks which workers perform best for specific task types or models. This enables:

- Improved task assignment based on past performance
- Better resource utilization by matching tasks to specialized workers
- Learning from historical execution patterns

Worker specialization scores are updated based on:

- Task success/failure outcomes
- Execution time metrics
- Stability and reliability factors

### Predictive Scheduling

Predictive scheduling uses historical data to estimate task execution times. This enables:

- More accurate load balancing
- Better task distribution based on estimated completion times
- Improved overall system throughput

Execution time predictions are made using:

- Historical execution times for similar tasks
- Worker-specific performance metrics
- Task-specific factors (model size, batch size, etc.)
- Default estimates for unknown task types

### Advanced Scoring Algorithm

The advanced scoring algorithm combines multiple factors to find the optimal worker for each task:

- Hardware compatibility (mandatory requirements)
- Memory availability and margins
- Computational capability matching
- Task affinity with currently running tasks
- Worker specialization scores
- Current workload and utilization
- Thermal and power considerations
- Estimated execution time and efficiency

## Configuration Options

The intelligent task scheduler can be configured with the following options:

```python
task_scheduler = TaskScheduler(
    coordinator,
    prioritize_hardware_match=True,      # Prioritize hardware compatibility
    load_balance=True,                   # Enable load balancing
    consider_worker_performance=True,    # Consider past worker performance
    max_tasks_per_worker=2,              # Maximum tasks per worker
    enable_task_affinity=True,           # Enable task affinity features
    enable_worker_specialization=True,   # Enable worker specialization tracking
    enable_predictive_scheduling=True    # Enable predictive execution time estimation
)
```

## Task Distribution Process

The intelligent task distribution process follows these steps:

1. Tasks are sorted by priority (higher priority tasks are scheduled first)
2. For each task, the system:
   - Identifies compatible workers based on hardware requirements
   - Calculates a comprehensive score for each compatible worker
   - Selects the worker with the highest score
   - Assigns the task to the selected worker
3. The system continuously updates performance metrics and specialization scores

## Hardware Capability Matching

The system matches task requirements with worker capabilities, considering:

- Required hardware types (CPU, CUDA, ROCm, etc.)
- Minimum memory requirements
- CUDA compute capability (for CUDA tasks)
- Specific device requirements
- Additional hardware features

## Performance Metrics and Statistics

The intelligent scheduler provides detailed performance metrics and statistics through the `get_scheduler_stats()` method, including:

- Task type and status counts
- Worker utilization statistics
- Average execution times by task type
- Worker specialization metrics
- Task affinity statistics
- Model family distribution
- Performance metric counts

## Example Usage

### Running the Test Script

To test the intelligent task scheduler with various workers and tasks:

```bash
python run_test_intelligent_scheduler.py --num-workers 4 --run-time 120
```

This script:
1. Creates a coordinator with intelligent scheduling enabled
2. Starts multiple workers with different hardware capabilities
3. Submits various benchmark and test tasks
4. Demonstrates dynamic task assignment based on the intelligent scheduling algorithm

### Configuration in Coordinator

The intelligent scheduler is enabled by default in the coordinator:

```python
# Configure task scheduler if enabled
if coordinator.enable_advanced_scheduler and coordinator.task_scheduler:
    coordinator.task_scheduler.max_tasks_per_worker = args.max_tasks_per_worker
    
    # Enable intelligent scheduling features by default
    coordinator.task_scheduler.enable_task_affinity = True
    coordinator.task_scheduler.enable_worker_specialization = True
    coordinator.task_scheduler.enable_predictive_scheduling = True
```

## Benefits of Intelligent Task Distribution

The intelligent task distribution system provides several benefits:

1. **Improved Performance**: Tasks are assigned to the most suitable workers, leading to better overall performance.
2. **Optimized Resource Utilization**: Resources are utilized more efficiently by matching tasks to appropriate workers.
3. **Better Load Balancing**: Workloads are distributed more evenly considering worker capabilities and current load.
4. **Reduced Task Execution Time**: Similar tasks benefit from cache locality and worker specialization.
5. **Adaptability**: The system adapts to changing workloads and worker conditions over time.
6. **Learning from Experience**: Performance improves over time as the system learns from task execution history.

## Implementation Details

### Task Affinity Map

The task affinity map contains relationships between models from the same family:

```python
# Text embedding models
"bert-base-uncased": ["bert-base-cased", "bert-large-uncased", "roberta-base", "distilbert-base-uncased"],
"roberta-base": ["roberta-large", "bert-base-uncased", "distilbert-base-uncased"],

# Text generation models
"t5-small": ["t5-base", "t5-large", "t5-3b", "t5-11b"],
"llama-7b": ["llama-13b", "llama-30b", "llama-65b", "opt-125m", "opt-1.3b"],

# Vision models
"vit-base": ["vit-large", "vit-huge", "deit-base"],
"clip-vit": ["clip-resnet", "vit-base", "blip-vit"],

# Audio models
"whisper-tiny": ["whisper-base", "whisper-small", "whisper-medium"],
"wav2vec2-base": ["wav2vec2-large", "hubert-base", "hubert-large"],

# Multimodal models
"llava-onevision-base": ["llava-onevision-large", "llava-1.5", "clip-vit", "llama-7b"],
"blip-base": ["blip-large", "clip-vit", "vit-base", "t5-base"]
```

### Scoring Factors

The scoring algorithm uses various factors with configurable weights:

```python
self.hardware_scoring_factors = {
    "hardware_match": 5.0,      # Base score for matching required hardware
    "memory_margin": 0.5,       # Score factor for available memory margin
    "compute_capability": 0.3,  # Score factor for CUDA compute capability
    "cores": 0.2,               # Score factor for CPU cores
    "device_match": 1.0,        # Score for exact device match
    "specialization": 2.0,      # Score for worker specialization
    "affinity": 1.5,            # Score for task affinity
    "efficiency": 1.0           # Score for energy/thermal efficiency
}
```

## Future Enhancements

Future enhancements to the intelligent task distribution system may include:

1. **Machine Learning-Based Predictions**: More sophisticated ML models for execution time prediction
2. **Dynamic Factor Adjustment**: Automatic adjustment of scoring factors based on system performance
3. **Task Dependency Handling**: Enhanced scheduling for tasks with dependencies
4. **Resource Reservation**: Preemptive resource reservation for high-priority tasks
5. **Multi-Objective Optimization**: Balancing multiple objectives like throughput, latency, and energy efficiency
6. **User-Defined Scheduling Policies**: Customizable scheduling policies for different use cases

## Conclusion

The intelligent task distribution system significantly improves the Distributed Testing Framework's ability to efficiently distribute and execute tasks across heterogeneous workers. By considering hardware capabilities, worker specialization, task affinities, and historical performance, the system achieves better resource utilization and overall performance.