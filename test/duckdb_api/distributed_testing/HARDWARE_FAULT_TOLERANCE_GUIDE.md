# Hardware-Aware Fault Tolerance System

**Implementation Date: March 13, 2025**  
**Original Schedule: June 12-19, 2025 (completed 3 months ahead of schedule)**

## Overview

The Hardware-Aware Fault Tolerance System is a comprehensive solution for handling failures in distributed testing environments with heterogeneous hardware. The system provides specialized recovery strategies for different hardware types (CPUs, GPUs, TPUs, browsers with WebGPU/WebNN), intelligent retry policies, failure pattern detection, task state persistence, and checkpoint/resume capabilities.

This document describes the architecture, components, and usage of the fault tolerance system, focusing on its hardware-aware aspects and integration with the existing Distributed Testing Framework.

## Key Features

1. **Hardware-Specific Recovery Strategies**
   - Tailored recovery approaches for different hardware types (CPU, GPU, TPU, NPU, WebGPU, WebNN)
   - Browser-specific strategies for WebGPU/WebNN environments
   - Error-specific handling for common failure modes (OOM, CUDA errors, browser crashes)

2. **Intelligent Retry Policies**
   - Exponential backoff with jitter for efficient retries
   - Configurable retry limits and delays
   - Progressive recovery strategies that adapt based on failure history

3. **Failure Pattern Detection**
   - Automated detection of recurring failure patterns
   - Classification by hardware type, error type, and worker instance
   - Recommended actions for addressing systemic issues

4. **Task State Persistence**
   - Tracking of task execution state across retries
   - Database integration for persistent state storage
   - Recovery of task state after system restarts

5. **Checkpoint and Resume**
   - Periodic checkpointing of long-running tasks
   - Resume capabilities from the last checkpoint after failure
   - Configurable checkpoint intervals and strategies

6. **Integration with Heterogeneous Hardware Scheduler**
   - Seamless integration with the recently implemented heterogeneous hardware scheduler
   - Hardware-aware task assignments and reassignments
   - Optimization of resource utilization during recovery

## Architecture

The fault tolerance system consists of the following core components:

### 1. HardwareAwareFaultToleranceManager

The central component that orchestrates fault tolerance operations:

- Tracks task state, failure history, and recovery actions
- Manages checkpoints and state persistence
- Analyzes failures to determine appropriate recovery strategies
- Integrates with the coordinator and heterogeneous scheduler

### 2. Failure Types and Recovery Strategies

The system defines a comprehensive taxonomy of failure types and recovery strategies:

**Failure Types**:
- Hardware errors (GPU crashes, TPU failures)
- Software errors (runtime exceptions)
- Resource exhaustion (out of memory, disk space)
- Timeouts
- Communication errors
- Browser failures (WebGPU context lost)
- Worker crashes and disconnections

**Recovery Strategies**:
- Immediate retry on the same worker
- Delayed retry with exponential backoff
- Retry on a different worker with similar capabilities
- Switch to a different hardware class
- Reduce precision (e.g., fp32 → fp16 → int8)
- Reduce batch size
- Fallback to CPU execution
- Browser restart for WebGPU/WebNN
- Human escalation for persistent issues

### 3. Failure Context and Recovery Actions

- `FailureContext`: Captures complete information about a failure (task, worker, hardware, error)
- `RecoveryAction`: Defines the action to take to recover from a failure

### 4. Failure Pattern Detection

The system includes a sophisticated pattern detection mechanism that:

- Analyzes historical failures to identify recurring patterns
- Groups failures by hardware type, error type, and worker instance
- Establishes thresholds for pattern recognition
- Recommends appropriate actions for addressing systemic issues

### 5. Checkpoint Management

Built-in checkpoint capabilities for task state persistence:

- Periodic checkpointing based on configurable intervals
- State synchronization with database for persistence
- Task resumption from the most recent checkpoint

## Hardware-Specific Recovery Strategies

The fault tolerance system implements specialized recovery strategies for different hardware types:

### CPU Failures

1. Delayed retry on the same worker
2. Retry on a different CPU worker
3. Reduce batch size (if applicable)

### GPU Failures

1. Delayed retry on the same GPU
2. Retry on a different GPU worker
3. Reduce precision (fp32 → fp16 → int8)
4. Reduce batch size
5. Fallback to CPU execution

### TPU Failures

1. Delayed retry on the same TPU
2. Retry on a different TPU worker
3. Reduce batch size
4. Fallback to CPU execution

### WebGPU/WebNN Failures

1. Browser restart
2. Delayed retry
3. Retry on a different browser/worker
4. Reduce precision (for WebGPU)
5. Fallback to CPU execution

### Error-Specific Strategies

The system also implements strategies tailored to specific error types:

#### Out of Memory (OOM) Errors

1. Reduce batch size
2. Reduce precision
3. Retry on a worker with more memory
4. Fallback to CPU

#### CUDA Errors

1. Retry on a different GPU worker
2. Delayed retry
3. Fallback to CPU

#### Browser Crashes

1. Restart browser
2. Retry on a different browser
3. Fallback to CPU

#### WebGPU Context Lost

1. Restart browser
2. Retry on a different browser
3. Fallback to CPU

## Usage Guide

### Basic Integration

To integrate the hardware-aware fault tolerance system with your distributed testing environment:

```python
from duckdb_api.distributed_testing.hardware_aware_fault_tolerance import (
    create_recovery_manager, apply_recovery_action
)

# Create a recovery manager
recovery_manager = create_recovery_manager(
    coordinator=coordinator,  # Your coordinator instance
    db_manager=db_manager,    # Your database manager
    scheduler=scheduler,      # Your heterogeneous scheduler
    enable_ml=True            # Enable machine learning-based pattern detection
)

# Handle a task failure
def on_task_failure(task_id, worker_id, error_info):
    # Determine recovery action
    recovery_action = recovery_manager.handle_failure(
        task_id=task_id,
        worker_id=worker_id,
        error_info=error_info
    )
    
    # Apply the recovery action
    apply_recovery_action(
        task_id=task_id,
        action=recovery_action,
        coordinator=coordinator,
        scheduler=scheduler
    )
```

### Custom Configuration

You can customize the fault tolerance system with your own configuration:

```python
# Create a manager with custom configuration
manager = create_recovery_manager(coordinator=coordinator)

# Update the configuration
manager.configure({
    "max_retries": 5,  # Maximum retry attempts
    "base_delay": 1.0,  # Base delay for retries (seconds)
    "max_delay": 30.0,  # Maximum delay for retries (seconds)
    "checkpoint_interval": 600,  # Checkpoint every 10 minutes
    "failure_pattern_threshold": 5,  # Threshold for pattern detection
    "recovery_strategies": {
        # Custom strategy order for each hardware class
        "GPU": [
            RecoveryStrategy.DIFFERENT_WORKER,
            RecoveryStrategy.REDUCED_PRECISION,
            RecoveryStrategy.FALLBACK_CPU
        ],
        # Add more customizations for other hardware classes
    }
})
```

### Task State and Checkpoints

You can work with task state and checkpoints directly:

```python
# Get current state of a task
task_state = recovery_manager.get_task_state(task_id)

# Update task state
recovery_manager.update_task_state(task_id, {
    "progress": 75,
    "completed_steps": 150,
    "current_phase": "validation"
})

# Create a checkpoint
checkpoint_data = {
    "model_state": model.state_dict(),
    "optimizer_state": optimizer.state_dict(),
    "epoch": current_epoch,
    "batch_idx": current_batch,
    "metrics": current_metrics
}
checkpoint_id = recovery_manager.create_checkpoint(task_id, checkpoint_data)

# Get the latest checkpoint for a task
latest_checkpoint = recovery_manager.get_latest_checkpoint(task_id)
if latest_checkpoint:
    # Resume from checkpoint
    model.load_state_dict(latest_checkpoint["model_state"])
    optimizer.load_state_dict(latest_checkpoint["optimizer_state"])
    start_epoch = latest_checkpoint["epoch"]
    start_batch = latest_checkpoint["batch_idx"]
```

### Failure Pattern Analysis

You can access and analyze detected failure patterns:

```python
# Get all detected failure patterns
patterns = recovery_manager.get_failure_patterns()

# Analyze patterns to identify systemic issues
for pattern_id, pattern in patterns.items():
    pattern_type = pattern["type"]  # "hardware_class", "error_type", "worker_id"
    pattern_key = pattern["key"]    # The specific value of the type
    count = pattern["count"]        # Number of failures in this pattern
    recommended_action = pattern["recommended_action"]
    
    print(f"Detected pattern: {pattern_type}={pattern_key}, count={count}")
    print(f"Recommended action: {recommended_action}")
    
    # Take appropriate action based on the pattern
    if pattern_type == "worker_id":
        # Consider taking the worker offline for investigation
        coordinator.disable_worker(pattern_key)
    elif pattern_type == "hardware_class" and pattern_key == "GPU":
        # Consider using CPU workers for affected tasks
        scheduler.update_hardware_preferences({"prefer_cpu": True})
```

### ML-Based Pattern Detection

When the ML-based pattern detection is enabled, you can also work with machine learning patterns:

```python
# Get ML-based patterns
if recovery_manager.enable_ml and recovery_manager.ml_detector:
    ml_patterns = recovery_manager.ml_detector.detect_patterns()
    
    # Process ML-detected patterns
    for pattern in ml_patterns:
        pattern_type = pattern["type"]  # "ml_cluster", "rapid_failure", etc.
        confidence = pattern["confidence"]  # Confidence score (0.0 to 1.0)
        description = pattern["description"]
        
        print(f"ML Pattern ({confidence:.2f}): {description}")
        
        # Take action based on pattern type and confidence
        if pattern_type == "ml_cluster" and confidence > 0.8:
            # High-confidence cluster pattern
            if "hardware_class" in pattern and pattern["hardware_class"] == "GPU":
                # GPU-related pattern
                scheduler.deprioritize_hardware_class("GPU")
        
        elif pattern_type == "rapid_failure" and confidence > 0.7:
            # Tasks failing in rapid succession
            task_id = pattern["task_id"]
            coordinator.pause_task(task_id)
            print(f"Paused task {task_id} due to rapid failures")
```

### Recovery History Analysis

You can analyze the recovery history for a task:

```python
# Get recovery history for a task
recovery_history = recovery_manager.get_recovery_history(task_id)

# Analyze recovery attempts
for i, action in enumerate(recovery_history):
    print(f"Recovery attempt {i+1}: {action.strategy.name}")
    print(f"Message: {action.message}")
    
    # Count strategies by type
    strategy_counts = {}
    for a in recovery_history:
        strategy = a.strategy.name
        strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
    
    print("Strategy distribution:")
    for strategy, count in strategy_counts.items():
        print(f"  {strategy}: {count}")
```

## Examples

### Handling GPU Out-of-Memory Errors

```python
# Task configuration with large batch size
task_config = {
    "model": "bert-large-uncased",
    "batch_size": 32,
    "precision": "fp16",
    "sequence_length": 512
}

# Create task
task_id = coordinator.add_task("benchmark", task_config)

# When an OOM error occurs
error_info = {
    "message": "CUDA out of memory. Tried to allocate 2.00 GiB.",
    "type": "cuda_error"
}

# Handle the failure
recovery_action = recovery_manager.handle_failure(task_id, worker_id, error_info)

# The system will automatically:
# 1. Reduce the batch size (32 → 16)
# 2. Update the task configuration
# 3. Retry the task with the new configuration
apply_recovery_action(task_id, recovery_action, coordinator)

# If it fails again, the system will:
# 1. Further reduce precision (fp16 → int8)
# 2. Update the task configuration
# 3. Retry the task with the new configuration

# If it still fails, the system will:
# 1. Fall back to CPU execution
# 2. Update the hardware requirements
# 3. Retry the task on a CPU worker
```

### Handling WebGPU Browser Crashes

```python
# Task configuration for WebGPU
task_config = {
    "model": "mobilenet_v2",
    "batch_size": 8,
    "browser": "chrome"
}

# Create task
task_id = coordinator.add_task("image_classification", task_config)

# When a browser crash occurs
error_info = {
    "message": "browser crash detected",
    "type": "browser_error"
}

# Handle the failure
recovery_action = recovery_manager.handle_failure(task_id, worker_id, error_info)

# The system will automatically:
# 1. Restart the browser
# 2. Retry the task on the same worker
apply_recovery_action(task_id, recovery_action, coordinator)

# If it fails again, the system will:
# 1. Try a different browser/worker
# 2. Retry the task on the new worker

# If it still fails, the system will:
# 1. Fall back to CPU execution
# 2. Update the hardware requirements
# 3. Retry the task on a CPU worker
```

### Long-Running Task with Checkpoints

```python
# Configure task for long training run
task_config = {
    "model": "t5-large",
    "epochs": 10,
    "checkpoint_every_n_steps": 1000
}

# Create task
task_id = coordinator.add_task("fine_tuning", task_config)

# During task execution, periodically create checkpoints
for epoch in range(10):
    for batch_idx, batch in enumerate(data_loader):
        # Training code...
        
        # Create checkpoint every 1000 steps
        if batch_idx % 1000 == 0:
            checkpoint_data = {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch,
                "batch_idx": batch_idx,
                "metrics": current_metrics
            }
            recovery_manager.create_checkpoint(task_id, checkpoint_data)
        
        # If a failure occurs, the system can resume from the last checkpoint
        # The checkpoint_loop background thread in the recovery manager will
        # automatically create checkpoints based on the configured interval
```

## Visualization and Reporting

The hardware-aware fault tolerance system includes comprehensive visualization capabilities to help users understand failure patterns, recovery strategies, and system performance:

1. **Failure Distribution Visualization**
   - Visual breakdown of failures by type and hardware class
   - Identify the most common failure modes in the system
   - Focus optimization efforts on the most problematic areas

2. **Recovery Strategy Analysis**
   - Visualize the usage and effectiveness of different recovery strategies
   - Compare success rates across strategies and hardware types
   - Identify the most effective recovery approaches for specific failures

3. **Temporal Analysis**
   - Track failures and patterns over time
   - Identify trends and recurring issues
   - Correlate failures with system events or changes

4. **Comprehensive HTML Reports**
   - Interactive HTML reports combining multiple visualizations
   - System statistics and summary metrics
   - Exportable format for sharing and archiving

### Visualization Usage

```python
# Create visualizations from a recovery manager
report_path = recovery_manager.create_visualization(output_dir="./visualizations")

# Access the report in a web browser
import webbrowser
webbrowser.open(f"file://{os.path.abspath(report_path)}")
```

A dedicated script is also provided for generating visualizations from live or simulated data:

```bash
# Generate visualizations from simulated data
python run_fault_tolerance_visualization.py --simulation --output-dir ./visualizations

# Generate and open in browser
python run_fault_tolerance_visualization.py --simulation --open-browser
```

## Integration with Distributed Testing Framework

The hardware-aware fault tolerance system is designed to integrate seamlessly with the existing Distributed Testing Framework:

1. **Coordinator Integration**
   - The recovery manager interacts with the coordinator to manage tasks and workers
   - Recovery actions are applied through the coordinator's API
   - Task state and recovery history are tracked alongside the coordinator's state

2. **Heterogeneous Scheduler Integration**
   - Recovery strategies consider hardware capabilities and affinities
   - The system can update task requirements based on recovery needs
   - Hardware-specific failures inform future scheduling decisions

3. **Database Integration**
   - Task state, failure patterns, and checkpoints are persisted in the database
   - The system can recover state after restarts or crashes
   - Historical data informs pattern detection and optimization

## Implementation Status

The hardware-aware fault tolerance system has been fully implemented and tested. All planned features are complete and operational, including:

- ✅ Hardware-specific recovery strategies for different hardware types
- ✅ Intelligent retry policies with exponential backoff and jitter
- ✅ Failure pattern detection and prevention
- ✅ Task state persistence and recovery
- ✅ Checkpoint and resume for long-running tasks
- ✅ Integration with heterogeneous hardware scheduler
- ✅ Comprehensive test suite validating all components

The system was completed ahead of schedule (March 13, 2025 vs. planned June 12-19, 2025), demonstrating the efficiency and capability of the development team. On the same day, we also added Machine Learning-based pattern detection capabilities, which was originally planned as a future enhancement.

## Machine Learning-Based Pattern Detection

The system now includes advanced pattern detection using machine learning techniques to identify subtle correlations between failures:

- **Feature extraction** from failure contexts (hardware types, error messages, etc.)
- **Clustering of similar failures** for more accurate pattern detection
- **Success rate tracking** for different recovery strategies
- **Recovery strategy recommendation** based on historical success rates
- **Automatic adaptation** to patterns as they emerge

This ML-based system works alongside the traditional pattern detection, providing an extra layer of intelligence for handling failures.

## Future Enhancements

While the current implementation provides comprehensive fault tolerance capabilities, several additional enhancements have been identified for future development:

2. **Fine-Grained Browser Fault Tolerance**
   - Browser-specific error fingerprinting for more targeted recovery
   - WebGPU shader compilation failure recovery
   - Advanced WebNN fallback strategies

3. **Power-Aware Recovery Strategies**
   - Consider power consumption implications of recovery strategies
   - Optimize for energy efficiency during recovery
   - Integrate with thermal management system

4. **Cross-Node Failure Coordination**
   - Coordinate recovery strategies across multiple physical nodes
   - Implement global pattern detection for cross-node issues
   - Develop cluster-wide recovery policies

5. **Predictive Fault Prevention**
   - Predict potential failures before they occur
   - Proactively migrate tasks from at-risk workers
   - Implement preventive maintenance scheduling

## Conclusion

The Hardware-Aware Fault Tolerance System represents a significant advancement in the reliability and robustness of the Distributed Testing Framework. By understanding the specific characteristics and failure modes of different hardware types, the system can apply targeted recovery strategies that maximize the chances of successful task completion while minimizing resource waste.

The system's integration with the heterogeneous hardware scheduler and the coordinator provides a comprehensive solution for fault tolerance in complex distributed testing environments with diverse hardware. Its configurable nature allows for customization to meet specific requirements and preferences.

This implementation solidifies the Distributed Testing Framework's position as a state-of-the-art solution for large-scale testing across heterogeneous hardware environments.