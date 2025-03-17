# Dynamic Resource Management System

## Overview

The Dynamic Resource Management (DRM) system is a key component of the Distributed Testing Framework, responsible for tracking, allocating, and optimizing computational resources across worker nodes. It enables efficient resource utilization, adaptive scaling, and intelligent task scheduling based on resource requirements.

This document provides a comprehensive technical reference for the Dynamic Resource Management system, including its architecture, components, implementation details, and integration with other parts of the framework.

## Architecture

The Dynamic Resource Management system consists of three main components:

1. **DynamicResourceManager**: Core component for resource tracking, allocation, and scaling decisions
2. **ResourcePerformancePredictor**: ML-based system for predicting resource requirements based on historical data
3. **CloudProviderManager**: Interface for deploying and managing worker nodes across different cloud platforms

These components work together to provide a comprehensive resource management solution for the Distributed Testing Framework.

## DynamicResourceManager

The `DynamicResourceManager` is the central component responsible for tracking and allocating resources across worker nodes.

### Key Features

- **Worker Resource Tracking**: Monitors resources (CPU, memory, GPU) across all worker nodes
- **Resource Reservation**: Allocates and tracks resources for specific tasks
- **Utilization Monitoring**: Calculates system-wide resource utilization metrics
- **Scaling Evaluation**: Determines when to scale the worker pool up or down
- **Resource Matching**: Matches tasks to suitable workers based on resource requirements

### Implementation Details

The `DynamicResourceManager` maintains several data structures to track resources:

- `worker_resources`: Maps worker IDs to resource data
- `resource_reservations`: Maps reservation IDs to reservation data
- `worker_tasks`: Maps worker IDs to sets of task IDs
- `task_reservation`: Maps task IDs to reservation IDs

It also includes background threads for continuous resource tracking and worker reassessment:

- `resource_tracker_thread`: Continuously monitors resource usage and updates utilization metrics
- `worker_reassessment_thread`: Periodically reassesses worker capabilities and resources

### Key Methods

- `register_worker`: Register a worker with its resource capabilities
- `update_worker_resources`: Update a worker's available resources
- `reserve_resources`: Reserve resources for a specific task
- `release_resources`: Release resources after task completion
- `calculate_task_worker_fitness`: Calculate fitness score for a worker-task pair
- `check_resource_availability`: Check if a worker has sufficient resources for a task
- `evaluate_scaling`: Evaluate whether to scale the worker pool up or down

## ResourcePerformancePredictor

The `ResourcePerformancePredictor` uses machine learning to predict resource requirements for tasks based on historical execution data.

### Key Features

- **Data Recording**: Records task execution data, including resource usage patterns
- **Model Training**: Trains ML models to predict resource requirements based on task parameters
- **Resource Prediction**: Predicts resource requirements for new tasks
- **Fallback Mechanisms**: Falls back to statistical methods or default values when ML prediction isn't possible

### Implementation Details

The predictor uses a tiered approach to prediction:

1. **ML-Based Prediction**: Uses trained models when sufficient data is available
2. **Statistics-Based Prediction**: Uses statistical aggregation when moderate data is available
3. **Default Values**: Uses reasonable defaults when minimal data is available

It maintains historical data in a local SQLite database and trains models in the background as more data becomes available.

### Key Methods

- `record_task_execution`: Record resource usage data for a task execution
- `predict_resource_requirements`: Predict resource requirements for a task
- `train_models`: Train or update ML models based on historical data
- `calculate_batch_scaling_factor`: Calculate how resource requirements scale with batch size

## CloudProviderManager

The `CloudProviderManager` provides a unified interface for deploying and managing worker nodes across different cloud platforms.

### Key Features

- **Multi-Provider Support**: Supports AWS EC2, Google Cloud Platform, and local Docker containers
- **Worker Deployment**: Creates worker nodes on specified platforms
- **Worker Termination**: Terminates worker nodes when they're no longer needed
- **Status Monitoring**: Monitors the status of deployed workers
- **Resource Discovery**: Discovers available resources on cloud platforms

### Implementation Details

The manager uses a plugin architecture with provider-specific implementations:

- `AWSCloudProvider`: Implementation for Amazon Web Services EC2
- `GCPCloudProvider`: Implementation for Google Cloud Platform
- `DockerLocalProvider`: Implementation for local Docker containers

Each provider implements a common interface defined by the abstract `CloudProviderBase` class.

### Key Methods

- `add_provider`: Register a cloud provider with the manager
- `create_worker`: Create a worker on a specified cloud platform
- `terminate_worker`: Terminate a worker on a specified cloud platform
- `get_worker_status`: Get the status of a worker on a specified cloud platform
- `get_available_resources`: Get available resources on a specified cloud platform

## Integration with Coordinator

The Dynamic Resource Management system is integrated with the coordinator server, which handles the coordination between workers and tasks.

### Registration Integration

When a worker registers with the coordinator, its resource information is passed to the Dynamic Resource Manager:

```python
# In Coordinator._process_message
if message_type == "register":
    worker_id = message.get("worker_id")
    resources = message.get("resources", {})
    
    # Register worker with resource information
    success = self.worker_manager.register_worker(
        worker_id, hostname, capabilities, websocket, tags, resources
    )
    
    # Register with dynamic resource manager if available
    if hasattr(self, 'dynamic_resource_manager') and self.dynamic_resource_manager and resources:
        self.dynamic_resource_manager.register_worker(worker_id, resources)
```

### Heartbeat Integration

Worker resources are updated during heartbeats:

```python
# In Coordinator._process_message
elif message_type == "heartbeat":
    worker_id = message.get("worker_id")
    resources = message.get("resources", {})
    
    # Update dynamic resource manager if available
    if hasattr(self, 'dynamic_resource_manager') and self.dynamic_resource_manager and resources:
        self.dynamic_resource_manager.update_worker_resources(worker_id, resources)
```

### Task Scheduling Integration

The task manager uses the Dynamic Resource Manager for resource-aware task scheduling:

```python
# In TaskManager.get_next_task
# Calculate fitness scores for each task-worker pair
task_fitness_scores = []
for i, task in enumerate(self.task_queue):
    if dynamic_resource_mgr and worker_resources:
        task_resources = self._estimate_task_resources(task, resource_predictor)
        fitness_score = dynamic_resource_mgr.calculate_task_worker_fitness(
            worker_id, task_resources
        )
        task_fitness_scores.append((i, fitness_score, task))

# Sort by fitness score (descending)
if task_fitness_scores:
    task_fitness_scores.sort(key=lambda x: -x[1])
    task_index, fitness_score, matching_task = task_fitness_scores[0]
    
    # Reserve resources
    if dynamic_resource_mgr and worker_resources:
        reservation_id = dynamic_resource_mgr.reserve_resources(
            worker_id=worker_id,
            task_id=matching_task["task_id"],
            resource_requirements=task_resources
        )
        matching_task["resource_reservation_id"] = reservation_id
```

### Task Completion Integration

Resources are released after task completion:

```python
# In TaskManager.complete_task and TaskManager.fail_task
# Release resources if there was a reservation
if "resource_reservation_id" in task:
    if dynamic_resource_mgr:
        dynamic_resource_mgr.release_resources(task["resource_reservation_id"])
```

### Scaling Integration

The coordinator periodically evaluates scaling decisions:

```python
# In CoordinatorServer._scaling_evaluation_loop
scaling_decision = self.dynamic_resource_manager.evaluate_scaling()

if scaling_decision.action == "scale_up":
    # Need to scale up, provision new workers
    for i in range(scaling_decision.count):
        self.cloud_provider_manager.create_worker(
            provider=provider_name,
            resources=scaling_decision.resource_requirements,
            worker_type=scaling_decision.worker_type
        )
elif scaling_decision.action == "scale_down":
    # Need to scale down, terminate excess workers
    for worker_id in scaling_decision.worker_ids:
        self.cloud_provider_manager.terminate_worker(
            provider=provider_name,
            worker_id=worker_id
        )
```

## Resource Reporting from Workers

Workers report their resource information during registration and heartbeats:

### Resource Detection

The worker uses the `HardwareDetector` class to detect available resources:

```python
# Worker initializes hardware detector
self.hardware_detector = HardwareDetector()
self.capabilities = self.hardware_detector.get_capabilities()
```

### Resource Reporting

The worker continuously monitors resource usage and reports it during heartbeats:

```python
# Get updated hardware metrics
hardware_metrics = self.task_runner._get_hardware_metrics()

# Send heartbeat request with resource updates
heartbeat_request = {
    "type": "heartbeat",
    "worker_id": self.worker_id,
    "timestamp": datetime.now().isoformat(),
    "resources": hardware_metrics.get("resources", {}),
    "hardware_metrics": {
        "cpu_percent": hardware_metrics.get("cpu_percent", 0),
        "memory_percent": hardware_metrics.get("memory_percent", 0),
        "gpu_utilization": hardware_metrics.get("gpu_utilization", 0)
    }
}
```

### Resource Metrics Collection

The worker collects detailed resource metrics using `psutil`, `GPUtil`, and other libraries:

```python
def _get_hardware_metrics(self) -> Dict[str, Any]:
    """Get current hardware metrics and resource information."""
    metrics = {
        "timestamp": datetime.now().isoformat()
    }
    
    # Add resource metrics for dynamic resource management
    resources = {
        "cpu": {},
        "memory": {},
        "gpu": {}
    }
    
    # Collect CPU metrics
    if PSUTIL_AVAILABLE:
        cpu_count = psutil.cpu_count(logical=True)
        cpu_physical = psutil.cpu_count(logical=False)
        cpu_load = [x / 100.0 for x in psutil.getloadavg()]
        
        resources["cpu"]["cores"] = cpu_count
        resources["cpu"]["physical_cores"] = cpu_physical
        resources["cpu"]["available_cores"] = max(0.1, cpu_count - cpu_load[0])
        
    # Collect memory metrics
    if PSUTIL_AVAILABLE:
        memory = psutil.virtual_memory()
        resources["memory"]["total_mb"] = int(memory.total / (1024 * 1024))
        resources["memory"]["available_mb"] = int(memory.available / (1024 * 1024))
        
    # Collect GPU metrics
    if GPUTIL_AVAILABLE:
        gpus = GPUtil.getGPUs()
        resources["gpu"]["devices"] = len(gpus)
        resources["gpu"]["available_devices"] = 0
        resources["gpu"]["total_memory_mb"] = 0
        resources["gpu"]["available_memory_mb"] = 0
        
        for gpu in gpus:
            available_memory_mb = gpu.memoryTotal - gpu.memoryUsed
            resources["gpu"]["total_memory_mb"] += gpu.memoryTotal
            resources["gpu"]["available_memory_mb"] += available_memory_mb
    
    # Add resources info to metrics
    metrics["resources"] = resources
    
    return metrics
```

## Configuration

The Dynamic Resource Management system is highly configurable:

### DynamicResourceManager Configuration

```python
# Default configuration
DEFAULT_TARGET_UTILIZATION = 0.7  # 70% target utilization
DEFAULT_SCALE_UP_THRESHOLD = 0.8  # Scale up at 80% utilization
DEFAULT_SCALE_DOWN_THRESHOLD = 0.3  # Scale down at 30% utilization
DEFAULT_EVALUATION_WINDOW = 300  # 5 minutes
DEFAULT_SCALE_UP_COOLDOWN = 300  # 5 minutes
DEFAULT_SCALE_DOWN_COOLDOWN = 600  # 10 minutes
DEFAULT_WORKER_REASSESSMENT_INTERVAL = 3600  # 1 hour
DEFAULT_HISTORY_RETENTION = 86400  # 24 hours
```

### ResourcePerformancePredictor Configuration

```python
# Default configuration
DEFAULT_PREDICTION_CONFIDENCE_THRESHOLD = 0.7  # Minimum confidence threshold
DEFAULT_PREDICTION_UPDATE_INTERVAL = 3600  # 1 hour
DEFAULT_MIN_SAMPLES_FOR_ML = 30  # Minimum samples for ML prediction
DEFAULT_MIN_SAMPLES_FOR_STATS = 5  # Minimum samples for statistical prediction
```

### CloudProviderManager Configuration

```python
# Example configuration file
{
  "aws": {
    "region": "us-west-2",
    "instance_types": {
      "cpu": "c5.xlarge",
      "gpu": "g4dn.xlarge"
    },
    "credentials": {
      "access_key_id": "${AWS_ACCESS_KEY_ID}",
      "secret_access_key": "${AWS_SECRET_ACCESS_KEY}"
    },
    "spot_instance_enabled": true
  },
  "gcp": {
    "project": "your-project-id",
    "zone": "us-central1-a",
    "machine_types": {
      "cpu": "n2-standard-4",
      "gpu": "n1-standard-4-nvidia-tesla-t4"
    },
    "credentials_file": "/path/to/credentials.json",
    "preemptible_enabled": true
  },
  "docker_local": {
    "image": "ipfs-accelerate-worker:latest",
    "cpu_limit": 4,
    "memory_limit": "16g",
    "network": "host"
  }
}
```

## Future Enhancements

Planned enhancements for the Dynamic Resource Management system include:

1. **Advanced Cost Optimization**: Multi-dimensional optimization considering both performance and cost
2. **Predictive Scaling**: Proactive scaling based on predicted workload patterns
3. **Resource Reservation Quotas**: Enforce limits on resource reservation to prevent monopolization
4. **Heterogeneous Resource Management**: Better support for specialized hardware like TPUs
5. **Cross-Region Optimization**: Optimize worker deployment across multiple regions for latency and cost
6. **Integration with Kubernetes**: Deploy workers as Kubernetes pods

## Conclusion

The Dynamic Resource Management system provides a powerful foundation for efficient resource utilization in the Distributed Testing Framework. By tracking resources, matching tasks to workers, and adaptively scaling the worker pool, it ensures optimal performance and cost efficiency for distributed testing workloads.

For more information on using the Dynamic Resource Management system, please refer to the [Distributed Testing Guide](DISTRIBUTED_TESTING_GUIDE.md).