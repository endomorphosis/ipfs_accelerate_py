# Dynamic Resource Management System

## Overview

The Dynamic Resource Management (DRM) system is a key component of the Distributed Testing Framework, responsible for tracking, allocating, and optimizing computational resources across worker nodes. It enables efficient resource utilization, adaptive scaling, and intelligent task scheduling based on resource requirements.

This document provides a comprehensive technical reference for the Dynamic Resource Management system, including its architecture, components, implementation details, and integration with other parts of the framework.

## Architecture

The Dynamic Resource Management system consists of four main components:

1. **DynamicResourceManager**: Core component for resource tracking, allocation, and scaling decisions
2. **ResourcePerformancePredictor**: ML-based system for predicting resource requirements based on historical data
3. **CloudProviderManager**: Interface for deploying and managing worker nodes across different cloud platforms
4. **ResourceOptimizer**: Integration component that combines the above components to enable intelligent resource allocation and workload optimization

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

## ResourceOptimizer

The `ResourceOptimizer` integrates the DynamicResourceManager, ResourcePerformancePredictor, and CloudProviderManager components to provide intelligent resource allocation and workload optimization.

### Key Features

- **Task Resource Prediction**: Uses historical performance data to predict resource requirements for tasks
- **Intelligent Resource Allocation**: Allocates optimal resources for task batches
- **Worker Type Recommendations**: Recommends worker types based on pending tasks
- **Workload Pattern Analysis**: Analyzes workload patterns to optimize resource allocation
- **Enhanced Scaling Recommendations**: Provides scaling recommendations based on workload patterns
- **Task Result Recording**: Records task execution results to improve future predictions

### Implementation Details

The ResourceOptimizer uses several data structures and algorithms:

- `TaskRequirements`: Structured representation of task resource requirements
- `WorkerTypeRecommendation`: Recommendations for worker types based on workload
- `ResourceAllocation`: Results of resource allocation attempts
- Fitness scoring algorithms for matching tasks to workers
- Resource capacity calculation for different worker types
- Workload clustering for batch optimization (uses K-means when available)

### Key Methods

- `predict_task_requirements`: Predicts resource requirements for a task
- `allocate_resources`: Allocates resources for a batch of tasks across available workers
- `recommend_worker_types`: Recommends worker types based on pending tasks
- `get_scaling_recommendations`: Provides enhanced scaling recommendations
- `record_task_result`: Records task execution results for future optimization

### Implementation Details

The CloudProviderManager uses a plugin architecture with provider-specific implementations:

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

## Testing Framework

The Dynamic Resource Management system includes a comprehensive testing framework to verify functionality, performance, and reliability.

### Test Components

1. **Unit Tests**: Individual component tests that verify specific functionality in isolation
2. **Integration Tests**: Tests that verify the interaction between DRM components
3. **End-to-End Tests**: Simulated environments that test the complete system workflow
4. **Performance Tests**: Tests that verify system efficiency under various loads

### Key Test Files

```
duckdb_api/distributed_testing/tests/
├── run_drm_tests.py              # Main test runner for all DRM tests
├── run_e2e_drm_test.py           # End-to-end test runner for real environment simulation
├── run_resource_optimization_tests.py  # ResourceOptimizer-specific test runner
├── test_cloud_provider_manager.py # Tests for CloudProviderManager
├── test_drm_integration.py       # Integration tests for DRM system
├── test_dynamic_resource_manager.py # Tests for DynamicResourceManager
├── test_resource_optimization.py  # Tests for ResourceOptimizer
└── test_resource_performance_predictor.py # Tests for ResourcePerformancePredictor
```

### Running Tests

```bash
# Run all DRM tests
python run_drm_tests.py

# Run tests for a specific component
python run_drm_tests.py --pattern resource_optimization

# Run end-to-end tests
python run_e2e_drm_test.py

# Run ResourceOptimizer tests
python run_resource_optimization_tests.py --verbose
```

For more detailed information on the testing framework, see [DYNAMIC_RESOURCE_MANAGEMENT_TESTING.md](DYNAMIC_RESOURCE_MANAGEMENT_TESTING.md).

## Real-Time Performance Metrics Dashboard

The Dynamic Resource Management system includes a comprehensive real-time dashboard for monitoring system performance, resource utilization, and scaling decisions.

### Dashboard Components

1. **DRM Real-Time Dashboard** (`drm_real_time_dashboard.py`): Interactive web dashboard for real-time monitoring
2. **Dashboard Runner** (`run_drm_real_time_dashboard.py`): Command-line script for launching the dashboard
3. **Data Collection Engine**: Background process for collecting and processing metrics
4. **Regression Detection System**: Statistical analysis for detecting performance anomalies
5. **Visualization Components**: Interactive charts and graphs for data visualization

### Key Features

- **Real-Time Resource Monitoring**: Live tracking of CPU, memory, and GPU utilization across workers
- **Worker-Level Metrics**: Detailed per-worker performance tracking and comparison
- **Performance Analytics**: Statistical analysis of task throughput and resource efficiency
- **Scaling Decision Visualization**: Timeline of scaling events with reasoning
- **Alerting System**: Automatic detection and alerting for performance anomalies

### Dashboard Sections

- **System Overview**: High-level metrics with trend indicators
- **Worker Details**: Detailed worker metrics with comparison capabilities
- **Performance Metrics**: Throughput, allocation time, and efficiency metrics
- **Scaling Decisions**: Visualization of scaling history and impact
- **Alerts and Notifications**: System alerts with severity classification

### Running the Dashboard

```bash
# Run with default settings
python run_drm_real_time_dashboard.py

# Run with custom configuration
python run_drm_real_time_dashboard.py --port 8085 --theme dark --update-interval 5 --retention 60 --browser
```

For detailed documentation on the real-time dashboard, see [REAL_TIME_PERFORMANCE_METRICS_DASHBOARD.md](REAL_TIME_PERFORMANCE_METRICS_DASHBOARD.md).

## CI/CD Integration

The Dynamic Resource Management system includes comprehensive CI/CD integration for automated testing, reporting, and status tracking.

### CI/CD Components

1. **DRM CI/CD Integration Module** (`drm_cicd_integration.py`): Extends the base CI/CD integration with DRM-specific functionality
2. **CI/CD Configuration Templates**:
   - GitHub Actions workflow (`drm_github_workflow.yml`)
   - GitLab CI configuration (`drm_gitlab_ci.yml`)
   - Jenkins pipeline (`drm_jenkinsfile`)
3. **Badge Generator**: Generates status badges for GitHub repositories

### CI/CD Test Stages

1. **Unit Tests**: Test individual DRM components in isolation
2. **Integration Tests**: Test interactions between components
3. **Performance Tests**: Measure system efficiency under various loads
4. **End-to-End Tests**: Test complete system workflow in a simulated environment
5. **Distributed Tests**: Use the Distributed Testing Framework for testing

### Status Badges

Status badges provide a visual indication of the health of the DRM system:

```markdown
![DRM Tests](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/username/repo/main/.github/badges/drm_status.json)
```

### Test Reports

The CI/CD integration generates comprehensive test reports with:
- Component test status
- Resource allocation metrics
- Scaling decision history
- Worker utilization visualization
- Performance benchmarks

For more detailed information on the CI/CD integration, see [duckdb_api/distributed_testing/CI_CD_INTEGRATION_SUMMARY.md](duckdb_api/distributed_testing/CI_CD_INTEGRATION_SUMMARY.md).

## Future Enhancements

Planned enhancements for the Dynamic Resource Management system include:

1. **Advanced Cost Optimization**: Multi-dimensional optimization considering both performance and cost
2. **Predictive Scaling**: Proactive scaling based on predicted workload patterns
3. **Resource Reservation Quotas**: Enforce limits on resource reservation to prevent monopolization
4. **Heterogeneous Resource Management**: Better support for specialized hardware like TPUs
5. **Cross-Region Optimization**: Optimize worker deployment across multiple regions for latency and cost
6. **Integration with Kubernetes**: Deploy workers as Kubernetes pods
7. **Machine Learning-Based Anomaly Detection**: Enhanced anomaly detection using ML techniques
8. **External Monitoring Integration**: Integration with Grafana, Prometheus, and other monitoring systems
9. **Customizable Dashboard Layouts**: User-configurable dashboard layouts and visualizations
10. **Alerting Integrations**: Integration with notification systems (Slack, Email, PagerDuty)

## Conclusion

The Dynamic Resource Management system provides a powerful foundation for efficient resource utilization in the Distributed Testing Framework. By tracking resources, matching tasks to workers, and adaptively scaling the worker pool, it ensures optimal performance and cost efficiency for distributed testing workloads.

For more information on using the Dynamic Resource Management system, please refer to the [Distributed Testing Guide](DISTRIBUTED_TESTING_GUIDE.md).