# Dynamic Resource Management for Distributed Testing Framework

This document provides detailed information about the Dynamic Resource Management (DRM) system for the Distributed Testing Framework, including the MultiDeviceOrchestrator and its integration with the CoordinatorServer.

## Table of Contents

1. [Introduction](#introduction)
2. [System Architecture](#system-architecture)
3. [Components](#components)
   - [MultiDeviceOrchestrator](#multideviceorchestrator)
   - [DynamicResourceManager](#dynamicresourcemanager)
   - [ResourcePerformancePredictor](#resourceperformancepredictor)
   - [CloudProviderIntegration](#cloudproviderintegration)
   - [CoordinatorOrchestratorIntegration](#coordinatororchestratorintegration)
   - [ResourceOptimizer](#resourceoptimizer)
4. [Orchestration Strategies](#orchestration-strategies)
5. [API Reference](#api-reference)
6. [Usage Examples](#usage-examples)
7. [Advanced Features](#advanced-features)
8. [Testing Framework](#testing-framework)
9. [Best Practices](#best-practices)

## Introduction

The Dynamic Resource Management (DRM) system is a core component of the Distributed Testing Framework that enables efficient allocation and utilization of computational resources across heterogeneous hardware environments. The system provides capabilities for orchestrating complex tasks, managing resource allocation, predicting performance requirements, and dynamically scaling resources based on workload demands.

**Key Features:**

- Multi-device task orchestration with 5 different splitting strategies
- Resource-aware task scheduling based on worker capabilities
- Hardware-specific task targeting for optimal performance
- Intelligent resource allocation and optimization
- Automatic resource scaling based on workload patterns
- Performance prediction using machine learning
- Fault tolerance and error recovery
- Cloud provider integration for dynamic worker provisioning
- Monitoring and visualization of resource utilization

## System Architecture

The DRM system is composed of several interconnected components that work together to provide a comprehensive resource management solution:

```
┌─────────────────────────────────────────────────────────────────┐
│                       Coordinator Server                        │
│                                                                 │
│  ┌─────────────────────┐  ┌────────────────────────────────┐   │
│  │   Task Management   │  │      Worker Management         │   │
│  └─────────────────────┘  └────────────────────────────────┘   │
│              │                           │                      │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │              Coordinator-Orchestrator Integration          │ │
│  └───────────────────────────────────────────────────────────┘ │
│                             │                                   │
└─────────────────────────────┼───────────────────────────────────┘
                              │
┌─────────────────────────────┼───────────────────────────────────┐
│                             ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                MultiDeviceOrchestrator                  │   │
│  │                                                         │   │
│  │  ┌───────────────┐  ┌───────────────┐  ┌────────────┐  │   │
│  │  │ Task Splitting │  │Result Merging │  │ Monitoring │  │   │
│  │  └───────────────┘  └───────────────┘  └────────────┘  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                             │                                   │
│  ┌─────────────────────────┼─────────────────────────────────┐ │
│  │                         ▼                                 │ │
│  │  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐ │ │
│  │  │    Dynamic    │  │   Resource    │  │     Cloud     │ │ │
│  │  │   Resource    │◄─┤  Performance  │◄─┤   Provider    │ │ │
│  │  │   Manager     │  │   Predictor   │  │  Integration  │ │ │
│  │  └───────────────┘  └───────────────┘  └───────────────┘ │ │
│  │         Dynamic Resource Management Components           │ │
│  └─────────────────────────────────────────────────────────┘ │
│                   Resource Management System                  │
└─────────────────────────────────────────────────────────────────┘
```

## Components

### MultiDeviceOrchestrator

The MultiDeviceOrchestrator is the core component responsible for orchestrating complex tasks across multiple worker nodes. It provides capabilities for:

1. Splitting tasks into subtasks based on various strategies
2. Scheduling subtasks across different workers based on their capabilities
3. Tracking and managing subtask execution
4. Merging results from subtasks into a coherent final result
5. Handling failures and recovery

**Key Classes and Enums:**

- `SplitStrategy`: Enumeration defining task splitting strategies:
  - `DATA_PARALLEL`: Split input data across workers
  - `MODEL_PARALLEL`: Split model across workers
  - `PIPELINE_PARALLEL`: Process data in stages across workers
  - `ENSEMBLE`: Run multiple versions in parallel
  - `FUNCTION_PARALLEL`: Split different functions across workers

- `TaskStatus`: Enumeration for orchestrated task status:
  - `PENDING`: Not yet started
  - `SPLITTING`: Being split into subtasks
  - `IN_PROGRESS`: At least one subtask is running
  - `MERGING`: Combining subtask results
  - `COMPLETED`: All subtasks completed successfully
  - `FAILED`: At least one subtask failed
  - `CANCELLED`: Task was cancelled

- `SubtaskStatus`: Enumeration for subtask status:
  - `PENDING`: Not yet assigned
  - `ASSIGNED`: Assigned to a worker
  - `RUNNING`: Currently executing
  - `COMPLETED`: Successfully completed
  - `FAILED`: Failed to complete
  - `CANCELLED`: Subtask was cancelled

**API Example:**

```python
# Initialize the orchestrator
orchestrator = MultiDeviceOrchestrator(
    coordinator=coordinator,
    task_manager=task_manager,
    worker_manager=worker_manager,
    resource_manager=resource_manager
)

# Orchestrate a task with data parallelism
task_data = {
    "type": "data_processing",
    "input_data": large_dataset,
    "config": {"processing_type": "transform"}
}
task_id = orchestrator.orchestrate_task(task_data, SplitStrategy.DATA_PARALLEL)

# Monitor task status
task_status = orchestrator.get_task_status(task_id)
print(f"Task status: {task_status['status']}, Completion: {task_status['completion_percentage']}%")

# Get the final result when completed
result = orchestrator.get_task_result(task_id)
```

### DynamicResourceManager

The DynamicResourceManager is responsible for tracking, allocating, and optimizing resources across the distributed system. It provides capabilities for:

1. Resource tracking and monitoring
2. Resource allocation based on task requirements
3. Resource utilization optimization
4. Resource scaling decisions
5. Resource reservation and release

**Key Features:**

- Hardware-aware resource allocation
- Resource reservation system
- Utilization monitoring and optimization
- Resource scaling recommendations
- Intelligent allocation strategies

**API Example:**

```python
# Initialize the resource manager
resource_manager = DynamicResourceManager()

# Track worker resources
worker_id = "worker-123"
resource_manager.register_worker_resources(
    worker_id,
    {
        "cpu": {"cores": 8, "memory_gb": 16},
        "gpu": {"count": 2, "memory_gb": 24, "type": "cuda"},
        "disk_gb": 100
    }
)

# Allocate resources for a task
task_id = "task-456"
allocation = resource_manager.allocate_resources(
    task_id,
    {
        "cpu_cores": 2,
        "memory_gb": 4,
        "gpu_memory_gb": 8
    }
)

# Check if allocation was successful
if allocation["success"]:
    print(f"Resources allocated on worker: {allocation['worker_id']}")
else:
    print(f"Resource allocation failed: {allocation['reason']}")

# Release resources when task completes
resource_manager.release_resources(task_id)
```

### ResourcePerformancePredictor

The ResourcePerformancePredictor uses machine learning to predict the resource requirements and performance characteristics of tasks based on historical execution data. It provides capabilities for:

1. Predicting resource requirements for tasks
2. Estimating execution time and throughput
3. Identifying optimal resource configurations
4. Providing confidence scores for predictions
5. Continuously learning from execution history

**Key Features:**

- ML-based resource requirement prediction
- Execution time and throughput estimation
- Hardware-specific performance modeling
- Confidence scoring for predictions
- Active learning for continuous improvement

**API Example:**

```python
# Initialize the performance predictor
predictor = ResourcePerformancePredictor()

# Predict resource requirements for a task
task_type = "bert_inference"
task_params = {
    "model_size": "base",
    "batch_size": 4,
    "sequence_length": 128
}
prediction = predictor.predict_resources(task_type, task_params)

print(f"Predicted resources: CPU: {prediction['cpu_cores']}, Memory: {prediction['memory_gb']}GB")
print(f"Estimated execution time: {prediction['execution_time_seconds']}s")
print(f"Prediction confidence: {prediction['confidence_score']}")

# Recommend optimal hardware
recommendation = predictor.recommend_hardware(task_type, task_params)
print(f"Recommended hardware: {recommendation['hardware_type']}")
```

### CloudProviderIntegration

The CloudProviderIntegration enables dynamic scaling of worker nodes using cloud providers. It provides capabilities for:

1. Provisioning new worker nodes based on demand
2. Scaling down underutilized resources
3. Managing cloud-specific configurations
4. Cost optimization strategies
5. Worker node lifecycle management

**Key Features:**

- Multi-cloud provider support
- Auto-scaling based on workload
- Cost optimization strategies
- Spot instance management
- Worker node lifecycle management

**API Example:**

```python
# Initialize the cloud provider manager
cloud_manager = CloudProviderManager(
    providers=["aws", "gcp"],
    config=cloud_config
)

# Scale up resources based on demand
scaling_request = {
    "cpu_cores": 16,
    "memory_gb": 64,
    "gpu_count": 4,
    "max_cost_per_hour": 2.5
}
scaling_result = cloud_manager.scale_up(scaling_request)

print(f"Provisioned {scaling_result['node_count']} new worker nodes")
print(f"Estimated cost per hour: ${scaling_result['cost_per_hour']}")

# Scale down underutilized resources
cloud_manager.scale_down(
    idle_threshold_minutes=30,
    utilization_threshold=0.2
)
```

### CoordinatorOrchestratorIntegration

The CoordinatorOrchestratorIntegration connects the MultiDeviceOrchestrator with the CoordinatorServer, enabling the coordinator to orchestrate complex tasks across multiple worker nodes. It provides capabilities for:

1. API endpoints for orchestration
2. Task orchestration and monitoring
3. Result callback handling
4. Integration with the coordinator's task and worker management
5. Coordination of the DRM components

### ResourceOptimizer

The ResourceOptimizer integrates the DynamicResourceManager and ResourcePerformancePredictor to enable intelligent resource allocation and workload optimization. It provides capabilities for:

1. Optimizing resource allocation for tasks based on historical performance data
2. Predicting resource requirements for different task types and batch sizes
3. Clustering similar workloads for efficient batch processing
4. Balancing resources across task types for optimal throughput
5. Providing enhanced scaling recommendations for different workload patterns

**Key Features:**

- Task resource requirement prediction
- Intelligent worker-task matching
- Worker type recommendations
- Workload pattern analysis
- Utilization-based scaling recommendations
- Performance history integration

**API Example:**

```python
# Initialize the resource optimizer
optimizer = ResourceOptimizer(
    resource_manager=dynamic_resource_manager,
    performance_predictor=resource_predictor,
    cloud_manager=cloud_provider_manager
)

# Get optimized resource allocation for a batch of tasks
task_batch = [task1, task2, task3]
allocation = optimizer.allocate_resources(
    task_batch=task_batch,
    available_workers=["worker1", "worker2"]
)

# Get worker type recommendations for pending tasks
recommendations = optimizer.recommend_worker_types(pending_tasks)

# Get scaling recommendations based on workload patterns
scaling_recommendations = optimizer.get_scaling_recommendations()

# Record task execution results for optimization
optimizer.record_task_result(
    task_id="task-123",
    worker_id="worker-1",
    result={
        "task_data": {...},
        "metrics": {
            "cpu_cores_used": 3.6,
            "memory_mb_used": 5120,
            "execution_time_ms": 850
        },
        "success": True
    }
)
```

**Key Features:**

- Seamless integration with coordinator
- API endpoints for orchestration
- Task tracking and monitoring
- Result aggregation and callback handling
- Comprehensive status reporting

**API Example:**

```python
# Integrate the orchestrator with the coordinator
integration = integrate_orchestrator_with_coordinator(coordinator)

# Access the integration for direct orchestration
task_data = {...}  # Task data
strategy = "model_parallel"  # Orchestration strategy
task_id = integration.orchestrate_task(task_data, strategy)

# Get the status of an orchestrated task
status = integration.get_task_status(task_id)

# Get the result of a completed task
result = integration.get_task_result(task_id)

# Clean up resources
integration.stop()
```

## Orchestration Strategies

The DRM system supports five different orchestration strategies for distributing tasks across multiple worker nodes:

### 1. Data Parallel

In data parallelism, the input data is divided into partitions, and each worker processes a different partition. The results are then merged, typically by concatenation.

**Best For:**
- Large datasets that can be processed independently
- Embarrassingly parallel tasks
- Batch processing workloads

**Example Scenarios:**
- Processing large datasets for analysis
- Running inference on many input samples
- Parallel data transformations

### 2. Model Parallel

In model parallelism, the model itself is divided into components, and each worker handles a different component. Data flows through the components in sequence.

**Best For:**
- Large models that exceed single-device memory
- Models with distinct components
- Deep networks with multiple stages

**Example Scenarios:**
- Distributing large transformer models
- Running BERT with encoder/decoder split
- Processing large vision models

### 3. Pipeline Parallel

In pipeline parallelism, the processing is divided into stages, and each worker handles a different stage. Data flows through the pipeline stages in sequence.

**Best For:**
- Multi-stage processing workflows
- Sequential dependencies between steps
- Balanced processing stages

**Example Scenarios:**
- Audio processing pipelines
- ETL workflows
- Multi-stage data transformations

### 4. Ensemble

In ensemble orchestration, multiple variants or configurations of a model or task are run in parallel, and the results are combined using aggregation methods like averaging or voting.

**Best For:**
- Improving accuracy through multiple models
- Testing different configurations
- Critical applications requiring redundancy

**Example Scenarios:**
- Running ensemble of vision models
- Combining results from multiple model sizes
- Testing different hyperparameter settings

### 5. Function Parallel

In function parallelism, different functions or operations are distributed across workers, with each worker handling a specific functional aspect.

**Best For:**
- Multi-faceted tasks with independent functions
- Comprehensive benchmarking
- Different analysis types on the same data

**Example Scenarios:**
- Running multiple benchmark functions
- Performing different analytical tasks
- Processing data with multiple algorithms

## API Reference

### MultiDeviceOrchestrator

```python
class MultiDeviceOrchestrator:
    def __init__(self, coordinator=None, task_manager=None, worker_manager=None, resource_manager=None):
        """Initialize the orchestrator."""
        
    def orchestrate_task(self, task_data, strategy):
        """Orchestrate a task for multi-device execution."""
        
    def get_task_status(self, task_id):
        """Get the status of an orchestrated task."""
        
    def cancel_task(self, task_id):
        """Cancel an orchestrated task and all its subtasks."""
        
    def process_subtask_result(self, subtask_id, result, success=True):
        """Process the result of a completed subtask."""
        
    def get_subtask_result(self, subtask_id):
        """Get the result of a completed subtask."""
        
    def get_task_result(self, task_id):
        """Get the merged result of a completed task."""
        
    def stop(self):
        """Stop the orchestrator and clean up resources."""
```

### CoordinatorOrchestratorIntegration

```python
class CoordinatorOrchestratorIntegration:
    def __init__(self, coordinator):
        """Initialize the integration."""
        
    def orchestrate_task(self, task_data, strategy):
        """Orchestrate a task for multi-device execution."""
        
    def get_task_status(self, task_id):
        """Get the status of an orchestrated task."""
        
    def get_task_result(self, task_id):
        """Get the merged result of a completed orchestrated task."""
        
    def stop(self):
        """Stop the integration and release resources."""
        
    # API Endpoints
    async def _handle_orchestrate_request(self, request_data):
        """Handle an API request to orchestrate a task."""
        
    async def _handle_orchestrated_task_request(self, request_data):
        """Handle an API request to get orchestrated task status."""
        
    async def _handle_list_orchestrated_tasks(self, request_data):
        """Handle an API request to list all orchestrated tasks."""
        
    async def _handle_cancel_orchestrated_task(self, request_data):
        """Handle an API request to cancel an orchestrated task."""
```

### Integration Helper Function

```python
def integrate_orchestrator_with_coordinator(coordinator):
    """
    Integrate the MultiDeviceOrchestrator with a CoordinatorServer.
    
    Args:
        coordinator: The CoordinatorServer instance
        
    Returns:
        CoordinatorOrchestratorIntegration: The integration instance
    """
```

## Usage Examples

### Basic Task Orchestration

```python
# Initialize the orchestrator
orchestrator = MultiDeviceOrchestrator(
    coordinator=coordinator,
    task_manager=task_manager,
    worker_manager=worker_manager,
    resource_manager=resource_manager
)

# Define a task with data parallelism
task_data = {
    "type": "data_processing",
    "input_data": [{"id": i, "value": f"item_{i}"} for i in range(100)],
    "config": {"processing_type": "transform"}
}

# Orchestrate the task
task_id = orchestrator.orchestrate_task(task_data, SplitStrategy.DATA_PARALLEL)

# Monitor task status
while True:
    status = orchestrator.get_task_status(task_id)
    print(f"Status: {status['status']}, Completion: {status['completion_percentage']}%")
    
    if status['status'] in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
        break
        
    time.sleep(5)

# Get the result if completed
if status['status'] == TaskStatus.COMPLETED:
    result = orchestrator.get_task_result(task_id)
    print(f"Task completed with result: {result}")
else:
    print(f"Task failed or was cancelled: {status['error']}")
```

### Integrating with Coordinator Server

```python
# Create and start the coordinator
coordinator = CoordinatorServer(host="0.0.0.0", port=8080, db_path="./benchmark_db.duckdb")
await coordinator.start()

# Integrate the orchestrator
integration = integrate_orchestrator_with_coordinator(coordinator)

# The integration adds API endpoints to the coordinator
# Clients can now use HTTP API to orchestrate tasks

# When done, stop integration and coordinator
integration.stop()
await coordinator.stop()
```

### Using API Endpoints

```python
# Using the HTTP API for orchestration

# 1. Orchestrate a task
POST /api/orchestrate
{
    "task_data": {
        "type": "benchmark",
        "model_name": "bert-base-uncased",
        "functions": ["latency", "throughput", "memory"]
    },
    "strategy": "function_parallel"
}

# Response
{
    "success": true,
    "task_id": "task-123",
    "message": "Task orchestrated with strategy: function_parallel"
}

# 2. Get task status
POST /api/orchestrated_task
{
    "task_id": "task-123"
}

# Response
{
    "success": true,
    "task_status": {
        "status": "in_progress",
        "completion_percentage": 33,
        "subtasks": [
            {"subtask_id": "task-123_0", "status": "completed"},
            {"subtask_id": "task-123_1", "status": "running"},
            {"subtask_id": "task-123_2", "status": "pending"}
        ]
    }
}

# 3. List all orchestrated tasks
POST /api/orchestrated_tasks
{
    "filters": {
        "limit": 10,
        "offset": 0,
        "status": "in_progress"
    }
}

# Response
{
    "success": true,
    "tasks": [
        {
            "task_id": "task-123",
            "status": "in_progress",
            "completion_percentage": 33,
            "strategy": "function_parallel"
        }
    ],
    "total": 1,
    "returned": 1
}

# 4. Cancel an orchestrated task
POST /api/cancel_orchestrated_task
{
    "task_id": "task-123"
}

# Response
{
    "success": true,
    "message": "Task canceled: task-123"
}
```

## Advanced Features

### Fault Tolerance and Recovery

The DRM system includes comprehensive fault tolerance mechanisms to handle failures at various levels:

1. **Subtask Failures**:
   - Failed subtasks can be automatically retried
   - The system can be configured to continue despite failed subtasks
   - Error information is propagated for analysis

2. **Worker Node Failures**:
   - Subtasks assigned to failed workers can be reassigned
   - The system detects worker disconnections
   - Resource allocation is adjusted dynamically

3. **Resource Exhaustion**:
   - Predictive scaling prevents resource exhaustion
   - Tasks can be queued or rescheduled based on resource availability
   - The system can provision additional resources dynamically

### Performance Optimization

The DRM system uses several strategies to optimize performance:

1. **Resource-Aware Scheduling**:
   - Tasks are matched to workers with appropriate capabilities
   - Hardware acceleration is utilized when available
   - Resource requirements are predicted to avoid bottlenecks

2. **Workload Balancing**:
   - Subtasks are distributed to balance load
   - Worker utilization is monitored in real-time
   - Work is redistributed if imbalances occur

3. **Caching and Reuse**:
   - Intermediate results can be cached
   - Common resources are shared when possible
   - Redundant computation is avoided

### Monitoring and Visualization

The DRM system provides comprehensive monitoring and visualization capabilities:

1. **Resource Utilization**:
   - Real-time tracking of CPU, memory, GPU utilization through heatmaps
   - Historical utilization patterns with trend analysis
   - Bottleneck identification with visual indicators

2. **Task Execution**:
   - Detailed task execution timelines
   - Subtask dependencies and status visualization
   - Performance metrics for each subtask with comparative analysis

3. **System Health**:
   - Worker node status and health dashboards
   - Network conditions and latency metrics
   - Error rates and failure modes with visual alerts

4. **Resource Allocation Visualization**:
   - Visual representation of resource distribution across workers
   - Efficiency metrics showing utilization vs. allocation
   - Scaling event timelines with impact analysis

5. **Cloud Resource Tracking**:
   - Cloud provider resource usage dashboards
   - Cost tracking and optimization visualizations
   - Instance type utilization metrics

6. **Interactive Dashboards**:
   - Real-time monitoring with WebSocket updates
   - Customizable refresh intervals
   - Tabbed interface for different visualization types
   - Comprehensive overview with summary metrics

For a complete demonstration of these visualization capabilities, use the example script:

```bash
python run_drm_visualization_example.py
```

For more information, see [DRM Visualization README](DRM_VISUALIZATION_README.md) and [DRM Dashboard Integration](docs/DRM_DASHBOARD_INTEGRATION.md).

## Best Practices

### Task Design

1. **Appropriate Splitting Strategy**:
   - Choose the right strategy for your task type
   - Consider data size, model size, and processing stages
   - Test different strategies for optimal performance

2. **Balanced Workloads**:
   - Design tasks to split into roughly equal subtasks
   - Avoid creating bottlenecks or dependencies when possible
   - Consider worker capabilities when designing tasks

3. **Resource Specification**:
   - Be explicit about resource requirements
   - Specify minimum and preferred resource configurations
   - Include timeout and priority information

### Resource Management

1. **Resource Allocation**:
   - Reserve resources before task execution
   - Release resources promptly when finished
   - Be mindful of resource contention

2. **Scaling Decisions**:
   - Use the predictor to anticipate resource needs
   - Scale ahead of demand when possible
   - Consider cost-performance tradeoffs

3. **Heterogeneous Hardware**:
   - Match tasks to appropriate hardware
   - Leverage specialized accelerators when available
   - Consider task-specific hardware optimizations

### Error Handling

1. **Retry Strategies**:
   - Implement appropriate retry policies
   - Use exponential backoff for transient failures
   - Set reasonable retry limits

2. **Fallback Mechanisms**:
   - Define fallback behavior for failures
   - Consider degraded execution modes
   - Prioritize critical tasks

3. **Logging and Monitoring**:
   - Log detailed error information
   - Monitor failure patterns
   - Use post-mortem analysis to improve reliability

## Testing Framework

The DRM system includes a comprehensive testing framework to verify functionality, performance, and reliability:

### Test Structure

The DRM testing infrastructure is organized as follows:

```
duckdb_api/distributed_testing/tests/
├── run_drm_tests.py              # Main test runner for all DRM tests
├── run_e2e_drm_test.py           # End-to-end test runner for real environment simulation
├── test_cloud_provider_manager.py # Tests for CloudProviderManager
├── test_drm_integration.py       # Integration tests for DRM system
├── test_dynamic_resource_manager.py # Tests for DynamicResourceManager
├── test_resource_optimization.py  # Tests for ResourceOptimizer
└── test_resource_performance_predictor.py # Tests for ResourcePerformancePredictor
```

### Test Components

1. **Unit Tests**: Testing individual components in isolation
   - Component-specific functionality
   - Error handling and edge cases
   - API contract verification

2. **Integration Tests**: Testing interactions between DRM components
   - Component communication
   - Resource allocation workflow
   - Task scheduling integration

3. **End-to-End Tests**: Testing the complete system in a simulated environment
   - Complete task lifecycle
   - Resource management under load
   - Scaling and fault tolerance

4. **Performance Tests**: Measuring system performance
   - Resource allocation efficiency
   - Scaling response time
   - System overhead

### Running Tests

To run the DRM tests:

```bash
# Run all DRM tests
python run_drm_tests.py

# Run tests for a specific component
python run_drm_tests.py --pattern resource_optimization

# Run with verbose output
python run_drm_tests.py --verbose

# Run end-to-end tests
python run_e2e_drm_test.py
```

### End-to-End Testing

The end-to-end test simulates a realistic environment with:

- Coordinator with DRM enabled
- Multiple worker nodes with different resource profiles
- Various task types with different resource requirements
- Dynamic workload patterns
- Fault injection and recovery

The test validates:
- Resource-aware task scheduling
- Dynamic scaling under varying load
- Fault tolerance and recovery
- Performance optimization

For more detailed information, see [DYNAMIC_RESOURCE_MANAGEMENT_TESTING.md](/home/barberb/ipfs_accelerate_py/test/DYNAMIC_RESOURCE_MANAGEMENT_TESTING.md).