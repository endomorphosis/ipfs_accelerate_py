# Enhanced Hardware Taxonomy and Heterogeneous Scheduler Integration

This document describes the integration between the Enhanced Hardware Taxonomy (EHT) and the Heterogeneous Scheduler (HS) components of the Distributed Testing Framework.

## Overview

The integration between these components enables more sophisticated hardware selection for tasks based on fine-grained hardware capabilities rather than just broad hardware classes. This results in better matching of workloads to the most appropriate hardware, improved resource utilization, and better overall performance.

## Architecture

The integration is implemented through a new component called the `HardwareTaxonomyIntegrator` which acts as a bridge between the Enhanced Hardware Taxonomy and the Heterogeneous Scheduler. This design minimizes changes to both existing systems while providing a clean interface for integration.

![Architecture Diagram](../diagrams/taxonomy_integration_architecture.png)

### Key Components

1. **Enhanced Hardware Taxonomy**: Provides capability definitions, relationships, and discovery
2. **Heterogeneous Scheduler**: Manages task scheduling to workers based on hardware characteristics
3. **HardwareTaxonomyIntegrator**: Connects the EHT and HS by enhancing worker states and task profiles with capability information
4. **WorkloadProfile**: Enhanced with capability requirements and preferences
5. **WorkerState**: Enhanced with capability profiles that represent available hardware capabilities

## Key Features

### 1. Capability-Based Worker Registration

When workers register with the scheduler, their hardware profiles are automatically enhanced with capabilities:

```python
# Register a worker with capability enhancement
worker = scheduler.register_worker(worker_id, capabilities)
```

The integrator analyzes the worker's hardware profiles and:
- Converts them to HardwareCapabilityProfile objects
- Auto-discovers capabilities based on hardware characteristics
- Stores the capability profiles for later use
- Updates workload specialization scores based on capability matches

### 2. Capability-Based Task Submission

When tasks are submitted to the scheduler, their workload profiles are enhanced with capability requirements:

```python
# Submit a task with capability enhancement
scheduler.submit_task(task)
```

The integrator analyzes the task's workload profile and:
- Adds required capabilities based on workload type
- Adds preferred capabilities based on workload type
- Ensures the workload profile has the proper capability attributes

### 3. Enhanced Affinity Calculation

The scheduler uses enhanced affinity calculation during scheduling:

```python
# Calculate enhanced affinity score for worker-task matching
score = integrator.calculate_enhanced_affinity(worker, task)
```

The enhanced affinity calculation:
- Starts with the standard affinity score (which considers specialization, load, thermal state)
- Enhances it with a capability match score based on required and preferred capabilities
- Produces a more accurate representation of worker suitability for specific tasks

### 4. Workload Capability Requirements

Each workload type has a set of required and preferred capabilities:

| Workload Type | Required Capabilities | Preferred Capabilities |
|---------------|------------------------|------------------------|
| nlp           | matrix_multiplication   | tensor_core_acceleration, int8_acceleration |
| vision        | matrix_multiplication   | conv_acceleration, int8_vision_optimization |
| audio         | matrix_multiplication   | fft_acceleration, audio_dsp_support |
| browser       | browser_compatibility   | webgpu_support, webnn_support |
| multimodal    | matrix_multiplication   | conv_acceleration, tensor_core_acceleration, parallel_execution |

### 5. Capability Discovery

The system can automatically discover capabilities based on hardware characteristics, for example:
- GPUs with tensor cores automatically get tensor_core_acceleration
- CPUs with AVX-512 automatically get avx512_vector_acceleration
- Browsers with WebGPU support automatically get webgpu_support

## Usage

To use the integrated system:

1. Enable the integration when creating the scheduler:
```python
scheduler = HeterogeneousScheduler(
    strategy="adaptive",
    use_enhanced_taxonomy=True  # Enable taxonomy integration
)
```

2. Create workload profiles with capability requirements:
```python
profile = WorkloadProfile(
    workload_type="nlp",
    # ...other parameters...
)
profile.add_required_capability("matrix_multiplication")
profile.add_preferred_capability("tensor_core_acceleration")
```

3. Let the scheduler handle the rest:
```python
# The scheduler will use enhanced capability matching automatically
scheduler.submit_task(task)
scheduler.schedule_tasks()
```

## Benefits

1. **More Precise Hardware Matching**: Tasks are matched to hardware based on specific capabilities rather than just broad hardware classes.

2. **Automatic Capability Discovery**: Hardware capabilities are automatically detected based on hardware characteristics.

3. **Improved Specialization**: Workers can specialize in specific capabilities rather than broad workload types.

4. **Future Hardware Support**: New hardware capabilities can be added without changing the scheduler code.

5. **Better Resource Utilization**: More optimal matching leads to better overall resource utilization.

## Implementation Status

The integration is fully implemented and ready for use. It includes:

- [x] HardwareTaxonomyIntegrator class
- [x] Enhanced WorkloadProfile with capability support
- [x] Updated HeterogeneousScheduler with taxonomy integration
- [x] Capability-based affinity calculation
- [x] Comprehensive test suite
- [x] Documentation

## Next Steps

1. **Expanded Capability Definitions**: Add more specialized capabilities for different hardware types.

2. **Capability-Based Performance Analysis**: Track performance by capability to improve future scheduling decisions.

3. **Dynamic Capability Learning**: Learn which capabilities matter most for different workloads based on historical performance.

4. **Multi-Device Orchestration**: Use capability information to split workloads across multiple devices.

5. **User Interface for Capability Exploration**: Develop a UI for exploring capabilities and their impact on performance.

## Related Documentation

- [Enhanced Hardware Taxonomy Implementation](ENHANCED_HARDWARE_TAXONOMY_IMPLEMENTATION.md)
- [Heterogeneous Hardware Guide](HETEROGENEOUS_HARDWARE_GUIDE.md)
- [Distributed Testing Framework Overview](DISTRIBUTED_TESTING_DESIGN.md)