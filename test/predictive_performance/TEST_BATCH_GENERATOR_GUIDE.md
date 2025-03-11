# Test Batch Generator Guide

**Version:** 1.0.0  
**Date:** March 15, 2025  
**Status:** Completed (100%)

## Overview

The Test Batch Generator is a critical component of the Active Learning Pipeline that creates optimized batches of test configurations for benchmarking. It ensures that test batches maximize expected information gain while respecting hardware constraints and ensuring diversity across configurations.

This guide explains how to use the Test Batch Generator, its key features, and how it integrates with the Active Learning System and Hardware Recommender.

## Key Features

The Test Batch Generator provides the following key features:

1. **Optimized Batch Creation**: Generate batches that maximize expected information gain
2. **Diversity-Aware Selection**: Ensure selected configurations cover diverse hardware and model combinations
3. **Hardware Constraints**: Respect hardware availability and maximum usage limits
4. **Integrated Scoring**: Consider both information gain and hardware optimization
5. **Customizable Weighting**: Configure diversity vs. information gain prioritization
6. **Hardware Availability Integration**: Account for hardware that might be partially available
7. **Selection Order Tracking**: Preserve the priority of selected configurations

## Using the Test Batch Generator

The Test Batch Generator is accessed through the `suggest_test_batch` method of the `ActiveLearningSystem` class:

```python
from predictive_performance.active_learning import ActiveLearningSystem

# Initialize the active learning system
active_learning = ActiveLearningSystem()

# Generate high-value configurations
high_value_configs = active_learning.recommend_configurations(budget=50)

# Create an optimized test batch
batch = active_learning.suggest_test_batch(
    configurations=high_value_configs,
    batch_size=10,
    ensure_diversity=True,
    hardware_constraints={
        'cuda': 3,    # Maximum 3 CUDA configurations
        'cpu': 2,     # Maximum 2 CPU configurations
        'webgpu': 1   # Maximum 1 WebGPU configuration
    },
    hardware_availability={
        'cuda': 0.8,  # CUDA is 80% available
        'webgpu': 0.5 # WebGPU is 50% available
    },
    diversity_weight=0.6  # Weighting between diversity and information gain
)

# Process the batch
for index, config in batch.iterrows():
    print(f"Configuration {config['selection_order']}: {config['model_name']} on {config['hardware']}")
    print(f"  Expected information gain: {config['expected_information_gain']:.4f}")
```

## Command-Line Interface

A command-line tool is available for testing and demonstrating the Test Batch Generator functionality:

```bash
# Run all tests of the Test Batch Generator
python predictive_performance/test_batch_generator.py --test all

# Test hardware constraints specifically
python predictive_performance/test_batch_generator.py --test hardware

# Test hardware availability weighting
python predictive_performance/test_batch_generator.py --test availability

# Test diversity weighting
python predictive_performance/test_batch_generator.py --test diversity

# Test combined constraints
python predictive_performance/test_batch_generator.py --test combined

# Run with verbose logging
python predictive_performance/test_batch_generator.py --test all --verbose
```

## API Reference

### suggest_test_batch() Method

```python
suggest_test_batch(
    configurations,
    batch_size=10, 
    ensure_diversity=True, 
    hardware_constraints=None, 
    hardware_availability=None,
    diversity_weight=0.5
)
```

**Parameters:**

- `configurations` (DataFrame or list): Configurations to select from
- `batch_size` (int): Maximum number of configurations to include in the batch
- `ensure_diversity` (bool): Whether to ensure diversity in the selected batch
- `hardware_constraints` (dict, optional): Dictionary mapping hardware types to maximum count in batch
- `hardware_availability` (dict, optional): Dictionary mapping hardware types to availability factor (0-1)
- `diversity_weight` (float): Weight to give diversity vs. information gain (0-1)

**Returns:**

- DataFrame of selected configurations for the test batch, with a `selection_order` column indicating the priority

## Integration with Hardware Recommender

The Test Batch Generator works seamlessly with the Hardware Recommender through the `integrate_with_hardware_recommender` method of the `ActiveLearningSystem` class:

```python
from predictive_performance.active_learning import ActiveLearningSystem
from predictive_performance.hardware_recommender import HardwareRecommender
from predictive_performance.predict import PerformancePredictor

# Initialize system components
predictor = PerformancePredictor()
hw_recommender = HardwareRecommender(predictor=predictor)
active_learning = ActiveLearningSystem()

# Get integrated recommendations
integrated_results = active_learning.integrate_with_hardware_recommender(
    hardware_recommender=hw_recommender,
    test_budget=20,
    optimize_for="throughput"
)

# Generate a batch from the integrated recommendations
batch = active_learning.suggest_test_batch(
    configurations=integrated_results["recommendations"],
    batch_size=10,
    ensure_diversity=True
)

# Use the batch for testing
for index, config in batch.iterrows():
    print(f"Configuration {config['selection_order']}: {config['model_name']} on {config['hardware']}")
    print(f"  Recommended hardware: {config.get('recommended_hardware', 'N/A')}")
    print(f"  Combined score: {config.get('combined_score', 0):.4f}")
```

## Algorithm Details

The Test Batch Generator uses sophisticated algorithms to create optimized batches:

### Diversity-Aware Selection Algorithm

The diversity-aware selection algorithm works as follows:

1. Select the first configuration with the highest expected information gain
2. For subsequent selections, calculate a score for each remaining configuration that considers:
   - The original score (expected information gain or combined score)
   - The diversity as measured by the minimum distance to already selected configurations
3. Weight these factors according to the specified diversity weight
4. Select the configuration with the highest combined score
5. Repeat until the batch is complete or constraints prevent further selections

### Hardware Constraint Handling

Hardware constraints are handled through several mechanisms:

1. **Hardware Platform Limits**: The generator enforces hard limits on the number of configurations for each hardware platform.
2. **Hardware Availability Weighting**: Configurations for less available hardware have their scores reduced proportionally.
3. **Combined Scoring**: When integrated with the Hardware Recommender, scores reflect both information gain and hardware optimization.

## Performance Considerations

For optimal performance, consider the following:

1. **Batch Size**: Larger batch sizes generally provide more diverse configurations but may exceed hardware constraints.
2. **Diversity Weight**: A higher diversity weight (closer to 1.0) prioritizes diversity over information gain, potentially selecting configurations with lower expected information gain but broader coverage of the feature space.
3. **Hardware Constraints**: Carefully consider hardware constraints to ensure efficient resource utilization without unnecessarily limiting the diversity of selected configurations.

## Best Practices

1. **Combine with Hardware Recommender**: For optimal results, use the Test Batch Generator with integrated recommendations from the Hardware Recommender.
2. **Balance Diversity**: Adjust the diversity weight based on your goals - higher for exploration, lower for exploitation.
3. **Realistic Hardware Constraints**: Set hardware constraints based on actual hardware availability and testing resources.
4. **Iterate and Update**: After testing a batch, update the active learning system with the results to improve future recommendations.

## Troubleshooting

### Common Issues

1. **Not Enough Diversity**: If the generated batch lacks diversity, increase the diversity weight parameter.
2. **Hardware Constraints Too Restrictive**: If hardware constraints prevent the selection of high-value configurations, consider relaxing constraints or prioritizing the most informative hardware platforms.
3. **Low Information Gain**: If configurations with low expected information gain are selected, reduce the diversity weight to prioritize higher-value configurations.

## Conclusion

The Test Batch Generator is a powerful tool for optimizing benchmark resource allocation by identifying the most valuable configurations to test. By ensuring diversity and respecting hardware constraints, it enables efficient exploration of the performance space while maximizing information gain.