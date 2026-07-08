# Power-Efficient Model Deployment Guide

**Date: April 2025**

This guide explains how to use the power-efficient model deployment pipeline for mobile and edge devices. The pipeline provides a comprehensive framework for deploying machine learning models with optimal power efficiency and performance.

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Components](#components)
4. [Power Profiles](#power-profiles)
5. [Deployment Targets](#deployment-targets)
6. [Basic Usage](#basic-usage)
7. [Advanced Usage](#advanced-usage)
8. [Integration with Thermal Monitoring](#integration-with-thermal-monitoring)
9. [Integration with Qualcomm AI Engine](#integration-with-qualcomm-ai-engine)
10. [Power Efficiency Reporting](#power-efficiency-reporting)
11. [Command-Line Interface](#command-line-interface)
12. [Best Practices](#best-practices)
13. [Examples](#examples)
14. [Troubleshooting](#troubleshooting)

## Overview

The Power-Efficient Model Deployment pipeline is designed to optimize machine learning models for deployment on power-constrained devices such as mobile phones, edge devices, and IoT devices. It provides:

- Intelligent power-aware model quantization and optimization
- Runtime thermal and power management
- Adaptive inference scheduling based on device state
- Comprehensive monitoring and reporting

By using this pipeline, you can significantly reduce power consumption, improve battery life, and prevent thermal throttling while maintaining model performance.

## Installation

The power-efficient deployment pipeline is included in the IPFS Accelerate Python Framework. It requires the following dependencies:

- Python 3.8 or later
- DuckDB (for storing benchmark and monitoring data)
- Mobile thermal monitoring module (for thermal management)
- Qualcomm quantization support (for Qualcomm-specific optimizations)

Optional dependencies:
- Hardware detection module (for adaptive hardware selection)
- Database API module (for storing results)

## Components

The pipeline consists of the following main components:

1. **PowerEfficientDeployment**: Main class that orchestrates the deployment process
2. **PowerProfile**: Enum defining different power consumption profiles
3. **DeploymentTarget**: Enum defining different deployment targets
4. **Integrations**:
   - Thermal monitoring system
   - Qualcomm quantization support
   - Database storage

## Power Profiles

The following power profiles are available:

| Profile | Description | Use Case |
|---------|-------------|----------|
| `MAXIMUM_PERFORMANCE` | Prioritizes performance over power consumption | High-performance tasks where power is not a constraint |
| `BALANCED` | Balances performance and power consumption | General-purpose deployment |
| `POWER_SAVER` | Prioritizes power efficiency over performance | Battery-constrained environments |
| `ULTRA_EFFICIENT` | Extremely conservative power usage | Critical battery conservation |
| `THERMAL_AWARE` | Focuses on thermal management | Devices with thermal constraints |
| `CUSTOM` | Custom profile with user-defined parameters | Specialized deployment scenarios |

## Deployment Targets

The following deployment targets are supported:

| Target | Description |
|--------|-------------|
| `ANDROID` | Android devices |
| `IOS` | iOS devices |
| `EMBEDDED` | General embedded systems |
| `BROWSER` | Web browser (WebNN/WebGPU) |
| `QUALCOMM` | Qualcomm-specific optimizations |
| `DESKTOP` | Desktop applications |
| `CUSTOM` | Custom deployment target |

## Basic Usage

Here's a basic example of how to use the power-efficient deployment pipeline:

```python
from power_efficient_deployment import PowerEfficientDeployment, PowerProfile, DeploymentTarget

# Initialize deployment with balanced power profile for Android
deployment = PowerEfficientDeployment(
    power_profile=PowerProfile.BALANCED,
    deployment_target=DeploymentTarget.ANDROID
)

# Prepare model for deployment
result = deployment.prepare_model_for_deployment(
    model_path="/path/to/model.onnx",
    model_type="text"  # Optional, can be inferred
)

# Load model for inference
model_info = deployment.load_model(
    model_path=result["output_model_path"]
)

# Run inference
inference_result = deployment.run_inference(
    model_path=result["output_model_path"],
    inputs="Sample input text"
)

# Print results
print(f"Inference time: {inference_result['inference_time_seconds']:.4f} seconds")
print(f"Output: {inference_result['outputs']}")

# Clean up when done
deployment.cleanup()
```

## Advanced Usage

### Custom Power Configuration

You can customize the power configuration for specific deployment scenarios:

```python
# Start with a predefined profile
deployment = PowerEfficientDeployment(
    power_profile=PowerProfile.BALANCED,
    deployment_target=DeploymentTarget.ANDROID
)

# Update configuration with custom settings
deployment.update_config({
    "quantization": {
        "preferred_method": "int4",
        "fallback_method": "int8"
    },
    "thermal_management": {
        "proactive_throttling": True,
        "temperature_check_interval_seconds": 2
    },
    "inference_optimization": {
        "batch_inference_when_possible": True,
        "optimal_batch_size": 8
    }
})
```

### Deployment with Specific Quantization Method

You can specify a particular quantization method for model deployment:

```python
# Prepare model with specific quantization method
result = deployment.prepare_model_for_deployment(
    model_path="/path/to/model.onnx",
    model_type="vision",
    quantization_method="int8"  # Specify quantization method
)

# Check power efficiency metrics
power_metrics = result["power_efficiency_metrics"]
print(f"Power consumption: {power_metrics['power_consumption_mw']:.2f} mW")
print(f"Battery impact: {power_metrics['battery_impact_percent_per_hour']:.2f}% per hour")
```

### Custom Model Loading and Inference

You can provide custom model loading and inference handlers:

```python
def custom_model_loader(model_path, **kwargs):
    # Custom model loading logic
    import onnxruntime as ort
    return ort.InferenceSession(model_path)

def custom_inference_handler(model, inputs, **kwargs):
    # Custom inference logic
    input_name = model.get_inputs()[0].name
    output_name = model.get_outputs()[0].name
    return model.run([output_name], {input_name: inputs})[0]

# Load model with custom loader
model_info = deployment.load_model(
    model_path=result["output_model_path"],
    model_loader=custom_model_loader
)

# Run inference with custom handler
inference_result = deployment.run_inference(
    model_path=result["output_model_path"],
    inputs=my_input_data,
    inference_handler=custom_inference_handler
)
```

## Integration with Thermal Monitoring

The power-efficient deployment pipeline integrates with the thermal monitoring system to prevent thermal throttling:

```python
# Initialize deployment with thermal-aware profile
deployment = PowerEfficientDeployment(
    power_profile=PowerProfile.THERMAL_AWARE,
    deployment_target=DeploymentTarget.ANDROID
)

# Thermal monitoring will automatically start when a model is loaded
model_info = deployment.load_model(model_path)

# Check current thermal status
thermal_status = deployment._check_thermal_status()
print(f"Thermal status: {thermal_status['thermal_status']}")
print(f"Thermal throttling: {thermal_status['thermal_throttling']}")

if thermal_status['thermal_throttling']:
    print(f"Throttling level: {thermal_status['throttling_level']}")
    
# Inference will automatically adapt to thermal conditions
inference_result = deployment.run_inference(model_path, inputs)
```

## Integration with Qualcomm AI Engine

For Qualcomm devices, the pipeline integrates with the Qualcomm AI Engine for optimal performance:

```python
# Initialize deployment with Qualcomm target
deployment = PowerEfficientDeployment(
    power_profile=PowerProfile.BALANCED,
    deployment_target=DeploymentTarget.QUALCOMM
)

# Prepare model for Qualcomm deployment
result = deployment.prepare_model_for_deployment(
    model_path="/path/to/model.onnx",
    model_type="text"
)

# Qualcomm-specific optimizations will be automatically applied
print(f"Quantization method: {result['quantization_method']}")
print(f"Optimizations applied: {result['optimizations_applied']}")
```

## Power Efficiency Reporting

The pipeline provides comprehensive power efficiency reporting:

```python
# Generate power efficiency report in different formats
json_report = deployment.get_power_efficiency_report(
    model_path=model_path,  # Optional, if None reports on all models
    report_format="json"
)

markdown_report = deployment.get_power_efficiency_report(
    report_format="markdown"
)

html_report = deployment.get_power_efficiency_report(
    report_format="html"
)

# Save report to file
with open("power_efficiency_report.md", "w") as f:
    f.write(markdown_report)
```

## Command-Line Interface

The pipeline also provides a command-line interface for common operations:

```bash
# Prepare model for deployment
python power_efficient_deployment.py prepare \
    --model-path /path/to/model.onnx \
    --model-type text \
    --power-profile POWER_SAVER \
    --deployment-target ANDROID

# Get deployment status
python power_efficient_deployment.py status

# Generate power efficiency report
python power_efficient_deployment.py report \
    --format markdown \
    --output power_report.md
```

## Best Practices

1. **Choose the right power profile**: Use `BALANCED` for general use, `POWER_SAVER` for battery-constrained scenarios, and `THERMAL_AWARE` for devices with thermal issues.

2. **Set the correct model type**: While the pipeline can infer model types, explicitly setting the model type ensures optimal optimizations.

3. **Monitor thermal status**: Regularly check thermal status during long-running inference sessions to avoid thermal throttling.

4. **Batch inference when possible**: Batching inferences can improve overall power efficiency for many models.

5. **Use hardware acceleration**: Always enable hardware acceleration for better power efficiency.

6. **Consider model size**: Smaller models generally consume less power. Use quantization to reduce model size.

7. **Unload unused models**: Unload models when they're not in use to free memory and reduce power consumption.

## Examples

### Deployment on a Mobile Phone

```python
# Initialize for mobile deployment
deployment = PowerEfficientDeployment(
    power_profile=PowerProfile.BALANCED,
    deployment_target=DeploymentTarget.ANDROID
)

# Prepare model
result = deployment.prepare_model_for_deployment(
    model_path="/path/to/model.onnx",
    model_type="vision"
)

# Load model
model_info = deployment.load_model(
    model_path=result["output_model_path"]
)

# Process camera frames
for frame in camera_frames:
    inference_result = deployment.run_inference(
        model_path=result["output_model_path"],
        inputs={"image": frame}
    )
    
    # Process results
    process_detection_results(inference_result["outputs"])
    
    # Check thermal status occasionally
    if frame_count % 100 == 0:
        thermal_status = deployment._check_thermal_status()
        if thermal_status["thermal_throttling"]:
            # Reduce frame rate or take other actions
            reduce_frame_rate()
```

### Low-Power IoT Device

```python
# Initialize for ultra-efficient deployment
deployment = PowerEfficientDeployment(
    power_profile=PowerProfile.ULTRA_EFFICIENT,
    deployment_target=DeploymentTarget.EMBEDDED
)

# Prepare model with aggressive quantization
result = deployment.prepare_model_for_deployment(
    model_path="/path/to/model.onnx",
    model_type="text",
    quantization_method="int4"  # Most aggressive quantization
)

# Load model
model_info = deployment.load_model(
    model_path=result["output_model_path"]
)

# Run inference only when needed
while True:
    if sensor_data_available():
        data = read_sensor_data()
        
        inference_result = deployment.run_inference(
            model_path=result["output_model_path"],
            inputs=data
        )
        
        process_results(inference_result["outputs"])
    
    # Sleep to save power
    time.sleep(60)  # Check every minute
```

### Web Browser Deployment

```python
# Initialize for browser deployment
deployment = PowerEfficientDeployment(
    power_profile=PowerProfile.BALANCED,
    deployment_target=DeploymentTarget.BROWSER
)

# Prepare model for browser
result = deployment.prepare_model_for_deployment(
    model_path="/path/to/model.onnx",
    model_type="text",
    quantization_method="int8"  # Good balance for browsers
)

# The rest of the API works the same way
model_info = deployment.load_model(
    model_path=result["output_model_path"]
)

inference_result = deployment.run_inference(
    model_path=result["output_model_path"],
    inputs="Sample input"
)
```

## Troubleshooting

### Common Issues

1. **Model preparation failure**:
   - Check that the model format is supported
   - Verify that the model type is correct
   - Try a different quantization method

2. **High power consumption**:
   - Use a more conservative power profile
   - Check for thermal throttling
   - Consider using a smaller or more optimized model

3. **Thermal throttling**:
   - Use `THERMAL_AWARE` power profile
   - Reduce batch size or inference frequency
   - Ensure proper ventilation for the device

4. **Slow inference**:
   - Check if thermal throttling is active
   - Verify that hardware acceleration is enabled
   - Consider using a more aggressive quantization method

### Debugging

The pipeline provides comprehensive logging for debugging:

```python
import logging

# Enable debug logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Initialize deployment
deployment = PowerEfficientDeployment(...)

# All operations will now produce detailed logs
```

## Conclusion

The Power-Efficient Model Deployment pipeline provides a comprehensive solution for deploying machine learning models on power-constrained devices. By following the guidelines in this document, you can optimize your models for power efficiency while maintaining good performance.

For more information, see:
- [Mobile Thermal Monitoring Guide](MOBILE_THERMAL_MONITORING_GUIDE.md)
- [Qualcomm AI Engine Integration Guide](QUALCOMM_AI_ENGINE_GUIDE.md)
- [Database Integration Guide](DATABASE_INTEGRATION_GUIDE.md)