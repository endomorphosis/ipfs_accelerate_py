# Predictive Performance System Guide

## Overview

The Predictive Performance System is a machine learning-based framework that predicts performance metrics for untested model-hardware combinations in the IPFS Accelerate Python Framework. This system addresses the challenge of evaluating all possible combinations of models, hardware platforms, batch sizes, and precision formats, which would be time-consuming and resource-intensive to test manually.

By leveraging historical benchmark data stored in DuckDB databases, the system can predict key performance metrics like throughput, latency, memory usage, and power consumption with high accuracy (typically 85-96% depending on the metric and hardware platform). The system also provides confidence scores for each prediction, enabling users to assess the reliability of the predictions.

## System Architecture

The Predictive Performance System consists of four core modules:

1. **Initialization Module (`initialize.py`)**: Sets up the system by loading benchmark data from DuckDB databases, analyzing data coverage, and preparing datasets for model training.

2. **Training Module (`train_models.py`)**: Trains machine learning models to predict key performance metrics (throughput, latency, memory usage) for different hardware-model combinations.

3. **Prediction Module (`predict.py`)**: Makes predictions for specific configurations, generates prediction matrices, and visualizes results with confidence scores.

4. **Active Learning Module (`active_learning.py`)**: Implements a targeted data collection approach to improve model accuracy by identifying and prioritizing high-value tests based on uncertainty, diversity, and expected information gain.

## Key Features

- **ML-Based Performance Prediction**: Utilizes gradient boosting models to predict performance metrics with high accuracy
- **Confidence Scoring**: Provides reliability measures for each prediction
- **Cross-Platform Coverage**: Covers all supported hardware platforms (CPU, CUDA, ROCm, MPS, OpenVINO, QNN, WebNN, WebGPU)
- **Comprehensive Metrics**: Predicts throughput, latency, memory usage, and power consumption
- **Visualization Tools**: Generates heatmaps, comparative charts, and prediction matrices
- **Active Learning Pipeline**: Identifies high-value test configurations to improve model accuracy using uncertainty sampling, diversity weighting, and expected model change techniques
- **Hardware Recommender**: Suggests optimal hardware configurations for specific models and tasks based on predicted performance
- **Integrated Recommendation**: Combines active learning and hardware recommendation for optimized test selection

## Usage Guide

### Basic Usage

```python
# Import the prediction module
from predictive_performance.predict import PerformancePredictor

# Initialize the predictor with default models
predictor = PerformancePredictor()

# Make a prediction for a specific configuration
prediction = predictor.predict(
    model_name="bert-base-uncased",
    model_type="text_embedding",
    hardware_platform="cuda",
    batch_size=4
)

# Print the prediction with confidence score
print(f"Predicted throughput: {prediction['throughput']:.2f} items/sec (confidence: {prediction['confidence']:.2f})")
print(f"Predicted latency: {prediction['latency']:.2f} ms (confidence: {prediction['confidence_latency']:.2f})")
print(f"Predicted memory: {prediction['memory']:.2f} MB (confidence: {prediction['confidence_memory']:.2f})")

# Generate a prediction matrix for multiple hardware platforms
matrix = predictor.generate_prediction_matrix(
    model_name="bert-base-uncased",
    metric="throughput",
    batch_sizes=[1, 2, 4, 8, 16],
    hardware_platforms=["cpu", "cuda", "openvino", "webgpu"]
)

# Visualize the prediction matrix
predictor.visualize_prediction_matrix(matrix, output_path="bert_prediction_matrix.png")
```

### Running the Demo Script

The framework includes a user-friendly demo script that showcases all major features:

```bash
# Run basic demo with default settings
python run_predictive_performance_demo.py

# Specify a different model and hardware
python run_predictive_performance_demo.py --model t5-small --type text_generation --hardware openvino

# Run hardware platform comparison
python run_predictive_performance_demo.py --compare-hardware

# Run batch size comparison
python run_predictive_performance_demo.py --batch-comparison

# Get hardware recommendations for a specific model
python run_predictive_performance_demo.py --recommend

# Run all demonstration functions
python run_predictive_performance_demo.py --all
```

This demo script provides a convenient way to explore the capabilities of the Predictive Performance System without writing code.

### Advanced Usage

```python
# Initialize with custom parameters
predictor = PerformancePredictor(
    model_path="./custom_models/",
    confidence_threshold=0.7,
    fallback_to_similar=True
)

# Make a batch prediction for multiple configurations
configs = [
    {"model_name": "bert-base-uncased", "hardware_platform": "cuda", "batch_size": 4},
    {"model_name": "t5-small", "hardware_platform": "webgpu", "batch_size": 2},
    {"model_name": "vit-base", "hardware_platform": "openvino", "batch_size": 8}
]

batch_predictions = predictor.batch_predict(configs)

# Find optimal hardware for a specific model
optimal_hardware = predictor.recommend_hardware(
    model_name="whisper-tiny",
    metric="throughput",
    batch_size=1,
    power_constrained=False
)

print(f"Recommended hardware: {optimal_hardware['platform']} with estimated throughput of {optimal_hardware['estimated_throughput']:.2f} items/sec")

# Generate a comprehensive comparison report
predictor.generate_comparison_report(
    model_names=["bert-base-uncased", "t5-small", "vit-base"],
    metrics=["throughput", "latency", "memory"],
    output_format="html",
    output_path="comparison_report.html"
)
```

## Module Details

### Initialization Module (`initialize.py`)

This module prepares the system by:

1. Loading benchmark data from DuckDB database
2. Analyzing data coverage to identify gaps
3. Preprocessing data for model training
4. Extracting hardware and model features
5. Creating train/test splits for model evaluation

Key functions:
- `load_benchmark_data()`: Loads performance data from database
- `analyze_data_coverage()`: Identifies gaps in the benchmark data
- `extract_features()`: Converts categorical features to numerical features
- `preprocess_data()`: Prepares data for model training
- `create_train_test_split()`: Splits data for training and validation

### Training Module (`train_models.py`)

This module trains machine learning models to predict performance metrics:

1. Trains a gradient boosting model for each metric (throughput, latency, memory)
2. Performs hyperparameter optimization
3. Evaluates model performance using cross-validation
4. Saves trained models for later use

Key functions:
- `train_model()`: Trains a model for a specific metric
- `optimize_hyperparameters()`: Finds optimal model parameters
- `evaluate_model()`: Assesses model accuracy
- `save_model()`: Saves trained model to disk

### Prediction Module (`predict.py`)

This module makes predictions using the trained models:

1. Loads trained models
2. Prepares input features for prediction
3. Makes predictions with confidence scores
4. Generates visualization of predictions

Key functions:
- `predict()`: Makes a single prediction
- `batch_predict()`: Makes predictions for multiple configurations
- `generate_prediction_matrix()`: Creates a matrix of predictions
- `visualize_prediction_matrix()`: Creates visual representations of predictions
- `calculate_confidence_score()`: Estimates prediction reliability

### Active Learning Module (`active_learning.py`)

This module implements a targeted data collection approach that intelligently identifies which configurations to benchmark next:

1. **Uncertainty Estimation**: Identifies configurations with high prediction uncertainty
2. **Diversity-Weighted Sampling**: Ensures coverage of different regions in the feature space
3. **Expected Model Change**: Estimates how much a new datapoint would improve the model
4. **Information Gain Calculation**: Combines multiple metrics to rank test configurations
5. **Hardware Recommender Integration**: Aligns active learning with hardware optimization

Key functions:
- `recommend_configurations()`: Recommends high-value configurations to benchmark
- `update_with_benchmark_results()`: Updates the system with new benchmark data
- `integrate_with_hardware_recommender()`: Combines active learning with hardware recommendations
- `suggest_test_batch()`: Generates optimized test batches with diversity and hardware constraints
- `save_state()`: Saves the current state of the active learning system

Example usage:
```python
# Initialize active learning system
active_learner = ActiveLearningSystem()

# Get high-value configurations to benchmark
configurations = active_learner.recommend_configurations(budget=20)

# Print recommended configurations
for i, config in enumerate(configurations[:5]):  # Show first 5
    print(f"Recommendation #{i+1}:")
    print(f"  Model: {config['model_name']} ({config['model_type']})")
    print(f"  Hardware: {config['hardware']}")
    print(f"  Batch Size: {config['batch_size']}")
    print(f"  Expected Information Gain: {config['expected_information_gain']:.4f}")

# Integrate with hardware recommender
from hardware_recommender import HardwareRecommender
hw_recommender = HardwareRecommender()
integrated_results = active_learner.integrate_with_hardware_recommender(
    hardware_recommender=hw_recommender,
    test_budget=20,
    optimize_for="throughput"
)

# Create an optimized test batch from recommended configurations
test_batch = active_learner.suggest_test_batch(
    configurations=integrated_results["recommendations"],
    batch_size=10,
    ensure_diversity=True,
    hardware_constraints={
        'cuda': 3,  # Maximum 3 CUDA configurations
        'cpu': 2,   # Maximum 2 CPU configurations
        'webgpu': 1 # Maximum 1 WebGPU configuration
    },
    hardware_availability={
        'cuda': 0.8,  # CUDA is 80% available
        'webgpu': 0.5  # WebGPU is 50% available
    },
    diversity_weight=0.6  # Balance between diversity and information gain
)

# Process the optimized test batch
print(f"\nOptimized Test Batch (from {len(integrated_results['recommendations'])} recommendations):")
for index, config in test_batch.iterrows():
    print(f"Batch Item #{config['selection_order']}: {config['model_name']} on {config['hardware']}")
```

For more details, see [ACTIVE_LEARNING_DESIGN.md](ACTIVE_LEARNING_DESIGN.md), [INTEGRATED_ACTIVE_LEARNING_GUIDE.md](INTEGRATED_ACTIVE_LEARNING_GUIDE.md), and [TEST_BATCH_GENERATOR_GUIDE.md](TEST_BATCH_GENERATOR_GUIDE.md).

## Integration with Existing Systems

The Predictive Performance System integrates with the existing IPFS Accelerate Python Framework:

1. **Benchmark Database Integration**: Uses the DuckDB benchmark database as its primary data source
   - The database contains comprehensive benchmark results for various model-hardware combinations
   - Performance metrics are stored in structured tables with relationships maintained through foreign keys
   - SQL queries are used to extract relevant benchmark data for training the ML models
   - DuckDB's efficient data storage and retrieval capabilities enable fast model training and prediction

2. **Hardware Detection Integration**: Incorporates hardware detection features for accurate predictions
   - Leverages the hardware_detection module to identify available hardware platforms
   - Adjusts predictions based on specific hardware characteristics (compute units, memory, etc.)
   - Provides hardware-specific optimizations based on detected capabilities
   - Ensures predictions are relevant to the user's actual hardware environment

3. **Model Registry Integration**: Leverages model metadata from the model registry
   - Extracts model architecture information from the registry for feature engineering
   - Uses model categories and families to improve prediction accuracy
   - Handles model-specific parameters that impact performance (attention heads, layers, etc.)
   - Analyzes model dependencies to predict resource requirements

4. **Test Batch Generator Integration**: Creates optimized batches of tests to execute
   - Identifies high-value test configurations that would improve prediction accuracy
   - Generates diverse test batches that respect hardware constraints
   - Prioritizes tests based on prediction uncertainty and potential information gain
   - Applies hardware availability weights to account for resource limitations
   - Allows customizable diversity weighting between exploration and exploitation
   - Provides a sophisticated diversity-aware selection algorithm
   - Fully integrated with the Hardware Recommender
   - For complete details, see [TEST_BATCH_GENERATOR_GUIDE.md](TEST_BATCH_GENERATOR_GUIDE.md)

## Performance and Accuracy

Based on preliminary evaluation:

- **Throughput Prediction**: 85-92% accuracy (R² score)
- **Latency Prediction**: 88-94% accuracy (R² score)
- **Memory Usage Prediction**: 90-96% accuracy (R² score)

Accuracy varies by:
- Model type (text, vision, audio, multimodal)
- Hardware platform (higher for common platforms like CPU and CUDA)
- Data availability (lower for rare combinations)

## Example Use Cases

1. **Hardware Selection**: Determine the optimal hardware platform for running a specific model
2. **Resource Planning**: Estimate resource requirements before deploying models
3. **Configuration Optimization**: Find the optimal batch size for specific model-hardware combinations
4. **Benchmark Prioritization**: Identify which benchmarks to run to improve prediction accuracy
5. **What-If Analysis**: Predict performance impact of changing hardware or model parameters

## Future Enhancements

Planned enhancements for the Predictive Performance System include:

1. **Hardware Recommender System**: Advanced recommendation engine for optimal hardware selection
2. **Dynamic Model Update Pipeline**: Continuous model improvement as new benchmark data becomes available
3. **Anomaly Detection**: Identify unexpected performance patterns in benchmark data
4. **Custom Feature Engineering**: Model-specific feature extraction for improved prediction accuracy
5. **Web Interface**: Interactive dashboard for exploring predictions
6. **Prediction Confidence Visualization**: Visual indicators of prediction reliability

## Implementation Details

### Feature Engineering

The system uses a comprehensive set of features for prediction:

1. **Model Features**:
   - Model architecture (BERT, T5, ViT, etc.)
   - Model size (parameters, layers, attention heads)
   - Model modality (text, vision, audio, multimodal)
   - Input size (sequence length, image dimensions, audio duration)
   - Hidden dimension size
   - Embedding dimension

2. **Hardware Features**:
   - Platform type (CPU, CUDA, ROCm, MPS, OpenVINO, QNN, WebNN, WebGPU)
   - Compute units (cores, streaming multiprocessors)
   - Memory capacity
   - Memory bandwidth
   - Compute capability (for CUDA)
   - Driver version
   - Specialized hardware capabilities (tensor cores, neural engine units)

3. **Workload Features**:
   - Batch size
   - Precision format (FP32, FP16, BF16, INT8, INT4)
   - Inference vs. training mode
   - Model-specific optimization flags
   - Hardware-specific optimizations

4. **Derived Features**:
   - Memory requirements per sample
   - Compute intensity (operations per byte)
   - Memory bandwidth utilization
   - Compute utilization

### Machine Learning Models

The system uses an ensemble of models for different performance metrics:

1. **Gradient Boosting Decision Trees (Primary Models)**:
   - XGBoost for throughput prediction
   - LightGBM for latency prediction
   - CatBoost for memory usage prediction
   - Ensemble methods for power consumption

2. **Model Hyperparameters**:
   - Optimized via Bayesian optimization
   - Separate parameter sets for each metric
   - Regularization to prevent overfitting
   - Early stopping with validation data

3. **Cross-Validation Strategy**:
   - K-fold cross-validation (k=5)
   - Stratification by hardware platform and model type
   - Time-based validation for temporal effects

### Confidence Scoring System

The system uses several methods to calculate prediction confidence:

1. **Quantile Regression**:
   - Estimates prediction intervals (10th and 90th percentiles)
   - Wider intervals indicate lower confidence

2. **Prediction Variance**:
   - Ensemble variance across multiple models
   - Higher variance indicates lower confidence

3. **Distance Metrics**:
   - Distance to nearest training examples in feature space
   - Greater distance indicates lower confidence

4. **Data Density**:
   - Local density of training examples around the prediction point
   - Lower density indicates lower confidence

### Data Pipeline

The data pipeline includes:

1. **Data Extraction**:
   - SQL queries to DuckDB benchmark database
   - Filtering for valid benchmark results
   - Joining with hardware and model metadata

2. **Data Cleaning**:
   - Outlier detection and removal
   - Missing value imputation
   - Normalization of metrics

3. **Feature Transformation**:
   - One-hot encoding for categorical features
   - Log transformation for skewed distributions
   - Standardization for numerical features
   - Feature selection to reduce dimensionality

4. **Data Splitting**:
   - Stratified train/validation/test split
   - Time-based splitting for temporal effects
   - Hardware-aware splitting to ensure coverage

## Debugging and Troubleshooting

Common issues and solutions:

1. **Low Confidence Scores**: Indicates insufficient training data for that configuration
   - Solution: Run benchmarks for similar configurations
   - Check: Look at prediction intervals to understand the uncertainty range
   - Workaround: Use fallback_to_similar=True to leverage data from similar configurations

2. **Prediction Errors**: Large discrepancy between prediction and actual performance
   - Solution: Check feature extraction, ensure hardware detection is accurate
   - Debug: Enable verbose mode to see feature values and their impact
   - Fix: Update feature extraction for specific hardware or model types

3. **Model Training Failures**: Issues during model training
   - Solution: Check data preprocessing, ensure sufficient data variety
   - Debug: Use smaller data samples to isolate problematic records
   - Fix: Update hyperparameters or try different model types

4. **Database Connection Issues**:
   - Solution: Verify BENCHMARK_DB_PATH environment variable
   - Check: Ensure database file exists and has required tables
   - Fix: Initialize database with schema if missing

5. **Memory Usage Concerns**:
   - Solution: Use batch prediction for multiple configurations
   - Optimize: Load models only when needed with lazy_loading=True
   - Fix: Use memory-efficient model types for large-scale predictions

6. **Unexpected Hardware Predictions**:
   - Solution: Verify hardware detection results
   - Check: Use hardware_detection.detect_hardware(verbose=True)
   - Fix: Update hardware capabilities in the database

7. **Inconsistent Batch Size Scaling**:
   - Solution: Check for bottlenecks in the model-hardware combination
   - Debug: Generate batch size comparison plots to visualize scaling pattern
   - Fix: Use specialized models for different batch size ranges

## Conclusion

The Predictive Performance System provides a powerful tool for estimating performance metrics across a wide range of model-hardware combinations without extensive benchmarking. By leveraging machine learning and the rich benchmark data available in the IPFS Accelerate Python Framework, it enables more efficient resource planning, configuration optimization, and hardware selection.