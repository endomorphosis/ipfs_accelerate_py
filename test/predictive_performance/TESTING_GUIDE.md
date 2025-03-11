# Predictive Performance System Testing Guide

**Date:** March 10, 2025  
**Version:** 1.0

## Overview

This guide provides comprehensive instructions for testing the Predictive Performance System, with a focus on verifying the Active Learning Pipeline and its integration with the Hardware Recommender. The testing approach covers unit tests, integration tests, and end-to-end validation to ensure all components work correctly together.

## Testing Environment Setup

### Prerequisites

Before running tests, ensure you have the following prerequisites installed:

- Python 3.9 or higher
- NumPy, pandas, scikit-learn, matplotlib, seaborn
- PyTest for unit testing
- DuckDB for database integration testing

### Setting Up the Test Environment

```bash
# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables for testing
export BENCHMARK_DB_PATH=test_db.duckdb
export TEST_MODE=True
```

## Unit Testing

Unit tests verify the functionality of individual components in isolation.

### Running Unit Tests

```bash
# Run all unit tests
python -m pytest test_*.py -v

# Run specific test files
python -m pytest test_active_learning.py -v
python -m pytest test_hardware_recommender.py -v

# Run specific test classes or methods
python -m pytest test_active_learning.py::TestUncertaintyEstimation -v
python -m pytest test_active_learning.py::TestUncertaintyEstimation::test_uncertainty_calculation -v
```

### Key Unit Test Files

- `test_active_learning.py`: Tests for the Active Learning Pipeline
- `test_hardware_recommender.py`: Tests for the Hardware Recommender
- `test_predict.py`: Tests for the Prediction Module
- `test_model_training.py`: Tests for the Model Training Module

## Integration Testing

Integration tests verify that different components work together correctly.

### Running Integration Tests

```bash
# Run the integrated test script
python test_integration.py

# Run with specific parameters
python test_integration.py --output-dir ./test_results
```

### Integration Test Scenarios

The integration tests cover the following scenarios:

1. **Active Learning + Hardware Recommender**: Tests the integration between active learning and hardware recommendation
2. **Prediction + Hardware Recommender**: Tests that hardware recommendations are based on accurate predictions
3. **Active Learning + DuckDB**: Tests active learning with database-backed benchmark data
4. **End-to-End Test**: Tests the complete workflow from prediction to recommendation to benchmark execution

## End-to-End Testing

End-to-end tests validate the entire system workflow.

### Running End-to-End Tests

```bash
# Run the example script with integration mode
python example.py integrate --budget 5 --metric throughput --output test_integrated_recommendations.json
```

### End-to-End Test Scenarios

1. **Full Workflow Test**: Tests the complete workflow from data loading to integrated recommendation
2. **DuckDB Integration Test**: Tests integration with the benchmark database
3. **Model Update Test**: Tests model updating with new benchmark results
4. **Visualization Test**: Tests generation of visualizations from recommendations

## Test Data

The test suite includes synthetic test data for running tests without requiring access to the full benchmark database.

### Using Test Data

```python
# In your test code
from test_utils import generate_test_data

# Generate synthetic benchmark data
benchmark_data = generate_test_data(
    model_types=["text_embedding", "vision"],
    hardware_platforms=["cpu", "cuda"],
    num_records=100
)
```

### Test Data Format

The synthetic test data follows the same schema as the actual benchmark database, with the following tables:

- `models`: Information about AI models
- `hardware_platforms`: Information about hardware platforms
- `performance_results`: Performance metrics for model-hardware combinations
- `benchmark_metadata`: Additional metadata about benchmarks

## Testing the Active Learning Pipeline

### Key Components to Test

1. **Uncertainty Estimation**: Tests for accurate uncertainty quantification
2. **Diversity Calculation**: Tests for configuration diversity metrics
3. **Expected Model Change**: Tests for accurate model change estimation
4. **Information Gain Calculation**: Tests for proper combination of uncertainty and diversity
5. **Recommendation Generation**: Tests for generating high-value recommendations

### Example Test Case: Uncertainty Estimation

```python
def test_uncertainty_estimation():
    """Test uncertainty estimation for active learning."""
    # Initialize active learning system with synthetic data
    active_learner = ActiveLearningSystem(data_file=None)
    
    # Get recommendations
    recommendations = active_learner.recommend_configurations(budget=5)
    
    # Check that uncertainty values are reasonable
    for config in recommendations:
        assert 'uncertainty' in config
        assert 0 <= config['uncertainty'] <= 1
        
    # Check that higher uncertainty leads to higher information gain
    uncertainties = [config['uncertainty'] for config in recommendations]
    info_gains = [config['expected_information_gain'] for config in recommendations]
    
    # Calculate correlation
    correlation = numpy.corrcoef(uncertainties, info_gains)[0, 1]
    assert correlation > 0.5  # Strong positive correlation expected
```

## Testing the Integration with Hardware Recommender

### Key Integration Points to Test

1. **Score Combination**: Tests that information gain and hardware scores are combined correctly
2. **Hardware Match Detection**: Tests that hardware matches are correctly identified
3. **Ranking Algorithm**: Tests that configurations are correctly ranked by combined score
4. **Metadata Generation**: Tests that result metadata is correctly generated

### Example Test Case: Integrated Recommendations

```python
def test_integrated_recommendations():
    """Test the integration between active learning and hardware recommender."""
    # Initialize components
    active_learner = ActiveLearningSystem()
    predictor = PerformancePredictor()
    hw_recommender = HardwareRecommender(predictor=predictor)
    
    # Get integrated recommendations
    results = active_learner.integrate_with_hardware_recommender(
        hardware_recommender=hw_recommender,
        test_budget=5,
        optimize_for="throughput"
    )
    
    # Verify structure of results
    assert 'recommendations' in results
    assert 'total_candidates' in results
    assert 'enhanced_candidates' in results
    assert 'final_recommendations' in results
    
    # Verify recommendations
    for config in results['recommendations']:
        assert 'model_name' in config
        assert 'hardware' in config
        assert 'batch_size' in config
        assert 'recommended_hardware' in config
        assert 'hardware_match' in config
        assert 'combined_score' in config
        
        # Verify scoring
        if config['hardware_match']:
            # If hardware matches recommendation, combined score should equal info gain
            assert abs(config['combined_score'] - config['expected_information_gain']) < 0.01
        else:
            # If hardware doesn't match, combined score should include hardware factor
            assert config['combined_score'] != config['expected_information_gain']
```

## Profiling and Performance Testing

Profiling tests measure the performance of the system to identify bottlenecks.

### Running Profiling Tests

```bash
# Profile prediction performance
python -m cProfile -o prediction_profile.prof test_prediction_performance.py

# Profile active learning performance
python -m cProfile -o active_learning_profile.prof test_active_learning_performance.py

# Analyze profile results
python -m pstats prediction_profile.prof
```

### Performance Metrics to Measure

1. **Prediction Time**: Time required to make predictions for different model-hardware combinations
2. **Recommendation Generation Time**: Time required to generate recommendations
3. **Memory Usage**: Memory consumption during different operations
4. **Scaling Behavior**: Performance scaling with increasing dataset size

## Regression Testing

Regression tests ensure that system updates don't break existing functionality.

### Running Regression Tests

```bash
# Run regression test suite
python regression_test_suite.py

# Check for performance regressions
python check_performance_regression.py --baseline baseline_metrics.json
```

### Regression Test Strategy

1. **Baseline Creation**: Create baseline performance and accuracy metrics
2. **Automated Comparison**: Automatically compare new test results against baselines
3. **Failure Thresholds**: Define thresholds for acceptable regressions
4. **History Tracking**: Track test results over time to identify trends

## Continuous Integration

The testing system integrates with CI/CD pipelines for automated testing.

### CI/CD Integration

```yaml
# Example CI configuration (GitHub Actions)
name: Predictive Performance Tests

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
      - name: Run tests
        run: |
          python -m pytest test_*.py
      - name: Run integration tests
        run: |
          python test_integration.py
```

## Troubleshooting Tests

### Common Test Issues

1. **Missing Dependencies**: Ensure all required packages are installed
2. **Database Access**: Verify that the test database is accessible
3. **Path Issues**: Check that file paths are correctly specified
4. **Random Seed**: Set random seeds for reproducible tests
5. **Test Data Size**: Use appropriately sized test data to avoid memory issues

### Debugging Failed Tests

```bash
# Run tests with increased verbosity
python -m pytest test_active_learning.py -vv

# Run tests with debugging output
python -m pytest test_active_learning.py --debug

# Run specific failed test
python -m pytest test_active_learning.py::TestUncertaintyEstimation::test_uncertainty_calculation -v
```

## Test Coverage

### Measuring Test Coverage

```bash
# Run tests with coverage
python -m pytest --cov=predictive_performance test_*.py

# Generate HTML coverage report
python -m pytest --cov=predictive_performance --cov-report=html test_*.py
```

### Coverage Goals

- **Unit Test Coverage**: Aim for >90% coverage of core functionality
- **Integration Test Coverage**: Aim for >80% coverage of integration points
- **Edge Case Coverage**: Ensure tests cover error conditions and edge cases

## Validation Tests

Validation tests ensure the system meets its requirements.

### Validation Criteria

1. **Prediction Accuracy**: Predictions must be within 15% of actual values
2. **Recommendation Quality**: Recommendations must improve prediction accuracy
3. **Performance Requirements**: System must respond within specified time limits
4. **Usability Requirements**: System must be easy to use with clear outputs

### Validation Test Example

```python
def test_prediction_accuracy():
    """Validate that predictions meet accuracy requirements."""
    # Load test data with known values
    test_data = load_test_data("known_values.csv")
    
    # Make predictions
    predictor = PerformancePredictor()
    predictions = []
    actuals = []
    
    for config in test_data:
        prediction = predictor.predict(
            model_name=config['model_name'],
            model_type=config['model_type'],
            hardware_platform=config['hardware'],
            batch_size=config['batch_size']
        )
        
        predictions.append(prediction['throughput'])
        actuals.append(config['actual_throughput'])
    
    # Calculate mean absolute percentage error
    mape = numpy.mean(numpy.abs((numpy.array(actuals) - numpy.array(predictions)) / numpy.array(actuals))) * 100
    
    # Validate accuracy requirement
    assert mape < 15, f"Prediction MAPE of {mape:.2f}% exceeds maximum allowed (15%)"
```

## Conclusion

Following this testing guide ensures the Predictive Performance System, particularly the newly implemented Active Learning Pipeline and its integration with the Hardware Recommender, functions correctly and reliably. The comprehensive test suite covers all aspects of the system from individual components to end-to-end workflows, providing confidence in the system's accuracy and robustness.

For more information about the Predictive Performance System, refer to the [PREDICTIVE_PERFORMANCE_GUIDE.md](PREDICTIVE_PERFORMANCE_GUIDE.md) and [INTEGRATED_ACTIVE_LEARNING_GUIDE.md](INTEGRATED_ACTIVE_LEARNING_GUIDE.md).