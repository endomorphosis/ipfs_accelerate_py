# Active Learning Pipeline Design Document

## Overview

The Active Learning Pipeline is a core component of the Predictive Performance System, designed to intelligently identify and prioritize high-value benchmark configurations for testing. By targeting configurations with high uncertainty or expected information gain, we can maximize the improvement in prediction accuracy while minimizing the number of actual benchmarks that need to be run.

## System Architecture

The Active Learning Pipeline consists of the following components:

1. **Uncertainty Estimation System**:
   - Measures prediction uncertainty using ensemble variance, quantile regression, and distance metrics
   - Identifies configurations with high uncertainty that would benefit from real benchmarking
   - Ranks configurations by uncertainty to prioritize testing

2. **Information Gain Estimator**:
   - Calculates expected information gain from testing specific configurations
   - Uses mutual information and entropy reduction techniques
   - Evaluates how much a test would improve overall prediction accuracy

3. **Exploration Strategy**:
   - Balances exploration (testing diverse configurations) vs. exploitation (refining high-uncertainty areas)
   - Implements diversity-aware sampling to ensure coverage across the feature space
   - Uses Thompson sampling and upper confidence bound techniques

4. **Test Batch Generator**:
   - Creates optimized batches of test configurations
   - Ensures batch diversity while maximizing information gain
   - Balances hardware and model constraints

5. **Model Update Pipeline**:
   - Incorporates new benchmark results back into the predictive models
   - Performs incremental model updates without full retraining
   - Tracks model improvement over successive updates

## Uncertainty Estimation Methods

### Ensemble Variance
We use an ensemble of gradient boosting models to estimate prediction uncertainty:

```python
def calculate_ensemble_variance(models, features):
    """Calculate variance across ensemble predictions."""
    predictions = [model.predict(features) for model in models]
    return np.var(predictions, axis=0)
```

### Quantile Regression
For direct uncertainty estimation using quantile regression:

```python
def calculate_prediction_interval(model, features):
    """Calculate prediction interval using quantile regression."""
    low = model_low.predict(features)  # 10th percentile
    high = model_high.predict(features)  # 90th percentile
    return high - low  # Width of prediction interval
```

### Distance-Based Uncertainty
Measure uncertainty based on distance to training examples:

```python
def calculate_distance_uncertainty(features, training_features, n_neighbors=5):
    """Calculate uncertainty based on distance to nearest training examples."""
    distances, _ = NearestNeighbors(n_neighbors=n_neighbors).fit(training_features).kneighbors(features)
    return np.mean(distances, axis=1)  # Average distance to k-nearest neighbors
```

## Information Gain Calculation

We compute expected information gain using the following approaches:

### Expected Model Change
Calculate how much the model would change if we added a new benchmark result:

```python
def expected_model_change(model, features, uncertainty):
    """Estimate expected model change from new benchmark data."""
    # Simulate possible benchmark outcomes based on prediction distribution
    simulated_outcomes = np.random.normal(
        model.predict(features), 
        uncertainty, 
        size=(n_simulations, len(features))
    )
    
    # Estimate model change for each simulation
    changes = []
    for outcomes in simulated_outcomes:
        model_copy = copy.deepcopy(model)
        model_copy.update(features, outcomes)
        changes.append(calculate_model_difference(model, model_copy))
    
    return np.mean(changes, axis=0)
```

### Expected Entropy Reduction
Estimate reduction in overall prediction uncertainty:

```python
def expected_entropy_reduction(model, features, training_features):
    """Estimate entropy reduction from new benchmark data."""
    current_entropy = calculate_model_entropy(model, training_features)
    
    # Simulate possible benchmark outcomes
    simulated_outcomes = sample_possible_outcomes(model, features)
    
    # Calculate expected entropy after adding new benchmark data
    expected_entropy = 0
    for outcome in simulated_outcomes:
        model_copy = copy.deepcopy(model)
        model_copy.update(features, outcome)
        new_entropy = calculate_model_entropy(model_copy, training_features)
        expected_entropy += new_entropy * outcome_probability(outcome)
    
    return current_entropy - expected_entropy
```

## Exploration Strategies

### Thompson Sampling
Balances exploration and exploitation:

```python
def thompson_sampling(models, features, n_samples=10):
    """Select configurations using Thompson sampling."""
    samples = []
    for _ in range(n_samples):
        # Randomly select a model from the ensemble
        model_idx = np.random.randint(len(models))
        model = models[model_idx]
        
        # Make predictions with the selected model
        predictions = model.predict(features)
        
        # Select configuration with highest predicted value
        best_idx = np.argmax(predictions)
        samples.append(best_idx)
    
    # Count frequency of each configuration
    counts = np.bincount(samples)
    
    # Return configurations in order of selection frequency
    return np.argsort(counts)[::-1]
```

### Diversity-Aware Sampling
Ensures diversity in selected configurations:

```python
def diversity_sampling(features, uncertainty, n_samples=10, diversity_weight=0.5):
    """Select diverse configurations with high uncertainty."""
    selected = []
    remaining = list(range(len(features)))
    
    # Select first configuration with highest uncertainty
    first_idx = np.argmax(uncertainty)
    selected.append(first_idx)
    remaining.remove(first_idx)
    
    # Select remaining configurations
    for _ in range(n_samples - 1):
        scores = []
        for idx in remaining:
            # Calculate diversity as minimum distance to already selected points
            diversity = min(euclidean(features[idx], features[s]) for s in selected)
            
            # Calculate score as weighted combination of uncertainty and diversity
            score = (1 - diversity_weight) * uncertainty[idx] + diversity_weight * diversity
            scores.append(score)
        
        # Select configuration with highest score
        best_idx = remaining[np.argmax(scores)]
        selected.append(best_idx)
        remaining.remove(best_idx)
    
    return selected
```

## Test Batch Generation

### Batch Optimization
Generates optimized test batches:

```python
def generate_test_batch(features, uncertainty, information_gain, 
                        batch_size=10, hardware_constraints=None):
    """Generate optimized batch of test configurations."""
    # Calculate combined score for each configuration
    scores = 0.4 * normalize(uncertainty) + 0.6 * normalize(information_gain)
    
    # Apply hardware constraints if provided
    if hardware_constraints is not None:
        scores = apply_hardware_constraints(scores, hardware_constraints)
    
    # Select configurations with highest scores while ensuring diversity
    selected = diversity_sampling(features, scores, n_samples=batch_size)
    
    return selected
```

### Hardware-Aware Batching
Accounts for hardware availability constraints:

```python
def hardware_aware_batching(features, scores, hardware_availability):
    """Adjust scores based on hardware availability."""
    hardware_type = features[:, hardware_type_idx]
    
    # Apply scaling factor based on hardware availability
    for hw_type, availability in hardware_availability.items():
        hw_mask = hardware_type == hw_type
        scores[hw_mask] *= availability
    
    return scores
```

## Model Update Pipeline

### Incremental Model Update
Updates models with new benchmark data without full retraining:

```python
def incremental_update(model, features, targets, learning_rate=0.1):
    """Update model incrementally with new benchmark data."""
    # Create mini-batch with new data
    X_batch = features
    y_batch = targets
    
    # Perform incremental update
    model.set_params(learning_rate=learning_rate)
    model.fit(X_batch, y_batch, xgb_model=model)
    
    return model
```

### Model Improvement Tracking
Tracks prediction accuracy improvement from new benchmark data:

```python
def track_model_improvement(model_before, model_after, eval_features, eval_targets):
    """Track model improvement after incorporating new benchmark data."""
    # Calculate metrics before update
    pred_before = model_before.predict(eval_features)
    rmse_before = np.sqrt(mean_squared_error(eval_targets, pred_before))
    r2_before = r2_score(eval_targets, pred_before)
    
    # Calculate metrics after update
    pred_after = model_after.predict(eval_features)
    rmse_after = np.sqrt(mean_squared_error(eval_targets, pred_after))
    r2_after = r2_score(eval_targets, pred_after)
    
    # Calculate improvement
    rmse_improvement = (rmse_before - rmse_after) / rmse_before * 100
    r2_improvement = (r2_after - r2_before) / (1 - r2_before) * 100
    
    return {
        "rmse_improvement_percent": rmse_improvement,
        "r2_improvement_percent": r2_improvement,
        "rmse_before": rmse_before,
        "rmse_after": rmse_after,
        "r2_before": r2_before,
        "r2_after": r2_after
    }
```

## API Design

### Key Functions

1. **identify_high_value_tests()**
   - Identifies configurations with high expected information gain
   - Returns ranked list of configurations to test

2. **prioritize_tests()**
   - Ranks configurations by expected value considering constraints
   - Accounts for hardware availability and testing costs

3. **suggest_test_batch()**
   - Generates a batch of tests optimized for information gain
   - Balances diversity and uncertainty

4. **update_models()**
   - Incorporates new benchmark results into predictive models
   - Updates uncertainty estimates based on new data

### Usage Example

```python
# Initialize active learning system
active_learner = ActiveLearningPipeline(trained_models, training_data)

# Generate candidate configurations to evaluate
candidates = generate_candidate_configurations(
    model_types=["bert", "vit", "whisper"],
    hardware_platforms=["cpu", "cuda", "webgpu"],
    batch_sizes=[1, 4, 16]
)

# Identify high-value tests
high_value_tests = active_learner.identify_high_value_tests(
    candidates, 
    uncertainty_threshold=0.3,
    max_tests=100
)

# Prioritize tests based on constraints
prioritized_tests = active_learner.prioritize_tests(
    high_value_tests,
    hardware_availability={"cuda": 0.8, "webgpu": 0.3, "cpu": 1.0},
    cost_weights={"time": 0.6, "resources": 0.4}
)

# Generate optimized test batch
test_batch = active_learner.suggest_test_batch(
    prioritized_tests,
    batch_size=10,
    ensure_diversity=True
)

# Run benchmarks for the test batch
benchmark_results = run_benchmarks(test_batch)

# Update models with new benchmark data
improvement_metrics = active_learner.update_models(benchmark_results)
```

## Implementation Timeline

| Component | Status | Target Completion |
|-----------|--------|------------------|
| Uncertainty Estimation System | ✅ COMPLETED (100%) | March 10, 2025 |
| Information Gain Estimator | ✅ COMPLETED (100%) | March 10, 2025 |
| Exploration Strategy | ✅ COMPLETED (100%) | March 10, 2025 |
| Hardware Recommender Integration | ✅ COMPLETED (100%) | March 10, 2025 |
| Test Batch Generator | ✅ COMPLETED (100%) | March 15, 2025 |
| Model Update Pipeline | ✅ COMPLETED (100%) | March 18, 2025 |
| API Integration | 🔄 IN PROGRESS (60%) | April 20, 2025 |
| Comprehensive Testing | 🔄 IN PROGRESS (30%) | May 10, 2025 |
| Documentation | 🔄 IN PROGRESS (70%) | May 20, 2025 |

## Hardware Recommender Integration

The Active Learning Pipeline has been integrated with the Hardware Recommender to create a powerful combined system that can:

1. Identify configurations with high uncertainty or expected information gain
2. Evaluate these configurations against optimal hardware recommendations
3. Prioritize configurations that would provide both valuable information and hardware optimization insights

### Integration Design

The integrated system combines the strengths of both components:

- **From Active Learning**: Identification of high-value tests based on uncertainty and diversity
- **From Hardware Recommender**: Optimal hardware selection based on model characteristics and constraints

The integration uses a weighted scoring system that combines:
- 70% information gain from active learning
- 30% hardware optimality from hardware recommender

This allows the system to prioritize configurations that are both informative for improving predictions and relevant for hardware optimization.

### Usage Example

```python
# Initialize components
active_learner = ActiveLearningSystem()
hardware_recommender = HardwareRecommender(predictor=predictor)

# Generate integrated recommendations
results = active_learner.integrate_with_hardware_recommender(
    hardware_recommender=hardware_recommender,
    test_budget=10,
    optimize_for="throughput"
)

# Access recommended configurations
for config in results["recommendations"]:
    print(f"Model: {config['model_name']}")
    print(f"Current Hardware: {config['hardware']}")
    print(f"Recommended Hardware: {config['recommended_hardware']}")
    print(f"Combined Score: {config['combined_score']}")
```

### Benefits of Integration

- **Efficiency**: Tests chosen maximize information gain while validating hardware selection
- **Intelligent Exploration**: Prioritizes configurations with hardware mismatches that would benefit from investigation
- **Resource Optimization**: Combines two selection processes into a single pipeline
- **Dual Improvement**: Simultaneously improves prediction accuracy and hardware recommendation reliability

## Conclusion

The Active Learning Pipeline, now integrated with the Hardware Recommender, enables strategic allocation of benchmark resources by identifying the most valuable configurations to test. By focusing on high-uncertainty areas and maximizing information gain while considering hardware optimization, we can achieve significantly higher prediction accuracy with fewer benchmark runs, reducing resource requirements while improving model quality and hardware selection.