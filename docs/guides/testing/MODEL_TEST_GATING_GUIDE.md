# Model Test Gating and Performance Monitoring

This document explains the new model test gating system and performance monitoring features.

## Model Test Gating

### Problem
Running 1000+ model tests every time you test the framework is slow and unnecessary.

### Solution
Model tests are now gated behind a special flag and don't run by default.

### Usage

**Run framework tests only (default):**
```bash
pytest
# or
pytest test/api test/distributed_testing
```

**Run with model tests:**
```bash
pytest --run-model-tests
```

**Run specific model tests:**
```bash
pytest --run-model-tests test/improved/test_hf_bert_improved.py
```

**Run model tests matching a pattern:**
```bash
pytest --run-model-tests -k "bert or gpt"
```

### How It Works

1. All model tests are marked with `@pytest.mark.model_test`
2. The `conftest.py` file has a `pytest_collection_modifyitems()` hook
3. Without `--run-model-tests`, model tests are automatically skipped
4. With `--run-model-tests`, all tests run normally

### Marking Your Tests

Add the `@pytest.mark.model_test` marker to test classes:

```python
@pytest.mark.model_test
@pytest.mark.model
@pytest.mark.text
class TestBERTInference:
    def test_forward_pass(self, model_and_tokenizer, sample_inputs):
        # Test implementation
        pass
```

## Performance Monitoring & Regression Detection

### Features

1. **Baseline Storage**: Performance baselines stored in `test/.performance_baselines.json`
2. **Automatic Comparison**: Tests automatically compare against baselines
3. **Regression Detection**: Warns if performance degrades beyond tolerance
4. **Configurable Tolerance**: Default 20% tolerance, adjustable via flag

### Usage

**Update baselines (first time or after optimization):**
```bash
pytest --run-model-tests --update-baselines
```

**Run with regression detection (default):**
```bash
pytest --run-model-tests
```

**Custom tolerance (e.g., 10%):**
```bash
pytest --run-model-tests --baseline-tolerance 0.10
```

### In Your Tests

Performance tests should use the `PerformanceTestUtils` helper:

```python
@pytest.mark.model_test
@pytest.mark.benchmark
class TestModelPerformance:
    def test_performance_with_baseline(self, model_and_tokenizer, 
                                       sample_inputs, pytest_config):
        model, _ = model_and_tokenizer
        
        # Measure performance
        timing_stats = ModelTestUtils.measure_inference_time(
            model, inputs, warmup_runs=3, test_runs=10
        )
        
        # Check for regressions
        result = PerformanceTestUtils.check_performance_regression(
            model_name="MyModel",
            timing_stats=timing_stats,
            device="cpu",
            tolerance=pytest_config.baseline_tolerance,
            update_baseline=pytest_config.update_baselines
        )
        
        # Log results
        if result.get('regressions'):
            import warnings
            warnings.warn(f"Performance regressions: {result['message']}")
```

### Baseline File Format

The baseline file (`test/.performance_baselines.json`) stores metrics per model and device:

```json
{
  "BERT_cpu": {
    "inference_time_mean": 0.0234,
    "inference_time_median": 0.0231,
    "inference_time_min": 0.0225,
    "inference_time_max": 0.0245,
    "memory_peak_mb": 456.78,
    "timestamp": "2026-02-02T01:00:00.000000",
    "device": "cpu"
  },
  "BERT_cuda": {
    "inference_time_mean": 0.0034,
    "memory_peak_mb": 512.34,
    "timestamp": "2026-02-02T01:00:00.000000",
    "device": "cuda"
  }
}
```

## Bulk Test Conversion

### Converting Legacy Tests

Use the bulk conversion script to convert existing test_hf_*.py files:

```bash
# Convert 10 tests (for testing)
python scripts/convert_tests_bulk.py --limit 10

# Convert all tests
python scripts/convert_tests_bulk.py

# Convert specific pattern
python scripts/convert_tests_bulk.py --pattern "test_hf_gpt*.py"

# Overwrite existing files
python scripts/convert_tests_bulk.py --overwrite
```

### What Gets Converted

The script automatically:
1. Extracts model ID, name, and task type
2. Detects model category (text, vision, audio, multimodal)
3. Generates pytest-compatible test from template
4. Adds `@pytest.mark.model_test` markers
5. Includes performance monitoring

### Conversion Output

```
======================================================================
Bulk Test Conversion
======================================================================
Input directory:  test
Output directory: test/improved
Pattern:          test_hf_*.py
Files found:      1018
======================================================================

[1/1018] ✅ Converted test_hf_bert.py → test_hf_bert_improved.py
[2/1018] ✅ Converted test_hf_gpt2.py → test_hf_gpt2_improved.py
...
======================================================================
Conversion complete: 1018/1018 successful
======================================================================
```

## Coverage Reporting

### Generate Coverage Report

```bash
# HTML report
pytest --run-model-tests --cov=ipfs_accelerate_py --cov-report=html

# Terminal report
pytest --run-model-tests --cov=ipfs_accelerate_py --cov-report=term

# Both
pytest --run-model-tests --cov=ipfs_accelerate_py --cov-report=html --cov-report=term
```

### View HTML Report

```bash
# Open in browser
open htmlcov/index.html
# or
firefox htmlcov/index.html
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Tests

on: [push, pull_request]

jobs:
  framework-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run framework tests
        run: pytest
      
  model-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run model tests with regression detection
        run: pytest --run-model-tests --baseline-tolerance 0.20
      
  model-tests-update-baselines:
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v2
      - name: Update performance baselines
        run: pytest --run-model-tests --update-baselines
      - name: Commit updated baselines
        run: |
          git add test/.performance_baselines.json
          git commit -m "Update performance baselines"
          git push
```

## FAQ

### Q: Why are my model tests not running?

A: Model tests require the `--run-model-tests` flag:
```bash
pytest --run-model-tests
```

### Q: How do I update baselines after making optimizations?

A: Use the `--update-baselines` flag:
```bash
pytest --run-model-tests --update-baselines
```

### Q: Can I adjust the regression tolerance?

A: Yes, use `--baseline-tolerance`:
```bash
pytest --run-model-tests --baseline-tolerance 0.30  # 30% tolerance
```

### Q: Where are baselines stored?

A: In `test/.performance_baselines.json` at the repository root.

### Q: Do regressions fail tests?

A: No, regressions trigger warnings but don't fail tests. This is intentional to avoid blocking development while still alerting to performance changes.

### Q: How do I run only fast tests?

A: Use pytest markers:
```bash
pytest --run-model-tests -m "not slow"
```

### Q: Can I run tests for specific hardware?

A: Yes, use hardware markers:
```bash
pytest --run-model-tests -m "cuda"
pytest --run-model-tests -m "cpu"
pytest --run-model-tests -m "mps"
```

## Best Practices

1. **Default Development**: Run `pytest` without flags for fast feedback
2. **Before PR**: Run `pytest --run-model-tests` to validate model tests
3. **After Optimization**: Run with `--update-baselines` to update benchmarks
4. **CI Pipeline**: Separate jobs for framework tests and model tests
5. **Performance Tracking**: Monitor baseline file in version control
6. **Coverage**: Generate coverage reports periodically to track trends

## Troubleshooting

### Tests not being skipped

Check that tests have the `@pytest.mark.model_test` marker:
```python
@pytest.mark.model_test  # Required!
class TestMyModel:
    pass
```

### Baselines not updating

Make sure you're using the flag:
```bash
pytest --run-model-tests --update-baselines
```

### Permission errors on baseline file

Ensure the test directory is writable:
```bash
chmod 755 test
```

## Summary

- 🚀 **Fast by default**: `pytest` runs only framework tests
- 🎯 **Selective testing**: `--run-model-tests` for model tests
- 📊 **Performance tracking**: Automatic regression detection
- 🔄 **Easy conversion**: Bulk conversion script for legacy tests
- 📈 **Coverage reports**: Integrated pytest-cov support
- ⚡ **CI/CD ready**: Easy integration with GitHub Actions
