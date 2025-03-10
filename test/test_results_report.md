# WebGPU Compute Shaders for 4-bit Inference Test Report
## Operation: mlp, 4-bit
Date: 2025-03-03 01:44:49

## Summary
- Operation: mlp
- Precision: 4-bit
- Browser: chrome
- Adaptive Precision: Enabled
- Model Size: tiny (1.1B)

## Shader Generation
- Generated Lines: 155
- Generation Time: 0.01ms

## Adaptive Precision Benchmark
- Baseline FP16 Memory: 126.07MB
- Best Configuration: Uniform 4-bit
- Memory Reduction: 75.0%
- Speed Improvement: 3.52x
- Accuracy Impact: 1.30%

### Configuration Comparison
| Configuration | Memory (MB) | Reduction | Speed | Accuracy Impact | Score |
|---------------|------------|-----------|-------|----------------|-------|
| Uniform 4-bit | 31.57 | 75.0% | 3.52x | 1.30% | 150.4 |
| 8-bit attention, 2-bit mlp | 36.07 | 71.4% | 3.44x | 1.50% | 145.9 |
| 8-bit attention, 4-bit rest | 45.07 | 64.2% | 2.80x | 0.80% | 122.5 |
| Fully adaptive | 40.57 | 67.8% | 2.69x | 0.80% | 121.4 |
| 16-bit attention, 4-bit rest | 72.07 | 42.8% | 2.20x | 0.50% | 91.6 |
