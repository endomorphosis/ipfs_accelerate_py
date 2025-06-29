# Simulation Validation Report - 2025-03-14 20:25:44

Generated on: 2025-03-14 20:25:44

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Overview](#overview)
3. [Hardware Comparison](#hardware-comparison)
4. [Model Comparison](#model-comparison)
5. [Metric Analysis](#metric-analysis)
6. [Statistical Analysis](#statistical-analysis)
7. [Detailed Results](#detailed-results)
8. [Recommendations](#recommendations)

## Executive Summary

### Key Metrics

- **Total Results:** 36
- **Hardware Types:** 3
- **Model Types:** 4
- **Overall MAPE:** 9.62%
- **Status:** good

### Statistical Metrics

- **Mean MAPE:** 9.62%
- **Median MAPE:** 9.43%
- **Standard Deviation:** 5.88%
- **95% Confidence Interval:** 8.64% - 10.59%

### Best and Worst Metrics

- **Best performing metric:** power_consumption_w (8.85% MAPE)
- **Worst performing metric:** throughput_items_per_second (10.92% MAPE)

### Best and Worst Hardware-Model Combinations

- **Best combination:** model_2 on hardware_0 (6.32% MAPE)
- **Worst combination:** model_0 on hardware_0 (12.09% MAPE)

## Overview

This report analyzes simulation validation results, comparing simulation predictions with actual hardware measurements.

### Summary

- **Total validation results:** 36
- **Overall MAPE:** 9.62%
- **Overall status:** good

### What is MAPE?

Mean Absolute Percentage Error (MAPE) measures the average percentage difference between simulated and actual values. Lower values indicate better simulation accuracy.

- **Excellent (< 5%):** Simulation is highly accurate
- **Good (5-10%):** Simulation is very reliable
- **Acceptable (10-15%):** Simulation is usable but could be improved
- **Problematic (15-25%):** Simulation needs calibration
- **Poor (> 25%):** Simulation requires significant improvement

## Hardware Comparison

This section compares simulation accuracy across different hardware types.

| Hardware | Count | MAPE | Status |
| --- | --- | --- | --- |
| hardware_0 | 12 | 8.91% | good |
| hardware_1 | 12 | 10.12% | acceptable |
| hardware_2 | 12 | 9.81% | good |

## Model Comparison

This section compares simulation accuracy across different model types.

| Model | Count | MAPE | Status |
| --- | --- | --- | --- |
| model_0 | 9 | 11.08% | acceptable |
| model_1 | 9 | 10.76% | acceptable |
| model_2 | 9 | 8.43% | good |
| model_3 | 9 | 8.19% | good |

## Metric Analysis

This section shows validation results grouped by hardware and model combinations.

| Hardware | Model | Count | MAPE | Status |
| --- | --- | --- | --- | --- |
| hardware_0 | model_0 | 3 | 12.09% | acceptable |
| hardware_0 | model_1 | 3 | 9.23% | good |
| hardware_0 | model_2 | 3 | 6.32% | good |
| hardware_0 | model_3 | 3 | 8.01% | good |
| hardware_1 | model_0 | 3 | 10.91% | acceptable |
| hardware_1 | model_1 | 3 | 11.32% | acceptable |
| hardware_1 | model_2 | 3 | 9.60% | good |
| hardware_1 | model_3 | 3 | 8.65% | good |
| hardware_2 | model_0 | 3 | 10.23% | acceptable |
| hardware_2 | model_1 | 3 | 11.75% | acceptable |
| hardware_2 | model_2 | 3 | 9.37% | good |
| hardware_2 | model_3 | 3 | 7.91% | good |

## Statistical Analysis

This section provides statistical analysis of the validation results, including confidence intervals and error distributions.

_Note: Visualizations are not available in Markdown format. Please use HTML format to view visualizations._

## Detailed Results

This section shows detailed validation results for individual simulations.

Showing up to 20 of 36 results

| Hardware | Model | Batch Size | Precision | Throughput MAPE | Latency MAPE | Memory MAPE | Power MAPE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| hardware_0 | model_0 | 4 | int8 | 18.26% | 15.01% | 16.85% | 5.15% |
| hardware_0 | model_0 | 4 | int8 | 17.22% | 10.92% | 14.21% | 4.79% |
| hardware_0 | model_0 | 4 | fp32 | 11.62% | 17.47% | 4.37% | 9.17% |
| hardware_0 | model_1 | 8 | fp16 | 5.60% | 7.38% | 11.04% | 18.72% |
| hardware_0 | model_1 | 4 | fp32 | 17.72% | 7.41% | 4.17% | 1.65% |
| hardware_0 | model_1 | 2 | int8 | 9.49% | 15.91% | 11.23% | 0.38% |
| hardware_0 | model_2 | 1 | fp32 | 13.60% | 3.86% | 3.84% | 9.94% |
| hardware_0 | model_2 | 1 | int8 | 7.27% | 9.33% | 15.54% | 1.85% |
| hardware_0 | model_2 | 4 | int8 | 0.31% | 5.01% | 0.31% | 5.04% |
| hardware_0 | model_3 | 16 | fp32 | 0.46% | 5.48% | 12.27% | 10.37% |
| hardware_0 | model_3 | 1 | fp32 | 12.64% | 1.31% | 6.39% | 13.39% |
| hardware_0 | model_3 | 16 | fp16 | 17.16% | 7.59% | 4.04% | 5.06% |
| hardware_1 | model_0 | 8 | fp32 | 17.44% | 2.16% | 19.71% | 17.65% |
| hardware_1 | model_0 | 1 | fp16 | 10.48% | 11.43% | 17.42% | 1.09% |
| hardware_1 | model_0 | 8 | fp32 | 6.36% | 3.56% | 17.83% | 5.85% |
| hardware_1 | model_1 | 4 | fp32 | 12.39% | 3.12% | 10.01% | 2.27% |
| hardware_1 | model_1 | 16 | fp32 | 19.97% | 18.76% | 13.95% | 0.57% |
| hardware_1 | model_1 | 8 | fp32 | 19.91% | 17.06% | 8.59% | 9.22% |
| hardware_1 | model_2 | 8 | fp32 | 11.29% | 9.43% | 7.46% | 0.93% |
| hardware_1 | model_2 | 16 | fp32 | 0.31% | 17.26% | 14.47% | 17.66% |

## Recommendations

Based on the validation results, the following recommendations are provided:

### Maintain Current Performance

The overall MAPE of 9.62% indicates good simulation accuracy. Continue monitoring for drift and consider further fine-tuning for critical workloads.

### Regular Drift Detection

Run drift detection regularly to identify changes in simulation accuracy over time.

---

*Generated by Simulation Accuracy and Validation Framework*