#!/usr/bin/env python3
"""
Advanced Analysis Methods for the Simulation Accuracy and Validation Framework.

This package provides advanced analysis capabilities for the framework, including:
- Statistical analysis beyond basic metrics
- Machine learning-based pattern recognition
- Anomaly detection for unusual validation results
- Predictive modeling for simulation accuracy
- Trend projection for future accuracy estimates
"""

from duckdb_api.simulation_validation.analysis.base import AnalysisMethod
from duckdb_api.simulation_validation.analysis.advanced_statistical_analysis import AdvancedStatisticalAnalysis
from duckdb_api.simulation_validation.analysis.ml_pattern_analysis import MLPatternAnalysis
from duckdb_api.simulation_validation.analysis.anomaly_detection import AnomalyDetection
from duckdb_api.simulation_validation.analysis.predictive_modeling import PredictiveModeling
from duckdb_api.simulation_validation.analysis.trend_projection import TrendProjection