#!/usr/bin/env python
"""
API Monitoring Dashboard

This script provides a monitoring dashboard for the API Distributed Testing Framework,
with real-time metrics visualization and performance analysis.

Features:
1. Real-time metrics display for API performance
2. Historical trend analysis and visualization
3. Anomaly detection with ML-based algorithms
4. Cost tracking and optimization insights
5. Integration with Prometheus/Grafana for advanced monitoring

Usage:
    python api_monitoring_dashboard.py [--port PORT] [--data-dir DIR]
    
    The dashboard will be available at http://localhost:PORT
"""

import os
import sys
import json
import time
import logging
import argparse
import threading
import random
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import uuid
import socketserver
import http.server
from http import HTTPStatus
import webbrowser
import socket
import traceback

# Add project root to python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import API distributed testing components
try:
    from test_api_distributed_integration import APIDistributedTesting, APITestType
    from api_backend_distributed_scheduler import (
        APIBackendScheduler, APIRateLimitStrategy, APICostStrategy
    )
    API_COMPONENTS_AVAILABLE = True
except ImportError:
    API_COMPONENTS_AVAILABLE = False
    print("Warning: API distributed testing components not available.")

# Import anomaly detection module
try:
    from api_anomaly_detection import (
        AnomalyDetector, AnomalyDetectionAlgorithm,
        AnomalySeverity, generate_anomaly_alert_html
    )
    ANOMALY_DETECTION_AVAILABLE = True
except ImportError:
    ANOMALY_DETECTION_AVAILABLE = False
    print("Warning: API anomaly detection module not available.")

# Import predictive analytics module
try:
    from api_predictive_analytics import (
        AnomalyPredictor, TimeSeriesPredictor, 
        PredictionHorizon, ModelType,
        generate_prediction_summary
    )
    PREDICTIVE_ANALYTICS_AVAILABLE = True
except ImportError:
    PREDICTIVE_ANALYTICS_AVAILABLE = False
    print("Warning: API predictive analytics module not available.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("api_dashboard")

# HTML templates
HTML_HEADER = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>API Distributed Testing Dashboard</title>
    <meta http-equiv="refresh" content="30">
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            margin: 0;
            padding: 0;
            background-color: #f8f9fa;
        }
        .container {
            width: 95%;
            margin: 0 auto;
            padding: 20px;
        }
        header {
            background-color: #343a40;
            color: white;
            padding: 1rem;
            margin-bottom: 2rem;
        }
        h1 {
            margin: 0;
            font-size: 2.2rem;
        }
        h2 {
            color: #495057;
            border-bottom: 2px solid #e9ecef;
            padding-bottom: 0.5rem;
            margin-top: 2rem;
        }
        .dashboard-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(500px, 1fr));
            gap: 20px;
            margin-bottom: 2rem;
        }
        .card {
            background: white;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            padding: 20px;
            margin-bottom: 20px;
        }
        .card h3 {
            margin-top: 0;
            border-bottom: 1px solid #e9ecef;
            padding-bottom: 10px;
            color: #495057;
        }
        .status-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 15px;
        }
        .status-item {
            background: #f8f9fa;
            border-radius: 5px;
            padding: 15px;
            border-left: 5px solid #6c757d;
        }
        .status-item.healthy {
            border-left-color: #28a745;
        }
        .status-item.warning {
            border-left-color: #ffc107;
        }
        .status-item.error {
            border-left-color: #dc3545;
        }
        .metric-value {
            font-size: 1.8rem;
            font-weight: bold;
            margin: 10px 0;
        }
        .metric-label {
            font-size: 0.9rem;
            color: #6c757d;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        table th, table td {
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #e9ecef;
        }
        table th {
            background-color: #f8f9fa;
            font-weight: bold;
        }
        tr:hover {
            background-color: #f1f3f5;
        }
        .chart-container {
            width: 100%;
            height: 300px;
            margin: 20px 0;
        }
        .footer {
            text-align: center;
            margin-top: 3rem;
            padding: 1rem;
            background-color: #f8f9fa;
            color: #6c757d;
            font-size: 0.9rem;
        }
        .btn {
            display: inline-block;
            font-weight: 400;
            color: #212529;
            text-align: center;
            vertical-align: middle;
            cursor: pointer;
            background-color: transparent;
            border: 1px solid transparent;
            padding: .375rem .75rem;
            font-size: 1rem;
            line-height: 1.5;
            border-radius: .25rem;
            transition: color .15s ease-in-out, background-color .15s ease-in-out, border-color .15s ease-in-out, box-shadow .15s ease-in-out;
            text-decoration: none;
        }
        .btn-primary {
            color: #fff;
            background-color: #007bff;
            border-color: #007bff;
        }
        .btn-primary:hover {
            background-color: #0069d9;
            border-color: #0062cc;
        }
        .btn-secondary {
            color: #fff;
            background-color: #6c757d;
            border-color: #6c757d;
        }
        .btn-secondary:hover {
            background-color: #5a6268;
            border-color: #545b62;
        }
        .text-success { color: #28a745; }
        .text-warning { color: #ffc107; }
        .text-danger { color: #dc3545; }
        .text-info { color: #17a2b8; }
        .system-status {
            display: flex;
            align-items: center;
            margin-bottom: 10px;
        }
        .status-indicator {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 10px;
        }
        .status-healthy { background-color: #28a745; }
        .status-warning { background-color: #ffc107; }
        .status-error { background-color: #dc3545; }
        .last-updated {
            font-style: italic;
            color: #6c757d;
            text-align: right;
            margin-top: 10px;
        }
        /* Anomaly alerts styling */
        .anomaly-alerts {
            margin-top: 20px;
        }
        .anomaly-alert {
            border-radius: 5px;
            padding: 15px;
            margin-bottom: 10px;
            border-left: 5px solid;
            background-color: #f8f9fa;
        }
        .anomaly-header {
            display: flex;
            justify-content: space-between;
            margin-bottom: 10px;
        }
        .anomaly-severity {
            font-weight: bold;
            text-transform: uppercase;
        }
        .anomaly-timestamp {
            font-size: 0.9rem;
            color: #6c757d;
        }
        .anomaly-title {
            font-size: 1.1rem;
            font-weight: 500;
            margin-bottom: 10px;
        }
        .anomaly-details {
            font-size: 0.9rem;
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 5px;
        }
        .text-danger-bg { border-left-color: #dc3545; }
        .text-warning-bg { border-left-color: #ffc107; }
        .text-info-bg { border-left-color: #17a2b8; }
    </style>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@3.7.1/dist/chart.min.js"></script>
</head>
<body>
    <header>
        <div class="container">
            <h1>API Distributed Testing Dashboard</h1>
        </div>
    </header>
    <div class="container">
"""

HTML_FOOTER = """
        <div class="footer">
            <p>API Distributed Testing Framework Dashboard | Last Updated: {timestamp}</p>
        </div>
    </div>
    <script>
        // Add any JavaScript for interactivity here
        function refreshPage() {
            location.reload();
        }
    </script>
</body>
</html>
"""


class APIDashboardManager:
    """
    Manager for the API monitoring dashboard.
    
    This class collects metrics from the API Distributed Testing Framework
    and generates visualizations for the dashboard.
    """
    
    def __init__(
        self,
        data_dir: str = "api_dashboard_data",
        update_interval: int = 30,
        history_days: int = 7,
        anomaly_detection_enabled: bool = True,
        predictive_analytics_enabled: bool = True
    ):
        """
        Initialize the dashboard manager.
        
        Args:
            data_dir: Directory to store dashboard data
            update_interval: How often to update metrics (seconds)
            history_days: How many days of history to maintain
            anomaly_detection_enabled: Whether to enable anomaly detection
            predictive_analytics_enabled: Whether to enable predictive analytics
        """
        self.data_dir = data_dir
        self.update_interval = update_interval
        self.history_days = history_days
        self.anomaly_detection_enabled = anomaly_detection_enabled
        self.predictive_analytics_enabled = predictive_analytics_enabled
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        
        # State variables
        self.is_running = False
        self.update_thread = None
        self.last_update_time = None
        
        # Dashboard data
        self.api_status = {}
        self.metrics = {
            "api_performance": {},
            "cost_tracking": {},
            "test_history": []
        }
        self.historical_data = {
            "latency": {},
            "throughput": {},
            "reliability": {},
            "cost": {}
        }
        
        # Predictive data
        self.predictions = {}
        self.pattern_analysis = {}
        self.predicted_anomalies = {}
        self.optimization_recommendations = {}
        
        # Initialize anomaly detector if available
        self.anomaly_detector = None
        if self.anomaly_detection_enabled and ANOMALY_DETECTION_AVAILABLE:
            try:
                self.anomaly_detector = AnomalyDetector(
                    algorithm=AnomalyDetectionAlgorithm.ENSEMBLE,
                    sensitivity=1.0,
                    window_size=20,
                    baseline_days=3
                )
                # Load any existing anomaly data
                anomaly_data_dir = os.path.join(data_dir, "anomaly_data")
                os.makedirs(anomaly_data_dir, exist_ok=True)
                self.anomaly_detector.load_anomaly_data(anomaly_data_dir)
                logger.info("Anomaly detector initialized")
            except Exception as e:
                logger.error(f"Error initializing anomaly detector: {e}")
                self.anomaly_detector = None
        
        # Initialize predictive analytics if available
        self.predictor = None
        if self.predictive_analytics_enabled and PREDICTIVE_ANALYTICS_AVAILABLE:
            try:
                predictive_data_dir = os.path.join(data_dir, "predictive_data")
                os.makedirs(predictive_data_dir, exist_ok=True)
                
                # Try to load existing predictive models
                self.predictor = AnomalyPredictor.load_models(predictive_data_dir)
                
                # If no existing models, create new predictor
                if self.predictor is None:
                    self.predictor = AnomalyPredictor(
                        prediction_window=24,  # 24 hours ahead
                        contamination=0.05,
                        min_train_size=48  # Need 2 days of data min
                    )
                
                logger.info("Predictive analytics initialized")
            except Exception as e:
                logger.error(f"Error initializing predictive analytics: {e}")
                self.predictor = None
        
        # Check if API components are available
        if API_COMPONENTS_AVAILABLE:
            try:
                # Initialize the API testing framework
                self.api_testing = APIDistributedTesting()
                
                # Initialize the API backend scheduler
                self.scheduler = APIBackendScheduler()
                
                # Start the scheduler
                self.scheduler.start()
                
                logger.info("API Distributed Testing Framework initialized")
            except Exception as e:
                logger.error(f"Error initializing API framework: {e}")
                self.api_testing = None
                self.scheduler = None
        else:
            self.api_testing = None
            self.scheduler = None
        
        # Initialize with mock data if API components are not available
        if self.scheduler is None:
            self._initialize_mock_data()
        
        logger.info(f"API Dashboard Manager initialized with data directory: {data_dir}")
    
    def _initialize_mock_data(self):
        """Initialize dashboard with mock data when API components aren't available."""
        logger.info("Initializing dashboard with mock data")
        
        # Mock API status
        self.api_status = {
            "openai": {
                "status": "healthy",
                "circuit_breaker_status": "closed",
                "success_rate": 0.98,
                "performance_score": 0.87
            },
            "claude": {
                "status": "healthy",
                "circuit_breaker_status": "closed",
                "success_rate": 0.99,
                "performance_score": 0.92
            },
            "groq": {
                "status": "warning",
                "circuit_breaker_status": "half_open",
                "success_rate": 0.85,
                "performance_score": 0.76
            }
        }
        
        # Mock performance metrics
        self.metrics["api_performance"] = {
            "openai": {
                "avg_latency": 1.2,
                "success_rate": 0.98,
                "throughput": 5.3
            },
            "claude": {
                "avg_latency": 1.5,
                "success_rate": 0.99,
                "throughput": 4.8
            },
            "groq": {
                "avg_latency": 0.9,
                "success_rate": 0.85,
                "throughput": 7.2
            }
        }
        
        # Mock cost tracking
        self.metrics["cost_tracking"] = {
            "openai": {
                "total_cost": 2.45,
                "total_requests": 135,
                "avg_cost_per_request": 0.018
            },
            "claude": {
                "total_cost": 3.81,
                "total_requests": 98,
                "avg_cost_per_request": 0.039
            },
            "groq": {
                "total_cost": 0.65,
                "total_requests": 112,
                "avg_cost_per_request": 0.0058
            }
        }
        
        # Mock test history
        for i in range(10):
            test_type = "latency" if i % 3 == 0 else "throughput" if i % 3 == 1 else "reliability"
            api_type = "openai" if i % 3 == 0 else "claude" if i % 3 == 1 else "groq"
            
            self.metrics["test_history"].append({
                "test_id": f"mock-test-{i}",
                "timestamp": time.time() - (i * 3600),  # 1 hour apart
                "api_type": api_type,
                "test_type": test_type,
                "status": "success",
                "summary": {
                    "avg_latency": 1.2 + (i * 0.1) if test_type == "latency" else None,
                    "requests_per_second": 5.0 + (i * 0.2) if test_type == "throughput" else None,
                    "success_rate": 0.9 + (i * 0.01) if test_type == "reliability" else None
                }
            })
        
        # Generate mock historical data
        self._generate_mock_historical_data()
    
    def _generate_mock_historical_data(self):
        """Generate mock historical data for visualizations."""
        apis = ["openai", "claude", "groq"]
        
        # Current timestamp as end date
        end_time = time.time()
        # Start date (days ago)
        start_time = end_time - (self.history_days * 24 * 60 * 60)
        
        # Generate data points (one per 6 hours)
        interval = 6 * 60 * 60  # 6 hours in seconds
        timestamps = []
        current_time = start_time
        while current_time <= end_time:
            timestamps.append(current_time)
            current_time += interval
        
        # Generate data for each API and metric
        for api in apis:
            # Latency data (seconds)
            latency_data = []
            base_latency = 1.0 if api == "openai" else 1.3 if api == "claude" else 0.8
            
            # Add artificial anomalies for testing
            # Each API will have a few anomalies to demonstrate detection
            anomaly_points = []
            if api == "openai":
                # Add a latency spike
                anomaly_points = [int(len(timestamps) * 0.3), int(len(timestamps) * 0.7)]
            elif api == "claude":
                # Add a latency spike
                anomaly_points = [int(len(timestamps) * 0.5)]
            elif api == "groq":
                # Add a latency spike
                anomaly_points = [int(len(timestamps) * 0.8)]
            
            for i, ts in enumerate(timestamps):
                # Add some trends and variance
                day_factor = i / len(timestamps)
                latency = base_latency * (1 - (day_factor * 0.2))  # Improve over time
                # Add daily fluctuation
                hour = datetime.fromtimestamp(ts).hour
                hour_factor = abs(hour - 12) / 12  # 0 at noon, 1 at midnight
                latency *= (1 + (hour_factor * 0.1))
                
                # Add artificial anomaly if needed
                if i in anomaly_points:
                    latency *= 3.0  # Significant spike
                else:
                    # Add normal noise
                    latency *= (0.9 + (0.2 * random.random()))
                
                latency_data.append({
                    "timestamp": ts,
                    "avg_latency": latency,
                    "min_latency": latency * 0.8,
                    "max_latency": latency * 1.3
                })
            
            self.historical_data["latency"][api] = latency_data
            
            # Throughput data
            throughput_data = []
            base_throughput = 5.0 if api == "openai" else 4.5 if api == "claude" else 7.0
            
            # Add anomaly points for throughput
            anomaly_points = []
            if api == "openai":
                anomaly_points = [int(len(timestamps) * 0.4)]
            elif api == "claude":
                anomaly_points = [int(len(timestamps) * 0.6)]
            elif api == "groq":
                anomaly_points = [int(len(timestamps) * 0.2)]
            
            for i, ts in enumerate(timestamps):
                # Add some trends and variance
                day_factor = i / len(timestamps)
                throughput = base_throughput * (1 + (day_factor * 0.3))  # Improve over time
                # Add daily fluctuation
                hour = datetime.fromtimestamp(ts).hour
                hour_factor = abs(hour - 12) / 12  # 0 at noon, 1 at midnight
                throughput *= (1 - (hour_factor * 0.15))
                
                # Add artificial anomaly if needed
                if i in anomaly_points:
                    throughput *= 0.3  # Significant drop
                else:
                    # Add normal noise
                    throughput *= (0.9 + (0.2 * random.random()))
                
                throughput_data.append({
                    "timestamp": ts,
                    "requests_per_second": throughput
                })
            
            self.historical_data["throughput"][api] = throughput_data
            
            # Reliability data
            reliability_data = []
            base_reliability = 0.98 if api == "openai" else 0.99 if api == "claude" else 0.85
            
            # Add anomaly points for reliability
            anomaly_points = []
            if api == "openai":
                anomaly_points = [int(len(timestamps) * 0.9)]
            elif api == "claude":
                anomaly_points = [int(len(timestamps) * 0.2)]
            elif api == "groq":
                anomaly_points = [int(len(timestamps) * 0.5)]
            
            for i, ts in enumerate(timestamps):
                # Add some trends and variance
                reliability = base_reliability
                
                # Add artificial anomaly if needed
                if i in anomaly_points:
                    reliability *= 0.7  # Significant drop
                elif random.random() < 0.1:  # 10% chance of a small dip
                    reliability *= 0.9
                
                # Add noise
                reliability *= (0.99 + (0.02 * random.random()))
                reliability = min(1.0, reliability)
                
                reliability_data.append({
                    "timestamp": ts,
                    "success_rate": reliability
                })
            
            self.historical_data["reliability"][api] = reliability_data
            
            # Cost data
            cost_data = []
            base_cost = 0.02 if api == "openai" else 0.04 if api == "claude" else 0.006
            
            # Add anomaly points for cost
            anomaly_points = []
            if api == "openai":
                anomaly_points = [int(len(timestamps) * 0.7)]
            elif api == "claude":
                anomaly_points = [int(len(timestamps) * 0.3)]
            elif api == "groq":
                anomaly_points = [int(len(timestamps) * 0.8)]
            
            for i, ts in enumerate(timestamps):
                # Add some trends and variance
                day_factor = i / len(timestamps)
                cost = base_cost * (1 - (day_factor * 0.1))  # Improve over time
                
                # Add artificial anomaly if needed
                if i in anomaly_points:
                    cost *= 3.0  # Significant spike
                else:
                    # Add normal noise
                    cost *= (0.95 + (0.1 * random.random()))
                
                cost_data.append({
                    "timestamp": ts,
                    "cost_per_request": cost,
                    "daily_cost": cost * (50 + int(50 * random.random()))
                })
            
            self.historical_data["cost"][api] = cost_data
        
        # Update anomaly detector with this data if available
        if self.anomaly_detector is not None:
            for api in apis:
                for metric_type in self.historical_data:
                    if api in self.historical_data[metric_type]:
                        # Update baseline
                        self.anomaly_detector.update_baseline(api, metric_type, self.historical_data[metric_type][api])
                        
                        # Detect initial anomalies
                        self.anomaly_detector.detect_anomalies(api, metric_type, self.historical_data[metric_type][api])
    
    def start(self) -> bool:
        """
        Start the dashboard manager.
        
        Returns:
            True if started successfully
        """
        if self.is_running:
            logger.warning("Dashboard manager already running")
            return False
        
        self.is_running = True
        
        # Start update thread
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()
        
        logger.info("Dashboard manager started")
        return True
    
    def stop(self) -> bool:
        """
        Stop the dashboard manager.
        
        Returns:
            True if stopped successfully
        """
        if not self.is_running:
            logger.warning("Dashboard manager not running")
            return False
        
        self.is_running = False
        
        # Wait for thread to stop
        if self.update_thread:
            self.update_thread.join(timeout=5)
        
        # Stop scheduler if it exists
        if self.scheduler:
            self.scheduler.stop()
        
        logger.info("Dashboard manager stopped")
        return True
    
    def _update_loop(self) -> None:
        """Background thread for updating metrics."""
        while self.is_running:
            try:
                self._update_metrics()
                self.last_update_time = time.time()
                
                # Save current data
                self._save_dashboard_data()
                
            except Exception as e:
                logger.error(f"Error updating metrics: {e}")
                logger.error(traceback.format_exc())
            
            # Sleep until next update
            for _ in range(self.update_interval):
                if not self.is_running:
                    break
                time.sleep(1)
    
    def _update_metrics(self) -> None:
        """Update metrics from the API testing framework."""
        logger.debug("Updating metrics...")
        
        # Skip if API components are not available
        if self.scheduler is None or self.api_testing is None:
            if random.random() < 0.3:  # Occasionally update mock data
                self._update_mock_data()
            return
        
        try:
            # Get API status from scheduler
            self.api_status = self.scheduler.get_api_status()
            
            # Get metrics from scheduler
            metrics = self.scheduler.get_scheduling_metrics()
            if "api_performance" in metrics:
                self.metrics["api_performance"] = metrics["api_performance"]
            
            # Get cost report
            cost_report = self.scheduler.get_cost_report()
            if "api_costs" in cost_report:
                self.metrics["cost_tracking"] = {
                    api: {
                        "total_cost": cost,
                        "avg_cost_per_request": cost / self.scheduler.api_profiles[api].total_requests
                        if api in self.scheduler.api_profiles and self.scheduler.api_profiles[api].total_requests > 0 
                        else 0
                    }
                    for api, cost in cost_report["api_costs"].items()
                }
            
            # Update historical data
            self._update_historical_data()
            
            # Update predictive analytics
            self._update_predictive_analytics()
            
        except Exception as e:
            logger.error(f"Error fetching metrics from API framework: {e}")
    
    def _update_predictive_analytics(self) -> None:
        """Update predictive analytics based on historical data."""
        if not self.predictive_analytics_enabled or self.predictor is None:
            return
        
        logger.debug("Updating predictive analytics...")
        
        try:
            # Process each API and metric type
            for metric_type in self.historical_data:
                for api in self.historical_data[metric_type]:
                    # Skip if not enough data
                    if len(self.historical_data[metric_type][api]) < 48:  # Need at least 2 days (48 hourly points)
                        continue
                    
                    # Extract timestamps and values
                    data_points = self.historical_data[metric_type][api]
                    timestamps = [point["timestamp"] for point in data_points]
                    
                    # Get value key for this metric type
                    value_key = {
                        "latency": "avg_latency",
                        "throughput": "requests_per_second",
                        "reliability": "success_rate",
                        "cost": "cost_per_request"
                    }.get(metric_type)
                    
                    if not value_key:
                        continue
                    
                    values = [point.get(value_key, 0) for point in data_points]
                    
                    # Skip if no values
                    if not values:
                        continue
                    
                    # Record data for prediction
                    self.predictor.record_data(api, metric_type, timestamps, values)
                    
                    # Get any previously detected anomalies for this API and metric
                    known_anomalies = self.get_recent_anomalies(hours=168)  # Last week
                    relevant_anomalies = [
                        a for a in known_anomalies 
                        if a.get("api") == api and a.get("metric_type") == metric_type
                    ]
                    
                    if relevant_anomalies:
                        # Update with known anomalies
                        self.predictor.record_data(api, metric_type, timestamps, values, relevant_anomalies)
                    
                    # Generate pattern analysis
                    self.pattern_analysis[(api, metric_type)] = self.predictor.analyze_patterns(
                        api, metric_type, timestamps, values
                    )
                    
                    # Predict future anomalies
                    predicted = self.predictor.predict_anomalies(
                        api, metric_type, timestamps, values,
                        prediction_horizon=PredictionHorizon.MEDIUM_TERM  # 24 hours
                    )
                    
                    if predicted:
                        self.predicted_anomalies[(api, metric_type)] = predicted
                    
                    # Get forecasted values from pattern analysis
                    if (api, metric_type) in self.pattern_analysis:
                        pattern = self.pattern_analysis[(api, metric_type)]
                        if "forecast" in pattern and "predictions" in pattern["forecast"]:
                            predictions = pattern["forecast"]["predictions"]
                            confidence_intervals = pattern["forecast"].get("confidence_intervals", [])
                            
                            # Generate prediction summary
                            if predictions:
                                self.predictions[(api, metric_type)] = generate_prediction_summary(
                                    api, 
                                    metric_type, 
                                    predictions,
                                    confidence_intervals if confidence_intervals else [(p*0.9, p*1.1) for p in predictions],
                                    predicted or []
                                )
                    
                    # Generate cost optimization recommendations if this is cost data
                    if metric_type == "cost":
                        # Create cost data format for recommendations
                        cost_data = [
                            {"timestamp": ts, "cost": val}
                            for ts, val in zip(timestamps, values)
                        ]
                        
                        # Get usage patterns from pattern analysis
                        usage_patterns = {}
                        if (api, "latency") in self.pattern_analysis:
                            usage_patterns = self.pattern_analysis[(api, "latency")].get("seasonality", {})
                        
                        # Generate recommendations
                        self.optimization_recommendations[api] = self.predictor.get_cost_optimization_recommendations(
                            api, cost_data, usage_patterns
                        )
            
            logger.debug("Predictive analytics updated")
            
            # Save predictive models periodically (every 24 hours)
            now = time.time()
            last_save_time = getattr(self, "_last_predictor_save_time", 0)
            if now - last_save_time > 24 * 60 * 60:
                predictive_data_dir = os.path.join(self.data_dir, "predictive_data")
                self.predictor.save_models(predictive_data_dir)
                self._last_predictor_save_time = now
                logger.info("Saved predictive models")
                
        except Exception as e:
            logger.error(f"Error updating predictive analytics: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def _update_mock_data(self) -> None:
        """Update mock data with some random variations."""
        # Update mock API status
        for api in self.api_status:
            # Occasionally change status
            if random.random() < 0.1:
                status_options = ["healthy", "warning", "healthy"]  # More weight to healthy
                self.api_status[api]["status"] = random.choice(status_options)
            
            # Update success rate with small variations
            current_rate = self.api_status[api]["success_rate"]
            self.api_status[api]["success_rate"] = min(1.0, max(0.7, current_rate + random.uniform(-0.02, 0.02)))
            
            # Update performance score with small variations
            current_score = self.api_status[api]["performance_score"]
            self.api_status[api]["performance_score"] = min(1.0, max(0.5, current_score + random.uniform(-0.03, 0.03)))
        
        # Update mock performance metrics
        for api in self.metrics["api_performance"]:
            # Occasionally add an anomaly (5% chance)
            is_anomaly = random.random() < 0.05
            
            # Update latency
            current_latency = self.metrics["api_performance"][api]["avg_latency"]
            if is_anomaly and random.random() < 0.7:  # 70% of anomalies affect latency
                # Add a significant spike (2-4x normal)
                new_latency = current_latency * (2.0 + random.random() * 2.0)
            else:
                # Normal variation
                new_latency = max(0.5, current_latency + random.uniform(-0.1, 0.1))
            self.metrics["api_performance"][api]["avg_latency"] = new_latency
            
            # Update success rate
            current_rate = self.metrics["api_performance"][api]["success_rate"]
            if is_anomaly and random.random() < 0.5:  # 50% of anomalies affect reliability
                # Add a significant drop (0.6-0.8x normal)
                new_rate = current_rate * (0.6 + random.random() * 0.2)
            else:
                # Normal variation
                new_rate = min(1.0, max(0.7, current_rate + random.uniform(-0.02, 0.02)))
            self.metrics["api_performance"][api]["success_rate"] = new_rate
            
            # Update throughput
            current_throughput = self.metrics["api_performance"][api]["throughput"]
            if is_anomaly and random.random() < 0.6:  # 60% of anomalies affect throughput
                # Add a significant drop (0.3-0.6x normal)
                new_throughput = current_throughput * (0.3 + random.random() * 0.3)
            else:
                # Normal variation
                new_throughput = max(1.0, current_throughput + random.uniform(-0.3, 0.3))
            self.metrics["api_performance"][api]["throughput"] = new_throughput
        
        # Update historical data with new data points
        now = time.time()
        for api in self.historical_data["latency"]:
            # Add a new data point to each metric
            current_latency = self.metrics["api_performance"][api]["avg_latency"]
            self.historical_data["latency"][api].append({
                "timestamp": now,
                "avg_latency": current_latency,
                "min_latency": current_latency * 0.8,
                "max_latency": current_latency * 1.3
            })
            
            current_throughput = self.metrics["api_performance"][api]["throughput"]
            self.historical_data["throughput"][api].append({
                "timestamp": now,
                "requests_per_second": current_throughput
            })
            
            current_rate = self.metrics["api_performance"][api]["success_rate"]
            self.historical_data["reliability"][api].append({
                "timestamp": now,
                "success_rate": current_rate
            })
            
            # Add cost data point
            current_cost = self.metrics["cost_tracking"].get(api, {}).get("avg_cost_per_request", 0.02)
            # Occasionally add cost anomaly (3% chance)
            if random.random() < 0.03:
                # Significant cost spike
                new_cost = current_cost * (2.0 + random.random() * 2.0)
            else:
                # Normal variation
                new_cost = current_cost * (0.95 + random.random() * 0.1)
                
            if api not in self.historical_data["cost"]:
                self.historical_data["cost"][api] = []
                
            self.historical_data["cost"][api].append({
                "timestamp": now,
                "cost_per_request": new_cost,
                "daily_cost": new_cost * (50 + int(50 * random.random()))
            })
        
        # Trim historical data to history_days
        cutoff_time = now - (self.history_days * 24 * 60 * 60)
        for metric in self.historical_data:
            for api in self.historical_data[metric]:
                self.historical_data[metric][api] = [
                    dp for dp in self.historical_data[metric][api]
                    if dp["timestamp"] >= cutoff_time
                ]
        
        # Detect anomalies with the anomaly detector if available
        if self.anomaly_detector is not None:
            for api in self.historical_data["latency"]:
                for metric_type in self.historical_data:
                    if api in self.historical_data[metric_type]:
                        # Get recent data (last 24 hours)
                        recent_cutoff = now - (24 * 60 * 60)
                        recent_data = [
                            dp for dp in self.historical_data[metric_type][api]
                            if dp["timestamp"] >= recent_cutoff
                        ]
                        
                        # Detect anomalies
                        if recent_data:
                            self.anomaly_detector.detect_anomalies(api, metric_type, recent_data)
    
    def _update_historical_data(self) -> None:
        """Update historical data with current metrics."""
        # Only update if we have API components
        if self.scheduler is None or self.api_testing is None:
            return
        
        now = time.time()
        
        # Get APIs from scheduler
        apis = list(self.scheduler.api_profiles.keys())
        
        for api in apis:
            profile = self.scheduler.api_profiles.get(api)
            if not profile:
                continue
            
            # Initialize historical data if needed
            for metric in ["latency", "throughput", "reliability", "cost"]:
                if api not in self.historical_data[metric]:
                    self.historical_data[metric][api] = []
            
            # Add latency data
            avg_latency = profile.performance.get_avg_latency()
            if avg_latency is not None:
                self.historical_data["latency"][api].append({
                    "timestamp": now,
                    "avg_latency": avg_latency,
                    "min_latency": min(profile.performance.recent_latencies or [avg_latency]),
                    "max_latency": max(profile.performance.recent_latencies or [avg_latency])
                })
            
            # Add throughput data
            avg_throughput = profile.performance.get_avg_throughput()
            if avg_throughput is not None:
                self.historical_data["throughput"][api].append({
                    "timestamp": now,
                    "requests_per_second": avg_throughput
                })
            
            # Add reliability data
            success_rate = profile.get_success_rate()
            self.historical_data["reliability"][api].append({
                "timestamp": now,
                "success_rate": success_rate
            })
            
            # Add cost data
            total_cost = profile.cost_profile.estimated_cost
            total_requests = profile.total_requests
            cost_per_request = total_cost / total_requests if total_requests > 0 else 0
            
            self.historical_data["cost"][api].append({
                "timestamp": now,
                "cost_per_request": cost_per_request,
                "total_cost": total_cost
            })
        
        # Trim historical data to history_days
        cutoff_time = now - (self.history_days * 24 * 60 * 60)
        for metric in self.historical_data:
            for api in self.historical_data[metric]:
                self.historical_data[metric][api] = [
                    dp for dp in self.historical_data[metric][api]
                    if dp["timestamp"] >= cutoff_time
                ]
    
    def _save_dashboard_data(self) -> None:
        """Save current dashboard data to disk."""
        try:
            # Create a full dashboard data snapshot
            dashboard_data = {
                "timestamp": time.time(),
                "api_status": self.api_status,
                "metrics": self.metrics,
                "historical_data": self.historical_data
            }
            
            # Save to file
            data_file = os.path.join(self.data_dir, "dashboard_data.json")
            with open(data_file, 'w') as f:
                json.dump(dashboard_data, f, indent=2)
            
            logger.debug(f"Saved dashboard data to {data_file}")
            
            # Save anomaly data if available
            if self.anomaly_detector is not None:
                anomaly_data_dir = os.path.join(self.data_dir, "anomaly_data")
                os.makedirs(anomaly_data_dir, exist_ok=True)
                self.anomaly_detector.save_anomaly_data(anomaly_data_dir)
            
        except Exception as e:
            logger.error(f"Error saving dashboard data: {e}")
    
    def get_recent_anomalies(self, hours: int = 24, min_severity: str = "low") -> List[Dict[str, Any]]:
        """
        Get recent anomalies filtered by minimum severity.
        
        Args:
            hours: Number of hours to look back
            min_severity: Minimum severity level to include (low, medium, high, critical)
            
        Returns:
            List of recent anomalies
        """
        if self.anomaly_detector is None:
            return []
        
        # Convert string to severity enum
        severity_map = {
            "low": AnomalySeverity.LOW,
            "medium": AnomalySeverity.MEDIUM,
            "high": AnomalySeverity.HIGH,
            "critical": AnomalySeverity.CRITICAL
        }
        
        min_severity_enum = severity_map.get(min_severity.lower(), AnomalySeverity.LOW)
        
        # Get anomalies from detector
        return self.anomaly_detector.get_recent_anomalies(hours, min_severity_enum)
    
    def get_anomaly_summary(self) -> Dict[str, Any]:
        """
        Get a summary of recent anomalies.
        
        Returns:
            Dictionary with anomaly summary
        """
        if self.anomaly_detector is None:
            return {
                "total_anomalies": 0,
                "severity_counts": {
                    "critical": 0,
                    "high": 0,
                    "medium": 0,
                    "low": 0
                },
                "type_counts": {},
                "api_counts": {},
                "metric_counts": {},
                "latest_anomaly": None
            }
        
        return self.anomaly_detector.get_anomaly_summary()
    
    def get_predictions(self, api: str, metric_type: str) -> Dict[str, Any]:
        """
        Get predictions for an API and metric type.
        
        Args:
            api: API name
            metric_type: Metric type
            
        Returns:
            Dictionary with predictions
        """
        key = (api, metric_type)
        if key in self.predictions:
            return self.predictions[key]
        
        # Return empty predictions structure
        return {
            "api": api,
            "metric": metric_type,
            "forecast_hours": 0,
            "generated_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "predictions": [],
            "anomalies": {
                "detected": False,
                "count": 0,
                "severity_counts": {
                    "critical": 0,
                    "high": 0,
                    "medium": 0,
                    "low": 0
                },
                "first_severe": None
            },
            "trend": {
                "direction": "stable",
                "percent_change": 0
            }
        }
    
    def get_pattern_analysis(self, api: str, metric_type: str) -> Dict[str, Any]:
        """
        Get pattern analysis for an API and metric type.
        
        Args:
            api: API name
            metric_type: Metric type
            
        Returns:
            Dictionary with pattern analysis
        """
        key = (api, metric_type)
        if key in self.pattern_analysis:
            return self.pattern_analysis[key]
        
        # Return empty pattern analysis
        return {
            "status": "insufficient_data"
        }
    
    def get_predicted_anomalies(self, api: str, metric_type: str) -> List[Dict[str, Any]]:
        """
        Get predicted anomalies for an API and metric type.
        
        Args:
            api: API name
            metric_type: Metric type
            
        Returns:
            List of predicted anomalies
        """
        key = (api, metric_type)
        if key in self.predicted_anomalies:
            return self.predicted_anomalies[key]
        
        return []
    
    def get_optimization_recommendations(self, api: str) -> List[Dict[str, Any]]:
        """
        Get cost optimization recommendations for an API.
        
        Args:
            api: API name
            
        Returns:
            List of recommendations
        """
        if api in self.optimization_recommendations:
            return self.optimization_recommendations[api]
        
        return []
    
    def get_all_predictions(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all predictions.
        
        Returns:
            Dictionary of predictions by API and metric type
        """
        # Convert tuple keys to string keys for JSON serialization
        return {f"{api}:{metric}": prediction for (api, metric), prediction in self.predictions.items()}
    
    def load_dashboard_data(self) -> bool:
        """
        Load dashboard data from disk.
        
        Returns:
            True if data was loaded successfully
        """
        data_file = os.path.join(self.data_dir, "dashboard_data.json")
        
        if not os.path.exists(data_file):
            logger.warning(f"Dashboard data file {data_file} does not exist")
            return False
        
        try:
            with open(data_file, 'r') as f:
                dashboard_data = json.load(f)
            
            # Update dashboard data
            if "api_status" in dashboard_data:
                self.api_status = dashboard_data["api_status"]
            
            if "metrics" in dashboard_data:
                self.metrics = dashboard_data["metrics"]
            
            if "historical_data" in dashboard_data:
                self.historical_data = dashboard_data["historical_data"]
            
            logger.info(f"Loaded dashboard data from {data_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading dashboard data: {e}")
            return False
    
    def generate_dashboard_html(self) -> str:
        """
        Generate HTML for the dashboard.
        
        Returns:
            HTML string for the dashboard
        """
        html = HTML_HEADER
        
        # System Status
        html += """
        <h2>System Status</h2>
        <div class="dashboard-grid">
            <div class="card">
                <h3>API Framework Status</h3>
                <div class="system-status">
                    <div class="status-indicator status-healthy"></div>
                    <div>Framework Status: <span class="text-success">Operational</span></div>
                </div>
        """
        
        if self.api_testing is not None and self.scheduler is not None:
            html += f"""
                <div class="system-status">
                    <div class="status-indicator status-healthy"></div>
                    <div>API Testing Framework: <span class="text-success">Connected</span></div>
                </div>
                <div class="system-status">
                    <div class="status-indicator status-healthy"></div>
                    <div>Scheduler: <span class="text-success">Running</span></div>
                </div>
            """
        else:
            html += f"""
                <div class="system-status">
                    <div class="status-indicator status-warning"></div>
                    <div>API Testing Framework: <span class="text-warning">Not Connected (Using Mock Data)</span></div>
                </div>
                <div class="system-status">
                    <div class="status-indicator status-warning"></div>
                    <div>Scheduler: <span class="text-warning">Not Running</span></div>
                </div>
            """
        
        # Add last update time
        if self.last_update_time:
            update_time = datetime.fromtimestamp(self.last_update_time).strftime('%Y-%m-%d %H:%M:%S')
            html += f"""
                <div class="last-updated">Last Updated: {update_time}</div>
            """
        
        html += """
            </div>
            
            <div class="card">
                <h3>API Status Overview</h3>
                <div class="status-grid">
        """
        
        # Add API status cards
        for api, status in self.api_status.items():
            status_class = "healthy"
            if isinstance(status, dict):
                if status.get("status") == "warning" or (status.get("success_rate", 1.0) < 0.9):
                    status_class = "warning"
                elif status.get("status") == "error" or (status.get("success_rate", 1.0) < 0.7):
                    status_class = "error"
            
            status_text = "Healthy"
            if status_class == "warning":
                status_text = "Warning"
            elif status_class == "error":
                status_text = "Error"
            
            circuit_breaker = "Unknown"
            if isinstance(status, dict) and "circuit_breaker_status" in status:
                circuit_breaker = status["circuit_breaker_status"].replace("_", " ").title()
            
            success_rate = "N/A"
            if isinstance(status, dict) and "success_rate" in status:
                success_rate = f"{status['success_rate']:.2%}"
            
            html += f"""
                <div class="status-item {status_class}">
                    <h4>{api.upper()}</h4>
                    <div>Status: <strong>{status_text}</strong></div>
                    <div>Circuit Breaker: {circuit_breaker}</div>
                    <div>Success Rate: {success_rate}</div>
                </div>
            """
        
        html += """
                </div>
            </div>
        </div>
        """
        
        # Performance Metrics
        html += """
        <h2>Performance Metrics</h2>
        <div class="dashboard-grid">
            <div class="card">
                <h3>API Latency (seconds)</h3>
                <table>
                    <thead>
                        <tr>
                            <th>API</th>
                            <th>Avg Latency</th>
                            <th>Min Latency</th>
                            <th>Max Latency</th>
                        </tr>
                    </thead>
                    <tbody>
        """
        
        # Add latency data
        for api in self.api_status:
            avg_latency = "N/A"
            min_latency = "N/A"
            max_latency = "N/A"
            
            # Check if we have performance data
            if api in self.metrics["api_performance"] and "avg_latency" in self.metrics["api_performance"][api]:
                avg_latency = f"{self.metrics['api_performance'][api]['avg_latency']:.3f}s"
            
            # Check if we have historical data
            if api in self.historical_data["latency"] and self.historical_data["latency"][api]:
                latest = self.historical_data["latency"][api][-1]
                if "min_latency" in latest:
                    min_latency = f"{latest['min_latency']:.3f}s"
                if "max_latency" in latest:
                    max_latency = f"{latest['max_latency']:.3f}s"
            
            html += f"""
                <tr>
                    <td>{api.upper()}</td>
                    <td>{avg_latency}</td>
                    <td>{min_latency}</td>
                    <td>{max_latency}</td>
                </tr>
            """
        
        html += """
                    </tbody>
                </table>
                <div class="chart-container">
                    <canvas id="latencyChart"></canvas>
                </div>
            </div>
            
            <div class="card">
                <h3>API Throughput (req/s)</h3>
                <table>
                    <thead>
                        <tr>
                            <th>API</th>
                            <th>Throughput</th>
                            <th>Success Rate</th>
                        </tr>
                    </thead>
                    <tbody>
        """
        
        # Add throughput data
        for api in self.api_status:
            throughput = "N/A"
            success_rate = "N/A"
            
            # Check if we have performance data
            if api in self.metrics["api_performance"]:
                if "throughput" in self.metrics["api_performance"][api]:
                    throughput = f"{self.metrics['api_performance'][api]['throughput']:.2f} req/s"
                if "success_rate" in self.metrics["api_performance"][api]:
                    success_rate = f"{self.metrics['api_performance'][api]['success_rate']:.2%}"
            
            html += f"""
                <tr>
                    <td>{api.upper()}</td>
                    <td>{throughput}</td>
                    <td>{success_rate}</td>
                </tr>
            """
        
        html += """
                    </tbody>
                </table>
                <div class="chart-container">
                    <canvas id="throughputChart"></canvas>
                </div>
            </div>
        </div>
        """
        
        # Cost Tracking
        html += """
        <h2>Cost Tracking</h2>
        <div class="dashboard-grid">
            <div class="card">
                <h3>API Cost Summary</h3>
                <table>
                    <thead>
                        <tr>
                            <th>API</th>
                            <th>Total Cost</th>
                            <th>Cost per Request</th>
                        </tr>
                    </thead>
                    <tbody>
        """
        
        # Add cost data
        for api in self.api_status:
            total_cost = "N/A"
            cost_per_request = "N/A"
            
            # Check if we have cost data
            if api in self.metrics["cost_tracking"]:
                if "total_cost" in self.metrics["cost_tracking"][api]:
                    total_cost = f"${self.metrics['cost_tracking'][api]['total_cost']:.4f}"
                if "avg_cost_per_request" in self.metrics["cost_tracking"][api]:
                    cost_per_request = f"${self.metrics['cost_tracking'][api]['avg_cost_per_request']:.4f}"
            
            html += f"""
                <tr>
                    <td>{api.upper()}</td>
                    <td>{total_cost}</td>
                    <td>{cost_per_request}</td>
                </tr>
            """
        
        html += """
                    </tbody>
                </table>
                <div class="chart-container">
                    <canvas id="costChart"></canvas>
                </div>
            </div>
            
            <div class="card">
                <h3>Recent Tests</h3>
                <table>
                    <thead>
                        <tr>
                            <th>Time</th>
                            <th>API</th>
                            <th>Test Type</th>
                            <th>Status</th>
                        </tr>
                    </thead>
                    <tbody>
        """
        
        # Add recent test data
        for test in self.metrics["test_history"][:10]:  # Show most recent 10
            timestamp = datetime.fromtimestamp(test["timestamp"]).strftime('%Y-%m-%d %H:%M:%S')
            api_type = test.get("api_type", "Unknown")
            test_type = test.get("test_type", "Unknown").capitalize()
            status = test.get("status", "Unknown")
            
            status_class = "text-success" if status == "success" else "text-danger"
            
            html += f"""
                <tr>
                    <td>{timestamp}</td>
                    <td>{api_type.upper()}</td>
                    <td>{test_type}</td>
                    <td class="{status_class}">{status.capitalize()}</td>
                </tr>
            """
        
        html += """
                    </tbody>
                </table>
            </div>
        </div>
        """
        
        # Anomaly Detection Section
        html += """
        <h2>Anomaly Detection</h2>
        <div class="dashboard-grid">
            <div class="card">
                <h3>Anomaly Summary</h3>
        """
        
        # Add anomaly summary
        anomaly_summary = self.get_anomaly_summary()
        total_anomalies = anomaly_summary.get("total_anomalies", 0)
        
        # Display anomaly count with appropriate color
        severity_class = "text-success"
        if total_anomalies > 10:
            severity_class = "text-danger"
        elif total_anomalies > 5:
            severity_class = "text-warning"
        elif total_anomalies > 0:
            severity_class = "text-info"
        
        html += f"""
                <div class="metric-value {severity_class}">{total_anomalies}</div>
                <div class="metric-label">Anomalies detected in the last 24 hours</div>
                
                <div class="status-grid" style="margin-top: 20px;">
        """
        
        # Add severity counts
        severity_counts = anomaly_summary.get("severity_counts", {})
        for severity, count in severity_counts.items():
            severity_class = ""
            if severity == "critical":
                severity_class = "text-danger"
            elif severity == "high":
                severity_class = "text-danger"
            elif severity == "medium":
                severity_class = "text-warning"
            elif severity == "low":
                severity_class = "text-info"
            
            html += f"""
                    <div class="status-item">
                        <strong class="{severity_class}">{severity.upper()}</strong>: {count}
                    </div>
            """
        
        html += """
                </div>
                
                <h4 style="margin-top: 20px;">By API</h4>
                <div class="status-grid">
        """
        
        # Add API counts
        api_counts = anomaly_summary.get("api_counts", {})
        for api, count in api_counts.items():
            html += f"""
                    <div class="status-item">
                        <strong>{api.upper()}</strong>: {count}
                    </div>
            """
        
        html += """
                </div>
                
                <h4 style="margin-top: 20px;">By Metric</h4>
                <div class="status-grid">
        """
        
        # Add metric counts
        metric_counts = anomaly_summary.get("metric_counts", {})
        for metric, count in metric_counts.items():
            html += f"""
                    <div class="status-item">
                        <strong>{metric.capitalize()}</strong>: {count}
                    </div>
            """
        
        html += """
                </div>
            </div>
            
            <div class="card">
                <h3>Recent Anomalies</h3>
                <div class="anomaly-alerts">
        """
        
        # Get recent anomalies (last 24 hours, medium severity or higher)
        recent_anomalies = self.get_recent_anomalies(hours=24, min_severity="medium")
        
        if not recent_anomalies:
            html += """
                    <p>No anomalies detected in the last 24 hours with medium or higher severity.</p>
            """
        else:
            # Show top 5 anomalies
            for anomaly in recent_anomalies[:5]:
                if ANOMALY_DETECTION_AVAILABLE:
                    # Use the HTML generator from the anomaly detection module
                    anomaly_html = generate_anomaly_alert_html(anomaly)
                    html += anomaly_html
                else:
                    # Simple fallback if module not available
                    timestamp = datetime.fromtimestamp(anomaly["timestamp"]).strftime('%Y-%m-%d %H:%M:%S')
                    severity = anomaly.get("severity", "unknown")
                    api = anomaly.get("api", "unknown")
                    metric = anomaly.get("metric_type", "unknown")
                    description = anomaly.get("description", "Anomaly detected")
                    
                    severity_class = "text-info"
                    if severity == "critical" or severity == "high":
                        severity_class = "text-danger"
                    elif severity == "medium":
                        severity_class = "text-warning"
                    
                    html += f"""
                        <div class="anomaly-alert">
                            <div class="anomaly-header">
                                <span class="anomaly-severity {severity_class}">{severity.upper()}</span>
                                <span class="anomaly-timestamp">{timestamp}</span>
                            </div>
                            <div class="anomaly-title">{description}</div>
                            <div class="anomaly-details">
                                <div><strong>API:</strong> {api.upper()}</div>
                                <div><strong>Metric:</strong> {metric}</div>
                            </div>
                        </div>
                    """
        
        html += """
                </div>
            </div>
        </div>
        """
        
        # Add JavaScript for charts
        html += self._generate_chart_javascript()
        
        # Add footer
        html += HTML_FOOTER.format(timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        
        return html
    
    def _generate_chart_javascript(self) -> str:
        """Generate JavaScript for charts."""
        js = """
        <script>
            document.addEventListener('DOMContentLoaded', function() {
                // Create charts once DOM is loaded
                createLatencyChart();
                createThroughputChart();
                createCostChart();
            });
            
            function createLatencyChart() {
                const ctx = document.getElementById('latencyChart').getContext('2d');
                
                // Prepare data
                const datasets = [];
        """
        
        # Add latency data
        for api, data in self.historical_data["latency"].items():
            if not data:
                continue
            
            # Get color based on API
            color = "rgba(54, 162, 235, 1)" if api == "openai" else "rgba(255, 99, 132, 1)" if api == "claude" else "rgba(75, 192, 192, 1)"
            
            # Format data
            sorted_data = sorted(data, key=lambda x: x["timestamp"])
            labels = [datetime.fromtimestamp(point["timestamp"]).strftime('%m-%d %H:%M') for point in sorted_data]
            values = [point["avg_latency"] for point in sorted_data]
            
            js += f"""
                datasets.push({{
                    label: '{api.upper()} Latency',
                    data: {json.dumps(values)},
                    borderColor: '{color}',
                    backgroundColor: '{color.replace("1)", "0.2)")}',
                    tension: 0.4,
                    pointRadius: 2
                }});
            """
        
        js += """
                const latencyChart = new Chart(ctx, {
                    type: 'line',
                    data: {
                        labels: """ + json.dumps(labels if 'labels' in locals() else []) + """,
                        datasets: datasets
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {
                            y: {
                                title: {
                                    display: true,
                                    text: 'Latency (seconds)'
                                },
                                beginAtZero: true
                            },
                            x: {
                                title: {
                                    display: true,
                                    text: 'Time'
                                }
                            }
                        }
                    }
                });
            }
            
            function createThroughputChart() {
                const ctx = document.getElementById('throughputChart').getContext('2d');
                
                // Prepare data
                const datasets = [];
        """
        
        # Add throughput data
        for api, data in self.historical_data["throughput"].items():
            if not data:
                continue
            
            # Get color based on API
            color = "rgba(54, 162, 235, 1)" if api == "openai" else "rgba(255, 99, 132, 1)" if api == "claude" else "rgba(75, 192, 192, 1)"
            
            # Format data
            sorted_data = sorted(data, key=lambda x: x["timestamp"])
            labels = [datetime.fromtimestamp(point["timestamp"]).strftime('%m-%d %H:%M') for point in sorted_data]
            values = [point["requests_per_second"] for point in sorted_data]
            
            js += f"""
                datasets.push({{
                    label: '{api.upper()} Throughput',
                    data: {json.dumps(values)},
                    borderColor: '{color}',
                    backgroundColor: '{color.replace("1)", "0.2)")}',
                    tension: 0.4,
                    pointRadius: 2
                }});
            """
        
        js += """
                const throughputChart = new Chart(ctx, {
                    type: 'line',
                    data: {
                        labels: """ + json.dumps(labels if 'labels' in locals() else []) + """,
                        datasets: datasets
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {
                            y: {
                                title: {
                                    display: true,
                                    text: 'Requests per Second'
                                },
                                beginAtZero: true
                            },
                            x: {
                                title: {
                                    display: true,
                                    text: 'Time'
                                }
                            }
                        }
                    }
                });
            }
            
            function createCostChart() {
                const ctx = document.getElementById('costChart').getContext('2d');
                
                // Prepare data
                const datasets = [];
        """
        
        # Add cost data
        for api, data in self.historical_data["cost"].items():
            if not data:
                continue
            
            # Get color based on API
            color = "rgba(54, 162, 235, 1)" if api == "openai" else "rgba(255, 99, 132, 1)" if api == "claude" else "rgba(75, 192, 192, 1)"
            
            # Format data
            sorted_data = sorted(data, key=lambda x: x["timestamp"])
            labels = [datetime.fromtimestamp(point["timestamp"]).strftime('%m-%d %H:%M') for point in sorted_data]
            
            # Use cost_per_request if available, otherwise use 0
            values = [point.get("cost_per_request", 0) for point in sorted_data]
            
            js += f"""
                datasets.push({{
                    label: '{api.upper()} Cost per Request',
                    data: {json.dumps(values)},
                    borderColor: '{color}',
                    backgroundColor: '{color.replace("1)", "0.2)")}',
                    tension: 0.4,
                    pointRadius: 2
                }});
            """
        
        js += """
                const costChart = new Chart(ctx, {
                    type: 'line',
                    data: {
                        labels: """ + json.dumps(labels if 'labels' in locals() else []) + """,
                        datasets: datasets
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {
                            y: {
                                title: {
                                    display: true,
                                    text: 'Cost per Request ($)'
                                },
                                beginAtZero: true
                            },
                            x: {
                                title: {
                                    display: true,
                                    text: 'Time'
                                }
                            }
                        }
                    }
                });
            }
        </script>
        """
        
        return js


class DashboardHandler(http.server.BaseHTTPRequestHandler):
    """HTTP request handler for the dashboard."""
    
    def __init__(self, *args, dashboard_manager=None, **kwargs):
        self.dashboard_manager = dashboard_manager
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        """Handle GET requests."""
        if self.path == '/' or self.path == '/index.html':
            self.send_response(HTTPStatus.OK)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            
            # Generate dashboard HTML
            html = self.dashboard_manager.generate_dashboard_html()
            self.wfile.write(html.encode('utf-8'))
            
        elif self.path == '/dashboard_data.json':
            self.send_response(HTTPStatus.OK)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            
            # Get dashboard data
            dashboard_data = {
                "timestamp": time.time(),
                "api_status": self.dashboard_manager.api_status,
                "metrics": self.dashboard_manager.metrics,
                "historical_data": self.dashboard_manager.historical_data
            }
            
            self.wfile.write(json.dumps(dashboard_data).encode('utf-8'))
            
        elif self.path == '/predictions.json':
            self.send_response(HTTPStatus.OK)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            
            # Get all predictions
            predictions_data = {
                "timestamp": time.time(),
                "predictions": self.dashboard_manager.get_all_predictions()
            }
            
            self.wfile.write(json.dumps(predictions_data).encode('utf-8'))
            
        elif self.path.startswith('/api/predictions/'):
            # Parse API and metric from path
            parts = self.path.split('/')
            if len(parts) >= 5:
                api = parts[3]
                metric_type = parts[4]
                
                self.send_response(HTTPStatus.OK)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                
                # Get predictions for specific API and metric
                predictions = self.dashboard_manager.get_predictions(api, metric_type)
                self.wfile.write(json.dumps(predictions).encode('utf-8'))
            else:
                self.send_response(HTTPStatus.BAD_REQUEST)
                self.end_headers()
                self.wfile.write(b'Invalid API or metric type')
                
        elif self.path.startswith('/api/anomalies/'):
            # Parse API and metric from path
            parts = self.path.split('/')
            if len(parts) >= 5:
                api = parts[3]
                metric_type = parts[4]
                
                self.send_response(HTTPStatus.OK)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                
                # Get anomalies for specific API and metric
                anomalies = self.dashboard_manager.get_predicted_anomalies(api, metric_type)
                self.wfile.write(json.dumps(anomalies).encode('utf-8'))
            else:
                self.send_response(HTTPStatus.BAD_REQUEST)
                self.end_headers()
                self.wfile.write(b'Invalid API or metric type')
                
        elif self.path.startswith('/api/patterns/'):
            # Parse API and metric from path
            parts = self.path.split('/')
            if len(parts) >= 5:
                api = parts[3]
                metric_type = parts[4]
                
                self.send_response(HTTPStatus.OK)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                
                # Get pattern analysis for specific API and metric
                pattern = self.dashboard_manager.get_pattern_analysis(api, metric_type)
                self.wfile.write(json.dumps(pattern).encode('utf-8'))
            else:
                self.send_response(HTTPStatus.BAD_REQUEST)
                self.end_headers()
                self.wfile.write(b'Invalid API or metric type')
        
        elif self.path.startswith('/api/recommendations/'):
            # Parse API from path
            parts = self.path.split('/')
            if len(parts) >= 4:
                api = parts[3]
                
                self.send_response(HTTPStatus.OK)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                
                # Get recommendations for specific API
                recommendations = self.dashboard_manager.get_optimization_recommendations(api)
                self.wfile.write(json.dumps(recommendations).encode('utf-8'))
            else:
                self.send_response(HTTPStatus.BAD_REQUEST)
                self.end_headers()
                self.wfile.write(b'Invalid API')
        
        else:
            self.send_response(HTTPStatus.NOT_FOUND)
            self.end_headers()
            self.wfile.write(b'Not Found')


def run_dashboard_server(
    port: int, 
    data_dir: str, 
    update_interval: int, 
    anomaly_detection: bool = True,
    enable_notifications: bool = False,
    anomaly_sensitivity: float = 1.0,
    algorithm: str = "ensemble",
    predictive_analytics: bool = True
):
    """
    Run the dashboard server.
    
    Args:
        port: Port to run the server on
        data_dir: Directory to store dashboard data
        update_interval: How often to update metrics (seconds)
        anomaly_detection: Whether to enable anomaly detection
        enable_notifications: Whether to enable anomaly notifications
        anomaly_sensitivity: Sensitivity for anomaly detection
        algorithm: Anomaly detection algorithm to use
        predictive_analytics: Whether to enable predictive analytics
    """
    # Convert algorithm string to enum if anomaly detection is available
    detection_algorithm = None
    if ANOMALY_DETECTION_AVAILABLE:
        try:
            detection_algorithm = AnomalyDetectionAlgorithm(algorithm)
            logger.info(f"Using anomaly detection algorithm: {algorithm}")
        except ValueError:
            logger.warning(f"Invalid algorithm: {algorithm}, using ensemble")
            detection_algorithm = AnomalyDetectionAlgorithm.ENSEMBLE
    
    # Initialize notification manager if enabled
    notification_manager = None
    if enable_notifications:
        try:
            from api_notification_manager import NotificationManager
            notification_manager = NotificationManager()
            notification_manager.start()
            logger.info("Initialized notification manager")
        except ImportError:
            logger.warning("Could not import NotificationManager, notifications disabled")
            enable_notifications = False
    
    # Initialize dashboard manager
    dashboard_manager = APIDashboardManager(
        data_dir=data_dir,
        update_interval=update_interval,
        anomaly_detection_enabled=anomaly_detection,
        predictive_analytics_enabled=predictive_analytics
    )
    
    # Configure anomaly detector if available
    if anomaly_detection and ANOMALY_DETECTION_AVAILABLE and dashboard_manager.anomaly_detector:
        dashboard_manager.anomaly_detector.sensitivity = anomaly_sensitivity
        if detection_algorithm:
            dashboard_manager.anomaly_detector.algorithm = detection_algorithm
        
        # Configure notifications
        dashboard_manager.anomaly_detector.notification_enabled = enable_notifications
        dashboard_manager.anomaly_detector.notification_manager = notification_manager
    
    # Load existing data if available
    dashboard_manager.load_dashboard_data()
    
    # Start the dashboard manager
    dashboard_manager.start()
    
    try:
        # Create a custom handler class that includes the dashboard manager
        handler = lambda *args, **kwargs: DashboardHandler(*args, dashboard_manager=dashboard_manager, **kwargs)
        
        # Create server
        server = socketserver.ThreadingTCPServer(('', port), handler)
        server.daemon_threads = True  # Don't hang on exit
        
        # Print URL
        print(f"Dashboard server started at http://localhost:{port}")
        print("Press Ctrl+C to stop")
        
        # Open browser
        try:
            webbrowser.open(f"http://localhost:{port}")
        except:
            pass
        
        # Start server
        server.serve_forever()
        
    except KeyboardInterrupt:
        print("\nShutting down server...")
    except socket.error as e:
        if e.errno == 98:  # Address already in use
            print(f"Error: Port {port} is already in use. Try a different port.")
        else:
            print(f"Socket error: {e}")
    finally:
        # Stop the dashboard manager
        dashboard_manager.stop()
        
        # Close server if it exists
        if 'server' in locals():
            server.server_close()


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="API Monitoring Dashboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument("--port", type=int, default=8080, help="Port to run dashboard on")
    parser.add_argument("--data-dir", type=str, default="api_dashboard_data", help="Directory to store dashboard data")
    parser.add_argument("--update-interval", type=int, default=30, help="How often to update metrics (seconds)")
    parser.add_argument("--disable-anomaly-detection", action="store_true", help="Disable automatic anomaly detection")
    parser.add_argument("--enable-notifications", action="store_true", help="Enable anomaly notifications")
    parser.add_argument("--anomaly-sensitivity", type=float, default=1.0, help="Sensitivity for anomaly detection (higher values = more sensitive)")
    parser.add_argument("--algorithm", type=str, default="ensemble", choices=["zscore", "moving_average", "iqr", "pattern_detection", "seasonality", "ensemble"], 
                        help="Anomaly detection algorithm to use")
    parser.add_argument("--disable-predictive-analytics", action="store_true", help="Disable predictive analytics features")
    
    args = parser.parse_args()
    
    # Run dashboard server
    run_dashboard_server(
        port=args.port, 
        data_dir=args.data_dir, 
        update_interval=args.update_interval,
        anomaly_detection=not args.disable_anomaly_detection,
        enable_notifications=args.enable_notifications,
        anomaly_sensitivity=args.anomaly_sensitivity,
        algorithm=args.algorithm,
        predictive_analytics=not args.disable_predictive_analytics
    )


if __name__ == "__main__":
    main()