#!/usr/bin/env python3
"""
Demo script for monitoring dashboard integration.

This script demonstrates how to use the monitoring dashboard integration
in the Simulation Accuracy and Validation Framework without requiring
external dependencies.

Usage:
    python demo_monitoring_dashboard.py --mode=panel
    python demo_monitoring_dashboard.py --mode=dashboard
    python demo_monitoring_dashboard.py --mode=monitor
"""

import os
import sys
import argparse
import logging
import json
import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("demo_monitoring_dashboard")

# Import the simplified ValidationVisualizerDBConnector from our test script
sys.path.append(os.path.dirname(__file__))
try:
    from test_basic_dashboard_integration import ValidationVisualizerDBConnector
    logger.info("Imported ValidationVisualizerDBConnector from test_basic_dashboard_integration")
except ImportError:
    logger.error("Could not import ValidationVisualizerDBConnector. Make sure test_basic_dashboard_integration.py exists.")
    sys.exit(1)

def create_panel_demo():
    """Demonstrate creating a dashboard panel."""
    logger.info("Running dashboard panel creation demo")
    
    # Create connector with dashboard integration enabled
    connector = ValidationVisualizerDBConnector(
        dashboard_integration=True,
        dashboard_url="http://localhost:8080/dashboard",
        dashboard_api_key="demo_api_key"
    )
    
    # Create a MAPE comparison panel
    result = connector.create_dashboard_panel_from_db(
        panel_type="mape_comparison",
        hardware_type="gpu_rtx3080",
        model_type="bert-base-uncased",
        metric="throughput_items_per_second",
        dashboard_id="demo_dashboard",
        panel_title="BERT GPU MAPE Comparison"
    )
    
    # Print result
    logger.info("Panel creation result:")
    logger.info(f"  Status: {result['status']}")
    logger.info(f"  Panel ID: {result.get('panel_id')}")
    logger.info(f"  Title: {result.get('title')}")
    
    # Create a hardware heatmap panel
    result = connector.create_dashboard_panel_from_db(
        panel_type="hardware_heatmap",
        model_type="bert-base-uncased",
        metric="average_latency_ms",
        dashboard_id="demo_dashboard",
        panel_title="BERT Hardware Latency Heatmap"
    )
    
    # Print result
    logger.info("Panel creation result:")
    logger.info(f"  Status: {result['status']}")
    logger.info(f"  Panel ID: {result.get('panel_id')}")
    logger.info(f"  Title: {result.get('title')}")

def create_dashboard_demo():
    """Demonstrate creating a comprehensive dashboard."""
    logger.info("Running comprehensive dashboard creation demo")
    
    # Create connector with dashboard integration enabled
    connector = ValidationVisualizerDBConnector(
        dashboard_integration=True,
        dashboard_url="http://localhost:8080/dashboard",
        dashboard_api_key="demo_api_key"
    )
    
    # Create a comprehensive dashboard for BERT on GPU
    result = connector.create_comprehensive_monitoring_dashboard(
        hardware_type="gpu_rtx3080",
        model_type="bert-base-uncased",
        dashboard_title="BERT GPU Monitoring Dashboard",
        include_panels=[
            "mape_comparison",
            "time_series",
            "drift_detection",
            "calibration_effectiveness"
        ]
    )
    
    # Print result
    logger.info("Comprehensive dashboard creation result:")
    logger.info(f"  Status: {result['status']}")
    logger.info(f"  Dashboard ID: {result.get('dashboard_id')}")
    logger.info(f"  Title: {result.get('dashboard_title')}")
    logger.info(f"  Panel Count: {result.get('panel_count')}")
    logger.info(f"  URL: {result.get('url')}")
    
    # Create a dashboard for ViT models across hardware types
    result = connector.create_comprehensive_monitoring_dashboard(
        model_type="vit-base-patch16-224",
        dashboard_title="ViT Model Monitoring Dashboard",
        include_panels=[
            "hardware_heatmap",
            "mape_comparison",
            "simulation_vs_hardware"
        ]
    )
    
    # Print result
    logger.info("Comprehensive dashboard creation result:")
    logger.info(f"  Status: {result['status']}")
    logger.info(f"  Dashboard ID: {result.get('dashboard_id')}")
    logger.info(f"  Title: {result.get('dashboard_title')}")
    logger.info(f"  Panel Count: {result.get('panel_count')}")
    logger.info(f"  URL: {result.get('url')}")

def create_monitoring_demo():
    """Demonstrate setting up real-time monitoring."""
    logger.info("Running real-time monitoring demo")
    
    # Create connector with dashboard integration enabled
    connector = ValidationVisualizerDBConnector(
        dashboard_integration=True,
        dashboard_url="http://localhost:8080/dashboard",
        dashboard_api_key="demo_api_key"
    )
    
    # Create a dashboard first
    dashboard_result = connector.create_comprehensive_monitoring_dashboard(
        hardware_type="gpu_rtx3080",
        model_type="bert-base-uncased",
        dashboard_title="BERT GPU Monitoring Dashboard"
    )
    
    # Set up real-time monitoring
    monitor_result = connector.set_up_real_time_monitoring(
        hardware_type="gpu_rtx3080",
        model_type="bert-base-uncased",
        metrics=["throughput_mape", "latency_mape", "memory_mape"],
        monitoring_interval=300,  # 5 minutes
        alert_thresholds={
            "throughput_mape": 10.0,  # Alert if MAPE exceeds 10%
            "latency_mape": 12.0,
            "memory_mape": 15.0
        },
        dashboard_id=dashboard_result.get("dashboard_id")
    )
    
    # Print result
    logger.info("Real-time monitoring setup result:")
    logger.info(f"  Status: {monitor_result['status']}")
    logger.info(f"  Monitoring Job ID: {monitor_result.get('monitoring_job_id')}")
    logger.info(f"  Metrics: {monitor_result.get('monitoring_config', {}).get('metrics')}")
    logger.info(f"  Alert Thresholds: {monitor_result.get('monitoring_config', {}).get('alert_thresholds')}")
    logger.info(f"  Created Panels: {monitor_result.get('created_panels')}")
    logger.info(f"  Next Check: {monitor_result.get('next_check')}")

def main():
    """Main function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Demo for monitoring dashboard integration")
    parser.add_argument("--mode", choices=["panel", "dashboard", "monitor", "all"],
                      default="all", help="Demo mode")
    
    args = parser.parse_args()
    
    # Run the selected demo
    if args.mode == "panel" or args.mode == "all":
        create_panel_demo()
        print()  # Empty line for separation
    
    if args.mode == "dashboard" or args.mode == "all":
        create_dashboard_demo()
        print()  # Empty line for separation
    
    if args.mode == "monitor" or args.mode == "all":
        create_monitoring_demo()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())