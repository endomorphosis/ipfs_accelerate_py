#!/usr/bin/env python3
"""
Setup script for the Distributed Testing Framework.

This script installs all required dependencies and
creates necessary directories for the framework to run.
"""

import os
import sys
import argparse
import subprocess
import json
from pathlib import Path


def create_directories(base_dir="./distributed_testing_data", force=False):
    """Create directory structure for the framework."""
    dirs = [
        "data",
        "models",
        "models/anomaly_detection",
        "logs",
        "dashboards",
        "configs",
    ]
    
    # Create base directory
    base = Path(base_dir)
    if base.exists() and not force:
        print(f"Base directory {base} already exists. Use --force to overwrite.")
        return False
        
    base.mkdir(exist_ok=True)
    print(f"Created base directory: {base}")
    
    # Create subdirectories
    for dir_name in dirs:
        dir_path = base / dir_name
        dir_path.mkdir(exist_ok=True)
        print(f"Created directory: {dir_path}")
        
    return True


def create_config_file(base_dir="./distributed_testing_data", prometheus_port=8000):
    """Create default configuration file."""
    config = {
        "scheduler": {
            "algorithm": "adaptive",
            "fairness_window": 100,
            "resource_match_weight": 0.7,
            "user_fair_share_enabled": True,
            "adaptive_interval": 50,
            "preemption_enabled": True,
            "max_task_retries": 3,
        },
        "monitoring": {
            "prometheus_port": prometheus_port,
            "prometheus_endpoint": "/metrics",
            "metrics_collection_interval": 30,
            "anomaly_detection_interval": 300,
            "dashboard_update_interval": 3600,
        },
        "ml": {
            "algorithms": ["isolation_forest", "dbscan", "threshold", "mad"],
            "forecasting": ["arima", "prophet", "exponential_smoothing"],
            "visualization": True,
            "model_persistence_dir": "models/anomaly_detection",
            "confidence_threshold": 0.85,
        },
        "metrics_interval": 30,
        "scheduling_interval": 5,
    }
    
    config_path = Path(base_dir) / "configs" / "default_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
        
    print(f"Created default configuration file: {config_path}")
    return config_path


def install_dependencies(prometheus=True, grafana=True, ml=True):
    """Install required dependencies."""
    # Base dependencies
    base_deps = [
        "numpy",
        "pandas",
        "requests",
        "pytest",
    ]
    
    # ML dependencies
    ml_deps = [
        "scikit-learn",
        "statsmodels",
        "prophet",
        "matplotlib",
        "seaborn",
        "joblib",
    ]
    
    # Prometheus dependencies
    prometheus_deps = [
        "prometheus-client",
    ]
    
    # Grafana dependencies
    grafana_deps = [
        "python-dotenv",
        "pygrafana",
    ]
    
    # Combine dependencies
    deps = base_deps
    if ml:
        deps.extend(ml_deps)
    if prometheus:
        deps.extend(prometheus_deps)
    if grafana:
        deps.extend(grafana_deps)
        
    # Install dependencies
    print("Installing Python dependencies...")
    try:
        cmd = [sys.executable, "-m", "pip", "install", "--upgrade"] + deps
        subprocess.run(cmd, check=True)
        print("Successfully installed Python dependencies")
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        return False
        
    return True


def check_prometheus_installation():
    """Check if Prometheus is installed and provide installation instructions."""
    try:
        # Check if prometheus is installed (Linux/macOS)
        result = subprocess.run(["which", "prometheus"], 
                              stdout=subprocess.PIPE, 
                              stderr=subprocess.PIPE)
        
        if result.returncode == 0:
            print("Prometheus is installed at:", result.stdout.decode().strip())
            return True
    except FileNotFoundError:
        # Command not found, which means we're on Windows or the command doesn't exist
        pass
    
    # Not installed, provide installation instructions
    print("\nPrometheus not found on system PATH. Install instructions:")
    print("- Linux: https://prometheus.io/docs/prometheus/latest/installation/")
    print("- macOS: brew install prometheus")
    print("- Windows: Download from https://prometheus.io/download/")
    
    return False


def check_grafana_installation():
    """Check if Grafana is installed and provide installation instructions."""
    try:
        # Check if grafana is installed (Linux/macOS)
        result = subprocess.run(["which", "grafana-server"], 
                              stdout=subprocess.PIPE, 
                              stderr=subprocess.PIPE)
        
        if result.returncode == 0:
            print("Grafana is installed at:", result.stdout.decode().strip())
            return True
    except FileNotFoundError:
        # Command not found, which means we're on Windows or the command doesn't exist
        pass
    
    # Not installed, provide installation instructions
    print("\nGrafana not found on system PATH. Install instructions:")
    print("- Linux: https://grafana.com/docs/grafana/latest/installation/debian/")
    print("- macOS: brew install grafana")
    print("- Windows: Download from https://grafana.com/grafana/download")
    
    return False


def create_init_file():
    """Create __init__.py file for the distributed_testing package."""
    init_path = Path("./distributed_testing/__init__.py")
    
    # Create directory if it doesn't exist
    init_path.parent.mkdir(exist_ok=True)
    
    with open(init_path, 'w') as f:
        f.write('''"""
Distributed Testing Framework.

This package provides a complete framework for distributed test execution
with advanced scheduling, monitoring, and anomaly detection.
"""

from distributed_testing.integration import DistributedTestingFramework, create_distributed_testing_framework
from distributed_testing.advanced_scheduling import AdvancedScheduler, Task, Worker
from distributed_testing.ml_anomaly_detection import MLAnomalyDetection
from distributed_testing.prometheus_grafana_integration import PrometheusGrafanaIntegration

__all__ = [
    'DistributedTestingFramework',
    'create_distributed_testing_framework',
    'AdvancedScheduler',
    'Task',
    'Worker',
    'MLAnomalyDetection',
    'PrometheusGrafanaIntegration',
]
''')
    
    print(f"Created __init__.py file at {init_path}")
    return init_path


def main():
    parser = argparse.ArgumentParser(description="Setup Distributed Testing Framework")
    parser.add_argument("--data-dir", default="./distributed_testing_data", 
                      help="Base directory for framework data")
    parser.add_argument("--force", action="store_true", 
                      help="Force creation of directories even if they exist")
    parser.add_argument("--skip-dependencies", action="store_true",
                      help="Skip installing Python dependencies")
    parser.add_argument("--prometheus-port", type=int, default=8000, 
                      help="Port for Prometheus metrics")
    
    args = parser.parse_args()
    
    # Create directory structure
    if not create_directories(args.data_dir, args.force):
        return 1
        
    # Create configuration file
    config_path = create_config_file(args.data_dir, args.prometheus_port)
    
    # Create __init__.py file
    create_init_file()
    
    # Install dependencies
    if not args.skip_dependencies:
        if not install_dependencies():
            return 1
    
    # Check external dependencies
    check_prometheus_installation()
    check_grafana_installation()
    
    print("\nSetup complete!")
    print(f"To start the framework, run:")
    print(f"  python -m distributed_testing.integration --config {config_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())