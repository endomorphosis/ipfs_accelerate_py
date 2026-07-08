#!/usr/bin/env python3
"""
Run Test Script for Web Dashboard

This script demonstrates the functionality of the Web Dashboard for the Result Aggregator.
It sets up a test environment, generates sample test results, and starts the dashboard.

Usage:
    python run_test_web_dashboard.py
"""

import anyio
import datetime
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

# Add the parent directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import required modules
try:
    from test.distributed_testing.result_aggregator.service import ResultAggregatorService
    from test.distributed_testing.result_aggregator.web_dashboard import app
except ImportError:
    logger.error("Required modules not found. Make sure you have the Result Aggregator installed.")
    sys.exit(1)

def generate_sample_data(service, num_results=100):
    """Generate sample test results for demonstration."""
    logger.info(f"Generating {num_results} sample test results...")
    
    # Test types
    test_types = ["benchmark", "unit", "integration", "performance", "compatibility"]
    
    # Statuses
    statuses = ["completed", "failed", "running"]
    status_weights = [0.85, 0.1, 0.05]  # 85% completed, 10% failed, 5% running
    
    # Worker IDs
    worker_ids = [f"worker_{i}" for i in range(1, 6)]
    
    # Metrics
    metrics = {
        "throughput": {"min": 80, "max": 150, "unit": "items/s"},
        "latency": {"min": 2, "max": 12, "unit": "ms"},
        "memory_usage": {"min": 500, "max": 2000, "unit": "MB"},
        "execution_time": {"min": 1, "max": 30, "unit": "s"}
    }
    
    # Task templates for adding variety
    task_templates = [
        {
            "type": "benchmark",
            "metrics": ["throughput", "latency", "memory_usage"]
        },
        {
            "type": "unit",
            "metrics": ["execution_time"]
        },
        {
            "type": "integration",
            "metrics": ["execution_time", "memory_usage"]
        },
        {
            "type": "performance",
            "metrics": ["throughput", "latency", "memory_usage", "execution_time"]
        },
        {
            "type": "compatibility",
            "metrics": ["execution_time"]
        }
    ]
    
    # Hardware requirements templates
    hardware_templates = [
        {"hardware": ["cpu"]},
        {"hardware": ["cuda"]},
        {"hardware": ["rocm"]},
        {"hardware": ["webgpu"]},
        {"hardware": ["webnn"]}
    ]
    
    # Generate results with timestamps going back in time
    now = datetime.datetime.now()
    result_ids = []
    
    for i in range(num_results):
        # Generate a timestamp within the last 7 days
        timestamp = now - datetime.timedelta(
            days=random.randint(0, 7),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59),
            seconds=random.randint(0, 59)
        )
        
        # Select a task template
        template = random.choice(task_templates)
        test_type = template["type"]
        
        # Generate a task ID
        task_id = f"{test_type}_{i}_{random.randint(1000, 9999)}"
        
        # Select a worker ID
        worker_id = random.choice(worker_ids)
        
        # Select a status based on weights
        status = random.choices(statuses, weights=status_weights)[0]
        
        # Generate duration
        duration = random.uniform(1, 60)
        
        # Generate metrics
        result_metrics = {}
        for metric_name in template["metrics"]:
            metric_config = metrics[metric_name]
            value = random.uniform(metric_config["min"], metric_config["max"])
            # Sometimes add unit information
            if random.random() < 0.3:  # 30% chance
                result_metrics[metric_name] = {
                    "value": value,
                    "unit": metric_config["unit"]
                }
            else:
                result_metrics[metric_name] = value
        
        # Add execution_time as a standard metric if not already there
        if "execution_time" not in result_metrics:
            result_metrics["execution_time"] = duration
        
        # Generate details
        details = {
            "priority": random.randint(1, 5),
            "requirements": random.choice(hardware_templates),
            "metadata": {
                "batch_size": random.choice([1, 2, 4, 8, 16]),
                "model": random.choice(["bert", "t5", "vit", "whisper", "llama"]),
                "precision": random.choice(["fp32", "fp16", "int8", "int4"])
            }
        }
        
        # Create result object
        result = {
            "task_id": task_id,
            "worker_id": worker_id,
            "timestamp": timestamp.isoformat(),
            "type": test_type,
            "status": status,
            "duration": duration,
            "metrics": result_metrics,
            "details": details
        }
        
        # Store in service
        result_id = service.store_result(result)
        result_ids.append(result_id)
        
        if (i + 1) % 10 == 0:
            logger.info(f"Generated {i + 1}/{num_results} results")
    
    logger.info(f"Generated {num_results} sample test results")
    
    # Generate some anomalies
    logger.info("Generating anomalies...")
    
    # Create a few anomalous results
    for _ in range(5):
        # Select a random template
        template = random.choice(task_templates)
        test_type = template["type"]
        
        # Generate a task ID
        task_id = f"{test_type}_anomaly_{random.randint(1000, 9999)}"
        
        # Select a worker ID
        worker_id = random.choice(worker_ids)
        
        # Generate a recent timestamp
        timestamp = now - datetime.timedelta(
            hours=random.randint(0, 12),
            minutes=random.randint(0, 59),
            seconds=random.randint(0, 59)
        )
        
        # Generate duration (anomalously high)
        duration = random.uniform(100, 200)  # Much higher than normal
        
        # Generate metrics
        result_metrics = {}
        for metric_name in template["metrics"]:
            metric_config = metrics[metric_name]
            # Generate anomalous values
            multiplier = random.choice([0.1, 10])  # Either very low or very high
            value = random.uniform(metric_config["min"], metric_config["max"]) * multiplier
            result_metrics[metric_name] = value
        
        # Add execution_time as a standard metric
        result_metrics["execution_time"] = duration
        
        # Generate details
        details = {
            "priority": random.randint(1, 5),
            "requirements": random.choice(hardware_templates),
            "metadata": {
                "batch_size": random.choice([1, 2, 4, 8, 16]),
                "model": random.choice(["bert", "t5", "vit", "whisper", "llama"]),
                "precision": random.choice(["fp32", "fp16", "int8", "int4"])
            }
        }
        
        # Create result object
        result = {
            "task_id": task_id,
            "worker_id": worker_id,
            "timestamp": timestamp.isoformat(),
            "type": test_type,
            "status": "completed",
            "duration": duration,
            "metrics": result_metrics,
            "details": details
        }
        
        # Store in service
        result_id = service.store_result(result)
        result_ids.append(result_id)
    
    logger.info("Generated anomalies")
    
    return result_ids

async def main():
    """Run the test script."""
    # Create a temporary database
    db_path = "./test_web_dashboard.duckdb"
    
    try:
        # Check if the database already exists
        if os.path.exists(db_path):
            logger.info(f"Using existing database at {db_path}")
        else:
            logger.info(f"Creating new database at {db_path}")
        
        # Initialize the service
        service = ResultAggregatorService(
            db_path=db_path,
            enable_ml=True,
            enable_visualization=True
        )
        
        # Generate sample data
        generate_sample_data(service, num_results=100)
        
        # Run anomaly detection on all results
        logger.info("Running anomaly detection...")
        anomalies = service.detect_anomalies()
        logger.info(f"Detected {len(anomalies)} anomalies")
        
        # Generate a summary report
        logger.info("Generating summary report...")
        report = service.generate_analysis_report(
            report_type="summary",
            format="markdown"
        )
        
        # Save the report
        with open("test_summary_report.md", "w") as f:
            f.write(report)
        
        logger.info("Report saved to test_summary_report.md")
        
        # Start the web dashboard
        logger.info("Starting web dashboard...")
        logger.info("Access the dashboard at http://localhost:8050")
        logger.info("Use the following credentials:")
        logger.info("  Username: admin")
        logger.info("  Password: admin_password")
        logger.info("Press Ctrl+C to stop the server")
        
        # Set environment variables for the dashboard
        os.environ["FLASK_APP"] = "result_aggregator.web_dashboard"
        
        # Run the dashboard with Flask
        from flask import Flask
        
        # Create the app with correct template and static folders
        app.config['TESTING'] = True
        app.config['DEBUG'] = True
        app.run(host='0.0.0.0', port=8050)
        
    except KeyboardInterrupt:
        logger.info("Stopping web dashboard...")
    except Exception as e:
        logger.error(f"Error running test: {e}")
    finally:
        # Close the service
        if 'service' in locals() and service:
            service.close()
            logger.info("Service closed")

if __name__ == "__main__":
    # For simplicity, use the main thread to run the web server
    anyio.run(main)