#!/usr/bin/env python3
"""
Run Monitoring Dashboard for Distributed Testing Framework

This script runs the comprehensive monitoring dashboard for the distributed 
testing framework, providing real-time monitoring of workers, tasks,
and system metrics.

Implementation Date: March 18, 2025 (Originally planned for June 19-26, 2025)

Usage:
    python run_monitoring_dashboard.py [options]

Options:
    --host HOST             Host to bind the server to (default: localhost)
    --port PORT             Port to bind the server to (default: 8080)
    --coordinator URL       URL of the coordinator server
    --db-path PATH          Path to SQLite database file
    --auto-open             Open dashboard in browser automatically
    --debug                 Enable debug logging
    --generate-sample-data  Generate sample data for demonstration
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# Import monitoring dashboard
from monitoring_dashboard import MonitoringDashboard

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point for the dashboard server."""
    parser = argparse.ArgumentParser(description="Monitoring Dashboard for Distributed Testing Framework")
    parser.add_argument("--host", type=str, default="localhost", help="Host to bind the dashboard server")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind the dashboard server")
    parser.add_argument("--coordinator-url", type=str, help="URL of the coordinator server")
    parser.add_argument("--db-path", type=str, help="Path to SQLite database for metrics")
    parser.add_argument("--auto-open", action="store_true", help="Automatically open the dashboard in a browser")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--generate-sample-data", action="store_true", help="Generate sample data for demonstration")
    
    args = parser.parse_args()
    
    # Set debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    # Set default database path if not provided
    if not args.db_path:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.db_path = os.path.join(script_dir, "monitoring_dashboard.db")
        logger.info(f"Using default database path: {args.db_path}")
    
    # Create dashboard
    dashboard = MonitoringDashboard(
        host=args.host,
        port=args.port,
        coordinator_url=args.coordinator_url,
        db_path=args.db_path,
        auto_open=args.auto_open
    )
    
    # Generate sample data if requested
    if args.generate_sample_data:
        generate_sample_data(dashboard.metrics)
    
    try:
        logger.info(f"Starting dashboard at http://{args.host}:{args.port}")
        dashboard.start()
    except KeyboardInterrupt:
        logger.info("Dashboard stopped by user")
        dashboard.stop()


def generate_sample_data(metrics):
    """Generate sample data for demonstration purposes."""
    logger.info("Generating sample data for demonstration")
    
    # Add workers
    for i in range(1, 6):
        worker_id = f"worker-{i}"
        status = "active" if i < 4 else ("inactive" if i == 4 else "error")
        capabilities = ["cpu"]
        if i % 2 == 0:
            capabilities.append("cuda")
        if i % 3 == 0:
            capabilities.append("rocm")
        
        metrics.record_entity(
            entity_id=worker_id,
            entity_type="worker",
            entity_name=f"Worker {i}",
            entity_data={
                "capabilities": capabilities,
                "host": f"worker-host-{i}.example.com",
                "port": 8000 + i,
                "last_seen": "2025-03-16T12:00:00",
                "version": "1.0.0",
                "os": "Linux",
                "resource_usage": {
                    "cpu": 20 + i * 10,
                    "memory": 30 + i * 5,
                    "gpu": 10 + i * 15
                },
                "task_count": i * 2
            },
            status=status
        )
        
        # Record resource usage metrics
        metrics.record_metric(
            metric_name="cpu_utilization",
            metric_value=20 + i * 10,
            entity_id=worker_id,
            entity_type="worker",
            category="resources"
        )
        
        metrics.record_metric(
            metric_name="memory_utilization",
            metric_value=30 + i * 5,
            entity_id=worker_id,
            entity_type="worker",
            category="resources"
        )
        
        metrics.record_metric(
            metric_name="gpu_utilization",
            metric_value=10 + i * 15,
            entity_id=worker_id,
            entity_type="worker",
            category="resources"
        )
    
    # Add tasks
    task_types = ["benchmark", "test", "validation", "conversion"]
    statuses = ["pending", "running", "completed", "failed"]
    
    for i in range(1, 21):
        task_id = f"task-{i}"
        task_type = task_types[i % len(task_types)]
        status = statuses[i % len(statuses)]
        worker_id = f"worker-{(i % 3) + 1}" if status in ["running", "completed"] else None
        
        # Execution time for completed tasks
        execution_time = None
        if status == "completed":
            execution_time = 10 + (i * 5)
        
        metrics.record_task(
            task_id=task_id,
            task_type=task_type,
            task_data={
                "parameters": {
                    "model": f"model-{i % 5 + 1}",
                    "batch_size": i % 8 + 1,
                    "precision": "fp16" if i % 2 == 0 else "int8"
                },
                "description": f"Sample task {i}"
            },
            status=status,
            worker_id=worker_id,
            priority=i % 5 + 1
        )
        
        # Add execution time for completed tasks
        if status == "completed":
            metrics.record_metric(
                metric_name="task_execution_time",
                metric_value=execution_time,
                entity_id=task_id,
                entity_type="task",
                category="performance"
            )
    
    # Add error events
    error_types = ["connection_error", "timeout_error", "resource_error", "task_error"]
    severities = ["info", "warning", "error"]
    
    for i in range(1, 11):
        error_type = error_types[i % len(error_types)]
        severity = severities[i % len(severities)]
        entity_id = f"worker-{i % 3 + 1}" if i % 2 == 0 else f"task-{i}"
        entity_type = "worker" if i % 2 == 0 else "task"
        
        metrics.record_event(
            event_type=error_type,
            event_data={
                "message": f"Sample error {i}: {error_type}",
                "details": f"This is a sample error event for demonstration purposes"
            },
            entity_id=entity_id,
            entity_type=entity_type,
            severity=severity
        )
    
    # Add alerts
    alert_types = ["system_alert", "resource_alert", "task_alert", "security_alert"]
    severities = ["info", "warning", "critical"]
    
    for i in range(1, 6):
        alert_type = alert_types[i % len(alert_types)]
        severity = severities[i % len(severities)]
        entity_id = f"worker-{i % 3 + 1}" if i % 2 == 0 else None
        entity_type = "worker" if i % 2 == 0 else None
        
        metrics.record_alert(
            alert_type=alert_type,
            alert_message=f"Sample alert {i}: {alert_type}",
            severity=severity,
            entity_id=entity_id,
            entity_type=entity_type
        )
    
    logger.info("Sample data generation complete")


if __name__ == "__main__":
    main()