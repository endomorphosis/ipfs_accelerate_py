#!/usr/bin/env python3
"""
Web Dashboard for Result Aggregator

This module provides a web-based dashboard for visualizing and interacting with the 
Result Aggregator data. It includes REST API endpoints and interactive visualizations.

Usage:
    # Start the web dashboard server
    python web_dashboard.py --port 8050 --db-path ./test_results.duckdb
"""

import argparse
import json
import logging
import os
import threading
import time
from datetime import datetime, timedelta
from functools import wraps
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

# Add parent directory to path so we can import modules
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import the Result Aggregator Service
from result_aggregator.service import ResultAggregatorService
from result_aggregator.visualization import ResultVisualizer

# Flask for web server
try:
    from flask import Flask, request, jsonify, render_template, Response, send_from_directory, session, redirect, url_for
    from flask_cors import CORS
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    print("Flask not available. Install with 'pip install flask flask-cors'")
    sys.exit(1)

# Optional: Flask SocketIO for real-time updates
try:
    from flask_socketio import SocketIO, emit
    SOCKETIO_AVAILABLE = True
except ImportError:
    SOCKETIO_AVAILABLE = False
    print("Flask-SocketIO not available. Real-time updates will be disabled.")
    print("Install with 'pip install flask-socketio'")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("web_dashboard.log")
    ]
)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__, 
            template_folder=os.path.join(os.path.dirname(__file__), 'templates'),
            static_folder=os.path.join(os.path.dirname(__file__), 'static'))
CORS(app)  # Enable CORS for all routes
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'development_key')

# Enable SocketIO if available
if SOCKETIO_AVAILABLE:
    socketio = SocketIO(app, cors_allowed_origins="*")
else:
    socketio = None

# Global service instance
service = None
visualizer = None

# Authentication configuration
USERS = {
    'admin': 'admin_password',
    'user': 'user_password'
}

# In-memory notification storage
notifications = []

# ===== Authentication Helpers =====

def login_required(f):
    """Decorator to require login for a route."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function

# ===== API Routes =====

@app.route('/api/results', methods=['GET'])
def get_results():
    """API endpoint to get test results."""
    start_time = request.args.get('start_time')
    end_time = request.args.get('end_time')
    test_type = request.args.get('test_type')
    status = request.args.get('status')
    worker_id = request.args.get('worker_id')
    limit = int(request.args.get('limit', 100))
    offset = int(request.args.get('offset', 0))
    
    # Build filter criteria
    filter_criteria = {}
    if start_time:
        filter_criteria['start_time'] = start_time
    if end_time:
        filter_criteria['end_time'] = end_time
    if test_type:
        filter_criteria['test_type'] = test_type
    if status:
        filter_criteria['status'] = status
    if worker_id:
        filter_criteria['worker_id'] = worker_id
    
    # Get results from service
    results = service.get_results(filter_criteria=filter_criteria, limit=limit, offset=offset)
    
    return jsonify(results)

@app.route('/api/result/<int:result_id>', methods=['GET'])
def get_result(result_id):
    """API endpoint to get a specific test result."""
    result = service.get_result(result_id)
    
    if not result:
        return jsonify({"error": f"Result with ID {result_id} not found"}), 404
    
    return jsonify(result)

@app.route('/api/aggregated', methods=['GET'])
def get_aggregated_results():
    """API endpoint to get aggregated test results."""
    # Parse query parameters
    aggregation_type = request.args.get('aggregation_type', 'mean')
    test_type = request.args.get('test_type')
    status = request.args.get('status')
    start_time = request.args.get('start_time')
    end_time = request.args.get('end_time')
    
    # Parse group_by parameter
    group_by_str = request.args.get('group_by')
    group_by = group_by_str.split(',') if group_by_str else None
    
    # Parse metrics parameter
    metrics_str = request.args.get('metrics')
    metrics = metrics_str.split(',') if metrics_str else None
    
    # Build filter criteria
    filter_criteria = {}
    if start_time:
        filter_criteria['start_time'] = start_time
    if end_time:
        filter_criteria['end_time'] = end_time
    if test_type:
        filter_criteria['test_type'] = test_type
    if status:
        filter_criteria['status'] = status
    
    # Get aggregated results from service
    results = service.get_aggregated_results(
        filter_criteria=filter_criteria,
        aggregation_type=aggregation_type,
        group_by=group_by,
        metrics=metrics
    )
    
    return jsonify(results)

@app.route('/api/trends', methods=['GET'])
def get_trends():
    """API endpoint to get performance trends."""
    # Parse query parameters
    test_type = request.args.get('test_type')
    start_time = request.args.get('start_time')
    end_time = request.args.get('end_time')
    metrics_str = request.args.get('metrics')
    metrics = metrics_str.split(',') if metrics_str else None
    window_size = int(request.args.get('window_size', 10))
    
    # Build filter criteria
    filter_criteria = {}
    if start_time:
        filter_criteria['start_time'] = start_time
    if end_time:
        filter_criteria['end_time'] = end_time
    if test_type:
        filter_criteria['test_type'] = test_type
    
    # Get performance trends from service
    trends = service.analyze_performance_trends(
        filter_criteria=filter_criteria,
        metrics=metrics,
        window_size=window_size
    )
    
    return jsonify(trends)

@app.route('/api/anomalies', methods=['GET'])
def get_anomalies():
    """API endpoint to get detected anomalies."""
    # Parse query parameters
    test_type = request.args.get('test_type')
    start_time = request.args.get('start_time')
    end_time = request.args.get('end_time')
    
    # Build filter criteria
    filter_criteria = {}
    if start_time:
        filter_criteria['start_time'] = start_time
    if end_time:
        filter_criteria['end_time'] = end_time
    if test_type:
        filter_criteria['test_type'] = test_type
    
    # Get anomalies from service
    anomalies = service.detect_anomalies(filter_criteria=filter_criteria)
    
    return jsonify(anomalies)

@app.route('/api/report', methods=['GET'])
def get_report():
    """API endpoint to generate an analysis report."""
    # Parse query parameters
    report_type = request.args.get('report_type', 'performance')
    format_type = request.args.get('format', 'json')
    test_type = request.args.get('test_type')
    start_time = request.args.get('start_time')
    end_time = request.args.get('end_time')
    
    # Build filter criteria
    filter_criteria = {}
    if start_time:
        filter_criteria['start_time'] = start_time
    if end_time:
        filter_criteria['end_time'] = end_time
    if test_type:
        filter_criteria['test_type'] = test_type
    
    # Generate report
    report = service.generate_analysis_report(
        filter_criteria=filter_criteria,
        report_type=report_type,
        format=format_type
    )
    
    if format_type == 'json':
        return jsonify(json.loads(report))
    elif format_type == 'html':
        return Response(report, mimetype='text/html')
    else:  # markdown or other
        return Response(report, mimetype='text/plain')

@app.route('/api/notifications', methods=['GET'])
def get_notifications():
    """API endpoint to get notifications."""
    # Get the last N notifications
    count = int(request.args.get('count', 10))
    return jsonify(notifications[-count:] if len(notifications) > 0 else [])

@app.route('/api/visualizations/performance', methods=['GET'])
def get_performance_visualization():
    """API endpoint to generate a performance visualization."""
    # Parse query parameters
    metrics_str = request.args.get('metrics')
    metrics = metrics_str.split(',') if metrics_str else None
    test_type = request.args.get('test_type')
    start_time = request.args.get('start_time')
    end_time = request.args.get('end_time')
    interactive = request.args.get('interactive', 'true').lower() == 'true'
    
    # Build filter criteria
    filter_criteria = {}
    if start_time:
        filter_criteria['start_time'] = start_time
    if end_time:
        filter_criteria['end_time'] = end_time
    if test_type:
        filter_criteria['test_type'] = test_type
    
    # Check if visualizer is available
    if not visualizer:
        return jsonify({"error": "Visualization is not available"}), 500
    
    try:
        # Generate temporary file path
        import tempfile
        import uuid
        
        filename = f"performance_{uuid.uuid4().hex}.html" if interactive else f"performance_{uuid.uuid4().hex}.png"
        output_path = os.path.join(tempfile.gettempdir(), filename)
        
        # Generate visualization
        visualizer.generate_performance_chart(
            metrics=metrics,
            filter_criteria=filter_criteria,
            output_path=output_path,
            interactive=interactive
        )
        
        # Return file path
        return jsonify({
            "success": True,
            "path": f"/visualizations/{filename}",
            "file": output_path
        })
    except Exception as e:
        logger.error(f"Error generating performance visualization: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/visualizations/trends', methods=['GET'])
def get_trends_visualization():
    """API endpoint to generate a trend analysis visualization."""
    # Parse query parameters
    metrics_str = request.args.get('metrics')
    metrics = metrics_str.split(',') if metrics_str else None
    test_type = request.args.get('test_type')
    start_time = request.args.get('start_time')
    end_time = request.args.get('end_time')
    interactive = request.args.get('interactive', 'true').lower() == 'true'
    
    # Build filter criteria
    filter_criteria = {}
    if start_time:
        filter_criteria['start_time'] = start_time
    if end_time:
        filter_criteria['end_time'] = end_time
    if test_type:
        filter_criteria['test_type'] = test_type
    
    # Check if visualizer is available
    if not visualizer:
        return jsonify({"error": "Visualization is not available"}), 500
    
    try:
        # Generate temporary file path
        import tempfile
        import uuid
        
        filename = f"trends_{uuid.uuid4().hex}.html" if interactive else f"trends_{uuid.uuid4().hex}.png"
        output_path = os.path.join(tempfile.gettempdir(), filename)
        
        # Generate visualization
        visualizer.generate_trend_analysis(
            metrics=metrics,
            filter_criteria=filter_criteria,
            output_path=output_path,
            interactive=interactive
        )
        
        # Return file path
        return jsonify({
            "success": True,
            "path": f"/visualizations/{filename}",
            "file": output_path
        })
    except Exception as e:
        logger.error(f"Error generating trend visualization: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/visualizations/anomalies', methods=['GET'])
def get_anomalies_visualization():
    """API endpoint to generate an anomaly dashboard."""
    # Parse query parameters
    test_type = request.args.get('test_type')
    start_time = request.args.get('start_time')
    end_time = request.args.get('end_time')
    
    # Build filter criteria
    filter_criteria = {}
    if start_time:
        filter_criteria['start_time'] = start_time
    if end_time:
        filter_criteria['end_time'] = end_time
    if test_type:
        filter_criteria['test_type'] = test_type
    
    # Check if visualizer is available
    if not visualizer:
        return jsonify({"error": "Visualization is not available"}), 500
    
    try:
        # Generate temporary file path
        import tempfile
        import uuid
        
        filename = f"anomalies_{uuid.uuid4().hex}.html"
        output_path = os.path.join(tempfile.gettempdir(), filename)
        
        # Generate visualization
        visualizer.generate_anomaly_dashboard(
            filter_criteria=filter_criteria,
            output_path=output_path
        )
        
        # Return file path
        return jsonify({
            "success": True,
            "path": f"/visualizations/{filename}",
            "file": output_path
        })
    except Exception as e:
        logger.error(f"Error generating anomaly visualization: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/visualizations/summary', methods=['GET'])
def get_summary_visualization():
    """API endpoint to generate a summary dashboard."""
    # Parse query parameters
    test_type = request.args.get('test_type')
    start_time = request.args.get('start_time')
    end_time = request.args.get('end_time')
    
    # Build filter criteria
    filter_criteria = {}
    if start_time:
        filter_criteria['start_time'] = start_time
    if end_time:
        filter_criteria['end_time'] = end_time
    if test_type:
        filter_criteria['test_type'] = test_type
    
    # Check if visualizer is available
    if not visualizer:
        return jsonify({"error": "Visualization is not available"}), 500
    
    try:
        # Generate temporary file path
        import tempfile
        import uuid
        
        filename = f"summary_{uuid.uuid4().hex}.html"
        output_path = os.path.join(tempfile.gettempdir(), filename)
        
        # Generate visualization
        visualizer.generate_summary_dashboard(
            filter_criteria=filter_criteria,
            output_path=output_path
        )
        
        # Return file path
        return jsonify({
            "success": True,
            "path": f"/visualizations/{filename}",
            "file": output_path
        })
    except Exception as e:
        logger.error(f"Error generating summary visualization: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/performance/regression', methods=['GET'])
def get_performance_regression():
    """API endpoint to detect performance regression."""
    # Parse query parameters
    metric_name = request.args.get('metric')
    baseline_period = request.args.get('baseline_period', '7d')
    comparison_period = request.args.get('comparison_period', '1d')
    test_type = request.args.get('test_type')
    
    # Build filter criteria
    filter_criteria = {}
    if test_type:
        filter_criteria['test_type'] = test_type
    
    # Check if service is available
    if not service:
        return jsonify({"error": "Service is not available"}), 500
    
    try:
        # Analyze performance regression
        results = service.analyze_performance_regression(
            metric_name=metric_name,
            baseline_period=baseline_period,
            comparison_period=comparison_period,
            filter_criteria=filter_criteria
        )
        
        return jsonify(results)
    except Exception as e:
        logger.error(f"Error analyzing performance regression: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/performance/hardware', methods=['GET'])
def get_hardware_performance():
    """API endpoint to compare hardware performance."""
    # Parse query parameters
    metrics_str = request.args.get('metrics')
    metrics = metrics_str.split(',') if metrics_str else None
    test_type = request.args.get('test_type')
    time_period = request.args.get('time_period', '30d')
    
    # Check if service is available
    if not service:
        return jsonify({"error": "Service is not available"}), 500
    
    try:
        # Compare hardware performance
        results = service.compare_hardware_performance(
            metrics=metrics,
            test_type=test_type,
            time_period=time_period
        )
        
        return jsonify(results)
    except Exception as e:
        logger.error(f"Error comparing hardware performance: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/performance/efficiency', methods=['GET'])
def get_resource_efficiency():
    """API endpoint to analyze resource efficiency."""
    # Parse query parameters
    test_type = request.args.get('test_type')
    time_period = request.args.get('time_period', '30d')
    
    # Check if service is available
    if not service:
        return jsonify({"error": "Service is not available"}), 500
    
    try:
        # Analyze resource efficiency
        results = service.analyze_resource_efficiency(
            test_type=test_type,
            time_period=time_period
        )
        
        return jsonify(results)
    except Exception as e:
        logger.error(f"Error analyzing resource efficiency: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/performance/time', methods=['GET'])
def get_performance_over_time():
    """API endpoint to analyze performance over time."""
    # Parse query parameters
    metric_name = request.args.get('metric')
    grouping = request.args.get('grouping', 'day')
    test_type = request.args.get('test_type')
    time_period = request.args.get('time_period', '90d')
    
    # Check if service is available
    if not service or not metric_name:
        return jsonify({"error": "Service is not available or metric is not specified"}), 500
    
    try:
        # Analyze performance over time
        results = service.analyze_performance_over_time(
            metric_name=metric_name,
            grouping=grouping,
            test_type=test_type,
            time_period=time_period
        )
        
        return jsonify(results)
    except Exception as e:
        logger.error(f"Error analyzing performance over time: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/performance/report', methods=['GET'])
def get_performance_report():
    """API endpoint to generate a performance report."""
    # Parse query parameters
    report_type = request.args.get('report_type', 'comprehensive')
    format_type = request.args.get('format', 'json')
    test_type = request.args.get('test_type')
    time_period = request.args.get('time_period', '30d')
    
    # Build filter criteria
    filter_criteria = {}
    if test_type:
        filter_criteria['test_type'] = test_type
    
    # Check if service is available
    if not service:
        return jsonify({"error": "Service is not available"}), 500
    
    try:
        # Generate performance report
        report = service.generate_performance_report(
            report_type=report_type,
            filter_criteria=filter_criteria,
            format=format_type,
            time_period=time_period
        )
        
        if format_type == 'json':
            return jsonify(json.loads(report))
        elif format_type == 'html':
            return Response(report, mimetype='text/html')
        else:  # markdown or other
            return Response(report, mimetype='text/plain')
    except Exception as e:
        logger.error(f"Error generating performance report: {e}")
        return jsonify({"error": str(e)}), 500

# ===== Monitoring API Routes =====

@app.route('/api/monitoring/cluster', methods=['GET'])
def get_cluster_status():
    """API endpoint to get cluster status."""
    try:
        # This would typically fetch data from the coordinator through the service
        # For demonstration, we're returning sample data
        import random
        
        active_workers = random.randint(3, 10)
        total_tasks = random.randint(20, 100)
        completed_tasks = random.randint(0, total_tasks - 5)
        failed_tasks = random.randint(0, 5)
        success_rate = int((completed_tasks / (completed_tasks + failed_tasks) * 100)) if completed_tasks + failed_tasks > 0 else 0
        
        # Calculate cluster health based on worker health statuses
        # In a real implementation, this would come from coordinator health data
        health_score = random.randint(70, 100)
        health_status = "healthy" if health_score >= 90 else "warning" if health_score >= 70 else "critical"
        
        # Generate trend data
        def create_trend(is_positive):
            direction = "up" if is_positive else "down"
            value = random.uniform(0, 5) if is_positive else random.uniform(0, 3)
            return {
                "direction": direction,
                "value": round(value, 1),
                "status": "stable" if value < 0.5 else direction
            }
        
        return jsonify({
            "active_workers": active_workers,
            "total_tasks": total_tasks,
            "success_rate": success_rate,
            "health": {
                "score": health_score,
                "status": health_status,
                "trend": create_trend(True)
            },
            "trends": {
                "workers": create_trend(random.choice([True, False])),
                "tasks": create_trend(True),
                "success_rate": create_trend(random.choice([True, False]))
            }
        })
    except Exception as e:
        logger.error(f"Error getting cluster status: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/monitoring/workers', methods=['GET'])
def get_worker_status():
    """API endpoint to get worker status."""
    try:
        # This would typically fetch data from the coordinator through the service
        # For demonstration, we're returning sample data
        import random
        
        # Hardware types
        hardware_types = ['cpu', 'cuda', 'rocm', 'mps', 'openvino', 'qualcomm', 'webnn', 'webgpu']
        health_statuses = ['healthy', 'warning', 'critical', 'unknown']
        
        workers = []
        for i in range(1, random.randint(5, 10)):
            worker_id = f"worker-{i:03d}"
            status = random.choice(['active', 'inactive']) if random.random() > 0.8 else 'active'
            health = random.choices(health_statuses, weights=[0.7, 0.2, 0.05, 0.05])[0]
            cpu_usage = random.randint(5, 95) if status == 'active' else 0
            memory_usage = round(random.uniform(0.5, 8.0), 1) if status == 'active' else 0.0
            tasks_completed = random.randint(0, 100) if status == 'active' else 0
            success_rate = random.randint(70, 100) if status == 'active' else 0
            
            # Random set of hardware
            num_hardware = random.randint(1, 5)
            available_hardware = ['cpu']  # Always include CPU
            available_hardware.extend(random.sample(hardware_types[1:], num_hardware))
            
            workers.append({
                "id": worker_id,
                "status": status,
                "health": health,
                "cpu": cpu_usage,
                "memory": memory_usage,
                "tasks_completed": tasks_completed,
                "success_rate": success_rate,
                "hardware": available_hardware
            })
        
        return jsonify(workers)
    except Exception as e:
        logger.error(f"Error getting worker status: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/monitoring/tasks', methods=['GET'])
def get_task_queue():
    """API endpoint to get task queue."""
    try:
        # Get filter parameter
        status_filter = request.args.get('status', 'all')
        
        # This would typically fetch data from the coordinator through the service
        # For demonstration, we're returning sample data
        import random
        
        task_types = ['benchmark', 'test', 'validation', 'integration']
        task_statuses = ['pending', 'running', 'completed', 'failed']
        
        tasks = []
        for i in range(1, random.randint(10, 25)):
            task_id = f"task-{random.randint(1000, 9999)}"
            task_type = random.choice(task_types)
            status = random.choice(task_statuses)
            priority = random.randint(1, 3)
            worker_id = f"worker-{random.randint(1, 10):03d}" if status in ['running', 'completed'] else None
            
            # Apply filter if needed
            if status_filter != 'all' and status != status_filter:
                continue
                
            tasks.append({
                "id": task_id,
                "type": task_type,
                "status": status,
                "priority": priority,
                "worker_id": worker_id
            })
        
        return jsonify(tasks)
    except Exception as e:
        logger.error(f"Error getting task queue: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/monitoring/resources', methods=['GET'])
def get_resource_usage():
    """API endpoint to get resource usage data."""
    try:
        # This would typically fetch data from the coordinator through the service
        # For demonstration, we're returning sample data
        import random
        from datetime import datetime, timedelta
        
        # Generate time points for last 24 hours
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=24)
        time_points = []
        labels = []
        
        for i in range(24):
            time_point = start_time + timedelta(hours=i)
            time_points.append(time_point)
            labels.append(time_point.strftime("%H:00"))
        
        # Generate CPU data
        cpu_avg = [random.randint(20, 60) for _ in range(24)]
        cpu_max = [random.randint(65, 95) for _ in range(24)]
        
        # Generate memory data
        memory_avg = [round(random.uniform(1, 4), 1) for _ in range(24)]
        memory_max = [round(random.uniform(4, 8), 1) for _ in range(24)]
        
        return jsonify({
            "cpu": {
                "labels": labels,
                "datasets": [
                    {
                        "label": "Average CPU Usage",
                        "data": cpu_avg,
                        "borderColor": "#4c9be8",
                        "backgroundColor": "rgba(76, 155, 232, 0.1)",
                        "fill": True
                    },
                    {
                        "label": "Max CPU Usage",
                        "data": cpu_max,
                        "borderColor": "#e86f4c",
                        "backgroundColor": "rgba(232, 111, 76, 0.1)",
                        "fill": True
                    }
                ]
            },
            "memory": {
                "labels": labels,
                "datasets": [
                    {
                        "label": "Average Memory Usage (GB)",
                        "data": memory_avg,
                        "borderColor": "#4ca6e8",
                        "backgroundColor": "rgba(76, 166, 232, 0.1)",
                        "fill": True
                    },
                    {
                        "label": "Max Memory Usage (GB)",
                        "data": memory_max,
                        "borderColor": "#e84ca6",
                        "backgroundColor": "rgba(232, 76, 166, 0.1)",
                        "fill": True
                    }
                ]
            }
        })
    except Exception as e:
        logger.error(f"Error getting resource usage: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/monitoring/hardware', methods=['GET'])
def get_hardware_availability():
    """API endpoint to get hardware availability data."""
    try:
        # This would typically fetch data from the coordinator through the service
        # For demonstration, we're returning sample data
        import random
        
        hardware_types = ['CPU', 'CUDA', 'ROCm', 'MPS', 'OpenVINO', 'QNN', 'WebNN', 'WebGPU']
        
        # Generate random data for available and total hardware
        available = []
        total = []
        
        for _ in range(len(hardware_types)):
            total_count = random.randint(1, 10)
            available_count = random.randint(0, total_count)
            
            available.append(available_count)
            total.append(total_count)
        
        return jsonify({
            "labels": hardware_types,
            "datasets": [
                {
                    "label": "Available",
                    "data": available,
                    "backgroundColor": "rgba(40, 167, 69, 0.7)"
                },
                {
                    "label": "Total",
                    "data": total,
                    "backgroundColor": "rgba(108, 117, 125, 0.3)"
                }
            ]
        })
    except Exception as e:
        logger.error(f"Error getting hardware availability: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/monitoring/network', methods=['GET'])
def get_network_topology():
    """API endpoint to get network topology data."""
    try:
        # This would typically fetch data from the coordinator through the service
        # For demonstration, we're returning sample data
        import random
        
        # Create nodes for coordinator and workers
        nodes = [
            {"id": "coordinator", "group": "coordinator", "status": "active"}
        ]
        
        # Add worker nodes
        links = []
        num_workers = random.randint(5, 10)
        
        for i in range(1, num_workers + 1):
            worker_id = f"worker-{i:03d}"
            status = "active" if random.random() > 0.2 else "inactive"
            
            nodes.append({
                "id": worker_id,
                "group": "worker",
                "status": status
            })
            
            # Link quality is higher for active workers
            link_quality = random.randint(8, 10) if status == "active" else random.randint(1, 3)
            
            links.append({
                "source": "coordinator",
                "target": worker_id,
                "value": link_quality
            })
        
        return jsonify({
            "nodes": nodes,
            "links": links
        })
    except Exception as e:
        logger.error(f"Error getting network topology: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/visualizations/<path:filename>')
def serve_visualization(filename):
    """Serve a visualization file."""
    import tempfile
    return send_from_directory(tempfile.gettempdir(), filename)

# ===== Web Routes =====

@app.route('/')
@login_required
def index():
    """Render the main dashboard page."""
    return render_template('index.html')

@app.route('/dashboard')
@login_required
def dashboard():
    """Render the dashboard page."""
    return render_template('dashboard.html')

@app.route('/results')
@login_required
def results_page():
    """Render the results page."""
    return render_template('results.html')

@app.route('/trends')
@login_required
def trends_page():
    """Render the trends page."""
    return render_template('trends.html')

@app.route('/anomalies')
@login_required
def anomalies_page():
    """Render the anomalies page."""
    return render_template('anomalies.html')

@app.route('/reports')
@login_required
def reports_page():
    """Render the reports page."""
    return render_template('reports.html')

@app.route('/settings')
@login_required
def settings_page():
    """Render the settings page."""
    return render_template('settings.html')

@app.route('/monitoring')
@login_required
def monitoring_dashboard():
    """Render the real-time monitoring dashboard page."""
    return render_template('monitoring_dashboard.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Handle login requests."""
    error = None
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        if username in USERS and USERS[username] == password:
            session['username'] = username
            return redirect(request.args.get('next') or url_for('index'))
        else:
            error = 'Invalid credentials'
    
    return render_template('login.html', error=error)

@app.route('/logout')
def logout():
    """Handle logout requests."""
    session.pop('username', None)
    return redirect(url_for('login'))

# ===== SocketIO Routes (if available) =====

if SOCKETIO_AVAILABLE:
    @socketio.on('connect')
    def handle_connect():
        """Handle SocketIO connection."""
        logger.info(f"Client connected: {request.sid}")

    @socketio.on('disconnect')
    def handle_disconnect():
        """Handle SocketIO disconnect."""
        logger.info(f"Client disconnected: {request.sid}")
    
    @socketio.on('subscribe_to_monitoring')
    def handle_subscribe_monitoring(data):
        """Handle monitoring subscription."""
        logger.info(f"Client {request.sid} subscribed to monitoring updates")
        # Join a room for monitoring updates
        from flask_socketio import join_room
        join_room('monitoring_subscribers')
        # Send initial data
        emit_monitoring_data()
    
    @socketio.on('request_monitoring_update')
    def handle_request_monitoring_update(data):
        """Handle request for immediate monitoring data update."""
        logger.info(f"Client {request.sid} requested monitoring data update")
        # Send updated data
        emit_monitoring_data()

# ===== Real-time Data Broadcasting =====

def emit_monitoring_data():
    """Emit real-time monitoring data to subscribed clients."""
    if not SOCKETIO_AVAILABLE:
        return

    try:
        # Generate monitoring data (in production, this would fetch real data)
        cluster_data = generate_cluster_data()
        worker_data = generate_worker_data()
        task_data = generate_task_data()
        resource_data = generate_resource_data()
        hardware_data = generate_hardware_data()
        network_data = generate_network_data()
        
        # Emit data to subscribed clients
        socketio.emit('monitoring_update', {
            'cluster': cluster_data,
            'workers': worker_data,
            'tasks': task_data,
            'resources': resource_data,
            'hardware': hardware_data,
            'network': network_data
        }, room='monitoring_subscribers')
        
        logger.debug("Monitoring data emitted via WebSocket")
    except Exception as e:
        logger.error(f"Error emitting monitoring data: {e}")

# Helper functions to generate mock data for demonstration
def generate_cluster_data():
    """Generate mock cluster data."""
    import random
    
    active_workers = random.randint(3, 10)
    total_tasks = random.randint(20, 100)
    completed_tasks = random.randint(0, total_tasks - 5)
    failed_tasks = random.randint(0, 5)
    success_rate = int((completed_tasks / (completed_tasks + failed_tasks) * 100)) if completed_tasks + failed_tasks > 0 else 0
    
    # Calculate cluster health based on worker health statuses
    health_score = random.randint(70, 100)
    health_status = "healthy" if health_score >= 90 else "warning" if health_score >= 70 else "critical"
    
    # Generate trend data
    def create_trend(is_positive):
        direction = "up" if is_positive else "down"
        value = random.uniform(0, 5) if is_positive else random.uniform(0, 3)
        return {
            "direction": direction,
            "value": round(value, 1),
            "status": "stable" if value < 0.5 else direction
        }
    
    return {
        "active_workers": active_workers,
        "total_tasks": total_tasks,
        "success_rate": success_rate,
        "health": {
            "score": health_score,
            "status": health_status,
            "trend": create_trend(True)
        },
        "trends": {
            "workers": create_trend(random.choice([True, False])),
            "tasks": create_trend(True),
            "success_rate": create_trend(random.choice([True, False]))
        }
    }

def generate_worker_data():
    """Generate mock worker data."""
    import random
    
    # Hardware types
    hardware_types = ['cpu', 'cuda', 'rocm', 'mps', 'openvino', 'qualcomm', 'webnn', 'webgpu']
    health_statuses = ['healthy', 'warning', 'critical', 'unknown']
    
    workers = []
    for i in range(1, random.randint(5, 10)):
        worker_id = f"worker-{i:03d}"
        status = random.choice(['active', 'inactive']) if random.random() > 0.8 else 'active'
        health = random.choices(health_statuses, weights=[0.7, 0.2, 0.05, 0.05])[0]
        cpu_usage = random.randint(5, 95) if status == 'active' else 0
        memory_usage = round(random.uniform(0.5, 8.0), 1) if status == 'active' else 0.0
        tasks_completed = random.randint(0, 100) if status == 'active' else 0
        success_rate = random.randint(70, 100) if status == 'active' else 0
        
        # Random set of hardware
        num_hardware = random.randint(1, 5)
        available_hardware = ['cpu']  # Always include CPU
        available_hardware.extend(random.sample(hardware_types[1:], num_hardware))
        
        workers.append({
            "id": worker_id,
            "status": status,
            "health": health,
            "cpu": cpu_usage,
            "memory": memory_usage,
            "tasks_completed": tasks_completed,
            "success_rate": success_rate,
            "hardware": available_hardware
        })
    
    return workers

def generate_task_data():
    """Generate mock task data."""
    import random
    
    task_types = ['benchmark', 'test', 'validation', 'integration']
    task_statuses = ['pending', 'running', 'completed', 'failed']
    
    tasks = []
    for i in range(1, random.randint(10, 25)):
        task_id = f"task-{random.randint(1000, 9999)}"
        task_type = random.choice(task_types)
        status = random.choice(task_statuses)
        priority = random.randint(1, 3)
        worker_id = f"worker-{random.randint(1, 10):03d}" if status in ['running', 'completed'] else None
            
        tasks.append({
            "id": task_id,
            "type": task_type,
            "status": status,
            "priority": priority,
            "worker_id": worker_id
        })
    
    return tasks

def generate_resource_data():
    """Generate mock resource usage data."""
    import random
    from datetime import datetime, timedelta
    
    # Generate time points for last 24 hours
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=24)
    time_points = []
    labels = []
    
    for i in range(24):
        time_point = start_time + timedelta(hours=i)
        time_points.append(time_point)
        labels.append(time_point.strftime("%H:00"))
    
    # Generate CPU data
    cpu_avg = [random.randint(20, 60) for _ in range(24)]
    cpu_max = [random.randint(65, 95) for _ in range(24)]
    
    # Generate memory data
    memory_avg = [round(random.uniform(1, 4), 1) for _ in range(24)]
    memory_max = [round(random.uniform(4, 8), 1) for _ in range(24)]
    
    return {
        "cpu": {
            "labels": labels,
            "datasets": [
                {
                    "label": "Average CPU Usage",
                    "data": cpu_avg,
                    "borderColor": "#4c9be8",
                    "backgroundColor": "rgba(76, 155, 232, 0.1)",
                    "fill": True
                },
                {
                    "label": "Max CPU Usage",
                    "data": cpu_max,
                    "borderColor": "#e86f4c",
                    "backgroundColor": "rgba(232, 111, 76, 0.1)",
                    "fill": True
                }
            ]
        },
        "memory": {
            "labels": labels,
            "datasets": [
                {
                    "label": "Average Memory Usage (GB)",
                    "data": memory_avg,
                    "borderColor": "#4ca6e8",
                    "backgroundColor": "rgba(76, 166, 232, 0.1)",
                    "fill": True
                },
                {
                    "label": "Max Memory Usage (GB)",
                    "data": memory_max,
                    "borderColor": "#e84ca6",
                    "backgroundColor": "rgba(232, 76, 166, 0.1)",
                    "fill": True
                }
            ]
        }
    }

def generate_hardware_data():
    """Generate mock hardware availability data."""
    import random
    
    hardware_types = ['CPU', 'CUDA', 'ROCm', 'MPS', 'OpenVINO', 'QNN', 'WebNN', 'WebGPU']
    
    # Generate random data for available and total hardware
    available = []
    total = []
    
    for _ in range(len(hardware_types)):
        total_count = random.randint(1, 10)
        available_count = random.randint(0, total_count)
        
        available.append(available_count)
        total.append(total_count)
    
    return {
        "labels": hardware_types,
        "datasets": [
            {
                "label": "Available",
                "data": available,
                "backgroundColor": "rgba(40, 167, 69, 0.7)"
            },
            {
                "label": "Total",
                "data": total,
                "backgroundColor": "rgba(108, 117, 125, 0.3)"
            }
        ]
    }

def generate_network_data():
    """Generate mock network topology data."""
    import random
    
    # Create nodes for coordinator and workers
    nodes = [
        {"id": "coordinator", "group": "coordinator", "status": "active"}
    ]
    
    # Add worker nodes
    links = []
    num_workers = random.randint(5, 10)
    
    for i in range(1, num_workers + 1):
        worker_id = f"worker-{i:03d}"
        status = "active" if random.random() > 0.2 else "inactive"
        
        nodes.append({
            "id": worker_id,
            "group": "worker",
            "status": status
        })
        
        # Link quality is higher for active workers
        link_quality = random.randint(8, 10) if status == "active" else random.randint(1, 3)
        
        links.append({
            "source": "coordinator",
            "target": worker_id,
            "value": link_quality
        })
    
    return {
        "nodes": nodes,
        "links": links
    }

# ===== Notification System =====

def add_notification(notification):
    """Add a notification to the notification system."""
    notifications.append({
        "id": len(notifications),
        "timestamp": datetime.now().isoformat(),
        "type": notification.get("type", "info"),
        "message": notification.get("message", ""),
        "details": notification.get("details", {})
    })
    
    # Emit notification via SocketIO if available
    if SOCKETIO_AVAILABLE:
        socketio.emit('notification', notifications[-1])

# ===== Notification Callback for Result Aggregator =====

def notification_callback(notification):
    """Callback function for result aggregator notifications."""
    add_notification(notification)
    logger.info(f"Received notification: {notification.get('message')}")

# ===== Background Monitoring Thread =====

def background_monitoring_thread(interval=5):
    """Background thread for emitting monitoring data periodically."""
    logger.info(f"Starting background monitoring thread with interval {interval} seconds")
    
    while True:
        try:
            # Emit monitoring data
            emit_monitoring_data()
            
            # Sleep for the interval
            time.sleep(interval)
        except Exception as e:
            logger.error(f"Error in background monitoring thread: {e}")
            time.sleep(5)  # Sleep and retry after error

# ===== Main Function =====

def main():
    global service, visualizer
    
    parser = argparse.ArgumentParser(description='Start the Result Aggregator Web Dashboard')
    parser.add_argument('--db-path', default='./test_results.duckdb', help='Path to DuckDB database')
    parser.add_argument('--port', type=int, default=8050, help='Port to run the web server on')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    parser.add_argument('--enable-ml', action='store_true', default=True, help='Enable machine learning features')
    parser.add_argument('--enable-visualization', action='store_true', default=True, help='Enable visualization features')
    parser.add_argument('--update-interval', type=int, default=5, help='Interval in seconds for real-time monitoring updates')
    
    args = parser.parse_args()
    
    # Create the Result Aggregator Service
    try:
        service = ResultAggregatorService(
            db_path=args.db_path,
            enable_ml=args.enable_ml,
            enable_visualization=args.enable_visualization
        )
        
        # Create the visualizer
        visualizer = ResultVisualizer(service)
        
        logger.info(f"Connected to database at {args.db_path}")
        
        # Add notification callback
        # This would typically be done by the coordinator_integration
        # But we're doing it here for demonstration purposes
        
        # Start background monitoring thread if SocketIO is available
        if SOCKETIO_AVAILABLE:
            # Run in a background thread
            monitoring_thread = threading.Thread(
                target=background_monitoring_thread,
                args=(args.update_interval,),
                daemon=True
            )
            monitoring_thread.start()
            logger.info(f"Background monitoring thread started with update interval of {args.update_interval} seconds")
        
        # Start the web server
        logger.info(f"Starting web server on port {args.port}")
        
        # Create template and static directories if they don't exist
        os.makedirs(os.path.join(os.path.dirname(__file__), 'templates'), exist_ok=True)
        os.makedirs(os.path.join(os.path.dirname(__file__), 'static'), exist_ok=True)
        
        # Run the app
        if SOCKETIO_AVAILABLE:
            socketio.run(app, host='0.0.0.0', port=args.port, debug=args.debug)
        else:
            app.run(host='0.0.0.0', port=args.port, debug=args.debug)
            
    except Exception as e:
        logger.error(f"Error starting web dashboard: {e}")
        sys.exit(1)
    finally:
        if service:
            service.close()
            logger.info("Service closed")

if __name__ == "__main__":
    main()