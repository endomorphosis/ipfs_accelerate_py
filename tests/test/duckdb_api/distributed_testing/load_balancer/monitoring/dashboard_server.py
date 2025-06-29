#!/usr/bin/env python3
"""
Dashboard Server for Load Balancer Monitoring

This module provides a REST API and WebSocket server for the
load balancer monitoring dashboard.
"""

import os
import sys
import json
import time
import logging
import argparse
import threading
import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

# Add parent directory to path to import modules from parent
parent_dir = str(Path(__file__).parent.parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import metrics collector
from duckdb_api.distributed_testing.load_balancer.monitoring.metrics_collector import (
    MetricsCollector, MetricType
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("dashboard_server")

try:
    from flask import Flask, jsonify, request, send_from_directory
    from flask_cors import CORS
    from flask_socketio import SocketIO
    FLASK_AVAILABLE = True
except ImportError:
    logger.warning("Flask or SocketIO not available. Install with 'pip install flask flask-cors flask-socketio'")
    FLASK_AVAILABLE = False

class DashboardServer:
    """
    Dashboard server for load balancer monitoring.
    
    Provides:
    - REST API for historical metrics access
    - WebSocket server for real-time updates
    - Serves static dashboard files
    """
    
    def __init__(self, 
                 metrics_collector: MetricsCollector,
                 host: str = "localhost",
                 port: int = 5000,
                 static_folder: Optional[str] = None):
        """
        Initialize the dashboard server.
        
        Args:
            metrics_collector: Metrics collector instance
            host: Host to bind the server to
            port: Port to bind the server to
            static_folder: Folder containing static dashboard files
        """
        if not FLASK_AVAILABLE:
            raise ImportError("Flask or SocketIO not available. Install with 'pip install flask flask-cors flask-socketio'")
        
        self.metrics_collector = metrics_collector
        self.host = host
        self.port = port
        self.static_folder = static_folder or os.path.join(os.path.dirname(__file__), 'static')
        
        # Ensure static folder exists
        os.makedirs(self.static_folder, exist_ok=True)
        
        # Create Flask app
        self.app = Flask(
            __name__, 
            static_folder=self.static_folder,
            static_url_path=''
        )
        CORS(self.app)
        
        # Create SocketIO server
        self.socketio = SocketIO(
            self.app, 
            cors_allowed_origins="*",
            async_mode="threading"
        )
        
        # Initialize routes
        self._setup_routes()
        
        # Initialize WebSocket handlers
        self._setup_socketio()
        
        # Client subscription tracking
        self.client_subscriptions = {}
        
        # Register metric callbacks
        self._register_callbacks()
        
        # Server thread
        self.server_thread = None
        self.running = False
    
    def _setup_routes(self):
        """Setup API routes."""
        # Root/index route
        @self.app.route('/')
        def index():
            return send_from_directory(self.static_folder, 'index.html')
        
        # System metrics API
        @self.app.route('/api/metrics/system', methods=['GET'])
        def get_system_metrics():
            return jsonify({
                'timestamp': datetime.datetime.now().isoformat(),
                'metrics': self.metrics_collector.get_current_system_metrics()
            })
        
        @self.app.route('/api/metrics/system/history', methods=['GET'])
        def get_system_metrics_history():
            # Get query parameters
            metrics = request.args.get('metrics', '').split(',')
            start_time = request.args.get('start_time')
            end_time = request.args.get('end_time')
            interval = request.args.get('interval', '1m')
            
            # Convert timestamps if provided
            if start_time:
                start_time = datetime.datetime.fromisoformat(start_time)
            if end_time:
                end_time = datetime.datetime.fromisoformat(end_time)
            
            # Get historical metrics
            metrics_data = self.metrics_collector.get_historical_system_metrics(
                metrics, start_time, end_time, interval
            )
            
            # Convert to format suitable for JSON
            formatted_data = {}
            for metric, values in metrics_data.items():
                formatted_data[metric] = [
                    {'timestamp': ts.isoformat(), 'value': val}
                    for ts, val in values
                ]
            
            return jsonify(formatted_data)
        
        # Worker metrics API
        @self.app.route('/api/metrics/workers', methods=['GET'])
        def get_workers_metrics():
            return jsonify({
                'timestamp': datetime.datetime.now().isoformat(),
                'metrics': self.metrics_collector.get_current_worker_metrics()
            })
        
        @self.app.route('/api/metrics/workers/<worker_id>', methods=['GET'])
        def get_worker_metrics(worker_id):
            worker_metrics = self.metrics_collector.get_current_worker_metrics()
            
            if worker_id in worker_metrics:
                return jsonify({
                    'timestamp': datetime.datetime.now().isoformat(),
                    'metrics': worker_metrics[worker_id]
                })
            else:
                return jsonify({'error': f"Worker {worker_id} not found"}), 404
        
        @self.app.route('/api/metrics/workers/<worker_id>/history', methods=['GET'])
        def get_worker_metrics_history(worker_id):
            # Get query parameters
            metrics = request.args.get('metrics', '').split(',')
            start_time = request.args.get('start_time')
            end_time = request.args.get('end_time')
            interval = request.args.get('interval', '1m')
            
            # Convert timestamps if provided
            if start_time:
                start_time = datetime.datetime.fromisoformat(start_time)
            if end_time:
                end_time = datetime.datetime.fromisoformat(end_time)
            
            # Get historical metrics
            metrics_data = self.metrics_collector.get_historical_worker_metrics(
                worker_id, metrics, start_time, end_time, interval
            )
            
            # Convert to format suitable for JSON
            formatted_data = {}
            for metric, values in metrics_data.items():
                formatted_data[metric] = [
                    {'timestamp': ts.isoformat(), 'value': val}
                    for ts, val in values
                ]
            
            return jsonify(formatted_data)
        
        @self.app.route('/api/metrics/workers/<worker_id>/performance', methods=['GET'])
        def get_worker_performance(worker_id):
            score = self.metrics_collector.get_worker_performance_score(worker_id)
            
            return jsonify({
                'worker_id': worker_id,
                'performance_score': score
            })
        
        # Worker metadata API
        @self.app.route('/api/workers/metadata', methods=['GET'])
        def get_workers_metadata():
            metadata = self.metrics_collector.get_worker_metadata()
            return jsonify(metadata)
        
        @self.app.route('/api/workers/metadata/<worker_id>', methods=['GET'])
        def get_worker_metadata(worker_id):
            metadata = self.metrics_collector.get_worker_metadata(worker_id)
            
            if worker_id in metadata:
                return jsonify(metadata[worker_id])
            else:
                return jsonify({'error': f"Worker {worker_id} not found"}), 404
        
        # Task metrics API
        @self.app.route('/api/metrics/tasks', methods=['GET'])
        def get_tasks_metrics():
            return jsonify({
                'timestamp': datetime.datetime.now().isoformat(),
                'metrics': self.metrics_collector.get_current_task_metrics()
            })
        
        @self.app.route('/api/metrics/tasks/<task_id>', methods=['GET'])
        def get_task_metrics(task_id):
            task_metrics = self.metrics_collector.get_current_task_metrics()
            
            if task_id in task_metrics:
                return jsonify({
                    'timestamp': datetime.datetime.now().isoformat(),
                    'metrics': task_metrics[task_id]
                })
            else:
                return jsonify({'error': f"Task {task_id} not found"}), 404
        
        # Task metadata API
        @self.app.route('/api/tasks/metadata', methods=['GET'])
        def get_tasks_metadata():
            metadata = self.metrics_collector.get_task_metadata()
            return jsonify(metadata)
        
        @self.app.route('/api/tasks/metadata/<task_id>', methods=['GET'])
        def get_task_metadata(task_id):
            metadata = self.metrics_collector.get_task_metadata(task_id)
            
            if task_id in metadata:
                return jsonify(metadata[task_id])
            else:
                return jsonify({'error': f"Task {task_id} not found"}), 404
        
        # Task statistics API
        @self.app.route('/api/metrics/tasks/stats', methods=['GET'])
        def get_task_stats():
            task_metrics = self.metrics_collector.get_current_task_metrics()
            
            # Count tasks by status
            status_counts = {
                'queued': 0,
                'assigned': 0,
                'running': 0,
                'completed': 0,
                'failed': 0
            }
            
            # Count tasks by model family
            family_counts = {}
            
            # Calculate average processing time
            processing_times = []
            
            # Get task metadata for model family
            task_metadata = self.metrics_collector.get_task_metadata()
            
            for task_id, metrics in task_metrics.items():
                # Status counts
                for status in status_counts.keys():
                    if metrics.get(f'status_{status}', 0) > 0:
                        status_counts[status] += 1
                
                # Processing time
                if 'processing_time' in metrics:
                    processing_times.append(metrics['processing_time'])
                
                # Model family counts
                if task_id in task_metadata:
                    model_family = task_metadata[task_id].get('model_family', 'unknown')
                    family_counts[model_family] = family_counts.get(model_family, 0) + 1
            
            # Calculate average processing time
            avg_processing_time = 0
            if processing_times:
                avg_processing_time = sum(processing_times) / len(processing_times)
            
            return jsonify({
                'status_counts': status_counts,
                'family_counts': family_counts,
                'avg_processing_time': avg_processing_time,
                'total_tasks': len(task_metrics)
            })
        
        # Anomaly detection API
        @self.app.route('/api/anomalies', methods=['GET'])
        def get_anomalies():
            anomalies = self.metrics_collector.detect_anomalies()
            return jsonify(anomalies)
    
    def _setup_socketio(self):
        """Setup WebSocket event handlers."""
        @self.socketio.on('connect')
        def handle_connect():
            client_id = request.sid
            logger.info(f"Client connected: {client_id}")
            self.client_subscriptions[client_id] = set()
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            client_id = request.sid
            logger.info(f"Client disconnected: {client_id}")
            if client_id in self.client_subscriptions:
                del self.client_subscriptions[client_id]
        
        @self.socketio.on('subscribe')
        def handle_subscribe(data):
            client_id = request.sid
            metric_type = data.get('metric_type')
            
            if metric_type:
                if client_id not in self.client_subscriptions:
                    self.client_subscriptions[client_id] = set()
                
                self.client_subscriptions[client_id].add(metric_type)
                logger.info(f"Client {client_id} subscribed to {metric_type} metrics")
        
        @self.socketio.on('unsubscribe')
        def handle_unsubscribe(data):
            client_id = request.sid
            metric_type = data.get('metric_type')
            
            if client_id in self.client_subscriptions and metric_type in self.client_subscriptions[client_id]:
                self.client_subscriptions[client_id].remove(metric_type)
                logger.info(f"Client {client_id} unsubscribed from {metric_type} metrics")
    
    def _register_callbacks(self):
        """Register callbacks with metrics collector."""
        # System metrics callback
        def system_metrics_callback(timestamp, metrics):
            self._emit_to_subscribers('system', {
                'timestamp': timestamp.isoformat(),
                'metrics': metrics
            })
        
        # Worker metrics callback
        def worker_metrics_callback(timestamp, metrics):
            self._emit_to_subscribers('worker', {
                'timestamp': timestamp.isoformat(),
                'metrics': metrics
            })
        
        # Task metrics callback
        def task_metrics_callback(timestamp, metrics):
            self._emit_to_subscribers('task', {
                'timestamp': timestamp.isoformat(),
                'metrics': metrics
            })
        
        # Register callbacks
        self.metrics_collector.register_callback(MetricType.SYSTEM, system_metrics_callback)
        self.metrics_collector.register_callback(MetricType.WORKER, worker_metrics_callback)
        self.metrics_collector.register_callback(MetricType.TASK, task_metrics_callback)
    
    def _emit_to_subscribers(self, metric_type, data):
        """
        Emit metrics data to subscribers.
        
        Args:
            metric_type: Type of metrics (system, worker, task)
            data: Metrics data to emit
        """
        for client_id, subscriptions in self.client_subscriptions.items():
            if metric_type in subscriptions:
                try:
                    self.socketio.emit(f'metrics_{metric_type}', data, room=client_id)
                except Exception as e:
                    logger.error(f"Error emitting to client {client_id}: {e}")
    
    def start(self):
        """Start the dashboard server."""
        if self.running:
            logger.warning("Dashboard server already running")
            return
        
        self.running = True
        
        # Start metrics collector if not already running
        if not getattr(self.metrics_collector, 'collection_thread', None) or not self.metrics_collector.collection_thread.is_alive():
            self.metrics_collector.start()
        
        # Start server in a thread
        self.server_thread = threading.Thread(
            target=self._run_server,
            daemon=True
        )
        self.server_thread.start()
        
        logger.info(f"Dashboard server started on {self.host}:{self.port}")
    
    def _run_server(self):
        """Run the dashboard server."""
        try:
            self.socketio.run(self.app, host=self.host, port=self.port, debug=False)
        except Exception as e:
            logger.error(f"Error running dashboard server: {e}")
            self.running = False
    
    def stop(self):
        """Stop the dashboard server."""
        if not self.running:
            logger.warning("Dashboard server not running")
            return
        
        self.running = False
        
        # Stop metrics collector
        self.metrics_collector.stop()
        
        # Server will stop when thread is stopped
        logger.info("Dashboard server stopped")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Load Balancer Monitoring Dashboard Server")
    parser.add_argument("--host", type=str, default="localhost", help="Host to bind to")
    parser.add_argument("--port", type=int, default=5000, help="Port to bind to")
    parser.add_argument("--db-path", type=str, default="metrics.duckdb", help="Path to metrics database")
    parser.add_argument("--static-folder", type=str, help="Folder containing static dashboard files")
    parser.add_argument("--collection-interval", type=float, default=1.0, help="Metrics collection interval in seconds")
    
    args = parser.parse_args()
    
    # Create metrics collector
    metrics_collector = MetricsCollector(
        db_path=args.db_path,
        collection_interval=args.collection_interval
    )
    
    # Create dashboard server
    dashboard_server = DashboardServer(
        metrics_collector=metrics_collector,
        host=args.host,
        port=args.port,
        static_folder=args.static_folder
    )
    
    # Start dashboard server
    dashboard_server.start()
    
    try:
        # Keep running until interrupted
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        # Stop dashboard server
        dashboard_server.stop()
        logger.info("Dashboard server stopped")


if __name__ == "__main__":
    main()