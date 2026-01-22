#!/usr/bin/env python3
"""
API Distributed Testing Coordinator Server

This script runs a coordinator server for the API Distributed Testing framework.
The coordinator server manages worker nodes, distributes API test tasks, and
collects test results.
"""

import os
import sys
import time
import json
import uuid
import argparse
import logging
import datetime
import threading
import traceback
from typing import Dict, List, Set, Any, Optional, Union, Tuple
from enum import Enum
from pathlib import Path

# Add parent directory to path for local development
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import needed modules
try:
    from distributed_testing.coordinator import TestCoordinator
    from distributed_testing.task import Task, TaskResult, TaskStatus
    from api_unified_testing_interface import (
        APIProvider, 
        APITestType, 
        APIBackendFactory, 
        APITester, 
        APIDistributedTesting
    )
    from api_anomaly_detection import AnomalyDetector, AnomalySeverity
    from api_predictive_analytics import TimeSeriesPredictor, PerformanceForecaster
    from api_notification_manager import NotificationManager, NotificationRule
    from api_monitoring_dashboard import APIMonitoringDashboard
except ImportError:
    # Add test directory to path
    sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "test"))
    # Try import again
    from distributed_testing.coordinator import TestCoordinator
    from distributed_testing.task import Task, TaskResult, TaskStatus
    from api_unified_testing_interface import (
        APIProvider, 
        APITestType, 
        APIBackendFactory, 
        APITester, 
        APIDistributedTesting
    )
    from api_anomaly_detection import AnomalyDetector, AnomalySeverity
    from api_predictive_analytics import TimeSeriesPredictor, PerformanceForecaster
    from api_notification_manager import NotificationManager, NotificationRule
    from api_monitoring_dashboard import APIMonitoringDashboard

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class APICoordinatorServer:
    """
    API Coordinator Server for managing API distributed testing.
    
    This class extends the base TestCoordinator to add API-specific functionality,
    including provider-specific task management, anomaly detection, predictive analytics,
    and monitoring dashboard integration.
    """
    
    def __init__(self,
                 host: str = '0.0.0.0',
                 port: int = 5555,
                 results_dir: str = "./api_test_results",
                 enable_anomaly_detection: bool = True,
                 enable_predictive_analytics: bool = True,
                 enable_dashboard: bool = True,
                 dashboard_port: int = 8080,
                 heartbeat_interval: int = 10,
                 worker_timeout: int = 30,
                 notification_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the API Coordinator Server.
        
        Args:
            host: Host to bind to
            port: Port to bind to
            results_dir: Directory for storing test results
            enable_anomaly_detection: Whether to enable anomaly detection
            enable_predictive_analytics: Whether to enable predictive analytics
            enable_dashboard: Whether to enable the monitoring dashboard
            dashboard_port: Port for the monitoring dashboard
            heartbeat_interval: Interval in seconds between heartbeats
            worker_timeout: Timeout in seconds for worker heartbeats
            notification_config: Configuration for notifications
        """
        # Initialize base coordinator
        self.coordinator = TestCoordinator(
            host=host,
            port=port,
            heartbeat_interval=heartbeat_interval,
            worker_timeout=worker_timeout
        )
        
        # API specific properties
        self.results_dir = results_dir
        self.enable_anomaly_detection = enable_anomaly_detection
        self.enable_predictive_analytics = enable_predictive_analytics
        self.enable_dashboard = enable_dashboard
        self.dashboard_port = dashboard_port
        
        # Ensure results directory exists
        os.makedirs(results_dir, exist_ok=True)
        
        # Initialize API capabilities registry
        self.provider_registry = {}
        self.model_registry = {}
        self.capability_registry = {}
        
        # Initialize anomaly detection if enabled
        if enable_anomaly_detection:
            self.anomaly_detector = AnomalyDetector()
        else:
            self.anomaly_detector = None
        
        # Initialize predictive analytics if enabled
        if enable_predictive_analytics:
            self.time_series_predictor = TimeSeriesPredictor()
            self.performance_forecaster = PerformanceForecaster()
        else:
            self.time_series_predictor = None
            self.performance_forecaster = None
        
        # Initialize notification manager
        self.notification_manager = NotificationManager(
            config=notification_config or {}
        )
        
        # Initialize dashboard if enabled
        if enable_dashboard:
            self.dashboard = APIMonitoringDashboard(
                port=dashboard_port,
                anomaly_detector=self.anomaly_detector,
                predictor=self.time_series_predictor,
                coordinator=self
            )
        else:
            self.dashboard = None
            
        # Initialize stats
        self.stats = {
            'tasks_created': 0,
            'tasks_completed': 0,
            'tasks_failed': 0,
            'api_providers': set(),
            'models_tested': set(),
            'worker_nodes': 0,
            'active_worker_nodes': 0,
            'start_time': time.time()
        }
        
        # Initialize locks
        self.stats_lock = threading.Lock()
        
        # Initialize worker-to-api_provider mapping
        self.worker_api_mapping = {}
        
        # Initialize event for stopping threads
        self.stop_event = threading.Event()
        
        # Initialize analysis thread
        self.analysis_thread = threading.Thread(target=self._analyze_results_loop)
        
        logger.info(f"API Coordinator Server initialized at {host}:{port}")
    
    def start(self) -> None:
        """Start the API Coordinator Server."""
        logger.info("Starting API Coordinator Server")
        
        # Start the base coordinator
        self.coordinator.start()
        
        # Start analysis thread
        self.analysis_thread.start()
        
        # Start dashboard if enabled
        if self.dashboard:
            self.dashboard.start()
        
        logger.info("API Coordinator Server started")
    
    def stop(self) -> None:
        """Stop the API Coordinator Server."""
        logger.info("Stopping API Coordinator Server")
        
        # Set stop event
        self.stop_event.set()
        
        # Stop the base coordinator
        self.coordinator.stop()
        
        # Stop analysis thread
        self.analysis_thread.join()
        
        # Stop dashboard if enabled
        if self.dashboard:
            self.dashboard.stop()
        
        logger.info("API Coordinator Server stopped")
    
    def register_worker(self, hostname: str, ip_address: str, capabilities: Dict[str, Any]) -> str:
        """
        Register a new worker node.
        
        Args:
            hostname: Hostname of the worker
            ip_address: IP address of the worker
            capabilities: Capabilities of the worker
            
        Returns:
            Worker ID
        """
        # Extract API provider capabilities
        api_providers = capabilities.get('providers', [])
        
        # Register with base coordinator
        worker_id = self.coordinator.register_worker(hostname, ip_address, capabilities)
        
        # Update API capabilities registry
        with self.stats_lock:
            self.worker_api_mapping[worker_id] = api_providers
            
            # Add to provider registry
            for provider in api_providers:
                if provider not in self.provider_registry:
                    self.provider_registry[provider] = set()
                self.provider_registry[provider].add(worker_id)
                self.stats['api_providers'].add(provider)
            
            # Update stats
            self.stats['worker_nodes'] += 1
            self.stats['active_worker_nodes'] += 1
        
        logger.info(f"Registered worker {worker_id} with API providers: {', '.join(api_providers)}")
        
        return worker_id
    
    def unregister_worker(self, worker_id: str) -> bool:
        """
        Unregister a worker node.
        
        Args:
            worker_id: Worker ID
            
        Returns:
            True if successful, False otherwise
        """
        # First unregister from API registries
        with self.stats_lock:
            if worker_id in self.worker_api_mapping:
                api_providers = self.worker_api_mapping[worker_id]
                
                # Remove from provider registry
                for provider in api_providers:
                    if provider in self.provider_registry and worker_id in self.provider_registry[provider]:
                        self.provider_registry[provider].remove(worker_id)
                
                # Remove from mapping
                del self.worker_api_mapping[worker_id]
                
                # Update stats
                self.stats['active_worker_nodes'] -= 1
            else:
                logger.warning(f"Attempted to unregister unknown worker {worker_id}")
                return False
        
        # Then unregister from base coordinator
        return self.coordinator.unregister_worker(worker_id)
    
    def process_result(self, task_id: str, result: Dict[str, Any]) -> None:
        """
        Process a task result.
        
        Args:
            task_id: Task ID
            result: Task result data
        """
        # Extract metadata
        provider = result.get('metadata', {}).get('provider')
        model = result.get('metadata', {}).get('model')
        test_type = result.get('metadata', {}).get('test_type')
        
        # Update stats
        with self.stats_lock:
            if result.get('status') == 'COMPLETED':
                self.stats['tasks_completed'] += 1
            elif result.get('status') == 'FAILED':
                self.stats['tasks_failed'] += 1
            
            if provider:
                self.stats['api_providers'].add(provider)
            
            if model:
                self.stats['models_tested'].add(model)
        
        # Save result to disk
        result_file = os.path.join(self.results_dir, f"{task_id}.json")
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        # Analyze result for anomalies if enabled
        if self.enable_anomaly_detection and self.anomaly_detector and provider and test_type:
            logger.debug(f"Analyzing result for anomalies: {task_id}")
            try:
                anomalies = []
                
                if test_type == 'latency':
                    if 'latencies_ms' in result:
                        anomalies = self.anomaly_detector.detect_latency_anomalies(
                            provider, result['latencies_ms']
                        )
                elif test_type == 'throughput':
                    if 'throughput_rps' in result:
                        anomalies = self.anomaly_detector.detect_throughput_anomalies(
                            provider, result['throughput_rps']
                        )
                elif test_type == 'reliability':
                    if 'success_rate' in result:
                        anomalies = self.anomaly_detector.detect_reliability_anomalies(
                            provider, result['success_rate']
                        )
                
                # Notify if anomalies detected
                if anomalies:
                    logger.warning(f"Anomalies detected in task {task_id}: {len(anomalies)} issues found")
                    
                    # Send notifications
                    for anomaly in anomalies:
                        if anomaly.get('severity', AnomalySeverity.LOW) >= AnomalySeverity.MEDIUM:
                            self.notification_manager.send_notification(
                                title=f"API Anomaly Detected: {provider}",
                                message=f"Anomaly detected in {test_type} test for {provider} ({model}): {anomaly.get('message')}",
                                severity=anomaly.get('severity', AnomalySeverity.MEDIUM),
                                data={
                                    'task_id': task_id,
                                    'provider': provider,
                                    'model': model,
                                    'test_type': test_type,
                                    'anomaly': anomaly
                                }
                            )
            except Exception as e:
                logger.error(f"Error analyzing result for anomalies: {e}")
    
    def create_api_test(self, 
                        api_type: Union[str, APIProvider],
                        test_type: Union[str, APITestType],
                        parameters: Dict[str, Any],
                        priority: int = 0) -> str:
        """
        Create an API test task.
        
        Args:
            api_type: API provider type
            test_type: Test type
            parameters: Test parameters
            priority: Task priority
            
        Returns:
            Task ID
        """
        # Convert enum values to strings if needed
        if isinstance(api_type, APIProvider):
            api_type = api_type.value
            
        if isinstance(test_type, APITestType):
            test_type = test_type.value
        
        # Create task parameters
        task_params = {
            'api_type': api_type,
            'test_type': test_type,
            'test_parameters': parameters
        }
        
        # Create task
        task_id = self.coordinator.create_task(
            test_path=f"api_test_{api_type}_{test_type}",
            parameters=task_params,
            priority=priority
        )
        
        # Update stats
        with self.stats_lock:
            self.stats['tasks_created'] += 1
            if api_type:
                self.stats['api_providers'].add(api_type)
        
        logger.info(f"Created API test task {task_id} for {api_type} ({test_type})")
        
        return task_id
    
    def create_api_comparison(self,
                             api_types: List[Union[str, APIProvider]],
                             test_type: Union[str, APITestType],
                             parameters: Dict[str, Any],
                             priority: int = 0) -> Dict[str, str]:
        """
        Create an API comparison test.
        
        Args:
            api_types: List of API provider types
            test_type: Test type
            parameters: Test parameters
            priority: Task priority
            
        Returns:
            Dictionary mapping API types to task IDs
        """
        # Create a task for each API type
        task_ids = {}
        
        for api_type in api_types:
            task_id = self.create_api_test(
                api_type=api_type,
                test_type=test_type,
                parameters=parameters,
                priority=priority
            )
            
            # Store task ID
            if isinstance(api_type, APIProvider):
                task_ids[api_type.value] = task_id
            else:
                task_ids[api_type] = task_id
        
        # Create a comparison record
        comparison_id = str(uuid.uuid4())
        comparison_file = os.path.join(self.results_dir, f"comparison_{comparison_id}.json")
        
        with open(comparison_file, 'w') as f:
            json.dump({
                'id': comparison_id,
                'api_types': [api.value if isinstance(api, APIProvider) else api for api in api_types],
                'test_type': test_type.value if isinstance(test_type, APITestType) else test_type,
                'parameters': parameters,
                'task_ids': task_ids,
                'created_at': datetime.datetime.now().isoformat()
            }, f, indent=2)
        
        logger.info(f"Created API comparison {comparison_id} for {len(api_types)} providers")
        
        return task_ids
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the status of a task.
        
        Args:
            task_id: Task ID
            
        Returns:
            Task status data or None if not found
        """
        # Get status from coordinator
        status = self.coordinator.get_task_status(task_id)
        
        if not status:
            # Try to load from results directory
            result_file = os.path.join(self.results_dir, f"{task_id}.json")
            if os.path.exists(result_file):
                try:
                    with open(result_file, 'r') as f:
                        return json.load(f)
                except Exception as e:
                    logger.error(f"Error loading task result file: {e}")
            
            return None
        
        return status
    
    def get_comparison_status(self, comparison_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the status of a comparison.
        
        Args:
            comparison_id: Comparison ID
            
        Returns:
            Comparison status data or None if not found
        """
        # Load comparison file
        comparison_file = os.path.join(self.results_dir, f"comparison_{comparison_id}.json")
        
        if not os.path.exists(comparison_file):
            return None
        
        try:
            with open(comparison_file, 'r') as f:
                comparison = json.load(f)
            
            # Get status of each task
            results = {}
            
            for api_type, task_id in comparison.get('task_ids', {}).items():
                status = self.get_task_status(task_id)
                results[api_type] = status or {'status': 'UNKNOWN'}
            
            # Add results to comparison
            comparison['results'] = results
            
            return comparison
        except Exception as e:
            logger.error(f"Error loading comparison file: {e}")
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get coordinator statistics.
        
        Returns:
            Statistics data
        """
        with self.stats_lock:
            # Copy stats to avoid modification during access
            stats = self.stats.copy()
            
            # Convert sets to lists for JSON serialization
            stats['api_providers'] = list(stats['api_providers'])
            stats['models_tested'] = list(stats['models_tested'])
            
            # Add uptime
            stats['uptime'] = time.time() - stats['start_time']
            
            # Add worker stats
            worker_stats = self.coordinator.get_statistics()
            stats.update({
                'tasks_pending': worker_stats.get('tasks_pending', 0),
                'tasks_running': worker_stats.get('tasks_running', 0)
            })
            
            return stats
    
    def get_provider_stats(self, provider: str) -> Dict[str, Any]:
        """
        Get statistics for a specific API provider.
        
        Args:
            provider: API provider name
            
        Returns:
            Provider statistics data
        """
        # Get all completed tasks for this provider
        provider_tasks = []
        
        for filename in os.listdir(self.results_dir):
            if filename.endswith('.json') and not filename.startswith('comparison_'):
                try:
                    with open(os.path.join(self.results_dir, filename), 'r') as f:
                        result = json.load(f)
                        
                        if result.get('metadata', {}).get('provider') == provider:
                            provider_tasks.append(result)
                except Exception as e:
                    logger.error(f"Error loading result file {filename}: {e}")
        
        # Calculate metrics
        stats = {
            'provider': provider,
            'total_tasks': len(provider_tasks),
            'models': set(),
            'latency_tests': 0,
            'throughput_tests': 0,
            'reliability_tests': 0,
            'cost_efficiency_tests': 0,
            'avg_latency_ms': None,
            'avg_throughput_rps': None,
            'avg_success_rate': None,
            'avg_tokens_per_dollar': None
        }
        
        # Collect data for averages
        latencies = []
        throughputs = []
        success_rates = []
        tokens_per_dollar = []
        
        for task in provider_tasks:
            # Add model to set
            model = task.get('metadata', {}).get('model')
            if model:
                stats['models'].add(model)
            
            # Count test types
            test_type = task.get('metadata', {}).get('test_type')
            if test_type == 'latency':
                stats['latency_tests'] += 1
                
                # Collect latency data
                if 'mean_latency_ms' in task:
                    latencies.append(task['mean_latency_ms'])
            elif test_type == 'throughput':
                stats['throughput_tests'] += 1
                
                # Collect throughput data
                if 'throughput_rps' in task:
                    throughputs.append(task['throughput_rps'])
            elif test_type == 'reliability':
                stats['reliability_tests'] += 1
                
                # Collect reliability data
                if 'success_rate' in task:
                    success_rates.append(task['success_rate'])
            elif test_type == 'cost_efficiency':
                stats['cost_efficiency_tests'] += 1
                
                # Collect cost efficiency data
                if 'cost_efficiency_metrics' in task and 'tokens_per_dollar' in task['cost_efficiency_metrics']:
                    tokens_per_dollar.append(task['cost_efficiency_metrics']['tokens_per_dollar'])
        
        # Calculate averages
        if latencies:
            stats['avg_latency_ms'] = sum(latencies) / len(latencies)
        
        if throughputs:
            stats['avg_throughput_rps'] = sum(throughputs) / len(throughputs)
        
        if success_rates:
            stats['avg_success_rate'] = sum(success_rates) / len(success_rates)
        
        if tokens_per_dollar:
            stats['avg_tokens_per_dollar'] = sum(tokens_per_dollar) / len(tokens_per_dollar)
        
        # Convert sets to lists for JSON serialization
        stats['models'] = list(stats['models'])
        
        return stats
    
    def _analyze_results_loop(self) -> None:
        """Background thread for analyzing test results."""
        logger.info("Starting result analysis thread")
        
        while not self.stop_event.is_set():
            try:
                # Only run analysis if anomaly detection or predictive analytics is enabled
                if self.enable_anomaly_detection or self.enable_predictive_analytics:
                    self._analyze_all_results()
                
                # Update dashboard if enabled
                if self.dashboard:
                    self.dashboard.update_data()
                
                # Wait for next iteration
                self.stop_event.wait(60)  # Run analysis every minute
            except Exception as e:
                logger.error(f"Error in result analysis loop: {e}")
                traceback.print_exc()
                self.stop_event.wait(5)  # Wait a bit before retrying
    
    def _analyze_all_results(self) -> None:
        """Analyze all test results for trends and anomalies."""
        # Skip if neither anomaly detection nor predictive analytics is enabled
        if not (self.enable_anomaly_detection or self.enable_predictive_analytics):
            return
        
        # Get all API providers
        providers = set()
        
        with self.stats_lock:
            providers = set(self.stats['api_providers'])
        
        # Analyze each provider
        for provider in providers:
            try:
                # Get provider stats
                provider_stats = self.get_provider_stats(provider)
                
                # Skip if no tasks
                if provider_stats['total_tasks'] == 0:
                    continue
                
                # Collect latency data
                latency_data = []
                
                # Collect throughput data
                throughput_data = []
                
                # Collect reliability data
                reliability_data = []
                
                # Load all results for this provider
                for filename in os.listdir(self.results_dir):
                    if filename.endswith('.json') and not filename.startswith('comparison_'):
                        try:
                            with open(os.path.join(self.results_dir, filename), 'r') as f:
                                result = json.load(f)
                                
                                if result.get('metadata', {}).get('provider') != provider:
                                    continue
                                
                                # Extract timestamp and convert to datetime
                                timestamp_str = result.get('metadata', {}).get('timestamp')
                                if not timestamp_str:
                                    continue
                                
                                try:
                                    timestamp = datetime.datetime.fromisoformat(timestamp_str)
                                except:
                                    continue
                                
                                # Add data points based on test type
                                test_type = result.get('metadata', {}).get('test_type')
                                if test_type == 'latency' and 'mean_latency_ms' in result:
                                    latency_data.append((timestamp, result['mean_latency_ms']))
                                elif test_type == 'throughput' and 'throughput_rps' in result:
                                    throughput_data.append((timestamp, result['throughput_rps']))
                                elif test_type == 'reliability' and 'success_rate' in result:
                                    reliability_data.append((timestamp, result['success_rate']))
                        except Exception as e:
                            logger.error(f"Error loading result file {filename}: {e}")
                
                # Sort data by timestamp
                latency_data.sort(key=lambda x: x[0])
                throughput_data.sort(key=lambda x: x[0])
                reliability_data.sort(key=lambda x: x[0])
                
                # Run predictive analytics if enabled
                if self.enable_predictive_analytics and self.time_series_predictor:
                    # Predict latency trends
                    if len(latency_data) >= 5:
                        timestamps = [x[0].timestamp() for x in latency_data]
                        values = [x[1] for x in latency_data]
                        
                        prediction = self.time_series_predictor.predict_timeseries(
                            provider, 'latency', timestamps, values
                        )
                        
                        logger.debug(f"Latency prediction for {provider}: {prediction}")
                        
                        # Check if prediction indicates deteriorating performance
                        if prediction.get('trend') == 'increasing' and prediction.get('confidence', 0) > 0.7:
                            # Send notification
                            self.notification_manager.send_notification(
                                title=f"API Performance Trend Alert: {provider}",
                                message=f"Latency for {provider} is trending upward with high confidence. Forecast: {prediction.get('forecast', 'Unknown')}",
                                severity=AnomalySeverity.MEDIUM,
                                data={
                                    'provider': provider,
                                    'metric': 'latency',
                                    'prediction': prediction
                                }
                            )
                    
                    # Predict throughput trends
                    if len(throughput_data) >= 5:
                        timestamps = [x[0].timestamp() for x in throughput_data]
                        values = [x[1] for x in throughput_data]
                        
                        prediction = self.time_series_predictor.predict_timeseries(
                            provider, 'throughput', timestamps, values
                        )
                        
                        logger.debug(f"Throughput prediction for {provider}: {prediction}")
                        
                        # Check if prediction indicates deteriorating performance
                        if prediction.get('trend') == 'decreasing' and prediction.get('confidence', 0) > 0.7:
                            # Send notification
                            self.notification_manager.send_notification(
                                title=f"API Performance Trend Alert: {provider}",
                                message=f"Throughput for {provider} is trending downward with high confidence. Forecast: {prediction.get('forecast', 'Unknown')}",
                                severity=AnomalySeverity.MEDIUM,
                                data={
                                    'provider': provider,
                                    'metric': 'throughput',
                                    'prediction': prediction
                                }
                            )
                    
                    # Predict reliability trends
                    if len(reliability_data) >= 5:
                        timestamps = [x[0].timestamp() for x in reliability_data]
                        values = [x[1] for x in reliability_data]
                        
                        prediction = self.time_series_predictor.predict_timeseries(
                            provider, 'reliability', timestamps, values
                        )
                        
                        logger.debug(f"Reliability prediction for {provider}: {prediction}")
                        
                        # Check if prediction indicates deteriorating performance
                        if prediction.get('trend') == 'decreasing' and prediction.get('confidence', 0) > 0.7:
                            # Send notification
                            self.notification_manager.send_notification(
                                title=f"API Reliability Trend Alert: {provider}",
                                message=f"Reliability for {provider} is trending downward with high confidence. Forecast: {prediction.get('forecast', 'Unknown')}",
                                severity=AnomalySeverity.HIGH,
                                data={
                                    'provider': provider,
                                    'metric': 'reliability',
                                    'prediction': prediction
                                }
                            )
            except Exception as e:
                logger.error(f"Error analyzing results for provider {provider}: {e}")
                traceback.print_exc()
    
    def generate_performance_report(self, output_file: Optional[str] = None) -> str:
        """
        Generate a performance report for all API providers.
        
        Args:
            output_file: Optional output file path
            
        Returns:
            Path to the generated report
        """
        # Get all API providers
        providers = set()
        
        with self.stats_lock:
            providers = set(self.stats['api_providers'])
        
        # Create report
        report = {
            'generated_at': datetime.datetime.now().isoformat(),
            'coordinator_stats': self.get_stats(),
            'providers': {}
        }
        
        # Add provider stats
        for provider in providers:
            report['providers'][provider] = self.get_provider_stats(provider)
        
        # Generate output file if not provided
        if not output_file:
            os.makedirs('reports', exist_ok=True)
            output_file = f"reports/api_performance_report_{int(time.time())}.json"
        
        # Write report to file
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Generated performance report: {output_file}")
        
        return output_file


def main():
    """Command-line interface for the API Coordinator Server."""
    parser = argparse.ArgumentParser(description='API Distributed Testing Coordinator Server')
    
    # Server configuration
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5555, help='Port to bind to')
    
    # Feature flags
    parser.add_argument('--disable-anomaly-detection', action='store_true', help='Disable anomaly detection')
    parser.add_argument('--disable-predictive-analytics', action='store_true', help='Disable predictive analytics')
    parser.add_argument('--disable-dashboard', action='store_true', help='Disable monitoring dashboard')
    
    # Dashboard configuration
    parser.add_argument('--dashboard-port', type=int, default=8080, help='Port for the monitoring dashboard')
    
    # Worker management
    parser.add_argument('--heartbeat-interval', type=int, default=10, help='Heartbeat interval in seconds')
    parser.add_argument('--worker-timeout', type=int, default=30, help='Worker timeout in seconds')
    
    # Output configuration
    parser.add_argument('--results-dir', type=str, default='./api_test_results', help='Directory for storing test results')
    
    args = parser.parse_args()
    
    # Create coordinator server
    coordinator = APICoordinatorServer(
        host=args.host,
        port=args.port,
        results_dir=args.results_dir,
        enable_anomaly_detection=not args.disable_anomaly_detection,
        enable_predictive_analytics=not args.disable_predictive_analytics,
        enable_dashboard=not args.disable_dashboard,
        dashboard_port=args.dashboard_port,
        heartbeat_interval=args.heartbeat_interval,
        worker_timeout=args.worker_timeout
    )
    
    try:
        # Start coordinator
        coordinator.start()
        
        logger.info(f"API Coordinator Server running at {args.host}:{args.port}")
        if not args.disable_dashboard:
            logger.info(f"Monitoring dashboard available at http://{args.host}:{args.dashboard_port}")
        
        # Wait for keyboard interrupt
        while True:
            try:
                time.sleep(1)
            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received, shutting down...")
                break
    finally:
        # Stop coordinator
        coordinator.stop()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())