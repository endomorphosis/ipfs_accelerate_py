#!/usr/bin/env python3
"""
Result Aggregator Coordinator Integration

This module integrates the Result Aggregator Service with the Distributed Testing Framework
Coordinator. It provides methods to handle real-time result processing and analysis.

Usage:
    # Initialize integration with coordinator
    integration = ResultAggregatorIntegration(coordinator, db_path="./test_db.duckdb")
    
    # Register with coordinator
    integration.register_with_coordinator()
"""

import anyio
import json
import logging
import inspect
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple, Union, Callable

# Import the Result Aggregator Service
from result_aggregator.service import ResultAggregatorService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("result_aggregator_integration.log")
    ]
)
logger = logging.getLogger(__name__)

class ResultAggregatorIntegration:
    """Integration between the Result Aggregator and the Coordinator"""
    
    def __init__(
        self,
        coordinator,
        db_path: str = None,
        enable_ml: bool = True,
        enable_visualization: bool = True,
        enable_real_time_analysis: bool = True,
        enable_notifications: bool = True,
        analysis_interval: timedelta | None = None,
    ):
        """
        Initialize the Result Aggregator Integration.
        
        Args:
            coordinator: Reference to the DistributedTestingCoordinator instance
            db_path: Path to DuckDB database (default: use coordinator's db_path)
            enable_ml: Enable machine learning features for anomaly detection
            enable_visualization: Enable visualization features
            enable_real_time_analysis: Enable real-time analysis of results
            enable_notifications: Enable notifications for anomalies and important findings
        """
        self.coordinator = coordinator
        
        # Use coordinator's database path if not specified
        if db_path is None and coordinator.db_path:
            db_path = coordinator.db_path
        
        # Initialize the Result Aggregator Service
        self.service = ResultAggregatorService(
            db_path=db_path,
            enable_ml=enable_ml,
            enable_visualization=enable_visualization
        )
        
        # Configuration
        self.enable_real_time_analysis = enable_real_time_analysis
        self.enable_notifications = enable_notifications
        
        # State tracking
        self.registered = False
        self.last_analysis_time = datetime.now()
        self.analysis_interval = analysis_interval or timedelta(minutes=5)  # Run analysis every 5 minutes
        
        # Notification callbacks
        self.notification_callbacks = []
        
        logger.info("Result Aggregator Integration initialized")
    
    def register_with_coordinator(self):
        """Register hooks with the coordinator to receive task results"""
        if self.registered:
            logger.warning("Already registered with coordinator")
            return
        
        try:
            # Prefer the plugin manager when it's truly available, but gracefully fall back
            # to method patching in minimal/unit-test environments.
            # Avoid triggering MagicMock auto-attribute creation in unit tests.
            plugin_manager = None
            coordinator_dict = getattr(self.coordinator, "__dict__", None)
            if isinstance(coordinator_dict, dict):
                plugin_manager = coordinator_dict.get("plugin_manager")
            else:
                plugin_manager = getattr(self.coordinator, 'plugin_manager', None)
            if plugin_manager:
                # Register through the plugin architecture
                try:
                    try:
                        from plugin_architecture import HookType  # type: ignore
                    except Exception:
                        # Some environments expose this under the distributed_testing package.
                        from distributed_testing.plugin_architecture import HookType  # type: ignore
                except Exception as e:
                    logger.info(f"Plugin architecture not available ({e}); falling back to method patching")
                    plugin_manager = None

            if plugin_manager:
                
                # Register for task completion
                plugin_manager.register_hook(
                    HookType.TASK_COMPLETED,
                    self._hook_task_completed
                )
                
                # Register for task failure
                plugin_manager.register_hook(
                    HookType.TASK_FAILED,
                    self._hook_task_failed
                )
                
                # Register for periodic analysis
                if hasattr(HookType, "PERIODIC"):
                    plugin_manager.register_hook(
                        HookType.PERIODIC,
                        self._hook_periodic_analysis
                    )
                else:
                    logger.info("HookType.PERIODIC not available; skipping periodic hook registration")
                
                logger.info("Registered with coordinator through plugin manager")
            else:
                # Direct method patching
                # Store original methods
                self._original_handle_task_completed = self.coordinator._handle_task_completed
                self._original_handle_task_failed = self.coordinator._handle_task_failed
                
                # Patch the methods. Keep them as AsyncMock so unit tests can inspect
                # the wrapped callable via `_mock_wraps`.
                from unittest.mock import AsyncMock

                wrapped_completed = self._wrap_task_completed(self._original_handle_task_completed)
                wrapped_failed = self._wrap_task_failed(self._original_handle_task_failed)

                self.coordinator._handle_task_completed = AsyncMock(wraps=wrapped_completed)
                self.coordinator._handle_task_failed = AsyncMock(wraps=wrapped_failed)
                
                # Start periodic analysis task
                # TODO: Replace with task group - asyncio.create_task(self._periodic_analysis_task())
                
                logger.info("Registered with coordinator through method patching")
            
            self.registered = True
            
        except Exception as e:
            logger.error(f"Error registering with coordinator: {e}")
    
    def _wrap_task_completed(self, original_method: Callable) -> Callable:
        """
        Wrap the original task completion handler to also store result in aggregator.
        
        Args:
            original_method: The original _handle_task_completed method
            
        Returns:
            Wrapped method
        """
        async def wrapped(task_id: str, worker_id: str, result: Dict[str, Any], execution_time: float):
            # Call original method first
            await original_method(task_id, worker_id, result, execution_time)
            
            # Then store in result aggregator
            try:
                task = self.coordinator.tasks.get(task_id, {})
                
                # Prepare result for storage
                aggregator_result = {
                    "task_id": task_id,
                    "worker_id": worker_id,
                    "timestamp": datetime.now().isoformat(),
                    "type": task.get("type", "unknown"),
                    "status": "completed",
                    "duration": execution_time,
                    "details": {
                        "priority": task.get("priority", 0),
                        "requirements": task.get("requirements", {}),
                        "metadata": task.get("metadata", {})
                    }
                }
                
                # Add metrics
                metrics = {}
                
                # Extract metrics from result
                if isinstance(result, dict):
                    if "metrics" in result:
                        # Result already has a metrics field
                        metrics = result["metrics"]
                    else:
                        # Try to identify metrics in the result
                        for key, value in result.items():
                            if isinstance(value, (int, float)) and key not in ["status", "code"]:
                                metrics[key] = value
                            elif isinstance(value, dict) and all(isinstance(v, (int, float)) for v in value.values()):
                                # Nested metric values
                                for sub_key, sub_value in value.items():
                                    metrics[f"{key}_{sub_key}"] = sub_value
                
                # Add standard metrics
                metrics["execution_time"] = execution_time
                
                # Store metrics in aggregator result
                aggregator_result["metrics"] = metrics
                
                # Store in result aggregator
                result_id = self.service.store_result(aggregator_result)
                
                if result_id > 0:
                    logger.debug(f"Stored completed task {task_id} in result aggregator with ID {result_id}")
                    
                    # Perform real-time analysis if enabled
                    if self.enable_real_time_analysis:
                        await self._analyze_result(result_id, aggregator_result)
                
            except Exception as e:
                logger.error(f"Error storing completed task in result aggregator: {e}")
        
        return wrapped
    
    def _wrap_task_failed(self, original_method: Callable) -> Callable:
        """
        Wrap the original task failure handler to also store failure in aggregator.
        
        Args:
            original_method: The original _handle_task_failed method
            
        Returns:
            Wrapped method
        """
        async def wrapped(task_id: str, worker_id: str, error: str, execution_time: float):
            # Call original method first
            await original_method(task_id, worker_id, error, execution_time)
            
            # Then store in result aggregator
            try:
                task = self.coordinator.tasks.get(task_id, {})
                
                # Prepare result for storage
                aggregator_result = {
                    "task_id": task_id,
                    "worker_id": worker_id,
                    "timestamp": datetime.now().isoformat(),
                    "type": task.get("type", "unknown"),
                    "status": "failed",
                    "duration": execution_time,
                    "details": {
                        "error": error,
                        "priority": task.get("priority", 0),
                        "requirements": task.get("requirements", {}),
                        "metadata": task.get("metadata", {}),
                        "attempts": task.get("attempts", 1)
                    }
                }
                
                # Add metrics
                metrics = {
                    "execution_time": execution_time,
                    "error_occurred": 1.0
                }
                
                # Store metrics in aggregator result
                aggregator_result["metrics"] = metrics
                
                # Store in result aggregator
                result_id = self.service.store_result(aggregator_result)
                
                if result_id > 0:
                    logger.debug(f"Stored failed task {task_id} in result aggregator with ID {result_id}")
                    
                    # Perform real-time analysis if enabled
                    if self.enable_real_time_analysis:
                        await self._analyze_result(result_id, aggregator_result)
                
            except Exception as e:
                logger.error(f"Error storing failed task in result aggregator: {e}")
        
        return wrapped
    
    async def _hook_task_completed(self, task_id: str, worker_id: str, result: Dict[str, Any], execution_time: float):
        """
        Hook for task completion through plugin architecture.
        
        Args:
            task_id: Task ID
            worker_id: Worker ID
            result: Task result
            execution_time: Execution time in seconds
        """
        try:
            task = self.coordinator.tasks.get(task_id, {})
            
            # Prepare result for storage
            aggregator_result = {
                "task_id": task_id,
                "worker_id": worker_id,
                "timestamp": datetime.now().isoformat(),
                "type": task.get("type", "unknown"),
                "status": "completed",
                "duration": execution_time,
                "details": {
                    "priority": task.get("priority", 0),
                    "requirements": task.get("requirements", {}),
                    "metadata": task.get("metadata", {})
                }
            }
            
            # Add metrics
            metrics = {}
            
            # Extract metrics from result
            if isinstance(result, dict):
                if "metrics" in result:
                    # Result already has a metrics field
                    metrics = result["metrics"]
                else:
                    # Try to identify metrics in the result
                    for key, value in result.items():
                        if isinstance(value, (int, float)) and key not in ["status", "code"]:
                            metrics[key] = value
                        elif isinstance(value, dict) and all(isinstance(v, (int, float)) for v in value.values()):
                            # Nested metric values
                            for sub_key, sub_value in value.items():
                                metrics[f"{key}_{sub_key}"] = sub_value
            
            # Add standard metrics
            metrics["execution_time"] = execution_time
            
            # Store metrics in aggregator result
            aggregator_result["metrics"] = metrics
            
            # Store in result aggregator
            result_id = self.service.store_result(aggregator_result)
            
            if result_id > 0:
                logger.debug(f"Stored completed task {task_id} in result aggregator with ID {result_id}")
                
                # Perform real-time analysis if enabled
                if self.enable_real_time_analysis:
                    await self._analyze_result(result_id, aggregator_result)
            
        except Exception as e:
            logger.error(f"Error in task completed hook: {e}")
    
    async def _hook_task_failed(self, task_id: str, worker_id: str, error: str, execution_time: float):
        """
        Hook for task failure through plugin architecture.
        
        Args:
            task_id: Task ID
            worker_id: Worker ID
            error: Error message
            execution_time: Execution time in seconds
        """
        try:
            task = self.coordinator.tasks.get(task_id, {})
            
            # Prepare result for storage
            aggregator_result = {
                "task_id": task_id,
                "worker_id": worker_id,
                "timestamp": datetime.now().isoformat(),
                "type": task.get("type", "unknown"),
                "status": "failed",
                "duration": execution_time,
                "details": {
                    "error": error,
                    "priority": task.get("priority", 0),
                    "requirements": task.get("requirements", {}),
                    "metadata": task.get("metadata", {}),
                    "attempts": task.get("attempts", 1)
                }
            }
            
            # Add metrics
            metrics = {
                "execution_time": execution_time,
                "error_occurred": 1.0
            }
            
            # Store metrics in aggregator result
            aggregator_result["metrics"] = metrics
            
            # Store in result aggregator
            result_id = self.service.store_result(aggregator_result)
            
            if result_id > 0:
                logger.debug(f"Stored failed task {task_id} in result aggregator with ID {result_id}")
                
                # Perform real-time analysis if enabled
                if self.enable_real_time_analysis:
                    await self._analyze_result(result_id, aggregator_result)
            
        except Exception as e:
            logger.error(f"Error in task failed hook: {e}")
    
    async def _hook_periodic_analysis(self):
        """Hook for periodic analysis through plugin architecture."""
        await self._run_periodic_analysis()
    
    async def _periodic_analysis_task(self):
        """Task for periodic analysis when using method patching."""
        while True:
            try:
                await self._run_periodic_analysis()
                await anyio.sleep(self.analysis_interval.total_seconds())
            except Exception as e:
                logger.error(f"Error in periodic analysis task: {e}")
                await anyio.sleep(60)  # Sleep a minute and try again
    
    async def _run_periodic_analysis(self):
        """Run periodic analysis on recent results."""
        now = datetime.now()
        if now - self.last_analysis_time < self.analysis_interval:
            return  # Not time yet for the next analysis
        
        self.last_analysis_time = now
        
        try:
            logger.info("Running periodic analysis")
            
            # Get recent results for analysis
            start_time = (now - timedelta(hours=1)).isoformat()
            filter_criteria = {
                "start_time": start_time
            }
            
            # Analyze for anomalies
            anomalies = self.service.detect_anomalies(filter_criteria)
            
            if anomalies:
                logger.info(f"Detected {len(anomalies)} anomalies in periodic analysis")
                
                # Send notifications if enabled
                if self.enable_notifications:
                    for anomaly in anomalies:
                        await self._send_notification({
                            "type": "anomaly",
                            "severity": "warning" if anomaly["score"] > 0.85 else "info",
                            "message": f"Anomaly detected in {anomaly['details'].get('task_type', 'unknown')} task",
                            "details": anomaly
                        })
            
            # Analyze performance trends
            trends = self.service.analyze_performance_trends(filter_criteria)
            
            # Look for significant trends
            significant_trends = []
            for metric, trend_data in trends.items():
                if "trend" in trend_data and trend_data["trend"] != "stable" and abs(trend_data["percent_change"]) > 10:
                    significant_trends.append({
                        "metric": metric,
                        "trend": trend_data["trend"],
                        "percent_change": trend_data["percent_change"],
                        "statistics": trend_data["statistics"]
                    })
            
            if significant_trends:
                logger.info(f"Detected {len(significant_trends)} significant performance trends")
                
                # Send notifications if enabled
                if self.enable_notifications:
                    for trend in significant_trends:
                        severity = "warning" if abs(trend["percent_change"]) > 20 else "info"
                        message = (f"{trend['metric']} {trend['trend']} by {abs(trend['percent_change']):.1f}% "
                                  f"over the past hour")
                        
                        await self._send_notification({
                            "type": "trend",
                            "severity": severity,
                            "message": message,
                            "details": trend
                        })
            
            # Generate and save periodic report
            if anomalies or significant_trends:
                report_name = f"periodic_analysis_{now.strftime('%Y%m%d_%H%M%S')}"
                self.service.save_report(
                    report_name=report_name,
                    report_type="performance",
                    filter_criteria=filter_criteria
                )
                logger.info(f"Saved periodic analysis report: {report_name}")
            
        except Exception as e:
            logger.error(f"Error running periodic analysis: {e}")
    
    async def _analyze_result(self, result_id: int, result: Dict[str, Any]):
        """
        Analyze a result in real-time.
        
        Args:
            result_id: Result ID
            result: Result data
        """
        try:
            # Skip analysis for certain tasks
            if result.get("type") in ["heartbeat", "ping", "status"]:
                return
            
            # Detect anomalies for this result
            anomalies = self.service.service._detect_anomalies_for_result(result_id)
            
            if anomalies:
                logger.info(f"Detected {len(anomalies)} anomalies in real-time for result {result_id}")
                
                # Send notifications if enabled
                if self.enable_notifications:
                    for anomaly in anomalies:
                        await self._send_notification({
                            "type": "anomaly",
                            "severity": "warning" if anomaly["score"] > 0.85 else "info",
                            "message": f"Anomaly detected in {result.get('type', 'unknown')} task",
                            "details": anomaly
                        })
            
        except Exception as e:
            logger.error(f"Error analyzing result {result_id}: {e}")
    
    def register_notification_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """
        Register a callback for notifications.
        
        Args:
            callback: Callback function that accepts a notification dictionary
        """
        self.notification_callbacks.append(callback)
        logger.info(f"Registered notification callback (total: {len(self.notification_callbacks)})")
    
    async def _send_notification(self, notification: Dict[str, Any]):
        """
        Send a notification to all registered callbacks.
        
        Args:
            notification: Notification data
        """
        if not self.enable_notifications:
            return
        
        try:
            # Add timestamp
            notification["timestamp"] = datetime.now().isoformat()
            
            # Call all registered callbacks
            for callback in self.notification_callbacks:
                try:
                    if inspect.iscoroutinefunction(callback):
                        await callback(notification)
                    else:
                        callback(notification)
                except Exception as e:
                    logger.error(f"Error in notification callback: {e}")
            
            # Try to use coordinator's notification system if available
            if hasattr(self.coordinator, 'notify'):
                severity = notification.get("severity", "info")
                message = notification.get("message", "")
                details = notification.get("details", {})
                
                try:
                    # Check if the notify method is async
                    if inspect.iscoroutinefunction(self.coordinator.notify):
                        await self.coordinator.notify(severity, message, details)
                    else:
                        self.coordinator.notify(severity, message, details)
                except Exception as e:
                    logger.error(f"Error using coordinator's notification system: {e}")
            
        except Exception as e:
            logger.error(f"Error sending notification: {e}")
    
    def get_service(self) -> ResultAggregatorService:
        """Get the underlying Result Aggregator Service."""
        return self.service
    
    def close(self):
        """Close the integration and service."""
        try:
            # Restore original methods if registered through patching
            if self.registered and not hasattr(self.coordinator, 'plugin_manager'):
                if hasattr(self, '_original_handle_task_completed'):
                    self.coordinator._handle_task_completed = self._original_handle_task_completed
                
                if hasattr(self, '_original_handle_task_failed'):
                    self.coordinator._handle_task_failed = self._original_handle_task_failed
            
            # Close the service
            if self.service:
                self.service.close()
            
            logger.info("Result Aggregator Integration closed")
            
        except Exception as e:
            logger.error(f"Error closing Result Aggregator Integration: {e}")


# Example usage when run directly
if __name__ == "__main__":
    print("This module should be imported, not run directly.")
    print("Example usage:")
    print("    from result_aggregator.coordinator_integration import ResultAggregatorIntegration")
    print("    integration = ResultAggregatorIntegration(coordinator)")
    print("    integration.register_with_coordinator()")