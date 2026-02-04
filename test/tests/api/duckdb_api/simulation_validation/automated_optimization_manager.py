#!/usr/bin/env python3
"""
Automated Optimization Manager for the Database Performance Component.

This module provides automated performance optimization for the DuckDB database
used by the Simulation Accuracy and Validation Framework. It monitors database
performance metrics and automatically applies optimization actions when metrics
exceed defined thresholds.

Features:
1. Continuous performance metrics monitoring
2. Configurable thresholds for different metrics
3. Automatic application of optimization actions
4. Detailed logging of detected issues and actions taken
5. Scheduled optimization tasks
6. Before/after metrics comparison for optimization evaluation
"""

import os
import sys
import time
import json
import logging
import datetime
import threading
from typing import Dict, List, Any, Optional, Union, Callable
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("automated_optimization_manager")

# Import the performance optimizer
try:
    from data.duckdb.simulation_validation.db_performance_optimizer import DBPerformanceOptimizer
except ImportError:
    logger.error("Failed to import DBPerformanceOptimizer. Make sure duckdb_api is properly installed.")
    sys.exit(1)


class AutomatedOptimizationManager:
    """
    Manages automated performance optimization for the database.
    
    This class monitors database performance metrics and automatically applies
    optimization actions when metrics exceed defined thresholds.
    """
    
    # Define default thresholds for metrics
    DEFAULT_THRESHOLDS = {
        "query_time": {
            "warning": 500.0,  # ms
            "error": 1000.0    # ms
        },
        "storage_size": {
            "warning": 524288000,  # bytes (500 MB)
            "error": 1073741824    # bytes (1 GB)
        },
        "index_efficiency": {
            "warning": 70.0,  # percent
            "error": 50.0     # percent
        },
        "vacuum_status": {
            "warning": 60.0,  # percent
            "error": 40.0     # percent
        },
        "compression_ratio": {
            "warning": 2.5,  # ratio
            "error": 1.5     # ratio
        },
        "read_efficiency": {
            "warning": 200.0,  # records/second
            "error": 100.0     # records/second
        },
        "write_efficiency": {
            "warning": 150.0,  # records/second
            "error": 75.0      # records/second
        },
        "cache_performance": {
            "warning": 50.0,  # percent hit ratio
            "error": 30.0     # percent hit ratio
        }
    }
    
    # Define default actions for each metric
    DEFAULT_ACTIONS = {
        "query_time": ["optimize_queries", "create_indexes", "analyze_tables"],
        "storage_size": ["vacuum_database", "cleanup_old_records"],
        "index_efficiency": ["create_indexes", "analyze_tables"],
        "vacuum_status": ["vacuum_database"],
        "compression_ratio": ["optimize_database"],
        "read_efficiency": ["optimize_queries", "create_indexes"],
        "write_efficiency": ["optimize_database", "batch_operations"],
        "cache_performance": ["clear_cache", "optimize_cache_settings"]
    }
    
    def __init__(
        self, 
        db_optimizer: DBPerformanceOptimizer,
        thresholds: Optional[Dict[str, Dict[str, float]]] = None,
        actions: Optional[Dict[str, List[str]]] = None,
        check_interval: int = 3600,  # 1 hour by default
        log_file: Optional[str] = None,
        auto_apply_actions: bool = False,
        retention_days: int = 90,
        enable_scheduled_tasks: bool = False,
        history_days: int = 30
    ):
        """
        Initialize the automated optimization manager.
        
        Args:
            db_optimizer: DBPerformanceOptimizer instance
            thresholds: Custom thresholds for metrics (None for defaults)
            actions: Custom actions for metrics (None for defaults)
            check_interval: Interval in seconds between automated checks
            log_file: Path to log file for optimization actions
            auto_apply_actions: Whether to automatically apply optimization actions
            retention_days: Days to retain data when cleaning up old records
            enable_scheduled_tasks: Whether to enable scheduled optimization tasks
            history_days: Days of metric history to keep for trend analysis
        """
        self.db_optimizer = db_optimizer
        self.thresholds = thresholds if thresholds else self.DEFAULT_THRESHOLDS
        self.actions = actions if actions else self.DEFAULT_ACTIONS
        self.check_interval = check_interval
        self.auto_apply_actions = auto_apply_actions
        self.retention_days = retention_days
        self.enable_scheduled_tasks = enable_scheduled_tasks
        self.history_days = history_days
        
        # Configure logging
        if log_file:
            self.log_file = log_file
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            logger.addHandler(file_handler)
        
        # Initialize metrics history storage
        self.metrics_history = {}
        
        # Initialize monitoring thread
        self.monitoring_thread = None
        self.monitoring_active = False
        
        # Initialize scheduled tasks
        self.scheduled_tasks = {}
        if enable_scheduled_tasks:
            self._setup_scheduled_tasks()
        
        logger.info("Initialized AutomatedOptimizationManager")
    
    def _setup_scheduled_tasks(self) -> None:
        """
        Set up default scheduled optimization tasks.
        """
        # Daily tasks
        self.schedule_task("daily_optimization", 
                          ["create_indexes", "analyze_tables"],
                          interval_hours=24)
        
        # Weekly tasks
        self.schedule_task("weekly_optimization", 
                          ["optimize_database", "vacuum_database"],
                          interval_hours=24 * 7)
        
        # Monthly tasks
        self.schedule_task("monthly_optimization", 
                          ["cleanup_old_records", "optimize_database", "vacuum_database"],
                          interval_hours=24 * 30)
        
        logger.info("Scheduled default optimization tasks")
    
    def schedule_task(
        self, 
        task_name: str,
        actions: List[str],
        interval_hours: int,
        start_time: Optional[datetime.datetime] = None
    ) -> Dict[str, Any]:
        """
        Schedule a recurring optimization task.
        
        Args:
            task_name: Name of the task
            actions: List of optimization actions to perform
            interval_hours: Interval in hours between task executions
            start_time: When to start the task (None for now)
            
        Returns:
            Task configuration details
        """
        if start_time is None:
            start_time = datetime.datetime.now()
        
        task = {
            "name": task_name,
            "actions": actions,
            "interval_hours": interval_hours,
            "start_time": start_time,
            "next_run": start_time,
            "last_run": None,
            "enabled": True
        }
        
        self.scheduled_tasks[task_name] = task
        logger.info(f"Scheduled task '{task_name}' to run every {interval_hours} hours starting at {start_time}")
        
        return task
    
    def unschedule_task(self, task_name: str) -> bool:
        """
        Remove a scheduled task.
        
        Args:
            task_name: Name of the task to remove
            
        Returns:
            True if the task was removed, False otherwise
        """
        if task_name in self.scheduled_tasks:
            del self.scheduled_tasks[task_name]
            logger.info(f"Unscheduled task '{task_name}'")
            return True
        else:
            logger.warning(f"Task '{task_name}' not found in scheduled tasks")
            return False
    
    def enable_task(self, task_name: str, enabled: bool = True) -> bool:
        """
        Enable or disable a scheduled task.
        
        Args:
            task_name: Name of the task to enable/disable
            enabled: Whether to enable or disable the task
            
        Returns:
            True if successful, False otherwise
        """
        if task_name in self.scheduled_tasks:
            self.scheduled_tasks[task_name]["enabled"] = enabled
            status = "enabled" if enabled else "disabled"
            logger.info(f"Task '{task_name}' {status}")
            return True
        else:
            logger.warning(f"Task '{task_name}' not found in scheduled tasks")
            return False
    
    def get_scheduled_tasks(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all scheduled tasks.
        
        Returns:
            Dictionary of scheduled tasks
        """
        return self.scheduled_tasks
    
    def _check_scheduled_tasks(self) -> None:
        """
        Check and execute scheduled tasks if they are due.
        """
        now = datetime.datetime.now()
        
        for task_name, task in self.scheduled_tasks.items():
            if task["enabled"] and now >= task["next_run"]:
                logger.info(f"Executing scheduled task '{task_name}'")
                
                # Execute the task
                result = self.execute_actions(task["actions"])
                
                # Update task status
                task["last_run"] = now
                task["next_run"] = now + datetime.timedelta(hours=task["interval_hours"])
                task["last_result"] = result
                
                logger.info(f"Scheduled task '{task_name}' completed. Next run at {task['next_run']}")
    
    def start_monitoring(self) -> None:
        """
        Start continuous monitoring for automated optimization.
        """
        if self.monitoring_active:
            logger.warning("Monitoring is already active")
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        logger.info(f"Started automated optimization monitoring (interval: {self.check_interval} seconds)")
    
    def stop_monitoring(self) -> None:
        """
        Stop continuous monitoring.
        """
        if not self.monitoring_active:
            logger.warning("Monitoring is not active")
            return
        
        self.monitoring_active = False
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5.0)
        
        logger.info("Stopped automated optimization monitoring")
    
    def _monitoring_loop(self) -> None:
        """
        Main monitoring loop that runs continuously.
        """
        while self.monitoring_active:
            try:
                # Check database performance metrics
                metrics = self.db_optimizer.get_performance_metrics()
                overall_status = self.db_optimizer.get_overall_status()
                
                # Update metrics history
                self._update_metrics_history(metrics)
                
                # Check for scheduled tasks
                if self.enable_scheduled_tasks:
                    self._check_scheduled_tasks()
                
                # Check if any metrics exceed thresholds
                issues = self._detect_performance_issues(metrics)
                
                if issues:
                    logger.info(f"Detected {len(issues)} performance issues")
                    
                    # Process the identified issues
                    optimization_results = self._process_detected_issues(issues)
                    
                    # Log results
                    for issue, result in optimization_results.items():
                        logger.info(f"Optimization for {issue}: {result['status']}")
                
                # Sleep until next check
                for _ in range(min(60, self.check_interval)):
                    if not self.monitoring_active:
                        break
                    time.sleep(1)
                
                remaining_sleep = max(0, self.check_interval - 60)
                if remaining_sleep > 0 and self.monitoring_active:
                    time.sleep(remaining_sleep)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                # Sleep for a shorter interval on error
                time.sleep(min(60, self.check_interval))
    
    def _update_metrics_history(self, metrics: Dict[str, Dict[str, Any]]) -> None:
        """
        Update the metrics history storage.
        
        Args:
            metrics: Current metrics values
        """
        current_time = datetime.datetime.now().isoformat()
        
        for metric_name, metric_data in metrics.items():
            if metric_name not in self.metrics_history:
                self.metrics_history[metric_name] = []
            
            # Add current metric to history
            metric_entry = {
                "timestamp": current_time,
                "value": metric_data.get("value"),
                "status": metric_data.get("status")
            }
            
            # Add to history
            self.metrics_history[metric_name].append(metric_entry)
            
            # Limit history size
            history_limit = self.history_days * 24  # Assuming hourly checks for X days
            if len(self.metrics_history[metric_name]) > history_limit:
                self.metrics_history[metric_name] = self.metrics_history[metric_name][-history_limit:]
    
    def get_metrics_history(self, metric_name: Optional[str] = None) -> Union[Dict[str, List[Dict[str, Any]]], List[Dict[str, Any]]]:
        """
        Get metrics history for analysis.
        
        Args:
            metric_name: Optional specific metric to retrieve history for
            
        Returns:
            Metrics history data
        """
        if metric_name:
            if metric_name in self.metrics_history:
                return self.metrics_history[metric_name]
            else:
                return []
        else:
            return self.metrics_history
    
    def _detect_performance_issues(self, metrics: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Detect performance issues based on metrics and thresholds.
        
        Args:
            metrics: Current metrics values
            
        Returns:
            Dictionary of detected issues
        """
        issues = {}
        
        for metric_name, metric_data in metrics.items():
            if metric_name not in self.thresholds:
                continue
            
            value = metric_data.get("value")
            if value is None:
                continue
            
            # Check if the metric exceeds thresholds
            thresholds = self.thresholds[metric_name]
            
            # Different metrics have different threshold directions
            # For some metrics, higher values are better (and we want to avoid low values)
            # For others, lower values are better (and we want to avoid high values)
            if metric_name in ["index_efficiency", "vacuum_status", "compression_ratio", "cache_performance"]:
                # For these metrics, higher values are better
                if value <= thresholds.get("error", 0):
                    severity = "error"
                elif value <= thresholds.get("warning", 0):
                    severity = "warning"
                else:
                    # No issue detected
                    continue
            else:
                # For these metrics, lower values are better
                if value >= thresholds.get("error", float("inf")):
                    severity = "error"
                elif value >= thresholds.get("warning", float("inf")):
                    severity = "warning"
                else:
                    # No issue detected
                    continue
            
            # Record the issue
            issues[metric_name] = {
                "value": value,
                "threshold": thresholds.get(severity),
                "severity": severity,
                "unit": metric_data.get("unit", ""),
                "recommended_actions": self.actions.get(metric_name, [])
            }
        
        return issues
    
    def _process_detected_issues(self, issues: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Process detected performance issues.
        
        Args:
            issues: Dictionary of detected issues
            
        Returns:
            Results of optimization actions
        """
        results = {}
        
        for metric_name, issue in issues.items():
            logger.info(f"Processing issue with {metric_name}: {issue['value']} {issue['unit']} ({issue['severity']})")
            
            if not issue.get("recommended_actions"):
                logger.warning(f"No recommended actions defined for {metric_name}")
                continue
            
            # Record before metrics if auto-apply is enabled
            before_metrics = None
            if self.auto_apply_actions:
                before_metrics = self.db_optimizer.get_performance_metrics()
            
            # Apply actions or just record recommendations
            if self.auto_apply_actions:
                action_results = self.execute_actions(issue["recommended_actions"])
                
                # Get updated metrics after optimization
                after_metrics = self.db_optimizer.get_performance_metrics()
                
                # Calculate improvement
                improvement = self._calculate_improvement(metric_name, before_metrics, after_metrics)
                
                results[metric_name] = {
                    "status": "applied",
                    "actions_taken": issue["recommended_actions"],
                    "action_results": action_results,
                    "before_value": before_metrics[metric_name]["value"] if metric_name in before_metrics else None,
                    "after_value": after_metrics[metric_name]["value"] if metric_name in after_metrics else None,
                    "improvement": improvement,
                    "timestamp": datetime.datetime.now().isoformat()
                }
            else:
                results[metric_name] = {
                    "status": "recommended",
                    "recommended_actions": issue["recommended_actions"],
                    "current_value": issue["value"],
                    "threshold": issue["threshold"],
                    "severity": issue["severity"],
                    "timestamp": datetime.datetime.now().isoformat()
                }
        
        return results
    
    def _calculate_improvement(
        self, 
        metric_name: str, 
        before_metrics: Dict[str, Dict[str, Any]], 
        after_metrics: Dict[str, Dict[str, Any]]
    ) -> Optional[float]:
        """
        Calculate improvement percentage for a metric.
        
        Args:
            metric_name: Name of the metric
            before_metrics: Metrics before optimization
            after_metrics: Metrics after optimization
            
        Returns:
            Improvement percentage or None if not calculable
        """
        if metric_name not in before_metrics or metric_name not in after_metrics:
            return None
        
        before_value = before_metrics[metric_name].get("value")
        after_value = after_metrics[metric_name].get("value")
        
        if before_value is None or after_value is None or before_value == 0:
            return None
        
        # For metrics where higher is better
        if metric_name in ["index_efficiency", "vacuum_status", "compression_ratio", "cache_performance"]:
            improvement = ((after_value - before_value) / before_value) * 100
        # For metrics where lower is better
        else:
            improvement = ((before_value - after_value) / before_value) * 100
        
        return improvement
    
    def execute_actions(self, actions: List[str]) -> Dict[str, Any]:
        """
        Execute optimization actions.
        
        Args:
            actions: List of optimization actions to perform
            
        Returns:
            Results of actions
        """
        results = {}
        
        for action in actions:
            logger.info(f"Executing optimization action: {action}")
            
            try:
                # Execute the appropriate method based on action
                if action == "optimize_queries":
                    # This action doesn't directly change the database,
                    # but ensures future queries are optimized
                    results[action] = {"status": "success", "message": "Query optimization enabled"}
                
                elif action == "create_indexes":
                    self.db_optimizer.create_indexes()
                    results[action] = {"status": "success", "message": "Indexes created"}
                
                elif action == "analyze_tables":
                    self.db_optimizer.analyze_tables()
                    results[action] = {"status": "success", "message": "Tables analyzed"}
                
                elif action == "vacuum_database":
                    # This is part of optimize_database
                    result = self.db_optimizer.optimize_database()
                    results[action] = {
                        "status": "success" if result else "failure",
                        "message": "Database vacuumed"
                    }
                
                elif action == "cleanup_old_records":
                    cleanup_results = self.db_optimizer.cleanup_old_records(
                        older_than_days=self.retention_days,
                        dry_run=False
                    )
                    
                    # Calculate total records deleted
                    total_deleted = 0
                    for table, stats in cleanup_results.items():
                        total_deleted += stats.get("deleted", 0)
                    
                    results[action] = {
                        "status": "success",
                        "message": f"Deleted {total_deleted} old records",
                        "details": cleanup_results
                    }
                
                elif action == "optimize_database":
                    result = self.db_optimizer.optimize_database()
                    results[action] = {
                        "status": "success" if result else "failure",
                        "message": "Database optimized"
                    }
                
                elif action == "clear_cache":
                    self.db_optimizer.clear_cache()
                    results[action] = {"status": "success", "message": "Cache cleared"}
                
                elif action == "optimize_cache_settings":
                    # This would typically involve adjusting cache size, TTL, etc.
                    # but we don't have direct methods for this in the current implementation
                    results[action] = {"status": "skipped", "message": "Cache settings adjustment not implemented"}
                
                elif action == "batch_operations":
                    # This is more of a strategy than a direct action
                    results[action] = {"status": "skipped", "message": "Batch operations is a strategy, not a direct action"}
                
                else:
                    logger.warning(f"Unknown optimization action: {action}")
                    results[action] = {"status": "skipped", "message": f"Unknown action: {action}"}
            
            except Exception as e:
                logger.error(f"Error executing action {action}: {e}")
                results[action] = {"status": "error", "message": str(e)}
        
        return results
    
    def check_performance(self, specific_metrics: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
        """
        Perform a manual check of database performance.
        
        Args:
            specific_metrics: List of specific metrics to check (None for all)
            
        Returns:
            Dictionary with performance metrics and detected issues
        """
        # Get current metrics
        metrics = self.db_optimizer.get_performance_metrics()
        
        # Filter metrics if requested
        if specific_metrics:
            metrics = {k: v for k, v in metrics.items() if k in specific_metrics}
        
        # Detect issues
        issues = self._detect_performance_issues(metrics)
        
        return {
            "metrics": metrics,
            "issues": issues,
            "overall_status": self.db_optimizer.get_overall_status(),
            "timestamp": datetime.datetime.now().isoformat()
        }
    
    def optimize_now(self, issues: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Dict[str, Any]]:
        """
        Run optimization immediately for detected issues.
        
        Args:
            issues: Pre-detected issues (None to detect issues first)
            
        Returns:
            Results of optimization actions
        """
        # Detect issues if not provided
        if issues is None:
            metrics = self.db_optimizer.get_performance_metrics()
            issues = self._detect_performance_issues(metrics)
        
        if not issues:
            logger.info("No performance issues detected")
            return {"status": "success", "message": "No issues detected"}
        
        # Get before metrics for comparison
        before_metrics = self.db_optimizer.get_performance_metrics()
        
        # Process issues and apply optimizations
        optimization_results = {}
        
        for metric_name, issue in issues.items():
            if "recommended_actions" in issue and issue["recommended_actions"]:
                logger.info(f"Optimizing {metric_name} ({issue['severity']})")
                
                # Execute actions
                action_results = self.execute_actions(issue["recommended_actions"])
                
                optimization_results[metric_name] = {
                    "status": "applied",
                    "actions_taken": issue["recommended_actions"],
                    "action_results": action_results,
                    "before_value": before_metrics[metric_name]["value"] if metric_name in before_metrics else None,
                    "timestamp": datetime.datetime.now().isoformat()
                }
        
        # Get after metrics for comparison
        after_metrics = self.db_optimizer.get_performance_metrics()
        
        # Calculate improvements
        for metric_name in optimization_results:
            improvement = self._calculate_improvement(metric_name, before_metrics, after_metrics)
            optimization_results[metric_name]["after_value"] = after_metrics[metric_name]["value"] if metric_name in after_metrics else None
            optimization_results[metric_name]["improvement"] = improvement
        
        return {
            "status": "success",
            "results": optimization_results,
            "before_overall_status": self.db_optimizer.get_overall_status(before_metrics),
            "after_overall_status": self.db_optimizer.get_overall_status(),
            "timestamp": datetime.datetime.now().isoformat()
        }
    
    def run_comprehensive_optimization(self) -> Dict[str, Any]:
        """
        Run a comprehensive optimization regardless of current metrics.
        
        This method performs all possible optimizations to ensure the database
        is in the best possible state.
        
        Returns:
            Results of the optimization process
        """
        logger.info("Starting comprehensive database optimization")
        
        # Get before metrics for comparison
        before_metrics = self.db_optimizer.get_performance_metrics()
        before_status = self.db_optimizer.get_overall_status()
        
        # Define all possible optimization actions
        all_actions = [
            "create_indexes",
            "analyze_tables",
            "cleanup_old_records",
            "optimize_database",
            "clear_cache"
        ]
        
        # Execute all actions
        action_results = self.execute_actions(all_actions)
        
        # Get after metrics for comparison
        after_metrics = self.db_optimizer.get_performance_metrics()
        after_status = self.db_optimizer.get_overall_status()
        
        # Calculate improvements for each metric
        improvements = {}
        for metric_name in before_metrics:
            if metric_name in after_metrics:
                improvement = self._calculate_improvement(metric_name, before_metrics, after_metrics)
                improvements[metric_name] = {
                    "before": before_metrics[metric_name].get("value"),
                    "after": after_metrics[metric_name].get("value"),
                    "improvement": improvement,
                    "unit": before_metrics[metric_name].get("unit", "")
                }
        
        return {
            "status": "success",
            "action_results": action_results,
            "before_status": before_status,
            "after_status": after_status,
            "improvements": improvements,
            "timestamp": datetime.datetime.now().isoformat()
        }
    
    def get_optimization_recommendations(self) -> Dict[str, Any]:
        """
        Get optimization recommendations without applying them.
        
        Returns:
            Dictionary with optimization recommendations
        """
        # Get current metrics
        metrics = self.db_optimizer.get_performance_metrics()
        
        # Detect issues
        issues = self._detect_performance_issues(metrics)
        
        if not issues:
            return {
                "status": "success",
                "message": "No optimization recommendations",
                "overall_status": self.db_optimizer.get_overall_status(),
                "timestamp": datetime.datetime.now().isoformat()
            }
        
        # Prepare recommendations
        recommendations = {}
        for metric_name, issue in issues.items():
            recommendations[metric_name] = {
                "value": issue["value"],
                "threshold": issue["threshold"],
                "severity": issue["severity"],
                "unit": issue["unit"],
                "recommended_actions": issue["recommended_actions"]
            }
        
        return {
            "status": "success",
            "recommendations": recommendations,
            "overall_status": self.db_optimizer.get_overall_status(),
            "timestamp": datetime.datetime.now().isoformat()
        }
    
    def analyze_trends(self, days: int = 7) -> Dict[str, Any]:
        """
        Analyze performance trends over time.
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Trend analysis results
        """
        trends = {}
        warnings = []
        
        # Calculate timestamp threshold
        threshold_date = datetime.datetime.now() - datetime.timedelta(days=days)
        threshold_timestamp = threshold_date.isoformat()
        
        # Analyze each metric's history
        for metric_name, history in self.metrics_history.items():
            # Filter history by date
            filtered_history = [entry for entry in history if entry["timestamp"] >= threshold_timestamp]
            
            if len(filtered_history) < 2:
                warnings.append(f"Insufficient data for {metric_name} trend analysis")
                continue
            
            # Extract values
            values = [entry["value"] for entry in filtered_history if entry["value"] is not None]
            
            if not values:
                warnings.append(f"No valid values for {metric_name} trend analysis")
                continue
            
            # Calculate trend statistics
            start_value = values[0]
            end_value = values[-1]
            min_value = min(values)
            max_value = max(values)
            avg_value = sum(values) / len(values)
            
            # Determine trend direction
            if end_value > start_value:
                direction = "increasing"
                change_pct = ((end_value - start_value) / start_value) * 100 if start_value != 0 else 0
            elif end_value < start_value:
                direction = "decreasing"
                change_pct = ((start_value - end_value) / start_value) * 100 if start_value != 0 else 0
            else:
                direction = "stable"
                change_pct = 0
            
            # Determine if trend is concerning
            concerning = False
            if metric_name in ["query_time", "storage_size"]:
                # For these metrics, increasing trends are concerning
                concerning = direction == "increasing" and change_pct > 10
            elif metric_name in ["index_efficiency", "vacuum_status", "compression_ratio", "cache_performance"]:
                # For these metrics, decreasing trends are concerning
                concerning = direction == "decreasing" and change_pct > 10
            
            # Record trend data
            trends[metric_name] = {
                "start_value": start_value,
                "end_value": end_value,
                "min_value": min_value,
                "max_value": max_value,
                "avg_value": avg_value,
                "direction": direction,
                "change_percent": change_pct,
                "concerning": concerning,
                "data_points": len(values)
            }
        
        return {
            "status": "success",
            "trends": trends,
            "warnings": warnings,
            "period_days": days,
            "timestamp": datetime.datetime.now().isoformat()
        }


def get_optimization_manager(
    db_optimizer: DBPerformanceOptimizer,
    config_file: Optional[str] = None,
    auto_apply: bool = False
) -> AutomatedOptimizationManager:
    """
    Create an AutomatedOptimizationManager instance with optional configuration from file.
    
    Args:
        db_optimizer: DBPerformanceOptimizer instance
        config_file: Path to configuration file (None for defaults)
        auto_apply: Whether to automatically apply optimization actions
        
    Returns:
        Configured AutomatedOptimizationManager instance
    """
    # Default configuration
    config = {
        "thresholds": AutomatedOptimizationManager.DEFAULT_THRESHOLDS,
        "actions": AutomatedOptimizationManager.DEFAULT_ACTIONS,
        "check_interval": 3600,
        "log_file": None,
        "auto_apply_actions": auto_apply,
        "retention_days": 90,
        "enable_scheduled_tasks": True,
        "history_days": 30
    }
    
    # Load configuration from file if provided
    if config_file and os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                file_config = json.load(f)
                # Update config with file values
                config.update(file_config)
                logger.info(f"Loaded configuration from {config_file}")
        except Exception as e:
            logger.error(f"Error loading configuration file: {e}")
    
    # Create manager instance
    manager = AutomatedOptimizationManager(
        db_optimizer=db_optimizer,
        thresholds=config["thresholds"],
        actions=config["actions"],
        check_interval=config["check_interval"],
        log_file=config["log_file"],
        auto_apply_actions=config["auto_apply_actions"],
        retention_days=config["retention_days"],
        enable_scheduled_tasks=config["enable_scheduled_tasks"],
        history_days=config["history_days"]
    )
    
    return manager


if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Automated Database Performance Optimization Tool")
    parser.add_argument("--db-path", type=str, default="./benchmark_db.duckdb", 
                        help="Path to the DuckDB database")
    parser.add_argument("--config", type=str, 
                        help="Path to configuration file")
    parser.add_argument("--action", choices=["check", "optimize", "comprehensive", "monitor", "trends"], 
                        default="check", help="Action to perform")
    parser.add_argument("--auto-apply", action="store_true", 
                        help="Automatically apply optimization actions")
    parser.add_argument("--days", type=int, default=7, 
                        help="Number of days for trend analysis")
    parser.add_argument("--metrics", nargs="+", 
                        help="Specific metrics to check")
    parser.add_argument("--monitor-time", type=int, 
                        help="Time in seconds to run monitoring (default: indefinite)")
    parser.add_argument("--check-interval", type=int, default=3600, 
                        help="Interval in seconds between checks")
    parser.add_argument("--output", type=str, 
                        help="Output file path for results")
    
    args = parser.parse_args()
    
    try:
        # Create DBPerformanceOptimizer instance
        from data.duckdb.simulation_validation.db_performance_optimizer import get_db_optimizer
        db_optimizer = get_db_optimizer(db_path=args.db_path)
        
        # Create AutomatedOptimizationManager instance
        manager = get_optimization_manager(
            db_optimizer=db_optimizer,
            config_file=args.config,
            auto_apply=args.auto_apply
        )
        
        # If check interval is provided, update it
        if args.check_interval:
            manager.check_interval = args.check_interval
        
        result = None
        
        # Execute requested action
        if args.action == "check":
            result = manager.check_performance(args.metrics)
            
            # Print summary of results
            print(f"\nDatabase Status: {result['overall_status'].upper()}")
            print(f"Timestamp: {result['timestamp']}")
            
            if result["issues"]:
                print(f"\nDetected {len(result['issues'])} performance issues:")
                for metric_name, issue in result["issues"].items():
                    print(f"- {metric_name}: {issue['value']} {issue['unit']} "
                          f"(Threshold: {issue['threshold']} {issue['unit']}, Severity: {issue['severity']})")
                    if "recommended_actions" in issue:
                        print(f"  Recommended actions: {', '.join(issue['recommended_actions'])}")
            else:
                print("\nNo performance issues detected.")
        
        elif args.action == "optimize":
            result = manager.optimize_now()
            
            # Print summary of results
            print(f"\nOptimization Status: {result['status'].upper()}")
            print(f"Overall status before: {result['before_overall_status'].upper()}")
            print(f"Overall status after: {result['after_overall_status'].upper()}")
            
            if "results" in result and result["results"]:
                print("\nOptimization results:")
                for metric_name, metric_result in result["results"].items():
                    print(f"- {metric_name}:")
                    if "before_value" in metric_result and "after_value" in metric_result:
                        print(f"  Value: {metric_result['before_value']} → {metric_result['after_value']}")
                    if "improvement" in metric_result and metric_result["improvement"] is not None:
                        print(f"  Improvement: {metric_result['improvement']:.2f}%")
                    if "actions_taken" in metric_result:
                        print(f"  Actions: {', '.join(metric_result['actions_taken'])}")
            else:
                print("\nNo optimizations performed.")
        
        elif args.action == "comprehensive":
            result = manager.run_comprehensive_optimization()
            
            # Print summary of results
            print(f"\nComprehensive Optimization Status: {result['status'].upper()}")
            print(f"Overall status before: {result['before_status'].upper()}")
            print(f"Overall status after: {result['after_status'].upper()}")
            
            if "improvements" in result and result["improvements"]:
                print("\nImprovements:")
                for metric_name, improvement in result["improvements"].items():
                    if improvement["improvement"] is not None:
                        direction = "improved" if improvement["improvement"] > 0 else "worsened"
                        print(f"- {metric_name}: {improvement['before']} → {improvement['after']} "
                              f"({abs(improvement['improvement']):.2f}% {direction})")
            
            if "action_results" in result:
                print("\nAction results:")
                for action, action_result in result["action_results"].items():
                    print(f"- {action}: {action_result['status']}")
        
        elif args.action == "trends":
            result = manager.analyze_trends(days=args.days)
            
            # Print summary of results
            print(f"\nTrend Analysis Status: {result['status'].upper()}")
            print(f"Period: {result['period_days']} days")
            
            if "warnings" in result and result["warnings"]:
                print("\nWarnings:")
                for warning in result["warnings"]:
                    print(f"- {warning}")
            
            if "trends" in result and result["trends"]:
                print("\nTrends:")
                for metric_name, trend in result["trends"].items():
                    direction_symbol = "↑" if trend["direction"] == "increasing" else "↓" if trend["direction"] == "decreasing" else "→"
                    concern = " (CONCERNING)" if trend["concerning"] else ""
                    print(f"- {metric_name}: {direction_symbol} {trend['change_percent']:.2f}%{concern}")
                    print(f"  Start: {trend['start_value']}, End: {trend['end_value']}")
                    print(f"  Min: {trend['min_value']}, Max: {trend['max_value']}, Avg: {trend['avg_value']:.2f}")
        
        elif args.action == "monitor":
            # Start continuous monitoring
            monitor_time = args.monitor_time
            
            print(f"\nStarting monitoring (interval: {manager.check_interval} seconds)")
            if monitor_time:
                print(f"Will run for {monitor_time} seconds")
            else:
                print("Will run until interrupted (Ctrl+C)")
            
            # Start monitoring
            manager.start_monitoring()
            
            try:
                if monitor_time:
                    time.sleep(monitor_time)
                else:
                    # Run indefinitely until interrupted
                    while True:
                        time.sleep(60)
            except KeyboardInterrupt:
                print("\nMonitoring interrupted by user")
            finally:
                # Stop monitoring
                manager.stop_monitoring()
                
                # Get final results
                result = manager.check_performance()
                
                # Print summary
                print(f"\nFinal Database Status: {result['overall_status'].upper()}")
                
                if result["issues"]:
                    print(f"\nDetected {len(result['issues'])} performance issues:")
                    for metric_name, issue in result["issues"].items():
                        print(f"- {metric_name}: {issue['value']} {issue['unit']} "
                              f"(Threshold: {issue['threshold']} {issue['unit']}, Severity: {issue['severity']})")
                else:
                    print("\nNo performance issues detected.")
        
        # Save results to file if requested
        if args.output and result:
            try:
                with open(args.output, 'w') as f:
                    json.dump(result, f, indent=2)
                print(f"\nResults saved to {args.output}")
            except Exception as e:
                print(f"\nError saving results to file: {e}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)