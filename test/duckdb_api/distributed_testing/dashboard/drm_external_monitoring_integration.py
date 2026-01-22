#!/usr/bin/env python3
"""
DRM External Monitoring Integration

This module provides integration between the Dynamic Resource Management (DRM)
system and external monitoring tools like Prometheus and Grafana. It exports
DRM metrics in Prometheus format and provides Grafana dashboard configurations.

Key components:
1. PrometheusExporter: Exports DRM metrics in Prometheus format
2. GrafanaDashboardGenerator: Generates Grafana dashboard JSON configurations
3. ExternalMonitoringBridge: Connects DRM dashboard to external monitoring systems

Usage:
    # Import the module
    from duckdb_api.distributed_testing.dashboard.drm_external_monitoring_integration import ExternalMonitoringBridge
    
    # Create a bridge instance
    bridge = ExternalMonitoringBridge(
        drm_dashboard=drm_dashboard_instance,
        metrics_port=9100,
        export_grafana_dashboard=True
    )
    
    # Start the bridge
    bridge.start()
"""

import os
import time
import json
import logging
import threading
import datetime
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import prometheus_client
try:
    import prometheus_client
    from prometheus_client import Gauge, Counter, Summary, Histogram, REGISTRY
    from prometheus_client.exposition import start_http_server
    PROMETHEUS_AVAILABLE = True
except ImportError:
    logger.warning("prometheus_client not available. Prometheus export features will be limited.")
    PROMETHEUS_AVAILABLE = False

# Try to import requests for Grafana API communication
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    logger.warning("requests not available. Grafana API features will be limited.")
    REQUESTS_AVAILABLE = False


class PrometheusExporter:
    """
    Exports DRM metrics in Prometheus format.
    
    Creates and maintains Prometheus metrics corresponding to DRM metrics,
    making them available for scraping by Prometheus servers.
    """
    
    def __init__(self, port: int = 9100):
        """
        Initialize the Prometheus exporter.
        
        Args:
            port: Port to expose metrics on (default: 9100)
        """
        self.port = port
        self.server = None
        self.running = False
        self.metrics = {}
        
        # Initialize metrics if Prometheus is available
        if PROMETHEUS_AVAILABLE:
            self._initialize_metrics()
        
        logger.info(f"Prometheus exporter initialized (port: {port})")
    
    def _initialize_metrics(self):
        """Initialize Prometheus metrics."""
        # Resource utilization metrics
        self.metrics["cpu_utilization"] = Gauge(
            "drm_cpu_utilization_percent",
            "CPU utilization percentage across all workers"
        )
        
        self.metrics["memory_utilization"] = Gauge(
            "drm_memory_utilization_percent",
            "Memory utilization percentage across all workers"
        )
        
        self.metrics["gpu_utilization"] = Gauge(
            "drm_gpu_utilization_percent",
            "GPU utilization percentage across all workers"
        )
        
        # Worker metrics
        self.metrics["worker_count"] = Gauge(
            "drm_worker_count",
            "Total number of workers"
        )
        
        self.metrics["active_workers"] = Gauge(
            "drm_active_workers",
            "Number of active workers"
        )
        
        # Task metrics
        self.metrics["active_tasks"] = Gauge(
            "drm_active_tasks",
            "Number of active tasks"
        )
        
        self.metrics["pending_tasks"] = Gauge(
            "drm_pending_tasks",
            "Number of pending tasks"
        )
        
        self.metrics["completed_tasks"] = Counter(
            "drm_completed_tasks_total",
            "Total number of completed tasks"
        )
        
        # Performance metrics
        self.metrics["task_throughput"] = Gauge(
            "drm_task_throughput_per_second",
            "Task throughput in tasks per second"
        )
        
        self.metrics["allocation_time"] = Gauge(
            "drm_allocation_time_milliseconds",
            "Resource allocation time in milliseconds"
        )
        
        self.metrics["resource_efficiency"] = Gauge(
            "drm_resource_efficiency_percent",
            "Resource efficiency percentage"
        )
        
        # Scaling metrics
        self.metrics["scaling_operations"] = Counter(
            "drm_scaling_operations_total",
            "Total number of scaling operations",
            ["direction", "reason"]
        )
        
        # Alert metrics
        self.metrics["alerts"] = Counter(
            "drm_alerts_total",
            "Total number of alerts",
            ["level", "source"]
        )
        
        # Worker-specific metrics using labels
        self.metrics["worker_cpu_utilization"] = Gauge(
            "drm_worker_cpu_utilization_percent",
            "Worker CPU utilization percentage",
            ["worker_id"]
        )
        
        self.metrics["worker_memory_utilization"] = Gauge(
            "drm_worker_memory_utilization_percent",
            "Worker memory utilization percentage",
            ["worker_id"]
        )
        
        self.metrics["worker_gpu_utilization"] = Gauge(
            "drm_worker_gpu_utilization_percent",
            "Worker GPU utilization percentage",
            ["worker_id"]
        )
        
        self.metrics["worker_tasks"] = Gauge(
            "drm_worker_tasks",
            "Number of tasks on worker",
            ["worker_id"]
        )
    
    def start(self):
        """Start the metrics server."""
        if not PROMETHEUS_AVAILABLE:
            logger.error("Cannot start Prometheus exporter: prometheus_client not available")
            return False
        
        if self.running:
            logger.warning("Prometheus exporter already running")
            return True
        
        try:
            # Start metrics server
            start_http_server(self.port)
            self.running = True
            logger.info(f"Prometheus metrics server running on port {self.port}")
            return True
        except Exception as e:
            logger.error(f"Error starting Prometheus metrics server: {e}")
            return False
    
    def stop(self):
        """Stop the metrics server."""
        if not self.running:
            return
        
        # There's no built-in way to stop the server, so we just mark it as not running
        self.running = False
        logger.info("Prometheus metrics server marked as stopped")
    
    def update_metrics(self, drm_metrics: Dict[str, Any]):
        """
        Update Prometheus metrics from DRM metrics.
        
        Args:
            drm_metrics: Dictionary of DRM metrics
        """
        if not PROMETHEUS_AVAILABLE or not self.running:
            return
        
        try:
            # Update resource utilization metrics
            if "resource_metrics" in drm_metrics:
                resource_metrics = drm_metrics["resource_metrics"]
                
                if resource_metrics.get("cpu_utilization"):
                    self.metrics["cpu_utilization"].set(resource_metrics["cpu_utilization"][-1])
                
                if resource_metrics.get("memory_utilization"):
                    self.metrics["memory_utilization"].set(resource_metrics["memory_utilization"][-1])
                
                if resource_metrics.get("gpu_utilization"):
                    self.metrics["gpu_utilization"].set(resource_metrics["gpu_utilization"][-1])
                
                if resource_metrics.get("worker_count"):
                    self.metrics["worker_count"].set(resource_metrics["worker_count"][-1])
                
                if resource_metrics.get("active_tasks"):
                    self.metrics["active_tasks"].set(resource_metrics["active_tasks"][-1])
                
                if resource_metrics.get("pending_tasks"):
                    self.metrics["pending_tasks"].set(resource_metrics["pending_tasks"][-1])
            
            # Update worker-specific metrics
            if "worker_metrics" in drm_metrics:
                worker_metrics = drm_metrics["worker_metrics"]
                
                # Count active workers
                active_workers = 0
                
                for worker_id, metrics in worker_metrics.items():
                    if metrics.get("cpu_utilization"):
                        self.metrics["worker_cpu_utilization"].labels(worker_id=worker_id).set(
                            metrics["cpu_utilization"][-1]
                        )
                        active_workers += 1
                    
                    if metrics.get("memory_utilization"):
                        self.metrics["worker_memory_utilization"].labels(worker_id=worker_id).set(
                            metrics["memory_utilization"][-1]
                        )
                    
                    if metrics.get("gpu_utilization"):
                        self.metrics["worker_gpu_utilization"].labels(worker_id=worker_id).set(
                            metrics["gpu_utilization"][-1]
                        )
                    
                    if metrics.get("tasks"):
                        self.metrics["worker_tasks"].labels(worker_id=worker_id).set(
                            metrics["tasks"][-1]
                        )
                
                self.metrics["active_workers"].set(active_workers)
            
            # Update performance metrics
            if "performance_metrics" in drm_metrics:
                performance_metrics = drm_metrics["performance_metrics"]
                
                if performance_metrics.get("task_throughput") and performance_metrics["task_throughput"]:
                    self.metrics["task_throughput"].set(
                        performance_metrics["task_throughput"][-1]["value"]
                    )
                
                if performance_metrics.get("allocation_time") and performance_metrics["allocation_time"]:
                    self.metrics["allocation_time"].set(
                        performance_metrics["allocation_time"][-1]["value"]
                    )
                
                if performance_metrics.get("resource_efficiency") and performance_metrics["resource_efficiency"]:
                    self.metrics["resource_efficiency"].set(
                        performance_metrics["resource_efficiency"][-1]["value"]
                    )
            
            # Update scaling metrics from new scaling decisions
            if "scaling_decisions" in drm_metrics and drm_metrics["scaling_decisions"]:
                latest_decision = drm_metrics["scaling_decisions"][-1]
                
                # Increment counter for scaling operation
                self.metrics["scaling_operations"].labels(
                    direction=latest_decision["action"],
                    reason=latest_decision["reason"]
                ).inc()
            
            # Update alert metrics from new alerts
            if "alerts" in drm_metrics and drm_metrics["alerts"]:
                latest_alert = drm_metrics["alerts"][-1]
                
                # Increment counter for alert
                self.metrics["alerts"].labels(
                    level=latest_alert["level"],
                    source=latest_alert["source"]
                ).inc()
            
            logger.debug("Prometheus metrics updated")
            
        except Exception as e:
            logger.error(f"Error updating Prometheus metrics: {e}")


class GrafanaDashboardGenerator:
    """
    Generates Grafana dashboard configurations for DRM metrics.
    
    Creates JSON dashboard configurations that can be imported into Grafana
    for visualizing DRM metrics with advanced panels and alerts.
    """
    
    def __init__(self, prometheus_url: str = "http://localhost:9090"):
        """
        Initialize the Grafana dashboard generator.
        
        Args:
            prometheus_url: URL of the Prometheus server (default: http://localhost:9090)
        """
        self.prometheus_url = prometheus_url
        self.output_dir = None
        
        logger.info(f"Grafana dashboard generator initialized (Prometheus URL: {prometheus_url})")
    
    def set_output_directory(self, output_dir: str):
        """
        Set the output directory for dashboard files.
        
        Args:
            output_dir: Directory to save dashboard files
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Grafana dashboard output directory set to: {output_dir}")
    
    def generate_drm_dashboard(self) -> Dict[str, Any]:
        """
        Generate a Grafana dashboard configuration for DRM metrics.
        
        Returns:
            Dashboard configuration as dictionary
        """
        # Base dashboard configuration
        dashboard = {
            "annotations": {
                "list": [
                    {
                        "builtIn": 1,
                        "datasource": "-- Grafana --",
                        "enable": True,
                        "hide": True,
                        "iconColor": "rgba(0, 211, 255, 1)",
                        "name": "Annotations & Alerts",
                        "type": "dashboard"
                    }
                ]
            },
            "editable": True,
            "gnetId": None,
            "graphTooltip": 0,
            "id": None,
            "links": [],
            "panels": [],
            "refresh": "10s",
            "schemaVersion": 27,
            "style": "dark",
            "tags": ["drm", "dynamic-resource-management"],
            "templating": {
                "list": []
            },
            "time": {
                "from": "now-1h",
                "to": "now"
            },
            "timepicker": {},
            "timezone": "",
            "title": "DRM Real-Time Performance Dashboard",
            "uid": "drm-performance",
            "version": 1
        }
        
        # Add worker variable template
        dashboard["templating"]["list"].append({
            "allValue": None,
            "current": {
                "selected": False,
                "text": "All",
                "value": "$__all"
            },
            "datasource": "Prometheus",
            "definition": "label_values(drm_worker_cpu_utilization_percent, worker_id)",
            "description": None,
            "error": None,
            "hide": 0,
            "includeAll": True,
            "label": "Worker",
            "multi": True,
            "name": "worker",
            "options": [],
            "query": {
                "query": "label_values(drm_worker_cpu_utilization_percent, worker_id)",
                "refId": "StandardVariableQuery"
            },
            "refresh": 2,
            "regex": "",
            "skipUrlSync": False,
            "sort": 1,
            "tagValuesQuery": "",
            "tags": [],
            "tagsQuery": "",
            "type": "query",
            "useTags": False
        })
        
        # Add panel: System Overview
        panel_id = 1
        dashboard["panels"].append({
            "datasource": "Prometheus",
            "fieldConfig": {
                "defaults": {
                    "color": {
                        "mode": "thresholds"
                    },
                    "mappings": [],
                    "thresholds": {
                        "mode": "absolute",
                        "steps": [
                            {
                                "color": "green",
                                "value": None
                            },
                            {
                                "color": "red",
                                "value": 80
                            }
                        ]
                    },
                    "unit": "none"
                },
                "overrides": []
            },
            "gridPos": {
                "h": 8,
                "w": 24,
                "x": 0,
                "y": 0
            },
            "id": panel_id,
            "options": {
                "colorMode": "value",
                "graphMode": "area",
                "justifyMode": "auto",
                "orientation": "auto",
                "reduceOptions": {
                    "calcs": [
                        "lastNotNull"
                    ],
                    "fields": "",
                    "values": False
                },
                "text": {},
                "textMode": "auto"
            },
            "pluginVersion": "7.5.3",
            "targets": [
                {
                    "expr": "drm_worker_count",
                    "interval": "",
                    "legendFormat": "Workers",
                    "refId": "A"
                },
                {
                    "expr": "drm_active_tasks",
                    "interval": "",
                    "legendFormat": "Active Tasks",
                    "refId": "B"
                },
                {
                    "expr": "drm_pending_tasks",
                    "interval": "",
                    "legendFormat": "Pending Tasks",
                    "refId": "C"
                }
            ],
            "title": "System Overview",
            "type": "stat"
        })
        panel_id += 1
        
        # Add panel: Resource Utilization
        dashboard["panels"].append({
            "aliasColors": {},
            "bars": False,
            "dashLength": 10,
            "dashes": False,
            "datasource": "Prometheus",
            "fieldConfig": {
                "defaults": {
                    "unit": "percent"
                },
                "overrides": []
            },
            "fill": 1,
            "fillGradient": 0,
            "gridPos": {
                "h": 9,
                "w": 12,
                "x": 0,
                "y": 8
            },
            "hiddenSeries": False,
            "id": panel_id,
            "legend": {
                "avg": False,
                "current": True,
                "max": False,
                "min": False,
                "show": True,
                "total": False,
                "values": True
            },
            "lines": True,
            "linewidth": 1,
            "nullPointMode": "null",
            "options": {
                "alertThreshold": True
            },
            "percentage": False,
            "pluginVersion": "7.5.3",
            "pointradius": 2,
            "points": False,
            "renderer": "flot",
            "seriesOverrides": [],
            "spaceLength": 10,
            "stack": False,
            "steppedLine": False,
            "targets": [
                {
                    "expr": "drm_cpu_utilization_percent",
                    "interval": "",
                    "legendFormat": "CPU",
                    "refId": "A"
                },
                {
                    "expr": "drm_memory_utilization_percent",
                    "interval": "",
                    "legendFormat": "Memory",
                    "refId": "B"
                },
                {
                    "expr": "drm_gpu_utilization_percent",
                    "interval": "",
                    "legendFormat": "GPU",
                    "refId": "C"
                }
            ],
            "thresholds": [],
            "timeRegions": [],
            "title": "Resource Utilization",
            "tooltip": {
                "shared": True,
                "sort": 0,
                "value_type": "individual"
            },
            "type": "graph",
            "xaxis": {
                "buckets": None,
                "mode": "time",
                "name": None,
                "show": True,
                "values": []
            },
            "yaxes": [
                {
                    "format": "percent",
                    "label": None,
                    "logBase": 1,
                    "max": "100",
                    "min": "0",
                    "show": True
                },
                {
                    "format": "short",
                    "label": None,
                    "logBase": 1,
                    "max": None,
                    "min": None,
                    "show": True
                }
            ],
            "yaxis": {
                "align": False,
                "alignLevel": None
            }
        })
        panel_id += 1
        
        # Add panel: Performance Metrics
        dashboard["panels"].append({
            "aliasColors": {},
            "bars": False,
            "dashLength": 10,
            "dashes": False,
            "datasource": "Prometheus",
            "fieldConfig": {
                "defaults": {},
                "overrides": []
            },
            "fill": 1,
            "fillGradient": 0,
            "gridPos": {
                "h": 9,
                "w": 12,
                "x": 12,
                "y": 8
            },
            "hiddenSeries": False,
            "id": panel_id,
            "legend": {
                "avg": False,
                "current": True,
                "max": False,
                "min": False,
                "show": True,
                "total": False,
                "values": True
            },
            "lines": True,
            "linewidth": 1,
            "nullPointMode": "null",
            "options": {
                "alertThreshold": True
            },
            "percentage": False,
            "pluginVersion": "7.5.3",
            "pointradius": 2,
            "points": False,
            "renderer": "flot",
            "seriesOverrides": [
                {
                    "alias": "Task Throughput",
                    "yaxis": 1
                },
                {
                    "alias": "Allocation Time",
                    "yaxis": 2
                }
            ],
            "spaceLength": 10,
            "stack": False,
            "steppedLine": False,
            "targets": [
                {
                    "expr": "drm_task_throughput_per_second",
                    "interval": "",
                    "legendFormat": "Task Throughput",
                    "refId": "A"
                },
                {
                    "expr": "drm_allocation_time_milliseconds",
                    "interval": "",
                    "legendFormat": "Allocation Time",
                    "refId": "B"
                }
            ],
            "thresholds": [],
            "timeRegions": [],
            "title": "Performance Metrics",
            "tooltip": {
                "shared": True,
                "sort": 0,
                "value_type": "individual"
            },
            "type": "graph",
            "xaxis": {
                "buckets": None,
                "mode": "time",
                "name": None,
                "show": True,
                "values": []
            },
            "yaxes": [
                {
                    "format": "short",
                    "label": "Tasks/s",
                    "logBase": 1,
                    "max": None,
                    "min": "0",
                    "show": True
                },
                {
                    "format": "ms",
                    "label": "Allocation Time",
                    "logBase": 1,
                    "max": None,
                    "min": "0",
                    "show": True
                }
            ],
            "yaxis": {
                "align": False,
                "alignLevel": None
            }
        })
        panel_id += 1
        
        # Add panel: Worker Metrics
        dashboard["panels"].append({
            "aliasColors": {},
            "bars": False,
            "dashLength": 10,
            "dashes": False,
            "datasource": "Prometheus",
            "fieldConfig": {
                "defaults": {
                    "unit": "percent"
                },
                "overrides": []
            },
            "fill": 1,
            "fillGradient": 0,
            "gridPos": {
                "h": 9,
                "w": 24,
                "x": 0,
                "y": 17
            },
            "hiddenSeries": False,
            "id": panel_id,
            "legend": {
                "avg": False,
                "current": True,
                "max": False,
                "min": False,
                "show": True,
                "total": False,
                "values": True
            },
            "lines": True,
            "linewidth": 1,
            "nullPointMode": "null",
            "options": {
                "alertThreshold": True
            },
            "percentage": False,
            "pluginVersion": "7.5.3",
            "pointradius": 2,
            "points": False,
            "renderer": "flot",
            "seriesOverrides": [],
            "spaceLength": 10,
            "stack": False,
            "steppedLine": False,
            "targets": [
                {
                    "expr": "drm_worker_cpu_utilization_percent{worker_id=~\"$worker\"}",
                    "interval": "",
                    "legendFormat": "CPU - {{worker_id}}",
                    "refId": "A"
                },
                {
                    "expr": "drm_worker_memory_utilization_percent{worker_id=~\"$worker\"}",
                    "interval": "",
                    "legendFormat": "Memory - {{worker_id}}",
                    "refId": "B"
                },
                {
                    "expr": "drm_worker_gpu_utilization_percent{worker_id=~\"$worker\"}",
                    "interval": "",
                    "legendFormat": "GPU - {{worker_id}}",
                    "refId": "C"
                }
            ],
            "thresholds": [],
            "timeRegions": [],
            "title": "Worker Resource Utilization",
            "tooltip": {
                "shared": True,
                "sort": 0,
                "value_type": "individual"
            },
            "type": "graph",
            "xaxis": {
                "buckets": None,
                "mode": "time",
                "name": None,
                "show": True,
                "values": []
            },
            "yaxes": [
                {
                    "format": "percent",
                    "label": "Utilization",
                    "logBase": 1,
                    "max": "100",
                    "min": "0",
                    "show": True
                },
                {
                    "format": "short",
                    "label": None,
                    "logBase": 1,
                    "max": None,
                    "min": None,
                    "show": False
                }
            ],
            "yaxis": {
                "align": False,
                "alignLevel": None
            }
        })
        panel_id += 1
        
        # Add panel: Worker Tasks
        dashboard["panels"].append({
            "aliasColors": {},
            "bars": False,
            "dashLength": 10,
            "dashes": False,
            "datasource": "Prometheus",
            "fieldConfig": {
                "defaults": {},
                "overrides": []
            },
            "fill": 1,
            "fillGradient": 0,
            "gridPos": {
                "h": 8,
                "w": 24,
                "x": 0,
                "y": 26
            },
            "hiddenSeries": False,
            "id": panel_id,
            "legend": {
                "avg": False,
                "current": True,
                "max": False,
                "min": False,
                "show": True,
                "total": False,
                "values": True
            },
            "lines": True,
            "linewidth": 1,
            "nullPointMode": "null",
            "options": {
                "alertThreshold": True
            },
            "percentage": False,
            "pluginVersion": "7.5.3",
            "pointradius": 2,
            "points": False,
            "renderer": "flot",
            "seriesOverrides": [],
            "spaceLength": 10,
            "stack": False,
            "steppedLine": False,
            "targets": [
                {
                    "expr": "drm_worker_tasks{worker_id=~\"$worker\"}",
                    "interval": "",
                    "legendFormat": "Tasks - {{worker_id}}",
                    "refId": "A"
                }
            ],
            "thresholds": [],
            "timeRegions": [],
            "title": "Worker Tasks",
            "tooltip": {
                "shared": True,
                "sort": 0,
                "value_type": "individual"
            },
            "type": "graph",
            "xaxis": {
                "buckets": None,
                "mode": "time",
                "name": None,
                "show": True,
                "values": []
            },
            "yaxes": [
                {
                    "format": "short",
                    "label": "Tasks",
                    "logBase": 1,
                    "max": None,
                    "min": "0",
                    "show": True
                },
                {
                    "format": "short",
                    "label": None,
                    "logBase": 1,
                    "max": None,
                    "min": None,
                    "show": False
                }
            ],
            "yaxis": {
                "align": False,
                "alignLevel": None
            }
        })
        panel_id += 1
        
        # Add panel: Scaling Operations
        dashboard["panels"].append({
            "aliasColors": {},
            "bars": True,
            "dashLength": 10,
            "dashes": False,
            "datasource": "Prometheus",
            "fieldConfig": {
                "defaults": {},
                "overrides": []
            },
            "fill": 1,
            "fillGradient": 0,
            "gridPos": {
                "h": 8,
                "w": 12,
                "x": 0,
                "y": 34
            },
            "hiddenSeries": False,
            "id": panel_id,
            "legend": {
                "avg": False,
                "current": True,
                "max": False,
                "min": False,
                "show": True,
                "total": False,
                "values": True
            },
            "lines": False,
            "linewidth": 1,
            "nullPointMode": "null",
            "options": {
                "alertThreshold": True
            },
            "percentage": False,
            "pluginVersion": "7.5.3",
            "pointradius": 2,
            "points": False,
            "renderer": "flot",
            "seriesOverrides": [],
            "spaceLength": 10,
            "stack": True,
            "steppedLine": False,
            "targets": [
                {
                    "expr": "sum(rate(drm_scaling_operations_total[5m])) by (direction)",
                    "interval": "",
                    "legendFormat": "{{direction}}",
                    "refId": "A"
                }
            ],
            "thresholds": [],
            "timeRegions": [],
            "title": "Scaling Operations (5m rate)",
            "tooltip": {
                "shared": True,
                "sort": 0,
                "value_type": "individual"
            },
            "type": "graph",
            "xaxis": {
                "buckets": None,
                "mode": "time",
                "name": None,
                "show": True,
                "values": []
            },
            "yaxes": [
                {
                    "format": "short",
                    "label": "Operations/sec",
                    "logBase": 1,
                    "max": None,
                    "min": "0",
                    "show": True
                },
                {
                    "format": "short",
                    "label": None,
                    "logBase": 1,
                    "max": None,
                    "min": None,
                    "show": False
                }
            ],
            "yaxis": {
                "align": False,
                "alignLevel": None
            }
        })
        panel_id += 1
        
        # Add panel: Alerts
        dashboard["panels"].append({
            "aliasColors": {},
            "bars": True,
            "dashLength": 10,
            "dashes": False,
            "datasource": "Prometheus",
            "fieldConfig": {
                "defaults": {},
                "overrides": []
            },
            "fill": 1,
            "fillGradient": 0,
            "gridPos": {
                "h": 8,
                "w": 12,
                "x": 12,
                "y": 34
            },
            "hiddenSeries": False,
            "id": panel_id,
            "legend": {
                "avg": False,
                "current": True,
                "max": False,
                "min": False,
                "show": True,
                "total": False,
                "values": True
            },
            "lines": False,
            "linewidth": 1,
            "nullPointMode": "null",
            "options": {
                "alertThreshold": True
            },
            "percentage": False,
            "pluginVersion": "7.5.3",
            "pointradius": 2,
            "points": False,
            "renderer": "flot",
            "seriesOverrides": [],
            "spaceLength": 10,
            "stack": True,
            "steppedLine": False,
            "targets": [
                {
                    "expr": "sum(rate(drm_alerts_total[5m])) by (level)",
                    "interval": "",
                    "legendFormat": "{{level}}",
                    "refId": "A"
                }
            ],
            "thresholds": [],
            "timeRegions": [],
            "title": "Alerts (5m rate)",
            "tooltip": {
                "shared": True,
                "sort": 0,
                "value_type": "individual"
            },
            "type": "graph",
            "xaxis": {
                "buckets": None,
                "mode": "time",
                "name": None,
                "show": True,
                "values": []
            },
            "yaxes": [
                {
                    "format": "short",
                    "label": "Alerts/sec",
                    "logBase": 1,
                    "max": None,
                    "min": "0",
                    "show": True
                },
                {
                    "format": "short",
                    "label": None,
                    "logBase": 1,
                    "max": None,
                    "min": None,
                    "show": False
                }
            ],
            "yaxis": {
                "align": False,
                "alignLevel": None
            }
        })
        
        return dashboard
    
    def save_dashboard(self, dashboard: Dict[str, Any], filename: str = "drm_dashboard.json") -> str:
        """
        Save a dashboard configuration to file.
        
        Args:
            dashboard: Dashboard configuration
            filename: File name to save to
        
        Returns:
            Path to saved file
        """
        if not self.output_dir:
            raise ValueError("Output directory not set. Call set_output_directory() first.")
        
        file_path = os.path.join(self.output_dir, filename)
        
        try:
            with open(file_path, 'w') as f:
                json.dump(dashboard, f, indent=2)
            
            logger.info(f"Grafana dashboard saved to: {file_path}")
            return file_path
        except Exception as e:
            logger.error(f"Error saving Grafana dashboard: {e}")
            return None
    
    def upload_dashboard(self, dashboard: Dict[str, Any], grafana_url: str, api_key: str, folder_id: int = 0) -> bool:
        """
        Upload a dashboard configuration to Grafana.
        
        Args:
            dashboard: Dashboard configuration
            grafana_url: Grafana base URL
            api_key: Grafana API key
            folder_id: Folder ID to place dashboard in
        
        Returns:
            Whether upload was successful
        """
        if not REQUESTS_AVAILABLE:
            logger.error("Cannot upload dashboard: requests not available")
            return False
        
        try:
            # Prepare dashboard payload
            payload = {
                "dashboard": dashboard,
                "overwrite": True,
                "folderId": folder_id
            }
            
            # Prepare headers
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
            
            # Send request
            url = f"{grafana_url.rstrip('/')}/api/dashboards/db"
            response = requests.post(url, headers=headers, json=payload)
            
            # Check response
            if response.status_code == 200:
                logger.info(f"Dashboard uploaded to Grafana successfully: {response.json().get('url')}")
                return True
            else:
                logger.error(f"Failed to upload dashboard: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error uploading dashboard to Grafana: {e}")
            return False


class ExternalMonitoringBridge:
    """
    Bridge between DRM dashboard and external monitoring systems.
    
    Connects the DRM dashboard to external monitoring systems like Prometheus and Grafana,
    enabling metrics export and integration with existing monitoring infrastructure.
    """
    
    def __init__(
        self,
        drm_dashboard=None,
        metrics_port: int = 9100,
        prometheus_url: str = "http://localhost:9090",
        grafana_url: str = "http://localhost:3000",
        grafana_api_key: str = None,
        export_grafana_dashboard: bool = True,
        output_dir: str = None
    ):
        """
        Initialize the external monitoring bridge.
        
        Args:
            drm_dashboard: DRMRealTimeDashboard instance
            metrics_port: Port to expose Prometheus metrics on
            prometheus_url: URL of the Prometheus server
            grafana_url: URL of the Grafana server
            grafana_api_key: Grafana API key for dashboard upload
            export_grafana_dashboard: Whether to export Grafana dashboard
            output_dir: Directory to save dashboard files
        """
        self.drm_dashboard = drm_dashboard
        self.prometheus_url = prometheus_url
        self.grafana_url = grafana_url
        self.grafana_api_key = grafana_api_key
        self.export_grafana_dashboard = export_grafana_dashboard
        
        # Set output directory
        if output_dir:
            self.output_dir = output_dir
        elif drm_dashboard and hasattr(drm_dashboard, "output_dir"):
            self.output_dir = os.path.join(drm_dashboard.output_dir, "external_monitoring")
        else:
            self.output_dir = "external_monitoring"
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize Prometheus exporter
        self.prometheus_exporter = PrometheusExporter(port=metrics_port)
        
        # Initialize Grafana dashboard generator
        self.grafana_generator = GrafanaDashboardGenerator(prometheus_url=prometheus_url)
        self.grafana_generator.set_output_directory(self.output_dir)
        
        # Background thread for metric updates
        self.update_thread = None
        self.update_stop_event = threading.Event()
        self.running = False
        
        logger.info(f"External monitoring bridge initialized (metrics port: {metrics_port})")
    
    def start(self, update_interval: int = 5):
        """
        Start the external monitoring bridge.
        
        Args:
            update_interval: Metrics update interval in seconds
        
        Returns:
            Whether start was successful
        """
        if self.running:
            logger.warning("External monitoring bridge already running")
            return True
        
        try:
            # Start Prometheus exporter
            if not self.prometheus_exporter.start():
                logger.error("Failed to start Prometheus exporter")
                return False
            
            # Export Grafana dashboard if requested
            if self.export_grafana_dashboard:
                self._export_grafana_dashboard()
            
            # Start metrics update thread if DRM dashboard is available
            if self.drm_dashboard:
                self.update_stop_event.clear()
                self.update_thread = threading.Thread(
                    target=self._update_metrics_loop,
                    args=(update_interval,),
                    daemon=True
                )
                self.update_thread.start()
            
            self.running = True
            logger.info("External monitoring bridge started")
            return True
            
        except Exception as e:
            logger.error(f"Error starting external monitoring bridge: {e}")
            self.stop()
            return False
    
    def stop(self):
        """Stop the external monitoring bridge."""
        if not self.running:
            return
        
        # Stop metrics update thread
        if self.update_thread and self.update_thread.is_alive():
            self.update_stop_event.set()
            self.update_thread.join(timeout=5.0)
        
        # Stop Prometheus exporter
        self.prometheus_exporter.stop()
        
        self.running = False
        logger.info("External monitoring bridge stopped")
    
    def _update_metrics_loop(self, update_interval: int):
        """
        Background thread for updating metrics.
        
        Args:
            update_interval: Update interval in seconds
        """
        while not self.update_stop_event.is_set():
            try:
                # Get current metrics from DRM dashboard
                metrics = self._get_dashboard_metrics()
                
                # Update Prometheus metrics
                self.prometheus_exporter.update_metrics(metrics)
                
            except Exception as e:
                logger.error(f"Error updating metrics: {e}")
            
            # Wait for next update
            self.update_stop_event.wait(update_interval)
    
    def _get_dashboard_metrics(self) -> Dict[str, Any]:
        """
        Get metrics from DRM dashboard.
        
        Returns:
            Dictionary of metrics
        """
        if not self.drm_dashboard:
            return {}
        
        # Get dashboard metrics
        metrics = {
            "resource_metrics": self.drm_dashboard.resource_metrics,
            "worker_metrics": self.drm_dashboard.worker_metrics,
            "performance_metrics": self.drm_dashboard.performance_metrics,
            "scaling_decisions": self.drm_dashboard.scaling_decisions,
            "alerts": self.drm_dashboard.alerts
        }
        
        return metrics
    
    def _export_grafana_dashboard(self):
        """Export Grafana dashboard configuration."""
        try:
            # Generate dashboard configuration
            dashboard = self.grafana_generator.generate_drm_dashboard()
            
            # Save dashboard to file
            file_path = self.grafana_generator.save_dashboard(dashboard)
            
            logger.info(f"Grafana dashboard exported to: {file_path}")
            
            # Upload dashboard if API key is provided
            if self.grafana_api_key:
                self.grafana_generator.upload_dashboard(
                    dashboard=dashboard,
                    grafana_url=self.grafana_url,
                    api_key=self.grafana_api_key
                )
            
        except Exception as e:
            logger.error(f"Error exporting Grafana dashboard: {e}")
    
    def get_prometheus_url(self) -> str:
        """
        Get the URL for accessing Prometheus metrics.
        
        Returns:
            URL for Prometheus metrics
        """
        return f"http://localhost:{self.prometheus_exporter.port}/metrics"
    
    def get_grafana_dashboard_path(self) -> str:
        """
        Get the path to the exported Grafana dashboard.
        
        Returns:
            Path to Grafana dashboard JSON file
        """
        return os.path.join(self.output_dir, "drm_dashboard.json")
    
    def integrate_with_drm_dashboard(self):
        """
        Integrate with the DRM dashboard.
        
        Adds external monitoring links to the DRM dashboard.
        """
        if not self.drm_dashboard or not hasattr(self.drm_dashboard, "dashboard_app"):
            logger.warning("DRM dashboard not available for integration")
            return
        
        try:
            # Import required modules
            import dash_bootstrap_components as dbc
            from dash import html
            
            # Get dashboard app
            app = self.drm_dashboard.dashboard_app
            
            # Add external monitoring links to the navbar
            if hasattr(app, "layout") and "navbar" in str(app.layout):
                # This is a simplistic approach that assumes the navbar structure exists
                # In a real implementation, you'd want to modify the layout more carefully
                prometheus_url = self.get_prometheus_url()
                grafana_url = self.grafana_url
                
                external_links = dbc.DropdownMenu(
                    children=[
                        dbc.DropdownMenuItem("Prometheus Metrics", href=prometheus_url, target="_blank"),
                        dbc.DropdownMenuItem("Grafana Dashboard", href=grafana_url, target="_blank"),
                    ],
                    nav=True,
                    in_navbar=True,
                    label="External Monitoring",
                )
                
                # This is a hook point that would need to be implemented in the DRM dashboard
                # to allow external components to add navbar items
                if hasattr(self.drm_dashboard, "add_navbar_component"):
                    self.drm_dashboard.add_navbar_component(external_links)
                
                logger.info("Integrated with DRM dashboard navbar")
            
        except Exception as e:
            logger.error(f"Error integrating with DRM dashboard: {e}")
    
    def get_metrics_integration_guide(self) -> str:
        """
        Get a guide for integrating DRM metrics with external monitoring.
        
        Returns:
            Text guide for integration
        """
        prometheus_url = self.get_prometheus_url()
        grafana_dashboard_path = self.get_grafana_dashboard_path()
        
        guide = f"""
# DRM External Monitoring Integration Guide

This guide explains how to integrate the DRM system with external monitoring tools like Prometheus and Grafana.

## Prometheus Integration

The DRM system exports metrics in Prometheus format on the following endpoint:

```
{prometheus_url}
```

Add the following scrape configuration to your Prometheus configuration:

```yaml
scrape_configs:
  - job_name: 'drm'
    scrape_interval: 15s
    static_configs:
      - targets: ['localhost:{self.prometheus_exporter.port}']
```

## Grafana Integration

A pre-configured Grafana dashboard is available at:

```
{grafana_dashboard_path}
```

To import this dashboard into Grafana:

1. Open Grafana web interface
2. Go to Dashboards > Import
3. Upload the JSON file or paste its contents
4. Select your Prometheus data source
5. Click Import

## Available Metrics

The following metrics are available:

### Resource Metrics
- `drm_cpu_utilization_percent`: CPU utilization percentage
- `drm_memory_utilization_percent`: Memory utilization percentage
- `drm_gpu_utilization_percent`: GPU utilization percentage

### Worker Metrics
- `drm_worker_count`: Total number of workers
- `drm_active_workers`: Number of active workers
- `drm_worker_cpu_utilization_percent`: Worker CPU utilization
- `drm_worker_memory_utilization_percent`: Worker memory utilization
- `drm_worker_gpu_utilization_percent`: Worker GPU utilization
- `drm_worker_tasks`: Number of tasks on worker

### Task Metrics
- `drm_active_tasks`: Number of active tasks
- `drm_pending_tasks`: Number of pending tasks
- `drm_completed_tasks_total`: Total number of completed tasks

### Performance Metrics
- `drm_task_throughput_per_second`: Task throughput
- `drm_allocation_time_milliseconds`: Resource allocation time
- `drm_resource_efficiency_percent`: Resource efficiency

### Scaling Metrics
- `drm_scaling_operations_total`: Number of scaling operations

### Alert Metrics
- `drm_alerts_total`: Number of alerts
"""
        
        return guide

def main():
    """Command-line entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="DRM External Monitoring Bridge")
    parser.add_argument("--metrics-port", type=int, default=9100, help="Port to expose Prometheus metrics on")
    parser.add_argument("--prometheus-url", default="http://localhost:9090", help="URL of the Prometheus server")
    parser.add_argument("--grafana-url", default="http://localhost:3000", help="URL of the Grafana server")
    parser.add_argument("--grafana-api-key", help="Grafana API key for dashboard upload")
    parser.add_argument("--output-dir", help="Directory to save dashboard files")
    parser.add_argument("--export-only", action="store_true", help="Only export dashboard, don't start metrics server")
    parser.add_argument("--update-interval", type=int, default=5, help="Metrics update interval in seconds")
    
    args = parser.parse_args()
    
    try:
        # Import DRM dashboard for integration
        from duckdb_api.distributed_testing.dashboard.drm_real_time_dashboard import DRMRealTimeDashboard
        from duckdb_api.distributed_testing.testing.mock_drm import MockDynamicResourceManager
        
        # Create mock DRM for testing
        logger.info("Creating mock DRM for testing")
        mock_drm = MockDynamicResourceManager()
        
        # Create DRM dashboard
        logger.info("Initializing DRM dashboard")
        dashboard = DRMRealTimeDashboard(
            dynamic_resource_manager=mock_drm,
            port=8085,
            update_interval=args.update_interval,
            debug=False
        )
        
        # Start dashboard in background if not export-only
        if not args.export_only:
            logger.info("Starting DRM dashboard in background")
            dashboard.start_in_background()
        
        # Create external monitoring bridge
        logger.info("Initializing external monitoring bridge")
        bridge = ExternalMonitoringBridge(
            drm_dashboard=dashboard,
            metrics_port=args.metrics_port,
            prometheus_url=args.prometheus_url,
            grafana_url=args.grafana_url,
            grafana_api_key=args.grafana_api_key,
            output_dir=args.output_dir
        )
        
        # Start bridge if not export-only
        if not args.export_only:
            logger.info("Starting external monitoring bridge")
            bridge.start(update_interval=args.update_interval)
            
            # Print integration information
            prometheus_url = bridge.get_prometheus_url()
            grafana_dashboard_path = bridge.get_grafana_dashboard_path()
            
            print(f"\nDRM External Monitoring Bridge started successfully!")
            print(f"Prometheus metrics URL: {prometheus_url}")
            print(f"Grafana dashboard file: {grafana_dashboard_path}\n")
            
            print("Press Ctrl+C to stop...")
            
            try:
                # Keep running until interrupted
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\nStopping...")
                
                # Stop bridge and dashboard
                bridge.stop()
                if hasattr(dashboard, "stop"):
                    dashboard.stop()
                
                print("Stopped.")
        else:
            # Export Grafana dashboard
            logger.info("Exporting Grafana dashboard")
            bridge._export_grafana_dashboard()
            
            # Print export information
            grafana_dashboard_path = bridge.get_grafana_dashboard_path()
            print(f"\nGrafana dashboard exported to: {grafana_dashboard_path}")
            
    except ImportError as e:
        logger.error(f"Import error: {e}")
        print(f"Error: {e}")
        print("Make sure the DRM dashboard and its dependencies are installed.")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    main()