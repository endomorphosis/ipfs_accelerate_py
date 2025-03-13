#!/usr/bin/env python3
"""
Prometheus Connector for Distributed Testing Framework

This module provides a connector for interacting with Prometheus's API and Push Gateway 
to query metrics data and submit metrics from the distributed testing framework.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from urllib.parse import quote

import aiohttp
import json

# Import the standardized interface
from distributed_testing.external_systems.api_interface import (
    ExternalSystemInterface,
    ConnectorCapabilities,
    ExternalSystemResult,
    ExternalSystemFactory
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PrometheusConnector(ExternalSystemInterface):
    """
    Connector for interacting with Prometheus API.
    
    This connector implements the standardized ExternalSystemInterface for Prometheus
    and provides methods for querying metrics and pushing metrics to Push Gateway.
    """
    
    def __init__(self):
        """
        Initialize the Prometheus connector.
        """
        self.prometheus_url = None
        self.pushgateway_url = None
        self.username = None
        self.password = None
        self.session = None
        self.use_basic_auth = False
        self.job_prefix = "distributed_testing"
        self.default_namespace = "distributed_testing"
        self.rate_limit_sleep = 1.0  # seconds to sleep when rate limited
        
        # Cache for commonly used data
        self.cache = {
            "metric_metadata": {},
            "targets": [],
            "alerts": []
        }
        
        # Capabilities
        self.capabilities = ConnectorCapabilities(
            supports_create=True,
            supports_update=True,
            supports_delete=True,
            supports_query=True,
            supports_batch_operations=True,
            supports_attachments=False,
            supports_comments=False,
            supports_custom_fields=False,
            supports_relationships=False,
            supports_history=True,
            item_types=["metric", "target", "alert", "rule"],
            query_operators=["=", "!=", ">", "<", ">=", "<="],
            max_batch_size=100,
            rate_limit=600,  # Common Prometheus API rate limit
            supports_instant_queries=True,
            supports_range_queries=True,
            supports_push_gateway=True
        )
    
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """
        Initialize the Prometheus connector with configuration.
        
        Args:
            config: Configuration dictionary containing:
                   - prometheus_url: Prometheus server URL
                   - pushgateway_url: Prometheus Push Gateway URL (optional)
                   - username: Basic auth username (optional)
                   - password: Basic auth password (optional)
                   - job_prefix: Job name prefix for Push Gateway (optional)
                   - default_namespace: Default metric namespace (optional)
            
        Returns:
            True if initialization succeeded
        """
        self.prometheus_url = config.get("prometheus_url")
        self.pushgateway_url = config.get("pushgateway_url")
        self.username = config.get("username")
        self.password = config.get("password")
        self.job_prefix = config.get("job_prefix", self.job_prefix)
        self.default_namespace = config.get("default_namespace", self.default_namespace)
        
        if not self.prometheus_url:
            logger.error("Prometheus URL is required")
            return False
        
        # Normalize URLs
        if self.prometheus_url.endswith("/"):
            self.prometheus_url = self.prometheus_url[:-1]
        
        if self.pushgateway_url and self.pushgateway_url.endswith("/"):
            self.pushgateway_url = self.pushgateway_url[:-1]
        
        # Check if using basic auth
        self.use_basic_auth = bool(self.username and self.password)
        
        logger.info(f"PrometheusConnector initialized for server {self.prometheus_url}")
        return True
    
    async def _ensure_session(self):
        """Ensure an aiohttp session exists with proper authentication."""
        if self.session is None:
            if self.use_basic_auth:
                auth = aiohttp.BasicAuth(self.username, self.password)
                
                self.session = aiohttp.ClientSession(
                    auth=auth,
                    headers={
                        "User-Agent": "DistributedTestingFramework",
                        "Content-Type": "application/json",
                        "Accept": "application/json"
                    }
                )
            else:
                self.session = aiohttp.ClientSession(
                    headers={
                        "User-Agent": "DistributedTestingFramework",
                        "Content-Type": "application/json",
                        "Accept": "application/json"
                    }
                )
    
    async def connect(self) -> bool:
        """
        Establish connection to Prometheus and validate accessibility.
        
        Returns:
            True if connection succeeded
        """
        await self._ensure_session()
        
        try:
            # Check connection by querying API status
            url = f"{self.prometheus_url}/api/v1/status/config"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get("status") == "success":
                        logger.info(f"Connected to Prometheus at {self.prometheus_url}")
                        
                        # Check Push Gateway connection if configured
                        if self.pushgateway_url:
                            try:
                                async with self.session.get(self.pushgateway_url + "/-/ready") as pgw_response:
                                    if pgw_response.status == 200:
                                        logger.info(f"Connected to Push Gateway at {self.pushgateway_url}")
                                    else:
                                        logger.warning(f"Push Gateway not ready: {pgw_response.status}")
                            except Exception as e:
                                logger.warning(f"Failed to connect to Push Gateway: {str(e)}")
                        
                        # Prefetch common metadata for better performance
                        await self._prefetch_metadata()
                        
                        return True
                    else:
                        logger.error(f"Prometheus connection status not success: {data}")
                        return False
                else:
                    error_text = await response.text()
                    logger.error(f"Failed to connect to Prometheus: {response.status} - {error_text}")
                    return False
                
        except Exception as e:
            logger.error(f"Exception connecting to Prometheus: {str(e)}")
            return False
    
    async def is_connected(self) -> bool:
        """
        Check if the connector is currently connected to Prometheus.
        
        Returns:
            True if connected
        """
        if self.session is None:
            return False
        
        try:
            # Simple ping to check connection
            url = f"{self.prometheus_url}/api/v1/status/runtimeinfo"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("status") == "success"
                return False
                
        except Exception:
            return False
    
    async def _prefetch_metadata(self):
        """Prefetch common metadata like targets and alert rules."""
        try:
            # Get targets
            url = f"{self.prometheus_url}/api/v1/targets"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get("status") == "success":
                        self.cache["targets"] = data.get("data", {}).get("activeTargets", [])
                        logger.debug(f"Cached {len(self.cache['targets'])} targets")
            
            # Get alerts
            url = f"{self.prometheus_url}/api/v1/alerts"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get("status") == "success":
                        self.cache["alerts"] = data.get("data", {}).get("alerts", [])
                        logger.debug(f"Cached {len(self.cache['alerts'])} alerts")
            
            # Get metric metadata
            url = f"{self.prometheus_url}/api/v1/metadata"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get("status") == "success":
                        self.cache["metric_metadata"] = data.get("data", {})
                        logger.debug(f"Cached metadata for {len(self.cache['metric_metadata'])} metrics")
                        
        except Exception as e:
            logger.warning(f"Error prefetching Prometheus metadata: {str(e)}")
    
    async def _handle_rate_limit(self, response):
        """
        Handle rate limiting by pausing when necessary.
        
        Args:
            response: The aiohttp response object
            
        Returns:
            The original response
        """
        if response.status == 429:  # Too Many Requests
            retry_after = response.headers.get("Retry-After")
            
            if retry_after:
                try:
                    seconds = int(retry_after)
                except ValueError:
                    seconds = self.rate_limit_sleep
            else:
                seconds = self.rate_limit_sleep
                
            logger.warning(f"Prometheus rate limit hit, pausing for {seconds} seconds")
            await asyncio.sleep(seconds)
        
        return response
    
    async def execute_operation(self, operation: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute an operation on Prometheus.
        
        Args:
            operation: The operation to execute
            params: Parameters for the operation
            
        Returns:
            Dictionary with operation result
        """
        await self._ensure_session()
        
        try:
            result = ExternalSystemResult(
                success=False,
                operation=operation,
                error_message="Operation not implemented",
                error_code="NOT_IMPLEMENTED"
            )
            
            # Map common operations to Prometheus API calls
            if operation == "query":
                result = await self._query(params)
            elif operation == "query_range":
                result = await self._query_range(params)
            elif operation == "push_metrics":
                result = await self._push_metrics(params)
            elif operation == "delete_metrics":
                result = await self._delete_metrics(params)
            elif operation == "get_metadata":
                result = await self._get_metadata(params)
            elif operation == "get_targets":
                result = await self._get_targets(params)
            elif operation == "get_alerts":
                result = await self._get_alerts(params)
            elif operation == "get_rules":
                result = await self._get_rules(params)
            elif operation == "get_series":
                result = await self._get_series(params)
            elif operation == "get_labels":
                result = await self._get_labels(params)
            elif operation == "get_label_values":
                result = await self._get_label_values(params)
            
            return result.to_dict()
            
        except Exception as e:
            logger.error(f"Exception executing Prometheus operation {operation}: {str(e)}")
            
            return ExternalSystemResult(
                success=False,
                operation=operation,
                error_message=str(e),
                error_code="EXCEPTION"
            ).to_dict()
    
    async def _query(self, params: Dict[str, Any]) -> ExternalSystemResult:
        """
        Execute an instant query on Prometheus.
        
        Args:
            params: Parameters for the instant query
            
        Returns:
            ExternalSystemResult with query results
        """
        query = params.get("query")
        
        if not query:
            return ExternalSystemResult(
                success=False,
                operation="query",
                error_message="Query expression is required",
                error_code="MISSING_QUERY"
            )
        
        # Prepare query parameters
        query_params = {"query": query}
        
        # Add optional parameters
        if "time" in params:
            query_params["time"] = params["time"]
        
        if "timeout" in params:
            query_params["timeout"] = params["timeout"]
        
        url = f"{self.prometheus_url}/api/v1/query"
        
        try:
            async with self.session.get(url, params=query_params) as response:
                response = await self._handle_rate_limit(response)
                
                if response.status == 200:
                    data = await response.json()
                    
                    if data.get("status") == "success":
                        return ExternalSystemResult(
                            success=True,
                            operation="query",
                            result_data=data.get("data", {})
                        )
                    else:
                        return ExternalSystemResult(
                            success=False,
                            operation="query",
                            error_message=f"Query failed: {data.get('error', 'Unknown error')}",
                            error_code=data.get("errorType", "QUERY_ERROR")
                        )
                else:
                    error_text = await response.text()
                    
                    return ExternalSystemResult(
                        success=False,
                        operation="query",
                        error_message=f"Query request failed: {response.status} - {error_text}",
                        error_code=f"HTTP_{response.status}"
                    )
                    
        except Exception as e:
            return ExternalSystemResult(
                success=False,
                operation="query",
                error_message=f"Exception executing query: {str(e)}",
                error_code="EXCEPTION"
            )
    
    async def _query_range(self, params: Dict[str, Any]) -> ExternalSystemResult:
        """
        Execute a range query on Prometheus.
        
        Args:
            params: Parameters for the range query
            
        Returns:
            ExternalSystemResult with query results
        """
        query = params.get("query")
        start = params.get("start")
        end = params.get("end")
        step = params.get("step")
        
        if not query:
            return ExternalSystemResult(
                success=False,
                operation="query_range",
                error_message="Query expression is required",
                error_code="MISSING_QUERY"
            )
        
        if not start:
            return ExternalSystemResult(
                success=False,
                operation="query_range",
                error_message="Start time is required",
                error_code="MISSING_START"
            )
        
        if not end:
            return ExternalSystemResult(
                success=False,
                operation="query_range",
                error_message="End time is required",
                error_code="MISSING_END"
            )
        
        if not step:
            return ExternalSystemResult(
                success=False,
                operation="query_range",
                error_message="Step interval is required",
                error_code="MISSING_STEP"
            )
        
        # Prepare query parameters
        query_params = {
            "query": query,
            "start": start,
            "end": end,
            "step": step
        }
        
        # Add optional parameters
        if "timeout" in params:
            query_params["timeout"] = params["timeout"]
        
        url = f"{self.prometheus_url}/api/v1/query_range"
        
        try:
            async with self.session.get(url, params=query_params) as response:
                response = await self._handle_rate_limit(response)
                
                if response.status == 200:
                    data = await response.json()
                    
                    if data.get("status") == "success":
                        return ExternalSystemResult(
                            success=True,
                            operation="query_range",
                            result_data=data.get("data", {})
                        )
                    else:
                        return ExternalSystemResult(
                            success=False,
                            operation="query_range",
                            error_message=f"Range query failed: {data.get('error', 'Unknown error')}",
                            error_code=data.get("errorType", "QUERY_ERROR")
                        )
                else:
                    error_text = await response.text()
                    
                    return ExternalSystemResult(
                        success=False,
                        operation="query_range",
                        error_message=f"Range query request failed: {response.status} - {error_text}",
                        error_code=f"HTTP_{response.status}"
                    )
                    
        except Exception as e:
            return ExternalSystemResult(
                success=False,
                operation="query_range",
                error_message=f"Exception executing range query: {str(e)}",
                error_code="EXCEPTION"
            )
    
    async def _push_metrics(self, params: Dict[str, Any]) -> ExternalSystemResult:
        """
        Push metrics to Prometheus Push Gateway.
        
        Args:
            params: Parameters for pushing metrics
            
        Returns:
            ExternalSystemResult with push operation result
        """
        if not self.pushgateway_url:
            return ExternalSystemResult(
                success=False,
                operation="push_metrics",
                error_message="Push Gateway URL is not configured",
                error_code="NO_PUSHGATEWAY_URL"
            )
        
        job_name = params.get("job_name", "unknown_job")
        instance = params.get("instance", "unknown_instance")
        metrics = params.get("metrics", [])
        labels = params.get("labels", {})
        
        if not metrics:
            return ExternalSystemResult(
                success=False,
                operation="push_metrics",
                error_message="Metrics are required",
                error_code="MISSING_METRICS"
            )
        
        # Format job name with prefix if not already included
        if not job_name.startswith(f"{self.job_prefix}_"):
            job_name = f"{self.job_prefix}_{job_name}"
        
        # Build URL with job and instance
        url = f"{self.pushgateway_url}/metrics/job/{quote(job_name)}/instance/{quote(instance)}"
        
        # Add additional labels if provided
        for key, value in labels.items():
            url += f"/{key}/{quote(value)}"
        
        # Format metrics in Prometheus text format
        metric_lines = []
        timestamp_ms = int(time.time() * 1000)
        
        for metric in metrics:
            name = metric.get("name")
            value = metric.get("value")
            help_text = metric.get("help", "")
            type_text = metric.get("type", "gauge")
            metric_labels = metric.get("labels", {})
            
            if not name or value is None:
                continue
            
            # Add namespace if not already included
            if not name.startswith(f"{self.default_namespace}_"):
                name = f"{self.default_namespace}_{name}"
            
            # Add help and type comments
            if help_text:
                metric_lines.append(f"# HELP {name} {help_text}")
            if type_text:
                metric_lines.append(f"# TYPE {name} {type_text}")
            
            # Format labels if present
            if metric_labels:
                label_str = ",".join([f'{k}="{v}"' for k, v in metric_labels.items()])
                metric_lines.append(f"{name}{{{label_str}}} {value} {timestamp_ms}")
            else:
                metric_lines.append(f"{name} {value} {timestamp_ms}")
        
        metric_data = "\n".join(metric_lines)
        
        try:
            async with self.session.post(
                url, 
                data=metric_data,
                headers={"Content-Type": "text/plain"}
            ) as response:
                response = await self._handle_rate_limit(response)
                
                if response.status in [200, 202]:
                    return ExternalSystemResult(
                        success=True,
                        operation="push_metrics",
                        result_data={
                            "job_name": job_name,
                            "instance": instance,
                            "metrics_count": len(metrics)
                        }
                    )
                else:
                    error_text = await response.text()
                    
                    return ExternalSystemResult(
                        success=False,
                        operation="push_metrics",
                        error_message=f"Failed to push metrics: {response.status} - {error_text}",
                        error_code=f"HTTP_{response.status}"
                    )
                    
        except Exception as e:
            return ExternalSystemResult(
                success=False,
                operation="push_metrics",
                error_message=f"Exception pushing metrics: {str(e)}",
                error_code="EXCEPTION"
            )
    
    async def _delete_metrics(self, params: Dict[str, Any]) -> ExternalSystemResult:
        """
        Delete metrics from Prometheus Push Gateway.
        
        Args:
            params: Parameters for deleting metrics
            
        Returns:
            ExternalSystemResult with delete operation result
        """
        if not self.pushgateway_url:
            return ExternalSystemResult(
                success=False,
                operation="delete_metrics",
                error_message="Push Gateway URL is not configured",
                error_code="NO_PUSHGATEWAY_URL"
            )
        
        job_name = params.get("job_name", "unknown_job")
        instance = params.get("instance")
        labels = params.get("labels", {})
        
        # Format job name with prefix if not already included
        if not job_name.startswith(f"{self.job_prefix}_"):
            job_name = f"{self.job_prefix}_{job_name}"
        
        # Build URL with job
        url = f"{self.pushgateway_url}/metrics/job/{quote(job_name)}"
        
        # Add instance if provided
        if instance:
            url += f"/instance/{quote(instance)}"
        
        # Add additional labels if provided
        for key, value in labels.items():
            url += f"/{key}/{quote(value)}"
        
        try:
            async with self.session.delete(url) as response:
                response = await self._handle_rate_limit(response)
                
                if response.status in [200, 202]:
                    return ExternalSystemResult(
                        success=True,
                        operation="delete_metrics",
                        result_data={
                            "job_name": job_name,
                            "instance": instance
                        }
                    )
                else:
                    error_text = await response.text()
                    
                    return ExternalSystemResult(
                        success=False,
                        operation="delete_metrics",
                        error_message=f"Failed to delete metrics: {response.status} - {error_text}",
                        error_code=f"HTTP_{response.status}"
                    )
                    
        except Exception as e:
            return ExternalSystemResult(
                success=False,
                operation="delete_metrics",
                error_message=f"Exception deleting metrics: {str(e)}",
                error_code="EXCEPTION"
            )
    
    async def _get_metadata(self, params: Dict[str, Any]) -> ExternalSystemResult:
        """
        Get metadata for metrics.
        
        Args:
            params: Parameters for getting metadata
            
        Returns:
            ExternalSystemResult with metadata
        """
        metric = params.get("metric")
        url = f"{self.prometheus_url}/api/v1/metadata"
        
        if metric:
            url += f"?metric={quote(metric)}"
        
        try:
            async with self.session.get(url) as response:
                response = await self._handle_rate_limit(response)
                
                if response.status == 200:
                    data = await response.json()
                    
                    if data.get("status") == "success":
                        return ExternalSystemResult(
                            success=True,
                            operation="get_metadata",
                            result_data=data.get("data", {})
                        )
                    else:
                        return ExternalSystemResult(
                            success=False,
                            operation="get_metadata",
                            error_message=f"Get metadata failed: {data.get('error', 'Unknown error')}",
                            error_code=data.get("errorType", "QUERY_ERROR")
                        )
                else:
                    error_text = await response.text()
                    
                    return ExternalSystemResult(
                        success=False,
                        operation="get_metadata",
                        error_message=f"Get metadata request failed: {response.status} - {error_text}",
                        error_code=f"HTTP_{response.status}"
                    )
                    
        except Exception as e:
            return ExternalSystemResult(
                success=False,
                operation="get_metadata",
                error_message=f"Exception getting metadata: {str(e)}",
                error_code="EXCEPTION"
            )
    
    async def _get_targets(self, params: Dict[str, Any]) -> ExternalSystemResult:
        """
        Get Prometheus targets.
        
        Args:
            params: Parameters for getting targets
            
        Returns:
            ExternalSystemResult with targets
        """
        state = params.get("state")
        url = f"{self.prometheus_url}/api/v1/targets"
        
        if state:
            url += f"?state={quote(state)}"
        
        try:
            async with self.session.get(url) as response:
                response = await self._handle_rate_limit(response)
                
                if response.status == 200:
                    data = await response.json()
                    
                    if data.get("status") == "success":
                        return ExternalSystemResult(
                            success=True,
                            operation="get_targets",
                            result_data=data.get("data", {})
                        )
                    else:
                        return ExternalSystemResult(
                            success=False,
                            operation="get_targets",
                            error_message=f"Get targets failed: {data.get('error', 'Unknown error')}",
                            error_code=data.get("errorType", "QUERY_ERROR")
                        )
                else:
                    error_text = await response.text()
                    
                    return ExternalSystemResult(
                        success=False,
                        operation="get_targets",
                        error_message=f"Get targets request failed: {response.status} - {error_text}",
                        error_code=f"HTTP_{response.status}"
                    )
                    
        except Exception as e:
            return ExternalSystemResult(
                success=False,
                operation="get_targets",
                error_message=f"Exception getting targets: {str(e)}",
                error_code="EXCEPTION"
            )
    
    async def _get_alerts(self, params: Dict[str, Any]) -> ExternalSystemResult:
        """
        Get Prometheus alerts.
        
        Args:
            params: Parameters for getting alerts
            
        Returns:
            ExternalSystemResult with alerts
        """
        url = f"{self.prometheus_url}/api/v1/alerts"
        
        try:
            async with self.session.get(url) as response:
                response = await self._handle_rate_limit(response)
                
                if response.status == 200:
                    data = await response.json()
                    
                    if data.get("status") == "success":
                        return ExternalSystemResult(
                            success=True,
                            operation="get_alerts",
                            result_data=data.get("data", {})
                        )
                    else:
                        return ExternalSystemResult(
                            success=False,
                            operation="get_alerts",
                            error_message=f"Get alerts failed: {data.get('error', 'Unknown error')}",
                            error_code=data.get("errorType", "QUERY_ERROR")
                        )
                else:
                    error_text = await response.text()
                    
                    return ExternalSystemResult(
                        success=False,
                        operation="get_alerts",
                        error_message=f"Get alerts request failed: {response.status} - {error_text}",
                        error_code=f"HTTP_{response.status}"
                    )
                    
        except Exception as e:
            return ExternalSystemResult(
                success=False,
                operation="get_alerts",
                error_message=f"Exception getting alerts: {str(e)}",
                error_code="EXCEPTION"
            )
    
    async def _get_rules(self, params: Dict[str, Any]) -> ExternalSystemResult:
        """
        Get Prometheus rules.
        
        Args:
            params: Parameters for getting rules
            
        Returns:
            ExternalSystemResult with rules
        """
        url = f"{self.prometheus_url}/api/v1/rules"
        
        try:
            async with self.session.get(url) as response:
                response = await self._handle_rate_limit(response)
                
                if response.status == 200:
                    data = await response.json()
                    
                    if data.get("status") == "success":
                        return ExternalSystemResult(
                            success=True,
                            operation="get_rules",
                            result_data=data.get("data", {})
                        )
                    else:
                        return ExternalSystemResult(
                            success=False,
                            operation="get_rules",
                            error_message=f"Get rules failed: {data.get('error', 'Unknown error')}",
                            error_code=data.get("errorType", "QUERY_ERROR")
                        )
                else:
                    error_text = await response.text()
                    
                    return ExternalSystemResult(
                        success=False,
                        operation="get_rules",
                        error_message=f"Get rules request failed: {response.status} - {error_text}",
                        error_code=f"HTTP_{response.status}"
                    )
                    
        except Exception as e:
            return ExternalSystemResult(
                success=False,
                operation="get_rules",
                error_message=f"Exception getting rules: {str(e)}",
                error_code="EXCEPTION"
            )
    
    async def _get_series(self, params: Dict[str, Any]) -> ExternalSystemResult:
        """
        Get Prometheus series.
        
        Args:
            params: Parameters for getting series
            
        Returns:
            ExternalSystemResult with series
        """
        match = params.get("match", [])
        start = params.get("start")
        end = params.get("end")
        
        if not match:
            return ExternalSystemResult(
                success=False,
                operation="get_series",
                error_message="Match series selector is required",
                error_code="MISSING_MATCH"
            )
        
        # Ensure match is a list
        if isinstance(match, str):
            match = [match]
        
        # Prepare query parameters
        query_params = {}
        for m in match:
            query_params.setdefault("match[]", []).append(m)
        
        if start:
            query_params["start"] = start
            
        if end:
            query_params["end"] = end
        
        url = f"{self.prometheus_url}/api/v1/series"
        
        try:
            async with self.session.get(url, params=query_params) as response:
                response = await self._handle_rate_limit(response)
                
                if response.status == 200:
                    data = await response.json()
                    
                    if data.get("status") == "success":
                        return ExternalSystemResult(
                            success=True,
                            operation="get_series",
                            result_data=data.get("data", [])
                        )
                    else:
                        return ExternalSystemResult(
                            success=False,
                            operation="get_series",
                            error_message=f"Get series failed: {data.get('error', 'Unknown error')}",
                            error_code=data.get("errorType", "QUERY_ERROR")
                        )
                else:
                    error_text = await response.text()
                    
                    return ExternalSystemResult(
                        success=False,
                        operation="get_series",
                        error_message=f"Get series request failed: {response.status} - {error_text}",
                        error_code=f"HTTP_{response.status}"
                    )
                    
        except Exception as e:
            return ExternalSystemResult(
                success=False,
                operation="get_series",
                error_message=f"Exception getting series: {str(e)}",
                error_code="EXCEPTION"
            )
    
    async def _get_labels(self, params: Dict[str, Any]) -> ExternalSystemResult:
        """
        Get Prometheus labels.
        
        Args:
            params: Parameters for getting labels
            
        Returns:
            ExternalSystemResult with labels
        """
        start = params.get("start")
        end = params.get("end")
        
        # Prepare query parameters
        query_params = {}
        
        if start:
            query_params["start"] = start
            
        if end:
            query_params["end"] = end
        
        url = f"{self.prometheus_url}/api/v1/labels"
        
        try:
            async with self.session.get(url, params=query_params) as response:
                response = await self._handle_rate_limit(response)
                
                if response.status == 200:
                    data = await response.json()
                    
                    if data.get("status") == "success":
                        return ExternalSystemResult(
                            success=True,
                            operation="get_labels",
                            result_data=data.get("data", [])
                        )
                    else:
                        return ExternalSystemResult(
                            success=False,
                            operation="get_labels",
                            error_message=f"Get labels failed: {data.get('error', 'Unknown error')}",
                            error_code=data.get("errorType", "QUERY_ERROR")
                        )
                else:
                    error_text = await response.text()
                    
                    return ExternalSystemResult(
                        success=False,
                        operation="get_labels",
                        error_message=f"Get labels request failed: {response.status} - {error_text}",
                        error_code=f"HTTP_{response.status}"
                    )
                    
        except Exception as e:
            return ExternalSystemResult(
                success=False,
                operation="get_labels",
                error_message=f"Exception getting labels: {str(e)}",
                error_code="EXCEPTION"
            )
    
    async def _get_label_values(self, params: Dict[str, Any]) -> ExternalSystemResult:
        """
        Get Prometheus label values.
        
        Args:
            params: Parameters for getting label values
            
        Returns:
            ExternalSystemResult with label values
        """
        label_name = params.get("label")
        start = params.get("start")
        end = params.get("end")
        
        if not label_name:
            return ExternalSystemResult(
                success=False,
                operation="get_label_values",
                error_message="Label name is required",
                error_code="MISSING_LABEL"
            )
        
        # Prepare query parameters
        query_params = {}
        
        if start:
            query_params["start"] = start
            
        if end:
            query_params["end"] = end
        
        url = f"{self.prometheus_url}/api/v1/label/{quote(label_name)}/values"
        
        try:
            async with self.session.get(url, params=query_params) as response:
                response = await self._handle_rate_limit(response)
                
                if response.status == 200:
                    data = await response.json()
                    
                    if data.get("status") == "success":
                        return ExternalSystemResult(
                            success=True,
                            operation="get_label_values",
                            result_data=data.get("data", [])
                        )
                    else:
                        return ExternalSystemResult(
                            success=False,
                            operation="get_label_values",
                            error_message=f"Get label values failed: {data.get('error', 'Unknown error')}",
                            error_code=data.get("errorType", "QUERY_ERROR")
                        )
                else:
                    error_text = await response.text()
                    
                    return ExternalSystemResult(
                        success=False,
                        operation="get_label_values",
                        error_message=f"Get label values request failed: {response.status} - {error_text}",
                        error_code=f"HTTP_{response.status}"
                    )
                    
        except Exception as e:
            return ExternalSystemResult(
                success=False,
                operation="get_label_values",
                error_message=f"Exception getting label values: {str(e)}",
                error_code="EXCEPTION"
            )
    
    async def query(self, query_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Query metrics from Prometheus.
        
        Args:
            query_params: Query parameters
            
        Returns:
            List of query results
        """
        query_expr = query_params.get("query")
        
        if not query_expr:
            logger.error("Query expression is required for Prometheus queries")
            return []
        
        # Determine if this is a range query
        if "start" in query_params and "end" in query_params and "step" in query_params:
            # Range query
            result = await self._query_range({
                "query": query_expr,
                "start": query_params["start"],
                "end": query_params["end"],
                "step": query_params["step"]
            })
        else:
            # Instant query
            params = {"query": query_expr}
            if "time" in query_params:
                params["time"] = query_params["time"]
            
            result = await self._query(params)
        
        if result.success:
            # Extract vector results from Prometheus response
            if "resultType" in result.result_data:
                result_type = result.result_data["resultType"]
                
                if result_type == "vector":
                    return result.result_data.get("result", [])
                elif result_type == "matrix":
                    return result.result_data.get("result", [])
                elif result_type == "scalar":
                    # Return scalar as a single item list
                    value = result.result_data.get("result", [0, "0"])
                    return [{"value": value}]
                elif result_type == "string":
                    # Return string as a single item list
                    value = result.result_data.get("result", [0, ""])
                    return [{"value": value}]
            
            # If no specific format, just return the result data
            return [result.result_data]
        else:
            logger.error(f"Error querying Prometheus: {result.error_message}")
            return []
    
    async def create_item(self, item_type: str, item_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create an item in Prometheus.
        
        This is primarily for pushing metrics to the Prometheus Push Gateway.
        
        Args:
            item_type: Type of item to create
            item_data: Item data
            
        Returns:
            Dictionary with created item details
        """
        if item_type == "metric":
            # Push a single metric
            result = await self._push_metrics({
                "job_name": item_data.get("job_name", "unknown_job"),
                "instance": item_data.get("instance", "unknown_instance"),
                "metrics": [item_data],
                "labels": item_data.get("extra_labels", {})
            })
        elif item_type == "metrics":
            # Push multiple metrics
            result = await self._push_metrics(item_data)
        else:
            result = ExternalSystemResult(
                success=False,
                operation=f"create_{item_type}",
                error_message=f"Unsupported item type: {item_type}",
                error_code="UNSUPPORTED_ITEM_TYPE"
            )
        
        if result.success:
            return result.result_data
        else:
            raise Exception(f"Failed to create {item_type}: {result.error_message}")
    
    async def update_item(self, item_type: str, item_id: str, update_data: Dict[str, Any]) -> bool:
        """
        Update an item in Prometheus.
        
        For Prometheus, this is the same as creating a new metric, as metrics are immutable
        time series data points. Updating essentially means creating a new data point.
        
        Args:
            item_type: Type of item to update
            item_id: ID of the item to update (metric name)
            update_data: Data to update
            
        Returns:
            True if update succeeded
        """
        if item_type == "metric":
            # For metrics, update is the same as create
            update_data["name"] = item_id
            result = await self._push_metrics({
                "job_name": update_data.get("job_name", "unknown_job"),
                "instance": update_data.get("instance", "unknown_instance"),
                "metrics": [update_data],
                "labels": update_data.get("extra_labels", {})
            })
            return result.success
        else:
            return False
    
    async def delete_item(self, item_type: str, item_id: str) -> bool:
        """
        Delete metrics from Prometheus.
        
        For Prometheus, this means deleting metrics from the Push Gateway using the
        job name as the identifier.
        
        Args:
            item_type: Type of item to delete
            item_id: ID of the item to delete (job name)
            
        Returns:
            True if deletion succeeded
        """
        if item_type == "metrics":
            # Delete metrics by job name
            result = await self._delete_metrics({"job_name": item_id})
            return result.success
        else:
            return False
    
    async def get_item(self, item_type: str, item_id: str) -> Dict[str, Any]:
        """
        Get an item from Prometheus.
        
        Args:
            item_type: Type of item to get
            item_id: ID of the item to get
            
        Returns:
            Dictionary with item details
        """
        if item_type == "metric":
            # Get metric data using instant query
            result = await self._query({"query": item_id})
        elif item_type == "target":
            # Get target data
            result = await self._get_targets({})
            
            # Find the specific target
            if result.success:
                targets = result.result_data.get("activeTargets", [])
                for target in targets:
                    if target.get("labels", {}).get("instance") == item_id:
                        return target
                
                raise Exception(f"Target not found: {item_id}")
        elif item_type == "metadata":
            # Get metric metadata
            result = await self._get_metadata({"metric": item_id})
        else:
            result = ExternalSystemResult(
                success=False,
                operation=f"get_{item_type}",
                error_message=f"Unsupported item type: {item_type}",
                error_code="UNSUPPORTED_ITEM_TYPE"
            )
        
        if result.success:
            return result.result_data
        else:
            raise Exception(f"Failed to get {item_type}: {result.error_message}")
    
    async def system_info(self) -> Dict[str, Any]:
        """
        Get information about the Prometheus system.
        
        Returns:
            Dictionary with system information
        """
        await self._ensure_session()
        
        try:
            # Get runtime info
            url = f"{self.prometheus_url}/api/v1/status/runtimeinfo"
            runtime_info = {}
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get("status") == "success":
                        runtime_info = data.get("data", {})
                
            # Get build info
            url = f"{self.prometheus_url}/api/v1/status/buildinfo"
            build_info = {}
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get("status") == "success":
                        build_info = data.get("data", {})
            
            # Check Push Gateway status if configured
            pushgateway_status = "not_configured"
            if self.pushgateway_url:
                try:
                    async with self.session.get(self.pushgateway_url + "/-/ready") as pgw_response:
                        pushgateway_status = "ready" if pgw_response.status == 200 else f"error_{pgw_response.status}"
                except Exception as e:
                    pushgateway_status = f"error: {str(e)}"
            
            # Build system info
            return {
                "system_type": "prometheus",
                "connected": await self.is_connected(),
                "server_url": self.prometheus_url,
                "pushgateway_url": self.pushgateway_url,
                "pushgateway_status": pushgateway_status,
                "runtime_info": runtime_info,
                "build_info": build_info,
                "capabilities": self.capabilities.to_dict()
            }
            
        except Exception as e:
            logger.error(f"Exception getting Prometheus system info: {str(e)}")
            
            return {
                "system_type": "prometheus",
                "connected": False,
                "server_url": self.prometheus_url,
                "pushgateway_url": self.pushgateway_url,
                "error": str(e),
                "capabilities": self.capabilities.to_dict()
            }
    
    async def close(self) -> None:
        """
        Close the connection to Prometheus and clean up resources.
        
        Returns:
            None
        """
        if self.session:
            await self.session.close()
            self.session = None
            logger.info("Prometheus connection closed")


# Register with factory
ExternalSystemFactory.register_connector("prometheus", PrometheusConnector)