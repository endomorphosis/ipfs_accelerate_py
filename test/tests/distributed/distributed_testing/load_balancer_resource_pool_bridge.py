#!/usr/bin/env python3
"""
Integration between Adaptive Load Balancer and Resource Pool Bridge

This module provides the integration layer between the Adaptive Load Balancer
from the Distributed Testing Framework and the WebGPU/WebNN Resource Pool Bridge.
It enables intelligent distribution of browser resources across worker nodes.

Usage:
    Import this module to create an integration between the load balancer
    and browser-based resource pools.
"""

from ipfs_accelerate_py.anyio_helpers import gather, wait_for
import anyio
import json
import logging
import time
import os
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple, NamedTuple, Callable, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import components
try:
    from resource_pool_bridge import ResourcePoolBridgeIntegration
    from resource_pool_bridge_recovery import BrowserStateManager, ResourcePoolRecoveryManager
    from model_sharding import ShardedModelExecution
except ImportError:
    # Try with full path
    from .resource_pool_bridge import ResourcePoolBridgeIntegration
    from .resource_pool_bridge_recovery import BrowserStateManager, ResourcePoolRecoveryManager
    from test.tests.distributed.distributed_testing.model_sharding import ShardedModelExecution

try:
    from data.duckdb.distributed_testing.load_balancer import LoadBalancerService, WorkerCapabilities, TestRequirements
    from data.duckdb.distributed_testing.load_balancer.models import WorkerCapabilities, TestRequirements, WorkerLoad
except ImportError:
    # Try alternative import path
    from data.duckdb.distributed_testing.load_balancer.service import LoadBalancerService
    from data.duckdb.distributed_testing.load_balancer.models import WorkerCapabilities, TestRequirements, WorkerLoad


class ResourcePoolWorker:
    """
    Worker implementation that integrates with the Resource Pool Bridge.
    
    This class represents a worker node in the distributed testing framework
    that can manage browser resources for testing.
    """
    
    def __init__(
        self,
        worker_id: str,
        max_browsers: int = 3,
        browser_preferences: Dict[str, str] = None,
        enable_fault_tolerance: bool = True,
        recovery_strategy: str = "progressive"
    ):
        """
        Initialize the resource pool worker.
        
        Args:
            worker_id: Unique identifier for the worker
            max_browsers: Maximum number of browser instances to manage
            browser_preferences: Preferred browsers for different model types
            enable_fault_tolerance: Whether to enable fault tolerance features
            recovery_strategy: Recovery strategy for browser failures
        """
        self.worker_id = worker_id
        self.max_browsers = max_browsers
        self.browser_preferences = browser_preferences or {
            'audio': 'firefox',
            'vision': 'chrome',
            'text_embedding': 'edge'
        }
        self.enable_fault_tolerance = enable_fault_tolerance
        self.recovery_strategy = recovery_strategy
        
        # Resource pool integration
        self.resource_pool = None
        
        # Active models
        self.active_models = {}
        
        # Performance metrics
        self.metrics = {
            'cpu_percent': 0.0,
            'memory_percent': 0.0,
            'gpu_utilization': 0.0
        }
        
        # Browser-specific metrics
        self.browser_metrics = {
            'chrome': {'utilization': 0.0, 'memory_usage': 0.0, 'active_models': 0},
            'firefox': {'utilization': 0.0, 'memory_usage': 0.0, 'active_models': 0},
            'edge': {'utilization': 0.0, 'memory_usage': 0.0, 'active_models': 0}
        }
        
        # Browser performance history
        self.browser_performance_history = {
            'chrome': {'avg_latency': 0.0, 'success_rate': 0.0, 'sample_count': 0},
            'firefox': {'avg_latency': 0.0, 'success_rate': 0.0, 'sample_count': 0},
            'edge': {'avg_latency': 0.0, 'success_rate': 0.0, 'sample_count': 0}
        }
        
        # Model type performance by browser
        self.model_type_browser_performance = {
            'audio': {'firefox': 1.0, 'chrome': 0.7, 'edge': 0.6},
            'vision': {'chrome': 1.0, 'firefox': 0.8, 'edge': 0.7},
            'text_embedding': {'edge': 1.0, 'chrome': 0.8, 'firefox': 0.7}
        }
        
        # Browser capability scores
        self.browser_capability_scores = {
            'chrome': {'webgpu': 0.9, 'webnn': 0.7, 'general': 0.8},
            'firefox': {'webgpu': 0.8, 'webnn': 0.6, 'audio_processing': 0.95},
            'edge': {'webgpu': 0.7, 'webnn': 0.9, 'text_processing': 0.9}
        }
        
        # Hardware capabilities
        self.capabilities = self._build_capabilities()
        
        # Worker load
        self.worker_load = WorkerLoad(worker_id=worker_id)
        
        # Browser capacity tracking
        self.browser_capacities = {}
        
        # Sharded model executions
        self.sharded_executions = {}
        
        # Browser instance status tracking
        self.browser_instances = {}
        
        # Load prediction data
        self.load_prediction = {
            'browser_requests': [],  # [(timestamp, browser_type, model_type)]
            'avg_request_rate': 0.0,
            'predicted_loads': {}  # browser_type -> predicted_load
        }
        
        # Cache for browser capability scoring
        self.capability_score_cache = {}
        
        logger.info(f"ResourcePoolWorker {worker_id} initialized")
    
    def _build_capabilities(self) -> WorkerCapabilities:
        """
        Build worker capabilities for the load balancer.
        
        Returns:
            WorkerCapabilities object
        """
        # Create hardware specs
        hardware_specs = {
            'cpu': {
                'cores': os.cpu_count() or 4,
                'arch': 'x64'
            },
            'memory': {
                'total_gb': 16  # Would get from system in real implementation
            }
        }
        
        # Check for GPU
        has_gpu = True  # Would detect in real implementation
        has_webgpu = True
        has_webnn = 'edge' in self.browser_preferences.values()
        
        if has_gpu:
            hardware_specs['gpu'] = {
                'name': 'Test GPU',
                'vram_gb': 8
            }
        
        # Get supported browsers
        browser_support = list(set(self.browser_preferences.values()))
        
        # Create capabilities
        capabilities = WorkerCapabilities(
            worker_id=self.worker_id,
            hardware_specs=hardware_specs,
            available_memory=14.0,  # Would get from system in real implementation
            cpu_cores=os.cpu_count() or 4,
            has_gpu=has_gpu,
            has_webgpu=has_webgpu,
            has_webnn=has_webnn,
            browser_support=browser_support,
            supported_backends=['webgpu', 'webnn'] if has_webnn else ['webgpu']
        )
        
        return capabilities
    
    async def initialize(self) -> None:
        """Initialize the resource pool worker."""
        logger.info(f"Initializing resource pool for worker {self.worker_id}")
        
        # Create resource pool integration
        self.resource_pool = ResourcePoolBridgeIntegration(
            max_connections=self.max_browsers,
            browser_preferences=self.browser_preferences,
            adaptive_scaling=True,
            enable_fault_tolerance=self.enable_fault_tolerance,
            recovery_strategy=self.recovery_strategy,
            state_sync_interval=5,
            redundancy_factor=2 if self.enable_fault_tolerance else 1
        )
        
        # Initialize resource pool
        await self.resource_pool.initialize()
        
        # Update browser capacities
        self._update_browser_capacities()
        
        logger.info(f"Resource pool initialized for worker {self.worker_id}")
    
    def _update_browser_capacities(self) -> None:
        """Update browser capacity tracking based on resource pool state."""
        if not self.resource_pool or not self.resource_pool.connection_pool:
            return
        
        # Reset capacities
        self.browser_capacities = {}
        
        # Track capacity by browser type
        browser_counts = {'chrome': 0, 'firefox': 0, 'edge': 0}
        browser_active = {'chrome': 0, 'firefox': 0, 'edge': 0}
        browser_memory = {'chrome': 0.0, 'firefox': 0.0, 'edge': 0.0}
        
        for browser_id, browser in self.resource_pool.connection_pool.items():
            browser_type = browser.get('type', 'unknown')
            if browser_type in browser_counts:
                browser_counts[browser_type] += 1
                if browser.get('status') == 'ready':
                    active_models = len(browser.get('active_models', set()))
                    browser_active[browser_type] += active_models
                    
                    # Track memory usage if available
                    memory_usage = browser.get('memory_usage', 0.0)
                    browser_memory[browser_type] += memory_usage
        
        # Calculate capacity percentages
        for browser_type, count in browser_counts.items():
            if count > 0:
                active = browser_active[browser_type]
                # Estimate capacity as percentage of slots available
                capacity = max(0.0, 1.0 - (active / (count * 3)))  # Assume each browser can handle 3 models
                self.browser_capacities[browser_type] = capacity
                
                # Update browser metrics
                if browser_type in self.browser_metrics:
                    self.browser_metrics[browser_type]['active_models'] = active
                    self.browser_metrics[browser_type]['memory_usage'] = browser_memory[browser_type]
                    self.browser_metrics[browser_type]['utilization'] = 1.0 - capacity
        
    def _update_load_prediction(self) -> None:
        """Update load prediction based on request patterns and performance history."""
        now = datetime.now()
        
        # Add current request data
        for model_info in self.active_models.values():
            model_type = model_info['test_req'].model_type
            browser_type = self.browser_preferences.get(model_type, 'chrome')
            
            # Record timestamp, browser type, and model type
            self.load_prediction['browser_requests'].append((now, browser_type, model_type))
        
        # Keep only recent requests (last 5 minutes)
        recent_cutoff = now - timedelta(minutes=5)
        self.load_prediction['browser_requests'] = [
            req for req in self.load_prediction['browser_requests']
            if req[0] >= recent_cutoff
        ]
        
        # Calculate request rates by browser type
        browser_request_counts = {'chrome': 0, 'firefox': 0, 'edge': 0}
        for _, browser_type, _ in self.load_prediction['browser_requests']:
            if browser_type in browser_request_counts:
                browser_request_counts[browser_type] += 1
        
        # Calculate time window in minutes (for rate calculation)
        time_window_minutes = 5.0  # Default to 5 minutes
        if self.load_prediction['browser_requests']:
            oldest_request = min(req[0] for req in self.load_prediction['browser_requests'])
            time_window_minutes = max(0.1, (now - oldest_request).total_seconds() / 60.0)
        
        # Calculate request rates
        browser_request_rates = {}
        for browser_type, count in browser_request_counts.items():
            rate = count / time_window_minutes  # requests per minute
            browser_request_rates[browser_type] = rate
        
        # Update average request rate
        total_requests = sum(browser_request_counts.values())
        if time_window_minutes > 0:
            self.load_prediction['avg_request_rate'] = total_requests / time_window_minutes
        
        # Predict future load based on request rates and active models
        predicted_loads = {}
        for browser_type in ['chrome', 'firefox', 'edge']:
            # Current utilization
            current_utilization = self.browser_metrics.get(browser_type, {}).get('utilization', 0.0)
            
            # Request rate for this browser type
            request_rate = browser_request_rates.get(browser_type, 0.0)
            
            # Predict load in next 1 minute based on current load and incoming requests
            # Assume each model stays active for an average of 2 minutes
            current_active = self.browser_metrics.get(browser_type, {}).get('active_models', 0)
            
            # Calculate expected completions in next minute (using current active count)
            expected_completions = current_active * (1.0 / 2.0)  # rate = active / avg_duration
            
            # Calculate expected new models in next minute
            expected_new_models = request_rate
            
            # Net change in active models
            net_change = expected_new_models - expected_completions
            
            # Predicted active models in 1 minute
            predicted_active = max(0, current_active + net_change)
            
            # Calculate predicted browser utilization
            browser_count = browser_request_counts.get(browser_type, 0)
            if browser_count > 0:
                # Calculate browser capacity (models per browser)
                model_capacity = 3  # Assume each browser can handle 3 models
                predicted_utilization = min(1.0, predicted_active / (browser_count * model_capacity))
            else:
                predicted_utilization = 0.0
            
            # Store prediction
            predicted_loads[browser_type] = {
                'current_utilization': current_utilization,
                'request_rate': request_rate,
                'current_active': current_active,
                'predicted_active': predicted_active,
                'predicted_utilization': predicted_utilization
            }
        
        # Update predicted loads
        self.load_prediction['predicted_loads'] = predicted_loads
        
        # Log predictions for high utilization browsers
        for browser_type, prediction in predicted_loads.items():
            if prediction['predicted_utilization'] > 0.8:
                logger.debug(f"Predicted high utilization for {browser_type}: {prediction['predicted_utilization']:.2f} "
                           f"(current: {prediction['current_utilization']:.2f}, rate: {prediction['request_rate']:.2f} req/min)")
    
    async def execute_test(self, test_req: TestRequirements) -> Dict[str, Any]:
        """
        Execute a test using the resource pool.
        
        Args:
            test_req: Test requirements
            
        Returns:
            Test execution results
        """
        logger.info(f"Executing test {test_req.test_id} ({test_req.model_type}/{test_req.model_id})")
        
        if not self.resource_pool:
            raise RuntimeError("Resource pool not initialized")
        
        # Check if this is a sharded model test
        if test_req.requires_sharding and test_req.sharding_requirements:
            return await self._execute_sharded_test(test_req)
            
        # Get model from resource pool
        model = await self.resource_pool.get_model(
            model_type=test_req.model_type,
            model_name=test_req.model_id,
            hardware_preferences=self._get_hardware_preferences(test_req),
            fault_tolerance=self._get_fault_tolerance_settings(test_req)
        )
        
        if not model:
            raise RuntimeError(f"Failed to get model for test {test_req.test_id}")
        
        # Store model
        self.active_models[test_req.test_id] = {
            'model': model,
            'test_req': test_req,
            'start_time': datetime.now().isoformat()
        }
        
        # Update browser capacities
        self._update_browser_capacities()
        
        # Execute model
        try:
            # Prepare inputs
            inputs = self._prepare_inputs(test_req)
            
            # Run inference
            start_time = time.time()
            result = await model(inputs)
            execution_time = time.time() - start_time
            
            # Create result
            test_result = {
                'test_id': test_req.test_id,
                'model_id': test_req.model_id,
                'execution_time': execution_time,
                'status': 'success',
                'result': result
            }
            
            logger.info(f"Completed test {test_req.test_id} in {execution_time:.2f}s")
            
            return test_result
            
        except Exception as e:
            logger.error(f"Error executing test {test_req.test_id}: {str(e)}")
            
            # Create error result
            error_result = {
                'test_id': test_req.test_id,
                'model_id': test_req.model_id,
                'status': 'error',
                'error': str(e)
            }
            
            return error_result
        finally:
            # Clean up
            if test_req.test_id in self.active_models:
                del self.active_models[test_req.test_id]
            
            # Update browser capacities
            self._update_browser_capacities()
    
    async def _execute_sharded_test(self, test_req: TestRequirements) -> Dict[str, Any]:
        """
        Execute a test using model sharding.
        
        Args:
            test_req: Test requirements
            
        Returns:
            Test execution results
        """
        logger.info(f"Executing sharded test {test_req.test_id} ({test_req.model_id})")
        
        if not self.resource_pool or not self.resource_pool.sharding_manager:
            raise RuntimeError("Resource pool or sharding manager not initialized")
        
        # Get sharding requirements
        sharding_strategy = test_req.sharding_requirements.get("strategy", "layer_balanced")
        num_shards = test_req.sharding_requirements.get("num_shards", 2)
        fault_tolerance_level = test_req.sharding_requirements.get("fault_tolerance_level", "medium")
        
        # Create sharded model execution
        sharded_execution = ShardedModelExecution(
            model_name=test_req.model_id,
            sharding_strategy=sharding_strategy,
            num_shards=num_shards,
            fault_tolerance_level=fault_tolerance_level,
            recovery_strategy="coordinated",
            connection_pool=self.resource_pool.connection_pool
        )
        
        # Store sharded execution
        self.sharded_executions[test_req.test_id] = sharded_execution
        
        # Initialize sharded execution
        await sharded_execution.initialize()
        
        try:
            # Prepare inputs
            inputs = self._prepare_inputs(test_req)
            
            # Run inference
            start_time = time.time()
            result = await sharded_execution.run_inference(inputs)
            execution_time = time.time() - start_time
            
            # Create result
            test_result = {
                'test_id': test_req.test_id,
                'model_id': test_req.model_id,
                'execution_time': execution_time,
                'status': 'success',
                'result': result,
                'sharding': {
                    'strategy': sharding_strategy,
                    'num_shards': num_shards
                }
            }
            
            logger.info(f"Completed sharded test {test_req.test_id} in {execution_time:.2f}s")
            
            return test_result
            
        except Exception as e:
            logger.error(f"Error executing sharded test {test_req.test_id}: {str(e)}")
            
            # Create error result
            error_result = {
                'test_id': test_req.test_id,
                'model_id': test_req.model_id,
                'status': 'error',
                'error': str(e),
                'sharding': {
                    'strategy': sharding_strategy,
                    'num_shards': num_shards
                }
            }
            
            return error_result
        finally:
            # Clean up
            if test_req.test_id in self.sharded_executions:
                # Close sharded execution
                await self.sharded_executions[test_req.test_id].close()
                del self.sharded_executions[test_req.test_id]
            
            # Update browser capacities
            self._update_browser_capacities()
    
    def _prepare_inputs(self, test_req: TestRequirements) -> Any:
        """
        Prepare inputs for model execution based on test requirements.
        
        Args:
            test_req: Test requirements
            
        Returns:
            Prepared inputs
        """
        # Default input
        default_input = {"text": "This is a test input"}
        
        # Use test parameters if available
        test_params = test_req.parameters or {}
        
        if test_req.model_type == "vision":
            return test_params.get("image_input", "placeholder_image_data")
        elif test_req.model_type == "audio":
            return test_params.get("audio_input", "placeholder_audio_data")
        elif test_req.model_type == "text_embedding" or test_req.model_type == "large_language_model":
            return test_params.get("text_input", default_input)
        else:
            return test_params.get("input", default_input)
    
    def _get_hardware_preferences(self, test_req: TestRequirements) -> Dict[str, Any]:
        """
        Get hardware preferences for a test based on model type and browser capability scoring.
        
        Args:
            test_req: Test requirements
            
        Returns:
            Hardware preferences
        """
        # Get browser requirements
        browser_reqs = test_req.browser_requirements or {}
        preferred_browser = browser_reqs.get("preferred")
        
        # Default hardware preferences
        hw_preferences = {
            'priority_list': ['webgpu', 'webnn', 'cpu']
        }
        
        # Generate a unique cache key for this request
        cache_key = f"{test_req.model_type}:{test_req.model_id}:{preferred_browser or 'none'}"
        
        # Check if we have cached results for this combination
        if cache_key in self.capability_score_cache:
            # Use cached result if it's recent (less than 5 minutes old)
            cached_result = self.capability_score_cache[cache_key]
            cache_age = (datetime.now() - cached_result['timestamp']).total_seconds()
            if cache_age < 300:  # 5 minutes
                return cached_result['preferences']
        
        # Compute browser capability score for this model type
        browser_scores = self._compute_browser_capability_scores(test_req)
        
        # Get the browser with highest capability score
        best_browser = max(browser_scores.items(), key=lambda x: x[1])[0]
        
        # Determine optimal hardware backends based on browser and model type
        if test_req.model_type == "text_embedding":
            if best_browser == "edge":
                hw_preferences['priority_list'] = ['webnn', 'webgpu', 'cpu']
            elif best_browser == "chrome":
                hw_preferences['priority_list'] = ['webgpu', 'webnn', 'cpu']
            else:  # firefox or other
                hw_preferences['priority_list'] = ['webgpu', 'cpu', 'webnn']
        elif test_req.model_type == "vision":
            if best_browser == "chrome":
                hw_preferences['priority_list'] = ['webgpu', 'webnn', 'cpu']
            else:
                hw_preferences['priority_list'] = ['webgpu', 'cpu', 'webnn']
        elif test_req.model_type == "audio":
            if best_browser == "firefox":
                hw_preferences['priority_list'] = ['webgpu', 'cpu', 'webnn']
            else:
                hw_preferences['priority_list'] = ['webgpu', 'webnn', 'cpu']
        elif test_req.model_type == "large_language_model":
            hw_preferences['priority_list'] = ['webgpu', 'cpu', 'webnn']
        
        # Override with explicit preferred browser if specified
        if preferred_browser:
            # Ensure preferred browser is used regardless of scoring
            hw_preferences['preferred_browser'] = preferred_browser
        else:
            # Otherwise use the best scored browser
            hw_preferences['preferred_browser'] = best_browser
        
        # Add capability scores to preferences
        hw_preferences['browser_scores'] = browser_scores
        
        # Cache the result
        self.capability_score_cache[cache_key] = {
            'timestamp': datetime.now(),
            'preferences': hw_preferences
        }
        
        return hw_preferences
        
    def _compute_browser_capability_scores(self, test_req: TestRequirements) -> Dict[str, float]:
        """
        Compute capability scores for each browser type for a specific test request.
        
        Args:
            test_req: Test requirements
            
        Returns:
            Dictionary mapping browser types to capability scores (0.0-1.0)
        """
        # Base scores from browser preferences
        base_scores = {
            'chrome': 0.5,
            'firefox': 0.5,
            'edge': 0.5
        }
        
        # Apply model type preference factors
        model_type = test_req.model_type
        if model_type in self.model_type_browser_performance:
            model_perf = self.model_type_browser_performance[model_type]
            for browser, perf_factor in model_perf.items():
                if browser in base_scores:
                    base_scores[browser] *= perf_factor
        
        # Apply runtime browser metrics
        for browser_type, metrics in self.browser_metrics.items():
            if browser_type in base_scores:
                # Penalty for high utilization
                utilization = metrics.get('utilization', 0.0)
                utilization_factor = max(0.1, 1.0 - utilization)
                base_scores[browser_type] *= utilization_factor
                
                # Penalty for many active models
                active_models = metrics.get('active_models', 0)
                if active_models > 2:
                    # Progressive penalty for more active models
                    active_penalty = max(0.2, 1.0 - ((active_models - 2) * 0.15))
                    base_scores[browser_type] *= active_penalty
        
        # Apply load prediction factors
        for browser_type, prediction in self.load_prediction.get('predicted_loads', {}).items():
            if browser_type in base_scores:
                predicted_util = prediction.get('predicted_utilization', 0.0)
                if predicted_util > 0.7:
                    # Penalty for high predicted utilization
                    prediction_factor = max(0.1, 1.0 - ((predicted_util - 0.7) * 2.0))
                    base_scores[browser_type] *= prediction_factor
        
        # Apply performance history factors if available
        for browser_type, history in self.browser_performance_history.items():
            if browser_type in base_scores and history.get('sample_count', 0) > 5:
                # Reward browsers with good success rate
                success_rate = history.get('success_rate', 0.0)
                success_factor = 0.2 + (success_rate * 0.8)  # Scale from 0.2 to 1.0
                base_scores[browser_type] *= success_factor
                
                # Reward browsers with low latency (if we have latency data)
                if 'avg_latency' in history and history['avg_latency'] > 0:
                    # Compare to average latency across browsers
                    avg_latencies = [h.get('avg_latency', 0.0) for h in self.browser_performance_history.values()
                                    if h.get('avg_latency', 0.0) > 0]
                    if avg_latencies:
                        overall_avg = sum(avg_latencies) / len(avg_latencies)
                        if overall_avg > 0:
                            latency_ratio = history['avg_latency'] / overall_avg
                            latency_factor = 1.0 / max(0.5, min(1.5, latency_ratio))  # Invert and clamp
                            base_scores[browser_type] *= latency_factor
        
        # Normalize scores (optional, but helps with interpretability)
        max_score = max(base_scores.values()) if base_scores else 1.0
        if max_score > 0:
            normalized_scores = {browser: score / max_score for browser, score in base_scores.items()}
        else:
            normalized_scores = base_scores
        
        return normalized_scores
    
    def _get_fault_tolerance_settings(self, test_req: TestRequirements) -> Dict[str, Any]:
        """
        Get fault tolerance settings for a test.
        
        Args:
            test_req: Test requirements
            
        Returns:
            Fault tolerance settings
        """
        if not self.enable_fault_tolerance:
            return None
        
        # Default settings
        settings = {
            'recovery_timeout': 30,  # 30 seconds
            'state_persistence': True,
            'failover_strategy': 'immediate'
        }
        
        # Adjust based on test priority
        if test_req.priority == 1:  # High priority
            settings['recovery_timeout'] = 15  # Faster recovery for high priority
            settings['failover_strategy'] = 'immediate'
        elif test_req.priority == 3:  # Low priority
            settings['recovery_timeout'] = 60  # Longer timeout for low priority
            settings['failover_strategy'] = 'progressive'
        
        return settings
    
    async def update_metrics(self) -> None:
        """Update worker metrics including browser-specific utilization."""
        # In a real implementation, would get metrics from system
        # For now, simulate metrics based on active models
        num_active = len(self.active_models)
        
        # Simulate CPU utilization (20% base + 10% per active model)
        self.metrics['cpu_percent'] = min(90.0, 20.0 + (num_active * 10.0))
        
        # Simulate memory utilization (30% base + 15% per active model)
        self.metrics['memory_percent'] = min(95.0, 30.0 + (num_active * 15.0))
        
        # Simulate GPU utilization if models use GPU
        gpu_models = sum(1 for model_info in self.active_models.values() 
                         if model_info['test_req'].model_type in ['vision', 'large_language_model'])
        self.metrics['gpu_utilization'] = min(95.0, 10.0 + (gpu_models * 25.0))
        
        # Update worker load
        self.worker_load.cpu_utilization = self.metrics['cpu_percent'] / 100.0
        self.worker_load.memory_utilization = self.metrics['memory_percent'] / 100.0
        self.worker_load.gpu_utilization = self.metrics['gpu_utilization'] / 100.0
        self.worker_load.active_tests = num_active
        
        # Update browser instance tracking
        if self.resource_pool and self.resource_pool.connection_pool:
            # Reset browser metrics
            for browser_type in self.browser_metrics:
                self.browser_metrics[browser_type] = {'utilization': 0.0, 'memory_usage': 0.0, 'active_models': 0}
                
            # Get active browser instances and their utilization
            self.browser_instances = {}
            for browser_id, browser_info in self.resource_pool.connection_pool.items():
                browser_type = browser_info.get('type', 'unknown')
                status = browser_info.get('status', 'unknown')
                active_models = browser_info.get('active_models', set())
                memory_usage = browser_info.get('memory_usage', 0.0)
                
                # Store browser instance information
                self.browser_instances[browser_id] = {
                    'type': browser_type,
                    'status': status,
                    'active_models': len(active_models),
                    'memory_usage': memory_usage
                }
                
                # Update browser type metrics if known type
                if browser_type in self.browser_metrics:
                    metrics = self.browser_metrics[browser_type]
                    metrics['active_models'] += len(active_models)
                    metrics['memory_usage'] += memory_usage
                    
                    # Calculate utilization based on active models and capacity
                    # Assuming each browser can handle 3 models at full capacity
                    model_capacity = 3
                    instance_utilization = min(1.0, len(active_models) / model_capacity)
                    metrics['utilization'] += instance_utilization
            
            # Normalize browser utilization by number of instances
            browser_counts = {}
            for browser_id, browser_info in self.browser_instances.items():
                browser_type = browser_info['type']
                if browser_type in browser_counts:
                    browser_counts[browser_type] += 1
                else:
                    browser_counts[browser_type] = 1
            
            # Update total browser metrics and capacities
            for browser_type, count in browser_counts.items():
                if count > 0 and browser_type in self.browser_metrics:
                    # Normalize utilization
                    self.browser_metrics[browser_type]['utilization'] /= count
                    
                    # Calculate capacity as percentage of slots available
                    active = self.browser_metrics[browser_type]['active_models']
                    capacity = max(0.0, 1.0 - (active / (count * 3)))  # Assume each browser can handle 3 models
                    self.browser_capacities[browser_type] = capacity
        
        # Update browser performance history from resource pool if available
        if self.resource_pool and hasattr(self.resource_pool, 'get_browser_performance_metrics'):
            try:
                browser_perf = await self.resource_pool.get_browser_performance_metrics()
                if browser_perf and isinstance(browser_perf, dict):
                    for browser_type, metrics in browser_perf.items():
                        if browser_type in self.browser_performance_history:
                            self.browser_performance_history[browser_type] = metrics
            except Exception as e:
                logger.error(f"Error getting browser performance metrics: {str(e)}")
        
        # Update load prediction based on recent request patterns
        self._update_load_prediction()
        
        # Include browser metrics in worker load
        if hasattr(self.worker_load, 'browser_metrics'):
            self.worker_load.browser_metrics = self.browser_metrics
            
        # Include browser capability scores in worker load
        if hasattr(self.worker_load, 'browser_capability_scores'):
            self.worker_load.browser_capability_scores = self.browser_capability_scores
            
        # Update custom properties if available
        if hasattr(self.worker_load, 'set_custom_property'):
            self.worker_load.set_custom_property('browser_capacities', self.browser_capacities)
            self.worker_load.set_custom_property('browser_instances', self.browser_instances)
            self.worker_load.set_custom_property('predicted_loads', self.load_prediction['predicted_loads'])
        
        logger.debug(f"Updated metrics for worker {self.worker_id}: CPU={self.metrics['cpu_percent']:.1f}%, "
                    f"MEM={self.metrics['memory_percent']:.1f}%, GPU={self.metrics['gpu_utilization']:.1f}%, "
                    f"Active: {num_active}")
    
    async def close(self) -> None:
        """Close the resource pool worker."""
        logger.info(f"Closing resource pool worker {self.worker_id}")
        
        # Close active models
        for test_id, model_info in list(self.active_models.items()):
            try:
                # Clean up model
                logger.info(f"Cleaning up model for test {test_id}")
                del self.active_models[test_id]
            except Exception as e:
                logger.error(f"Error cleaning up model for test {test_id}: {str(e)}")
        
        # Close sharded executions
        for test_id, execution in list(self.sharded_executions.items()):
            try:
                # Close execution
                await execution.close()
                del self.sharded_executions[test_id]
            except Exception as e:
                logger.error(f"Error closing sharded execution for test {test_id}: {str(e)}")
        
        # Close resource pool
        if self.resource_pool:
            try:
                if hasattr(self.resource_pool, 'close'):
                    await self.resource_pool.close()
            except Exception as e:
                logger.error(f"Error closing resource pool: {str(e)}")


class LoadBalancerResourcePoolBridge:
    """
    Integration between Load Balancer and Resource Pool Bridge.
    
    This class provides the integration layer between the Adaptive Load Balancer
    and the WebGPU/WebNN Resource Pool Bridge, enabling intelligent distribution
    of browser resources across worker nodes.
    """
    
    def __init__(
        self,
        db_path: Optional[str] = None,
        max_browsers_per_worker: int = 3,
        enable_fault_tolerance: bool = True,
        browser_preferences: Dict[str, str] = None,
        recovery_strategy: str = "progressive"
    ):
        """
        Initialize the load balancer resource pool bridge.
        
        Args:
            db_path: Path to SQLite database for performance tracking
            max_browsers_per_worker: Maximum number of browser instances per worker
            enable_fault_tolerance: Whether to enable fault tolerance features
            browser_preferences: Preferred browsers for different model types
            recovery_strategy: Recovery strategy for browser failures
        """
        self.db_path = db_path
        self.max_browsers_per_worker = max_browsers_per_worker
        self.enable_fault_tolerance = enable_fault_tolerance
        self.browser_preferences = browser_preferences or {
            'audio': 'firefox',
            'vision': 'chrome',
            'text_embedding': 'edge'
        }
        self.recovery_strategy = recovery_strategy
        
        # Load balancer
        self.load_balancer = LoadBalancerService(db_path=db_path)
        
        # Workers
        self.workers: Dict[str, ResourcePoolWorker] = {}
        
        # Test execution callbacks
        self.test_execution_callbacks: List[Callable[[str, Dict[str, Any]], None]] = []
        
        # Monitoring
        self.monitoring_interval = 10  # seconds
        self._stop_monitoring = anyio.Event()
        self.monitoring_task = None
        self._monitoring_group = None
        
        logger.info("LoadBalancerResourcePoolBridge initialized")
    
    async def start(self) -> None:
        """Start the load balancer resource pool bridge."""
        logger.info("Starting load balancer resource pool bridge")
        
        # Start load balancer
        self.load_balancer.start()
        
        # Start monitoring task
        self._stop_monitoring.clear()
        self._monitoring_group = anyio.create_task_group()
        await self._monitoring_group.__aenter__()
        self._monitoring_group.start_soon(self._monitoring_loop)
        
        logger.info("Load balancer resource pool bridge started")
    
    async def stop(self) -> None:
        """Stop the load balancer resource pool bridge."""
        logger.info("Stopping load balancer resource pool bridge")
        
        # Stop monitoring task
        if self._monitoring_group:
            self._stop_monitoring.set()
            self._monitoring_group.cancel_scope.cancel()
            await self._monitoring_group.__aexit__(None, None, None)
            self._monitoring_group = None
        
        # Stop load balancer
        self.load_balancer.stop()
        
        # Close workers
        for worker_id, worker in list(self.workers.items()):
            try:
                await worker.close()
            except Exception as e:
                logger.error(f"Error closing worker {worker_id}: {str(e)}")
        
        logger.info("Load balancer resource pool bridge stopped")
    
    async def register_worker(
        self,
        worker_id: str,
        max_browsers: Optional[int] = None,
        browser_preferences: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Register a worker with the load balancer.
        
        Args:
            worker_id: Unique identifier for the worker
            max_browsers: Maximum number of browser instances for this worker
            browser_preferences: Preferred browsers for this worker
        """
        logger.info(f"Registering worker {worker_id}")
        
        # Create worker
        worker = ResourcePoolWorker(
            worker_id=worker_id,
            max_browsers=max_browsers or self.max_browsers_per_worker,
            browser_preferences=browser_preferences or self.browser_preferences,
            enable_fault_tolerance=self.enable_fault_tolerance,
            recovery_strategy=self.recovery_strategy
        )
        
        # Initialize worker
        await worker.initialize()
        
        # Register with load balancer
        self.load_balancer.register_worker(worker_id, worker.capabilities)
        
        # Store worker
        self.workers[worker_id] = worker
        
        logger.info(f"Worker {worker_id} registered successfully")
    
    async def unregister_worker(self, worker_id: str) -> None:
        """
        Unregister a worker from the load balancer.
        
        Args:
            worker_id: Worker ID to unregister
        """
        if worker_id in self.workers:
            logger.info(f"Unregistering worker {worker_id}")
            
            # Unregister from load balancer
            self.load_balancer.unregister_worker(worker_id)
            
            # Close worker
            try:
                await self.workers[worker_id].close()
            except Exception as e:
                logger.error(f"Error closing worker {worker_id}: {str(e)}")
            
            # Remove worker
            del self.workers[worker_id]
            
            logger.info(f"Worker {worker_id} unregistered")
    
    async def submit_test(self, test_requirements: TestRequirements) -> str:
        """
        Submit a test for scheduling.
        
        Args:
            test_requirements: Test requirements
            
        Returns:
            Test ID
        """
        logger.info(f"Submitting test {test_requirements.test_id} ({test_requirements.model_type}/{test_requirements.model_id})")
        
        # Submit to load balancer
        test_id = self.load_balancer.submit_test(test_requirements)
        
        # Process assignments
        # TODO: Replace with task group - anyio task group for assignment processing
        
        return test_id
    
    async def _process_assignments(self) -> None:
        """Process test assignments from the load balancer."""
        # Get all assignments
        all_assignments = {}
        
        for worker_id, worker in self.workers.items():
            assignments = self.load_balancer.get_worker_assignments(worker_id)
            if assignments:
                all_assignments[worker_id] = assignments
        
        # Process each worker's assignments
        for worker_id, assignments in all_assignments.items():
            for assignment in assignments:
                # Skip if not assigned or already started
                if assignment.status != "assigned":
                    continue
                
                # Get worker
                worker = self.workers.get(worker_id)
                if not worker:
                    logger.warning(f"Worker {worker_id} not found for assignment {assignment.test_id}")
                    continue
                
                # Execute test
                # TODO: Replace with task group - anyio task group for test execution
    
    async def _execute_test(
        self,
        worker: ResourcePoolWorker,
        test_id: str,
        test_req: TestRequirements
    ) -> None:
        """
        Execute a test on a worker.
        
        Args:
            worker: Worker to execute the test
            test_id: Test ID
            test_req: Test requirements
        """
        logger.info(f"Executing test {test_id} on worker {worker.worker_id}")
        
        # Mark test as started
        self.load_balancer.update_assignment_status(test_id, "running")
        
        try:
            # Execute test
            result = await worker.execute_test(test_req)
            
            # Check result
            if result.get('status') == 'success':
                # Mark test as completed
                self.load_balancer.update_assignment_status(test_id, "completed", result)
                
                # Call callbacks
                for callback in self.test_execution_callbacks:
                    try:
                        callback(test_id, result)
                    except Exception as e:
                        logger.error(f"Error in test execution callback: {str(e)}")
            else:
                # Mark test as failed
                self.load_balancer.update_assignment_status(test_id, "failed", result)
            
        except Exception as e:
            logger.error(f"Error executing test {test_id}: {str(e)}")
            
            # Mark test as failed
            self.load_balancer.update_assignment_status(
                test_id, "failed", {"error": str(e)}
            )
    
    async def _monitoring_loop(self) -> None:
        """Background monitoring loop."""
        logger.info("Starting monitoring loop")
        
        while not self._stop_monitoring.is_set():
            try:
                # Update worker metrics
                for worker_id, worker in self.workers.items():
                    try:
                        await worker.update_metrics()
                        
                        # Update load balancer
                        self.load_balancer.update_worker_load(
                            worker_id, worker.worker_load
                        )
                    except Exception as e:
                        logger.error(f"Error updating metrics for worker {worker_id}: {str(e)}")
                
                # Process assignments
                await self._process_assignments()
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")
            
            # Wait for next interval
            try:
                await wait_for(
                    self._stop_monitoring.wait(), 
                    timeout=self.monitoring_interval
                )
            except TimeoutError:
                # Normal timeout, continue
                pass
        
        logger.info("Monitoring loop stopped")
    
    def register_test_execution_callback(
        self,
        callback: Callable[[str, Dict[str, Any]], None]
    ) -> None:
        """
        Register a callback for test execution results.
        
        Args:
            callback: Function to call with test ID and results
        """
        self.test_execution_callbacks.append(callback)
    
    async def get_worker_browser_capacities(self, worker_id: str) -> Dict[str, float]:
        """
        Get browser capacities for a worker.
        
        Args:
            worker_id: Worker ID
            
        Returns:
            Dictionary of browser capacities (browser_type -> capacity percentage)
        """
        if worker_id in self.workers:
            return self.workers[worker_id].browser_capacities
        return {}
    
    async def get_model_performance_history(
        self,
        model_type: str = None,
        time_range: str = "7d"
    ) -> Dict[str, Any]:
        """
        Get performance history for browser models.
        
        Args:
            model_type: Optional filter for model type
            time_range: Time range for history (e.g., "7d" for 7 days)
            
        Returns:
            Performance history data
        """
        # Collect performance data from all workers
        performance_data = {}
        
        for worker_id, worker in self.workers.items():
            if worker.resource_pool:
                try:
                    history = await worker.resource_pool.get_performance_history(
                        model_type=model_type,
                        time_range=time_range
                    )
                    
                    if history and "error" not in history:
                        performance_data[worker_id] = history
                except Exception as e:
                    logger.error(f"Error getting performance history for worker {worker_id}: {str(e)}")
        
        return {
            "performance_data": performance_data,
            "time_range": time_range,
            "model_type": model_type
        }
    
    async def analyze_system_performance(self) -> Dict[str, Any]:
        """
        Analyze system performance and generate recommendations.
        
        Returns:
            Analysis and recommendations
        """
        # Get performance history
        history = await self.get_model_performance_history(time_range="7d")
        
        # Analyze worker performance
        worker_analysis = {}
        
        for worker_id, worker_data in history.get("performance_data", {}).items():
            if worker_id not in self.workers:
                continue
                
            worker = self.workers[worker_id]
            
            # Analyze browser preferences based on model type performance
            model_type_performance = {}
            for model_type, performance in worker_data.get("model_types", {}).items():
                # Calculate average latency by browser
                browser_latency = {}
                for browser, stats in performance.get("browsers", {}).items():
                    if "avg_latency" in stats:
                        browser_latency[browser] = stats["avg_latency"]
                
                if browser_latency:
                    # Find best browser for this model type
                    best_browser = min(browser_latency.items(), key=lambda x: x[1])[0]
                    current_browser = worker.browser_preferences.get(model_type)
                    
                    model_type_performance[model_type] = {
                        "best_browser": best_browser,
                        "current_browser": current_browser,
                        "latency_by_browser": browser_latency,
                        "change_recommended": best_browser != current_browser
                    }
            
            # Generate worker recommendations
            worker_analysis[worker_id] = {
                "model_type_performance": model_type_performance,
                "browser_capacities": worker.browser_capacities,
                "recommendations": {
                    "browser_preferences": {
                        model_type: perf["best_browser"]
                        for model_type, perf in model_type_performance.items()
                        if perf["change_recommended"]
                    }
                }
            }
        
        # Generate system recommendations
        system_recommendations = {
            "worker_load_balance": self._analyze_worker_load_balance(),
            "browser_type_distribution": self._analyze_browser_type_distribution(),
            "model_type_assignment": self._analyze_model_type_assignment()
        }
        
        return {
            "worker_analysis": worker_analysis,
            "system_recommendations": system_recommendations,
            "timestamp": datetime.now().isoformat()
        }
    
    def _analyze_worker_load_balance(self) -> Dict[str, Any]:
        """
        Analyze worker load balance.
        
        Returns:
            Analysis of worker load balance
        """
        worker_loads = {}
        for worker_id, worker in self.workers.items():
            worker_loads[worker_id] = worker.worker_load.calculate_load_score()
        
        if not worker_loads:
            return {"status": "no_workers"}
        
        # Calculate statistics
        avg_load = sum(worker_loads.values()) / len(worker_loads)
        min_load = min(worker_loads.values())
        max_load = max(worker_loads.values())
        imbalance = max_load - min_load
        
        # Get highest and lowest loaded workers
        highest_worker = max(worker_loads.items(), key=lambda x: x[1])[0]
        lowest_worker = min(worker_loads.items(), key=lambda x: x[1])[0]
        
        return {
            "status": "imbalanced" if imbalance > 0.3 else "balanced",
            "avg_load": avg_load,
            "min_load": min_load,
            "max_load": max_load,
            "imbalance": imbalance,
            "highest_worker": highest_worker,
            "lowest_worker": lowest_worker,
            "recommendation": "rebalance" if imbalance > 0.3 else "maintain"
        }
    
    def _analyze_browser_type_distribution(self) -> Dict[str, Any]:
        """
        Analyze browser type distribution.
        
        Returns:
            Analysis of browser type distribution
        """
        browser_counts = {'chrome': 0, 'firefox': 0, 'edge': 0}
        browser_active = {'chrome': 0, 'firefox': 0, 'edge': 0}
        
        # Count browsers across all workers
        for worker_id, worker in self.workers.items():
            if not worker.resource_pool or not worker.resource_pool.connection_pool:
                continue
                
            for browser_id, browser in worker.resource_pool.connection_pool.items():
                browser_type = browser.get('type', 'unknown')
                if browser_type in browser_counts:
                    browser_counts[browser_type] += 1
                    if browser.get('status') == 'ready':
                        active_models = len(browser.get('active_models', set()))
                        browser_active[browser_type] += active_models
        
        # Calculate utilization percentages
        browser_utilization = {}
        for browser_type, count in browser_counts.items():
            if count > 0:
                active = browser_active[browser_type]
                # Utilization as percentage of capacity
                utilization = min(1.0, active / (count * 3))  # Assume each browser can handle 3 models
                browser_utilization[browser_type] = utilization
        
        # Generate recommendations
        recommendations = {}
        
        for browser_type, utilization in browser_utilization.items():
            if utilization > 0.8:
                recommendations[browser_type] = "increase"
            elif utilization < 0.2 and browser_counts[browser_type] > 1:
                recommendations[browser_type] = "decrease"
            else:
                recommendations[browser_type] = "maintain"
        
        return {
            "browser_counts": browser_counts,
            "browser_active_models": browser_active,
            "browser_utilization": browser_utilization,
            "recommendations": recommendations
        }
    
    def _analyze_model_type_assignment(self) -> Dict[str, Any]:
        """
        Analyze model type assignment patterns.
        
        Returns:
            Analysis of model type assignment patterns
        """
        # Count model types across all workers
        model_type_counts = {}
        
        for worker_id, worker in self.workers.items():
            for test_id, model_info in worker.active_models.items():
                model_type = model_info['test_req'].model_type
                
                if model_type not in model_type_counts:
                    model_type_counts[model_type] = 0
                    
                model_type_counts[model_type] += 1
        
        # Get browser preferences for each model type
        browser_preferences = {}
        
        for worker_id, worker in self.workers.items():
            for model_type, browser in worker.browser_preferences.items():
                if model_type not in browser_preferences:
                    browser_preferences[model_type] = set()
                    
                browser_preferences[model_type].add(browser)
        
        # Check if all workers have consistent browser preferences
        consistent_preferences = {}
        
        for model_type, browsers in browser_preferences.items():
            consistent_preferences[model_type] = len(browsers) == 1
        
        return {
            "model_type_counts": model_type_counts,
            "browser_preferences": {
                model_type: list(browsers)
                for model_type, browsers in browser_preferences.items()
            },
            "consistent_preferences": consistent_preferences,
            "recommendations": {
                model_type: "standardize_browser_preference"
                for model_type, consistent in consistent_preferences.items()
                if not consistent
            }
        }
    
    async def apply_optimization_recommendations(self, recommendations: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply optimization recommendations.
        
        Args:
            recommendations: Optimization recommendations
            
        Returns:
            Results of applying recommendations
        """
        logger.info("Applying optimization recommendations")
        
        results = {
            "applied": [],
            "failed": [],
            "ignored": []
        }
        
        # Apply worker-specific recommendations
        for worker_id, worker_recs in recommendations.get("worker_analysis", {}).items():
            if worker_id not in self.workers:
                results["ignored"].append({
                    "worker_id": worker_id,
                    "reason": "worker_not_found"
                })
                continue
                
            worker = self.workers[worker_id]
            
            # Apply browser preferences
            browser_prefs = worker_recs.get("recommendations", {}).get("browser_preferences", {})
            if browser_prefs:
                try:
                    # Update worker browser preferences
                    for model_type, browser in browser_prefs.items():
                        worker.browser_preferences[model_type] = browser
                    
                    # Apply to resource pool if available
                    if worker.resource_pool:
                        await worker.resource_pool.apply_performance_optimizations({
                            "browser_preferences": browser_prefs
                        })
                    
                    results["applied"].append({
                        "worker_id": worker_id,
                        "type": "browser_preferences",
                        "changes": browser_prefs
                    })
                except Exception as e:
                    logger.error(f"Error applying browser preferences for worker {worker_id}: {str(e)}")
                    results["failed"].append({
                        "worker_id": worker_id,
                        "type": "browser_preferences",
                        "error": str(e)
                    })
        
        # Apply system-wide recommendations
        system_recs = recommendations.get("system_recommendations", {})
        
        # Apply worker load balance recommendations
        load_balance_rec = system_recs.get("worker_load_balance", {}).get("recommendation")
        if load_balance_rec == "rebalance":
            try:
                # Trigger load balancer rebalance
                self.load_balancer.rebalance()
                
                results["applied"].append({
                    "type": "load_balance",
                    "action": "rebalanced"
                })
            except Exception as e:
                logger.error(f"Error rebalancing loads: {str(e)}")
                results["failed"].append({
                    "type": "load_balance",
                    "error": str(e)
                })
        
        # Apply browser type distribution recommendations
        browser_recs = system_recs.get("browser_type_distribution", {}).get("recommendations", {})
        if browser_recs:
            try:
                # Apply to all workers with resource pools
                for worker_id, worker in self.workers.items():
                    if worker.resource_pool:
                        # Determine target connection pool size
                        current_size = len(worker.resource_pool.connection_pool)
                        target_size = current_size
                        
                        for browser_type, action in browser_recs.items():
                            if action == "increase":
                                # Increase connections for this browser type
                                target_size += 1
                            elif action == "decrease":
                                # Decrease connections for this browser type
                                target_size = max(1, target_size - 1)
                        
                        # Apply scaling if needed
                        if target_size != current_size:
                            await worker.resource_pool._scale_connection_pool(target_size)
                
                results["applied"].append({
                    "type": "browser_distribution",
                    "actions": browser_recs
                })
            except Exception as e:
                logger.error(f"Error applying browser distribution recommendations: {str(e)}")
                results["failed"].append({
                    "type": "browser_distribution",
                    "error": str(e)
                })
        
        logger.info(f"Applied optimization recommendations: {len(results['applied'])} applied, {len(results['failed'])} failed, {len(results['ignored'])} ignored")
        
        return results


# Factory function for creating the bridge
def create_bridge(config: Dict[str, Any]) -> LoadBalancerResourcePoolBridge:
    """
    Create a load balancer resource pool bridge.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        LoadBalancerResourcePoolBridge instance
    """
    # Extract config values
    db_path = config.get("db_path")
    max_browsers_per_worker = config.get("max_browsers_per_worker", 3)
    enable_fault_tolerance = config.get("enable_fault_tolerance", True)
    recovery_strategy = config.get("recovery_strategy", "progressive")
    browser_preferences = config.get("browser_preferences", {
        'audio': 'firefox',
        'vision': 'chrome',
        'text_embedding': 'edge'
    })
    
    # Create bridge
    bridge = LoadBalancerResourcePoolBridge(
        db_path=db_path,
        max_browsers_per_worker=max_browsers_per_worker,
        enable_fault_tolerance=enable_fault_tolerance,
        browser_preferences=browser_preferences,
        recovery_strategy=recovery_strategy
    )
    
    return bridge