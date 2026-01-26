#!/usr/bin/env python3
"""
Resource Pool Plugin for Distributed Testing Framework

This plugin integrates the WebGPU/WebNN resource pool with the distributed testing
framework, enabling fault-tolerant model execution across distributed workers.

The plugin provides:
1. Task scheduling across browser instances
2. Fault tolerance with automatic recovery
3. Performance tracking and optimization
4. State replication for resilience
5. Metrics collection and reporting

Usage:
    from distributed_testing.plugins.resource_pool_plugin import ResourcePoolPlugin
    
    # Create plugin with resource pool integration
    plugin = ResourcePoolPlugin(
        integration=resource_pool_integration, 
        fault_tolerance_level="high"
    )
    
    # Initialize plugin
    await plugin.initialize()
    
    # Execute a task
    result = await plugin.execute_task({
        "action": "run_model",
        "model_name": "bert-base-uncased",
        "model_type": "text_embedding",
        "platform": "webgpu",
        "inputs": inputs
    })
    
    # Get metrics
    metrics = await plugin.get_metrics()
    
    # Shutdown plugin
    await plugin.shutdown()
"""

import os
import sys
import json
import time
import anyio
import logging
import traceback
from typing import Dict, List, Any, Optional, Union, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import distributed testing framework components
from distributed_testing.plugin_base import PluginBase
from distributed_testing.circuit_breaker import CircuitBreaker
from distributed_testing.state_manager import StateManager
from distributed_testing.worker_registry import WorkerRegistry
from distributed_testing.transaction_log import TransactionLog

class ResourcePoolPlugin(PluginBase):
    """
    Plugin for integrating WebGPU/WebNN resource pool with distributed testing framework
    """
    
    def __init__(self, 
                 integration,
                 fault_tolerance_level: str = "medium",
                 recovery_strategy: str = "progressive",
                 state_storage_path: str = None,
                 plugin_id: str = "resource-pool-plugin"):
        """
        Initialize resource pool plugin
        
        Args:
            integration: WebGPU/WebNN resource pool integration
            fault_tolerance_level: Fault tolerance level (low, medium, high, critical)
            recovery_strategy: Recovery strategy (simple, progressive, parallel, coordinated)
            state_storage_path: Path for state persistence
            plugin_id: Unique identifier for this plugin
        """
        super().__init__(plugin_id)
        self.integration = integration
        self.fault_tolerance_level = fault_tolerance_level
        self.recovery_strategy = recovery_strategy
        
        # Plugin state
        self.initialized = False
        self.metrics = {
            "tasks_executed": 0,
            "successful_tasks": 0,
            "failed_tasks": 0,
            "recovery_attempts": 0,
            "successful_recoveries": 0,
            "execution_times_ms": [],
            "browser_usage": {}
        }
        
        # Create state storage path if provided
        if state_storage_path:
            self.state_storage_path = state_storage_path
            os.makedirs(state_storage_path, exist_ok=True)
        else:
            self.state_storage_path = None
        
        # Create distributed testing components
        self.state_manager = StateManager(plugin_id)
        self.worker_registry = WorkerRegistry(plugin_id)
        self.transaction_log = TransactionLog(plugin_id)
        
        # Create circuit breakers for fault tolerance
        self.circuit_breakers = {}
        
        # Reference to active models
        self.active_models = {}
        
        logger.info(f"ResourcePoolPlugin initialized with {fault_tolerance_level} fault tolerance")
    
    async def initialize(self) -> bool:
        """
        Initialize plugin and register with distributed testing framework
        
        Returns:
            Success status
        """
        try:
            logger.info("Initializing ResourcePoolPlugin")
            
            # Register available browsers as workers
            browser_info = self._get_browser_info()
            for i, (browser, info) in enumerate(browser_info.items()):
                worker_id = f"browser-{i}"
                await self.worker_registry.register(worker_id, {
                    "type": browser,
                    "capabilities": info.get("capabilities", ["webgpu"]),
                    "status": "ready",
                    "startup_time": time.time()
                })
            
            # Create circuit breakers for each browser
            for browser in browser_info.keys():
                self.circuit_breakers[browser] = CircuitBreaker(
                    failure_threshold=3,
                    recovery_timeout=30,
                    half_open_timeout=5,
                    name=f"{browser}-circuit"
                )
            
            # Initialize state
            await self.state_manager.update_state("plugin_state", {
                "initialized": True,
                "fault_tolerance_level": self.fault_tolerance_level,
                "recovery_strategy": self.recovery_strategy,
                "browser_info": browser_info
            })
            
            # Record initialization in transaction log
            await self.transaction_log.append({
                "action": "initialize",
                "timestamp": time.time(),
                "fault_tolerance_level": self.fault_tolerance_level,
                "recovery_strategy": self.recovery_strategy
            })
            
            self.initialized = True
            logger.info("ResourcePoolPlugin initialization complete")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing ResourcePoolPlugin: {e}")
            traceback.print_exc()
            return False
    
    async def execute_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a task using the resource pool
        
        Args:
            task_data: Task definition and parameters
            
        Returns:
            Task execution results
        """
        if not self.initialized:
            return {"success": False, "error": "Plugin not initialized"}
        
        # Record task execution start
        self.metrics["tasks_executed"] += 1
        start_time = time.time()
        
        try:
            # Record in transaction log
            await self.transaction_log.append({
                "action": "execute_task",
                "task_type": task_data.get("action", "unknown"),
                "timestamp": time.time()
            })
            
            # Determine action type
            action = task_data.get("action", "run_model")
            
            if action == "run_model":
                # Extract model parameters
                model_name = task_data.get("model_name")
                model_type = task_data.get("model_type")
                platform = task_data.get("platform", "webgpu")
                inputs = task_data.get("inputs", {})
                
                # Handle model execution with fault tolerance
                result = await self._execute_model_with_fault_tolerance(
                    model_name=model_name,
                    model_type=model_type,
                    platform=platform,
                    inputs=inputs
                )
                
            elif action == "run_batch":
                # Execute batch inference
                model_name = task_data.get("model_name")
                model_type = task_data.get("model_type")
                platform = task_data.get("platform", "webgpu")
                batch_inputs = task_data.get("batch_inputs", [])
                
                # Handle batch execution with fault tolerance
                result = await self._execute_batch_with_fault_tolerance(
                    model_name=model_name,
                    model_type=model_type,
                    platform=platform,
                    batch_inputs=batch_inputs
                )
                
            elif action == "run_concurrent":
                # Execute concurrent models
                model_configs = task_data.get("model_configs", [])
                
                # Handle concurrent execution with fault tolerance
                result = await self._execute_concurrent_with_fault_tolerance(
                    model_configs=model_configs
                )
                
            else:
                result = {
                    "success": False,
                    "error": f"Unknown action: {action}"
                }
            
            # Calculate execution time
            execution_time = (time.time() - start_time) * 1000  # ms
            
            # Update metrics
            if result.get("success", False):
                self.metrics["successful_tasks"] += 1
            else:
                self.metrics["failed_tasks"] += 1
                
            self.metrics["execution_times_ms"].append(execution_time)
            
            # Update browser usage metrics
            browser = result.get("browser", "unknown")
            if browser in self.metrics["browser_usage"]:
                self.metrics["browser_usage"][browser] += 1
            else:
                self.metrics["browser_usage"][browser] = 1
            
            # Add execution time to result
            result["execution_time"] = execution_time
            
            # Record task completion in transaction log
            await self.transaction_log.append({
                "action": "task_complete",
                "task_type": task_data.get("action", "unknown"),
                "success": result.get("success", False),
                "execution_time_ms": execution_time,
                "timestamp": time.time()
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing task: {e}")
            traceback.print_exc()
            
            # Update metrics
            self.metrics["failed_tasks"] += 1
            
            # Calculate execution time
            execution_time = (time.time() - start_time) * 1000  # ms
            
            # Record error in transaction log
            await self.transaction_log.append({
                "action": "task_error",
                "task_type": task_data.get("action", "unknown"),
                "error": str(e),
                "timestamp": time.time()
            })
            
            return {
                "success": False,
                "error": str(e),
                "execution_time": execution_time
            }
    
    async def _execute_model_with_fault_tolerance(self, 
                                                model_name: str, 
                                                model_type: str, 
                                                platform: str, 
                                                inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a model with fault tolerance
        
        Args:
            model_name: Name of the model
            model_type: Type of model
            platform: Platform to run on
            inputs: Model inputs
            
        Returns:
            Model execution results
        """
        # Determine optimal browser for this model type
        browser = self._get_optimal_browser(model_type, platform)
        
        # Configure circuit breaker for this browser
        circuit_breaker = self.circuit_breakers.get(browser)
        
        try:
            # Use circuit breaker to execute model if available
            if circuit_breaker:
                # Execute with circuit breaker for fault isolation
                async def execute_model():
                    # Configure hardware preferences
                    hardware_preferences = {
                        'priority_list': [platform, 'cpu'],
                        'model_family': model_type,
                        'browser': browser
                    }
                    
                    # Get model from resource pool
                    model = self.integration.get_model(
                        model_type=model_type,
                        model_name=model_name,
                        hardware_preferences=hardware_preferences
                    )
                    
                    # Cache model reference for recovery
                    model_key = f"{model_type}:{model_name}"
                    self.active_models[model_key] = model
                    
                    # Run inference
                    return model(inputs)
                
                # Execute with circuit breaker
                result = await circuit_breaker.execute(execute_model)
                
                # Record success in circuit breaker
                circuit_breaker.record_success()
                
                return result
            else:
                # Execute normally without circuit breaker
                # Configure hardware preferences
                hardware_preferences = {
                    'priority_list': [platform, 'cpu'],
                    'model_family': model_type,
                    'browser': browser
                }
                
                # Get model from resource pool
                model = self.integration.get_model(
                    model_type=model_type,
                    model_name=model_name,
                    hardware_preferences=hardware_preferences
                )
                
                # Cache model reference for recovery
                model_key = f"{model_type}:{model_name}"
                self.active_models[model_key] = model
                
                # Run inference
                return model(inputs)
            
        except Exception as e:
            logger.error(f"Error executing model {model_name}: {e}")
            
            # Record failure in circuit breaker if available
            if circuit_breaker:
                circuit_breaker.record_failure()
            
            # Attempt recovery based on fault tolerance level
            self.metrics["recovery_attempts"] += 1
            
            try:
                if self.fault_tolerance_level in ["medium", "high", "critical"]:
                    # Attempt recovery
                    recovery_result = await self._recover_model_execution(
                        model_name=model_name,
                        model_type=model_type,
                        platform=platform,
                        inputs=inputs,
                        original_browser=browser,
                        original_error=e
                    )
                    
                    if recovery_result.get("success", False):
                        self.metrics["successful_recoveries"] += 1
                        recovery_result["recovery"] = {
                            "recovered": True,
                            "strategy": self.recovery_strategy,
                            "original_error": str(e)
                        }
                        return recovery_result
                    else:
                        # Recovery failed
                        return {
                            "success": False,
                            "error": f"Execution failed and recovery unsuccessful: {e}",
                            "recovery": {
                                "recovered": False,
                                "strategy": self.recovery_strategy,
                                "original_error": str(e),
                                "recovery_error": recovery_result.get("error")
                            }
                        }
                else:
                    # No recovery for low fault tolerance
                    return {
                        "success": False,
                        "error": f"Error executing model: {e}"
                    }
            except Exception as recovery_error:
                logger.error(f"Error during recovery: {recovery_error}")
                return {
                    "success": False,
                    "error": f"Execution failed and recovery error: {recovery_error}",
                    "original_error": str(e)
                }
    
    async def _recover_model_execution(self,
                                      model_name: str,
                                      model_type: str,
                                      platform: str,
                                      inputs: Dict[str, Any],
                                      original_browser: str,
                                      original_error: Exception) -> Dict[str, Any]:
        """
        Attempt to recover model execution after failure
        
        Args:
            model_name: Name of the model
            model_type: Type of model
            platform: Platform to run on
            inputs: Model inputs
            original_browser: Original browser that failed
            original_error: Original error that occurred
            
        Returns:
            Recovery results
        """
        logger.info(f"Attempting to recover model execution for {model_name} using {self.recovery_strategy} strategy")
        
        try:
            # Apply different recovery strategies based on configuration
            if self.recovery_strategy == "simple":
                # Simple retry on the same browser
                hardware_preferences = {
                    'priority_list': [platform, 'cpu'],
                    'model_family': model_type,
                    'browser': original_browser
                }
                
                # Get model from resource pool
                model = self.integration.get_model(
                    model_type=model_type,
                    model_name=model_name,
                    hardware_preferences=hardware_preferences
                )
                
                # Run inference
                result = model(inputs)
                
                return result
                
            elif self.recovery_strategy == "progressive":
                # Try different browser
                alternate_browser = self._get_alternate_browser(original_browser, model_type)
                
                hardware_preferences = {
                    'priority_list': [platform, 'cpu'],
                    'model_family': model_type,
                    'browser': alternate_browser
                }
                
                # Get model from resource pool
                model = self.integration.get_model(
                    model_type=model_type,
                    model_name=model_name,
                    hardware_preferences=hardware_preferences
                )
                
                # Run inference
                result = model(inputs)
                
                # Mark as recovered using alternate browser
                if isinstance(result, dict):
                    result["alternate_browser"] = alternate_browser
                
                return result
                
            elif self.recovery_strategy == "parallel":
                # Try multiple browsers in parallel
                browsers = self._get_all_available_browsers(exclude=[original_browser])
                
                if not browsers:
                    return {
                        "success": False,
                        "error": "No alternate browsers available for recovery"
                    }
                
                # Create tasks for each browser
                send_stream, receive_stream = anyio.create_memory_object_stream(len(browsers))
                
                for browser in browsers:
                    async def run_on_browser(b):
                        try:
                            hardware_preferences = {
                                'priority_list': [platform, 'cpu'],
                                'model_family': model_type,
                                'browser': b
                            }
                            
                            # Get model from resource pool
                            model = self.integration.get_model(
                                model_type=model_type,
                                model_name=model_name,
                                hardware_preferences=hardware_preferences
                            )
                            
                            # Run inference
                            result = model(inputs)
                            return {"browser": b, "result": result, "success": True}
                        except Exception as e:
                            return {"browser": b, "error": str(e), "success": False}
                    
                async def run_and_send(browser_name: str) -> None:
                    task_result = await run_on_browser(browser_name)
                    await send_stream.send(task_result)

                async with anyio.create_task_group() as task_group:
                    for browser in browsers:
                        task_group.start_soon(run_and_send, browser)

                    # Wait for first successful result or all failures
                    for _ in browsers:
                        task_result = await receive_stream.receive()
                        if task_result.get("success", False):
                            result = task_result["result"]
                            result["recovery_browser"] = task_result["browser"]
                            task_group.cancel_scope.cancel()
                            return result
                
                # All parallel recovery attempts failed
                return {
                    "success": False,
                    "error": "All parallel recovery attempts failed"
                }
                
            elif self.recovery_strategy == "coordinated":
                # Use state management for coordinated recovery
                await self.state_manager.update_state(f"model_recovery_{model_name}", {
                    "in_progress": True,
                    "original_browser": original_browser,
                    "original_error": str(original_error),
                    "timestamp": time.time()
                })
                
                # Try alternate browser
                alternate_browser = self._get_alternate_browser(original_browser, model_type)
                
                hardware_preferences = {
                    'priority_list': [platform, 'cpu'],
                    'model_family': model_type,
                    'browser': alternate_browser
                }
                
                # Get model from resource pool
                model = self.integration.get_model(
                    model_type=model_type,
                    model_name=model_name,
                    hardware_preferences=hardware_preferences
                )
                
                # Run inference
                result = model(inputs)
                
                # Update recovery state
                await self.state_manager.update_state(f"model_recovery_{model_name}", {
                    "in_progress": False,
                    "success": True,
                    "recovery_browser": alternate_browser,
                    "timestamp": time.time()
                })
                
                # Mark as recovered using alternate browser
                if isinstance(result, dict):
                    result["recovery_browser"] = alternate_browser
                
                return result
            
            else:
                # Unknown strategy, fall back to simple retry
                hardware_preferences = {
                    'priority_list': [platform, 'cpu'],
                    'model_family': model_type,
                    'browser': original_browser
                }
                
                # Get model from resource pool
                model = self.integration.get_model(
                    model_type=model_type,
                    model_name=model_name,
                    hardware_preferences=hardware_preferences
                )
                
                # Run inference
                return model(inputs)
                
        except Exception as e:
            logger.error(f"Error during recovery: {e}")
            return {
                "success": False,
                "error": f"Recovery failed: {e}"
            }
    
    async def _execute_batch_with_fault_tolerance(self,
                                                model_name: str,
                                                model_type: str,
                                                platform: str,
                                                batch_inputs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Execute batch inference with fault tolerance
        
        Args:
            model_name: Name of the model
            model_type: Type of model
            platform: Platform to run on
            batch_inputs: List of model inputs
            
        Returns:
            Batch execution results
        """
        logger.info(f"Executing batch inference for {model_name} with {len(batch_inputs)} inputs")
        
        try:
            # Determine optimal browser
            browser = self._get_optimal_browser(model_type, platform)
            
            # Configure hardware preferences
            hardware_preferences = {
                'priority_list': [platform, 'cpu'],
                'model_family': model_type,
                'browser': browser
            }
            
            # Get model from resource pool
            model = self.integration.get_model(
                model_type=model_type,
                model_name=model_name,
                hardware_preferences=hardware_preferences
            )
            
            # Cache model reference for recovery
            model_key = f"{model_type}:{model_name}"
            self.active_models[model_key] = model
            
            # Run batch inference
            batch_results = []
            
            # Process batch with fault tolerance
            for i, inputs in enumerate(batch_inputs):
                try:
                    # Execute single input
                    result = model(inputs)
                    batch_results.append(result)
                except Exception as e:
                    logger.error(f"Error in batch item {i}: {e}")
                    
                    # Attempt recovery if fault tolerance enabled
                    if self.fault_tolerance_level in ["medium", "high", "critical"]:
                        self.metrics["recovery_attempts"] += 1
                        
                        try:
                            # Recover single input
                            recovery_result = await self._recover_model_execution(
                                model_name=model_name,
                                model_type=model_type,
                                platform=platform,
                                inputs=inputs,
                                original_browser=browser,
                                original_error=e
                            )
                            
                            if recovery_result.get("success", False):
                                self.metrics["successful_recoveries"] += 1
                                recovery_result["recovery"] = {
                                    "recovered": True,
                                    "strategy": self.recovery_strategy,
                                    "original_error": str(e)
                                }
                                batch_results.append(recovery_result)
                            else:
                                # Add error result
                                batch_results.append({
                                    "success": False,
                                    "error": f"Execution failed and recovery unsuccessful: {e}",
                                    "recovery": {
                                        "recovered": False,
                                        "strategy": self.recovery_strategy,
                                        "original_error": str(e)
                                    }
                                })
                        except Exception as recovery_error:
                            logger.error(f"Recovery error in batch item {i}: {recovery_error}")
                            batch_results.append({
                                "success": False,
                                "error": f"Execution failed and recovery error: {recovery_error}",
                                "original_error": str(e)
                            })
                    else:
                        # No recovery, just add error result
                        batch_results.append({
                            "success": False,
                            "error": f"Error in batch item {i}: {e}"
                        })
            
            # Calculate success rate
            success_count = sum(1 for r in batch_results if r.get("success", False))
            success_rate = success_count / len(batch_inputs) if batch_inputs else 0
            
            return {
                "success": success_rate > 0.5,  # Success if more than half succeeded
                "batch_results": batch_results,
                "batch_size": len(batch_inputs),
                "success_count": success_count,
                "success_rate": success_rate,
                "browser": browser
            }
            
        except Exception as e:
            logger.error(f"Error in batch execution: {e}")
            return {
                "success": False,
                "error": f"Batch execution failed: {e}"
            }
    
    async def _execute_concurrent_with_fault_tolerance(self,
                                                     model_configs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Execute multiple models concurrently with fault tolerance
        
        Args:
            model_configs: List of model configurations
            
        Returns:
            Concurrent execution results
        """
        logger.info(f"Executing {len(model_configs)} models concurrently")
        
        try:
            # Prepare models for execution
            models = []
            model_inputs = []
            
            for config in model_configs:
                model_name = config.get("model_name")
                model_type = config.get("model_type")
                platform = config.get("platform", "webgpu")
                inputs = config.get("inputs", {})
                
                # Determine browser
                browser = config.get("browser", self._get_optimal_browser(model_type, platform))
                
                # Configure hardware preferences
                hardware_preferences = {
                    'priority_list': [platform, 'cpu'],
                    'model_family': model_type,
                    'browser': browser
                }
                
                # Get model from resource pool
                model = self.integration.get_model(
                    model_type=model_type,
                    model_name=model_name,
                    hardware_preferences=hardware_preferences
                )
                
                if model:
                    models.append(model)
                    model_inputs.append((model.model_id, inputs))
                    
                    # Cache model reference
                    model_key = f"{model_type}:{model_name}"
                    self.active_models[model_key] = model
            
            # Execute models concurrently
            concurrent_results = self.integration.execute_concurrent(model_inputs)
            
            # Process results and handle failures
            processed_results = []
            
            for i, result in enumerate(concurrent_results):
                success = result.get("success", False) or result.get("status") == "success"
                
                if success:
                    processed_results.append(result)
                else:
                    # Handle failure with recovery if enabled
                    if i < len(model_configs) and self.fault_tolerance_level in ["medium", "high", "critical"]:
                        config = model_configs[i]
                        model_name = config.get("model_name")
                        model_type = config.get("model_type")
                        platform = config.get("platform", "webgpu")
                        inputs = config.get("inputs", {})
                        browser = config.get("browser", "unknown")
                        error = result.get("error", "Unknown error")
                        
                        self.metrics["recovery_attempts"] += 1
                        
                        try:
                            # Attempt recovery
                            recovery_result = await self._recover_model_execution(
                                model_name=model_name,
                                model_type=model_type,
                                platform=platform,
                                inputs=inputs,
                                original_browser=browser,
                                original_error=Exception(error)
                            )
                            
                            if recovery_result.get("success", False):
                                self.metrics["successful_recoveries"] += 1
                                recovery_result["recovery"] = {
                                    "recovered": True,
                                    "strategy": self.recovery_strategy,
                                    "original_error": error
                                }
                                processed_results.append(recovery_result)
                            else:
                                # Add original failed result
                                processed_results.append(result)
                        except Exception as recovery_error:
                            logger.error(f"Recovery error: {recovery_error}")
                            result["recovery_error"] = str(recovery_error)
                            processed_results.append(result)
                    else:
                        # No recovery, add original result
                        processed_results.append(result)
            
            # Calculate success rate
            success_count = sum(1 for r in processed_results if r.get("success", False) or r.get("status") == "success")
            success_rate = success_count / len(model_configs) if model_configs else 0
            
            return {
                "success": success_rate > 0.5,  # Success if more than half succeeded
                "model_results": processed_results,
                "model_count": len(model_configs),
                "success_count": success_count,
                "success_rate": success_rate
            }
            
        except Exception as e:
            logger.error(f"Error in concurrent execution: {e}")
            return {
                "success": False,
                "error": f"Concurrent execution failed: {e}"
            }
    
    def _get_browser_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about available browsers
        
        Returns:
            Dictionary with browser information
        """
        # Start with default browser information
        browser_info = {
            "chrome": {
                "capabilities": ["webgpu"],
                "strengths": ["vision", "multimodal", "parallel"],
                "status": "ready"
            },
            "firefox": {
                "capabilities": ["webgpu"],
                "strengths": ["audio", "speech", "compute_shaders"],
                "status": "ready"
            },
            "edge": {
                "capabilities": ["webgpu", "webnn"],
                "strengths": ["text", "embedding", "webnn"],
                "status": "ready"
            }
        }
        
        # If integration provides browser information, use it
        if hasattr(self.integration, "get_browser_info"):
            try:
                integration_browser_info = self.integration.get_browser_info()
                if integration_browser_info:
                    # Merge with default info
                    for browser, info in integration_browser_info.items():
                        if browser in browser_info:
                            browser_info[browser].update(info)
                        else:
                            browser_info[browser] = info
            except Exception as e:
                logger.warning(f"Error getting browser info from integration: {e}")
        
        return browser_info
    
    def _get_optimal_browser(self, model_type: str, platform: str) -> str:
        """
        Get optimal browser for model type and platform
        
        Args:
            model_type: Type of model
            platform: Platform to run on
            
        Returns:
            Browser name
        """
        # Optimal browser mappings
        if platform == "webnn":
            # Edge has best WebNN support
            return "edge"
        
        # Browser strengths by model type
        browser_strengths = {
            "audio": "firefox",  # Firefox has better compute shader performance for audio
            "vision": "chrome",  # Chrome has good WebGPU support for vision models
            "text_embedding": "edge",  # Edge has excellent WebNN support for text embeddings
            "text": "edge",
            "multimodal": "chrome"
        }
        
        # Return optimal browser if available
        return browser_strengths.get(model_type, "chrome")
    
    def _get_alternate_browser(self, original_browser: str, model_type: str) -> str:
        """
        Get alternate browser when original browser fails
        
        Args:
            original_browser: Original browser that failed
            model_type: Type of model
            
        Returns:
            Alternate browser name
        """
        # Get all browsers except the original
        browsers = [b for b in ["chrome", "firefox", "edge"] if b != original_browser]
        
        # If no alternatives, return original
        if not browsers:
            return original_browser
        
        # Try to find browser that's good for this model type
        optimal_browser = self._get_optimal_browser(model_type, "webgpu")
        
        if optimal_browser != original_browser and optimal_browser in browsers:
            return optimal_browser
        
        # Otherwise return any alternative
        return browsers[0]
    
    def _get_all_available_browsers(self, exclude: List[str] = None) -> List[str]:
        """
        Get all available browsers, optionally excluding some
        
        Args:
            exclude: List of browsers to exclude
            
        Returns:
            List of available browser names
        """
        exclude = exclude or []
        
        # Get browser info
        browser_info = self._get_browser_info()
        
        # Filter available browsers
        browsers = [
            browser for browser, info in browser_info.items()
            if info.get("status") == "ready" and browser not in exclude
        ]
        
        return browsers
    
    async def get_status(self) -> Dict[str, Any]:
        """
        Get plugin status
        
        Returns:
            Dictionary with status information
        """
        status = {
            "initialized": self.initialized,
            "fault_tolerance_level": self.fault_tolerance_level,
            "recovery_strategy": self.recovery_strategy,
            "tasks_executed": self.metrics["tasks_executed"],
            "successful_tasks": self.metrics["successful_tasks"],
            "failed_tasks": self.metrics["failed_tasks"],
            "recovery_attempts": self.metrics["recovery_attempts"],
            "successful_recoveries": self.metrics["successful_recoveries"],
            "browser_usage": self.metrics["browser_usage"]
        }
        
        # Calculate success rate
        if self.metrics["tasks_executed"] > 0:
            status["success_rate"] = self.metrics["successful_tasks"] / self.metrics["tasks_executed"]
        else:
            status["success_rate"] = 0
        
        # Calculate recovery rate
        if self.metrics["recovery_attempts"] > 0:
            status["recovery_rate"] = self.metrics["successful_recoveries"] / self.metrics["recovery_attempts"]
        else:
            status["recovery_rate"] = 0
        
        # Add browser information
        status["browser_info"] = self._get_browser_info()
        
        # Add active models
        status["active_models"] = len(self.active_models)
        
        return status
    
    async def get_metrics(self) -> Dict[str, Any]:
        """
        Get detailed plugin metrics
        
        Returns:
            Dictionary with metrics information
        """
        metrics = dict(self.metrics)
        
        # Calculate average execution time
        if metrics["execution_times_ms"]:
            metrics["avg_execution_time_ms"] = sum(metrics["execution_times_ms"]) / len(metrics["execution_times_ms"])
        else:
            metrics["avg_execution_time_ms"] = 0
        
        # Calculate success rate
        if metrics["tasks_executed"] > 0:
            metrics["success_rate"] = metrics["successful_tasks"] / metrics["tasks_executed"]
        else:
            metrics["success_rate"] = 0
        
        # Calculate recovery rate
        if metrics["recovery_attempts"] > 0:
            metrics["recovery_rate"] = metrics["successful_recoveries"] / metrics["recovery_attempts"]
        else:
            metrics["recovery_rate"] = 0
        
        # Get browser usage distribution
        total_browser_usage = sum(metrics["browser_usage"].values())
        if total_browser_usage > 0:
            metrics["browser_distribution"] = {
                browser: count / total_browser_usage
                for browser, count in metrics["browser_usage"].items()
            }
        else:
            metrics["browser_distribution"] = {}
        
        # Add integration metrics if available
        if hasattr(self.integration, "get_metrics"):
            try:
                integration_metrics = self.integration.get_metrics()
                metrics["integration_metrics"] = integration_metrics
            except Exception as e:
                logger.warning(f"Error getting integration metrics: {e}")
        
        return metrics
    
    async def inject_fault(self, fault_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Inject fault for testing fault tolerance
        
        Args:
            fault_config: Fault configuration
            
        Returns:
            Fault injection results
        """
        logger.info(f"Injecting fault: {fault_config}")
        
        try:
            fault_type = fault_config.get("type", "browser_crash")
            browser = fault_config.get("browser", "chrome")
            
            # Log fault injection in transaction log
            await self.transaction_log.append({
                "action": "inject_fault",
                "fault_type": fault_type,
                "browser": browser,
                "timestamp": time.time()
            })
            
            if fault_type == "browser_crash":
                # Inject browser crash
                if hasattr(self.integration, "_simulate_browser_crash"):
                    # Get browser index
                    browsers = list(self._get_browser_info().keys())
                    if browser in browsers:
                        browser_index = browsers.index(browser)
                        
                        # Simulate crash
                        await self.integration._simulate_browser_crash(browser_index)
                        
                        return {
                            "success": True,
                            "fault_type": fault_type,
                            "browser": browser,
                            "browser_index": browser_index
                        }
                    else:
                        return {
                            "success": False,
                            "error": f"Browser {browser} not found"
                        }
                else:
                    return {
                        "success": False,
                        "error": "Browser crash simulation not supported"
                    }
                    
            elif fault_type == "connection_lost":
                # Inject connection loss
                if hasattr(self.integration, "_simulate_connection_loss"):
                    # Get browser index
                    browsers = list(self._get_browser_info().keys())
                    if browser in browsers:
                        browser_index = browsers.index(browser)
                        
                        # Simulate connection loss
                        await self.integration._simulate_connection_loss(browser_index)
                        
                        return {
                            "success": True,
                            "fault_type": fault_type,
                            "browser": browser,
                            "browser_index": browser_index
                        }
                    else:
                        return {
                            "success": False,
                            "error": f"Browser {browser} not found"
                        }
                else:
                    return {
                        "success": False,
                        "error": "Connection loss simulation not supported"
                    }
                    
            elif fault_type == "component_timeout":
                # Inject component timeout
                if hasattr(self.integration, "_simulate_operation_timeout"):
                    # Get browser index
                    browsers = list(self._get_browser_info().keys())
                    if browser in browsers:
                        browser_index = browsers.index(browser)
                        
                        # Simulate timeout
                        await self.integration._simulate_operation_timeout(browser_index)
                        
                        return {
                            "success": True,
                            "fault_type": fault_type,
                            "browser": browser,
                            "browser_index": browser_index
                        }
                    else:
                        return {
                            "success": False,
                            "error": f"Browser {browser} not found"
                        }
                else:
                    return {
                        "success": False,
                        "error": "Operation timeout simulation not supported"
                    }
                    
            else:
                return {
                    "success": False,
                    "error": f"Unknown fault type: {fault_type}"
                }
                
        except Exception as e:
            logger.error(f"Error injecting fault: {e}")
            return {
                "success": False,
                "error": f"Error injecting fault: {e}"
            }
    
    async def shutdown(self) -> Dict[str, Any]:
        """
        Shutdown plugin and clean up resources
        
        Returns:
            Shutdown status
        """
        logger.info("Shutting down ResourcePoolPlugin")
        
        try:
            # Record shutdown in transaction log
            await self.transaction_log.append({
                "action": "shutdown",
                "timestamp": time.time()
            })
            
            # Clear active models
            self.active_models.clear()
            
            # Reset metrics
            self.metrics.clear()
            
            # Update state
            await self.state_manager.update_state("plugin_state", {
                "initialized": False,
                "shutdown_time": time.time()
            })
            
            # Set flag
            self.initialized = False
            
            return {
                "success": True,
                "message": "Plugin shutdown complete"
            }
            
        except Exception as e:
            logger.error(f"Error shutting down plugin: {e}")
            return {
                "success": False,
                "error": f"Error shutting down plugin: {e}"
            }