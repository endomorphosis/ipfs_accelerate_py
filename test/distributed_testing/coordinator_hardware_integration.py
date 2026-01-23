#!/usr/bin/env python3
"""
Coordinator Hardware Integration Module

This module integrates the enhanced hardware capability detector with the coordinator
component of the distributed testing framework. It enables:

1. Hardware capability detection during worker registration
2. Storage of hardware capabilities in the database
3. Hardware-aware task assignment based on task requirements
4. Worker selection based on hardware compatibility

Usage:
    # In coordinator.py
    from coordinator_hardware_integration import CoordinatorHardwareIntegration
    hardware_integration = CoordinatorHardwareIntegration(coordinator, db_path="./test_db.duckdb")
    await hardware_integration.initialize()
"""

import os
import sys
import json
import logging
import anyio
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from datetime import datetime

# Import hardware capability detector
from hardware_capability_detector import (
    HardwareCapabilityDetector,
    HardwareType,
    HardwareVendor,
    PrecisionType,
    CapabilityScore,
    HardwareCapability,
    WorkerHardwareCapabilities
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("coordinator_hardware_integration")


class CoordinatorHardwareIntegration:
    """
    Integrates hardware capability detection with the coordinator.
    """
    
    def __init__(
        self,
        coordinator,
        db_path: Optional[str] = None,
        enable_browser_detection: bool = False,
        browser_executable_path: Optional[str] = None,
        cache_capabilities: bool = True,
        capability_cache_timeout: int = 3600  # 1 hour in seconds
    ):
        """
        Initialize the hardware integration.
        
        Args:
            coordinator: Reference to the DistributedTestingCoordinator instance
            db_path: Path to DuckDB database for storing results
            enable_browser_detection: Whether to enable browser-based detection
            browser_executable_path: Path to browser executable for automated detection
            cache_capabilities: Whether to cache capabilities in memory
            capability_cache_timeout: How long to cache capabilities (in seconds)
        """
        self.coordinator = coordinator
        self.db_path = db_path or coordinator.db_path
        self.enable_browser_detection = enable_browser_detection
        self.browser_executable_path = browser_executable_path
        self.cache_capabilities = cache_capabilities
        self.capability_cache_timeout = capability_cache_timeout
        
        # Initialize hardware capability detector
        self.detector = HardwareCapabilityDetector(
            db_path=self.db_path,
            enable_browser_detection=self.enable_browser_detection,
            browser_executable_path=self.browser_executable_path
        )
        
        # Initialize capability cache
        self.capability_cache = {}
        self.cache_timestamps = {}
        
        # Register handlers with coordinator
        self._register_handlers()
    
    async def initialize(self):
        """Initialize the hardware integration."""
        logger.info("Initializing coordinator hardware integration")
        
        # Create schema if needed
        await self._ensure_database_schema()
        
        # Load existing capabilities into cache if enabled
        if self.cache_capabilities:
            await self._load_capabilities_cache()
        
        logger.info("Coordinator hardware integration initialized")
    
    async def _ensure_database_schema(self):
        """Ensure the database schema is set up."""
        # The HardwareCapabilityDetector will handle creating tables
        # This is just a wrapper to make it async-compatible
        pass
    
    async def _load_capabilities_cache(self):
        """Load existing capabilities into cache."""
        try:
            # Get all worker_ids from the database
            if not self.detector.db_connection:
                logger.warning("No database connection, cannot load capabilities cache")
                return
            
            # Get worker IDs
            worker_results = self.detector.db_connection.execute("""
                SELECT DISTINCT worker_id FROM worker_hardware
            """).fetchall()
            
            worker_ids = [row[0] for row in worker_results]
            logger.info(f"Loading capabilities for {len(worker_ids)} workers into cache")
            
            # Load capabilities for each worker
            now = datetime.now().timestamp()
            for worker_id in worker_ids:
                capabilities = self.detector.get_worker_capabilities(worker_id)
                if capabilities:
                    self.capability_cache[worker_id] = capabilities
                    self.cache_timestamps[worker_id] = now
            
            logger.info(f"Loaded capabilities for {len(self.capability_cache)} workers into cache")
        
        except Exception as e:
            logger.error(f"Error loading capabilities cache: {str(e)}")
    
    def _register_handlers(self):
        """Register integration handlers with the coordinator."""
        # Store original handler
        self.original_worker_register_handler = self.coordinator.handle_worker_register
        
        # Replace with enhanced handler
        self.coordinator.handle_worker_register = self._enhanced_worker_register_handler
        
        # Enhance the worker capability check function
        self.original_can_worker_handle_task = self.coordinator._can_worker_handle_task
        self.coordinator._can_worker_handle_task = self._enhanced_can_worker_handle_task
        
        # Enhance task assignment logic
        self.original_assign_pending_tasks = self.coordinator._assign_pending_tasks
        self.coordinator._assign_pending_tasks = self._enhanced_assign_pending_tasks
        
        logger.info("Enhanced handlers registered with coordinator")
    
    async def _enhanced_worker_register_handler(self, request):
        """
        Enhanced worker registration handler that processes hardware capabilities.
        This wraps the original handler and adds hardware capability processing.
        """
        # Call original handler first
        response = await self.original_worker_register_handler(request)
        
        # If registration was successful, process hardware capabilities
        if response.status == 200:
            try:
                # Parse response and get worker_id
                response_data = await response.json()
                worker_id = response_data.get("worker_id")
                
                if not worker_id:
                    logger.warning("Worker registration successful but no worker_id in response")
                    return response
                
                # Get request data
                request_data = await request.json()
                
                # Process hardware capabilities if provided
                if "hardware_capabilities" in request_data:
                    hardware_capabilities = request_data["hardware_capabilities"]
                    
                    # Convert to WorkerHardwareCapabilities object
                    worker_capabilities = self._parse_hardware_capabilities(
                        worker_id, 
                        hardware_capabilities
                    )
                    
                    # Store capabilities
                    if worker_capabilities:
                        # Store in database
                        self.detector.store_capabilities(worker_capabilities)
                        
                        # Update cache if enabled
                        if self.cache_capabilities:
                            self.capability_cache[worker_id] = worker_capabilities
                            self.cache_timestamps[worker_id] = datetime.now().timestamp()
                        
                        logger.info(f"Stored hardware capabilities for worker {worker_id}")
                
                # Return original response
                return response
            
            except Exception as e:
                logger.error(f"Error processing hardware capabilities: {str(e)}")
                # Return original response even if there was an error
                return response
        
        # If registration failed, just return the original response
        return response
    
    def _parse_hardware_capabilities(self, worker_id: str, 
                                   capabilities_data: Dict[str, Any]) -> Optional[WorkerHardwareCapabilities]:
        """
        Parse hardware capabilities data from worker registration.
        
        Args:
            worker_id: Worker ID
            capabilities_data: Hardware capabilities data from request
            
        Returns:
            WorkerHardwareCapabilities object or None if parsing failed
        """
        try:
            # Extract basic worker info
            hostname = capabilities_data.get("hostname", "unknown")
            os_type = capabilities_data.get("os_type", "unknown")
            os_version = capabilities_data.get("os_version", "unknown")
            cpu_count = capabilities_data.get("cpu_count", 0)
            total_memory_gb = capabilities_data.get("total_memory_gb", 0.0)
            
            # Create hardware capabilities list
            hardware_capabilities = []
            
            # Process hardware components
            for hw_data in capabilities_data.get("hardware_capabilities", []):
                try:
                    # Convert hardware type and vendor to enums
                    hw_type_str = hw_data.get("hardware_type", "other")
                    try:
                        hardware_type = HardwareType(hw_type_str)
                    except (ValueError, TypeError):
                        hardware_type = HardwareType.OTHER
                    
                    vendor_str = hw_data.get("vendor", "unknown")
                    try:
                        vendor = HardwareVendor(vendor_str)
                    except (ValueError, TypeError):
                        vendor = HardwareVendor.UNKNOWN
                    
                    # Convert precision types
                    supported_precisions = []
                    for precision_str in hw_data.get("supported_precisions", []):
                        try:
                            supported_precisions.append(PrecisionType(precision_str))
                        except (ValueError, TypeError):
                            pass
                    
                    # Convert scores
                    scores = {}
                    for score_type, score_value in hw_data.get("scores", {}).items():
                        try:
                            scores[score_type] = CapabilityScore(score_value)
                        except (ValueError, TypeError):
                            scores[score_type] = CapabilityScore.UNKNOWN
                    
                    # Create hardware capability
                    hw_capability = HardwareCapability(
                        hardware_type=hardware_type,
                        vendor=vendor,
                        model=hw_data.get("model", "Unknown"),
                        version=hw_data.get("version", None),
                        driver_version=hw_data.get("driver_version", None),
                        compute_units=hw_data.get("compute_units", None),
                        cores=hw_data.get("cores", None),
                        memory_gb=hw_data.get("memory_gb", None),
                        supported_precisions=supported_precisions,
                        capabilities=hw_data.get("capabilities", {}),
                        scores=scores
                    )
                    
                    # Add to list
                    hardware_capabilities.append(hw_capability)
                
                except Exception as e:
                    logger.warning(f"Error parsing hardware component: {str(e)}")
                    continue
            
            # Create worker capabilities object
            worker_capabilities = WorkerHardwareCapabilities(
                worker_id=worker_id,
                hostname=hostname,
                os_type=os_type,
                os_version=os_version,
                cpu_count=cpu_count,
                total_memory_gb=total_memory_gb,
                hardware_capabilities=hardware_capabilities,
                last_updated=datetime.now().timestamp()
            )
            
            return worker_capabilities
        
        except Exception as e:
            logger.error(f"Error parsing hardware capabilities: {str(e)}")
            return None
    
    async def _enhanced_can_worker_handle_task(self, worker_id: str, task: Dict[str, Any]) -> bool:
        """
        Enhanced version of _can_worker_handle_task that uses hardware capabilities.
        
        Args:
            worker_id: Worker ID
            task: Task information
            
        Returns:
            True if worker can handle the task, False otherwise
        """
        # Call original method first for basic checks
        if not await self.original_can_worker_handle_task(worker_id, task):
            return False
        
        # Get hardware requirements from task
        requirements = task.get("requirements", {})
        hardware_requirements = requirements.get("hardware", {})
        
        # If no hardware requirements, worker can handle the task
        if not hardware_requirements:
            return True
        
        # Get worker capabilities from cache or database
        worker_capabilities = await self._get_worker_capabilities(worker_id)
        
        # If no capabilities found, can't determine compatibility
        if not worker_capabilities:
            # Log warning but allow task if no hardware requirements are critical
            if not hardware_requirements.get("critical", False):
                logger.warning(f"No hardware capabilities for worker {worker_id}, but continuing as requirements not critical")
                return True
            else:
                logger.warning(f"No hardware capabilities for worker {worker_id}, can't verify critical hardware requirements")
                return False
        
        # Check hardware requirements against worker capabilities
        if "required_types" in hardware_requirements:
            required_types = hardware_requirements["required_types"]
            if not self._has_required_hardware_types(worker_capabilities, required_types):
                return False
        
        # Check for specific hardware features
        if "features" in hardware_requirements:
            required_features = hardware_requirements["features"]
            if not self._has_required_hardware_features(worker_capabilities, required_features):
                return False
        
        # Check minimum memory requirements
        if "min_memory_gb" in hardware_requirements:
            min_memory_gb = hardware_requirements["min_memory_gb"]
            if not self._has_sufficient_memory(worker_capabilities, min_memory_gb):
                return False
        
        # Check precision type requirements
        if "precision_types" in hardware_requirements:
            required_precision_types = hardware_requirements["precision_types"]
            if not self._has_required_precision_types(worker_capabilities, required_precision_types):
                return False
        
        # All checks passed, worker can handle the task
        return True
    
    async def _enhanced_assign_pending_tasks(self):
        """
        Enhanced version of _assign_pending_tasks that uses hardware capabilities
        for more intelligent task assignment.
        """
        # If no pending tasks, nothing to do
        if not self.coordinator.pending_tasks:
            return 0
        
        # Get available workers
        available_workers = []
        for worker_id, worker in self.coordinator.workers.items():
            if (worker.get("status") == "active" and 
                worker_id in self.coordinator.worker_connections):
                available_workers.append(worker_id)
        
        # If no available workers, nothing to do
        if not available_workers:
            return 0
        
        # Get pending tasks
        pending_tasks = []
        for task_id in self.coordinator.pending_tasks:
            if task_id in self.coordinator.tasks:
                task = self.coordinator.tasks[task_id]
                pending_tasks.append(task)
        
        # Group tasks by hardware requirements
        tasks_by_hardware = {}
        for task in pending_tasks:
            hardware_key = self._get_hardware_requirements_key(task)
            if hardware_key not in tasks_by_hardware:
                tasks_by_hardware[hardware_key] = []
            tasks_by_hardware[hardware_key].append(task)
        
        # Find compatible workers for each hardware requirement
        compatible_workers = {}
        for hardware_key, tasks in tasks_by_hardware.items():
            if not tasks:
                continue
                
            # Use the first task as representative for this group
            sample_task = tasks[0]
            
            # Find workers that can handle this task
            workers_for_task = []
            for worker_id in available_workers:
                if await self._enhanced_can_worker_handle_task(worker_id, sample_task):
                    workers_for_task.append(worker_id)
            
            compatible_workers[hardware_key] = workers_for_task
        
        # Call original method to do the actual assignment
        # Our enhanced method just improves the selection of workers
        return await self.original_assign_pending_tasks()
    
    def _get_hardware_requirements_key(self, task: Dict[str, Any]) -> str:
        """
        Generate a key for task hardware requirements for grouping purposes.
        
        Args:
            task: Task information
            
        Returns:
            String key representing hardware requirements
        """
        requirements = task.get("requirements", {})
        hardware_requirements = requirements.get("hardware", {})
        
        # If no hardware requirements, use a default key
        if not hardware_requirements:
            return "default"
        
        # Create a string representation of the requirements
        try:
            # Sort keys for consistency
            sorted_requirements = json.dumps(hardware_requirements, sort_keys=True)
            return sorted_requirements
        except:
            # If serialization fails, use a default key
            return f"custom_{hash(str(hardware_requirements))}"
    
    def _has_required_hardware_types(self, worker_capabilities: WorkerHardwareCapabilities, 
                                  required_types: List[str]) -> bool:
        """
        Check if worker has all required hardware types.
        
        Args:
            worker_capabilities: Worker hardware capabilities
            required_types: List of required hardware types
            
        Returns:
            True if worker has all required types, False otherwise
        """
        # Get all hardware types from worker
        worker_hardware_types = set()
        for hw in worker_capabilities.hardware_capabilities:
            hw_type = hw.hardware_type.value if isinstance(hw.hardware_type, Enum) else str(hw.hardware_type)
            worker_hardware_types.add(hw_type)
        
        # Check if all required types are present
        for required_type in required_types:
            if required_type not in worker_hardware_types:
                return False
        
        return True
    
    def _has_required_hardware_features(self, worker_capabilities: WorkerHardwareCapabilities,
                                     required_features: Dict[str, Any]) -> bool:
        """
        Check if worker has all required hardware features.
        
        Args:
            worker_capabilities: Worker hardware capabilities
            required_features: Dictionary of required features
            
        Returns:
            True if worker has all required features, False otherwise
        """
        # For each hardware type specified in required_features
        for hw_type, features in required_features.items():
            # Find matching hardware in worker capabilities
            matching_hardware = None
            for hw in worker_capabilities.hardware_capabilities:
                hw_type_val = hw.hardware_type.value if isinstance(hw.hardware_type, Enum) else str(hw.hardware_type)
                if hw_type_val == hw_type:
                    matching_hardware = hw
                    break
            
            # If required hardware type not found, check if it's critical
            if not matching_hardware:
                if features.get("critical", False):
                    return False
                else:
                    continue
            
            # Check each feature
            for feature_name, feature_value in features.items():
                # Skip "critical" flag
                if feature_name == "critical":
                    continue
                
                # Check if feature exists in hardware capabilities
                if (feature_name not in matching_hardware.capabilities or
                    matching_hardware.capabilities[feature_name] != feature_value):
                    # If feature doesn't match and is critical, return False
                    if features.get("critical", False):
                        return False
        
        # All checks passed
        return True
    
    def _has_sufficient_memory(self, worker_capabilities: WorkerHardwareCapabilities,
                            min_memory_gb: float) -> bool:
        """
        Check if worker has sufficient memory.
        
        Args:
            worker_capabilities: Worker hardware capabilities
            min_memory_gb: Minimum required memory in GB
            
        Returns:
            True if worker has sufficient memory, False otherwise
        """
        # Check if any hardware component has sufficient memory
        for hw in worker_capabilities.hardware_capabilities:
            if hw.memory_gb and hw.memory_gb >= min_memory_gb:
                return True
        
        # Also check total worker memory
        if worker_capabilities.total_memory_gb >= min_memory_gb:
            return True
        
        return False
    
    def _has_required_precision_types(self, worker_capabilities: WorkerHardwareCapabilities,
                                   required_precision_types: List[str]) -> bool:
        """
        Check if worker has all required precision types.
        
        Args:
            worker_capabilities: Worker hardware capabilities
            required_precision_types: List of required precision types
            
        Returns:
            True if worker has all required precision types, False otherwise
        """
        # Check each hardware component
        for hw in worker_capabilities.hardware_capabilities:
            # Convert precision types to strings
            supported_precisions = set()
            for p in hw.supported_precisions:
                p_str = p.value if isinstance(p, Enum) else str(p)
                supported_precisions.add(p_str)
            
            # Check if all required types are supported by this component
            if all(req_type in supported_precisions for req_type in required_precision_types):
                return True
        
        return False
    
    async def _get_worker_capabilities(self, worker_id: str) -> Optional[WorkerHardwareCapabilities]:
        """
        Get worker capabilities from cache or database.
        
        Args:
            worker_id: Worker ID
            
        Returns:
            WorkerHardwareCapabilities or None if not found
        """
        # Check cache first if enabled
        if self.cache_capabilities and worker_id in self.capability_cache:
            # Check if cache is still valid
            cache_time = self.cache_timestamps.get(worker_id, 0)
            now = datetime.now().timestamp()
            
            if now - cache_time <= self.capability_cache_timeout:
                return self.capability_cache[worker_id]
        
        # Otherwise, get from database
        capabilities = self.detector.get_worker_capabilities(worker_id)
        
        # Update cache if enabled
        if self.cache_capabilities and capabilities:
            self.capability_cache[worker_id] = capabilities
            self.cache_timestamps[worker_id] = datetime.now().timestamp()
        
        return capabilities


async def main():
    """Main function for testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test coordinator hardware integration")
    parser.add_argument("--db-path", help="Path to DuckDB database")
    parser.add_argument("--enable-browser-detection", action="store_true", help="Enable browser detection")
    
    args = parser.parse_args()
    
    # Create a dummy coordinator for testing
    class DummyCoordinator:
        def __init__(self):
            self.db_path = args.db_path
            self.workers = {}
            self.worker_connections = {}
            self.tasks = {}
            self.pending_tasks = set()
            self.running_tasks = {}
        
        async def handle_worker_register(self, request):
            # Dummy handler
            from aiohttp import web
            return web.json_response({"status": "success", "worker_id": "test_worker"})
        
        async def _can_worker_handle_task(self, worker_id, task):
            # Dummy handler
            return True
        
        async def _assign_pending_tasks(self):
            # Dummy handler
            return 0
    
    # Create coordinator and integration
    coordinator = DummyCoordinator()
    integration = CoordinatorHardwareIntegration(
        coordinator,
        db_path=args.db_path,
        enable_browser_detection=args.enable_browser_detection
    )
    
    # Initialize integration
    await integration.initialize()
    
    # Test parsing capabilities
    test_capabilities = {
        "hostname": "test-worker",
        "os_type": "Linux",
        "os_version": "5.10.0",
        "cpu_count": 8,
        "total_memory_gb": 16.0,
        "hardware_capabilities": [
            {
                "hardware_type": "cpu",
                "vendor": "intel",
                "model": "Intel Core i7",
                "cores": 8,
                "memory_gb": 16.0,
                "supported_precisions": ["fp32", "fp16", "int8"],
                "scores": {
                    "compute": 4,
                    "memory": 3
                }
            },
            {
                "hardware_type": "gpu",
                "vendor": "nvidia",
                "model": "NVIDIA RTX 3080",
                "compute_units": 68,
                "memory_gb": 10.0,
                "supported_precisions": ["fp32", "fp16", "int8"],
                "scores": {
                    "compute": 5,
                    "memory": 4
                }
            }
        ]
    }
    
    worker_capabilities = integration._parse_hardware_capabilities("test_worker", test_capabilities)
    
    if worker_capabilities:
        print("Successfully parsed worker capabilities:")
        print(f"Worker ID: {worker_capabilities.worker_id}")
        print(f"Hostname: {worker_capabilities.hostname}")
        print(f"OS: {worker_capabilities.os_type} {worker_capabilities.os_version}")
        print(f"CPU Count: {worker_capabilities.cpu_count}")
        print(f"Total Memory: {worker_capabilities.total_memory_gb:.2f} GB")
        print(f"Hardware Components: {len(worker_capabilities.hardware_capabilities)}")
        
        # Store capabilities
        success = integration.detector.store_capabilities(worker_capabilities)
        print(f"Stored capabilities: {success}")
        
        # Test retrieval
        retrieved = await integration._get_worker_capabilities("test_worker")
        print(f"Retrieved capabilities: {retrieved is not None}")
        
        # Test hardware requirements checks
        if retrieved:
            has_cpu = integration._has_required_hardware_types(retrieved, ["cpu"])
            print(f"Has CPU: {has_cpu}")
            
            has_gpu = integration._has_required_hardware_types(retrieved, ["gpu"])
            print(f"Has GPU: {has_gpu}")
            
            has_sufficient_memory = integration._has_sufficient_memory(retrieved, 8.0)
            print(f"Has sufficient memory (8GB): {has_sufficient_memory}")
            
            has_fp16 = integration._has_required_precision_types(retrieved, ["fp16"])
            print(f"Has FP16 support: {has_fp16}")
    
    print("\nTest completed")


if __name__ == "__main__":
    anyio.run(main())