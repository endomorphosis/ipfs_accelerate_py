#!/usr/bin/env python3
"""
WebGPU/WebNN Resource Pool Recovery Manager

This module provides recovery management for browser-based models and resources,
implementing fault tolerance and state management for the resource pool.

Usage:
    Import this module in resource_pool_bridge.py to enable fault tolerance
    for browser-based model inference.
"""

import asyncio
import hashlib
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple, Union
import uuid

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BrowserStateManager:
    """
    State manager for browser-based models and resources.
    
    This class manages the state of browser instances, models, and operations 
    with transaction-based state updates for consistency across failures.
    """
    
    def __init__(
        self,
        sync_interval: int = 5,
        redundancy_factor: int = 2
    ):
        """
        Initialize the browser state manager.
        
        Args:
            sync_interval: Interval for state synchronization in seconds
            redundancy_factor: Number of redundant copies to maintain for critical state
        """
        self.sync_interval = sync_interval
        self.redundancy_factor = redundancy_factor
        
        # State partitions
        self.state = {
            "browsers": {},
            "models": {},
            "operations": {},
            "transactions": {},
            "performance": {}
        }
        
        # Transaction log
        self.transaction_log = []
        
        # Checksum tracking
        self.checksums = {}
        
        # Sync task
        self.sync_task = None
        
        logger.info(f"BrowserStateManager initialized with redundancy_factor={redundancy_factor}")
    
    async def initialize(self):
        """Initialize the browser state manager."""
        logger.info("Initializing BrowserStateManager...")
        
        # Initialize state
        await self._initialize_state()
        
        # Start state synchronization task
        self.sync_task = asyncio.create_task(self._state_sync_loop())
        
        logger.info("BrowserStateManager initialization complete")
    
    async def _initialize_state(self):
        """Initialize state with default values."""
        # Initialize state with default values
        self.state = {
            "browsers": {},
            "models": {},
            "operations": {},
            "transactions": {},
            "performance": {}
        }
        
        # Calculate initial checksums
        await self._update_checksums()
        
        logger.info("State initialized with default values")
    
    async def _state_sync_loop(self):
        """State synchronization loop."""
        while True:
            try:
                # Simulate state synchronization with redundant storage
                await self._sync_state()
                
                # Update checksums
                await self._update_checksums()
                
                # Verify state consistency
                await self._verify_state_consistency()
                
            except Exception as e:
                logger.error(f"Error in state sync loop: {str(e)}")
            
            # Wait for next sync
            await asyncio.sleep(self.sync_interval)
    
    async def _sync_state(self):
        """Synchronize state with redundant storage."""
        # Simulate synchronizing state to redundant storage
        # In a real implementation, would write to durable storage, replicate to other nodes, etc.
        
        # For now, just log that sync occurred
        logger.debug(f"State synchronized at {datetime.now().isoformat()}")
    
    async def _update_checksums(self):
        """Update checksums for state partitions."""
        for partition, data in self.state.items():
            # Calculate checksum
            checksum = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
            self.checksums[partition] = checksum
    
    async def _verify_state_consistency(self):
        """Verify consistency of state using checksums."""
        # In a real implementation, would verify checksums across redundant copies
        # For now, just calculate and verify checksums
        
        for partition, data in self.state.items():
            # Calculate current checksum
            current_checksum = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
            
            # Check against stored checksum
            if current_checksum != self.checksums.get(partition, ""):
                logger.error(f"State consistency check failed for partition {partition}")
                
                # Attempt recovery
                await self._recover_state_partition(partition)
    
    async def _recover_state_partition(self, partition: str):
        """
        Recover a state partition from redundant storage.
        
        Args:
            partition: Name of the partition to recover
        """
        # In a real implementation, would recover from redundant storage
        # For now, just log the recovery attempt
        
        logger.warning(f"Attempting to recover state partition: {partition}")
        
        # Mark partition for full sync
        # await self._full_sync_partition(partition)
        
        # Update checksum after recovery
        await self._update_checksums()
    
    async def register_browser(
        self, 
        browser_id: str, 
        browser_type: str, 
        capabilities: Dict[str, Any]
    ):
        """
        Register a browser in the state manager.
        
        Args:
            browser_id: Unique identifier for the browser
            browser_type: Type of browser (chrome, firefox, edge)
            capabilities: Browser capabilities
        """
        transaction_id = await self.start_transaction("register_browser", {
            "browser_id": browser_id,
            "browser_type": browser_type
        })
        
        try:
            # Add browser to state
            self.state["browsers"][browser_id] = {
                "id": browser_id,
                "type": browser_type,
                "capabilities": capabilities,
                "registered_at": datetime.now().isoformat(),
                "last_active": datetime.now().isoformat(),
                "status": "ready",
                "models": [],
                "operations": []
            }
            
            # Update checksums
            await self._update_checksums()
            
            # Commit transaction
            await self.commit_transaction(transaction_id)
            
            logger.info(f"Browser {browser_id} registered")
            return True
            
        except Exception as e:
            logger.error(f"Error registering browser {browser_id}: {str(e)}")
            
            # Rollback transaction
            await self.rollback_transaction(transaction_id)
            
            return False
    
    async def register_model(
        self,
        model_id: str,
        model_name: str,
        model_type: str,
        browser_id: str
    ):
        """
        Register a model in the state manager.
        
        Args:
            model_id: Unique identifier for the model
            model_name: Name of the model
            model_type: Type of model
            browser_id: ID of the browser running the model
        """
        transaction_id = await self.start_transaction("register_model", {
            "model_id": model_id,
            "model_name": model_name,
            "browser_id": browser_id
        })
        
        try:
            # Add model to state
            self.state["models"][model_id] = {
                "id": model_id,
                "name": model_name,
                "type": model_type,
                "browser_id": browser_id,
                "created_at": datetime.now().isoformat(),
                "last_active": datetime.now().isoformat(),
                "status": "ready",
                "operations": []
            }
            
            # Add model to browser's model list
            if browser_id in self.state["browsers"]:
                if "models" not in self.state["browsers"][browser_id]:
                    self.state["browsers"][browser_id]["models"] = []
                
                self.state["browsers"][browser_id]["models"].append(model_id)
            
            # Update checksums
            await self._update_checksums()
            
            # Commit transaction
            await self.commit_transaction(transaction_id)
            
            logger.info(f"Model {model_id} registered on browser {browser_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error registering model {model_id}: {str(e)}")
            
            # Rollback transaction
            await self.rollback_transaction(transaction_id)
            
            return False
    
    async def update_model_browser(self, model_id: str, browser_id: str):
        """
        Update the browser assignment for a model.
        
        Args:
            model_id: Model ID
            browser_id: New browser ID
        """
        if model_id not in self.state["models"]:
            logger.warning(f"Model {model_id} not found, cannot update browser")
            return False
        
        transaction_id = await self.start_transaction("update_model_browser", {
            "model_id": model_id,
            "browser_id": browser_id
        })
        
        try:
            # Get current browser ID
            old_browser_id = self.state["models"][model_id]["browser_id"]
            
            # Remove model from old browser's model list
            if old_browser_id in self.state["browsers"]:
                if "models" in self.state["browsers"][old_browser_id]:
                    if model_id in self.state["browsers"][old_browser_id]["models"]:
                        self.state["browsers"][old_browser_id]["models"].remove(model_id)
            
            # Update model's browser ID
            self.state["models"][model_id]["browser_id"] = browser_id
            
            # Add model to new browser's model list
            if browser_id in self.state["browsers"]:
                if "models" not in self.state["browsers"][browser_id]:
                    self.state["browsers"][browser_id]["models"] = []
                
                self.state["browsers"][browser_id]["models"].append(model_id)
            
            # Update last active timestamp
            self.state["models"][model_id]["last_active"] = datetime.now().isoformat()
            
            # Update checksums
            await self._update_checksums()
            
            # Commit transaction
            await self.commit_transaction(transaction_id)
            
            logger.info(f"Model {model_id} moved from browser {old_browser_id} to {browser_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating browser for model {model_id}: {str(e)}")
            
            # Rollback transaction
            await self.rollback_transaction(transaction_id)
            
            return False
    
    async def record_operation(
        self,
        operation_id: str,
        model_id: str,
        operation_type: str,
        start_time: str,
        status: str = "started",
        metadata: Dict[str, Any] = None
    ):
        """
        Record an operation in the state manager.
        
        Args:
            operation_id: Unique identifier for the operation
            model_id: ID of the model performing the operation
            operation_type: Type of operation
            start_time: Start time of the operation
            status: Status of the operation
            metadata: Additional metadata for the operation
        """
        # Add operation to state
        self.state["operations"][operation_id] = {
            "id": operation_id,
            "model_id": model_id,
            "type": operation_type,
            "start_time": start_time,
            "status": status,
            "metadata": metadata or {}
        }
        
        # Add operation to model's operation list
        if model_id in self.state["models"]:
            if "operations" not in self.state["models"][model_id]:
                self.state["models"][model_id]["operations"] = []
            
            self.state["models"][model_id]["operations"].append(operation_id)
        
        # Update last active timestamp for model
        if model_id in self.state["models"]:
            self.state["models"][model_id]["last_active"] = datetime.now().isoformat()
        
        # Update checksums
        await self._update_checksums()
        
        logger.debug(f"Operation {operation_id} recorded for model {model_id}")
        return True
    
    async def complete_operation(
        self,
        operation_id: str,
        status: str = "completed",
        end_time: str = None,
        result: Any = None
    ):
        """
        Mark an operation as completed.
        
        Args:
            operation_id: Operation ID
            status: Final status of the operation
            end_time: End time of the operation
            result: Result of the operation
        """
        if operation_id not in self.state["operations"]:
            logger.warning(f"Operation {operation_id} not found, cannot complete")
            return False
        
        # Update operation status
        self.state["operations"][operation_id]["status"] = status
        self.state["operations"][operation_id]["end_time"] = end_time or datetime.now().isoformat()
        
        if result is not None:
            self.state["operations"][operation_id]["result"] = result
        
        # Update checksums
        await self._update_checksums()
        
        logger.debug(f"Operation {operation_id} marked as {status}")
        return True
    
    async def start_transaction(self, transaction_type: str, metadata: Dict[str, Any] = None) -> str:
        """
        Start a new transaction.
        
        Args:
            transaction_type: Type of transaction
            metadata: Additional metadata for the transaction
            
        Returns:
            Transaction ID
        """
        transaction_id = f"tx-{uuid.uuid4().hex}"
        
        # Record transaction
        self.state["transactions"][transaction_id] = {
            "id": transaction_id,
            "type": transaction_type,
            "start_time": datetime.now().isoformat(),
            "status": "started",
            "metadata": metadata or {},
            "changes": []
        }
        
        # Add to transaction log
        self.transaction_log.append({
            "id": transaction_id,
            "type": transaction_type,
            "time": datetime.now().isoformat(),
            "action": "start"
        })
        
        logger.debug(f"Transaction {transaction_id} started: {transaction_type}")
        return transaction_id
    
    async def commit_transaction(self, transaction_id: str):
        """
        Commit a transaction.
        
        Args:
            transaction_id: Transaction ID
        """
        if transaction_id not in self.state["transactions"]:
            logger.warning(f"Transaction {transaction_id} not found, cannot commit")
            return False
        
        # Update transaction status
        self.state["transactions"][transaction_id]["status"] = "committed"
        self.state["transactions"][transaction_id]["end_time"] = datetime.now().isoformat()
        
        # Add to transaction log
        self.transaction_log.append({
            "id": transaction_id,
            "type": self.state["transactions"][transaction_id]["type"],
            "time": datetime.now().isoformat(),
            "action": "commit"
        })
        
        # Trigger state sync
        await self._sync_state()
        
        logger.debug(f"Transaction {transaction_id} committed")
        return True
    
    async def rollback_transaction(self, transaction_id: str):
        """
        Rollback a transaction.
        
        Args:
            transaction_id: Transaction ID
        """
        if transaction_id not in self.state["transactions"]:
            logger.warning(f"Transaction {transaction_id} not found, cannot rollback")
            return False
        
        # Update transaction status
        self.state["transactions"][transaction_id]["status"] = "rolled_back"
        self.state["transactions"][transaction_id]["end_time"] = datetime.now().isoformat()
        
        # Add to transaction log
        self.transaction_log.append({
            "id": transaction_id,
            "type": self.state["transactions"][transaction_id]["type"],
            "time": datetime.now().isoformat(),
            "action": "rollback"
        })
        
        # Apply rollback logic
        # In a real implementation, would undo changes recorded in the transaction
        
        # Trigger state sync
        await self._sync_state()
        
        logger.debug(f"Transaction {transaction_id} rolled back")
        return True
    
    def get_browser_state(self, browser_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the current state of a browser.
        
        Args:
            browser_id: Browser ID
            
        Returns:
            Browser state or None if not found
        """
        return self.state["browsers"].get(browser_id)
    
    def get_model_state(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the current state of a model.
        
        Args:
            model_id: Model ID
            
        Returns:
            Model state or None if not found
        """
        return self.state["models"].get(model_id)
    
    def get_operation_state(self, operation_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the current state of an operation.
        
        Args:
            operation_id: Operation ID
            
        Returns:
            Operation state or None if not found
        """
        return self.state["operations"].get(operation_id)
    
    def get_transaction_state(self, transaction_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the current state of a transaction.
        
        Args:
            transaction_id: Transaction ID
            
        Returns:
            Transaction state or None if not found
        """
        return self.state["transactions"].get(transaction_id)


class ResourcePoolRecoveryManager:
    """
    Recovery manager for browser-based models and resources.
    
    This class provides recovery capabilities for browser failures, model failures,
    and operation failures using various recovery strategies.
    """
    
    def __init__(
        self,
        strategy: str = "progressive",
        state_manager: BrowserStateManager = None
    ):
        """
        Initialize the recovery manager.
        
        Args:
            strategy: Default recovery strategy to use
            state_manager: Reference to the state manager
        """
        self.strategy = strategy
        self.state_manager = state_manager
        
        # Recovery tracking
        self.recovery_attempts = {}
        self.recovery_history = []
        
        # Model recovery settings
        self.model_recovery_settings = {}
        
        # Active operations
        self.active_operations = {}
        
        logger.info(f"ResourcePoolRecoveryManager initialized with strategy={strategy}")
    
    async def initialize(self):
        """Initialize the recovery manager."""
        logger.info("Initializing ResourcePoolRecoveryManager...")
        
        # Initialize recovery tracking
        self.recovery_attempts = {}
        self.recovery_history = []
        
        logger.info("ResourcePoolRecoveryManager initialization complete")
    
    async def start_operation(
        self,
        model_id: str,
        operation_type: str,
        metadata: Dict[str, Any] = None
    ) -> str:
        """
        Start tracking an operation for recovery purposes.
        
        Args:
            model_id: ID of the model performing the operation
            operation_type: Type of operation
            metadata: Additional metadata for the operation
            
        Returns:
            Operation ID
        """
        operation_id = f"op-{uuid.uuid4().hex}"
        
        # Record operation
        self.active_operations[operation_id] = {
            "id": operation_id,
            "model_id": model_id,
            "type": operation_type,
            "start_time": datetime.now().isoformat(),
            "status": "started",
            "metadata": metadata or {}
        }
        
        # Record in state manager if available
        if self.state_manager:
            await self.state_manager.record_operation(
                operation_id=operation_id,
                model_id=model_id,
                operation_type=operation_type,
                start_time=datetime.now().isoformat(),
                status="started",
                metadata=metadata
            )
        
        logger.debug(f"Started tracking operation {operation_id} for model {model_id}")
        return operation_id
    
    async def complete_operation(self, operation_id: str):
        """
        Mark an operation as completed.
        
        Args:
            operation_id: Operation ID
        """
        if operation_id not in self.active_operations:
            logger.warning(f"Operation {operation_id} not found, cannot complete")
            return False
        
        # Update operation status
        self.active_operations[operation_id]["status"] = "completed"
        self.active_operations[operation_id]["end_time"] = datetime.now().isoformat()
        
        # Record in state manager if available
        if self.state_manager:
            await self.state_manager.complete_operation(
                operation_id=operation_id,
                status="completed",
                end_time=datetime.now().isoformat()
            )
        
        # Remove from active operations
        del self.active_operations[operation_id]
        
        logger.debug(f"Operation {operation_id} completed")
        return True
    
    async def set_model_recovery_settings(
        self,
        model_id: str,
        recovery_timeout: int,
        state_persistence: bool,
        failover_strategy: str
    ):
        """
        Set recovery settings for a specific model.
        
        Args:
            model_id: Model ID
            recovery_timeout: Maximum recovery time in seconds
            state_persistence: Whether to persist state between sessions
            failover_strategy: Strategy for failover (immediate, progressive, etc.)
        """
        self.model_recovery_settings[model_id] = {
            "recovery_timeout": recovery_timeout,
            "state_persistence": state_persistence,
            "failover_strategy": failover_strategy
        }
        
        logger.info(f"Recovery settings updated for model {model_id}")
    
    async def recover_model_operation(
        self,
        model_id: str,
        operation_type: str,
        error: str,
        inputs: Any = None
    ) -> Tuple[bool, Any]:
        """
        Recover from a failed model operation.
        
        Args:
            model_id: ID of the model that failed
            operation_type: Type of operation that failed
            error: Error message
            inputs: Inputs to the operation (for retry)
            
        Returns:
            Tuple of (success, recovered_model)
        """
        # Increment recovery attempts
        model_attempts = self.recovery_attempts.get(model_id, 0) + 1
        self.recovery_attempts[model_id] = model_attempts
        
        # Log recovery attempt
        logger.info(f"Attempting to recover model {model_id} (attempt {model_attempts})")
        
        # Get model recovery settings
        settings = self.model_recovery_settings.get(model_id, {
            "recovery_timeout": 30,
            "state_persistence": True,
            "failover_strategy": "immediate"
        })
        
        # Get model state
        model_state = None
        if self.state_manager:
            model_state = self.state_manager.get_model_state(model_id)
        
        if not model_state:
            logger.warning(f"Model {model_id} state not found, cannot recover")
            return False, None
        
        # Get browser state
        browser_id = model_state.get("browser_id")
        browser_state = None
        
        if self.state_manager and browser_id:
            browser_state = self.state_manager.get_browser_state(browser_id)
        
        # Choose recovery strategy based on settings and error
        strategy = settings.get("failover_strategy", "immediate")
        
        if strategy == "progressive" and model_attempts < 3:
            # Try simpler recovery first
            success, recovered_model = await self._recover_model_retry(
                model_id, browser_id, model_state, inputs
            )
        else:
            # Go straight to browser failover
            success, recovered_model = await self._recover_model_failover(
                model_id, browser_id, model_state, inputs
            )
        
        # Record recovery result
        self.recovery_history.append({
            "model_id": model_id,
            "operation_type": operation_type,
            "error": error,
            "attempt": model_attempts,
            "strategy": strategy,
            "success": success,
            "time": datetime.now().isoformat()
        })
        
        # Reset recovery attempts if successful
        if success:
            self.recovery_attempts[model_id] = 0
        
        return success, recovered_model
    
    async def _recover_model_retry(
        self,
        model_id: str,
        browser_id: str,
        model_state: Dict[str, Any],
        inputs: Any
    ) -> Tuple[bool, Any]:
        """
        Attempt to recover a model by retrying the operation on the same browser.
        
        Args:
            model_id: Model ID
            browser_id: Browser ID
            model_state: Model state
            inputs: Inputs to the operation
            
        Returns:
            Tuple of (success, recovered_model)
        """
        logger.info(f"Attempting to recover model {model_id} by retrying on browser {browser_id}")
        
        # Simulate recovery attempt
        try:
            # In a real implementation, would try to restart the model on the same browser
            # For now, just simulate a recovery attempt
            
            # Create a simple model proxy
            from resource_pool_bridge import ModelProxy
            
            # Create a recovered model proxy
            recovered_model = ModelProxy(
                model_id=model_id,
                model_name=model_state.get("name", "unknown"),
                model_type=model_state.get("type", "unknown"),
                browser_id=browser_id
            )
            
            logger.info(f"Model {model_id} recovered on browser {browser_id}")
            return True, recovered_model
            
        except Exception as e:
            logger.error(f"Error recovering model {model_id} by retry: {str(e)}")
            return False, None
    
    async def _recover_model_failover(
        self,
        model_id: str,
        browser_id: str,
        model_state: Dict[str, Any],
        inputs: Any
    ) -> Tuple[bool, Any]:
        """
        Attempt to recover a model by failing over to a different browser.
        
        Args:
            model_id: Model ID
            browser_id: Original browser ID
            model_state: Model state
            inputs: Inputs to the operation
            
        Returns:
            Tuple of (success, recovered_model)
        """
        logger.info(f"Attempting to recover model {model_id} by failing over from browser {browser_id}")
        
        # Simulate failover to another browser
        try:
            # Generate a new browser ID
            new_browser_id = f"recovery-{uuid.uuid4().hex[:8]}"
            
            # Update model state if state manager is available
            if self.state_manager:
                await self.state_manager.update_model_browser(model_id, new_browser_id)
            
            # Create a simple model proxy
            from resource_pool_bridge import ModelProxy
            
            # Create a recovered model proxy
            recovered_model = ModelProxy(
                model_id=model_id,
                model_name=model_state.get("name", "unknown"),
                model_type=model_state.get("type", "unknown"),
                browser_id=new_browser_id
            )
            
            logger.info(f"Model {model_id} recovered by failover to browser {new_browser_id}")
            return True, recovered_model
            
        except Exception as e:
            logger.error(f"Error recovering model {model_id} by failover: {str(e)}")
            return False, None


class PerformanceHistoryTracker:
    """
    Tracker for browser and model performance history.
    
    This class tracks and analyzes performance metrics for browsers and models,
    enabling optimization based on historical performance data.
    """
    
    def __init__(self):
        """Initialize the performance history tracker."""
        # Performance history - browser -> model_type -> metrics
        self.history = {}
        
        # Raw performance data
        self.raw_data = []
        
        # Last analysis time
        self.last_analysis_time = None
        
        logger.info("PerformanceHistoryTracker initialized")
    
    async def initialize(self):
        """Initialize the performance history tracker."""
        logger.info("Initializing PerformanceHistoryTracker...")
        
        # Initialize history storage
        self.history = {}
        
        # Start performance analysis task
        asyncio.create_task(self._periodic_analysis())
        
        logger.info("PerformanceHistoryTracker initialization complete")
    
    async def _periodic_analysis(self):
        """Periodically analyze performance data to generate new insights."""
        while True:
            try:
                # Analyze performance data
                await self._analyze_performance_data()
                
                # Record analysis time
                self.last_analysis_time = datetime.now()
                
            except Exception as e:
                logger.error(f"Error in performance analysis: {str(e)}")
            
            # Wait for next analysis cycle (every hour)
            await asyncio.sleep(3600)
    
    async def _analyze_performance_data(self):
        """Analyze performance data to generate insights."""
        # In a real implementation, would perform statistical analysis on raw data
        # For now, just simulate a simple analysis
        
        # Skip if no raw data
        if not self.raw_data:
            return
        
        # Group data by browser and model type
        grouped_data = {}
        
        for entry in self.raw_data:
            browser_id = entry.get("browser_id")
            model_type = entry.get("model_type")
            
            if not browser_id or not model_type:
                continue
                
            if browser_id not in grouped_data:
                grouped_data[browser_id] = {}
                
            if model_type not in grouped_data[browser_id]:
                grouped_data[browser_id][model_type] = []
                
            grouped_data[browser_id][model_type].append(entry)
        
        # Calculate metrics for each browser and model type
        for browser_id, browser_data in grouped_data.items():
            if browser_id not in self.history:
                self.history[browser_id] = {}
                
            for model_type, entries in browser_data.items():
                # Calculate metrics
                latencies = [e.get("latency", 0) for e in entries if "latency" in e]
                success_count = sum(1 for e in entries if e.get("success", False))
                
                avg_latency = sum(latencies) / len(latencies) if latencies else 0
                success_rate = success_count / len(entries) if entries else 0
                
                # Store metrics
                self.history[browser_id][model_type] = {
                    "avg_latency": avg_latency,
                    "success_rate": success_rate,
                    "sample_count": len(entries),
                    "last_updated": datetime.now().isoformat()
                }
        
        logger.info(f"Performance analysis completed, analyzed {len(self.raw_data)} data points")
    
    async def record_operation_performance(
        self,
        browser_id: str,
        model_id: str,
        model_type: str,
        operation_type: str,
        latency: float,
        success: bool,
        metadata: Dict[str, Any] = None
    ):
        """
        Record performance data for an operation.
        
        Args:
            browser_id: Browser ID
            model_id: Model ID
            model_type: Model type
            operation_type: Operation type
            latency: Operation latency in milliseconds
            success: Whether the operation succeeded
            metadata: Additional metadata
        """
        # Add entry to raw data
        self.raw_data.append({
            "browser_id": browser_id,
            "model_id": model_id,
            "model_type": model_type,
            "operation_type": operation_type,
            "latency": latency,
            "success": success,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        })
        
        # Limit raw data size
        if len(self.raw_data) > 10000:
            # Keep most recent 10000 entries
            self.raw_data = self.raw_data[-10000:]
        
        logger.debug(f"Recorded performance data for model {model_id} on browser {browser_id}")
    
    async def get_history(
        self,
        model_type: str = None,
        time_range: str = "7d",
        metrics: List[str] = None
    ) -> Dict[str, Any]:
        """
        Get performance history data.
        
        Args:
            model_type: Optional filter for model type
            time_range: Time range for history
            metrics: Specific metrics to return
            
        Returns:
            Performance history data
        """
        # Parse time range
        days = 7
        if time_range.endswith("d"):
            try:
                days = int(time_range[:-1])
            except ValueError:
                days = 7
        
        # Calculate cutoff time
        cutoff_time = datetime.now() - timedelta(days=days)
        
        # Filter raw data by time and model type
        filtered_data = []
        
        for entry in self.raw_data:
            # Skip if timestamp is missing
            if "timestamp" not in entry:
                continue
                
            # Parse timestamp
            try:
                timestamp = datetime.fromisoformat(entry["timestamp"])
            except ValueError:
                continue
                
            # Skip if older than cutoff
            if timestamp < cutoff_time:
                continue
                
            # Skip if model type doesn't match filter
            if model_type and entry.get("model_type") != model_type:
                continue
                
            filtered_data.append(entry)
        
        # Group by browser ID and model type
        grouped_data = {}
        
        for entry in filtered_data:
            browser_id = entry.get("browser_id")
            entry_model_type = entry.get("model_type")
            
            if not browser_id or not entry_model_type:
                continue
                
            if browser_id not in grouped_data:
                grouped_data[browser_id] = {}
                
            if entry_model_type not in grouped_data[browser_id]:
                grouped_data[browser_id][entry_model_type] = []
                
            grouped_data[browser_id][entry_model_type].append(entry)
        
        # Calculate metrics
        result = {
            "browsers": {},
            "time_range": time_range,
            "total_operations": len(filtered_data)
        }
        
        for browser_id, browser_data in grouped_data.items():
            result["browsers"][browser_id] = {}
            
            for entry_model_type, entries in browser_data.items():
                # Calculate metrics
                latencies = [e.get("latency", 0) for e in entries if "latency" in e]
                success_count = sum(1 for e in entries if e.get("success", False))
                
                avg_latency = sum(latencies) / len(latencies) if latencies else 0
                success_rate = success_count / len(entries) if entries else 0
                
                # Create metrics object
                metrics_obj = {
                    "avg_latency": avg_latency,
                    "success_rate": success_rate,
                    "sample_count": len(entries)
                }
                
                # Filter metrics if specified
                if metrics:
                    metrics_obj = {k: v for k, v in metrics_obj.items() if k in metrics}
                
                result["browsers"][browser_id][entry_model_type] = metrics_obj
        
        return result
    
    async def analyze_trends(self, history: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze performance trends and generate recommendations.
        
        Args:
            history: Performance history data
            
        Returns:
            Analysis and recommendations
        """
        # Simple analysis to determine best browser for each model type
        recommendations = {
            "browser_preferences": {},
            "connection_pool_scaling": None,
            "model_migrations": []
        }
        
        # Analyze browser preferences
        for browser_id, browser_data in history.get("browsers", {}).items():
            for model_type, metrics in browser_data.items():
                # Skip if insufficient data
                if metrics.get("sample_count", 0) < 5:
                    continue
                
                # Check if we already have a recommendation for this model type
                if model_type in recommendations["browser_preferences"]:
                    # Compare with existing recommendation
                    existing_browser = recommendations["browser_preferences"][model_type]
                    existing_metrics = None
                    
                    # Find metrics for existing browser
                    for b_id, b_data in history.get("browsers", {}).items():
                        if b_id == existing_browser and model_type in b_data:
                            existing_metrics = b_data[model_type]
                            break
                    
                    if existing_metrics:
                        # Compare performance
                        if (metrics.get("success_rate", 0) > existing_metrics.get("success_rate", 0) and
                            metrics.get("avg_latency", 100) < existing_metrics.get("avg_latency", 100)):
                            # This browser is better
                            recommendations["browser_preferences"][model_type] = browser_id
                else:
                    # No existing recommendation, add this one
                    recommendations["browser_preferences"][model_type] = browser_id
        
        # Analyze connection pool scaling
        # For now, just keep current size
        recommendations["connection_pool_scaling"] = None
        
        # Generate model migration recommendations
        # Would be more sophisticated in a real implementation
        
        return recommendations