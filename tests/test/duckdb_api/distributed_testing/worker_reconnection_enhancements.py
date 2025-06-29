#!/usr/bin/env python3
"""
Enhancements for the Worker Reconnection System.

This module extends the Worker Reconnection System with additional features:
1. Enhanced security (authentication, token validation)
2. Performance metrics and telemetry
3. Adaptive connection parameters
4. Traffic compression for efficient bandwidth usage
5. Priority-based message handling
"""

import os
import sys
import time
import json
import zlib
import hmac
import base64
import hashlib
import logging
import threading
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Callable
from pathlib import Path
from enum import Enum
from dataclasses import dataclass, field

# Add parent directory to path to import modules from parent
parent_dir = str(Path(__file__).parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import worker reconnection module
from duckdb_api.distributed_testing.worker_reconnection import (
    ConnectionState, ConnectionStats, WorkerReconnectionManager,
    WorkerReconnectionPlugin
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("worker_reconnection_enhancements")


class MessagePriority(Enum):
    """Priority levels for worker-coordinator messages."""
    CRITICAL = 0    # Highest priority: authentication, errors
    HIGH = 1        # High priority: task results, checkpoints
    NORMAL = 2      # Normal priority: task states, registrations
    LOW = 3         # Low priority: heartbeats, status updates
    BACKGROUND = 4  # Lowest priority: telemetry, statistics


@dataclass
class PerformanceMetrics:
    """Performance metrics for worker reconnection system."""
    message_count: int = 0                       # Total messages sent/received
    message_size_total: int = 0                  # Total size of all messages in bytes
    message_sizes: List[int] = field(default_factory=list)  # Individual message sizes
    message_latencies: List[float] = field(default_factory=list)  # Message latencies in ms
    message_errors: int = 0                      # Number of message errors
    reconnections: int = 0                       # Number of reconnections
    reconnection_durations: List[float] = field(default_factory=list)  # Reconnection durations in seconds
    task_execution_count: int = 0                # Number of tasks executed
    task_success_count: int = 0                  # Number of successfully completed tasks
    task_error_count: int = 0                    # Number of failed tasks
    task_durations: List[float] = field(default_factory=list)  # Task execution durations in seconds
    checkpoints_created: int = 0                 # Number of checkpoints created
    checkpoints_resumed: int = 0                 # Number of tasks resumed from checkpoints
    compression_original_size: int = 0           # Original size of compressed messages
    compression_compressed_size: int = 0         # Compressed size of messages
    created_at: datetime = field(default_factory=datetime.now)  # Creation timestamp
    last_updated: datetime = field(default_factory=datetime.now)  # Last update timestamp
    
    def update_last_updated(self):
        """Update the last_updated timestamp."""
        self.last_updated = datetime.now()
    
    @property
    def compression_ratio(self) -> float:
        """Calculate compression ratio (original/compressed)."""
        if self.compression_compressed_size == 0:
            return 1.0
        return self.compression_original_size / self.compression_compressed_size
    
    @property
    def average_message_size(self) -> float:
        """Calculate average message size in bytes."""
        if not self.message_count:
            return 0.0
        return self.message_size_total / self.message_count
    
    @property
    def average_message_latency(self) -> float:
        """Calculate average message latency in ms."""
        if not self.message_latencies:
            return 0.0
        return sum(self.message_latencies) / len(self.message_latencies)
    
    @property
    def average_reconnection_duration(self) -> float:
        """Calculate average reconnection duration in seconds."""
        if not self.reconnection_durations:
            return 0.0
        return sum(self.reconnection_durations) / len(self.reconnection_durations)
    
    @property
    def average_task_duration(self) -> float:
        """Calculate average task duration in seconds."""
        if not self.task_durations:
            return 0.0
        return sum(self.task_durations) / len(self.task_durations)
    
    @property
    def task_success_rate(self) -> float:
        """Calculate task success rate (0.0-1.0)."""
        if not self.task_execution_count:
            return 1.0
        return self.task_success_count / self.task_execution_count
    
    @property
    def checkpoint_usage_rate(self) -> float:
        """Calculate rate of checkpoint usage (resumed/created)."""
        if not self.checkpoints_created:
            return 0.0
        return self.checkpoints_resumed / self.checkpoints_created
    
    def add_message_size(self, size: int):
        """Add a message size measurement."""
        self.message_count += 1
        self.message_size_total += size
        self.message_sizes.append(size)
        # Keep a reasonable history
        if len(self.message_sizes) > 1000:
            self.message_sizes = self.message_sizes[-1000:]
        self.update_last_updated()
    
    def add_message_latency(self, latency: float):
        """Add a message latency measurement."""
        self.message_latencies.append(latency)
        # Keep a reasonable history
        if len(self.message_latencies) > 1000:
            self.message_latencies = self.message_latencies[-1000:]
        self.update_last_updated()
    
    def add_message_error(self):
        """Increment message error count."""
        self.message_errors += 1
        self.update_last_updated()
    
    def add_reconnection(self, duration: float):
        """Add a reconnection measurement."""
        self.reconnections += 1
        self.reconnection_durations.append(duration)
        # Keep a reasonable history
        if len(self.reconnection_durations) > 100:
            self.reconnection_durations = self.reconnection_durations[-100:]
        self.update_last_updated()
    
    def add_task_execution(self, duration: float, success: bool):
        """Add a task execution measurement."""
        self.task_execution_count += 1
        if success:
            self.task_success_count += 1
        else:
            self.task_error_count += 1
        self.task_durations.append(duration)
        # Keep a reasonable history
        if len(self.task_durations) > 1000:
            self.task_durations = self.task_durations[-1000:]
        self.update_last_updated()
    
    def add_checkpoint_created(self):
        """Increment checkpoint created count."""
        self.checkpoints_created += 1
        self.update_last_updated()
    
    def add_checkpoint_resumed(self):
        """Increment checkpoint resumed count."""
        self.checkpoints_resumed += 1
        self.update_last_updated()
    
    def add_compression_metrics(self, original_size: int, compressed_size: int):
        """Add compression metrics."""
        self.compression_original_size += original_size
        self.compression_compressed_size += compressed_size
        self.update_last_updated()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "message_count": self.message_count,
            "message_size_total": self.message_size_total,
            "average_message_size": self.average_message_size,
            "average_message_latency": self.average_message_latency,
            "message_errors": self.message_errors,
            "reconnections": self.reconnections,
            "average_reconnection_duration": self.average_reconnection_duration,
            "task_execution_count": self.task_execution_count,
            "task_success_count": self.task_success_count,
            "task_error_count": self.task_error_count,
            "task_success_rate": self.task_success_rate,
            "average_task_duration": self.average_task_duration,
            "checkpoints_created": self.checkpoints_created,
            "checkpoints_resumed": self.checkpoints_resumed,
            "checkpoint_usage_rate": self.checkpoint_usage_rate,
            "compression_ratio": self.compression_ratio,
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "uptime": (self.last_updated - self.created_at).total_seconds()
        }


class SecurityEnhancement:
    """Security enhancements for worker reconnection."""
    
    def __init__(self, api_key: str, hmac_key: Optional[str] = None):
        """
        Initialize security enhancement.
        
        Args:
            api_key: API key for authentication
            hmac_key: HMAC key for message signing (if None, a key will be derived from API key)
        """
        self.api_key = api_key
        self.hmac_key = hmac_key or self._derive_hmac_key(api_key)
    
    def _derive_hmac_key(self, api_key: str) -> str:
        """
        Derive HMAC key from API key.
        
        Args:
            api_key: API key
            
        Returns:
            HMAC key
        """
        # Use a simple key derivation: hash the API key
        hash_obj = hashlib.sha256(api_key.encode())
        return base64.b64encode(hash_obj.digest()).decode()
    
    def sign_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sign a message.
        
        Args:
            message: Message to sign
            
        Returns:
            Signed message
        """
        # Create a copy of the message
        signed_message = message.copy()
        
        # Remove existing signature if present
        if "signature" in signed_message:
            del signed_message["signature"]
        
        # Add timestamp if not present
        if "timestamp" not in signed_message:
            signed_message["timestamp"] = datetime.now().isoformat()
        
        # Sort keys for consistent serialization
        message_json = json.dumps(signed_message, sort_keys=True)
        
        # Create HMAC signature
        hmac_obj = hmac.new(
            self.hmac_key.encode(),
            message_json.encode(),
            hashlib.sha256
        )
        signature = base64.b64encode(hmac_obj.digest()).decode()
        
        # Add signature to message
        signed_message["signature"] = signature
        
        return signed_message
    
    def verify_message(self, message: Dict[str, Any]) -> bool:
        """
        Verify a signed message.
        
        Args:
            message: Signed message
            
        Returns:
            True if signature is valid, False otherwise
        """
        # Extract signature
        if "signature" not in message:
            return False
        
        signature = message["signature"]
        
        # Create a copy of the message without signature
        message_copy = message.copy()
        del message_copy["signature"]
        
        # Sort keys for consistent serialization
        message_json = json.dumps(message_copy, sort_keys=True)
        
        # Compute expected signature
        hmac_obj = hmac.new(
            self.hmac_key.encode(),
            message_json.encode(),
            hashlib.sha256
        )
        expected_signature = base64.b64encode(hmac_obj.digest()).decode()
        
        # Compare signatures
        return hmac.compare_digest(signature, expected_signature)
    
    def generate_auth_headers(self) -> Dict[str, str]:
        """
        Generate authentication headers.
        
        Returns:
            Dictionary of authentication headers
        """
        # Create timestamp for replay protection
        timestamp = datetime.now().isoformat()
        
        # Create nonce for additional protection
        nonce = base64.b64encode(os.urandom(16)).decode()
        
        # Create signature
        message = f"{self.api_key}:{timestamp}:{nonce}"
        hmac_obj = hmac.new(
            self.hmac_key.encode(),
            message.encode(),
            hashlib.sha256
        )
        signature = base64.b64encode(hmac_obj.digest()).decode()
        
        # Create headers
        headers = {
            "X-API-Key": self.api_key,
            "X-Timestamp": timestamp,
            "X-Nonce": nonce,
            "X-Signature": signature
        }
        
        return headers


class CompressionEnhancement:
    """Message compression enhancement for worker reconnection."""
    
    def __init__(self, compression_level: int = 6,
                 min_size_for_compression: int = 1024,
                 compression_threshold_ratio: float = 0.9):
        """
        Initialize compression enhancement.
        
        Args:
            compression_level: Compression level (0-9, higher is more compression but slower)
            min_size_for_compression: Minimum message size in bytes for compression
            compression_threshold_ratio: Minimum ratio (compressed/original) to accept compression
        """
        self.compression_level = compression_level
        self.min_size_for_compression = min_size_for_compression
        self.compression_threshold_ratio = compression_threshold_ratio
    
    def compress_message(self, message: str) -> Tuple[str, int, int]:
        """
        Compress a message.
        
        Args:
            message: Message to compress
            
        Returns:
            Tuple of (compressed message, original size, compressed size)
        """
        # Convert message to bytes if needed
        if isinstance(message, str):
            message_bytes = message.encode("utf-8")
        else:
            message_bytes = message
        
        original_size = len(message_bytes)
        
        # Skip compression for small messages
        if original_size < self.min_size_for_compression:
            if isinstance(message, str):
                return message, original_size, original_size
            else:
                return message.decode("utf-8"), original_size, original_size
        
        # Compress message
        compressed_bytes = zlib.compress(message_bytes, self.compression_level)
        compressed_size = len(compressed_bytes)
        
        # Skip compression if not efficient enough
        compression_ratio = compressed_size / original_size
        if compression_ratio > self.compression_threshold_ratio:
            if isinstance(message, str):
                return message, original_size, original_size
            else:
                return message.decode("utf-8"), original_size, original_size
        
        # Encode compressed bytes as base64 for JSON compatibility
        compressed_message = base64.b64encode(compressed_bytes).decode("utf-8")
        
        # Add compression flag and metadata
        result = json.dumps({
            "compressed": True,
            "original_size": original_size,
            "compressed_size": compressed_size,
            "data": compressed_message
        })
        
        return result, original_size, compressed_size
    
    def decompress_message(self, message: str) -> str:
        """
        Decompress a message.
        
        Args:
            message: Compressed message
            
        Returns:
            Decompressed message
        """
        try:
            # Check if message is compressed
            message_data = json.loads(message)
            if isinstance(message_data, dict) and message_data.get("compressed"):
                # Extract compressed data
                compressed_data = base64.b64decode(message_data["data"])
                # Decompress data
                decompressed_data = zlib.decompress(compressed_data)
                # Convert back to string
                return decompressed_data.decode("utf-8")
            
            # Not compressed, return as is
            return message
            
        except (json.JSONDecodeError, KeyError, base64.binascii.Error, zlib.error):
            # Not a compressed message or error in decompression
            return message


class PriorityMessageQueue:
    """Priority-based message queue for worker reconnection."""
    
    def __init__(self):
        """Initialize the priority message queue."""
        # Create queues for different priorities
        self.queues = {
            MessagePriority.CRITICAL: [],
            MessagePriority.HIGH: [],
            MessagePriority.NORMAL: [],
            MessagePriority.LOW: [],
            MessagePriority.BACKGROUND: []
        }
        self.lock = threading.RLock()
    
    def put(self, message: Dict[str, Any], priority: MessagePriority = MessagePriority.NORMAL):
        """
        Add a message to the queue with priority.
        
        Args:
            message: Message to queue
            priority: Message priority
        """
        with self.lock:
            self.queues[priority].append(message)
    
    def get(self) -> Optional[Dict[str, Any]]:
        """
        Get the next message from the queue, respecting priorities.
        
        Returns:
            Next message or None if queue is empty
        """
        with self.lock:
            # Check queues in priority order
            for priority in sorted(MessagePriority, key=lambda p: p.value):
                queue = self.queues[priority]
                if queue:
                    return queue.pop(0)
            
            # No messages available
            return None
    
    def empty(self) -> bool:
        """
        Check if all queues are empty.
        
        Returns:
            True if all queues are empty, False otherwise
        """
        with self.lock:
            return all(not queue for queue in self.queues.values())
    
    def qsize(self) -> int:
        """
        Get total size of all queues.
        
        Returns:
            Total number of queued messages
        """
        with self.lock:
            return sum(len(queue) for queue in self.queues.values())
    
    def qsize_by_priority(self) -> Dict[MessagePriority, int]:
        """
        Get size of each queue by priority.
        
        Returns:
            Dictionary mapping priorities to queue sizes
        """
        with self.lock:
            return {priority: len(queue) for priority, queue in self.queues.items()}


class EnhancedWorkerReconnectionManager(WorkerReconnectionManager):
    """Enhanced worker reconnection manager with additional features."""
    
    def __init__(self, worker_id: str, coordinator_url: str, 
                 api_key: Optional[str] = None, capabilities: Optional[Dict] = None,
                 task_executor: Optional[Callable] = None,
                 hmac_key: Optional[str] = None,
                 enable_compression: bool = True,
                 compression_level: int = 6,
                 enable_priority_queue: bool = True,
                 adaptive_parameters: bool = True):
        """
        Initialize the enhanced worker reconnection manager.
        
        Args:
            worker_id: Unique identifier for this worker
            coordinator_url: URL of the coordinator server
            api_key: API key for authentication
            capabilities: Worker capabilities to report
            task_executor: Callback for executing tasks
            hmac_key: HMAC key for message signing
            enable_compression: Whether to enable message compression
            compression_level: Compression level (0-9)
            enable_priority_queue: Whether to enable priority-based message queue
            adaptive_parameters: Whether to enable adaptive connection parameters
        """
        # Initialize base class
        super().__init__(
            worker_id=worker_id,
            coordinator_url=coordinator_url,
            api_key=api_key,
            capabilities=capabilities,
            task_executor=task_executor
        )
        
        # Security enhancement
        self.security = SecurityEnhancement(api_key or "default-key", hmac_key)
        
        # Compression enhancement
        self.enable_compression = enable_compression
        self.compression = CompressionEnhancement(compression_level=compression_level)
        
        # Priority queue
        self.enable_priority_queue = enable_priority_queue
        if enable_priority_queue:
            self.priority_queue = PriorityMessageQueue()
        
        # Performance metrics
        self.metrics = PerformanceMetrics()
        
        # Adaptive parameters
        self.adaptive_parameters = adaptive_parameters
        self.adaptive_stats = {
            "last_adjustment_time": datetime.now(),
            "adjustment_interval": timedelta(minutes=5),
            "connection_quality_history": [],
            "message_latency_history": []
        }
        
        # Configure with initial enhancements
        self._apply_enhancements()
    
    def _apply_enhancements(self):
        """Apply enhancements to the base configuration."""
        # Override original message queue if using priority queue
        if self.enable_priority_queue:
            # Store original queue for compatibility
            self._original_queue = self.message_queue
            # Replace with our priority queue
            self.message_queue = self.priority_queue
        
        # Enhance authentication headers
        self._original_get_auth_headers = self._get_auth_headers
        self._get_auth_headers = self._enhanced_get_auth_headers
    
    def _enhanced_get_auth_headers(self) -> Dict[str, str]:
        """
        Enhanced version of get_auth_headers using security enhancement.
        
        Returns:
            Dictionary of authentication headers
        """
        # Get base headers
        base_headers = self._original_get_auth_headers()
        
        # Add enhanced security headers
        security_headers = self.security.generate_auth_headers()
        
        # Merge headers
        headers = {**base_headers, **security_headers}
        
        return headers
    
    def _queue_message(self, message: Dict[str, Any], priority: MessagePriority = MessagePriority.NORMAL):
        """
        Queue a message for sending with priority and enhancements.
        
        Args:
            message: Message to send
            priority: Message priority
        """
        # Apply message signing
        signed_message = self.security.sign_message(message)
        
        # Track original message size for metrics
        message_json = json.dumps(signed_message)
        original_size = len(message_json)
        
        # Apply compression if enabled
        if self.enable_compression:
            compressed_message, orig_size, compressed_size = self.compression.compress_message(message_json)
            # Update compression metrics
            self.metrics.add_compression_metrics(orig_size, compressed_size)
            # Use compressed message
            message_to_send = compressed_message
        else:
            # Use original message
            message_to_send = message_json
            
        # Track message metrics
        self.metrics.add_message_size(len(message_to_send))
        
        # Queue message with priority if enabled
        if self.enable_priority_queue:
            self.priority_queue.put(message_to_send, priority)
        else:
            # Fall back to base implementation
            super()._queue_message(signed_message)
    
    def _message_sender_loop(self):
        """Enhanced message sender thread function with priority handling."""
        # If not using priority queue, use base implementation
        if not self.enable_priority_queue:
            return super()._message_sender_loop()
        
        while not self.stop_event.is_set():
            try:
                # Get a message from the priority queue
                message = self.priority_queue.get()
                if message is None:
                    # No message available, sleep briefly
                    time.sleep(0.1)
                    continue
                
                # Send the message
                if self.connection_state == ConnectionState.CONNECTED and self.ws_conn:
                    try:
                        # Record send time for latency measurement
                        send_time = time.time()
                        
                        # Send message
                        if isinstance(message, str):
                            self.ws_conn.send(message)
                        else:
                            self.ws_conn.send(json.dumps(message))
                        
                        logger.debug(f"Sent message: {message[:100] if isinstance(message, str) else message.get('type', 'unknown')}")
                        
                        # Track message send latency (time to acknowledge from WebSocket)
                        # This is approximate since we don't have direct acknowledgement
                        latency = (time.time() - send_time) * 1000  # ms
                        self.metrics.add_message_latency(latency)
                        
                    except Exception as e:
                        logger.error(f"Error sending message: {e}")
                        # Track message error
                        self.metrics.add_message_error()
                        # Requeue the message
                        self.priority_queue.put(message, MessagePriority.HIGH)  # Increase priority for retry
                else:
                    # Not connected, requeue the message with HIGH priority
                    self.priority_queue.put(message, MessagePriority.HIGH)
                
            except Exception as e:
                logger.error(f"Error in message sender loop: {e}")
                import traceback
                logger.debug(traceback.format_exc())
            
            # Small sleep to prevent CPU spinning
            time.sleep(0.01)
    
    def _handle_connection_failure(self):
        """Enhanced connection failure handling with metrics."""
        # Track reconnection start time
        reconnection_start_time = time.time()
        
        # Call base implementation
        super()._handle_connection_failure()
        
        # Track reconnection metrics
        self.metrics.reconnections += 1
        
        # Add reconnection hook for tracking
        def on_reconnection_complete():
            """Callback when reconnection is complete."""
            reconnection_duration = time.time() - reconnection_start_time
            self.metrics.add_reconnection(reconnection_duration)
        
        # Add the hook to be called when reconnected
        # We'll use a thread to check for reconnection completion
        def reconnection_monitor():
            """Monitor reconnection completion."""
            # Wait for reconnection or timeout
            max_wait = 300  # Maximum wait time in seconds
            start_wait = time.time()
            while (time.time() - start_wait < max_wait and 
                   self.connection_state != ConnectionState.CONNECTED and
                   not self.stop_event.is_set()):
                time.sleep(1)
            
            # Check if reconnection completed
            if self.connection_state == ConnectionState.CONNECTED:
                on_reconnection_complete()
        
        # Start monitor thread
        threading.Thread(target=reconnection_monitor, daemon=True).start()
        
        # Apply adaptive parameters if enabled
        if self.adaptive_parameters:
            self._adapt_connection_parameters()
    
    def _adapt_connection_parameters(self):
        """Adapt connection parameters based on connection statistics."""
        now = datetime.now()
        
        # Only adjust parameters periodically
        if now - self.adaptive_stats["last_adjustment_time"] < self.adaptive_stats["adjustment_interval"]:
            return
        
        # Update adjustment time
        self.adaptive_stats["last_adjustment_time"] = now
        
        # Get connection quality metrics
        stability = self.connection_stats.connection_stability
        success_rate = self.connection_stats.connection_success_rate
        
        # Store in history
        self.adaptive_stats["connection_quality_history"].append((stability, success_rate))
        # Keep reasonable history size
        if len(self.adaptive_stats["connection_quality_history"]) > 10:
            self.adaptive_stats["connection_quality_history"] = self.adaptive_stats["connection_quality_history"][-10:]
        
        # Calculate average connection quality
        avg_stability = sum(s for s, _ in self.adaptive_stats["connection_quality_history"]) / len(self.adaptive_stats["connection_quality_history"])
        avg_success_rate = sum(r for _, r in self.adaptive_stats["connection_quality_history"]) / len(self.adaptive_stats["connection_quality_history"])
        
        # Adjust heartbeat interval based on connection quality
        if avg_stability < 0.5:
            # Poor stability: reduce heartbeat interval
            new_heartbeat = max(1.0, self.config["heartbeat_interval"] * 0.8)
            logger.info(f"Adapting heartbeat interval: {self.config['heartbeat_interval']:.1f}s -> {new_heartbeat:.1f}s (low stability)")
            self.config["heartbeat_interval"] = new_heartbeat
        elif avg_stability > 0.9 and self.config["heartbeat_interval"] < 10.0:
            # Good stability: increase heartbeat interval to reduce overhead
            new_heartbeat = min(10.0, self.config["heartbeat_interval"] * 1.2)
            logger.info(f"Adapting heartbeat interval: {self.config['heartbeat_interval']:.1f}s -> {new_heartbeat:.1f}s (high stability)")
            self.config["heartbeat_interval"] = new_heartbeat
        
        # Adjust reconnect delay based on success rate
        if avg_success_rate < 0.5:
            # Poor success rate: increase initial delay to avoid rapid retries
            new_delay = min(5.0, self.config["initial_reconnect_delay"] * 1.5)
            logger.info(f"Adapting reconnect delay: {self.config['initial_reconnect_delay']:.1f}s -> {new_delay:.1f}s (low success rate)")
            self.config["initial_reconnect_delay"] = new_delay
        elif avg_success_rate > 0.8 and self.config["initial_reconnect_delay"] > 0.5:
            # Good success rate: decrease initial delay for faster recovery
            new_delay = max(0.5, self.config["initial_reconnect_delay"] * 0.8)
            logger.info(f"Adapting reconnect delay: {self.config['initial_reconnect_delay']:.1f}s -> {new_delay:.1f}s (high success rate)")
            self.config["initial_reconnect_delay"] = new_delay
    
    def execute_task_with_metrics(self, task_id: str, task_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a task with metrics tracking.
        
        Args:
            task_id: ID of the task
            task_config: Task configuration
            
        Returns:
            Task result
        """
        # Get original task executor
        task_executor = self.task_executor
        
        if not task_executor:
            return {"error": "No task executor available"}
        
        # Check if we have a checkpoint to resume from
        checkpoint_data = self.get_latest_checkpoint(task_id)
        if checkpoint_data:
            # Track checkpoint resumption
            self.metrics.add_checkpoint_resumed()
            
            # Add checkpoint data to task config so worker can resume
            updated_config = task_config.copy()
            updated_config["_checkpoint_data"] = checkpoint_data
            task_config = updated_config
        
        # Note: The actual execution of the task is now handled by the caller,
        # typically EnhancedWorkerReconnectionPlugin._task_executor_wrapper,
        # which directly calls worker.execute_task and updates the metrics.
        # This method is now used primarily for checkpoint handling and
        # for direct execution in scenarios where task_executor is not the plugin wrapper.
        
        # For direct execution cases (not through the plugin), execute the task:
        if task_executor and task_executor.__qualname__ != 'EnhancedWorkerReconnectionPlugin._task_executor_wrapper':
            # Track task execution start time
            start_time = time.time()
            
            try:
                # Execute task using the provided executor
                result = task_executor(task_id, task_config)
                
                # Track successful task completion
                duration = time.time() - start_time
                self.metrics.add_task_execution(duration, True)
                
                return result
                
            except Exception as e:
                # Track failed task completion
                duration = time.time() - start_time
                self.metrics.add_task_execution(duration, False)
                
                # Re-raise exception
                raise
        
        # For execution through the plugin, just pass through to the task executor
        # which will handle metrics tracking
        return task_executor(task_id, task_config)
    
    def create_checkpoint(self, task_id: str, checkpoint_data: Dict[str, Any]) -> str:
        """
        Create a checkpoint for a task with metrics tracking.
        
        Args:
            task_id: ID of the task
            checkpoint_data: Checkpoint data
            
        Returns:
            Checkpoint ID
        """
        # Call base implementation
        checkpoint_id = super().create_checkpoint(task_id, checkpoint_data)
        
        # Track checkpoint creation
        self.metrics.add_checkpoint_created()
        
        return checkpoint_id
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics.
        
        Returns:
            Dictionary of performance metrics
        """
        return self.metrics.to_dict()
    
    def get_queue_status(self) -> Dict[str, Any]:
        """
        Get message queue status.
        
        Returns:
            Dictionary with queue status information
        """
        if self.enable_priority_queue:
            queue_sizes = self.priority_queue.qsize_by_priority()
            return {
                "total_messages": self.priority_queue.qsize(),
                "priority_queues": {str(priority.name): size for priority, size in queue_sizes.items()}
            }
        else:
            return {
                "total_messages": self.message_queue.qsize() if hasattr(self.message_queue, "qsize") else 0
            }
    
    def submit_task_result(self, task_id: str, result: Dict[str, Any], error: Optional[Dict] = None):
        """
        Submit a task result to the coordinator with priority.
        
        Args:
            task_id: ID of the task
            result: Task result data
            error: Optional error information
        """
        # Determine priority based on task result type
        priority = MessagePriority.HIGH
        if error:
            # Task errors get higher priority
            priority = MessagePriority.CRITICAL
        
        # Create result message
        message = {
            "type": "task_result",
            "task_id": task_id,
            "worker_id": self.worker_id,
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
        
        if error:
            message["error"] = error
        
        # Queue message with priority
        self._queue_message(message, priority)
    
    def update_task_state(self, task_id: str, state_updates: Dict[str, Any]):
        """
        Update the state of a task with priority.
        
        Args:
            task_id: ID of the task
            state_updates: Dictionary with state updates
        """
        # Store state in task_states
        with self.task_lock:
            if task_id not in self.task_states:
                self.task_states[task_id] = {}
            
            # Update state
            self.task_states[task_id].update(state_updates)
        
        # Determine priority based on state updates
        priority = MessagePriority.NORMAL
        
        # Higher priority for important state changes
        if "status" in state_updates:
            if state_updates["status"] in ["completed", "failed", "error"]:
                priority = MessagePriority.HIGH
        
        # Critical priority for error states
        if "error" in state_updates or "failure" in state_updates:
            priority = MessagePriority.CRITICAL
        
        # Lower priority for routine progress updates
        if set(state_updates.keys()) == {"progress"} or "heartbeat" in state_updates:
            priority = MessagePriority.LOW
        
        # Send state update message
        message = {
            "type": "task_state",
            "task_id": task_id,
            "worker_id": self.worker_id,
            "state": state_updates,
            "timestamp": datetime.now().isoformat()
        }
        
        # Queue message with priority
        self._queue_message(message, priority)
    
    def send_heartbeat(self):
        """Send a heartbeat message with low priority."""
        # Generate sequence number for latency calculation
        self.heartbeat_sequence += 1
        
        # Create heartbeat message
        message = {
            "type": "heartbeat",
            "worker_id": self.worker_id,
            "sequence": self.heartbeat_sequence,
            "timestamp": datetime.now().isoformat()
        }
        
        # Add current tasks info
        with self.task_lock:
            message["active_tasks"] = list(self.current_tasks.keys())
        
        # Queue message with LOW priority
        self._queue_message(message, MessagePriority.LOW)
        
        # Record sent time
        self.last_heartbeat_sent = datetime.now()
    
    def send_telemetry(self):
        """Send telemetry data with lowest priority."""
        # Create telemetry message
        message = {
            "type": "telemetry",
            "worker_id": self.worker_id,
            "metrics": self.get_performance_metrics(),
            "queue_status": self.get_queue_status(),
            "connection_stats": {
                "state": str(self.connection_state),
                "stability": self.connection_stats.connection_stability,
                "success_rate": self.connection_stats.connection_success_rate,
                "average_latency": self.connection_stats.average_latency,
                "connection_attempts": self.connection_stats.connection_attempts,
                "successful_connections": self.connection_stats.successful_connections,
                "failed_connections": self.connection_stats.failed_connections
            },
            "timestamp": datetime.now().isoformat()
        }
        
        # Queue message with BACKGROUND priority
        self._queue_message(message, MessagePriority.BACKGROUND)


class EnhancedWorkerReconnectionPlugin(WorkerReconnectionPlugin):
    """Enhanced worker reconnection plugin with additional features."""
    
    def __init__(self, worker, api_key: Optional[str] = None, hmac_key: Optional[str] = None,
                 enable_compression: bool = True, compression_level: int = 6,
                 enable_priority_queue: bool = True, adaptive_parameters: bool = True):
        """
        Initialize the enhanced worker reconnection plugin.
        
        Args:
            worker: Worker instance
            api_key: API key for authentication (if None, extracted from worker)
            hmac_key: HMAC key for message signing
            enable_compression: Whether to enable message compression
            compression_level: Compression level (0-9)
            enable_priority_queue: Whether to enable priority-based message queue
            adaptive_parameters: Whether to enable adaptive connection parameters
        """
        self.worker = worker
        
        # Extract API key from worker if not provided
        if api_key is None and hasattr(worker, "api_key"):
            api_key = worker.api_key
        
        # Configure worker attributes for enhanced reconnection
        worker_id = getattr(worker, "worker_id", str(uuid.uuid4()))
        coordinator_url = getattr(worker, "coordinator_url", "")
        capabilities = getattr(worker, "capabilities", {})
        
        # Create enhanced reconnection manager
        self.reconnection_manager = EnhancedWorkerReconnectionManager(
            worker_id=worker_id,
            coordinator_url=coordinator_url,
            api_key=api_key,
            capabilities=capabilities,
            task_executor=self._task_executor_wrapper,
            hmac_key=hmac_key,
            enable_compression=enable_compression,
            compression_level=compression_level,
            enable_priority_queue=enable_priority_queue,
            adaptive_parameters=adaptive_parameters
        )
        
        # Configure based on worker settings
        config = {}
        
        # Check for worker-specific configuration
        if hasattr(worker, "reconnection_config"):
            config.update(worker.reconnection_config)
        
        # Apply configuration
        if config:
            self.reconnection_manager.configure(config)
        
        # Start telemetry thread
        self.telemetry_thread = None
        self.stop_telemetry = threading.Event()
    
    def _task_executor_wrapper(self, task_id: str, task_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Wrap the worker's task executor with metrics tracking.
        
        Args:
            task_id: ID of the task
            task_config: Task configuration
            
        Returns:
            Task result
        """
        # Call worker's execute_task method if available
        if hasattr(self.worker, "execute_task"):
            # Track task execution start time
            start_time = time.time()
            
            try:
                # Execute task directly using the worker's implementation
                result = self.worker.execute_task(task_id, task_config)
                
                # Track successful task completion in metrics
                duration = time.time() - start_time
                self.reconnection_manager.metrics.add_task_execution(duration, True)
                
                return result
                
            except Exception as e:
                # Track failed task completion in metrics
                duration = time.time() - start_time
                self.reconnection_manager.metrics.add_task_execution(duration, False)
                
                # Re-raise exception
                raise
        
        # Default implementation
        return {"error": "Worker does not implement execute_task method"}
    
    def start(self):
        """Start the reconnection manager and telemetry."""
        if self.reconnection_manager:
            self.reconnection_manager.start()
            
            # Start telemetry thread
            self.stop_telemetry.clear()
            self.telemetry_thread = threading.Thread(
                target=self._telemetry_loop,
                daemon=True
            )
            self.telemetry_thread.start()
    
    def stop(self):
        """Stop the reconnection manager and telemetry."""
        # Stop telemetry thread
        self.stop_telemetry.set()
        if self.telemetry_thread and self.telemetry_thread.is_alive():
            self.telemetry_thread.join(timeout=2)
        
        # Stop reconnection manager
        if self.reconnection_manager:
            self.reconnection_manager.stop()
    
    def _telemetry_loop(self):
        """Telemetry thread function."""
        # Wait a bit before starting telemetry
        time.sleep(30)
        
        # Send telemetry periodically
        while not self.stop_telemetry.is_set():
            try:
                # Only send telemetry if connected
                if self.reconnection_manager.is_connected():
                    self.reconnection_manager.send_telemetry()
            except Exception as e:
                logger.error(f"Error sending telemetry: {e}")
            
            # Wait for next telemetry interval or stop
            self.stop_telemetry.wait(300)  # 5 minutes
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics.
        
        Returns:
            Dictionary of performance metrics
        """
        if self.reconnection_manager:
            return self.reconnection_manager.get_performance_metrics()
        return {}


def create_enhanced_worker_reconnection_plugin(
    worker, api_key: Optional[str] = None, hmac_key: Optional[str] = None,
    enable_compression: bool = True, compression_level: int = 6,
    enable_priority_queue: bool = True, adaptive_parameters: bool = True
) -> EnhancedWorkerReconnectionPlugin:
    """
    Create an enhanced worker reconnection plugin for a worker.
    
    Args:
        worker: Worker instance
        api_key: API key for authentication
        hmac_key: HMAC key for message signing
        enable_compression: Whether to enable message compression
        compression_level: Compression level (0-9)
        enable_priority_queue: Whether to enable priority-based message queue
        adaptive_parameters: Whether to enable adaptive connection parameters
        
    Returns:
        Configured EnhancedWorkerReconnectionPlugin instance
    """
    plugin = EnhancedWorkerReconnectionPlugin(
        worker=worker,
        api_key=api_key,
        hmac_key=hmac_key,
        enable_compression=enable_compression,
        compression_level=compression_level,
        enable_priority_queue=enable_priority_queue,
        adaptive_parameters=adaptive_parameters
    )
    plugin.start()
    
    logger.info(f"Created enhanced worker reconnection plugin for worker {getattr(worker, 'worker_id', 'unknown')}")
    return plugin