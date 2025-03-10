#!/usr/bin/env python3
"""
Distributed Testing Framework - Worker Node

This module implements the worker node component of the distributed testing framework.
The worker registers with the coordinator, receives tasks, executes them, and reports results.

Usage:
    python worker.py --coordinator http://localhost:8080 --db-path ./benchmark_db.duckdb --api-key YOUR_API_KEY
    """

    import argparse
    import asyncio
    import json
    import logging
    import os
    import platform
    import signal
    import sys
    import uuid
    from datetime import datetime
    from pathlib import Path
    from typing import Dict, List, Optional, Any, Set, Tuple, Callable

    import aiohttp
    import duckdb
    import psutil
    import websockets

# Import security module
    from security import SecurityManager

# Configure logging
    logging.basicConfig()))
    level=logging.INFO,
    format='%()))asctime)s - %()))name)s - %()))levelname)s - %()))message)s',
    handlers=[],
    logging.StreamHandler()))),
    logging.FileHandler()))"worker.log")
    ]
    )
    logger = logging.getLogger()))__name__)

try:
    # Try to import GPU-related libraries if available::
    import torch
    import torch.cuda as cuda
    HAS_TORCH = True:
except ImportError:
    HAS_TORCH = False

try:
    # Try to import OpenVINO if available::
    import openvino
    HAS_OPENVINO = True:
except ImportError:
    HAS_OPENVINO = False

class DistributedTestingWorker:
    """Worker node for distributed testing framework."""
    
    def __init__()))
    self,
    coordinator_url: str,
    hostname: Optional[],str] = None,
    db_path: Optional[],str] = None,
    worker_id: Optional[],str] = None,
    api_key: Optional[],str] = None,
    token: Optional[],str] = None,
    ):
        """
        Initialize the worker.
        
        Args:
            coordinator_url: URL of the coordinator server
            hostname: Hostname of the worker node ()))default: system hostname)
            db_path: Path to the DuckDB database ()))optional)
            worker_id: Worker ID ()))default: generated UUID)
            api_key: API key for authentication with coordinator
            token: JWT token for authentication ()))alternative to API key)
            """
            self.coordinator_url = coordinator_url
            self.hostname = hostname or platform.node())))
            self.db_path = db_path
            self.worker_id = worker_id or str()))uuid.uuid4()))))
            self.api_key = api_key
            self.token = token
        
        # Validate authentication parameters
        if not api_key and not token:
            logger.warning()))"No API key or token provided. Authentication will likely fail.")
        
        # Create security manager
            self.security_manager = SecurityManager())))
        
        # Task state
            self.current_task: Optional[],Dict[],str, Any]] = None
            self.current_task_future: Optional[],asyncio.Future] = None
            self.task_executors: Dict[],str, Callable] = self._register_task_executors())))
        
        # WebSocket connection
            self.ws: Optional[],websockets.WebSocketClientProtocol] = None
            self.ws_connected = False
            self.heartbeat_interval = 10  # seconds
        
        # Performance metrics
            self.current_hardware_metrics: Dict[],str, Any] = {}}}}}}}}}}}}}}}}}}}}}}}}}}
        
        # Database connection
            self.db = None
        if db_path:
            self._init_database())))
        
        # Signal handlers
            self._setup_signal_handlers())))
        
            logger.info()))f"Worker initialized with ID {}}}}}}}}}}}}}}}}}}}}}}}}}self.worker_id}")
    
    def _init_database()))self):
        """Initialize the database connection."""
        try:
            # Connect to database
            self.db = duckdb.connect()))self.db_path)
            logger.info()))f"Database connection established to {}}}}}}}}}}}}}}}}}}}}}}}}}self.db_path}")
        except Exception as e:
            logger.error()))f"Failed to initialize database: {}}}}}}}}}}}}}}}}}}}}}}}}}str()))e)}")
    
    def _setup_signal_handlers()))self):
        """Set up signal handlers for graceful shutdown."""
        signals = ()))signal.SIGHUP, signal.SIGTERM, signal.SIGINT)
        for s in signals:
            signal.signal()))s, self._handle_shutdown_signal)
            logger.info()))"Signal handlers configured")
    
    def _handle_shutdown_signal()))self, sig, frame):
        """Handle shutdown signals."""
        logger.info()))f"Received shutdown signal {}}}}}}}}}}}}}}}}}}}}}}}}}sig}, shutting down...")
        
        # Close database connection
        if self.db:
            self.db.close())))
        
        # Cancel running task if any:
        if self.current_task_future and not self.current_task_future.done()))):
            self.current_task_future.cancel())))
        
        # Exit
            sys.exit()))0)
    
    def _register_task_executors()))self) -> Dict[],str, Callable]:
        """Register task executors for different task types."""
            return {}}}}}}}}}}}}}}}}}}}}}}}}}
            "benchmark": self._execute_benchmark_task,
            "test": self._execute_test_task,
            "custom": self._execute_custom_task,
            }
    
    async def _detect_hardware_capabilities()))self) -> Dict[],str, Any]:
        """Detect hardware capabilities of the worker node."""
        capabilities = {}}}}}}}}}}}}}}}}}}}}}}}}}
        "hostname": self.hostname,
        "platform": platform.platform()))),
        "hardware": [],],
        "cpu": {}}}}}}}}}}}}}}}}}}}}}}}}}
        "model": platform.processor()))) or "Unknown",
        "cores": psutil.cpu_count()))logical=False),
        "threads": psutil.cpu_count()))logical=True),
        "freq_mhz": psutil.cpu_freq()))).max if psutil.cpu_freq()))) else 0,
            },:
                "memory": {}}}}}}}}}}}}}}}}}}}}}}}}}
                "total_gb": round()))psutil.virtual_memory()))).total / ()))1024 ** 3), 2),
                "available_gb": round()))psutil.virtual_memory()))).available / ()))1024 ** 3), 2),
                }
                }
        
        # Always add CPU to hardware list
                capabilities[],"hardware"].append()))"cpu")
        
        # Detect GPU capabilities if PyTorch is available:
        if HAS_TORCH:
            cuda_available = torch.cuda.is_available())))
            if cuda_available:
                capabilities[],"hardware"].append()))"cuda")
                device_count = torch.cuda.device_count())))
                devices = [],]
                
                for i in range()))device_count):
                    device_info = {}}}}}}}}}}}}}}}}}}}}}}}}}
                    "index": i,
                    "name": torch.cuda.get_device_name()))i),
                    "memory_gb": round()))torch.cuda.get_device_properties()))i).total_memory / ()))1024 ** 3), 2),
                    "compute_capability": f"{}}}}}}}}}}}}}}}}}}}}}}}}}torch.cuda.get_device_capability()))i)[],0]}.{}}}}}}}}}}}}}}}}}}}}}}}}}torch.cuda.get_device_capability()))i)[],1]}",
                    }
                    devices.append()))device_info)
                
                    capabilities[],"gpu"] = {}}}}}}}}}}}}}}}}}}}}}}}}}
                    "count": device_count,
                    "devices": devices,
                    "cuda_version": torch.version.cuda,
                    "cudnn_version": torch.backends.cudnn.version()))) if torch.backends.cudnn.is_available()))) else None,
                    }
                
                # Extract first GPU info for summary:
                if devices:
                    capabilities[],"gpu"][],"name"] = devices[],0][],"name"]
                    capabilities[],"gpu"][],"memory_gb"] = devices[],0][],"memory_gb"]
                    capabilities[],"gpu"][],"cuda_compute"] = float()))devices[],0][],"compute_capability"])
            
            # Check for ROCm support
            if hasattr()))torch.version, 'hip') and torch.version.hip:
                capabilities[],"hardware"].append()))"rocm")
                capabilities[],"rocm"] = {}}}}}}}}}}}}}}}}}}}}}}}}}
                "version": torch.version.hip,
                "available": True,
                }
            
            # Check for MPS ()))Apple Silicon) support
            if hasattr()))torch.backends, 'mps') and torch.backends.mps.is_available()))):
                capabilities[],"hardware"].append()))"mps")
                capabilities[],"mps"] = {}}}}}}}}}}}}}}}}}}}}}}}}}
                "available": True,
                }
        
        # Detect OpenVINO if available::
        if HAS_OPENVINO:
            capabilities[],"hardware"].append()))"openvino")
            capabilities[],"openvino"] = {}}}}}}}}}}}}}}}}}}}}}}}}}
            "version": openvino.__version__,
            "available": True,
            }
        
        # Check for WebNN support ()))just for reporting, not actually useful on worker node)
            capabilities[],"webnn"] = {}}}}}}}}}}}}}}}}}}}}}}}}}"available": False}
        
        # Check for WebGPU support ()))just for reporting, not actually useful on worker node)
            capabilities[],"webgpu"] = {}}}}}}}}}}}}}}}}}}}}}}}}}"available": False}
        
                return capabilities
    
    async def connect_to_coordinator()))self):
        """Connect to the coordinator server via WebSocket."""
        ws_url = self.coordinator_url.replace()))"http://", "ws://").replace()))"https://", "wss://") + "/ws"
        
        try:
            self.ws = await websockets.connect()))ws_url)
            self.ws_connected = True
            logger.info()))f"Connected to coordinator at {}}}}}}}}}}}}}}}}}}}}}}}}}ws_url}")
            
            # Authenticate with coordinator
            if not await self._authenticate()))):
                logger.error()))"Authentication failed")
                self.ws_connected = False
                await self.ws.close())))
            return False
            
            # Detect hardware capabilities
            capabilities = await self._detect_hardware_capabilities())))
            
            # Register with coordinator
            register_msg = {}}}}}}}}}}}}}}}}}}}}}}}}}
            "type": "register",
            "worker_id": self.worker_id,
            "hostname": self.hostname,
            "capabilities": capabilities
            }
            
            # Sign message
            signed_msg = self.security_manager.sign_message()))register_msg)
            await self.ws.send()))json.dumps()))signed_msg))
            
            # Receive registration response
            response_data = await self.ws.recv())))
            response = json.loads()))response_data)
            
            # Verify response signature
            if not self.security_manager.verify_message()))response.copy())))):
                logger.warning()))"Received response with invalid signature")
            
            if response.get()))"type") == "register_response" and "worker_id" in response:
                # Update worker ID if provided:: by coordinator
                self.worker_id = response[],"worker_id"]:
                    logger.info()))f"Registered with coordinator, assigned worker ID: {}}}}}}}}}}}}}}}}}}}}}}}}}self.worker_id}")
            
                return True
        except Exception as e:
            logger.error()))f"Failed to connect to coordinator: {}}}}}}}}}}}}}}}}}}}}}}}}}str()))e)}")
            self.ws_connected = False
                return False
            
    async def _authenticate()))self):
        """Authenticate with the coordinator."""
        if not self.ws_connected:
        return False
            
        try:
            # Prepare authentication message
            if self.token:
                # Authenticate with token
                auth_msg = {}}}}}}}}}}}}}}}}}}}}}}}}}
                "type": "auth",
                "auth_type": "token",
                "token": self.token
                }
            elif self.api_key:
                # Authenticate with API key
                auth_msg = {}}}}}}}}}}}}}}}}}}}}}}}}}
                "type": "auth",
                "auth_type": "api_key",
                "api_key": self.api_key,
                "worker_id": self.worker_id
                }
            else:
                logger.error()))"No authentication credentials available")
                return False
                
            # Send authentication message
                await self.ws.send()))json.dumps()))auth_msg))
            
            # Receive response
                response_data = await self.ws.recv())))
                response = json.loads()))response_data)
            
            if response.get()))"type") == "auth_response" and response.get()))"status") == "success":
                logger.info()))"Authentication successful")
                
                # Store token if provided::
                if "token" in response:
                    self.token = response[],"token"]
                    logger.info()))"Received new authentication token")
                
                # Update worker ID if provided::
                if "worker_id" in response:
                    self.worker_id = response[],"worker_id"]
                    logger.info()))f"Worker ID updated to: {}}}}}}}}}}}}}}}}}}}}}}}}}self.worker_id}")
                
                    return True
            else:
                error_msg = response.get()))"message", "Unknown error")
                logger.error()))f"Authentication failed: {}}}}}}}}}}}}}}}}}}}}}}}}}error_msg}")
                    return False
                
        except Exception as e:
            logger.error()))f"Error during authentication: {}}}}}}}}}}}}}}}}}}}}}}}}}str()))e)}")
                    return False
    
    async def reconnect_to_coordinator()))self):
        """Reconnect to the coordinator server."""
        if self.ws:
            try:
                await self.ws.close())))
            except:
                pass
            
                self.ws = None
                self.ws_connected = False
        
                retry_count = 0
                max_retries = 10
                retry_delay = 5  # seconds
        
        while retry_count < max_retries:
            logger.info()))f"Attempting to reconnect to coordinator ()))attempt {}}}}}}}}}}}}}}}}}}}}}}}}}retry_count + 1}/{}}}}}}}}}}}}}}}}}}}}}}}}}max_retries})...")
            
            if await self.connect_to_coordinator()))):
            return True
            
            retry_count += 1
            await asyncio.sleep()))retry_delay)
        
            logger.error()))f"Failed to reconnect to coordinator after {}}}}}}}}}}}}}}}}}}}}}}}}}max_retries} attempts")
                return False
    
    async def send_heartbeat()))self):
        """Send heartbeat to coordinator."""
        if not self.ws_connected:
            logger.warning()))"Not connected to coordinator, cannot send heartbeat")
        return False
        
        try:
            # Collect hardware metrics
            hardware_metrics = self._collect_hardware_metrics())))
            self.current_hardware_metrics = hardware_metrics
            
            # Prepare heartbeat message
            heartbeat = {}}}}}}}}}}}}}}}}}}}}}}}}}
            "type": "heartbeat",
            "worker_id": self.worker_id,
            "timestamp": datetime.now()))).isoformat()))),
            "hardware_metrics": hardware_metrics,
            }
            
            # Add current task status if running::
            if self.current_task:
                heartbeat[],"task_status"] = {}}}}}}}}}}}}}}}}}}}}}}}}}
                "task_id": self.current_task[],"task_id"],
                "status": self.current_task[],"status"],
                "progress": self.current_task.get()))"progress", 0),
                }
            
            # Sign the message
                signed_heartbeat = self.security_manager.sign_message()))heartbeat)
            
            # Send heartbeat
                await self.ws.send()))json.dumps()))signed_heartbeat))
            
            # Receive heartbeat response
                response_data = await self.ws.recv())))
                response = json.loads()))response_data)
            
            # Verify response signature
            if not self.security_manager.verify_message()))response.copy())))):
                logger.warning()))"Received response with invalid signature")
            
            if response.get()))"type") == "heartbeat_response":
                return True
            else:
                logger.warning()))f"Unexpected response to heartbeat: {}}}}}}}}}}}}}}}}}}}}}}}}}response}")
                return False
                
        except Exception as e:
            logger.error()))f"Error sending heartbeat: {}}}}}}}}}}}}}}}}}}}}}}}}}str()))e)}")
            self.ws_connected = False
                return False
    
    async def heartbeat_loop()))self):
        """Heartbeat loop to keep connection alive."""
        while True:
            if self.ws_connected:
                heartbeat_success = await self.send_heartbeat())))
                if not heartbeat_success:
                    # Try to reconnect
                    self.ws_connected = False
                    await self.reconnect_to_coordinator())))
            else:
                # Try to reconnect
                await self.reconnect_to_coordinator())))
            
            # Sleep for heartbeat interval
                await asyncio.sleep()))self.heartbeat_interval)
    
    async def listen_for_tasks()))self):
        """Listen for tasks from coordinator."""
        while True:
            if not self.ws_connected:
                # If not connected, wait and try again
                await asyncio.sleep()))5)
            continue
            
            try:
                # Receive message from coordinator
                message = await self.ws.recv())))
                data = json.loads()))message)
                msg_type = data.get()))"type")
                
                if msg_type == "execute_task":
                    # Task execution request
                    task_id = data.get()))"task_id")
                    task_type = data.get()))"task_type")
                    task_config = data.get()))"config", {}}}}}}}}}}}}}}}}}}}}}}}}}})
                    
                    logger.info()))f"Received task execution request: {}}}}}}}}}}}}}}}}}}}}}}}}}task_id} ())){}}}}}}}}}}}}}}}}}}}}}}}}}task_type})")
                    
                    # Create task data
                    task = {}}}}}}}}}}}}}}}}}}}}}}}}}
                    "task_id": task_id,
                    "type": task_type,
                    "config": task_config,
                    "status": "received",
                    "received": datetime.now()))).isoformat()))),
                    }
                    
                    # Start task execution in background
                    self.current_task = task
                    self.current_task_future = asyncio.create_task()))self._execute_task()))task))
                    
                elif msg_type == "cancel_task":
                    # Task cancellation request
                    task_id = data.get()))"task_id")
                    reason = data.get()))"reason", "user_request")
                    logger.info()))f"Received task cancellation request: {}}}}}}}}}}}}}}}}}}}}}}}}}task_id} ()))reason: {}}}}}}}}}}}}}}}}}}}}}}}}}reason})")
                    
                    # Cancel task if running::
                    if self.current_task and self.current_task[],"task_id"] == task_id:
                        if self.current_task_future and not self.current_task_future.done()))):
                            # Store task data before cancelling
                            task_data = self.current_task.copy())))
                            
                            # Cancel the task
                            self.current_task_future.cancel())))
                            try:
                                await self.current_task_future
                            except asyncio.CancelledError:
                                logger.info()))f"Task {}}}}}}}}}}}}}}}}}}}}}}}}}task_id} cancelled")
                            finally:
                                # Update task status
                                self.current_task[],"status"] = "cancelled"
                                
                                # Send task cancellation notification if this is for migration:
                                if reason == "migration":
                                    # Notify coordinator that task has been cancelled for migration
                                    await self._send_task_cancellation_notification()))task_id, reason)
                                
                                # Clear current task
                                    self.current_task = None
                                    self.current_task_future = None
                        else:
                            logger.warning()))f"Task {}}}}}}}}}}}}}}}}}}}}}}}}}task_id} is not running, cannot cancel")
                    else:
                        logger.warning()))f"Task {}}}}}}}}}}}}}}}}}}}}}}}}}task_id} is not the current task, cannot cancel")
                
                elif msg_type == "error":
                    # Error message from coordinator
                    error_message = data.get()))"message", "Unknown error")
                    logger.error()))f"Received error from coordinator: {}}}}}}}}}}}}}}}}}}}}}}}}}error_message}")
                
                else:
                    # Unknown message type
                    logger.warning()))f"Received unknown message type: {}}}}}}}}}}}}}}}}}}}}}}}}}msg_type}")
                    
            except websockets.exceptions.ConnectionClosed:
                logger.warning()))"Connection to coordinator closed")
                self.ws_connected = False
                await self.reconnect_to_coordinator())))
                
            except Exception as e:
                logger.error()))f"Error listening for tasks: {}}}}}}}}}}}}}}}}}}}}}}}}}str()))e)}")
                self.ws_connected = False
                await self.reconnect_to_coordinator())))
    
    async def _execute_task()))self, task: Dict[],str, Any]) -> Dict[],str, Any]:
        """
        Execute a task.
        
        Args:
            task: Task information
            
        Returns:
            Task result information
            """
            task_id = task[],"task_id"]
            task_type = task[],"type"]
        
        # Update task status
            task[],"status"] = "running"
            task[],"started"] = datetime.now()))).isoformat())))
        
            logger.info()))f"Starting execution of task {}}}}}}}}}}}}}}}}}}}}}}}}}task_id} ())){}}}}}}}}}}}}}}}}}}}}}}}}}task_type})")
        
        try:
            # Execute task based on type
            if task_type in self.task_executors:
                result = await self.task_executors[],task_type]()))task)
            else:
                # Unknown task type
                raise ValueError()))f"Unknown task type: {}}}}}}}}}}}}}}}}}}}}}}}}}task_type}")
            
            # Task completed successfully
                task[],"status"] = "completed"
                task[],"ended"] = datetime.now()))).isoformat())))
                task[],"result"] = result
            
            # Calculate execution time
                started = datetime.fromisoformat()))task[],"started"])
                ended = datetime.fromisoformat()))task[],"ended"])
                execution_time = ()))ended - started).total_seconds())))
            
            # Send result to coordinator
                await self._send_task_result()))task, execution_time)
            
                logger.info()))f"Task {}}}}}}}}}}}}}}}}}}}}}}}}}task_id} completed successfully in {}}}}}}}}}}}}}}}}}}}}}}}}}execution_time:.2f} seconds")
            
        except asyncio.CancelledError:
            # Task was cancelled
            task[],"status"] = "cancelled"
            task[],"ended"] = datetime.now()))).isoformat())))
            
            logger.info()))f"Task {}}}}}}}}}}}}}}}}}}}}}}}}}task_id} was cancelled")
            
        except Exception as e:
            # Task failed
            task[],"status"] = "failed"
            task[],"ended"] = datetime.now()))).isoformat())))
            task[],"error"] = str()))e)
            
            # Calculate execution time
            started = datetime.fromisoformat()))task[],"started"])
            ended = datetime.fromisoformat()))task[],"ended"])
            execution_time = ()))ended - started).total_seconds())))
            
            # Send error to coordinator
            await self._send_task_error()))task, execution_time, str()))e))
            
            logger.error()))f"Task {}}}}}}}}}}}}}}}}}}}}}}}}}task_id} failed after {}}}}}}}}}}}}}}}}}}}}}}}}}execution_time:.2f} seconds: {}}}}}}}}}}}}}}}}}}}}}}}}}str()))e)}")
        
        # Clear current task
            self.current_task = None
        
            return task
    
    async def _send_task_result()))self, task: Dict[],str, Any], execution_time: float):
        """
        Send task result to coordinator.
        
        Args:
            task: Task information
            execution_time: Execution time in seconds
            """
        if not self.ws_connected:
            logger.warning()))"Not connected to coordinator, cannot send task result")
            return
        
        # Collect hardware metrics
            hardware_metrics = self._collect_hardware_metrics())))
        
        # Prepare result message
            result = {}}}}}}}}}}}}}}}}}}}}}}}}}
            "type": "task_result",
            "worker_id": self.worker_id,
            "task_id": task[],"task_id"],
            "status": "completed",
            "execution_time_seconds": execution_time,
            "hardware_metrics": hardware_metrics,
            "result": task.get()))"result", {}}}}}}}}}}}}}}}}}}}}}}}}}}),
            }
        
        try:
            # Sign the message
            signed_result = self.security_manager.sign_message()))result)
            
            # Send result
            await self.ws.send()))json.dumps()))signed_result))
            
            # Receive response
            response_data = await self.ws.recv())))
            response = json.loads()))response_data)
            
            # Verify response signature
            if not self.security_manager.verify_message()))response.copy())))):
                logger.warning()))"Received response with invalid signature")
            
            if response.get()))"type") != "task_result_response":
                logger.warning()))f"Unexpected response to task result: {}}}}}}}}}}}}}}}}}}}}}}}}}response}")
                
        except Exception as e:
            logger.error()))f"Error sending task result: {}}}}}}}}}}}}}}}}}}}}}}}}}str()))e)}")
            self.ws_connected = False
    
    async def _send_task_cancellation_notification()))self, task_id: str, reason: str):
        """
        Send task cancellation notification to coordinator.
        
        Args:
            task_id: Task ID
            reason: Cancellation reason
            """
        if not self.ws_connected:
            logger.warning()))"Not connected to coordinator, cannot send task cancellation notification")
            return
        
        try:
            # Prepare cancellation notification
            notification = {}}}}}}}}}}}}}}}}}}}}}}}}}
            "type": "heartbeat",
            "worker_id": self.worker_id,
            "timestamp": datetime.now()))).isoformat()))),
            "hardware_metrics": self.current_hardware_metrics,
            "task_cancelled": {}}}}}}}}}}}}}}}}}}}}}}}}}
            "task_id": task_id,
            "reason": reason,
            "timestamp": datetime.now()))).isoformat())))
            }
            }
            
            # Sign the message
            signed_notification = self.security_manager.sign_message()))notification)
            
            # Send notification
            await self.ws.send()))json.dumps()))signed_notification))
            
            # Log cancellation
            logger.info()))f"Sent cancellation notification for task {}}}}}}}}}}}}}}}}}}}}}}}}}task_id} to coordinator ()))reason: {}}}}}}}}}}}}}}}}}}}}}}}}}reason})")
            
        except Exception as e:
            logger.error()))f"Error sending task cancellation notification: {}}}}}}}}}}}}}}}}}}}}}}}}}str()))e)}")
    
    async def _send_task_error()))self, task: Dict[],str, Any], execution_time: float, error: str):
        """
        Send task error to coordinator.
        
        Args:
            task: Task information
            execution_time: Execution time in seconds
            error: Error message
            """
        if not self.ws_connected:
            logger.warning()))"Not connected to coordinator, cannot send task error")
            return
        
        # Collect hardware metrics
            hardware_metrics = self._collect_hardware_metrics())))
        
        # Prepare error message
            result = {}}}}}}}}}}}}}}}}}}}}}}}}}
            "type": "task_result",
            "worker_id": self.worker_id,
            "task_id": task[],"task_id"],
            "status": "failed",
            "execution_time_seconds": execution_time,
            "hardware_metrics": hardware_metrics,
            "error": error,
            }
        
        try:
            # Sign the message
            signed_result = self.security_manager.sign_message()))result)
            
            # Send error
            await self.ws.send()))json.dumps()))signed_result))
            
            # Receive response
            response_data = await self.ws.recv())))
            response = json.loads()))response_data)
            
            # Verify response signature
            if not self.security_manager.verify_message()))response.copy())))):
                logger.warning()))"Received response with invalid signature")
            
            if response.get()))"type") != "task_result_response":
                logger.warning()))f"Unexpected response to task error: {}}}}}}}}}}}}}}}}}}}}}}}}}response}")
                
        except Exception as e:
            logger.error()))f"Error sending task error: {}}}}}}}}}}}}}}}}}}}}}}}}}str()))e)}")
            self.ws_connected = False
    
    def _collect_hardware_metrics()))self) -> Dict[],str, Any]:
        """
        Collect hardware metrics.
        
        Returns:
            Hardware metrics information
            """
            metrics = {}}}}}}}}}}}}}}}}}}}}}}}}}
            "cpu_percent": psutil.cpu_percent()))),
            "memory_percent": psutil.virtual_memory()))).percent,
            "memory_used_gb": round()))psutil.virtual_memory()))).used / ()))1024 ** 3), 2),
            "memory_available_gb": round()))psutil.virtual_memory()))).available / ()))1024 ** 3), 2),
            }
        
        # Collect GPU metrics if available::
        if HAS_TORCH and torch.cuda.is_available()))):
            try:
                gpu_metrics = [],]
                for i in range()))torch.cuda.device_count())))):
                    # Get memory stats
                    memory_allocated = round()))torch.cuda.memory_allocated()))i) / ()))1024 ** 3), 2)
                    memory_reserved = round()))torch.cuda.memory_reserved()))i) / ()))1024 ** 3), 2)
                    max_memory = round()))torch.cuda.get_device_properties()))i).total_memory / ()))1024 ** 3), 2)
                    
                    # Calculate utilization
                    memory_utilization = round()))memory_allocated / max_memory * 100, 2) if max_memory > 0 else 0
                    
                    gpu_metrics.append())){}}}}}}}}}}}}}}}}}}}}}}}}}:
                        "index": i,
                        "memory_allocated_gb": memory_allocated,
                        "memory_reserved_gb": memory_reserved,
                        "memory_total_gb": max_memory,
                        "memory_utilization_percent": memory_utilization,
                        })
                
                        metrics[],"gpu"] = gpu_metrics
                
            except Exception as e:
                logger.warning()))f"Error collecting GPU metrics: {}}}}}}}}}}}}}}}}}}}}}}}}}str()))e)}")
        
                        return metrics
    
    async def _execute_benchmark_task()))self, task: Dict[],str, Any]) -> Dict[],str, Any]:
        """
        Execute a benchmark task.
        
        Args:
            task: Task information
            
        Returns:
            Benchmark result information
            """
            config = task[],"config"]
            model_name = config.get()))"model")
            batch_sizes = config.get()))"batch_sizes", [],1, 2, 4, 8, 16])
            precision = config.get()))"precision", "fp32")
            iterations = config.get()))"iterations", 10)
        
            logger.info()))f"Executing benchmark task for model {}}}}}}}}}}}}}}}}}}}}}}}}}model_name} with {}}}}}}}}}}}}}}}}}}}}}}}}}iterations} iterations")
        
        # For now, simulate benchmark execution
            results = {}}}}}}}}}}}}}}}}}}}}}}}}}}
        
        # Simulate results for each batch size
        for batch_size in batch_sizes:
            await asyncio.sleep()))0.5)  # Simulate benchmark execution
            
            # Update progress
            progress = batch_sizes.index()))batch_size) / len()))batch_sizes) * 100
            task[],"progress"] = progress
            
            # Simulate benchmark metrics
            latency = 10 + batch_size * 2  # Simulated latency ()))ms)
            throughput = 1000 / latency * batch_size  # Simulated throughput ()))items/s)
            memory = 1024 + batch_size * 128  # Simulated memory usage ()))MB)
            
            results[],str()))batch_size)] = {}}}}}}}}}}}}}}}}}}}}}}}}}
            "latency_ms": latency,
            "throughput_items_per_second": throughput,
            "memory_mb": memory,
            }
            
            logger.info()))f"Completed batch size {}}}}}}}}}}}}}}}}}}}}}}}}}batch_size} ())){}}}}}}}}}}}}}}}}}}}}}}}}}progress:.1f}%)")
        
        # Return benchmark results
            return {}}}}}}}}}}}}}}}}}}}}}}}}}
            "model": model_name,
            "precision": precision,
            "iterations": iterations,
            "batch_sizes": results,
            }
    
    async def _execute_test_task()))self, task: Dict[],str, Any]) -> Dict[],str, Any]:
        """
        Execute a test task.
        
        Args:
            task: Task information
            
        Returns:
            Test result information
            """
            config = task[],"config"]
            test_file = config.get()))"test_file")
            test_args = config.get()))"test_args", [],])
        
            logger.info()))f"Executing test task for file {}}}}}}}}}}}}}}}}}}}}}}}}}test_file}")
        
        # For now, simulate test execution
            await asyncio.sleep()))2)  # Simulate test execution
        
        # Simulate test results
            results = {}}}}}}}}}}}}}}}}}}}}}}}}}
            "test_file": test_file,
            "success": True,
            "test_count": 10,
            "passed": 9,
            "failed": 1,
            "skipped": 0,
            "duration_seconds": 2.0,
            }
        
            logger.info()))f"Test completed: {}}}}}}}}}}}}}}}}}}}}}}}}}results[],'passed']}/{}}}}}}}}}}}}}}}}}}}}}}}}}results[],'test_count']} tests passed")
        
            return results
    
    async def _execute_custom_task()))self, task: Dict[],str, Any]) -> Dict[],str, Any]:
        """
        Execute a custom task.
        
        Args:
            task: Task information
            
        Returns:
            Custom task result information
            """
            config = task[],"config"]
            task_name = config.get()))"name", "unknown")
        
            logger.info()))f"Executing custom task: {}}}}}}}}}}}}}}}}}}}}}}}}}task_name}")
        
        # For now, simulate custom task execution
            await asyncio.sleep()))1)  # Simulate task execution
        
        # Simulate custom task results
            results = {}}}}}}}}}}}}}}}}}}}}}}}}}
            "name": task_name,
            "success": True,
            "output": f"Executed custom task {}}}}}}}}}}}}}}}}}}}}}}}}}task_name}",
            }
        
            logger.info()))f"Custom task completed: {}}}}}}}}}}}}}}}}}}}}}}}}}task_name}")
        
            return results
    
    async def start()))self):
        """Start the worker node."""
        # Connect to coordinator
        connection_successful = await self.connect_to_coordinator())))
        if not connection_successful:
            logger.error()))"Failed to connect to coordinator, exiting")
        return
        
        # Start background tasks
        heartbeat_task = asyncio.create_task()))self.heartbeat_loop()))))
        listen_task = asyncio.create_task()))self.listen_for_tasks()))))
        
        # Wait for tasks to complete ()))they won't unless they fail)
        await asyncio.gather()))heartbeat_task, listen_task)


async def main()))):
    """Main function."""
    parser = argparse.ArgumentParser()))description="Distributed Testing Framework Worker Node")
    parser.add_argument()))"--coordinator", default="http://localhost:8080", help="URL of the coordinator server")
    parser.add_argument()))"--hostname", help="Hostname of the worker node ()))default: system hostname)")
    parser.add_argument()))"--db-path", help="Path to DuckDB database ()))optional)")
    parser.add_argument()))"--worker-id", help="Worker ID ()))default: generated UUID)")
    parser.add_argument()))"--api-key", help="API key for authentication with coordinator")
    parser.add_argument()))"--token", help="JWT token for authentication ()))alternative to API key)")
    parser.add_argument()))"--token-file", help="Path to file containing JWT token")
    
    args = parser.parse_args())))
    
    # Load token from file if specified
    token = args.token:
    if args.token_file and not token:
        try:
            with open()))args.token_file, 'r') as f:
                token = f.read()))).strip())))
        except Exception as e:
            logger.error()))f"Failed to read token from file: {}}}}}}}}}}}}}}}}}}}}}}}}}str()))e)}")
    
    # Create worker
            worker = DistributedTestingWorker()))
            coordinator_url=args.coordinator,
            hostname=args.hostname,
            db_path=args.db_path,
            worker_id=args.worker_id,
            api_key=args.api_key,
            token=token,
            )
    
            await worker.start())))


if __name__ == "__main__":
    asyncio.run()))main()))))