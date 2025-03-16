#!/usr/bin/env python3
"""
Distributed Testing Framework - Coordinator Server

This module implements the coordinator server component of the distributed testing framework.
The coordinator manages worker nodes, distributes tasks, and aggregates results.

Usage:
    python coordinator.py --host 0.0.0.0 --port 8080 --db-path ./benchmark_db.duckdb
"""

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple

import aiohttp
from aiohttp import web
import duckdb
import websockets

# Import security module
from security import SecurityManager, auth_middleware

# Import health monitoring module
from health_monitor import HealthMonitor

# Import task scheduler module
from task_scheduler import TaskScheduler

# Import load balancer module
from load_balancer import AdaptiveLoadBalancer

# Import plugin architecture
from plugin_architecture import PluginManager, HookType

# Try to import coordinator redundancy
try:
    from coordinator_redundancy import RedundancyManager
    REDUNDANCY_AVAILABLE = True
except ImportError:
    REDUNDANCY_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("coordinator.log")
    ]
)
logger = logging.getLogger(__name__)

class DistributedTestingCoordinator:
    """Coordinator server for distributed testing framework."""
    
    def __init__(self, db_path: str, host: str = "0.0.0.0", port: int = 8080,
                 security_config: str = None, enable_advanced_scheduler: bool = True,
                 enable_health_monitor: bool = True, enable_load_balancer: bool = True,
                 enable_auto_recovery: bool = True, enable_redundancy: bool = True,
                 enable_plugins: bool = True, plugin_dirs: List[str] = None,
                 cluster_nodes: List[str] = None, node_id: str = None,
                 enable_enhanced_error_handling: bool = True):
        """
        Initialize the coordinator.
        
        Args:
            db_path: Path to DuckDB database
            host: Host to bind to
            port: Port to bind to
            security_config: Path to security configuration
            enable_advanced_scheduler: Enable advanced task scheduler
            enable_health_monitor: Enable health monitoring
            enable_load_balancer: Enable adaptive load balancer
            enable_auto_recovery: Enable auto recovery
            enable_redundancy: Enable coordinator redundancy
            enable_plugins: Enable plugin system
            plugin_dirs: List of directories to search for plugins
            cluster_nodes: List of coordinator nodes in the cluster (for redundancy)
            node_id: Unique identifier for this node (for redundancy)
            enable_enhanced_error_handling: Enable enhanced error handling with performance tracking
        """
        
        # Store configuration
        self.db_path = db_path
        self.host = host
        self.port = port
        self.security_config = security_config
        
        # Initialize database
        self.db = None
        if db_path:
            try:
                self.db = duckdb.connect(db_path)
                logger.info(f"Connected to database: {db_path}")
            except Exception as e:
                logger.error(f"Error connecting to database: {e}")
        
        # Initialize components
        self.security_manager = None
        self.health_monitor = None
        self.task_scheduler = None
        self.load_balancer = None
        self.auto_recovery = None
        self.redundancy_manager = None
        self.plugin_manager = None
        self.enhanced_error_handling = None
        
        # Features enabled
        self.enable_advanced_scheduler = enable_advanced_scheduler
        self.enable_health_monitor = enable_health_monitor
        self.enable_load_balancer = enable_load_balancer
        self.enable_auto_recovery = enable_auto_recovery
        self.enable_redundancy = enable_redundancy and REDUNDANCY_AVAILABLE
        self.enable_plugins = enable_plugins
        self.enable_enhanced_error_handling = enable_enhanced_error_handling
        
        # Plugin configuration
        self.plugin_dirs = plugin_dirs or ["plugins"]
        
        # Redundancy configuration
        self.cluster_nodes = cluster_nodes or [f"http://{host}:{port}"]
        self.node_id = node_id or f"node-{uuid.uuid4().hex[:8]}"
        
        # Worker and task tracking
        self.workers = {}
        self.tasks = {}
        self.pending_tasks = set()
        self.running_tasks = {}
        self.completed_tasks = set()
        self.failed_tasks = set()
        self.worker_connections = {}
        
        # Initialize security manager
        self._init_security_manager()
        
        # Initialize plugin manager if enabled
        if self.enable_plugins:
            self._init_plugin_manager()
        
        # Initialize task scheduler if enabled
        if self.enable_advanced_scheduler:
            self._init_task_scheduler()
        
        # Initialize health monitor if enabled
        if self.enable_health_monitor:
            self._init_health_monitor()
        
        # Initialize load balancer if enabled
        if self.enable_load_balancer:
            self._init_load_balancer()
        
        # Initialize auto recovery if enabled
        if self.enable_auto_recovery:
            self._init_auto_recovery()
        
        # Initialize redundancy manager if enabled
        if self.enable_redundancy:
            self._init_redundancy_manager()
            
        # Initialize enhanced error handling if enabled
        if self.enable_enhanced_error_handling:
            self._init_enhanced_error_handling()
    
    def _init_security_manager(self):
        """Initialize the security manager."""
        try:
            self.security_manager = SecurityManager(
                self.db if self.db_path else None,
                config_path=self.security_config
            )
            logger.info("Security manager initialized")
        except Exception as e:
            logger.error(f"Error initializing security manager: {e}")
            self.security_manager = None
    
    def _init_task_scheduler(self):
        """Initialize the advanced task scheduler."""
        try:
            self.task_scheduler = TaskScheduler(self)
            logger.info("Advanced task scheduler initialized")
        except Exception as e:
            logger.error(f"Error initializing task scheduler: {e}")
            self.task_scheduler = None
    
    def _init_health_monitor(self):
        """Initialize the health monitor."""
        try:
            self.health_monitor = HealthMonitor(self)
            logger.info("Health monitor initialized")
        except Exception as e:
            logger.error(f"Error initializing health monitor: {e}")
            self.health_monitor = None
    
    def _init_load_balancer(self):
        """Initialize the adaptive load balancer."""
        try:
            self.load_balancer = AdaptiveLoadBalancer(self)
            logger.info("Adaptive load balancer initialized")
        except Exception as e:
            logger.error(f"Error initializing load balancer: {e}")
            self.load_balancer = None
    
    def _init_auto_recovery(self):
        """Initialize the auto recovery system."""
        try:
            # Import here to avoid circular imports
            from auto_recovery import AutoRecoveryManager
            
            self.auto_recovery = AutoRecoveryManager(
                coordinator=self,
                health_monitor=self.health_monitor
            )
            logger.info("Auto recovery system initialized")
        except Exception as e:
            logger.error(f"Error initializing auto recovery: {e}")
            self.auto_recovery = None
    
    def _init_redundancy_manager(self):
        """Initialize the redundancy manager for coordinator failover."""
        if not REDUNDANCY_AVAILABLE:
            logger.warning("Redundancy module not available, skipping initialization")
            return
            
        try:
            self.redundancy_manager = RedundancyManager(
                coordinator=self,
                cluster_nodes=self.cluster_nodes,
                node_id=self.node_id,
                db_path=self.db_path
            )
            logger.info(f"Redundancy manager initialized with node_id={self.node_id}, cluster_size={len(self.cluster_nodes)}")
        except Exception as e:
            logger.error(f"Error initializing redundancy manager: {e}")
            self.redundancy_manager = None
    
    def _init_plugin_manager(self):
        """Initialize the plugin manager."""
        try:
            self.plugin_manager = PluginManager(
                coordinator=self,
                plugin_dirs=self.plugin_dirs
            )
            logger.info(f"Plugin manager initialized with plugin directories: {self.plugin_dirs}")
        except Exception as e:
            logger.error(f"Error initializing plugin manager: {e}")
            self.plugin_manager = None
            
    def _init_enhanced_error_handling(self):
        """Initialize the enhanced error handling system with performance tracking."""
        try:
            # Import the enhanced error handling integration
            from enhanced_error_handling_integration import install_enhanced_error_handling
            
            # Install enhanced error handling
            self.enhanced_error_handling = install_enhanced_error_handling(self)
            logger.info("Enhanced error handling system initialized with performance tracking")
        except ImportError:
            logger.warning("Enhanced error handling module not available, using standard error handling")
            self.enhanced_error_handling = None
        except Exception as e:
            logger.error(f"Error initializing enhanced error handling: {str(e)}")
            self.enhanced_error_handling = None
    
    async def start(self):
        """Start the coordinator server."""
        # Create web application
        self.app = web.Application(middlewares=[auth_middleware])
        
        # Setup routes
        self._setup_routes()
        
        # Discover and load plugins if enabled
        if self.enable_plugins and self.plugin_manager:
            try:
                # Discover available plugins
                discovered_plugins = await self.plugin_manager.discover_plugins()
                logger.info(f"Discovered {len(discovered_plugins)} plugins: {', '.join(discovered_plugins)}")
                
                # Load discovered plugins
                for plugin_module in discovered_plugins:
                    plugin_id = await self.plugin_manager.load_plugin(plugin_module)
                    if plugin_id:
                        logger.info(f"Loaded plugin: {plugin_id}")
                    else:
                        logger.warning(f"Failed to load plugin: {plugin_module}")
            except Exception as e:
                logger.error(f"Error loading plugins: {str(e)}")
        
        # Start health monitor if enabled
        if self.health_monitor:
            await self.health_monitor.start()
            
        # Start auto recovery if enabled
        if self.auto_recovery:
            await self.auto_recovery.start_recovery_monitoring()
            
        # Start redundancy manager if enabled
        if self.redundancy_manager:
            await self.redundancy_manager.start()
            
        # Start web server
        runner = web.AppRunner(self.app)
        await runner.setup()
        
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()
        
        logger.info(f"Coordinator server started on {self.host}:{self.port}")
        
        # Invoke startup hook for plugins
        if self.enable_plugins and self.plugin_manager:
            await self.plugin_manager.invoke_hook(HookType.COORDINATOR_STARTUP, self)
        
        return site, runner
    
    async def stop(self):
        """Stop the coordinator server."""
        # Invoke shutdown hook for plugins
        if self.enable_plugins and self.plugin_manager:
            await self.plugin_manager.invoke_hook(HookType.COORDINATOR_SHUTDOWN, self)
            
            # Shutdown plugin manager
            await self.plugin_manager.shutdown()
            
        # Run final diagnostics for enhanced error handling
        if self.enhanced_error_handling:
            try:
                diagnostics = await self.enhanced_error_handling.run_diagnostics()
                if diagnostics.get('issues'):
                    logger.warning(f"Enhanced error handling found {len(diagnostics['issues'])} issues:")
                    for issue in diagnostics['issues']:
                        logger.warning(f"  - {issue}")
                logger.info("Enhanced error handling stopped")
        
        # Stop redundancy manager if enabled
        if self.redundancy_manager:
            await self.redundancy_manager.stop()
            
        # Stop health monitor if enabled
        if self.health_monitor:
            await self.health_monitor.stop()
            
        # Close database connection
        if self.db:
            self.db.close()
            
        logger.info("Coordinator server stopped")
        
    def _setup_routes(self):
        """Setup web routes for the coordinator server."""
        # Setup main routes
        self.app.router.add_get('/', self.handle_index)
        self.app.router.add_get('/status', self.handle_status)
        
        # Setup worker routes
        self.app.router.add_post('/api/workers/register', self.handle_worker_register)
        self.app.router.add_get('/api/workers', self.handle_list_workers)
        self.app.router.add_get('/api/workers/{worker_id}', self.handle_get_worker)
        
        # Setup task routes
        self.app.router.add_post('/api/tasks', self.handle_create_task)
        self.app.router.add_get('/api/tasks', self.handle_list_tasks)
        self.app.router.add_get('/api/tasks/{task_id}', self.handle_get_task)
        self.app.router.add_post('/api/tasks/{task_id}/cancel', self.handle_cancel_task)
        
        # Setup WebSocket route for worker connections
        self.app.router.add_get('/ws', self.handle_websocket)
        
        # Setup redundancy routes if redundancy is enabled
        if self.redundancy_manager:
            self.app.router.add_post('/raft', self.handle_raft_request)
            self.app.router.add_post('/raft/sync', self.handle_raft_sync)
            self.app.router.add_post('/raft/forward', self.handle_raft_forward)
            self.app.router.add_get('/raft/status', self.handle_raft_status)
    
    # Redundancy handler methods
    async def handle_raft_request(self, request):
        """Handle Raft consensus protocol requests."""
        if not self.redundancy_manager:
            return web.json_response({'error': 'Redundancy not enabled'}, status=400)
            
        try:
            # Parse request
            data = await request.json()
            request_type = data.get('type')
            
            # Handle request based on type
            if request_type == 'request_vote':
                # Handle RequestVote RPC
                response = await self.redundancy_manager.handle_request_vote(data)
                return web.json_response(response)
            elif request_type == 'append_entries':
                # Handle AppendEntries RPC
                response = await self.redundancy_manager.handle_append_entries(data)
                return web.json_response(response)
            else:
                return web.json_response({'error': f'Unknown request type: {request_type}'}, status=400)
        except Exception as e:
            logger.error(f"Error handling Raft request: {e}")
            return web.json_response({'error': str(e)}, status=500)
    
    async def handle_raft_sync(self, request):
        """Handle state synchronization between coordinators."""
        if not self.redundancy_manager:
            return web.json_response({'error': 'Redundancy not enabled'}, status=400)
            
        try:
            # Parse request
            data = await request.json()
            
            # Handle state sync
            response = await self.redundancy_manager.handle_state_sync(data)
            return web.json_response(response)
        except Exception as e:
            logger.error(f"Error handling state sync: {e}")
            return web.json_response({'error': str(e)}, status=500)
    
    async def handle_raft_forward(self, request):
        """Handle requests forwarded from other coordinators."""
        if not self.redundancy_manager:
            return web.json_response({'error': 'Redundancy not enabled'}, status=400)
            
        try:
            # Parse request
            data = await request.json()
            
            # Handle forwarded request
            response = await self.redundancy_manager.handle_forwarded_request(data)
            return web.json_response(response)
        except Exception as e:
            logger.error(f"Error handling forwarded request: {e}")
            return web.json_response({'error': str(e)}, status=500)
    
    async def handle_raft_status(self, request):
        """Get the current status of the Raft consensus algorithm."""
        if not self.redundancy_manager:
            return web.json_response({'error': 'Redundancy not enabled'}, status=400)
            
        try:
            # Get status
            status = self.redundancy_manager.get_status()
            return web.json_response(status)
        except Exception as e:
            logger.error(f"Error getting Raft status: {e}")
            return web.json_response({'error': str(e)}, status=500)
    
    # Add stubs for the required HTTP handlers
    async def handle_index(self, request):
        """Handle index page request."""
        return web.json_response({
            "status": "ok",
            "version": "1.0.0",
            "name": "Distributed Testing Framework Coordinator"
        })
    
    async def handle_status(self, request):
        """Handle status request."""
        status_data = {
            "status": "ok",
            "workers": len(getattr(self, "workers", {})),
            "tasks": len(getattr(self, "tasks", {})),
            "redundancy_enabled": self.enable_redundancy,
            "redundancy_role": self.redundancy_manager.current_role.value if self.redundancy_manager else "none",
            "enhanced_error_handling_enabled": self.enable_enhanced_error_handling
        }
        
        # Add error handling metrics if available
        if self.enhanced_error_handling:
            try:
                # Get performance metrics
                metrics = self.enhanced_error_handling.get_performance_metrics()
                
                # Get error metrics
                error_metrics = self.enhanced_error_handling.get_error_metrics()
                
                # Add to status
                status_data["error_handling"] = {
                    "total_errors": error_metrics.get("total_errors", 0),
                    "unresolved_errors": error_metrics.get("unresolved_errors", 0),
                    "recovery_strategies": len(metrics.get("strategies", {})),
                    "recovery_executions": metrics.get("overall", {}).get("total_executions", 0),
                    "success_rate": metrics.get("overall", {}).get("overall_success_rate", 0)
                }
            except Exception as e:
                status_data["error_handling"] = {"error": str(e)}
        
        return web.json_response(status_data)
    
    # Worker API handlers
    async def handle_worker_register(self, request):
        """Handle worker registration."""
        try:
            # Verify authentication
            api_key = request.headers.get("X-API-Key")
            if not self.security_manager.verify_api_key(api_key, required_role="worker"):
                return web.json_response({"error": "Unauthorized"}, status=401)
            
            # Parse request data
            data = await request.json()
            hostname = data.get("hostname")
            capabilities = data.get("capabilities", {})
            worker_id = data.get("worker_id", str(uuid.uuid4()))
            
            # Log registration
            logger.info(f"Worker registration request: {hostname} ({worker_id})")
            
            # Validate input
            if not hostname:
                return web.json_response({"error": "Hostname is required"}, status=400)
            
            # Check if worker already exists
            existing_worker = None
            if worker_id in self.workers:
                # Update existing worker
                existing_worker = self.workers[worker_id]
                logger.info(f"Updating existing worker: {worker_id}")
            
            # Create or update worker
            now = datetime.now()
            worker = {
                "worker_id": worker_id,
                "hostname": hostname,
                "capabilities": capabilities,
                "status": "active",
                "registered": now.isoformat(),
                "last_heartbeat": now.isoformat(),
                "tasks_completed": existing_worker.get("tasks_completed", 0) if existing_worker else 0,
                "tasks_failed": existing_worker.get("tasks_failed", 0) if existing_worker else 0
            }
            
            # Save to memory
            self.workers[worker_id] = worker
            
            # Save to database
            if self.db:
                try:
                    # Check if worker exists in database
                    result = self.db.execute(
                        """
                        SELECT worker_id FROM worker_nodes
                        WHERE worker_id = ?
                        """,
                        (worker_id,)
                    ).fetchone()
                    
                    if result:
                        # Update existing record
                        self.db.execute(
                            """
                            UPDATE worker_nodes
                            SET hostname = ?, capabilities = ?, status = ?, last_heartbeat = ?
                            WHERE worker_id = ?
                            """,
                            (hostname, json.dumps(capabilities), "active", now, worker_id)
                        )
                    else:
                        # Insert new record
                        self.db.execute(
                            """
                            INSERT INTO worker_nodes
                            (worker_id, hostname, capabilities, status, registered, last_heartbeat)
                            VALUES (?, ?, ?, ?, ?, ?)
                            """,
                            (worker_id, hostname, json.dumps(capabilities), "active", now, now)
                        )
                except Exception as e:
                    logger.error(f"Error saving worker to database: {e}")
            
            # Return success with worker ID
            return web.json_response({
                "status": "success",
                "worker_id": worker_id,
                "message": "Worker registered successfully"
            })
            
        except Exception as e:
            logger.error(f"Error handling worker registration: {e}")
            return web.json_response({"error": str(e)}, status=500)
    
    async def handle_list_workers(self, request):
        """Handle worker listing."""
        try:
            # Verify authentication (require admin role for full details)
            api_key = request.headers.get("X-API-Key")
            if not self.security_manager.verify_api_key(api_key, required_role="admin"):
                # If not admin, check if they have a valid worker role
                if not self.security_manager.verify_api_key(api_key, required_role="worker"):
                    return web.json_response({"error": "Unauthorized"}, status=401)
                # Workers get limited info (for discovery)
                limited_info = True
            else:
                limited_info = False
            
            # Get query parameters
            status = request.query.get("status")
            hardware = request.query.get("hardware")
            limit = request.query.get("limit")
            offset = request.query.get("offset", "0")
            
            # Convert to appropriate types
            try:
                limit = int(limit) if limit else None
                offset = int(offset)
            except ValueError:
                return web.json_response({"error": "Invalid limit or offset parameter"}, status=400)
            
            # Filter workers based on query parameters
            filtered_workers = []
            
            for worker_id, worker in self.workers.items():
                # Filter by status if specified
                if status and worker.get("status") != status:
                    continue
                
                # Filter by hardware if specified
                if hardware:
                    hardware_types = hardware.split(",")
                    worker_hardware = worker.get("capabilities", {}).get("hardware", [])
                    if not all(hw in worker_hardware for hw in hardware_types):
                        continue
                
                # For limited info, only include basic fields
                if limited_info:
                    filtered_workers.append({
                        "worker_id": worker_id,
                        "hostname": worker.get("hostname"),
                        "status": worker.get("status"),
                        "hardware": worker.get("capabilities", {}).get("hardware", [])
                    })
                else:
                    # Include full worker info for admins
                    worker_copy = worker.copy()
                    
                    # Calculate task stats
                    active_tasks = sum(1 for task_id, wid in self.running_tasks.items() if wid == worker_id)
                    worker_copy["active_tasks"] = active_tasks
                    
                    filtered_workers.append(worker_copy)
            
            # Apply pagination
            if offset >= len(filtered_workers):
                paginated_workers = []
            else:
                paginated_workers = filtered_workers[offset:offset + limit] if limit else filtered_workers[offset:]
            
            # Return worker list
            return web.json_response({
                "status": "success",
                "total": len(filtered_workers),
                "offset": offset,
                "limit": limit,
                "workers": paginated_workers
            })
            
        except Exception as e:
            logger.error(f"Error handling worker listing: {e}")
            return web.json_response({"error": str(e)}, status=500)
    
    async def handle_get_worker(self, request):
        """Handle get worker details."""
        try:
            # Verify authentication (require admin role for full details)
            api_key = request.headers.get("X-API-Key")
            if not self.security_manager.verify_api_key(api_key, required_role="admin"):
                # If not admin, check if they have a valid worker role
                if not self.security_manager.verify_api_key(api_key, required_role="worker"):
                    return web.json_response({"error": "Unauthorized"}, status=401)
                # Workers get limited info
                limited_info = True
            else:
                limited_info = False
            
            # Get worker ID from URL path
            worker_id = request.match_info.get("worker_id")
            
            # Check if worker exists
            if worker_id not in self.workers:
                return web.json_response({"error": "Worker not found"}, status=404)
            
            worker = self.workers[worker_id]
            
            # Include health data if health monitor is enabled
            worker_health = None
            if self.health_monitor:
                worker_health = self.health_monitor.get_worker_details(worker_id)
            
            # For limited info, only include basic fields
            if limited_info:
                result = {
                    "worker_id": worker_id,
                    "hostname": worker.get("hostname"),
                    "status": worker.get("status"),
                    "hardware": worker.get("capabilities", {}).get("hardware", [])
                }
            else:
                # Include full worker info for admins
                worker_copy = worker.copy()
                
                # Calculate task stats
                active_tasks = []
                for task_id, wid in self.running_tasks.items():
                    if wid == worker_id and task_id in self.tasks:
                        task = self.tasks[task_id]
                        active_tasks.append({
                            "task_id": task_id,
                            "type": task.get("type"),
                            "started": task.get("started")
                        })
                
                worker_copy["active_tasks"] = active_tasks
                worker_copy["active_task_count"] = len(active_tasks)
                
                if worker_health:
                    worker_copy["health"] = worker_health
                
                result = worker_copy
            
            # Return worker details
            return web.json_response({
                "status": "success",
                "worker": result
            })
            
        except Exception as e:
            logger.error(f"Error handling get worker: {e}")
            return web.json_response({"error": str(e)}, status=500)
    
    # Task API handlers
    async def handle_create_task(self, request):
        """Handle task creation."""
        try:
            # Verify authentication
            api_key = request.headers.get("X-API-Key")
            if not self.security_manager.verify_api_key(api_key):
                return web.json_response({"error": "Unauthorized"}, status=401)
            
            # Parse request data
            data = await request.json()
            
            # Validate required fields
            task_type = data.get("type")
            if not task_type:
                return web.json_response({"error": "Task type is required"}, status=400)
            
            # Generate task ID
            task_id = data.get("task_id", str(uuid.uuid4()))
            
            # Check if task ID is already in use
            if task_id in self.tasks:
                return web.json_response({"error": f"Task ID {task_id} is already in use"}, status=400)
            
            # Get other fields with defaults
            priority = int(data.get("priority", 1))
            config = data.get("config", {})
            requirements = data.get("requirements", {})
            
            # Create task
            now = datetime.now()
            task = {
                "task_id": task_id,
                "type": task_type,
                "priority": priority,
                "config": config,
                "requirements": requirements,
                "status": "pending",
                "created": now.isoformat(),
                "attempts": 0
            }
            
            # Store task in memory
            self.tasks[task_id] = task
            self.pending_tasks.add(task_id)
            
            # Save to database
            if self.db:
                try:
                    self.db.execute(
                        """
                        INSERT INTO distributed_tasks
                        (task_id, type, priority, config, requirements, status, created)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            task_id,
                            task_type,
                            priority,
                            json.dumps(config),
                            json.dumps(requirements),
                            "pending",
                            now
                        )
                    )
                except Exception as e:
                    logger.error(f"Error saving task to database: {e}")
            
            # Try to assign the task to a worker immediately
            await self._assign_pending_tasks()
            
            # Return task info
            return web.json_response({
                "status": "success",
                "task_id": task_id,
                "message": "Task created successfully"
            })
            
        except Exception as e:
            logger.error(f"Error creating task: {e}")
            return web.json_response({"error": str(e)}, status=500)
    
    async def handle_list_tasks(self, request):
        """Handle task listing."""
        try:
            # Verify authentication
            api_key = request.headers.get("X-API-Key")
            if not self.security_manager.verify_api_key(api_key):
                return web.json_response({"error": "Unauthorized"}, status=401)
            
            # Get query parameters
            status = request.query.get("status")
            type = request.query.get("type")
            worker_id = request.query.get("worker_id")
            limit = request.query.get("limit")
            offset = request.query.get("offset", "0")
            
            # Convert to appropriate types
            try:
                limit = int(limit) if limit else None
                offset = int(offset)
            except ValueError:
                return web.json_response({"error": "Invalid limit or offset parameter"}, status=400)
            
            # Filter tasks based on query parameters
            filtered_tasks = []
            
            for task_id, task in self.tasks.items():
                # Filter by status if specified
                if status and task.get("status") != status:
                    continue
                
                # Filter by type if specified
                if type and task.get("type") != type:
                    continue
                
                # Filter by worker ID if specified
                if worker_id:
                    # For running tasks, check against running_tasks mapping
                    if task.get("status") == "running":
                        assigned_worker = self.running_tasks.get(task_id)
                        if assigned_worker != worker_id:
                            continue
                    else:
                        # For non-running tasks, they're not assigned to this worker
                        continue
                
                # Add task to filtered list
                filtered_tasks.append(task.copy())
            
            # Apply pagination
            if offset >= len(filtered_tasks):
                paginated_tasks = []
            else:
                paginated_tasks = filtered_tasks[offset:offset + limit] if limit else filtered_tasks[offset:]
            
            # Return task list
            return web.json_response({
                "status": "success",
                "total": len(filtered_tasks),
                "offset": offset,
                "limit": limit,
                "tasks": paginated_tasks
            })
            
        except Exception as e:
            logger.error(f"Error listing tasks: {e}")
            return web.json_response({"error": str(e)}, status=500)
    
    async def handle_get_task(self, request):
        """Handle get task details."""
        try:
            # Verify authentication
            api_key = request.headers.get("X-API-Key")
            if not self.security_manager.verify_api_key(api_key):
                return web.json_response({"error": "Unauthorized"}, status=401)
            
            # Get task ID from URL path
            task_id = request.match_info.get("task_id")
            
            # Check if task exists
            if task_id not in self.tasks:
                return web.json_response({"error": "Task not found"}, status=404)
            
            task = self.tasks[task_id]
            
            # Add worker info if task is running
            if task.get("status") == "running" and task_id in self.running_tasks:
                worker_id = self.running_tasks[task_id]
                task["worker_id"] = worker_id
                
                # Add worker hostname if available
                if worker_id in self.workers:
                    task["worker_hostname"] = self.workers[worker_id].get("hostname")
            
            # Add health data if health monitor is enabled
            task_health = None
            if self.health_monitor:
                task_health = self.health_monitor.get_task_details(task_id)
            
            # Create result with task info and health data
            result = task.copy()
            if task_health:
                result["health"] = task_health
            
            # Return task details
            return web.json_response({
                "status": "success",
                "task": result
            })
            
        except Exception as e:
            logger.error(f"Error getting task: {e}")
            return web.json_response({"error": str(e)}, status=500)
    
    async def handle_cancel_task(self, request):
        """Handle task cancellation."""
        try:
            # Verify authentication
            api_key = request.headers.get("X-API-Key")
            if not self.security_manager.verify_api_key(api_key):
                return web.json_response({"error": "Unauthorized"}, status=401)
            
            # Get task ID from URL path
            task_id = request.match_info.get("task_id")
            
            # Check if task exists
            if task_id not in self.tasks:
                return web.json_response({"error": "Task not found"}, status=404)
            
            task = self.tasks[task_id]
            task_status = task.get("status")
            
            # Check if task is already completed or failed
            if task_status in ["completed", "failed", "cancelled"]:
                return web.json_response({
                    "status": "success",
                    "message": f"Task was already {task_status}"
                })
            
            # If task is pending, just remove from pending tasks
            if task_status == "pending":
                if task_id in self.pending_tasks:
                    self.pending_tasks.remove(task_id)
                
                # Update task status
                task["status"] = "cancelled"
                task["cancelled"] = datetime.now().isoformat()
                
                # Update in database
                if self.db:
                    try:
                        self.db.execute(
                            """
                            UPDATE distributed_tasks
                            SET status = 'cancelled', cancelled = ?
                            WHERE task_id = ?
                            """,
                            (datetime.now(), task_id)
                        )
                    except Exception as e:
                        logger.error(f"Error updating task in database: {e}")
                
                return web.json_response({
                    "status": "success",
                    "message": "Task cancelled successfully"
                })
            
            # If task is running, notify worker to cancel it
            if task_status == "running" and task_id in self.running_tasks:
                worker_id = self.running_tasks[task_id]
                
                # Check if worker is connected
                if worker_id not in self.worker_connections:
                    return web.json_response({"error": "Worker is not connected"}, status=500)
                
                # Send cancellation message to worker
                try:
                    await self.worker_connections[worker_id].send_json({
                        "type": "cancel_task",
                        "task_id": task_id,
                        "reason": "user_request"
                    })
                except Exception as e:
                    logger.error(f"Error sending cancellation request: {e}")
                    return web.json_response({"error": f"Error sending cancellation request: {str(e)}"}, status=500)
                
                # Update task status
                task["status"] = "cancelling"
                task["cancelling"] = datetime.now().isoformat()
                
                # Update in database
                if self.db:
                    try:
                        self.db.execute(
                            """
                            UPDATE distributed_tasks
                            SET status = 'cancelling', cancelling = ?
                            WHERE task_id = ?
                            """,
                            (datetime.now(), task_id)
                        )
                    except Exception as e:
                        logger.error(f"Error updating task in database: {e}")
                
                return web.json_response({
                    "status": "success",
                    "message": "Task cancellation request sent"
                })
            
            # Some other status
            return web.json_response({"error": f"Cannot cancel task with status: {task_status}"}, status=400)
            
        except Exception as e:
            logger.error(f"Error cancelling task: {e}")
            return web.json_response({"error": str(e)}, status=500)
    
    # WebSocket handler
    async def handle_websocket(self, request):
        """Handle WebSocket connection from worker."""
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        
        worker_id = None
        authenticated = False
        
        logger.info(f"WebSocket connection established from {request.remote}")
        
        try:
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        msg_type = data.get("type")
                        
                        # Handle authentication first
                        if msg_type == "auth":
                            auth_type = data.get("auth_type")
                            
                            if auth_type == "api_key":
                                api_key = data.get("api_key")
                                worker_id = data.get("worker_id")
                                
                                # Verify API key
                                if self.security_manager.verify_api_key(api_key, required_role="worker"):
                                    authenticated = True
                                    
                                    # Generate JWT token for future authentication
                                    token = self.security_manager.generate_token(worker_id, "worker")
                                    
                                    # Send success response with token
                                    await ws.send_json({
                                        "type": "auth_response",
                                        "status": "success",
                                        "token": token,
                                        "worker_id": worker_id
                                    })
                                    
                                    logger.info(f"Worker {worker_id} authenticated via API key")
                                else:
                                    # Send failure response
                                    await ws.send_json({
                                        "type": "auth_response",
                                        "status": "error",
                                        "message": "Invalid API key"
                                    })
                                    
                                    logger.warning(f"Failed authentication attempt from {request.remote} with API key")
                            
                            elif auth_type == "token":
                                token = data.get("token")
                                
                                # Verify token
                                decoded = self.security_manager.verify_token(token)
                                if decoded:
                                    authenticated = True
                                    worker_id = decoded.get("sub")
                                    
                                    # Send success response
                                    await ws.send_json({
                                        "type": "auth_response",
                                        "status": "success",
                                        "worker_id": worker_id
                                    })
                                    
                                    logger.info(f"Worker {worker_id} authenticated via token")
                                else:
                                    # Send failure response
                                    await ws.send_json({
                                        "type": "auth_response",
                                        "status": "error",
                                        "message": "Invalid token"
                                    })
                                    
                                    logger.warning(f"Failed authentication attempt from {request.remote} with token")
                            
                            else:
                                # Unknown auth type
                                await ws.send_json({
                                    "type": "auth_response",
                                    "status": "error",
                                    "message": "Unknown authentication type"
                                })
                                
                                logger.warning(f"Unknown authentication type from {request.remote}: {auth_type}")
                                
                        # Require authentication for all other message types
                        elif not authenticated:
                            await ws.send_json({
                                "type": "error",
                                "message": "Not authenticated"
                            })
                            
                            logger.warning(f"Unauthenticated message from {request.remote}")
                            
                        # Handle register message
                        elif msg_type == "register":
                            worker_id = data.get("worker_id", worker_id)
                            hostname = data.get("hostname")
                            capabilities = data.get("capabilities", {})
                            
                            if not worker_id or not hostname:
                                await ws.send_json({
                                    "type": "register_response",
                                    "status": "error",
                                    "message": "Worker ID and hostname are required"
                                })
                                continue
                            
                            # Register worker
                            now = datetime.now()
                            
                            # Check if worker already exists
                            existing_worker = None
                            if worker_id in self.workers:
                                existing_worker = self.workers[worker_id]
                            
                            # Create or update worker
                            worker = {
                                "worker_id": worker_id,
                                "hostname": hostname,
                                "capabilities": capabilities,
                                "status": "active",
                                "registered": existing_worker.get("registered", now.isoformat()) if existing_worker else now.isoformat(),
                                "last_heartbeat": now.isoformat(),
                                "tasks_completed": existing_worker.get("tasks_completed", 0) if existing_worker else 0,
                                "tasks_failed": existing_worker.get("tasks_failed", 0) if existing_worker else 0
                            }
                            
                            # Save to memory
                            self.workers[worker_id] = worker
                            self.worker_connections[worker_id] = ws
                            
                            # Save to database
                            if self.db:
                                try:
                                    # Check if worker exists in database
                                    result = self.db.execute(
                                        """
                                        SELECT worker_id FROM worker_nodes
                                        WHERE worker_id = ?
                                        """,
                                        (worker_id,)
                                    ).fetchone()
                                    
                                    if result:
                                        # Update existing record
                                        self.db.execute(
                                            """
                                            UPDATE worker_nodes
                                            SET hostname = ?, capabilities = ?, status = ?, last_heartbeat = ?
                                            WHERE worker_id = ?
                                            """,
                                            (hostname, json.dumps(capabilities), "active", now, worker_id)
                                        )
                                    else:
                                        # Insert new record
                                        self.db.execute(
                                            """
                                            INSERT INTO worker_nodes
                                            (worker_id, hostname, capabilities, status, registered, last_heartbeat)
                                            VALUES (?, ?, ?, ?, ?, ?)
                                            """,
                                            (worker_id, hostname, json.dumps(capabilities), "active", now, now)
                                        )
                                except Exception as e:
                                    logger.error(f"Error saving worker to database: {e}")
                            
                            # Send registration response
                            await ws.send_json({
                                "type": "register_response",
                                "status": "success",
                                "worker_id": worker_id
                            })
                            
                            logger.info(f"Worker {worker_id} ({hostname}) registered")
                            
                            # Send pending tasks if any are available
                            await self._assign_pending_tasks()
                        
                        # Handle heartbeat message
                        elif msg_type == "heartbeat":
                            if not worker_id or worker_id not in self.workers:
                                await ws.send_json({
                                    "type": "error",
                                    "message": "Worker not registered"
                                })
                                continue
                            
                            # Update worker heartbeat
                            now = datetime.now()
                            self.workers[worker_id]["last_heartbeat"] = now.isoformat()
                            
                            # Update hardware metrics if provided
                            if "hardware_metrics" in data:
                                self.workers[worker_id]["hardware_metrics"] = data["hardware_metrics"]
                            
                            # Update health status if provided
                            if "health_status" in data:
                                self.workers[worker_id]["health_status"] = data["health_status"]
                            
                            # Check for task_status updates
                            if "task_status" in data:
                                task_status = data["task_status"]
                                task_id = task_status.get("task_id")
                                status = task_status.get("status")
                                progress = task_status.get("progress", 0)
                                
                                # Update task progress if task exists
                                if task_id in self.tasks:
                                    self.tasks[task_id]["progress"] = progress
                            
                            # Check for task_cancelled notification
                            if "task_cancelled" in data:
                                task_cancelled = data["task_cancelled"]
                                task_id = task_cancelled.get("task_id")
                                reason = task_cancelled.get("reason")
                                
                                # Handle task cancellation (especially for migrations)
                                if reason == "migration" and self.load_balancer and self.load_balancer.enable_task_migration:
                                    await self.load_balancer.handle_task_cancelled_for_migration(task_id, worker_id)
                            
                            # Update in database
                            if self.db:
                                try:
                                    self.db.execute(
                                        """
                                        UPDATE worker_nodes
                                        SET last_heartbeat = ?, status = ?
                                        WHERE worker_id = ?
                                        """,
                                        (now, "active", worker_id)
                                    )
                                except Exception as e:
                                    logger.error(f"Error updating worker heartbeat in database: {e}")
                            
                            # Send heartbeat response
                            await ws.send_json({
                                "type": "heartbeat_response",
                                "timestamp": now.isoformat()
                            })
                            
                            # If worker was previously offline, try to assign tasks now
                            if self.workers[worker_id].get("status") != "active":
                                self.workers[worker_id]["status"] = "active"
                                await self._assign_pending_tasks()
                        
                        # Handle task_result message
                        elif msg_type == "task_result":
                            task_id = data.get("task_id")
                            status = data.get("status")
                            result = data.get("result", {})
                            error = data.get("error")
                            execution_time = data.get("execution_time_seconds", 0)
                            
                            if not task_id:
                                await ws.send_json({
                                    "type": "error",
                                    "message": "Task ID is required"
                                })
                                continue
                            
                            # Check if task exists
                            if task_id not in self.tasks:
                                await ws.send_json({
                                    "type": "error",
                                    "message": f"Task {task_id} not found"
                                })
                                continue
                            
                            # Check if this worker is assigned to this task
                            if task_id in self.running_tasks and self.running_tasks[task_id] != worker_id:
                                await ws.send_json({
                                    "type": "error",
                                    "message": f"Task {task_id} is not assigned to this worker"
                                })
                                continue
                            
                            # Handle task result based on status
                            if status == "completed":
                                await self._handle_task_completed(task_id, worker_id, result, execution_time)
                            elif status == "failed":
                                await self._handle_task_failed(task_id, worker_id, error, execution_time)
                            else:
                                await ws.send_json({
                                    "type": "error",
                                    "message": f"Unknown task status: {status}"
                                })
                                continue
                            
                            # Send task result response
                            await ws.send_json({
                                "type": "task_result_response",
                                "task_id": task_id,
                                "received": datetime.now().isoformat()
                            })
                            
                            # Try to assign more tasks to this worker
                            await self._assign_pending_tasks()
                        
                        # Unknown message type
                        else:
                            await ws.send_json({
                                "type": "error",
                                "message": f"Unknown message type: {msg_type}"
                            })
                            
                            logger.warning(f"Unknown message type from worker {worker_id}: {msg_type}")
                    
                    except json.JSONDecodeError:
                        await ws.send_json({
                            "type": "error",
                            "message": "Invalid JSON format"
                        })
                        
                        logger.warning(f"Invalid JSON format from {request.remote}")
                    
                    except Exception as e:
                        logger.error(f"Error processing WebSocket message: {e}")
                        
                        await ws.send_json({
                            "type": "error",
                            "message": f"Error processing message: {str(e)}"
                        })
                
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error(f"WebSocket connection closed with exception: {ws.exception()}")
                    break
        
        finally:
            # Connection closed, clean up
            if worker_id and worker_id in self.worker_connections:
                del self.worker_connections[worker_id]
                
                # Update worker status to offline
                if worker_id in self.workers:
                    self.workers[worker_id]["status"] = "offline"
                    
                    # Update in database
                    if self.db:
                        try:
                            self.db.execute(
                                """
                                UPDATE worker_nodes
                                SET status = 'offline'
                                WHERE worker_id = ?
                                """,
                                (worker_id,)
                            )
                        except Exception as e:
                            logger.error(f"Error updating worker status in database: {e}")
                    
                    logger.info(f"Worker {worker_id} disconnected")
                    
                    # If health monitoring and auto recovery is enabled, handle worker failure
                    if self.health_monitor and self.health_monitor.auto_recovery:
                        await self.health_monitor.handle_worker_failure(worker_id)
            
            await ws.close()
        
        return ws
        
    async def _assign_pending_tasks(self):
        """Assign pending tasks to available workers with enhanced batching and optimization."""
        # Skip if no pending tasks
        if not self.pending_tasks:
            return 0
        
        # Get available workers
        available_workers = {}
        for worker_id, worker in self.workers.items():
            if worker.get("status") == "active" and worker_id in self.worker_connections:
                available_workers[worker_id] = worker
        
        # Skip if no available workers
        if not available_workers:
            return 0
        
        # Use advanced scheduler if enabled
        if self.enable_advanced_scheduler and self.task_scheduler:
            return await self.task_scheduler.schedule_pending_tasks()
        
        # Enhanced scheduling: assign tasks to workers with batching support
        tasks_assigned = 0
        
        # Sort pending tasks by priority (higher number = higher priority)
        pending_tasks = sorted(
            [self.tasks[task_id] for task_id in self.pending_tasks],
            key=lambda t: t.get("priority", 0),
            reverse=True
        )
        
        # Organize tasks by type to enable batching
        tasks_by_type = {}
        for task in pending_tasks:
            task_type = task.get("type", "unknown")
            if task_type not in tasks_by_type:
                tasks_by_type[task_type] = []
            tasks_by_type[task_type].append(task)
        
        # Prepare worker capacity tracking
        worker_capacity = {worker_id: self.max_tasks_per_worker for worker_id in available_workers}
        
        # Track assigned tasks for batch assignment
        batch_assignments = {}  # worker_id -> [task_ids]
        
        # First pass: try to assign tasks in batches to suitable workers
        for task_type, tasks in tasks_by_type.items():
            # Group tasks by model for more efficient batching
            tasks_by_model = {}
            for task in tasks:
                model_name = task.get("config", {}).get("model", "unknown")
                if model_name not in tasks_by_model:
                    tasks_by_model[model_name] = []
                tasks_by_model[model_name].append(task)
            
            # Assign tasks in model-specific batches
            for model_name, model_tasks in tasks_by_model.items():
                # Find best worker for this batch of tasks
                worker_scores = []
                
                for worker_id, worker in available_workers.items():
                    # Skip if worker has no remaining capacity
                    if worker_capacity[worker_id] <= 0:
                        continue
                    
                    # Check if worker can handle these tasks
                    if not await self._can_worker_handle_task(worker_id, model_tasks[0]):
                        continue
                    
                    # Calculate batch size (minimum of worker capacity and task count)
                    batch_size = min(worker_capacity[worker_id], len(model_tasks))
                    
                    # Calculate score based on hardware match, current load, and specialization
                    score = 10.0
                    
                    # Add score for hardware match
                    score += 5.0
                    
                    # Prefer workers that are specialized for this model type
                    if self.task_scheduler and self.task_scheduler.worker_specialization.get(worker_id, {}).get(model_name, 0) > 0.7:
                        score += 3.0
                    
                    # Add score for batch efficiency (higher for larger batches)
                    score += batch_size * 0.5
                    
                    worker_scores.append((worker_id, score, batch_size))
                
                # Sort workers by score
                worker_scores.sort(key=lambda x: x[1], reverse=True)
                
                # Assign tasks to the best worker
                if worker_scores:
                    best_worker_id, _, batch_size = worker_scores[0]
                    
                    # Initialize batch if needed
                    if best_worker_id not in batch_assignments:
                        batch_assignments[best_worker_id] = []
                    
                    # Add tasks to batch (limited by batch size)
                    for i in range(min(batch_size, len(model_tasks))):
                        batch_assignments[best_worker_id].append(model_tasks[i]["task_id"])
                        worker_capacity[best_worker_id] -= 1
                        
                        # Stop if worker reaches capacity
                        if worker_capacity[best_worker_id] <= 0:
                            break
        
        # Second pass: assign any remaining tasks individually
        remaining_tasks = []
        for task in pending_tasks:
            task_id = task["task_id"]
            # Check if this task is not already in a batch
            if not any(task_id in batch for batch in batch_assignments.values()):
                remaining_tasks.append(task)
        
        # Sort workers by remaining capacity
        workers_by_capacity = sorted(
            [(worker_id, worker_capacity[worker_id]) for worker_id in available_workers],
            key=lambda x: x[1],
            reverse=True
        )
        
        # Assign remaining tasks
        for task in remaining_tasks:
            task_id = task["task_id"]
            
            # Find suitable worker with capacity
            suitable_worker = None
            for worker_id, capacity in workers_by_capacity:
                if capacity > 0 and await self._can_worker_handle_task(worker_id, task):
                    suitable_worker = worker_id
                    break
            
            # Skip if no suitable worker found
            if not suitable_worker:
                continue
            
            # Add to batch
            if suitable_worker not in batch_assignments:
                batch_assignments[suitable_worker] = []
            batch_assignments[suitable_worker].append(task_id)
            
            # Update worker capacity
            for i, (worker_id, capacity) in enumerate(workers_by_capacity):
                if worker_id == suitable_worker:
                    workers_by_capacity[i] = (worker_id, capacity - 1)
                    break
            
            # Re-sort workers by capacity
            workers_by_capacity.sort(key=lambda x: x[1], reverse=True)
        
        # Execute batch assignments
        logger.info(f"Batch assignments prepared: {len(batch_assignments)} workers, {sum(len(batch) for batch in batch_assignments.values())} tasks")
        
        # Process batch assignments
        for worker_id, task_ids in batch_assignments.items():
            if not task_ids:
                continue
                
            # For single task, use normal assignment
            if len(task_ids) == 1:
                task_id = task_ids[0]
                success = await self._assign_task_to_worker(task_id, worker_id)
                
                if success:
                    tasks_assigned += 1
                    self.pending_tasks.remove(task_id)
                    self.running_tasks[task_id] = worker_id
            else:
                # For multiple tasks, use batch assignment
                success = await self._assign_task_batch_to_worker(task_ids, worker_id)
                
                if success:
                    tasks_assigned += len(task_ids)
                    for task_id in task_ids:
                        self.pending_tasks.remove(task_id)
                        self.running_tasks[task_id] = worker_id
        
        logger.info(f"Assigned {tasks_assigned} tasks to {len(batch_assignments)} workers")
        return tasks_assigned
        
    async def _assign_task_batch_to_worker(self, task_ids: List[str], worker_id: str) -> bool:
        """
        Assign a batch of tasks to a worker.
        
        Args:
            task_ids: List of task IDs to assign
            worker_id: Worker ID
            
        Returns:
            True if tasks were assigned successfully, False otherwise
        """
        # Check if worker exists and is connected
        if worker_id not in self.workers or worker_id not in self.worker_connections:
            logger.error(f"Cannot assign tasks to worker {worker_id}: Worker not found or not connected")
            return False
        
        # Get tasks
        tasks = [self.tasks[task_id] for task_id in task_ids]
        
        try:
            # Update task statuses
            now = datetime.now()
            for task in tasks:
                task["status"] = "running"
                task["started"] = now.isoformat()
                task["worker_id"] = worker_id
                task["attempts"] = task.get("attempts", 0) + 1
            
            # Prepare batch execution message
            batch_data = {
                "type": "execute_task_batch",
                "tasks": [
                    {
                        "task_id": task["task_id"],
                        "task_type": task["type"],
                        "config": task.get("config", {})
                    }
                    for task in tasks
                ]
            }
            
            # Send batch to worker
            await self.worker_connections[worker_id].send_json(batch_data)
            
            # Update database (in a single transaction if possible)
            if self.db:
                try:
                    # Start transaction
                    self.db.execute("BEGIN TRANSACTION;")
                    
                    # Update each task
                    for task in tasks:
                        self.db.execute(
                            """
                            UPDATE distributed_tasks
                            SET status = 'running', start_time = ?, worker_id = ?, attempts = ?
                            WHERE task_id = ?
                            """,
                            (now, worker_id, task["attempts"], task["task_id"])
                        )
                    
                    # Commit transaction
                    self.db.execute("COMMIT;")
                except Exception as e:
                    # Rollback on error
                    self.db.execute("ROLLBACK;")
                    logger.error(f"Error updating tasks in database: {e}")
            
            logger.info(f"Assigned batch of {len(task_ids)} tasks to worker {worker_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error assigning tasks to worker {worker_id}: {e}")
            return False
    
    async def _can_worker_handle_task(self, worker_id: str, task: Dict[str, Any]) -> bool:
        """
        Check if a worker can handle a task.
        
        Args:
            worker_id: Worker ID
            task: Task information
            
        Returns:
            True if worker can handle the task, False otherwise
        """
        # Skip if worker doesn't exist
        if worker_id not in self.workers:
            return False
        
        worker = self.workers[worker_id]
        
        # Skip inactive workers
        if worker.get("status") != "active":
            return False
        
        # Check task requirements against worker capabilities
        requirements = task.get("requirements", {})
        capabilities = worker.get("capabilities", {})
        
        # Check required hardware
        required_hardware = requirements.get("hardware", [])
        if required_hardware:
            worker_hardware = capabilities.get("hardware", [])
            if not all(hw in worker_hardware for hw in required_hardware):
                return False
        
        # Check memory requirements
        min_memory_gb = requirements.get("min_memory_gb", 0)
        if min_memory_gb > 0:
            worker_memory_gb = capabilities.get("memory", {}).get("total_gb", 0)
            if worker_memory_gb < min_memory_gb:
                return False
        
        # Check CUDA compute capability
        min_cuda_compute = requirements.get("min_cuda_compute", 0)
        if min_cuda_compute > 0:
            worker_cuda_compute = float(capabilities.get("gpu", {}).get("cuda_compute", 0))
            if worker_cuda_compute < min_cuda_compute:
                return False
        
        return True
    
    async def _assign_task_to_worker(self, task_id: str, worker_id: str) -> bool:
        """
        Assign a task to a worker.
        
        Args:
            task_id: Task ID
            worker_id: Worker ID
            
        Returns:
            True if task was assigned successfully, False otherwise
        """
        # Check if task exists
        if task_id not in self.tasks:
            logger.error(f"Cannot assign task {task_id}: Task not found")
            return False
        
        # Check if worker exists and is connected
        if worker_id not in self.workers or worker_id not in self.worker_connections:
            logger.error(f"Cannot assign task {task_id}: Worker {worker_id} not found or not connected")
            return False
        
        task = self.tasks[task_id]
        
        try:
            # Update task status
            now = datetime.now()
            task["status"] = "running"
            task["started"] = now.isoformat()
            task["worker_id"] = worker_id
            task["attempts"] = task.get("attempts", 0) + 1
            
            # Send task to worker
            await self.worker_connections[worker_id].send_json({
                "type": "execute_task",
                "task_id": task_id,
                "task_type": task["type"],
                "config": task.get("config", {})
            })
            
            # Update database
            if self.db:
                try:
                    self.db.execute(
                        """
                        UPDATE distributed_tasks
                        SET status = 'running', start_time = ?, worker_id = ?, attempts = ?
                        WHERE task_id = ?
                        """,
                        (now, worker_id, task["attempts"], task_id)
                    )
                except Exception as e:
                    logger.error(f"Error updating task in database: {e}")
            
            logger.info(f"Task {task_id} assigned to worker {worker_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error assigning task {task_id} to worker {worker_id}: {e}")
            return False
    
    async def _handle_task_completed(self, task_id: str, worker_id: str, result: Dict[str, Any], execution_time: float):
        """
        Handle completion of a task.
        
        Args:
            task_id: Task ID
            worker_id: Worker ID
            result: Task result
            execution_time: Execution time in seconds
        """
        # Update task status
        now = datetime.now()
        task = self.tasks[task_id]
        task["status"] = "completed"
        task["ended"] = now.isoformat()
        task["result"] = result
        task["execution_time"] = execution_time
        
        # Update worker stats
        if worker_id in self.workers:
            self.workers[worker_id]["tasks_completed"] = self.workers[worker_id].get("tasks_completed", 0) + 1
        
        # Remove from running tasks
        if task_id in self.running_tasks:
            del self.running_tasks[task_id]
        
        # Add to completed tasks
        self.completed_tasks.add(task_id)
        
        # Update database
        if self.db:
            try:
                # Update task status
                self.db.execute(
                    """
                    UPDATE distributed_tasks
                    SET status = 'completed', end_time = ?, result = ?, execution_time = ?
                    WHERE task_id = ?
                    """,
                    (now, json.dumps(result), execution_time, task_id)
                )
                
                # Record execution history
                self.db.execute(
                    """
                    INSERT INTO task_execution_history
                    (task_id, worker_id, attempt, status, start_time, end_time, execution_time_seconds, result)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        task_id,
                        worker_id,
                        task["attempts"],
                        "completed",
                        datetime.fromisoformat(task["started"]) if "started" in task else None,
                        now,
                        execution_time,
                        json.dumps(result)
                    )
                )
                
                # Update worker stats
                self.db.execute(
                    """
                    UPDATE worker_nodes
                    SET tasks_completed = tasks_completed + 1
                    WHERE worker_id = ?
                    """,
                    (worker_id,)
                )
            except Exception as e:
                logger.error(f"Error updating task completion in database: {e}")
        
        # Update task scheduler performance metrics if available
        if self.enable_advanced_scheduler and self.task_scheduler:
            self.task_scheduler.update_worker_performance(worker_id, {
                "task_id": task_id,
                "type": task["type"],
                "status": "completed",
                "execution_time_seconds": execution_time,
                "result": result
            })
        
        # Update health monitor task timeout estimates if available
        if self.health_monitor:
            self.health_monitor.update_task_timeout_estimate(task["type"], execution_time)
        
        logger.info(f"Task {task_id} completed by worker {worker_id} in {execution_time:.2f} seconds")
    
    async def _handle_task_failed(self, task_id: str, worker_id: str, error: str, execution_time: float):
        """
        Handle failure of a task.
        
        Args:
            task_id: Task ID
            worker_id: Worker ID
            error: Error message
            execution_time: Execution time in seconds
        """
        # Update task status
        now = datetime.now()
        task = self.tasks[task_id]
        task["status"] = "failed"
        task["ended"] = now.isoformat()
        task["error"] = error
        task["execution_time"] = execution_time
        
        # Update worker stats
        if worker_id in self.workers:
            self.workers[worker_id]["tasks_failed"] = self.workers[worker_id].get("tasks_failed", 0) + 1
        
        # Remove from running tasks
        if task_id in self.running_tasks:
            del self.running_tasks[task_id]
        
        # Add to failed tasks
        self.failed_tasks.add(task_id)
        
        # Update database
        if self.db:
            try:
                # Update task status
                self.db.execute(
                    """
                    UPDATE distributed_tasks
                    SET status = 'failed', end_time = ?, error = ?, execution_time = ?
                    WHERE task_id = ?
                    """,
                    (now, error, execution_time, task_id)
                )
                
                # Record execution history
                self.db.execute(
                    """
                    INSERT INTO task_execution_history
                    (task_id, worker_id, attempt, status, start_time, end_time, execution_time_seconds, error_message)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        task_id,
                        worker_id,
                        task["attempts"],
                        "failed",
                        datetime.fromisoformat(task["started"]) if "started" in task else None,
                        now,
                        execution_time,
                        error
                    )
                )
                
                # Update worker stats
                self.db.execute(
                    """
                    UPDATE worker_nodes
                    SET tasks_failed = tasks_failed + 1
                    WHERE worker_id = ?
                    """,
                    (worker_id,)
                )
            except Exception as e:
                logger.error(f"Error updating task failure in database: {e}")
        
        # Update task scheduler performance metrics if available
        if self.enable_advanced_scheduler and self.task_scheduler:
            self.task_scheduler.update_worker_performance(worker_id, {
                "task_id": task_id,
                "type": task["type"],
                "status": "failed",
                "execution_time_seconds": execution_time,
                "error": error
            })
        
        # If health monitoring and auto recovery is enabled, consider requeing the task
        if self.health_monitor and self.health_monitor.auto_recovery:
            await self.health_monitor.requeue_task(task_id)
        
        logger.warning(f"Task {task_id} failed on worker {worker_id}: {error}")
    
    async def _update_task_status(self, task_id: str, status: str, additional_data: Dict[str, Any] = None):
        """
        Update task status.
        
        Args:
            task_id: Task ID
            status: New status
            additional_data: Additional data to update
        """
        # Check if task exists
        if task_id not in self.tasks:
            logger.warning(f"Cannot update status for task {task_id}: Task not found")
            return
        
        # Update task status
        task = self.tasks[task_id]
        old_status = task.get("status")
        task["status"] = status
        
        # Add additional data
        if additional_data:
            task.update(additional_data)
        
        # Add timestamp for status change
        now = datetime.now()
        task[f"{status}_time"] = now.isoformat()
        
        # Update status-specific task sets
        if old_status == "pending" and task_id in self.pending_tasks:
            self.pending_tasks.remove(task_id)
        
        if old_status == "running" and task_id in self.running_tasks:
            del self.running_tasks[task_id]
        
        if status == "completed":
            self.completed_tasks.add(task_id)
        elif status == "failed":
            self.failed_tasks.add(task_id)
        
        # Update database
        if self.db:
            try:
                # Prepare fields to update
                update_fields = {
                    "status": status,
                    f"{status}_time": now
                }
                
                # Add additional fields from additional_data
                if additional_data:
                    for key, value in additional_data.items():
                        if isinstance(value, (str, int, float, bool)) or value is None:
                            update_fields[key] = value
                        elif isinstance(value, (dict, list)):
                            update_fields[key] = json.dumps(value)
                
                # Build SQL query
                fields = ", ".join([f"{k} = ?" for k in update_fields.keys()])
                values = list(update_fields.values())
                values.append(task_id)
                
                sql = f"UPDATE distributed_tasks SET {fields} WHERE task_id = ?"
                
                # Execute update
                self.db.execute(sql, values)
                
            except Exception as e:
                logger.error(f"Error updating task status in database: {e}")
        
        logger.info(f"Task {task_id} status updated from {old_status} to {status}")
    
    async def start(self):
        """Start the coordinator server."""
        # Call the parent start method
        site, runner = await super().start()
        
        # Start the task scheduler if enabled
        if self.enable_advanced_scheduler and self.task_scheduler:
            logger.info("Starting advanced task scheduler")
            asyncio.create_task(self.task_scheduler.start_scheduling())
        
        # Start the health monitor if enabled
        if self.enable_health_monitor and self.health_monitor:
            logger.info("Starting health monitor")
            asyncio.create_task(self.health_monitor.start_monitoring())
        
        # Start the load balancer if enabled
        if self.enable_load_balancer and self.load_balancer:
            logger.info("Starting adaptive load balancer")
            asyncio.create_task(self.load_balancer.start_balancing())
        
        # Initialize database tables
        await self._init_database_tables()
        
        return site, runner
    
    async def _init_database_tables(self):
        """Initialize database tables."""
        if not self.db:
            return
            
        try:
            # Worker nodes table
            self.db.execute("""
            CREATE TABLE IF NOT EXISTS worker_nodes (
                worker_id VARCHAR PRIMARY KEY,
                hostname VARCHAR,
                capabilities JSON,
                status VARCHAR,
                registered TIMESTAMP,
                last_heartbeat TIMESTAMP,
                tasks_completed INTEGER DEFAULT 0,
                tasks_failed INTEGER DEFAULT 0
            )
            """)
            
            # Tasks table
            self.db.execute("""
            CREATE TABLE IF NOT EXISTS distributed_tasks (
                task_id VARCHAR PRIMARY KEY,
                type VARCHAR,
                priority INTEGER,
                config JSON,
                requirements JSON,
                status VARCHAR,
                created TIMESTAMP,
                start_time TIMESTAMP,
                end_time TIMESTAMP,
                worker_id VARCHAR,
                attempts INTEGER DEFAULT 0,
                result JSON,
                error TEXT,
                execution_time FLOAT,
                cancelled TIMESTAMP,
                cancelling TIMESTAMP
            )
            """)
            
            # Task execution history table
            self.db.execute("""
            CREATE TABLE IF NOT EXISTS task_execution_history (
                id INTEGER PRIMARY KEY,
                task_id VARCHAR,
                worker_id VARCHAR,
                attempt INTEGER,
                status VARCHAR,
                start_time TIMESTAMP,
                end_time TIMESTAMP,
                execution_time_seconds FLOAT,
                result JSON,
                error_message TEXT,
                hardware_metrics JSON
            )
            """)
            
            logger.info("Database tables initialized")
        except Exception as e:
            logger.error(f"Error initializing database tables: {e}")
            
    async def _check_and_create_db_schema(self):
        """Check if database schema exists and create it if not."""
        if not self.db:
            return
            
        try:
            # Check if worker_nodes table exists
            result = self.db.execute("""
            SELECT name FROM sqlite_master WHERE type='table' AND name='worker_nodes'
            """).fetchone()
            
            if not result:
                await self._init_database_tables()
            
        except Exception as e:
            logger.error(f"Error checking database schema: {e}")
            
            # Try to create tables anyway
            await self._init_database_tables()

async def main():
    """Main entry point for the coordinator server."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Distributed Testing Framework Coordinator")
    
    # Server configuration
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind the server to")
    parser.add_argument("--db-path", default="./benchmark_db.duckdb", help="Path to the DuckDB database")
    
    # Feature configuration
    parser.add_argument("--security-config", help="Path to security configuration file")
    parser.add_argument("--disable-advanced-scheduler", action="store_true", help="Disable advanced task scheduler")
    parser.add_argument("--disable-health-monitor", action="store_true", help="Disable health monitoring")
    parser.add_argument("--disable-load-balancer", action="store_true", help="Disable adaptive load balancer")
    parser.add_argument("--disable-auto-recovery", action="store_true", help="Disable auto recovery")
    parser.add_argument("--disable-enhanced-error-handling", action="store_true", help="Disable enhanced error handling with performance tracking")
    
    # Redundancy configuration
    parser.add_argument("--disable-redundancy", action="store_true", help="Disable coordinator redundancy")
    parser.add_argument("--cluster-nodes", help="Comma-separated list of coordinator nodes in the cluster")
    parser.add_argument("--node-id", help="Unique identifier for this node")
    
    # Plugin configuration
    parser.add_argument("--disable-plugins", action="store_true", help="Disable plugin system")
    parser.add_argument("--plugin-dirs", help="Comma-separated list of plugin directories")
    parser.add_argument("--list-plugins", action="store_true", help="List available plugins and exit")
    parser.add_argument("--enable-plugin", action="append", help="Enable specific plugin(s) by name")
    parser.add_argument("--disable-plugin", action="append", help="Disable specific plugin(s) by name")
    
    # API key generation
    parser.add_argument("--generate-api-key", help="Generate an API key for a worker")
    parser.add_argument("--generate-admin-key", action="store_true", help="Generate an admin API key")
    
    args = parser.parse_args()
    
    # Handle API key generation
    if args.generate_api_key or args.generate_admin_key:
        # Import security module
        try:
            from security import SecurityManager
            
            # Initialize security manager
            security_manager = SecurityManager(db_path=args.db_path, config_path=args.security_config)
            
            if args.generate_api_key:
                key_info = security_manager.generate_api_key(args.generate_api_key, "worker")
                print(f"Generated worker API key: {key_info['api_key']}")
                print(f"Key ID: {key_info['key_id']}")
                print(f"Name: {key_info['name']}")
                print(f"Role: {key_info['role']}")
            
            if args.generate_admin_key:
                key_info = security_manager.generate_api_key("admin", "admin")
                print(f"Generated admin API key: {key_info['api_key']}")
                print(f"Key ID: {key_info['key_id']}")
                print(f"Name: {key_info['name']}")
                print(f"Role: {key_info['role']}")
            
            return 0
        except Exception as e:
            print(f"Error generating API key: {e}")
            return 1
    
    # Parse cluster nodes
    cluster_nodes = None
    if args.cluster_nodes:
        cluster_nodes = args.cluster_nodes.split(",")
    
    # Parse plugin directories
    plugin_dirs = None
    if args.plugin_dirs:
        plugin_dirs = args.plugin_dirs.split(",")
    
    # Handle list plugins option
    if args.list_plugins:
        # Create temporary plugin manager
        plugin_manager = PluginManager(coordinator=None, plugin_dirs=plugin_dirs or ["plugins"])
        
        # Discover plugins
        discovered_plugins = asyncio.run(plugin_manager.discover_plugins())
        
        print("Discovered plugins:")
        if discovered_plugins:
            for plugin in discovered_plugins:
                print(f"  - {plugin}")
        else:
            print("  No plugins found")
            
        return 0
    
    # Create coordinator
    coordinator = DistributedTestingCoordinator(
        db_path=args.db_path,
        host=args.host,
        port=args.port,
        security_config=args.security_config,
        enable_advanced_scheduler=not args.disable_advanced_scheduler,
        enable_health_monitor=not args.disable_health_monitor,
        enable_load_balancer=not args.disable_load_balancer,
        enable_auto_recovery=not args.disable_auto_recovery,
        enable_redundancy=not args.disable_redundancy,
        enable_plugins=not args.disable_plugins,
        plugin_dirs=plugin_dirs,
        cluster_nodes=cluster_nodes,
        node_id=args.node_id,
        enable_enhanced_error_handling=not args.disable_enhanced_error_handling
    )
    
    # Setup signal handlers for graceful shutdown
    loop = asyncio.get_event_loop()
    
    async def shutdown(signal=None):
        """Shutdown the coordinator server."""
        if signal:
            logger.info(f"Received exit signal {signal.name}...")
        
        logger.info("Shutting down coordinator...")
        await coordinator.stop()
        
        tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        
        for task in tasks:
            task.cancel()
            
        logger.info(f"Cancelling {len(tasks)} outstanding tasks...")
        await asyncio.gather(*tasks, return_exceptions=True)
        
        loop.stop()
        logger.info("Shutdown complete")
    
    # Register signal handlers
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda s=sig: asyncio.create_task(shutdown(s)))
    
    # Start coordinator
    try:
        site, runner = await coordinator.start()
        
        # Keep running until stopped
        await asyncio.Event().wait()
    except Exception as e:
        logger.error(f"Error starting coordinator: {e}")
        await shutdown()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))