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
                 cluster_nodes: List[str] = None, node_id: str = None):
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
            cluster_nodes: List of coordinator nodes in the cluster (for redundancy)
            node_id: Unique identifier for this node (for redundancy)
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
        
        # Features enabled
        self.enable_advanced_scheduler = enable_advanced_scheduler
        self.enable_health_monitor = enable_health_monitor
        self.enable_load_balancer = enable_load_balancer
        self.enable_auto_recovery = enable_auto_recovery
        self.enable_redundancy = enable_redundancy and REDUNDANCY_AVAILABLE
        
        # Redundancy configuration
        self.cluster_nodes = cluster_nodes or [f"http://{host}:{port}"]
        self.node_id = node_id or f"node-{uuid.uuid4().hex[:8]}"
        
        # Initialize security manager
        self._init_security_manager()
        
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
    
    async def start(self):
        """Start the coordinator server."""
        # Create web application
        self.app = web.Application(middlewares=[auth_middleware])
        
        # Setup routes
        self._setup_routes()
        
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
        
        return site, runner
    
    async def stop(self):
        """Stop the coordinator server."""
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
        return web.json_response({
            "status": "ok",
            "workers": len(getattr(self, "workers", {})),
            "tasks": len(getattr(self, "tasks", {})),
            "redundancy_enabled": self.enable_redundancy,
            "redundancy_role": self.redundancy_manager.current_role.value if self.redundancy_manager else "none"
        })
    
    # Worker API handlers
    async def handle_worker_register(self, request):
        """Handle worker registration."""
        return web.json_response({"status": "not_implemented"}, status=501)
    
    async def handle_list_workers(self, request):
        """Handle worker listing."""
        return web.json_response({"status": "not_implemented"}, status=501)
    
    async def handle_get_worker(self, request):
        """Handle get worker details."""
        return web.json_response({"status": "not_implemented"}, status=501)
    
    # Task API handlers
    async def handle_create_task(self, request):
        """Handle task creation."""
        return web.json_response({"status": "not_implemented"}, status=501)
    
    async def handle_list_tasks(self, request):
        """Handle task listing."""
        return web.json_response({"status": "not_implemented"}, status=501)
    
    async def handle_get_task(self, request):
        """Handle get task details."""
        return web.json_response({"status": "not_implemented"}, status=501)
    
    async def handle_cancel_task(self, request):
        """Handle task cancellation."""
        return web.json_response({"status": "not_implemented"}, status=501)
    
    # WebSocket handler
    async def handle_websocket(self, request):
        """Handle WebSocket connection from worker."""
        return web.Response(text="WebSocket not implemented", status=501)

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
    
    # Redundancy configuration
    parser.add_argument("--disable-redundancy", action="store_true", help="Disable coordinator redundancy")
    parser.add_argument("--cluster-nodes", help="Comma-separated list of coordinator nodes in the cluster")
    parser.add_argument("--node-id", help="Unique identifier for this node")
    
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
        cluster_nodes=cluster_nodes,
        node_id=args.node_id
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