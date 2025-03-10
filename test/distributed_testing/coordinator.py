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

# Configure logging
    logging.basicConfig())))))))))))))))))))
    level=logging.INFO,
    format='%())))))))))))))))))))asctime)s - %())))))))))))))))))))name)s - %())))))))))))))))))))levelname)s - %())))))))))))))))))))message)s',
    handlers=[]],,
    logging.StreamHandler())))))))))))))))))))),
    logging.FileHandler())))))))))))))))))))"coordinator.log")
    ]
    )
    logger = logging.getLogger())))))))))))))))))))__name__)

class DistributedTestingCoordinator:
    """Coordinator server for distributed testing framework."""
    
    def __init__())))))))))))))))))))self, db_path: str, host: str = "0.0.0.0", port: int = 8080,
    security_config: str = None, enable_advanced_scheduler: bool = True,
                 enable_health_monitor: bool = True, enable_load_balancer: bool = True):
                     """
                     Initialize the coordinator.
        
        Args:
            db_path: Path to the DuckDB database
            host: Host to bind the server to
            port: Port to bind the server to
            security_config: Path to security configuration file
            enable_advanced_scheduler: Whether to enable the advanced task scheduler
            enable_health_monitor: Whether to enable the health monitor
            enable_load_balancer: Whether to enable the adaptive load balancer
            """
            self.db_path = db_path
            self.host = host
            self.port = port
            self.app = web.Application())))))))))))))))))))middlewares=[]],,auth_middleware])
        
        # Initialize security manager
        if security_config and os.path.exists())))))))))))))))))))security_config):
            self.security_manager = SecurityManager.load_config())))))))))))))))))))security_config)
            logger.info())))))))))))))))))))f"Loaded security configuration from {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}security_config}")
        else:
            self.security_manager = SecurityManager()))))))))))))))))))))
            logger.info())))))))))))))))))))"Initialized new security manager")
            
            # Generate default API key for testing
            self.security_manager.generate_api_key())))))))))))))))))))"default-admin", []],,"admin"])
            logger.info())))))))))))))))))))"Generated default admin API key")
            
            # Save security configuration if path provided:
            if security_config:
                self.security_manager.save_config())))))))))))))))))))security_config)
                logger.info())))))))))))))))))))f"Saved security configuration to {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}security_config}")
        
        # Store security manager in app context
                self.app[]],,"security_manager"] = self.security_manager
        
                self.setup_routes()))))))))))))))))))))
        
        # Worker state
                self.workers: Dict[]],,str, Dict[]],,str, Any]] = {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                self.worker_connections: Dict[]],,str, web.WebSocketResponse] = {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
        
        # Task state
                self.tasks: Dict[]],,str, Dict[]],,str, Any]] = {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                self.pending_tasks: Set[]],,str] = set()))))))))))))))))))))
                self.running_tasks: Dict[]],,str, str] = {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}  # task_id -> worker_id
                self.completed_tasks: Set[]],,str] = set()))))))))))))))))))))
                self.failed_tasks: Set[]],,str] = set()))))))))))))))))))))
        
        # Database connection
                self._init_database()))))))))))))))))))))
        
        # Signal handlers
                self._setup_signal_handlers()))))))))))))))))))))
        
        # Initialize health monitor if enabled::
        self.enable_health_monitor = enable_health_monitor:
        if enable_health_monitor:
            self.health_monitor = HealthMonitor())))))))))))))))))))self)
            logger.info())))))))))))))))))))"Health monitor initialized")
        else:
            self.health_monitor = None
            
        # Initialize task scheduler if enabled::
        self.enable_advanced_scheduler = enable_advanced_scheduler:
        if enable_advanced_scheduler:
            self.task_scheduler = TaskScheduler())))))))))))))))))))self)
            logger.info())))))))))))))))))))"Advanced task scheduler initialized")
        else:
            self.task_scheduler = None
            
        # Initialize load balancer if enabled::
        self.enable_load_balancer = enable_load_balancer:
        if enable_load_balancer:
            self.load_balancer = AdaptiveLoadBalancer())))))))))))))))))))self)
            logger.info())))))))))))))))))))"Adaptive load balancer initialized")
        else:
            self.load_balancer = None
        
            logger.info())))))))))))))))))))f"Coordinator initialized with database at {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}db_path}")
    
    def _init_database())))))))))))))))))))self):
        """Initialize the database connection and schema."""
        try:
            # Connect to database
            self.db = duckdb.connect())))))))))))))))))))self.db_path)
            
            # Create schema if it doesn't exist
            self._create_schema()))))))))))))))))))))
            
            logger.info())))))))))))))))))))f"Database connection established to {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}self.db_path}"):
        except Exception as e:
            logger.error())))))))))))))))))))f"Failed to initialize database: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}str())))))))))))))))))))e)}")
                raise
    
    def _create_schema())))))))))))))))))))self):
        """Create the database schema if it doesn't exist.""":
        try:
            # Worker nodes table
            self.db.execute())))))))))))))))))))"""
            CREATE TABLE IF NOT EXISTS worker_nodes ())))))))))))))))))))
            worker_id VARCHAR PRIMARY KEY,
            hostname VARCHAR,
            registration_time TIMESTAMP,
            last_heartbeat TIMESTAMP,
            status VARCHAR,
            capabilities JSON,
            hardware_metrics JSON,
            tags JSON
            )
            """)
            
            # Distributed tasks table
            self.db.execute())))))))))))))))))))"""
            CREATE TABLE IF NOT EXISTS distributed_tasks ())))))))))))))))))))
            task_id VARCHAR PRIMARY KEY,
            type VARCHAR,
            priority INTEGER,
            status VARCHAR,
            create_time TIMESTAMP,
            start_time TIMESTAMP,
            end_time TIMESTAMP,
            worker_id VARCHAR,
            attempts INTEGER,
            config JSON,
            requirements JSON
            )
            """)
            
            # Task execution history table
            self.db.execute())))))))))))))))))))"""
            CREATE TABLE IF NOT EXISTS task_execution_history ())))))))))))))))))))
            id INTEGER PRIMARY KEY,
            task_id VARCHAR,
            worker_id VARCHAR,
            attempt INTEGER,
            status VARCHAR,
            start_time TIMESTAMP,
            end_time TIMESTAMP,
            execution_time_seconds FLOAT,
            error_message VARCHAR,
            hardware_metrics JSON
            )
            """)
            
            logger.info())))))))))))))))))))"Database schema created successfully")
        except Exception as e:
            logger.error())))))))))))))))))))f"Failed to create database schema: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}str())))))))))))))))))))e)}")
            raise
    
    def _setup_signal_handlers())))))))))))))))))))self):
        """Set up signal handlers for graceful shutdown."""
        signals = ())))))))))))))))))))signal.SIGHUP, signal.SIGTERM, signal.SIGINT)
        for s in signals:
            signal.signal())))))))))))))))))))s, self._handle_shutdown_signal)
            logger.info())))))))))))))))))))"Signal handlers configured")
    
    def _handle_shutdown_signal())))))))))))))))))))self, sig, frame):
        """Handle shutdown signals."""
        logger.info())))))))))))))))))))f"Received shutdown signal {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}sig}, shutting down...")
        # Close database connection
        if hasattr())))))))))))))))))))self, 'db'):
            self.db.close()))))))))))))))))))))
        # Exit
            sys.exit())))))))))))))))))))0)
    
    def setup_routes())))))))))))))))))))self):
        """Set up API routes."""
        self.app.add_routes())))))))))))))))))))[]],,
        web.get())))))))))))))))))))'/', self.handle_index),
        web.get())))))))))))))))))))'/status', self.handle_status),
        web.get())))))))))))))))))))'/ws', self.handle_websocket),
        web.post())))))))))))))))))))'/api/workers/register', self.handle_worker_registration),
        web.get())))))))))))))))))))'/api/workers', self.handle_list_workers),
        web.get())))))))))))))))))))'/api/workers/{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}worker_id}', self.handle_get_worker),
        web.post())))))))))))))))))))'/api/tasks', self.handle_create_task),
        web.get())))))))))))))))))))'/api/tasks', self.handle_list_tasks),
        web.get())))))))))))))))))))'/api/tasks/{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}task_id}', self.handle_get_task),
        web.post())))))))))))))))))))'/api/tasks/{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}task_id}/cancel', self.handle_cancel_task),
        ])
        logger.info())))))))))))))))))))"API routes configured")
    
    async def handle_index())))))))))))))))))))self, request):
        """Handle the index route."""
        return web.Response())))))))))))))))))))text="Distributed Testing Framework Coordinator")
    
    async def handle_status())))))))))))))))))))self, request):
        """Handle the status route."""
        status = {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
        "workers": {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
        "total": len())))))))))))))))))))self.workers),
                "active": sum())))))))))))))))))))1 for w in self.workers.values())))))))))))))))))))) if w.get())))))))))))))))))))"status") == "active"),:
                "idle": sum())))))))))))))))))))1 for w in self.workers.values())))))))))))))))))))) if w.get())))))))))))))))))))"status") == "idle"),:
                    "offline": sum())))))))))))))))))))1 for w in self.workers.values())))))))))))))))))))) if w.get())))))))))))))))))))"status") == "offline"),
            },:
                "tasks": {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                "total": len())))))))))))))))))))self.tasks),
                "pending": len())))))))))))))))))))self.pending_tasks),
                "running": len())))))))))))))))))))self.running_tasks),
                "completed": len())))))))))))))))))))self.completed_tasks),
                "failed": len())))))))))))))))))))self.failed_tasks),
                },
                "uptime": 0,  # TODO: Implement uptime tracking
                "version": "0.1.0",
                }
                    return web.json_response())))))))))))))))))))status)
    
    async def handle_websocket())))))))))))))))))))self, request):
        """Handle WebSocket connections."""
        ws = web.WebSocketResponse()))))))))))))))))))))
        await ws.prepare())))))))))))))))))))request)
        
        worker_id = None
        authenticated = False
        
        try:
            # First message must be authentication
            auth_msg = await ws.receive_json()))))))))))))))))))))
            
            if auth_msg.get())))))))))))))))))))"type") != "auth":
                logger.warning())))))))))))))))))))"First message from worker was not authentication")
                await ws.send_json()))))))))))))))))))){}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}"type": "error", "message": "Authentication required"})
            return ws
            
            # Check authentication type
            auth_type = auth_msg.get())))))))))))))))))))"auth_type")
            
            if auth_type == "api_key":
                # API key authentication
                api_key = auth_msg.get())))))))))))))))))))"api_key")
                
                if not api_key:
                    await ws.send_json()))))))))))))))))))){}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}"type": "error", "message": "API key required"})
                return ws
                
                # Validate API key
                is_valid, roles = self.security_manager.validate_api_key())))))))))))))))))))api_key)
                
                if not is_valid or not self.security_manager.has_required_role())))))))))))))))))))roles):
                    await ws.send_json()))))))))))))))))))){}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}"type": "error", "message": "Invalid API key or insufficient permissions"})
                return ws
                
                # Generate worker token
                worker_id = auth_msg.get())))))))))))))))))))"worker_id", str())))))))))))))))))))uuid.uuid4()))))))))))))))))))))))
                token = self.security_manager.generate_worker_token())))))))))))))))))))worker_id, api_key)
                
                if not token:
                    await ws.send_json()))))))))))))))))))){}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}"type": "error", "message": "Failed to generate worker token"})
                return ws
                
                # Send token to worker
                await ws.send_json()))))))))))))))))))){}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                "type": "auth_response",
                "status": "success",
                "token": token,
                "worker_id": worker_id
                })
                
                authenticated = True
                
            elif auth_type == "token":
                # Token authentication
                token = auth_msg.get())))))))))))))))))))"token")
                
                if not token:
                    await ws.send_json()))))))))))))))))))){}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}"type": "error", "message": "Token required"})
                return ws
                
                # Validate token
                is_valid, token_worker_id, roles = self.security_manager.validate_worker_token())))))))))))))))))))token)
                
                if not is_valid:
                    await ws.send_json()))))))))))))))))))){}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}"type": "error", "message": "Invalid token"})
                return ws
                
                worker_id = token_worker_id
                
                # Send success response
                await ws.send_json()))))))))))))))))))){}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                "type": "auth_response",
                "status": "success",
                "worker_id": worker_id
                })
                
                authenticated = True
                
            else:
                # Unknown authentication type
                await ws.send_json()))))))))))))))))))){}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}"type": "error", "message": "Invalid authentication type"})
                return ws
            
            if not authenticated:
                return ws
            
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        data = json.loads())))))))))))))))))))msg.data)
                        
                        # Verify message signature
                        if not self.security_manager.verify_message())))))))))))))))))))data.copy()))))))))))))))))))))):
                            await ws.send_json()))))))))))))))))))){}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}"type": "error", "message": "Invalid message signature"})
                        continue
                        
                        msg_type = data.get())))))))))))))))))))"type")
                        
                        if msg_type == "register":
                            # Worker registration
                            worker_id = data.get())))))))))))))))))))"worker_id")
                            if not worker_id:
                                worker_id = str())))))))))))))))))))uuid.uuid4())))))))))))))))))))))
                                response = {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}"type": "register_response", "worker_id": worker_id}
                                await ws.send_json())))))))))))))))))))self.security_manager.sign_message())))))))))))))))))))response))
                            
                            # Store worker connection
                                self.worker_connections[]],,worker_id] = ws
                            
                            # Update worker status
                                self.workers[]],,worker_id] = {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                                "status": "active",
                                "last_heartbeat": datetime.now())))))))))))))))))))).isoformat())))))))))))))))))))),
                                "capabilities": data.get())))))))))))))))))))"capabilities", {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}),
                                "hostname": data.get())))))))))))))))))))"hostname", "unknown"),
                                }
                            
                                logger.info())))))))))))))))))))f"Worker {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}worker_id} registered")
                            
                        elif msg_type == "heartbeat":
                            # Worker heartbeat
                            worker_id = data.get())))))))))))))))))))"worker_id")
                            if not worker_id or worker_id not in self.workers:
                                await ws.send_json())))))))))))))))))))self.security_manager.sign_message()))))))))))))))))))){}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                                "type": "error",
                                "message": "Unknown worker"
                                }))
                            continue
                            
                            # Update worker status
                            self.workers[]],,worker_id][]],,"last_heartbeat"] = datetime.now())))))))))))))))))))).isoformat()))))))))))))))))))))
                            self.workers[]],,worker_id][]],,"status"] = "active"
                            
                            # Store hardware metrics if provided:
                            if "hardware_metrics" in data:
                                self._update_worker_hardware_metrics())))))))))))))))))))worker_id, data[]],,"hardware_metrics"])
                            
                            # Check for task updates
                            if "task_status" in data:
                                task_id = data[]],,"task_status"].get())))))))))))))))))))"task_id")
                                status = data[]],,"task_status"].get())))))))))))))))))))"status")
                                
                                if task_id and status:
                                    await self._update_task_status())))))))))))))))))))task_id, status, data[]],,"task_status"])
                            
                            # Check for task cancellation for migration
                            if "task_cancelled" in data:
                                task_id = data[]],,"task_cancelled"].get())))))))))))))))))))"task_id")
                                reason = data[]],,"task_cancelled"].get())))))))))))))))))))"reason")
                                
                                if task_id and reason == "migration" and self.enable_load_balancer and self.load_balancer:
                                    # Handle task cancellation for migration
                                    await self.load_balancer.handle_task_cancelled_for_migration())))))))))))))))))))task_id, worker_id)
                            
                                    await ws.send_json())))))))))))))))))))self.security_manager.sign_message()))))))))))))))))))){}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                                    "type": "heartbeat_response"
                                    }))
                            
                        elif msg_type == "task_result":
                            # Task result
                            worker_id = data.get())))))))))))))))))))"worker_id")
                            task_id = data.get())))))))))))))))))))"task_id")
                            
                            if not worker_id or worker_id not in self.workers:
                                await ws.send_json())))))))))))))))))))self.security_manager.sign_message()))))))))))))))))))){}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                                "type": "error",
                                "message": "Unknown worker"
                                }))
                            continue
                            
                            if not task_id or task_id not in self.tasks:
                                await ws.send_json())))))))))))))))))))self.security_manager.sign_message()))))))))))))))))))){}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                                "type": "error",
                                "message": "Unknown task"
                                }))
                            continue
                            
                            # Process task result
                            await self._process_task_result())))))))))))))))))))task_id, worker_id, data)
                            
                            await ws.send_json())))))))))))))))))))self.security_manager.sign_message()))))))))))))))))))){}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                            "type": "task_result_response",
                            "task_id": task_id
                            }))
                            
                        else:
                            # Unknown message type
                            await ws.send_json())))))))))))))))))))self.security_manager.sign_message()))))))))))))))))))){}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                            "type": "error",
                            "message": "Unknown message type"
                            }))
                    
                    except json.JSONDecodeError:
                        await ws.send_json())))))))))))))))))))self.security_manager.sign_message()))))))))))))))))))){}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                        "type": "error",
                        "message": "Invalid JSON"
                        }))
                    except Exception as e:
                        logger.error())))))))))))))))))))f"Error processing WebSocket message: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}str())))))))))))))))))))e)}")
                        await ws.send_json())))))))))))))))))))self.security_manager.sign_message()))))))))))))))))))){}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                        "type": "error",
                        "message": f"Error: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}str())))))))))))))))))))e)}"
                        }))
                
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error())))))))))))))))))))f"WebSocket connection closed with exception {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}ws.exception()))))))))))))))))))))}")
        
        finally:
            # Clean up when connection is closed
            if worker_id and worker_id in self.worker_connections:
                del self.worker_connections[]],,worker_id]
                
                if worker_id in self.workers:
                    self.workers[]],,worker_id][]],,"status"] = "offline"
                    logger.info())))))))))))))))))))f"Worker {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}worker_id} disconnected")
            
                    await ws.close()))))))))))))))))))))
        
                return ws
    
    async def handle_worker_registration())))))))))))))))))))self, request):
        """Handle worker registration via HTTP API."""
        try:
            data = await request.json()))))))))))))))))))))
            
            hostname = data.get())))))))))))))))))))"hostname")
            capabilities = data.get())))))))))))))))))))"capabilities", {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}})
            worker_id = data.get())))))))))))))))))))"worker_id", str())))))))))))))))))))uuid.uuid4()))))))))))))))))))))))
            
            # Register worker in database
            self.db.execute())))))))))))))))))))
            """
            INSERT INTO worker_nodes ())))))))))))))))))))
            worker_id, hostname, registration_time, last_heartbeat,
            status, capabilities, hardware_metrics, tags
            ) VALUES ())))))))))))))))))))?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT ())))))))))))))))))))worker_id) DO UPDATE SET
            hostname = excluded.hostname,
            last_heartbeat = excluded.last_heartbeat,
            status = excluded.status,
            capabilities = excluded.capabilities
            """,
            ())))))))))))))))))))
            worker_id, hostname, datetime.now())))))))))))))))))))), datetime.now())))))))))))))))))))),
            "active", json.dumps())))))))))))))))))))capabilities), json.dumps()))))))))))))))))))){}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}), json.dumps()))))))))))))))))))){}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}})
            )
            )
            
            # Update in-memory worker state
            self.workers[]],,worker_id] = {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "worker_id": worker_id,
            "hostname": hostname,
            "status": "active",
            "last_heartbeat": datetime.now())))))))))))))))))))).isoformat())))))))))))))))))))),
            "capabilities": capabilities,
            }
            
            logger.info())))))))))))))))))))f"Worker {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}worker_id} registered via HTTP API")
            
        return web.json_response()))))))))))))))))))){}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
        "worker_id": worker_id,
        "registered": True,
        "message": "Worker registered successfully"
        })
            
        except Exception as e:
            logger.error())))))))))))))))))))f"Error registering worker: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}str())))))))))))))))))))e)}")
        return web.json_response()))))))))))))))))))){}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
        "error": "Failed to register worker",
        "message": str())))))))))))))))))))e)
        }, status=500)
    
    async def handle_list_workers())))))))))))))))))))self, request):
        """Handle listing all workers."""
        try:
            workers = []],,]
            for worker_id, worker in self.workers.items())))))))))))))))))))):
                workers.append()))))))))))))))))))){}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                "worker_id": worker_id,
                "hostname": worker.get())))))))))))))))))))"hostname", "unknown"),
                "status": worker.get())))))))))))))))))))"status", "unknown"),
                "last_heartbeat": worker.get())))))))))))))))))))"last_heartbeat"),
                "capabilities_summary": self._summarize_capabilities())))))))))))))))))))worker.get())))))))))))))))))))"capabilities", {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}})),
                })
            
            return web.json_response()))))))))))))))))))){}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}"workers": workers})
            
        except Exception as e:
            logger.error())))))))))))))))))))f"Error listing workers: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}str())))))))))))))))))))e)}")
            return web.json_response()))))))))))))))))))){}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "error": "Failed to list workers",
            "message": str())))))))))))))))))))e)
            }, status=500)
    
    def _summarize_capabilities())))))))))))))))))))self, capabilities: Dict[]],,str, Any]) -> Dict[]],,str, Any]:
        """Generate a summary of worker capabilities."""
        summary = {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
        
        # Extract hardware type
        if "hardware" in capabilities:
            summary[]],,"hardware"] = capabilities[]],,"hardware"]
        
        # Extract GPU info if available:::
        if "gpu" in capabilities:
            summary[]],,"gpu"] = {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "name": capabilities[]],,"gpu"].get())))))))))))))))))))"name", "unknown"),
            "count": capabilities[]],,"gpu"].get())))))))))))))))))))"count", 0),
            "memory_gb": capabilities[]],,"gpu"].get())))))))))))))))))))"memory_gb", 0),
            }
        
        # Extract CPU info if available:::
        if "cpu" in capabilities:
            summary[]],,"cpu"] = {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "model": capabilities[]],,"cpu"].get())))))))))))))))))))"model", "unknown"),
            "cores": capabilities[]],,"cpu"].get())))))))))))))))))))"cores", 0),
            "threads": capabilities[]],,"cpu"].get())))))))))))))))))))"threads", 0),
            }
        
        # Extract memory info if available:::
        if "memory" in capabilities:
            summary[]],,"memory_gb"] = capabilities[]],,"memory"].get())))))))))))))))))))"total_gb", 0)
        
            return summary
    
    async def handle_get_worker())))))))))))))))))))self, request):
        """Handle getting a specific worker."""
        worker_id = request.match_info.get())))))))))))))))))))"worker_id")
        
        if not worker_id or worker_id not in self.workers:
        return web.json_response()))))))))))))))))))){}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
        "error": "Worker not found"
        }, status=404)
        
        try:
            worker = self.workers[]],,worker_id]
            
            # Get worker tasks
            worker_tasks = []],,]
            for task_id, task in self.tasks.items())))))))))))))))))))):
                if task.get())))))))))))))))))))"worker_id") == worker_id:
                    worker_tasks.append()))))))))))))))))))){}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                    "task_id": task_id,
                    "type": task.get())))))))))))))))))))"type"),
                    "status": task.get())))))))))))))))))))"status"),
                    "created": task.get())))))))))))))))))))"created"),
                    "started": task.get())))))))))))))))))))"started"),
                    "ended": task.get())))))))))))))))))))"ended"),
                    })
            
                    response = {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                    "worker_id": worker_id,
                    "hostname": worker.get())))))))))))))))))))"hostname", "unknown"),
                    "status": worker.get())))))))))))))))))))"status", "unknown"),
                    "last_heartbeat": worker.get())))))))))))))))))))"last_heartbeat"),
                    "capabilities": worker.get())))))))))))))))))))"capabilities", {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}),
                    "tasks": worker_tasks,
                    }
            
                return web.json_response())))))))))))))))))))response)
            
        except Exception as e:
            logger.error())))))))))))))))))))f"Error getting worker {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}worker_id}: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}str())))))))))))))))))))e)}")
                return web.json_response()))))))))))))))))))){}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                "error": f"Failed to get worker {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}worker_id}",
                "message": str())))))))))))))))))))e)
                }, status=500)
    
    async def handle_create_task())))))))))))))))))))self, request):
        """Handle creating a new task."""
        try:
            data = await request.json()))))))))))))))))))))
            
            task_type = data.get())))))))))))))))))))"type")
            priority = data.get())))))))))))))))))))"priority", 1)
            config = data.get())))))))))))))))))))"config", {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}})
            requirements = data.get())))))))))))))))))))"requirements", {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}})
            
            # Validate required fields
            if not task_type:
            return web.json_response()))))))))))))))))))){}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "error": "Missing required field: type"
            }, status=400)
            
            # Generate task ID
            task_id = str())))))))))))))))))))uuid.uuid4())))))))))))))))))))))
            
            # Create task in database
            self.db.execute())))))))))))))))))))
            """
            INSERT INTO distributed_tasks ())))))))))))))))))))
            task_id, type, priority, status, create_time,
            start_time, end_time, worker_id, attempts, config, requirements
            ) VALUES ())))))))))))))))))))?, ?, ?, ?, ?, NULL, NULL, NULL, ?, ?, ?)
            """,
            ())))))))))))))))))))
            task_id, task_type, priority, "pending", datetime.now())))))))))))))))))))),
            0, json.dumps())))))))))))))))))))config), json.dumps())))))))))))))))))))requirements)
            )
            )
            
            # Update in-memory task state
            self.tasks[]],,task_id] = {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "task_id": task_id,
            "type": task_type,
            "priority": priority,
            "status": "pending",
            "created": datetime.now())))))))))))))))))))).isoformat())))))))))))))))))))),
            "config": config,
            "requirements": requirements,
            "attempts": 0,
            }
            
            # Add to pending tasks
            self.pending_tasks.add())))))))))))))))))))task_id)
            
            logger.info())))))))))))))))))))f"Task {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}task_id} created")
            
            # Attempt to assign the task
            await self._assign_pending_tasks()))))))))))))))))))))
            
        return web.json_response()))))))))))))))))))){}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
        "task_id": task_id,
        "created": True,
        "message": "Task created successfully"
        })
            
        except Exception as e:
            logger.error())))))))))))))))))))f"Error creating task: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}str())))))))))))))))))))e)}")
        return web.json_response()))))))))))))))))))){}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
        "error": "Failed to create task",
        "message": str())))))))))))))))))))e)
        }, status=500)
    
    async def handle_list_tasks())))))))))))))))))))self, request):
        """Handle listing all tasks."""
        try:
            status_filter = request.query.get())))))))))))))))))))"status")
            limit = int())))))))))))))))))))request.query.get())))))))))))))))))))"limit", 100))
            
            tasks = []],,]
            for task_id, task in self.tasks.items())))))))))))))))))))):
                if status_filter and task.get())))))))))))))))))))"status") != status_filter:
                continue
                
                tasks.append()))))))))))))))))))){}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                "task_id": task_id,
                "type": task.get())))))))))))))))))))"type"),
                "status": task.get())))))))))))))))))))"status"),
                "created": task.get())))))))))))))))))))"created"),
                "started": task.get())))))))))))))))))))"started", None),
                "ended": task.get())))))))))))))))))))"ended", None),
                "worker_id": task.get())))))))))))))))))))"worker_id"),
                "priority": task.get())))))))))))))))))))"priority", 1),
                })
                
                if len())))))))))))))))))))tasks) >= limit:
                break
            
            return web.json_response()))))))))))))))))))){}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}"tasks": tasks})
            
        except Exception as e:
            logger.error())))))))))))))))))))f"Error listing tasks: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}str())))))))))))))))))))e)}")
            return web.json_response()))))))))))))))))))){}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "error": "Failed to list tasks",
            "message": str())))))))))))))))))))e)
            }, status=500)
    
    async def handle_get_task())))))))))))))))))))self, request):
        """Handle getting a specific task."""
        task_id = request.match_info.get())))))))))))))))))))"task_id")
        
        if not task_id or task_id not in self.tasks:
        return web.json_response()))))))))))))))))))){}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
        "error": "Task not found"
        }, status=404)
        
        try:
            task = self.tasks[]],,task_id]
            
            response = {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "task_id": task_id,
            "type": task.get())))))))))))))))))))"type"),
            "status": task.get())))))))))))))))))))"status"),
            "created": task.get())))))))))))))))))))"created"),
            "started": task.get())))))))))))))))))))"started", None),
            "ended": task.get())))))))))))))))))))"ended", None),
            "worker_id": task.get())))))))))))))))))))"worker_id"),
            "priority": task.get())))))))))))))))))))"priority", 1),
            "config": task.get())))))))))))))))))))"config", {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}),
            "requirements": task.get())))))))))))))))))))"requirements", {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}),
            "result": task.get())))))))))))))))))))"result"),
            "error": task.get())))))))))))))))))))"error"),
            }
            
        return web.json_response())))))))))))))))))))response)
            
        except Exception as e:
            logger.error())))))))))))))))))))f"Error getting task {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}task_id}: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}str())))))))))))))))))))e)}")
        return web.json_response()))))))))))))))))))){}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
        "error": f"Failed to get task {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}task_id}",
        "message": str())))))))))))))))))))e)
        }, status=500)
    
    async def handle_cancel_task())))))))))))))))))))self, request):
        """Handle cancelling a task."""
        task_id = request.match_info.get())))))))))))))))))))"task_id")
        
        if not task_id or task_id not in self.tasks:
        return web.json_response()))))))))))))))))))){}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
        "error": "Task not found"
        }, status=404)
        
        try:
            task = self.tasks[]],,task_id]
            
            # Can only cancel pending or running tasks
            if task.get())))))))))))))))))))"status") not in []],,"pending", "running"]:
            return web.json_response()))))))))))))))))))){}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "error": f"Cannot cancel task in {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}task.get())))))))))))))))))))'status')} state"
            }, status=400)
            
            # Update task status in database
            self.db.execute())))))))))))))))))))
            """
            UPDATE distributed_tasks
            SET status = 'cancelled', end_time = ?
            WHERE task_id = ?
            """,
            ())))))))))))))))))))datetime.now())))))))))))))))))))), task_id)
            )
            
            # Update in-memory task state
            task[]],,"status"] = "cancelled"
            task[]],,"ended"] = datetime.now())))))))))))))))))))).isoformat()))))))))))))))))))))
            
            # Remove from tracking sets
            if task_id in self.pending_tasks:
                self.pending_tasks.remove())))))))))))))))))))task_id)
            
            if task_id in self.running_tasks:
                worker_id = self.running_tasks[]],,task_id]
                del self.running_tasks[]],,task_id]
                
                # Notify worker if connected:
                if worker_id in self.worker_connections:
                    try:
                        await self.worker_connections[]],,worker_id].send_json()))))))))))))))))))){}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                        "type": "cancel_task",
                        "task_id": task_id
                        })
                    except Exception as e:
                        logger.error())))))))))))))))))))f"Error notifying worker {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}worker_id} of task cancellation: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}str())))))))))))))))))))e)}")
            
                        logger.info())))))))))))))))))))f"Task {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}task_id} cancelled")
            
                        return web.json_response()))))))))))))))))))){}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                        "task_id": task_id,
                        "cancelled": True,
                        "message": "Task cancelled successfully"
                        })
            
        except Exception as e:
            logger.error())))))))))))))))))))f"Error cancelling task {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}task_id}: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}str())))))))))))))))))))e)}")
                        return web.json_response()))))))))))))))))))){}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                        "error": f"Failed to cancel task {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}task_id}",
                        "message": str())))))))))))))))))))e)
                        }, status=500)
    
    def _update_worker_hardware_metrics())))))))))))))))))))self, worker_id: str, hardware_metrics: Dict[]],,str, Any]):
        """
        Update worker hardware metrics.
        
        Args:
            worker_id: Worker ID
            hardware_metrics: Hardware metrics
            """
        # Skip if worker doesn't exist:
        if worker_id not in self.workers:
            return
        
        # Update hardware metrics in worker record
            self.workers[]],,worker_id][]],,"hardware_metrics"] = hardware_metrics
        
        # Log utilization to help identify performance issues
        if "cpu_percent" in hardware_metrics or "memory_percent" in hardware_metrics:
            cpu = hardware_metrics.get())))))))))))))))))))"cpu_percent", 0)
            memory = hardware_metrics.get())))))))))))))))))))"memory_percent", 0)
            
            # Only log if utilization is high:
            if cpu > 80 or memory > 80:
                logger.debug())))))))))))))))))))f"Worker {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}worker_id} high utilization: CPU {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}cpu}%, Memory {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}memory}%")
    
    async def _task_scheduler_loop())))))))))))))))))))self):
        """Run the task scheduler in a loop."""
        while True:
            try:
                if self.pending_tasks and self.task_scheduler:
                    # Use advanced scheduler
                    tasks_assigned = await self.task_scheduler.schedule_pending_tasks()))))))))))))))))))))
                    if tasks_assigned > 0:
                        logger.info())))))))))))))))))))f"Advanced scheduler assigned {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}tasks_assigned} tasks")
                else:
                    # Wait for tasks
                    await asyncio.sleep())))))))))))))))))))1)
            except Exception as e:
                logger.error())))))))))))))))))))f"Error in task scheduler loop: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}str())))))))))))))))))))e)}")
            
            # Check every 5 seconds
                await asyncio.sleep())))))))))))))))))))5)
    
    async def _assign_pending_tasks())))))))))))))))))))self):
        """Attempt to assign pending tasks to available workers."""
        # If advanced scheduler is enabled, use it
        if self.enable_advanced_scheduler and self.task_scheduler:
        return await self.task_scheduler.schedule_pending_tasks()))))))))))))))))))))
        
        # Otherwise use simple assignment logic
        if not self.pending_tasks:
        return 0  # No pending tasks
        
        # Get available workers
        available_workers = {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
        worker_id: worker
        for worker_id, worker in self.workers.items()))))))))))))))))))))
        if worker.get())))))))))))))))))))"status") == "active" and worker_id in self.worker_connections
        }
        :
        if not available_workers:
            return 0  # No available workers
        
        # Sort pending tasks by priority ())))))))))))))))))))higher number = higher priority)
            pending_tasks = sorted())))))))))))))))))))
            []],,self.tasks[]],,task_id] for task_id in self.pending_tasks],:
                key=lambda t: t.get())))))))))))))))))))"priority", 0),
                reverse=True
                )
        
                tasks_assigned = 0
        
        for task in pending_tasks:
            task_id = task[]],,"task_id"]
            
            # Find a suitable worker for this task
            worker_id = await self._find_suitable_worker())))))))))))))))))))task, available_workers)
            
            if worker_id:
                # Assign task to worker
                await self._assign_task_to_worker())))))))))))))))))))task_id, worker_id)
                
                # Remove from pending tasks
                self.pending_tasks.remove())))))))))))))))))))task_id)
                
                # Add to running tasks
                self.running_tasks[]],,task_id] = worker_id
                
                # Update worker availability
                available_workers[]],,worker_id][]],,"status"] = "busy"
                
                logger.info())))))))))))))))))))f"Task {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}task_id} assigned to worker {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}worker_id}")
                
                tasks_assigned += 1
                
                # Remove worker from available workers
                del available_workers[]],,worker_id]
                
                # Stop if no more workers available:
                if not available_workers:
                break
        
            return tasks_assigned
    
    async def _find_suitable_worker())))))))))))))))))))self, task: Dict[]],,str, Any], available_workers: Dict[]],,str, Dict[]],,str, Any]]) -> Optional[]],,str]:
        """
        Find a suitable worker for a task.
        
        Args:
            task: Task information
            available_workers: Dictionary of available workers
            
        Returns:
            Worker ID if a suitable worker is found, None otherwise
            """
            task_requirements = task.get())))))))))))))))))))"requirements", {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}})
        
        # For now, use a simple matching algorithm:
        # TODO: Implement more sophisticated matching based on capabilities and load
        
        for worker_id, worker in available_workers.items())))))))))))))))))))):
            capabilities = worker.get())))))))))))))))))))"capabilities", {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}})
            
            # Check if worker has required hardware
            required_hardware = task_requirements.get())))))))))))))))))))"hardware", []],,]):
            if required_hardware:
                worker_hardware = capabilities.get())))))))))))))))))))"hardware", []],,])
                if not all())))))))))))))))))))hw in worker_hardware for hw in required_hardware):
                continue
            
            # Check if worker has required memory
            min_memory_gb = task_requirements.get())))))))))))))))))))"min_memory_gb", 0):
            if min_memory_gb > 0:
                worker_memory_gb = capabilities.get())))))))))))))))))))"memory", {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}).get())))))))))))))))))))"total_gb", 0)
                if worker_memory_gb < min_memory_gb:
                continue
            
            # Check if worker has required CUDA compute capability
            min_cuda_compute = task_requirements.get())))))))))))))))))))"min_cuda_compute", 0):
            if min_cuda_compute > 0:
                worker_cuda_compute = capabilities.get())))))))))))))))))))"gpu", {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}).get())))))))))))))))))))"cuda_compute", 0)
                if worker_cuda_compute < min_cuda_compute:
                continue
            
            # If we reach here, worker is suitable
                return worker_id
        
                return None
    
    async def _assign_task_to_worker())))))))))))))))))))self, task_id: str, worker_id: str):
        """
        Assign a task to a worker.
        
        Args:
            task_id: Task ID
            worker_id: Worker ID
            """
        # Get task information
            task = self.tasks[]],,task_id]
        
        # Update task status in database
            self.db.execute())))))))))))))))))))
            """
            UPDATE distributed_tasks
            SET status = 'running', start_time = ?, worker_id = ?, attempts = attempts + 1
            WHERE task_id = ?
            """,
            ())))))))))))))))))))datetime.now())))))))))))))))))))), worker_id, task_id)
            )
        
        # Update in-memory task state
            task[]],,"status"] = "running"
            task[]],,"started"] = datetime.now())))))))))))))))))))).isoformat()))))))))))))))))))))
            task[]],,"worker_id"] = worker_id
            task[]],,"attempts"] = task.get())))))))))))))))))))"attempts", 0) + 1
        
        # Notify worker
        if worker_id in self.worker_connections:
            await self.worker_connections[]],,worker_id].send_json()))))))))))))))))))){}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "type": "execute_task",
            "task_id": task_id,
            "task_type": task.get())))))))))))))))))))"type"),
            "config": task.get())))))))))))))))))))"config", {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}})
            })
    
    async def _update_task_status())))))))))))))))))))self, task_id: str, status: str, status_data: Dict[]],,str, Any]):
        """
        Update the status of a task.
        
        Args:
            task_id: Task ID
            status: New status
            status_data: Additional status data
            """
        if task_id not in self.tasks:
            logger.warning())))))))))))))))))))f"Received status update for unknown task {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}task_id}")
            return
        
            task = self.tasks[]],,task_id]
            old_status = task.get())))))))))))))))))))"status")
        
        # Update task status
            task[]],,"status"] = status
        
        # Update additional fields based on status
        if status == "running":
            task[]],,"started"] = datetime.now())))))))))))))))))))).isoformat()))))))))))))))))))))
            task[]],,"progress"] = status_data.get())))))))))))))))))))"progress", 0)
            
        elif status == "completed":
            task[]],,"ended"] = datetime.now())))))))))))))))))))).isoformat()))))))))))))))))))))
            if "result" in status_data:
                task[]],,"result"] = status_data[]],,"result"]
            
            # Update database
                self.db.execute())))))))))))))))))))
                """
                UPDATE distributed_tasks
                SET status = 'completed', end_time = ?
                WHERE task_id = ?
                """,
                ())))))))))))))))))))datetime.now())))))))))))))))))))), task_id)
                )
            
            # Move from running to completed
            if task_id in self.running_tasks:
                worker_id = self.running_tasks[]],,task_id]
                del self.running_tasks[]],,task_id]
                self.completed_tasks.add())))))))))))))))))))task_id)
                
                # Update worker status
                if worker_id in self.workers:
                    self.workers[]],,worker_id][]],,"status"] = "active"
            
        elif status == "failed":
            task[]],,"ended"] = datetime.now())))))))))))))))))))).isoformat()))))))))))))))))))))
            if "error" in status_data:
                task[]],,"error"] = status_data[]],,"error"]
            
            # Update database
                self.db.execute())))))))))))))))))))
                """
                UPDATE distributed_tasks
                SET status = 'failed', end_time = ?
                WHERE task_id = ?
                """,
                ())))))))))))))))))))datetime.now())))))))))))))))))))), task_id)
                )
            
            # Move from running to failed
            if task_id in self.running_tasks:
                worker_id = self.running_tasks[]],,task_id]
                del self.running_tasks[]],,task_id]
                self.failed_tasks.add())))))))))))))))))))task_id)
                
                # Update worker status
                if worker_id in self.workers:
                    self.workers[]],,worker_id][]],,"status"] = "active"
        
        # Log status change
                    logger.info())))))))))))))))))))f"Task {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}task_id} status changed from {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}old_status} to {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}status}")
        
        # Assign pending tasks if workers are available:
        if status in []],,"completed", "failed"] and self.pending_tasks:
            await self._assign_pending_tasks()))))))))))))))))))))
    
    async def _process_task_result())))))))))))))))))))self, task_id: str, worker_id: str, result_data: Dict[]],,str, Any]):
        """
        Process a task result from a worker.
        
        Args:
            task_id: Task ID
            worker_id: Worker ID
            result_data: Result data
            """
        # Extract result information
            status = result_data.get())))))))))))))))))))"status", "completed")
            execution_time = result_data.get())))))))))))))))))))"execution_time_seconds", 0)
            hardware_metrics = result_data.get())))))))))))))))))))"hardware_metrics", {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}})
            error_message = result_data.get())))))))))))))))))))"error", "")
        
        # Check if task exists:
        if task_id not in self.tasks:
            logger.warning())))))))))))))))))))f"Received result for unknown task {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}task_id}")
            return
        
        # Check if worker is assigned to task
        task = self.tasks[]],,task_id]:
        if task.get())))))))))))))))))))"worker_id") != worker_id:
            logger.warning())))))))))))))))))))f"Received result for task {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}task_id} from unassigned worker {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}worker_id}")
            return
        
        # Store execution in history
        try:
            self.db.execute())))))))))))))))))))
            """
            INSERT INTO task_execution_history ())))))))))))))))))))
            task_id, worker_id, attempt, status, start_time, end_time,
            execution_time_seconds, error_message, hardware_metrics
            ) VALUES ())))))))))))))))))))?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ())))))))))))))))))))
            task_id, worker_id, task.get())))))))))))))))))))"attempts", 1), status,
            datetime.fromisoformat())))))))))))))))))))task.get())))))))))))))))))))"started", datetime.now())))))))))))))))))))).isoformat())))))))))))))))))))))),
            datetime.now())))))))))))))))))))), execution_time, error_message,
            json.dumps())))))))))))))))))))hardware_metrics)
            )
            )
            
            logger.info())))))))))))))))))))f"Stored execution history for task {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}task_id}")
            
        except Exception as e:
            logger.error())))))))))))))))))))f"Error storing execution history for task {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}task_id}: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}str())))))))))))))))))))e)}")
        
        # Update worker performance metrics in task scheduler
        if self.enable_advanced_scheduler and self.task_scheduler:
            # Create combined task result for performance tracking
            task_result = {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "task_id": task_id,
            "type": task.get())))))))))))))))))))"type", "unknown"),
            "status": status,
            "execution_time_seconds": execution_time,
            "error": error_message,
            "hardware_metrics": hardware_metrics
            }
            
            # Update worker performance metrics
            self.task_scheduler.update_worker_performance())))))))))))))))))))worker_id, task_result)
            
            # If health monitor is enabled, update task timeout estimates
            if self.enable_health_monitor and self.health_monitor and execution_time > 0:
                self.health_monitor.update_task_timeout_estimate())))))))))))))))))))task.get())))))))))))))))))))"type", "unknown"), execution_time)
        
        # Update task status
                await self._update_task_status())))))))))))))))))))task_id, status, result_data)
    
    async def start())))))))))))))))))))self):
        """Start the coordinator server."""
        runner = web.AppRunner())))))))))))))))))))self.app)
        await runner.setup()))))))))))))))))))))
        site = web.TCPSite())))))))))))))))))))runner, self.host, self.port)
        await site.start()))))))))))))))))))))
        
        logger.info())))))))))))))))))))f"Coordinator server started at http://{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}self.host}:{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}self.port}")
        
        # Start background tasks
        background_tasks = []],,]
        
        # Start legacy heartbeat checker if health monitor is disabled:
        if not self.enable_health_monitor:
            background_tasks.append())))))))))))))))))))asyncio.create_task())))))))))))))))))))self._heartbeat_checker()))))))))))))))))))))))
        
        # Start health monitor if enabled::
        if self.enable_health_monitor and self.health_monitor:
            background_tasks.append())))))))))))))))))))asyncio.create_task())))))))))))))))))))self.health_monitor.start_monitoring()))))))))))))))))))))))
            logger.info())))))))))))))))))))"Health monitoring started")
        
        # Start task scheduler loop
        if self.enable_advanced_scheduler and self.task_scheduler:
            background_tasks.append())))))))))))))))))))asyncio.create_task())))))))))))))))))))self._task_scheduler_loop()))))))))))))))))))))))
            logger.info())))))))))))))))))))"Advanced task scheduler started")
        
        # Start load balancer loop
        if self.enable_load_balancer and self.load_balancer:
            background_tasks.append())))))))))))))))))))asyncio.create_task())))))))))))))))))))self.load_balancer.start_balancing()))))))))))))))))))))))
            logger.info())))))))))))))))))))"Adaptive load balancer started")
        
        # Keep server running
        while True:
            await asyncio.sleep())))))))))))))))))))3600)  # Sleep for 1 hour
    
    async def _heartbeat_checker())))))))))))))))))))self):
        """Check worker heartbeats periodically."""
        while True:
            try:
                now = datetime.now()))))))))))))))))))))
                
                for worker_id, worker in list())))))))))))))))))))self.workers.items()))))))))))))))))))))):
                    # Skip offline workers
                    if worker.get())))))))))))))))))))"status") == "offline":
                    continue
                    
                    last_heartbeat = datetime.fromisoformat())))))))))))))))))))worker.get())))))))))))))))))))"last_heartbeat", "1970-01-01T00:00:00"))
                    heartbeat_age = ())))))))))))))))))))now - last_heartbeat).total_seconds()))))))))))))))))))))
                    
                    # If heartbeat is older than 30 seconds, mark as offline
                    if heartbeat_age > 30:
                        logger.warning())))))))))))))))))))f"Worker {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}worker_id} heartbeat is {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}heartbeat_age:.1f} seconds old, marking as offline")
                        worker[]],,"status"] = "offline"
                        
                        # Update database
                        self.db.execute())))))))))))))))))))
                        """
                        UPDATE worker_nodes
                        SET status = 'offline'
                        WHERE worker_id = ?
                        """,
                        ())))))))))))))))))))worker_id,)
                        )
                        
                        # Check for running tasks assigned to this worker
                        for task_id, assigned_worker_id in list())))))))))))))))))))self.running_tasks.items()))))))))))))))))))))):
                            if assigned_worker_id == worker_id:
                                logger.warning())))))))))))))))))))f"Worker {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}worker_id} is offline, marking task {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}task_id} as failed")
                                
                                # Update task status
                                await self._update_task_status())))))))))))))))))))task_id, "failed", {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                                "error": "Worker disconnected"
                                })
                
                # Re-assign tasks if needed:
                if self.pending_tasks:
                    await self._assign_pending_tasks()))))))))))))))))))))
                
            except Exception as e:
                logger.error())))))))))))))))))))f"Error in heartbeat checker: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}str())))))))))))))))))))e)}")
            
            # Sleep for 10 seconds
                await asyncio.sleep())))))))))))))))))))10)


async def main())))))))))))))))))))):
    """Main function."""
    parser = argparse.ArgumentParser())))))))))))))))))))description="Distributed Testing Framework Coordinator")
    parser.add_argument())))))))))))))))))))"--host", default="0.0.0.0", help="Host to bind the server to")
    parser.add_argument())))))))))))))))))))"--port", type=int, default=8080, help="Port to bind the server to")
    parser.add_argument())))))))))))))))))))"--db-path", default="./benchmark_db.duckdb", help="Path to DuckDB database")
    parser.add_argument())))))))))))))))))))"--security-config", default="./security_config.json", 
    help="Path to security configuration file")
    parser.add_argument())))))))))))))))))))"--generate-admin-key", action="store_true", 
    help="Generate a new admin API key and display it")
    parser.add_argument())))))))))))))))))))"--generate-worker-key", action="store_true", 
    help="Generate a new worker API key and display it")
    parser.add_argument())))))))))))))))))))"--disable-health-monitor", action="store_true",
    help="Disable the health monitoring system")
    parser.add_argument())))))))))))))))))))"--disable-advanced-scheduler", action="store_true",
    help="Disable the advanced task scheduler")
    parser.add_argument())))))))))))))))))))"--disable-load-balancer", action="store_true",
    help="Disable the adaptive load balancer")
    parser.add_argument())))))))))))))))))))"--max-tasks-per-worker", type=int, default=1,
    help="Maximum number of tasks that can be assigned to a worker simultaneously")
    
    args = parser.parse_args()))))))))))))))))))))
    
    # Create coordinator
    coordinator = DistributedTestingCoordinator())))))))))))))))))))
    db_path=args.db_path,
    host=args.host,
    port=args.port,
    security_config=args.security_config,
    enable_health_monitor=not args.disable_health_monitor,
    enable_advanced_scheduler=not args.disable_advanced_scheduler,
    enable_load_balancer=not args.disable_load_balancer
    )
    
    # Configure task scheduler if enabled::
    if coordinator.enable_advanced_scheduler and coordinator.task_scheduler:
        coordinator.task_scheduler.max_tasks_per_worker = args.max_tasks_per_worker
    
    # Generate and display API keys if requested:
    if args.generate_admin_key:
        admin_key = coordinator.security_manager.generate_api_key())))))))))))))))))))"admin-user", []],,"admin"])
        print())))))))))))))))))))f"\n=== ADMIN API KEY ===\n{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}admin_key}\n")
        print())))))))))))))))))))"Save this key securely as it will not be displayed again.")
        print())))))))))))))))))))"Use this key for administrative access to the coordinator API.\n")
        coordinator.security_manager.save_config())))))))))))))))))))args.security_config)
        
    if args.generate_worker_key:
        worker_key = coordinator.security_manager.generate_api_key())))))))))))))))))))"worker-node", []],,"worker"])
        print())))))))))))))))))))f"\n=== WORKER API KEY ===\n{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}worker_key}\n")
        print())))))))))))))))))))"Save this key securely as it will not be displayed again.")
        print())))))))))))))))))))"Use this key for worker nodes to connect to the coordinator.\n")
        coordinator.security_manager.save_config())))))))))))))))))))args.security_config)
    
    # Log configuration
        logger.info())))))))))))))))))))f"Starting coordinator with configuration:")
        logger.info())))))))))))))))))))f"  - Database path: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}args.db_path}")
        logger.info())))))))))))))))))))f"  - Host: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}args.host}")
        logger.info())))))))))))))))))))f"  - Port: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}args.port}")
    logger.info())))))))))))))))))))f"  - Health monitor: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}'Enabled' if not args.disable_health_monitor else 'Disabled'}"):
    logger.info())))))))))))))))))))f"  - Advanced scheduler: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}'Enabled' if not args.disable_advanced_scheduler else 'Disabled'}"):
    logger.info())))))))))))))))))))f"  - Adaptive load balancer: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}'Enabled' if not args.disable_load_balancer else 'Disabled'}"):
    if not args.disable_advanced_scheduler:
        logger.info())))))))))))))))))))f"  - Max tasks per worker: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}args.max_tasks_per_worker}")
    
        await coordinator.start()))))))))))))))))))))


if __name__ == "__main__":
    asyncio.run())))))))))))))))))))main())))))))))))))))))))))