#!/usr/bin/env python3
"""
Real-Time Monitoring for End-to-End Testing

This module provides real-time monitoring of end-to-end test execution, 
sending live updates to the monitoring dashboard via WebSockets.

Usage:
    python -m duckdb_api.distributed_testing.tests.realtime_monitoring [options]

Options:
    --dashboard-url URL           URL of monitoring dashboard (default: http://localhost:8082)
    --test-id ID                  Specific test ID to monitor
    --update-interval SECONDS     Update interval in seconds (default: 1)
    --debug                       Enable debug logging
"""

import anyio
import json
import logging
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

import aiohttp
import argparse
import websockets

# Add parent directory to path to ensure imports work properly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

logger = logging.getLogger(__name__)

class RealTimeMonitor:
    """Real-time monitor for end-to-end test execution."""
    
    def __init__(
        self, 
        dashboard_url: str = "http://localhost:8082",
        test_id: Optional[str] = None,
        update_interval: float = 1.0,
        debug: bool = False
    ):
        """Initialize the real-time monitor.
        
        Args:
            dashboard_url: URL of the monitoring dashboard
            test_id: Specific test ID to monitor (if None, monitor latest test)
            update_interval: Update interval in seconds
            debug: Enable debug logging
        """
        self.dashboard_url = dashboard_url
        self.test_id = test_id
        self.update_interval = update_interval
        self.debug = debug
        
        # Convert HTTP URL to WebSocket URL
        if dashboard_url.startswith("http://"):
            self.ws_url = dashboard_url.replace("http://", "ws://")
        elif dashboard_url.startswith("https://"):
            self.ws_url = dashboard_url.replace("https://", "wss://")
        else:
            self.ws_url = f"ws://{dashboard_url}"
        
        # Ensure WS URL ends with /
        if not self.ws_url.endswith("/"):
            self.ws_url += "/"
        
        # WebSocket endpoint
        self.ws_endpoint = f"{self.ws_url}ws/e2e-test-monitoring"
        
        # HTTP session
        self.session = None
        
        # WebSocket connection
        self.ws_connection = None
        
        # Test state
        self.test_state = {
            "test_id": self.test_id or f"e2e_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "start_time": datetime.now().isoformat(),
            "status": "initializing",
            "phases": {},
            "components": {},
            "resources": {},
            "errors": [],
            "progress": 0.0
        }
        
        # Running flag
        self.running = False
        
        logger.info(f"Initialized real-time monitor for test {self.test_state['test_id']}")
        logger.info(f"Dashboard URL: {self.dashboard_url}")
        logger.info(f"WebSocket endpoint: {self.ws_endpoint}")
    
    async def connect_websocket(self) -> bool:
        """Connect to the WebSocket endpoint.
        
        Returns:
            Success status
        """
        try:
            logger.info(f"Connecting to WebSocket endpoint: {self.ws_endpoint}")
            self.ws_connection = await websockets.connect(self.ws_endpoint)
            
            # Send initial message
            await self.ws_connection.send(json.dumps({
                "type": "e2e_test_monitoring_init",
                "test_id": self.test_state["test_id"],
                "timestamp": datetime.now().isoformat()
            }))
            
            logger.info("WebSocket connection established")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to WebSocket: {e}")
            return False
    
    async def send_update(self) -> bool:
        """Send an update via WebSocket.
        
        Returns:
            Success status
        """
        if not self.ws_connection:
            return False
        
        try:
            # Add timestamp to state
            self.test_state["timestamp"] = datetime.now().isoformat()
            
            # Send update
            await self.ws_connection.send(json.dumps({
                "type": "e2e_test_monitoring_update",
                "data": self.test_state
            }))
            
            return True
        except Exception as e:
            logger.error(f"Failed to send update: {e}")
            return False
    
    def update_test_state(self, update: Dict[str, Any]):
        """Update the test state with new information.
        
        Args:
            update: Dict with state updates
        """
        # Update top-level fields
        for key, value in update.items():
            if key not in ["phases", "components", "resources", "errors"]:
                self.test_state[key] = value
        
        # Update phases
        if "phases" in update:
            for phase_name, phase_data in update["phases"].items():
                if phase_name not in self.test_state["phases"]:
                    self.test_state["phases"][phase_name] = {}
                
                self.test_state["phases"][phase_name].update(phase_data)
        
        # Update components
        if "components" in update:
            for component_name, component_data in update["components"].items():
                if component_name not in self.test_state["components"]:
                    self.test_state["components"][component_name] = {}
                
                self.test_state["components"][component_name].update(component_data)
        
        # Update resources
        if "resources" in update:
            for resource_name, resource_data in update["resources"].items():
                if resource_name not in self.test_state["resources"]:
                    self.test_state["resources"][resource_name] = {}
                
                self.test_state["resources"][resource_name].update(resource_data)
        
        # Update errors
        if "errors" in update and isinstance(update["errors"], list):
            self.test_state["errors"].extend(update["errors"])
    
    async def monitor_test_process(self, process_id: int = None):
        """Monitor the test process and send updates.
        
        Args:
            process_id: Optional process ID to monitor
        """
        if process_id:
            logger.info(f"Monitoring test process with PID: {process_id}")
        
        # Update test state to running
        self.update_test_state({
            "status": "running",
            "phases": {
                "initialization": {
                    "status": "completed",
                    "start_time": self.test_state["start_time"],
                    "end_time": datetime.now().isoformat()
                },
                "services_startup": {
                    "status": "running",
                    "start_time": datetime.now().isoformat()
                }
            },
            "progress": 5.0
        })
        
        # Send initial update
        await self.send_update()
        
        # Simulate phases of E2E test execution
        await self.simulate_services_startup_phase()
        await self.simulate_worker_startup_phase()
        await self.simulate_workload_submission_phase()
        await self.simulate_test_execution_phase()
        await self.simulate_validation_phase()
        await self.simulate_cleanup_phase()
        
        # Update test state to completed
        self.update_test_state({
            "status": "completed",
            "end_time": datetime.now().isoformat(),
            "progress": 100.0
        })
        
        # Send final update
        await self.send_update()
        
        logger.info("Test monitoring completed")
    
    async def simulate_services_startup_phase(self):
        """Simulate the services startup phase."""
        logger.info("Simulating services startup phase")
        
        # Initialize components
        components = {
            "result_aggregator": {"status": "starting"},
            "coordinator": {"status": "starting"},
            "monitoring_dashboard": {"status": "starting"}
        }
        
        # Update test state
        self.update_test_state({
            "components": components,
            "progress": 10.0
        })
        await self.send_update()
        
        # Simulate services starting up one by one
        for component, delay in [
            ("result_aggregator", 2),
            ("coordinator", 3),
            ("monitoring_dashboard", 2)
        ]:
            await anyio.sleep(delay)
            components[component]["status"] = "running"
            components[component]["start_time"] = datetime.now().isoformat()
            self.update_test_state({
                "components": components,
                "progress": self.test_state["progress"] + 5.0
            })
            await self.send_update()
        
        # Complete phase
        self.update_test_state({
            "phases": {
                "services_startup": {
                    "status": "completed",
                    "end_time": datetime.now().isoformat()
                },
                "worker_startup": {
                    "status": "running",
                    "start_time": datetime.now().isoformat()
                }
            },
            "progress": 25.0
        })
        await self.send_update()
    
    async def simulate_worker_startup_phase(self):
        """Simulate the worker startup phase."""
        logger.info("Simulating worker startup phase")
        
        # Initialize workers
        num_workers = 5
        workers = {}
        
        for i in range(num_workers):
            hardware_type = ["cpu", "gpu", "webgpu", "webnn", "multi"][i % 5]
            workers[f"worker_{i}"] = {
                "status": "starting",
                "hardware_type": hardware_type
            }
        
        # Update test state
        self.update_test_state({
            "components": {
                "workers": workers
            },
            "progress": 30.0
        })
        await self.send_update()
        
        # Simulate workers starting up
        worker_names = list(workers.keys())
        for i, worker_name in enumerate(worker_names):
            await anyio.sleep(1)
            workers[worker_name]["status"] = "running"
            workers[worker_name]["start_time"] = datetime.now().isoformat()
            self.update_test_state({
                "components": {
                    "workers": workers
                },
                "progress": 30.0 + (i + 1) * 10.0 / len(worker_names)
            })
            await self.send_update()
        
        # Complete phase
        self.update_test_state({
            "phases": {
                "worker_startup": {
                    "status": "completed",
                    "end_time": datetime.now().isoformat()
                },
                "workload_submission": {
                    "status": "running",
                    "start_time": datetime.now().isoformat()
                }
            },
            "progress": 40.0
        })
        await self.send_update()
    
    async def simulate_workload_submission_phase(self):
        """Simulate the workload submission phase."""
        logger.info("Simulating workload submission phase")
        
        # Initialize task types
        task_types = ["performance", "compatibility", "integration", "web_platform"]
        
        # Update test state
        self.update_test_state({
            "resources": {
                "tasks": {
                    task_type: {"submitted": 0, "total": 10} for task_type in task_types
                }
            },
            "progress": 45.0
        })
        await self.send_update()
        
        # Simulate task submission
        for task_type in task_types:
            for i in range(10):  # 10 tasks per type
                await anyio.sleep(0.2)
                self.test_state["resources"]["tasks"][task_type]["submitted"] = i + 1
                self.update_test_state({
                    "progress": 45.0 + (task_type.index(task_type) * 10 + (i + 1)) / (len(task_types) * 10) * 5.0
                })
                await self.send_update()
        
        # Complete phase
        self.update_test_state({
            "phases": {
                "workload_submission": {
                    "status": "completed",
                    "end_time": datetime.now().isoformat()
                },
                "test_execution": {
                    "status": "running",
                    "start_time": datetime.now().isoformat()
                }
            },
            "progress": 50.0
        })
        await self.send_update()
    
    async def simulate_test_execution_phase(self):
        """Simulate the test execution phase."""
        logger.info("Simulating test execution phase")
        
        # Initialize task tracking
        task_types = ["performance", "compatibility", "integration", "web_platform"]
        task_tracking = {
            task_type: {"completed": 0, "total": 10, "succeeded": 0, "failed": 0} 
            for task_type in task_types
        }
        
        # Update test state
        self.update_test_state({
            "resources": {
                "task_execution": task_tracking
            },
            "progress": 55.0
        })
        await self.send_update()
        
        # Simulate task execution with occasionally injected failures
        for task_type in task_types:
            for i in range(10):  # 10 tasks per type
                await anyio.sleep(0.5)
                
                # Simulate occasional failure
                success = not (i == 7 and task_type == "compatibility")  # Fail one task
                
                task_tracking[task_type]["completed"] = i + 1
                if success:
                    task_tracking[task_type]["succeeded"] = task_tracking[task_type]["succeeded"] + 1
                else:
                    task_tracking[task_type]["failed"] = task_tracking[task_type]["failed"] + 1
                    self.update_test_state({
                        "errors": [{
                            "timestamp": datetime.now().isoformat(),
                            "type": "task_failure",
                            "message": f"Task {task_type}_{i} failed",
                            "details": {
                                "task_type": task_type,
                                "task_id": i
                            }
                        }]
                    })
                
                self.update_test_state({
                    "resources": {
                        "task_execution": task_tracking
                    },
                    "progress": 55.0 + (task_type.index(task_type) * 10 + (i + 1)) / (len(task_types) * 10) * 20.0
                })
                await self.send_update()
        
        # Complete phase
        self.update_test_state({
            "phases": {
                "test_execution": {
                    "status": "completed",
                    "end_time": datetime.now().isoformat()
                },
                "validation": {
                    "status": "running",
                    "start_time": datetime.now().isoformat()
                }
            },
            "progress": 75.0
        })
        await self.send_update()
    
    async def simulate_validation_phase(self):
        """Simulate the validation phase."""
        logger.info("Simulating validation phase")
        
        # Initialize validation areas
        validation_areas = [
            "dashboard_accessibility", 
            "results_page", 
            "result_aggregation", 
            "integration"
        ]
        
        validation_status = {area: {"status": "pending"} for area in validation_areas}
        
        # Update test state
        self.update_test_state({
            "resources": {
                "validation": validation_status
            },
            "progress": 80.0
        })
        await self.send_update()
        
        # Simulate validation checks
        for i, area in enumerate(validation_areas):
            await anyio.sleep(1)
            validation_status[area]["status"] = "completed"
            validation_status[area]["result"] = "passed"
            validation_status[area]["timestamp"] = datetime.now().isoformat()
            
            self.update_test_state({
                "resources": {
                    "validation": validation_status
                },
                "progress": 80.0 + (i + 1) / len(validation_areas) * 10.0
            })
            await self.send_update()
        
        # Complete phase
        self.update_test_state({
            "phases": {
                "validation": {
                    "status": "completed",
                    "end_time": datetime.now().isoformat()
                },
                "cleanup": {
                    "status": "running",
                    "start_time": datetime.now().isoformat()
                }
            },
            "progress": 90.0
        })
        await self.send_update()
    
    async def simulate_cleanup_phase(self):
        """Simulate the cleanup phase."""
        logger.info("Simulating cleanup phase")
        
        # Update component status for cleanup
        component_status = {
            "workers": {"status": "stopping"},
            "result_aggregator": {"status": "running"},
            "coordinator": {"status": "running"},
            "monitoring_dashboard": {"status": "running"}
        }
        
        # Update test state
        self.update_test_state({
            "components": component_status,
            "progress": 92.0
        })
        await self.send_update()
        
        # Simulate workers stopping
        await anyio.sleep(2)
        component_status["workers"] = {"status": "stopped"}
        
        self.update_test_state({
            "components": component_status,
            "progress": 95.0
        })
        await self.send_update()
        
        # Simulate services stopping
        for component in ["coordinator", "result_aggregator", "monitoring_dashboard"]:
            await anyio.sleep(1)
            component_status[component] = {"status": "stopped"}
            
            self.update_test_state({
                "components": component_status,
                "progress": 95.0 + ["coordinator", "result_aggregator", "monitoring_dashboard"].index(component) / 3 * 5.0
            })
            await self.send_update()
        
        # Complete phase
        self.update_test_state({
            "phases": {
                "cleanup": {
                    "status": "completed",
                    "end_time": datetime.now().isoformat()
                }
            },
            "progress": 100.0
        })
        await self.send_update()
    
    async def start(self):
        """Start the real-time monitor."""
        self.running = True
        
        # Connect to WebSocket
        if not await self.connect_websocket():
            logger.error("Failed to connect to WebSocket, monitoring disabled")
            return
        
        try:
            # Monitor test process
            await self.monitor_test_process()
        except asyncio.CancelledError:
            logger.info("Monitoring cancelled")
        finally:
            self.running = False
            
            # Close WebSocket connection
            if self.ws_connection:
                await self.ws_connection.close()
                self.ws_connection = None

async def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Real-Time Monitoring for End-to-End Testing")
    
    # Basic options
    parser.add_argument("--dashboard-url", default="http://localhost:8082",
                      help="URL of monitoring dashboard")
    parser.add_argument("--test-id", help="Specific test ID to monitor")
    parser.add_argument("--update-interval", type=float, default=1.0,
                      help="Update interval in seconds")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
    )
    
    # Create and start monitor
    monitor = RealTimeMonitor(
        dashboard_url=args.dashboard_url,
        test_id=args.test_id,
        update_interval=args.update_interval,
        debug=args.debug
    )
    
    # Handle keyboard interrupt
    loop = # TODO: Remove event loop management - asyncio.get_event_loop()
    for s in [signal.SIGINT, signal.SIGTERM]:
        loop.add_signal_handler(s, lambda: # TODO: Replace with task group - asyncio.create_task(shutdown(monitor)))
    
    # Start monitoring
    await monitor.start()

async def shutdown(monitor):
    """Gracefully shutdown the monitor."""
    if monitor.running:
        logger.info("Shutting down...")
        monitor.running = False

if __name__ == "__main__":
    anyio.run(main())