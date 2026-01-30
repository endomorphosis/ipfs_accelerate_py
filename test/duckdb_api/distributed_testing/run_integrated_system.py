#!/usr/bin/env python3
"""
Run Integrated Distributed Testing System

This script provides a complete solution for running all major components of the Distributed Testing Framework:
1. Coordinator Server with DuckDB integration
2. Load Balancer with intelligent task distribution
3. Multi-Device Orchestrator for distributed task execution
4. Fault Tolerance System with circuit breaking and recovery
5. Comprehensive Monitoring Dashboard with real-time visualization

Features:
- Complete integration of all framework components
- Real-time web-based dashboard with visualization
- Intelligent task distribution with load balancing
- Multi-device task orchestration with 5 splitting strategies
- Comprehensive fault tolerance mechanisms
- Dynamic resource management
- Mock workers for local testing
- Stress test capabilities
- Enhanced error handling and visualization
- Cross-browser WebNN/WebGPU testing support
- Predictive performance analytics
- Hardware-aware task routing
- Circuit breaker patterns for fault isolation

Usage examples:
    # Basic usage with default settings
    python run_integrated_system.py

    # Custom ports and database
    python run_integrated_system.py --port 8080 --dashboard-port 8888 --db-path ./benchmark_db.duckdb

    # Advanced orchestration configuration
    python run_integrated_system.py --orchestrator-strategy data_parallel --enable-distributed

    # Run with terminal-based dashboard (no web interface)
    python run_integrated_system.py --terminal-dashboard

    # Run with mock workers for testing
    python run_integrated_system.py --mock-workers 5

    # Run with stress testing enabled
    python run_integrated_system.py --stress-test --test-workers 10 --test-tasks 50
    
    # Run with fault injection for testing fault tolerance
    python run_integrated_system.py --fault-injection --fault-rate 0.1
    
    # Run with high availability cluster configuration
    python run_integrated_system.py --high-availability --coordinator-id coordinator1
    
    # Run with performance analytics enabled
    python run_integrated_system.py --performance-analytics --visualization-path ./visualizations
    
    # Run with enhanced hardware taxonomy
    python run_integrated_system.py --enhanced-hardware-taxonomy
"""

import os
import sys
import time
import json
import anyio
import logging
import argparse
import threading
import subprocess
import webbrowser
import random
import uuid
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union

# Add parent directory to path to import modules from parent
parent_dir = str(Path(__file__).parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("integrated_system")

def generate_default_security_config(output_path=None):
    """Generate a default security configuration."""
    import secrets
    
    config = {
        "token_secret": secrets.token_hex(32),
        "api_keys": {
            "admin": secrets.token_hex(16),
            "worker": secrets.token_hex(16),
            "user": secrets.token_hex(16)
        }
    }
    
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)
        
    return config

def start_terminal_dashboard(coordinator, load_balancer, refresh_interval=1.0):
    """Start a terminal-based dashboard for monitoring."""
    try:
        from data.duckdb.distributed_testing.load_balancer_live_dashboard import TerminalDashboard
        
        # Create and start the terminal dashboard
        dashboard = TerminalDashboard(
            coordinator=coordinator,
            load_balancer=load_balancer,
            refresh_interval=refresh_interval
        )
        
        # Start dashboard in new thread
        dashboard_thread = threading.Thread(target=dashboard.start)
        dashboard_thread.daemon = True
        dashboard_thread.start()
        
        return dashboard
    except ImportError as e:
        logger.error(f"Failed to start terminal dashboard: {e}")
        logger.error("Make sure you have the required packages installed:")
        logger.error("pip install blessed numpy")
        return None

def launch_mock_workers(coordinator_url, api_key, count=3, capabilities=None):
    """Launch mock worker processes for testing."""
    worker_processes = []
    
    # Create temporary directory for worker files
    import tempfile
    temp_dir = tempfile.mkdtemp(prefix="mock_workers_")
    
    # Default capabilities for different worker types
    default_capabilities = [
        # Basic CPU worker
        {
            "hardware_types": ["cpu"],
            "memory_gb": 4
        },
        # CPU + GPU worker
        {
            "hardware_types": ["cpu", "cuda"],
            "cuda_compute": 7.5,
            "memory_gb": 16
        },
        # CPU + WebGPU worker
        {
            "hardware_types": ["cpu", "webgpu"],
            "browsers": ["chrome", "firefox"],
            "memory_gb": 8
        },
        # CPU + TPU worker
        {
            "hardware_types": ["cpu", "tpu"],
            "tpu_version": "v3",
            "memory_gb": 32
        },
        # Low-powered CPU worker
        {
            "hardware_types": ["cpu"],
            "cpu_cores": 2,
            "memory_gb": 2
        }
    ]
    
    # Override with custom capabilities if provided
    if capabilities:
        default_capabilities = capabilities
    
    # Launch worker processes
    for i in range(count):
        worker_id = f"mock_worker_{i}"
        worker_dir = os.path.join(temp_dir, f"worker_{i}")
        os.makedirs(worker_dir, exist_ok=True)
        
        # Select capability set (cycling through the available options)
        capability_index = i % len(default_capabilities)
        capabilities_json = json.dumps(default_capabilities[capability_index])
        
        worker_cmd = [
            sys.executable,
            os.path.join(parent_dir, "duckdb_api/distributed_testing/worker.py"),
            "--coordinator", coordinator_url,
            "--api-key", api_key,
            "--worker-id", worker_id,
            "--work-dir", worker_dir,
            "--reconnect-interval", "2",
            "--heartbeat-interval", "3",
            "--capabilities", capabilities_json
        ]
        
        logger.info(f"Starting mock worker {worker_id} with capabilities: {default_capabilities[capability_index]}")
        process = subprocess.Popen(worker_cmd)
        worker_processes.append({
            "process": process,
            "worker_id": worker_id,
            "work_dir": worker_dir,
            "capabilities": default_capabilities[capability_index]
        })
    
    return worker_processes, temp_dir

def run_stress_test(coordinator, task_count=20, duration=60, task_types=None, fault_injection=False, fault_rate=0.05):
    """Run a stress test with configurable parameters."""
    logger.info(f"Starting stress test with {task_count} tasks for {duration} seconds " + 
                f"{'with fault injection' if fault_injection else 'without fault injection'}")
    
    # Default task types if none provided
    if not task_types:
        task_types = [
            {
                "type": "command",
                "config": {"command": ["sleep", "5"]},
                "requirements": {"hardware": ["cpu"]},
                "priority": 1
            },
            {
                "type": "command",
                "config": {"command": ["echo", "GPU task"]},
                "requirements": {"hardware": ["cuda"]},
                "priority": 2
            },
            {
                "type": "command",
                "config": {"command": ["echo", "WebGPU task"]},
                "requirements": {"hardware": ["webgpu"]},
                "priority": 3
            },
            {
                "type": "benchmark",
                "config": {
                    "model": "bert-base-uncased",
                    "batch_sizes": [1, 2, 4],
                    "precision": "fp16",
                    "iterations": 3
                },
                "requirements": {"hardware": ["cpu"], "min_memory_gb": 8},
                "priority": 1
            },
            {
                "type": "benchmark",
                "config": {
                    "model": "t5-small",
                    "batch_sizes": [1],
                    "precision": "fp32",
                    "iterations": 2
                },
                "requirements": {"hardware": ["cpu"]},
                "priority": 4
            }
        ]
    
    # Add orchestration tasks if multi-device orchestrator is enabled
    task_types.extend([
        {
            "type": "multi_device_orchestration",
            "config": {
                "model": "llama-7b",
                "strategy": "model_parallel",
                "num_workers": 4,
                "device_map": None  # Auto-determine
            },
            "requirements": {"hardware": ["cuda"], "min_memory_gb": 8},
            "priority": 5
        },
        {
            "type": "multi_device_orchestration",
            "config": {
                "model": "clip-vit-base",
                "strategy": "data_parallel",
                "num_workers": 3,
                "batch_size": 32
            },
            "requirements": {"hardware": ["cuda", "cpu"]},
            "priority": 3
        }
    ])
    
    # Create tasks
    task_ids = []
    start_time = time.time()
    
    # Submit initial batch of tasks
    for i in range(task_count // 2):  # Submit half initially
        task_type = task_types[i % len(task_types)]
        
        # Apply fault injection if enabled (create invalid tasks occasionally)
        if fault_injection and random.random() < fault_rate:
            # Create a faulty task by introducing errors
            faulty_task = task_type.copy()
            faulty_task["config"] = task_type["config"].copy()
            fault_type = random.choice(["missing_param", "invalid_hardware", "high_priority", "invalid_command"])
            
            if fault_type == "missing_param":
                # Remove a critical parameter
                if "model" in faulty_task["config"]:
                    del faulty_task["config"]["model"]
            elif fault_type == "invalid_hardware":
                # Request non-existent hardware
                faulty_task["requirements"] = {"hardware": ["quantum_gpu"]}
            elif fault_type == "high_priority":
                # Set extremely high priority to test load balancer
                faulty_task["priority"] = 9999
            elif fault_type == "invalid_command":
                # Set invalid command
                if faulty_task["type"] == "command":
                    faulty_task["config"]["command"] = ["invalid_command_that_does_not_exist"]
            
            # Use the faulty task
            task_type = faulty_task
            logger.info(f"Created faulty task with {fault_type} fault")
        
        task_id = coordinator.add_task(
            task_type["type"],
            task_type["config"],
            task_type["requirements"],
            task_type["priority"]
        )
        task_ids.append(task_id)
    
    # Continue submitting tasks at a steady rate until duration is reached
    remaining_tasks = task_count - (task_count // 2)
    task_interval = duration / (remaining_tasks + 1)  # Spread over the duration
    
    def submit_remaining_tasks():
        nonlocal remaining_tasks
        next_task_index = task_count // 2
        
        while time.time() - start_time < duration and remaining_tasks > 0:
            # Submit next task
            task_type = task_types[next_task_index % len(task_types)]
            
            # Apply fault injection if enabled
            if fault_injection and random.random() < fault_rate:
                # Create a faulty task
                faulty_task = task_type.copy()
                faulty_task["config"] = task_type["config"].copy()
                fault_type = random.choice(["missing_param", "invalid_hardware", "high_priority", "invalid_command"])
                
                if fault_type == "missing_param":
                    if "model" in faulty_task["config"]:
                        del faulty_task["config"]["model"]
                elif fault_type == "invalid_hardware":
                    faulty_task["requirements"] = {"hardware": ["quantum_gpu"]}
                elif fault_type == "high_priority":
                    faulty_task["priority"] = 9999
                elif fault_type == "invalid_command":
                    if faulty_task["type"] == "command":
                        faulty_task["config"]["command"] = ["invalid_command_that_does_not_exist"]
                
                task_type = faulty_task
                logger.info(f"Created faulty task with {fault_type} fault")
            
            task_id = coordinator.add_task(
                task_type["type"],
                task_type["config"],
                task_type["requirements"],
                task_type["priority"]
            )
            task_ids.append(task_id)
            
            # Update counters
            next_task_index += 1
            remaining_tasks -= 1
            
            # Wait before submitting next task
            time.sleep(task_interval)
    
    # Start task submission in separate thread
    submission_thread = threading.Thread(target=submit_remaining_tasks)
    submission_thread.daemon = True
    submission_thread.start()
    
    return task_ids

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run Integrated Distributed Testing System")
    
    # Coordinator settings
    coordinator_group = parser.add_argument_group("Coordinator Settings")
    coordinator_group.add_argument("--host", type=str, default="localhost", help="Host to bind to")
    coordinator_group.add_argument("--port", type=int, default=8080, help="Port to bind to")
    coordinator_group.add_argument("--db-path", type=str, default="./benchmark_db.duckdb", help="Path to DuckDB database")
    coordinator_group.add_argument("--heartbeat-timeout", type=int, default=60, help="Heartbeat timeout in seconds")
    coordinator_group.add_argument("--security-config", type=str, help="Path to security configuration JSON file")
    coordinator_group.add_argument("--visualization-path", type=str, help="Path for performance visualizations")
    
    # Load Balancer settings
    lb_group = parser.add_argument_group("Load Balancer Settings")
    lb_group.add_argument("--disable-load-balancer", action="store_true", help="Disable load balancer integration")
    lb_group.add_argument("--scheduler", type=str, default="performance_based", 
                         choices=["performance_based", "round_robin", "weighted_round_robin", "priority_based", "affinity_based", "composite"],
                         help="Load balancer scheduler type")
    lb_group.add_argument("--monitoring-interval", type=int, default=15, help="Monitoring interval in seconds")
    lb_group.add_argument("--rebalance-interval", type=int, default=90, help="Rebalance interval in seconds")
    lb_group.add_argument("--enable-work-stealing", action="store_true", help="Enable work stealing between workers")
    lb_group.add_argument("--worker-concurrency", type=int, default=2, help="Default worker concurrency setting")
    
    # Multi-Device Orchestrator settings
    orchestrator_group = parser.add_argument_group("Multi-Device Orchestrator Settings")
    orchestrator_group.add_argument("--disable-orchestrator", action="store_true", help="Disable multi-device orchestrator")
    orchestrator_group.add_argument("--orchestrator-strategy", type=str, default="auto", 
                                  choices=["auto", "data_parallel", "model_parallel", "pipeline_parallel", "ensemble", "function_parallel"],
                                  help="Default orchestration strategy")
    orchestrator_group.add_argument("--enable-distributed", action="store_true", help="Enable distributed execution across workers")
    orchestrator_group.add_argument("--max-split-workers", type=int, default=4, help="Maximum number of workers for splitting tasks")
    
    # Fault Tolerance settings
    ft_group = parser.add_argument_group("Fault Tolerance Settings")
    ft_group.add_argument("--disable-fault-tolerance", action="store_true", help="Disable fault tolerance system")
    ft_group.add_argument("--max-retries", type=int, default=3, help="Maximum number of retries for operations")
    ft_group.add_argument("--circuit-break-threshold", type=int, default=5, help="Number of errors before circuit breaking")
    ft_group.add_argument("--circuit-break-timeout", type=int, default=300, help="Timeout in seconds for circuit breaker reset")
    ft_group.add_argument("--error-window-size", type=int, default=100, help="Size of sliding window for error rate calculation")
    ft_group.add_argument("--error-rate-threshold", type=float, default=0.5, help="Threshold for error rate alerting")
    
    # Dashboard settings
    dashboard_group = parser.add_argument_group("Dashboard Settings")
    dashboard_group.add_argument("--disable-dashboard", action="store_true", help="Disable all monitoring dashboards")
    dashboard_group.add_argument("--dashboard-port", type=int, default=8888, help="Port for web monitoring dashboard")
    dashboard_group.add_argument("--metrics-db-path", type=str, help="Path to metrics database (defaults to <db-path>_metrics.duckdb)")
    dashboard_group.add_argument("--terminal-dashboard", action="store_true", help="Use terminal-based dashboard instead of web interface")
    dashboard_group.add_argument("--terminal-refresh", type=float, default=1.0, help="Terminal dashboard refresh interval in seconds")
    dashboard_group.add_argument("--open-browser", action="store_true", help="Open web browser to dashboard automatically")
    
    # Testing options
    testing_group = parser.add_argument_group("Testing Options")
    testing_group.add_argument("--mock-workers", type=int, help="Launch N mock workers for testing")
    testing_group.add_argument("--stress-test", action="store_true", help="Run a stress test after starting")
    testing_group.add_argument("--test-tasks", type=int, default=20, help="Number of tasks to create for stress test")
    testing_group.add_argument("--test-duration", type=int, default=60, help="Duration of stress test in seconds")
    testing_group.add_argument("--fault-injection", action="store_true", help="Enable fault injection for testing fault tolerance")
    testing_group.add_argument("--fault-rate", type=float, default=0.1, help="Rate of fault injection (0.0 to 1.0)")
    
    # High availability settings
    ha_group = parser.add_argument_group("High Availability Settings")
    ha_group.add_argument("--high-availability", action="store_true", help="Enable high availability mode")
    ha_group.add_argument("--coordinator-id", type=str, default=f"coordinator-{uuid.uuid4().hex[:8]}", help="Unique coordinator ID")
    ha_group.add_argument("--coordinator-addresses", type=str, help="Comma-separated list of other coordinator addresses")
    ha_group.add_argument("--auto-leader-election", action="store_true", help="Enable automatic leader election")
    
    # WebNN/WebGPU integration
    web_group = parser.add_argument_group("Web Platform Integration")
    web_group.add_argument("--enable-web-integration", action="store_true", help="Enable WebNN/WebGPU integration")
    web_group.add_argument("--browsers", type=str, default="chrome,firefox,edge", help="Comma-separated list of browsers to use")
    web_group.add_argument("--enable-web-optimizations", action="store_true", help="Enable web platform optimizations")
    
    # Advanced analytics
    analytics_group = parser.add_argument_group("Advanced Analytics")
    analytics_group.add_argument("--performance-analytics", action="store_true", help="Enable performance analytics")
    analytics_group.add_argument("--visualization-path", type=str, help="Path to store visualization files")
    analytics_group.add_argument("--enhanced-hardware-taxonomy", action="store_true", help="Enable enhanced hardware taxonomy")
    analytics_group.add_argument("--prediction-model", type=str, choices=["basic", "advanced"], help="Performance prediction model to use")
    
    # Advanced features
    advanced_group = parser.add_argument_group("Advanced Features")
    advanced_group.add_argument("--enable-result-aggregation", action="store_true", help="Enable advanced result aggregation")
    advanced_group.add_argument("--auto-recovery", action="store_true", help="Enable automatic recovery mechanisms")
    advanced_group.add_argument("--dynamic-resource-management", action="store_true", help="Enable dynamic resource management")
    advanced_group.add_argument("--health-monitoring", action="store_true", default=True, help="Enable health monitoring")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set default metrics DB path if not specified
    if not args.metrics_db_path and args.db_path:
        metrics_db_base = os.path.splitext(args.db_path)[0]
        args.metrics_db_path = f"{metrics_db_base}_metrics.duckdb"
    
    # Load or generate security configuration
    security_config = None
    security_config_path = args.security_config
    
    if security_config_path and os.path.exists(security_config_path):
        # Load existing configuration
        with open(security_config_path, 'r') as f:
            security_config = json.load(f)
        logger.info(f"Loaded security configuration from {security_config_path}")
    else:
        # Generate new configuration
        if not security_config_path:
            security_config_path = os.path.join(os.path.dirname(args.db_path), "security_config.json")
        
        security_config = generate_default_security_config(security_config_path)
        logger.info(f"Generated new security configuration at {security_config_path}")
    
    # Import the modules we need
    try:
        # Apply coordinator patches for integrations
        from data.duckdb.distributed_testing.coordinator_patch import apply_patches, remove_patches
        
        # Apply patches if load balancer is enabled
        if not args.disable_load_balancer:
            apply_patches()
            logger.info("Applied coordinator load balancer integration patches")
        
        # Import coordinator 
        from data.duckdb.distributed_testing.coordinator import CoordinatorServer
        
        # Import Multi-Device Orchestrator
        from data.duckdb.distributed_testing.multi_device_orchestrator import MultiDeviceOrchestrator
        from data.duckdb.distributed_testing.coordinator_orchestrator_integration import CoordinatorOrchestratorIntegration
        
        # Import Fault Tolerance System
        from data.duckdb.distributed_testing.fault_tolerance_system import FaultToleranceSystem
        
        # Import Comprehensive Monitoring Dashboard
        from data.duckdb.distributed_testing.comprehensive_monitoring_dashboard import ComprehensiveMonitoringDashboard
        from data.duckdb.distributed_testing.fault_tolerance_visualization import FaultToleranceVisualization
        
        # Import additional components based on options
        if args.high_availability:
            from data.duckdb.distributed_testing.auto_recovery import AutoRecoverySystem
            logger.info("Imported High Availability components")
            
        if args.enable_web_integration:
            from data.duckdb.distributed_testing.hardware_taxonomy import EnhancedHardwareTaxonomy
            from data.duckdb.distributed_testing.enhanced_hardware_detector import EnhancedHardwareDetector
            logger.info("Imported Web Platform Integration components")
            
        if args.performance_analytics:
            from data.duckdb.distributed_testing.performance_trend_analyzer import PerformanceTrendAnalyzer
            from data.duckdb.distributed_testing.resource_performance_predictor import ResourcePerformancePredictor
            logger.info("Imported Performance Analytics components")
            
        if args.dynamic_resource_management:
            from data.duckdb.distributed_testing.dynamic_resource_manager import DynamicResourceManager
            logger.info("Imported Dynamic Resource Management components")
            
        if args.enable_result_aggregation:
            from data.duckdb.distributed_testing.result_aggregator.service import ResultAggregatorService
            logger.info("Imported Result Aggregation components")
            
        if args.health_monitoring:
            from data.duckdb.distributed_testing.health_monitor import HealthMonitor
            logger.info("Imported Health Monitoring components")
        
    except ImportError as e:
        logger.error(f"Failed to import required modules: {e}")
        logger.error("Make sure all required components are installed.")
        sys.exit(1)
    
    # Load balancer configuration
    load_balancer_config = {
        "db_path": args.db_path,
        "monitoring_interval": args.monitoring_interval,
        "rebalance_interval": args.rebalance_interval,
        "default_scheduler": {
            "type": args.scheduler
        },
        "worker_concurrency": args.worker_concurrency,
        "enable_work_stealing": args.enable_work_stealing,
        "test_type_schedulers": {
            "performance": {"type": "performance_based"},
            "compatibility": {"type": "affinity_based"},
            "integration": {
                "type": "composite",
                "algorithms": [
                    {"type": "performance_based", "weight": 0.7},
                    {"type": "priority_based", "weight": 0.3}
                ]
            }
        }
    }
    
    # Configure test type to scheduler mapping based on model family
    load_balancer_config["model_family_schedulers"] = {
        "vision": {"type": "performance_based"},
        "text": {"type": "weighted_round_robin"},
        "audio": {"type": "affinity_based"},
        "multimodal": {
            "type": "composite",
            "algorithms": [
                {"type": "performance_based", "weight": 0.6},
                {"type": "affinity_based", "weight": 0.4}
            ]
        }
    }
    
    # Create and start coordinator with all components
    try:
        # Ensure the database directory exists
        db_dir = os.path.dirname(os.path.abspath(args.db_path))
        os.makedirs(db_dir, exist_ok=True)
        
        # Create additional components based on args
        enhanced_hardware_detector = None
        hardware_taxonomy = None
        dynamic_resource_manager = None
        health_monitor = None
        performance_analyzer = None
        result_aggregator = None
        auto_recovery = None
        
        # Initialize enhanced hardware taxonomy if enabled
        if args.enhanced_hardware_taxonomy:
            hardware_taxonomy = EnhancedHardwareTaxonomy()
            logger.info("Enhanced Hardware Taxonomy initialized")
            
            # Initialize hardware detector if enabled
            if args.enable_web_integration:
                enhanced_hardware_detector = EnhancedHardwareDetector(
                    browsers=args.browsers.split(',') if args.browsers else ["chrome", "firefox", "edge"],
                    enable_optimizations=args.enable_web_optimizations
                )
                logger.info(f"Enhanced Hardware Detector initialized with browsers: {args.browsers}")
        
        # Initialize dynamic resource manager if enabled
        if args.dynamic_resource_management:
            dynamic_resource_manager = DynamicResourceManager(
                db_path=args.db_path,
                hardware_taxonomy=hardware_taxonomy
            )
            logger.info("Dynamic Resource Manager initialized")
            
        # Initialize health monitor if enabled
        if args.health_monitoring:
            health_monitor = HealthMonitor()
            logger.info("Health Monitor initialized")
            
        # Initialize performance analyzer if enabled
        if args.performance_analytics:
            performance_analyzer = PerformanceTrendAnalyzer(
                db_path=args.db_path,
                visualization_path=args.visualization_path
            )
            
            # Initialize resource performance predictor if prediction model specified
            if args.prediction_model:
                resource_predictor = ResourcePerformancePredictor(
                    model_type=args.prediction_model,
                    db_path=args.db_path
                )
                performance_analyzer.set_predictor(resource_predictor)
                logger.info(f"Performance Analyzer initialized with {args.prediction_model} prediction model")
            else:
                logger.info("Performance Analyzer initialized without prediction model")
        
        # Initialize result aggregator if enabled
        if args.enable_result_aggregation:
            result_aggregator = ResultAggregatorService(
                db_path=args.db_path
            )
            logger.info("Result Aggregator initialized")
            
        # Initialize auto recovery system if high availability is enabled
        if args.high_availability:
            auto_recovery = AutoRecoverySystem(
                coordinator_id=args.coordinator_id,
                coordinator_addresses=args.coordinator_addresses.split(',') if args.coordinator_addresses else None,
                db_path=args.db_path,
                auto_leader_election=args.auto_leader_election,
                visualization_path=args.visualization_path
            )
            logger.info(f"Auto Recovery System initialized with coordinator ID: {args.coordinator_id}")
        
        # Create coordinator with all components
        coordinator = CoordinatorServer(
            host=args.host,
            port=args.port,
            db_path=args.db_path,
            heartbeat_timeout=args.heartbeat_timeout,
            visualization_path=args.visualization_path,
            performance_analyzer=True if args.performance_analytics else False,
            enable_load_balancer=not args.disable_load_balancer,
            load_balancer_config=load_balancer_config,
            token_secret=security_config["token_secret"],
            hardware_taxonomy=hardware_taxonomy,
            dynamic_resource_manager=dynamic_resource_manager,
            health_monitor=health_monitor,
            result_aggregator=result_aggregator,
            auto_recovery=auto_recovery
        )
        
        # Create and attach the multi-device orchestrator if enabled
        if not args.disable_orchestrator:
            orchestrator = MultiDeviceOrchestrator(
                coordinator=coordinator,
                task_manager=getattr(coordinator, 'task_manager', None),
                worker_manager=getattr(coordinator, 'worker_manager', None),
                resource_manager=dynamic_resource_manager,
                default_strategy=args.orchestrator_strategy,
                enable_distributed=args.enable_distributed,
                max_workers=args.max_split_workers
            )
            
            # Create orchestrator integration
            orchestrator_integration = CoordinatorOrchestratorIntegration(coordinator)
            
            # Attach orchestrator and integration to coordinator
            coordinator.multi_device_orchestrator = orchestrator
            coordinator.orchestrator_integration = orchestrator_integration
            logger.info(f"Multi-Device Orchestrator initialized with {args.orchestrator_strategy} strategy")
        
        # Create and attach the fault tolerance system if enabled
        if not args.disable_fault_tolerance:
            fault_tolerance_system = FaultToleranceSystem(
                coordinator=coordinator,
                task_manager=getattr(coordinator, 'task_manager', None),
                worker_manager=getattr(coordinator, 'worker_manager', None),
                max_retries=args.max_retries,
                circuit_break_threshold=args.circuit_break_threshold,
                circuit_break_timeout=args.circuit_break_timeout,
                error_window_size=args.error_window_size,
                error_rate_threshold=args.error_rate_threshold
            )
            
            # Create fault tolerance visualization if monitoring dashboard is enabled
            if not args.disable_dashboard:
                fault_tolerance_visualization = FaultToleranceVisualization(
                    fault_tolerance_system=fault_tolerance_system
                )
                coordinator.fault_tolerance_visualization = fault_tolerance_visualization
            
            # Attach fault tolerance system to coordinator
            coordinator.fault_tolerance_system = fault_tolerance_system
            logger.info("Fault Tolerance System initialized")
        
        # Start coordinator in a separate thread
        coordinator_thread = threading.Thread(target=lambda: anyio.run(coordinator.start()))
        coordinator_thread.daemon = True
        coordinator_thread.start()
        
        # Wait for coordinator to initialize
        time.sleep(2)
        
        logger.info(f"Coordinator started on ws://{args.host}:{args.port} with:")
        logger.info(f"- Load Balancer: {'enabled' if not args.disable_load_balancer else 'disabled'}")
        logger.info(f"- Multi-Device Orchestrator: {'enabled' if not args.disable_orchestrator else 'disabled'}")
        logger.info(f"- Fault Tolerance System: {'enabled' if not args.disable_fault_tolerance else 'disabled'}")
        logger.info(f"Admin API Key: {security_config['api_keys']['admin']}")
        logger.info(f"Worker API Key: {security_config['api_keys']['worker']}")
        logger.info(f"User API Key: {security_config['api_keys']['user']}")
        
        # Start monitoring dashboard if enabled
        dashboard_integration = None
        terminal_dashboard = None
        comprehensive_dashboard = None
        
        if not args.disable_dashboard:
            if args.terminal_dashboard:
                # Start terminal-based dashboard
                terminal_dashboard = start_terminal_dashboard(
                    coordinator=coordinator, 
                    load_balancer=coordinator.load_balancer if hasattr(coordinator, 'load_balancer') else None,
                    refresh_interval=args.terminal_refresh
                )
                
                if terminal_dashboard:
                    logger.info(f"Terminal dashboard started with refresh interval of {args.terminal_refresh}s")
            else:
                # Start comprehensive monitoring dashboard
                try:
                    # Configure dashboard components
                    dashboard_components = {
                        "fault_tolerance_visualization": getattr(coordinator, "fault_tolerance_visualization", None),
                        "performance_analyzer": performance_analyzer
                    }
                    
                    # Add result aggregator if enabled
                    if args.enable_result_aggregation:
                        dashboard_components["result_aggregator"] = result_aggregator
                    
                    # Add health monitor if enabled
                    if args.health_monitoring:
                        dashboard_components["health_monitor"] = health_monitor
                    
                    # Add dynamic resource manager if enabled
                    if args.dynamic_resource_management:
                        dashboard_components["dynamic_resource_manager"] = dynamic_resource_manager
                    
                    # Create comprehensive dashboard with all components
                    comprehensive_dashboard = ComprehensiveMonitoringDashboard(
                        coordinator=coordinator,
                        port=args.dashboard_port,
                        coordinator_url=f"ws://{args.host}:{args.port}",
                        db_path=args.metrics_db_path,
                        static_path=None,
                        template_path=None,
                        debug=args.debug if hasattr(args, 'debug') else False,
                        **dashboard_components
                    )
                    
                    # Register custom visualizations based on enabled features
                    if args.performance_analytics:
                        # Register performance trend visualization
                        comprehensive_dashboard.register_visualization(
                            "performance_trends", 
                            "Performance Trends", 
                            performance_analyzer.generate_trend_visualization
                        )
                        
                        # Register performance prediction visualization if enabled
                        if args.prediction_model:
                            comprehensive_dashboard.register_visualization(
                                "performance_predictions",
                                "Performance Predictions",
                                performance_analyzer.generate_prediction_visualization
                            )
                    
                    # Register web integration visualization if enabled
                    if args.enable_web_integration and enhanced_hardware_detector:
                        comprehensive_dashboard.register_visualization(
                            "web_capabilities",
                            "Web Browser Capabilities",
                            enhanced_hardware_detector.generate_capabilities_visualization
                        )
                    
                    # Register enhanced hardware visualization if enabled
                    if args.enhanced_hardware_taxonomy and hardware_taxonomy:
                        comprehensive_dashboard.register_visualization(
                            "hardware_taxonomy",
                            "Hardware Taxonomy",
                            hardware_taxonomy.generate_taxonomy_visualization
                        )
                    
                    # Start dashboard in a separate thread
                    dashboard_thread = threading.Thread(target=comprehensive_dashboard.start)
                    dashboard_thread.daemon = True
                    dashboard_thread.start()
                    
                    logger.info(f"Comprehensive Monitoring Dashboard started at http://{args.host}:{args.dashboard_port}/")
                    
                    # Open browser if requested
                    if args.open_browser:
                        url = f"http://{args.host}:{args.dashboard_port}/"
                        threading.Timer(2.0, lambda: webbrowser.open(url)).start()
                
                except Exception as e:
                    logger.error(f"Failed to start comprehensive dashboard: {e}")
                    logger.error("Make sure you have all required packages installed:")
                    logger.error("pip install tornado websockets plotly pandas numpy")
                    
                    # Fall back to basic monitoring dashboard if available
                    try:
                        # Import monitoring integration
                        from data.duckdb.distributed_testing.load_balancer.monitoring.integration import MonitoringIntegration
                        
                        # Create monitoring integration
                        dashboard_integration = MonitoringIntegration(
                            coordinator=coordinator,
                            load_balancer=coordinator.load_balancer if hasattr(coordinator, 'load_balancer') else None,
                            db_path=args.metrics_db_path,
                            dashboard_host=args.host,
                            dashboard_port=args.dashboard_port,
                            collection_interval=1.0  # Collect metrics every second
                        )
                        
                        # Start monitoring
                        dashboard_integration.start()
                        
                        logger.info(f"Basic Monitoring Dashboard started at http://{args.host}:{args.dashboard_port}/")
                        
                        # Open browser if requested
                        if args.open_browser:
                            url = f"http://{args.host}:{args.dashboard_port}/"
                            threading.Timer(2.0, lambda: webbrowser.open(url)).start()
                    
                    except ImportError as e:
                        logger.error(f"Failed to start basic monitoring dashboard: {e}")
                        logger.error("Make sure you have Flask and Flask-SocketIO installed:")
                        logger.error("pip install flask flask-cors flask-socketio")
        
        # Start mock workers if requested
        mock_workers = []
        temp_dir = None
        
        if args.mock_workers and args.mock_workers > 0:
            coordinator_url = f"ws://{args.host}:{args.port}"
            mock_workers, temp_dir = launch_mock_workers(
                coordinator_url=coordinator_url,
                api_key=security_config["api_keys"]["worker"],
                count=args.mock_workers
            )
            
            logger.info(f"Started {len(mock_workers)} mock workers")
            
            # Wait for workers to register
            time.sleep(5)
        
        # Run stress test if requested
        stress_test_tasks = []
        if args.stress_test:
            stress_test_tasks = run_stress_test(
                coordinator=coordinator,
                task_count=args.test_tasks,
                duration=args.test_duration,
                fault_injection=args.fault_injection,
                fault_rate=args.fault_rate
            )
            
            logger.info(f"Started stress test with {len(stress_test_tasks)} tasks " +
                       f"{'with fault injection' if args.fault_injection else 'without fault injection'}")
        
        # Keep running until interrupted
        try:
            logger.info("Press Ctrl+C to stop the integrated system")
            
            while True:
                time.sleep(1)
        
        except KeyboardInterrupt:
            logger.info("Shutting down...")
            
            # Stop comprehensive dashboard if running
            if comprehensive_dashboard:
                logger.info("Stopping comprehensive dashboard...")
                comprehensive_dashboard.stop()
            
            # Stop basic dashboard if running
            if dashboard_integration:
                logger.info("Stopping basic dashboard...")
                dashboard_integration.stop()
            
            if terminal_dashboard:
                logger.info("Stopping terminal dashboard...")
                terminal_dashboard.stop()
            
            # Stop result aggregator if running
            if result_aggregator:
                logger.info("Stopping result aggregator...")
                if hasattr(result_aggregator, 'stop'):
                    result_aggregator.stop()
                
            # Stop performance analyzer if running
            if performance_analyzer:
                logger.info("Stopping performance analyzer...")
                if hasattr(performance_analyzer, 'stop'):
                    performance_analyzer.stop()
            
            # Stop health monitor if running
            if health_monitor:
                logger.info("Stopping health monitor...")
                if hasattr(health_monitor, 'stop'):
                    health_monitor.stop()
            
            # Stop dynamic resource manager if running
            if dynamic_resource_manager:
                logger.info("Stopping dynamic resource manager...")
                if hasattr(dynamic_resource_manager, 'stop'):
                    dynamic_resource_manager.stop()
            
            # Stop enhanced hardware detector if running
            if enhanced_hardware_detector:
                logger.info("Stopping enhanced hardware detector...")
                if hasattr(enhanced_hardware_detector, 'stop'):
                    enhanced_hardware_detector.stop()
            
            # Stop auto recovery system if running
            if auto_recovery:
                logger.info("Stopping auto recovery system...")
                if hasattr(auto_recovery, 'stop'):
                    auto_recovery.stop()
            
            # Stop fault tolerance system if running
            if not args.disable_fault_tolerance:
                logger.info("Stopping fault tolerance system...")
                if hasattr(coordinator, 'fault_tolerance_system'):
                    coordinator.fault_tolerance_system.stop()
            
            # Stop orchestrator integration if running
            if hasattr(coordinator, 'orchestrator_integration'):
                logger.info("Stopping orchestrator integration...")
                coordinator.orchestrator_integration.stop()
            
            # Terminate mock workers
            if mock_workers:
                logger.info(f"Terminating {len(mock_workers)} mock workers...")
                for worker in mock_workers:
                    worker["process"].terminate()
                
                # Wait for workers to terminate
                for worker in mock_workers:
                    worker["process"].wait()
                
                # Clean up temporary directory
                if temp_dir and os.path.exists(temp_dir):
                    import shutil
                    shutil.rmtree(temp_dir)
            
            # Stop coordinator
            logger.info("Stopping coordinator...")
            anyio.run(coordinator.stop())
            
            # Remove patches if applied
            if not args.disable_load_balancer:
                try:
                    remove_patches()
                    logger.info("Removed coordinator patches")
                except Exception as e:
                    logger.warning(f"Error removing patches: {e}")
            
            logger.info("Shutdown complete")
            
    except Exception as e:
        logger.error(f"Error running integrated system: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
        # Remove patches if error occurs
        if not args.disable_load_balancer:
            remove_patches()

if __name__ == "__main__":
    main()