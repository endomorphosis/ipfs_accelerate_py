#!/usr/bin/env python3
"""
Load Balancer Integration with Hardware-Aware Workload Management

This module provides utility functions to integrate the Load Balancer component 
with the Hardware-Aware Workload Management system for more intelligent scheduling.
"""

import logging
from typing import Optional, Dict, Any, List, Set, Tuple

# Import load balancer components
from duckdb_api.distributed_testing.load_balancer.service import LoadBalancerService
from duckdb_api.distributed_testing.load_balancer.scheduling_algorithms import (
    SchedulingAlgorithm, CompositeScheduler, AdaptiveScheduler
)

# Import hardware workload management components
from distributed_testing.hardware_workload_management import (
    HardwareWorkloadManager, HardwareTaxonomy
)

# Import hardware-aware scheduler
from distributed_testing.hardware_aware_scheduler import HardwareAwareScheduler

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("load_balancer_integration")


def create_hardware_aware_load_balancer(
        hardware_taxonomy: Optional[HardwareTaxonomy] = None,
        hardware_workload_manager: Optional[HardwareWorkloadManager] = None,
        db_path: Optional[str] = None,
        use_composite: bool = False,
        hardware_scheduler_weight: float = 0.7
    ) -> Tuple[LoadBalancerService, HardwareWorkloadManager, HardwareAwareScheduler]:
    """
    Create a load balancer with hardware-aware scheduling capabilities.
    
    This function creates and configures a load balancer service that uses the
    Hardware-Aware Workload Management system for more intelligent scheduling.
    
    Args:
        hardware_taxonomy: Optional hardware taxonomy (created if not provided)
        hardware_workload_manager: Optional workload manager (created if not provided)
        db_path: Optional path to database for performance tracking
        use_composite: Whether to use composite scheduler with hardware-aware and adaptive schedulers
        hardware_scheduler_weight: Weight for hardware-aware scheduler in composite (0.0-1.0)
        
    Returns:
        Tuple of (load_balancer, workload_manager, hardware_aware_scheduler)
    """
    # Create hardware taxonomy if not provided
    if hardware_taxonomy is None:
        hardware_taxonomy = HardwareTaxonomy()
    
    # Create hardware workload manager if not provided
    if hardware_workload_manager is None:
        hardware_workload_manager = HardwareWorkloadManager(hardware_taxonomy, db_path=db_path)
    
    # Create hardware-aware scheduler
    hardware_aware_scheduler = HardwareAwareScheduler(hardware_workload_manager, hardware_taxonomy)
    
    # Create load balancer service
    load_balancer = LoadBalancerService(db_path=db_path)
    
    # Configure scheduler
    if use_composite:
        # Use composite scheduler with hardware-aware and adaptive schedulers
        adaptive_scheduler = AdaptiveScheduler()
        
        composite = CompositeScheduler([
            (hardware_aware_scheduler, hardware_scheduler_weight),
            (adaptive_scheduler, 1.0 - hardware_scheduler_weight)
        ])
        
        load_balancer.default_scheduler = composite
    else:
        # Use hardware-aware scheduler directly
        load_balancer.default_scheduler = hardware_aware_scheduler
    
    # Start workload manager
    hardware_workload_manager.start()
    
    return load_balancer, hardware_workload_manager, hardware_aware_scheduler


def register_type_specific_schedulers(
        load_balancer: LoadBalancerService,
        hardware_aware_scheduler: HardwareAwareScheduler,
        type_scheduler_map: Dict[str, SchedulingAlgorithm]
    ) -> None:
    """
    Register type-specific schedulers for the load balancer.
    
    Args:
        load_balancer: Load balancer service
        hardware_aware_scheduler: Hardware-aware scheduler for default scheduling
        type_scheduler_map: Mapping of test types to specialized schedulers
    """
    # Set hardware-aware scheduler as default
    load_balancer.default_scheduler = hardware_aware_scheduler
    
    # Register type-specific schedulers
    for test_type, scheduler in type_scheduler_map.items():
        load_balancer.test_type_schedulers[test_type] = scheduler
        
    logger.info(f"Registered {len(type_scheduler_map)} type-specific schedulers")


def shutdown_integration(
        load_balancer: LoadBalancerService,
        workload_manager: HardwareWorkloadManager
    ) -> None:
    """
    Properly shut down the load balancer and workload manager.
    
    Args:
        load_balancer: Load balancer service to shut down
        workload_manager: Workload manager to shut down
    """
    # Stop load balancer
    load_balancer.stop()
    
    # Stop workload manager
    workload_manager.stop()
    
    logger.info("Load balancer and workload manager stopped")