"""
Distributed Testing Framework

A high-performance distributed testing system that enables parallel execution of
benchmarks and tests across multiple machines with heterogeneous hardware. This
framework provides intelligent workload distribution and centralized result aggregation.

Components:
- coordinator.py: Coordinator server for task distribution
- worker.py: Worker node for task execution
- task_scheduler.py: Intelligent task scheduling and distribution
- load_balancer.py: Adaptive load balancing system
- health_monitor.py: Worker health monitoring and recovery
- dashboard_server.py: Web-based dashboard for monitoring
- run_test.py: CLI for running tests with the distributed framework

Implementation Status:
- Phase 1: Core Infrastructure (COMPLETED)
- Phase 2: Security and Worker Management (COMPLETED)
- Phase 3: Intelligent Task Distribution (IN PROGRESS)
- Phase 4: Adaptive Load Balancing (IN PROGRESS)
- Phase 5: Fault Tolerance (PLANNED)
- Phase 6: Monitoring Dashboard (PLANNED)

Target completion: June 20, 2025
"""

__version__ = "0.1.0"