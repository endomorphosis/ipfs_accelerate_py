#!/usr/bin/env python3
"""
Parallel Test Execution Runner for Distributed Testing Framework

This script demonstrates the usage of the Test Dependency Manager and Execution Orchestrator
for running tests in a distributed manner with dependency management.
"""

import argparse
import logging
import time
import json
import sys
import os
from typing import Dict, List, Set, Optional, Any

# Import test dependency manager
from test_dependency_manager import (
    TestDependencyManager, Dependency, DependencyType
)

# Import execution orchestrator
from execution_orchestrator import (
    ExecutionOrchestrator, ExecutionStrategy
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("parallel_execution.log")
    ]
)
logger = logging.getLogger("parallel_execution_runner")


def create_test_dependency_graph(
        num_tests: int = 20,
        dependency_probability: float = 0.3,
        group_count: int = 3,
        group_assignment_probability: float = 0.5
    ) -> TestDependencyManager:
    """
    Create a test dependency graph with simulated tests.
    
    Args:
        num_tests: Number of tests to create
        dependency_probability: Probability of a test depending on another test
        group_count: Number of groups to create
        group_assignment_probability: Probability of a test being assigned to a group
        
    Returns:
        TestDependencyManager with simulated tests
    """
    import random
    
    # Create dependency manager
    dependency_manager = TestDependencyManager()
    
    # Create groups
    groups = [f"group_{i}" for i in range(group_count)]
    
    # Create tests
    for i in range(num_tests):
        test_id = f"test_{i}"
        
        # Create dependencies on previous tests with some probability
        dependencies = []
        for j in range(i):
            if random.random() < dependency_probability:
                # 20% chance of soft dependency, 10% chance of optional dependency
                dependency_type_rand = random.random()
                if dependency_type_rand < 0.2:
                    dependency_type = DependencyType.SOFT
                elif dependency_type_rand < 0.3:
                    dependency_type = DependencyType.OPTIONAL
                else:
                    dependency_type = DependencyType.HARD
                
                # 10% chance of group dependency
                if random.random() < 0.1 and groups:
                    dependencies.append(Dependency(
                        random.choice(groups),
                        dependency_type=dependency_type,
                        is_group=True
                    ))
                else:
                    dependencies.append(Dependency(
                        f"test_{j}",
                        dependency_type=dependency_type
                    ))
        
        # Assign to groups with some probability
        test_groups = []
        for group in groups:
            if random.random() < group_assignment_probability:
                test_groups.append(group)
        
        # Register test
        dependency_manager.register_test(test_id, dependencies, test_groups)
    
    return dependency_manager


def pre_execution_hook(max_workers: int, strategy: ExecutionStrategy, total_tests: int) -> None:
    """Hook called before execution starts."""
    logger.info(f"Starting execution with {max_workers} workers, {strategy.name} strategy, "
               f"and {total_tests} total tests")


def post_execution_hook(metrics: Dict[str, Any]) -> None:
    """Hook called after execution completes."""
    logger.info(f"Execution metrics: {json.dumps(metrics, indent=2)}")


def test_start_hook(test_id: str, worker_id: str) -> None:
    """Hook called when a test starts."""
    logger.debug(f"Test {test_id} started on worker {worker_id}")


def test_end_hook(test_id: str, status: Any, result: Any, error: Optional[str]) -> None:
    """Hook called when a test ends."""
    if error:
        logger.debug(f"Test {test_id} ended with status {status.name} and error: {error}")
    else:
        logger.debug(f"Test {test_id} ended with status {status.name}")


def run_parallel_execution(args: argparse.Namespace) -> None:
    """
    Run parallel test execution with the specified arguments.
    
    Args:
        args: Command-line arguments
    """
    logger.info(f"Running parallel execution with strategy: {args.strategy}")
    
    # Create test dependency graph
    if args.dependency_file:
        # Load dependencies from file
        logger.info(f"Loading dependencies from {args.dependency_file}")
        try:
            with open(args.dependency_file, 'r') as f:
                dependency_data = json.load(f)
            
            dependency_manager = TestDependencyManager()
            dependency_manager.import_dependency_data(dependency_data)
        except Exception as e:
            logger.error(f"Error loading dependencies from file: {str(e)}")
            sys.exit(1)
    else:
        # Create simulated dependencies
        logger.info("Creating simulated test dependency graph")
        dependency_manager = create_test_dependency_graph(
            num_tests=args.num_tests,
            dependency_probability=args.dependency_probability,
            group_count=args.group_count,
            group_assignment_probability=args.group_probability
        )
    
    # Validate dependencies
    logger.info("Validating dependencies")
    is_valid, errors = dependency_manager.validate_dependencies()
    if not is_valid:
        logger.error(f"Invalid dependencies: {len(errors)} errors")
        for error in errors[:5]:  # Show first 5 errors
            logger.error(f"  - {error}")
        sys.exit(1)
    
    # Create execution orchestrator
    logger.info(f"Creating execution orchestrator with {args.max_workers} workers")
    orchestrator = ExecutionOrchestrator(
        dependency_manager=dependency_manager,
        max_workers=args.max_workers,
        strategy=ExecutionStrategy(args.strategy),
        timeout_seconds=args.timeout
    )
    
    # Set hooks
    if not args.no_hooks:
        orchestrator.set_pre_execution_hook(pre_execution_hook)
        orchestrator.set_post_execution_hook(post_execution_hook)
        orchestrator.set_test_start_hook(test_start_hook)
        orchestrator.set_test_end_hook(test_end_hook)
    
    # Visualize execution plan if requested
    if args.visualize:
        plan = orchestrator.visualize_execution_plan(output_format=args.visualize_format)
        
        if args.visualize_output:
            # Write to file
            with open(args.visualize_output, 'w') as f:
                f.write(plan)
        else:
            # Print to console
            print("\nExecution Plan:")
            print(plan)
    
    # Run execution
    logger.info("Starting test execution")
    start_time = time.time()
    
    if args.async_execution:
        # Run asynchronously
        results = anyio.run(orchestrator.execute_all_tests_async)
    else:
        # Run synchronously
        results = orchestrator.execute_all_tests()
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Print results
    logger.info(f"Execution completed in {elapsed_time:.2f} seconds")
    logger.info(f"Total Tests: {results['metrics']['total_tests']}")
    logger.info(f"Completed: {results['metrics']['completed_tests']}")
    logger.info(f"Failed: {results['metrics']['failed_tests']}")
    logger.info(f"Skipped: {results['metrics']['skipped_tests']}")
    logger.info(f"Max Parallelism: {results['metrics']['max_parallelism']}")
    
    # Save results if requested
    if args.output:
        logger.info(f"Saving results to {args.output}")
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)


def main() -> None:
    """Parse command-line arguments and run script."""
    parser = argparse.ArgumentParser(
        description='Run parallel test execution with dependency management.'
    )
    
    # Test creation options
    test_group = parser.add_argument_group('Test Creation Options')
    test_group.add_argument('--dependency-file', type=str,
                          help='Path to a JSON file with dependency data')
    test_group.add_argument('--num-tests', type=int, default=20,
                          help='Number of simulated tests to create (default: 20)')
    test_group.add_argument('--dependency-probability', type=float, default=0.3,
                          help='Probability of a test depending on another test (default: 0.3)')
    test_group.add_argument('--group-count', type=int, default=3,
                          help='Number of groups to create (default: 3)')
    test_group.add_argument('--group-probability', type=float, default=0.5,
                          help='Probability of a test being assigned to a group (default: 0.5)')
    
    # Execution options
    exec_group = parser.add_argument_group('Execution Options')
    exec_group.add_argument('--max-workers', type=int, default=0,
                          help='Maximum number of worker threads (default: auto)')
    exec_group.add_argument('--strategy', type=str, default='resource_aware',
                          choices=[s.value for s in ExecutionStrategy],
                          help='Execution strategy to use (default: resource_aware)')
    exec_group.add_argument('--timeout', type=int, default=None,
                          help='Execution timeout in seconds (default: None)')
    exec_group.add_argument('--async-execution', action='store_true',
                          help='Use asynchronous execution with AnyIO')
    exec_group.add_argument('--no-hooks', action='store_true',
                          help='Disable execution hooks')
    
    # Output options
    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument('--output', type=str,
                            help='Path to save execution results as JSON')
    output_group.add_argument('--visualize', action='store_true',
                            help='Visualize the execution plan')
    output_group.add_argument('--visualize-format', type=str, default='text',
                            choices=['text', 'json', 'mermaid'],
                            help='Format for visualization (default: text)')
    output_group.add_argument('--visualize-output', type=str,
                            help='Path to save visualization (default: print to console)')
    output_group.add_argument('--log-level', type=str, default='INFO',
                            choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                            help='Logging level (default: INFO)')
    
    args = parser.parse_args()
    
    # Set log level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Run parallel execution
    run_parallel_execution(args)


if __name__ == '__main__':
    main()