#!/usr/bin/env python3
"""
Test Dependency Manager for Distributed Testing Framework

This module provides a comprehensive system for managing test dependencies in a distributed
testing environment. It enables dependency tracking, resolution, validation, and execution
ordering based on the dependencies between tests.

The dependency manager supports:
- Direct dependencies between tests
- Dependency groups and tags for flexible dependency specification
- Dependency validation including cycle detection
- Execution order generation based on dependencies
- Parallel execution planning with dependency constraints
"""

import logging
import time
from enum import Enum
from typing import Dict, List, Set, Optional, Tuple, Any, Union, DefaultDict
from collections import defaultdict, deque
import networkx as nx
from dataclasses import dataclass, field

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("test_dependency_manager")


class DependencyType(Enum):
    """Types of dependencies between tests."""
    HARD = "hard"  # Test cannot run until dependency completes successfully
    SOFT = "soft"  # Test can run if dependency fails, but not until it completes
    OPTIONAL = "optional"  # Test can run even if dependency is not scheduled


class DependencyStatus(Enum):
    """Status of a dependency resolution."""
    SATISFIED = "satisfied"  # Dependency is satisfied and test can run
    PENDING = "pending"  # Dependency has not completed yet
    FAILED = "failed"  # Dependency failed, test cannot run (for HARD dependencies)
    IGNORED = "ignored"  # Dependency is ignored (for OPTIONAL dependencies)


@dataclass
class Dependency:
    """Represents a dependency between tests."""
    dependency_id: str  # ID of the dependency (test ID or group ID)
    dependency_type: DependencyType = DependencyType.HARD
    timeout_seconds: Optional[int] = None  # Timeout for waiting for dependency
    is_group: bool = False  # Whether this is a dependency on a group/tag
    custom_properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TestDependencyInfo:
    """Dependency information for a test."""
    test_id: str
    dependencies: List[Dependency] = field(default_factory=list)
    dependents: List[str] = field(default_factory=list)  # Tests that depend on this test
    group_memberships: List[str] = field(default_factory=list)  # Groups this test belongs to


class TestDependencyManager:
    """
    Manages dependencies between distributed tests.
    
    This class is responsible for tracking dependencies, validating them,
    resolving them, and generating execution orders based on them.
    """
    
    def __init__(self):
        """Initialize the dependency manager."""
        # Maps test IDs to dependency information
        self.test_dependencies: Dict[str, TestDependencyInfo] = {}
        
        # Maps group IDs to sets of test IDs
        self.groups: Dict[str, Set[str]] = defaultdict(set)
        
        # Dependency graph
        self.graph = nx.DiGraph()
        
        # Test statuses
        self.test_statuses: Dict[str, Dict[str, Any]] = {}
        
        # Dependency resolution cache
        self.resolution_cache: Dict[str, Dict[str, DependencyStatus]] = {}
        
        # Track whether validation has been performed
        self.is_validated = False
        
        logger.info("Test Dependency Manager initialized")
    
    def register_test(self, test_id: str, dependencies: List[Dependency] = None, 
                     groups: List[str] = None) -> None:
        """
        Register a test with its dependencies and group memberships.
        
        Args:
            test_id: Unique identifier for the test
            dependencies: List of dependencies for this test
            groups: List of groups/tags this test belongs to
        """
        if test_id in self.test_dependencies:
            logger.warning(f"Test {test_id} is already registered. Updating dependencies.")
        
        # Create or update dependency info
        if test_id not in self.test_dependencies:
            self.test_dependencies[test_id] = TestDependencyInfo(test_id=test_id)
        
        # Add test to graph
        if not self.graph.has_node(test_id):
            self.graph.add_node(test_id)
        
        # Update dependencies
        if dependencies:
            self.test_dependencies[test_id].dependencies = dependencies
            
            # Update dependency graph
            for dep in dependencies:
                # For direct dependencies (not groups)
                if not dep.is_group:
                    # Add edge from dependency to this test
                    self.graph.add_edge(dep.dependency_id, test_id)
                    
                    # Register the dependency test if it doesn't exist
                    if dep.dependency_id not in self.test_dependencies:
                        self.test_dependencies[dep.dependency_id] = TestDependencyInfo(
                            test_id=dep.dependency_id
                        )
                    
                    # Add this test as a dependent
                    if test_id not in self.test_dependencies[dep.dependency_id].dependents:
                        self.test_dependencies[dep.dependency_id].dependents.append(test_id)
        
        # Update group memberships
        if groups:
            self.test_dependencies[test_id].group_memberships = groups
            
            # Add test to groups
            for group in groups:
                self.groups[group].add(test_id)
        
        # Invalidate validation since we've changed the dependency graph
        self.is_validated = False
        
        logger.debug(f"Registered test {test_id} with {len(dependencies or [])} dependencies "
                    f"and {len(groups or [])} group memberships")
    
    def register_test_batch(self, tests: List[Dict[str, Any]]) -> None:
        """
        Register multiple tests at once.
        
        Args:
            tests: List of test dictionaries, each containing:
                  - test_id: Unique identifier for the test
                  - dependencies: (Optional) List of dependencies
                  - groups: (Optional) List of groups/tags
        """
        for test in tests:
            self.register_test(
                test_id=test["test_id"],
                dependencies=test.get("dependencies"),
                groups=test.get("groups")
            )
        
        logger.info(f"Registered {len(tests)} tests in batch")
    
    def validate_dependencies(self) -> Tuple[bool, List[str]]:
        """
        Validate all registered dependencies.
        
        Checks for:
        - Cyclic dependencies
        - Missing dependency tests
        - Invalid group references
        
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Check for cycles in the dependency graph
        try:
            cycles = list(nx.simple_cycles(self.graph))
            if cycles:
                for cycle in cycles:
                    cycle_str = " -> ".join(cycle) + " -> " + cycle[0]
                    errors.append(f"Cyclic dependency detected: {cycle_str}")
        except nx.NetworkXNoCycle:
            # No cycles found
            pass
        
        # Check for dependencies on non-existent tests
        for test_id, info in self.test_dependencies.items():
            for dep in info.dependencies:
                if not dep.is_group and dep.dependency_id not in self.test_dependencies:
                    errors.append(f"Test {test_id} depends on non-existent test {dep.dependency_id}")
        
        # Check for dependencies on non-existent groups
        for test_id, info in self.test_dependencies.items():
            for dep in info.dependencies:
                if dep.is_group and dep.dependency_id not in self.groups:
                    errors.append(f"Test {test_id} depends on non-existent group {dep.dependency_id}")
        
        is_valid = len(errors) == 0
        self.is_validated = is_valid
        
        if is_valid:
            logger.info("Dependency validation successful")
        else:
            logger.error(f"Dependency validation failed with {len(errors)} errors")
            for error in errors:
                logger.error(f"  - {error}")
        
        return is_valid, errors
    
    def update_test_status(self, test_id: str, status: str, result: Any = None,
                          metadata: Dict[str, Any] = None) -> None:
        """
        Update the status of a test.
        
        Args:
            test_id: ID of the test to update
            status: New status of the test (e.g., "pending", "running", "completed", "failed")
            result: Optional result data
            metadata: Optional additional metadata
        """
        if test_id not in self.test_dependencies:
            logger.warning(f"Updating status for unknown test {test_id}. Registering it.")
            self.register_test(test_id)
        
        self.test_statuses[test_id] = {
            "status": status,
            "result": result,
            "timestamp": time.time(),
            "metadata": metadata or {}
        }
        
        # Invalidate resolution cache for dependent tests
        for dependent_id in self.test_dependencies[test_id].dependents:
            if dependent_id in self.resolution_cache:
                self.resolution_cache.pop(dependent_id)
        
        logger.debug(f"Updated status of test {test_id} to {status}")
    
    def batch_update_test_statuses(self, updates: List[Dict[str, Any]]) -> None:
        """
        Update the status of multiple tests at once.
        
        Args:
            updates: List of status update dictionaries, each containing:
                    - test_id: ID of the test to update
                    - status: New status of the test
                    - result: (Optional) Result data
                    - metadata: (Optional) Additional metadata
        """
        for update in updates:
            self.update_test_status(
                test_id=update["test_id"],
                status=update["status"],
                result=update.get("result"),
                metadata=update.get("metadata")
            )
        
        logger.info(f"Updated status for {len(updates)} tests in batch")
    
    def get_test_status(self, test_id: str) -> Dict[str, Any]:
        """
        Get the current status of a test.
        
        Args:
            test_id: ID of the test
            
        Returns:
            Dictionary with test status information
        """
        if test_id not in self.test_statuses:
            return {"status": "unknown", "timestamp": None, "result": None, "metadata": {}}
        
        return self.test_statuses[test_id]
    
    def resolve_dependencies(self, test_id: str) -> Tuple[bool, Dict[str, DependencyStatus]]:
        """
        Resolve dependencies for a test to determine if it can be executed.
        
        Args:
            test_id: ID of the test
            
        Returns:
            Tuple of (can_execute, dependency_statuses)
        """
        if test_id not in self.test_dependencies:
            logger.warning(f"Resolving dependencies for unknown test {test_id}")
            return False, {}
        
        # Return cached resolution if available
        if test_id in self.resolution_cache:
            all_satisfied = all(status == DependencyStatus.SATISFIED or status == DependencyStatus.IGNORED
                              for status in self.resolution_cache[test_id].values())
            return all_satisfied, self.resolution_cache[test_id]
        
        dependency_statuses = {}
        info = self.test_dependencies[test_id]
        
        # Check each dependency
        for dep in info.dependencies:
            if dep.is_group:
                # Group dependency
                group_id = dep.dependency_id
                if group_id not in self.groups:
                    logger.warning(f"Unknown group dependency {group_id} for test {test_id}")
                    dependency_statuses[f"group:{group_id}"] = DependencyStatus.PENDING
                    continue
                
                # Check all tests in the group
                group_satisfied = True
                for group_test_id in self.groups[group_id]:
                    test_status = self.get_test_status(group_test_id)
                    if test_status["status"] == "failed" and dep.dependency_type == DependencyType.HARD:
                        dependency_statuses[f"group:{group_id}:{group_test_id}"] = DependencyStatus.FAILED
                        group_satisfied = False
                        break
                    elif test_status["status"] != "completed":
                        dependency_statuses[f"group:{group_id}:{group_test_id}"] = DependencyStatus.PENDING
                        if dep.dependency_type != DependencyType.OPTIONAL:
                            group_satisfied = False
                
                if group_satisfied:
                    dependency_statuses[f"group:{group_id}"] = DependencyStatus.SATISFIED
            else:
                # Direct dependency
                dep_id = dep.dependency_id
                dep_status = self.get_test_status(dep_id)
                
                if dep_status["status"] == "completed":
                    dependency_statuses[dep_id] = DependencyStatus.SATISFIED
                elif dep_status["status"] == "failed":
                    if dep.dependency_type == DependencyType.HARD:
                        dependency_statuses[dep_id] = DependencyStatus.FAILED
                    elif dep.dependency_type == DependencyType.SOFT:
                        dependency_statuses[dep_id] = DependencyStatus.SATISFIED
                    else:  # OPTIONAL
                        dependency_statuses[dep_id] = DependencyStatus.IGNORED
                else:
                    if dep.dependency_type == DependencyType.OPTIONAL:
                        dependency_statuses[dep_id] = DependencyStatus.IGNORED
                    else:
                        dependency_statuses[dep_id] = DependencyStatus.PENDING
        
        # Cache the resolution
        self.resolution_cache[test_id] = dependency_statuses
        
        # Check if all dependencies are satisfied
        all_satisfied = all(status == DependencyStatus.SATISFIED or status == DependencyStatus.IGNORED
                          for status in dependency_statuses.values())
        
        logger.debug(f"Dependencies for test {test_id} resolved: can_execute={all_satisfied}")
        
        return all_satisfied, dependency_statuses
    
    def get_execution_order(self) -> List[str]:
        """
        Generate a valid execution order for all tests based on dependencies.
        
        Returns:
            List of test IDs in a valid execution order
        """
        if not self.is_validated:
            is_valid, errors = self.validate_dependencies()
            if not is_valid:
                raise ValueError(f"Cannot generate execution order: {len(errors)} validation errors")
        
        # Use topological sort to get a valid execution order
        try:
            order = list(nx.topological_sort(self.graph))
            
            # Add tests that don't have dependencies and aren't dependencies themselves
            isolated_tests = set(self.test_dependencies.keys()) - set(order)
            order.extend(isolated_tests)
            
            logger.info(f"Generated execution order with {len(order)} tests")
            return order
        except nx.NetworkXUnfeasible:
            logger.error("Cannot generate execution order: graph contains cycles")
            raise ValueError("Cannot generate execution order: dependency graph contains cycles")
    
    def get_parallel_execution_groups(self) -> List[List[str]]:
        """
        Generate groups of tests that can be executed in parallel.
        
        Returns:
            List of lists, where each inner list contains test IDs that can run in parallel
        """
        if not self.is_validated:
            is_valid, errors = self.validate_dependencies()
            if not is_valid:
                raise ValueError(f"Cannot generate parallel groups: {len(errors)} validation errors")
        
        # First get a valid execution order
        order = self.get_execution_order()
        
        # Create a map of tests to their levels in the dependency graph
        levels = {}
        for test_id in order:
            # Get all dependencies for this test
            dependencies = set()
            for dep in self.test_dependencies[test_id].dependencies:
                if not dep.is_group:
                    dependencies.add(dep.dependency_id)
                else:
                    # Add all tests in the group
                    dependencies.update(self.groups[dep.dependency_id])
            
            # Calculate the level (max level of dependencies + 1)
            if not dependencies:
                levels[test_id] = 0
            else:
                levels[test_id] = max(levels.get(dep, 0) for dep in dependencies) + 1
        
        # Group tests by level
        groups = defaultdict(list)
        for test_id, level in levels.items():
            groups[level].append(test_id)
        
        # Convert to list of lists
        parallel_groups = [groups[level] for level in sorted(groups.keys())]
        
        logger.info(f"Generated {len(parallel_groups)} parallel execution groups")
        
        return parallel_groups
    
    def prune_satisfied_dependencies(self, test_id: str) -> None:
        """
        Remove dependencies that are already satisfied for a test.
        
        This is useful to simplify the dependency graph for visualization or
        to reduce unnecessary dependency checks.
        
        Args:
            test_id: ID of the test to prune dependencies for
        """
        if test_id not in self.test_dependencies:
            logger.warning(f"Cannot prune dependencies for unknown test {test_id}")
            return
        
        info = self.test_dependencies[test_id]
        original_count = len(info.dependencies)
        
        pruned_dependencies = []
        for dep in info.dependencies:
            # Check if dependency is satisfied
            if dep.is_group:
                # Group dependency
                group_id = dep.dependency_id
                if group_id not in self.groups:
                    pruned_dependencies.append(dep)
                    continue
                
                # Check if all tests in group are completed
                all_completed = True
                for group_test_id in self.groups[group_id]:
                    test_status = self.get_test_status(group_test_id)
                    if test_status["status"] != "completed":
                        all_completed = False
                        break
                
                if not all_completed:
                    pruned_dependencies.append(dep)
            else:
                # Direct dependency
                dep_status = self.get_test_status(dep.dependency_id)
                if dep_status["status"] != "completed":
                    pruned_dependencies.append(dep)
        
        # Update dependencies
        info.dependencies = pruned_dependencies
        
        # Update graph if needed
        for dep in list(self.graph.predecessors(test_id)):
            if not any(d.dependency_id == dep and not d.is_group for d in pruned_dependencies):
                self.graph.remove_edge(dep, test_id)
        
        pruned_count = original_count - len(pruned_dependencies)
        if pruned_count > 0:
            logger.debug(f"Pruned {pruned_count} satisfied dependencies for test {test_id}")
    
    def get_dependency_graph_json(self) -> Dict[str, Any]:
        """
        Get a JSON representation of the dependency graph for visualization.
        
        Returns:
            Dictionary with nodes and links for visualization
        """
        nodes = []
        for test_id, info in self.test_dependencies.items():
            status = self.get_test_status(test_id)
            nodes.append({
                "id": test_id,
                "status": status["status"],
                "groups": info.group_memberships,
                "dependents_count": len(info.dependents),
                "dependencies_count": len(info.dependencies)
            })
        
        links = []
        for test_id, info in self.test_dependencies.items():
            for dep in info.dependencies:
                if not dep.is_group:
                    links.append({
                        "source": dep.dependency_id,
                        "target": test_id,
                        "type": dep.dependency_type.value
                    })
        
        return {
            "nodes": nodes,
            "links": links,
            "groups": {group: list(tests) for group, tests in self.groups.items()}
        }
    
    def get_ready_tests(self) -> List[str]:
        """
        Get a list of tests that are ready to be executed based on dependency resolution.
        
        Returns:
            List of test IDs that can be executed
        """
        ready_tests = []
        
        for test_id in self.test_dependencies:
            # Skip tests that are already running or completed
            status = self.get_test_status(test_id)
            if status["status"] in ["running", "completed", "failed"]:
                continue
            
            # Check dependencies
            can_execute, _ = self.resolve_dependencies(test_id)
            if can_execute:
                ready_tests.append(test_id)
        
        logger.debug(f"Found {len(ready_tests)} tests ready for execution")
        return ready_tests
    
    def clear_cache(self) -> None:
        """Clear the dependency resolution cache."""
        self.resolution_cache.clear()
        logger.debug("Cleared dependency resolution cache")
    
    def export_dependency_data(self) -> Dict[str, Any]:
        """
        Export all dependency data for persistence or transfer.
        
        Returns:
            Dictionary with all dependency data
        """
        # Convert test dependencies to serializable format
        test_deps = {}
        for test_id, info in self.test_dependencies.items():
            test_deps[test_id] = {
                "dependencies": [
                    {
                        "dependency_id": dep.dependency_id,
                        "dependency_type": dep.dependency_type.value,
                        "timeout_seconds": dep.timeout_seconds,
                        "is_group": dep.is_group,
                        "custom_properties": dep.custom_properties
                    }
                    for dep in info.dependencies
                ],
                "dependents": info.dependents,
                "group_memberships": info.group_memberships
            }
        
        # Export groups
        group_data = {group: list(tests) for group, tests in self.groups.items()}
        
        # Export statuses
        status_data = self.test_statuses
        
        return {
            "test_dependencies": test_deps,
            "groups": group_data,
            "test_statuses": status_data
        }
    
    def import_dependency_data(self, data: Dict[str, Any]) -> None:
        """
        Import dependency data from a previously exported dataset.
        
        Args:
            data: Dependency data dictionary from export_dependency_data()
        """
        # Reset current state
        self.test_dependencies = {}
        self.groups = defaultdict(set)
        self.graph = nx.DiGraph()
        self.test_statuses = {}
        self.resolution_cache = {}
        self.is_validated = False
        
        # Import test dependencies
        for test_id, info in data["test_dependencies"].items():
            dependencies = []
            for dep_data in info["dependencies"]:
                dependencies.append(Dependency(
                    dependency_id=dep_data["dependency_id"],
                    dependency_type=DependencyType(dep_data["dependency_type"]),
                    timeout_seconds=dep_data["timeout_seconds"],
                    is_group=dep_data["is_group"],
                    custom_properties=dep_data["custom_properties"]
                ))
            
            self.test_dependencies[test_id] = TestDependencyInfo(
                test_id=test_id,
                dependencies=dependencies,
                dependents=info["dependents"],
                group_memberships=info["group_memberships"]
            )
            
            # Add to graph
            self.graph.add_node(test_id)
            
            # Add edges for dependencies
            for dep in dependencies:
                if not dep.is_group:
                    self.graph.add_edge(dep.dependency_id, test_id)
            
            # Add to groups
            for group in info["group_memberships"]:
                self.groups[group].add(test_id)
        
        # Import statuses
        self.test_statuses = data["test_statuses"]
        
        logger.info(f"Imported dependency data for {len(self.test_dependencies)} tests "
                   f"and {len(self.groups)} groups")
        
    def __str__(self) -> str:
        """String representation of the dependency manager."""
        return (f"TestDependencyManager(tests={len(self.test_dependencies)}, "
                f"groups={len(self.groups)}, validated={self.is_validated})")
    
    def __repr__(self) -> str:
        """Detailed representation of the dependency manager."""
        return str(self)


# Example usage
if __name__ == "__main__":
    # Create a dependency manager
    manager = TestDependencyManager()
    
    # Register some tests with dependencies
    manager.register_test("test1", [], ["group1"])
    manager.register_test("test2", [Dependency("test1")], ["group1"])
    manager.register_test("test3", [Dependency("test1"), Dependency("test2")], ["group2"])
    manager.register_test("test4", [Dependency("group1", is_group=True)], ["group2"])
    manager.register_test("test5", [
        Dependency("test3"), 
        Dependency("test4", dependency_type=DependencyType.SOFT)
    ])
    
    # Validate dependencies
    is_valid, errors = manager.validate_dependencies()
    print(f"Dependencies valid: {is_valid}")
    if not is_valid:
        for error in errors:
            print(f"  - {error}")
    
    # Generate execution order
    print("\nExecution Order:")
    order = manager.get_execution_order()
    for idx, test_id in enumerate(order):
        print(f"{idx+1}. {test_id}")
    
    # Generate parallel execution groups
    print("\nParallel Execution Groups:")
    groups = manager.get_parallel_execution_groups()
    for idx, group in enumerate(groups):
        print(f"Group {idx+1}: {', '.join(group)}")
    
    # Update some test statuses
    manager.update_test_status("test1", "completed")
    manager.update_test_status("test2", "running")
    
    # Check which tests are ready
    print("\nReady Tests:")
    ready = manager.get_ready_tests()
    print(", ".join(ready) if ready else "None")
    
    # Resolve dependencies for a specific test
    print("\nDependency Resolution for test5:")
    can_execute, statuses = manager.resolve_dependencies("test5")
    print(f"Can execute: {can_execute}")
    for dep_id, status in statuses.items():
        print(f"  - {dep_id}: {status}")