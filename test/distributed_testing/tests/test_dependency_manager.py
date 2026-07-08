#!/usr/bin/env python3
"""
Unit tests for the Test Dependency Manager component.

This module contains comprehensive tests for the TestDependencyManager class
to ensure proper functionality for dependency tracking, resolution, validation,
and execution ordering.
"""

import unittest
import sys
import os
import logging
import pytest

nx = pytest.importorskip("networkx")
from typing import Dict, List, Set, Optional, Tuple, Any, Union

# Add the parent directory to the path to import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from test_dependency_manager import (
    TestDependencyManager, Dependency, TestDependencyInfo, DependencyType, DependencyStatus
)


class TestDependencyManagerTests(unittest.TestCase):
    """Unit tests for the TestDependencyManager class."""
    
    def setUp(self):
        """Set up a test dependency manager before each test."""
        # Disable logging for tests
        logging.disable(logging.CRITICAL)
        
        # Create a fresh dependency manager for each test
        self.manager = TestDependencyManager()
    
    def tearDown(self):
        """Clean up after each test."""
        # Re-enable logging
        logging.disable(logging.NOTSET)
    
    def test_register_test(self):
        """Test registering a test with dependencies and groups."""
        # Register a test
        self.manager.register_test("test1", [
            Dependency("dep1"),
            Dependency("dep2", dependency_type=DependencyType.SOFT)
        ], ["group1", "group2"])
        
        # Verify the test was registered
        self.assertIn("test1", self.manager.test_dependencies)
        
        # Verify dependencies were registered
        test_info = self.manager.test_dependencies["test1"]
        self.assertEqual(len(test_info.dependencies), 2)
        self.assertEqual(test_info.dependencies[0].dependency_id, "dep1")
        self.assertEqual(test_info.dependencies[1].dependency_id, "dep2")
        self.assertEqual(test_info.dependencies[1].dependency_type, DependencyType.SOFT)
        
        # Verify group memberships were registered
        self.assertEqual(test_info.group_memberships, ["group1", "group2"])
        self.assertIn("test1", self.manager.groups["group1"])
        self.assertIn("test1", self.manager.groups["group2"])
        
        # Verify dependency tests were created
        self.assertIn("dep1", self.manager.test_dependencies)
        self.assertIn("dep2", self.manager.test_dependencies)
        
        # Verify dependent relationships
        self.assertIn("test1", self.manager.test_dependencies["dep1"].dependents)
        self.assertIn("test1", self.manager.test_dependencies["dep2"].dependents)
        
        # Verify graph structure
        self.assertTrue(self.manager.graph.has_node("test1"))
        self.assertTrue(self.manager.graph.has_node("dep1"))
        self.assertTrue(self.manager.graph.has_node("dep2"))
        self.assertTrue(self.manager.graph.has_edge("dep1", "test1"))
        self.assertTrue(self.manager.graph.has_edge("dep2", "test1"))
    
    def test_register_test_batch(self):
        """Test registering multiple tests at once."""
        # Register a batch of tests
        self.manager.register_test_batch([
            {
                "test_id": "test1",
                "dependencies": [Dependency("dep1")],
                "groups": ["group1"]
            },
            {
                "test_id": "test2",
                "dependencies": [Dependency("test1")],
                "groups": ["group2"]
            },
            {
                "test_id": "test3",
                "dependencies": [
                    Dependency("test1"),
                    Dependency("test2")
                ],
                "groups": ["group1", "group2"]
            }
        ])
        
        # Verify all tests were registered
        self.assertIn("test1", self.manager.test_dependencies)
        self.assertIn("test2", self.manager.test_dependencies)
        self.assertIn("test3", self.manager.test_dependencies)
        
        # Verify dependencies
        self.assertEqual(len(self.manager.test_dependencies["test3"].dependencies), 2)
        
        # Verify group memberships
        self.assertIn("test1", self.manager.groups["group1"])
        self.assertIn("test2", self.manager.groups["group2"])
        self.assertIn("test3", self.manager.groups["group1"])
        self.assertIn("test3", self.manager.groups["group2"])
    
    def test_validate_dependencies_no_cycles(self):
        """Test dependency validation with no cycles."""
        # Create a simple dependency chain
        self.manager.register_test("test1")
        self.manager.register_test("test2", [Dependency("test1")])
        self.manager.register_test("test3", [Dependency("test2")])
        
        # Validate dependencies
        is_valid, errors = self.manager.validate_dependencies()
        
        # Should be valid with no errors
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)
        self.assertTrue(self.manager.is_validated)
    
    def test_validate_dependencies_with_cycles(self):
        """Test dependency validation with cycles."""
        # Create a dependency cycle
        self.manager.register_test("test1", [Dependency("test3")])
        self.manager.register_test("test2", [Dependency("test1")])
        self.manager.register_test("test3", [Dependency("test2")])
        
        # Validate dependencies
        is_valid, errors = self.manager.validate_dependencies()
        
        # Should be invalid with cycle errors
        self.assertFalse(is_valid)
        self.assertGreater(len(errors), 0)
        self.assertFalse(self.manager.is_validated)
        
        # Verify error message contains cycle information
        self.assertTrue(any("Cyclic dependency" in error for error in errors))
    
    def test_validate_dependencies_missing_tests(self):
        """Test dependency validation with dependencies on non-existent tests."""
        # Create a dependency on a non-existent test
        self.manager.register_test("test1", [Dependency("missing_test")])
        
        # Validate dependencies
        is_valid, errors = self.manager.validate_dependencies()
        
        # Should be invalid with missing test errors
        self.assertFalse(is_valid)
        self.assertGreater(len(errors), 0)
        self.assertFalse(self.manager.is_validated)
        
        # Verify error message contains missing test information
        self.assertTrue(any("non-existent test" in error for error in errors))
    
    def test_validate_dependencies_missing_groups(self):
        """Test dependency validation with dependencies on non-existent groups."""
        # Create a dependency on a non-existent group
        self.manager.register_test("test1", [Dependency("missing_group", is_group=True)])
        
        # Validate dependencies
        is_valid, errors = self.manager.validate_dependencies()
        
        # Should be invalid with missing group errors
        self.assertFalse(is_valid)
        self.assertGreater(len(errors), 0)
        self.assertFalse(self.manager.is_validated)
        
        # Verify error message contains missing group information
        self.assertTrue(any("non-existent group" in error for error in errors))
    
    def test_update_test_status(self):
        """Test updating the status of a test."""
        # Register a test
        self.manager.register_test("test1")
        
        # Update status
        self.manager.update_test_status(
            "test1",
            "completed",
            result={"passed": True},
            metadata={"duration": 10.5}
        )
        
        # Verify status was updated
        status = self.manager.get_test_status("test1")
        self.assertEqual(status["status"], "completed")
        self.assertEqual(status["result"], {"passed": True})
        self.assertEqual(status["metadata"], {"duration": 10.5})
        self.assertIn("timestamp", status)
    
    def test_update_status_unknown_test(self):
        """Test updating the status of an unknown test."""
        # Update status for unknown test
        self.manager.update_test_status("unknown_test", "running")
        
        # Verify test was automatically registered
        self.assertIn("unknown_test", self.manager.test_dependencies)
        
        # Verify status was updated
        status = self.manager.get_test_status("unknown_test")
        self.assertEqual(status["status"], "running")
    
    def test_batch_update_test_statuses(self):
        """Test updating the status of multiple tests at once."""
        # Register tests
        self.manager.register_test("test1")
        self.manager.register_test("test2")
        
        # Batch update statuses
        self.manager.batch_update_test_statuses([
            {
                "test_id": "test1",
                "status": "running"
            },
            {
                "test_id": "test2",
                "status": "completed",
                "result": {"passed": True}
            }
        ])
        
        # Verify statuses were updated
        self.assertEqual(self.manager.get_test_status("test1")["status"], "running")
        self.assertEqual(self.manager.get_test_status("test2")["status"], "completed")
        self.assertEqual(self.manager.get_test_status("test2")["result"], {"passed": True})
    
    def test_get_test_status_unknown(self):
        """Test getting the status of an unknown test."""
        # Get status for unknown test
        status = self.manager.get_test_status("unknown_test")
        
        # Verify default status was returned
        self.assertEqual(status["status"], "unknown")
        self.assertIsNone(status["timestamp"])
        self.assertIsNone(status["result"])
        self.assertEqual(status["metadata"], {})
    
    def test_resolve_dependencies_all_satisfied(self):
        """Test resolving dependencies when all are satisfied."""
        # Set up tests with dependencies
        self.manager.register_test("test1")
        self.manager.register_test("test2", [Dependency("test1")])
        
        # Mark dependency as completed
        self.manager.update_test_status("test1", "completed")
        
        # Resolve dependencies
        can_execute, statuses = self.manager.resolve_dependencies("test2")
        
        # Should be executable with all dependencies satisfied
        self.assertTrue(can_execute)
        self.assertEqual(statuses["test1"], DependencyStatus.SATISFIED)
    
    def test_resolve_dependencies_pending(self):
        """Test resolving dependencies when some are pending."""
        # Set up tests with dependencies
        self.manager.register_test("test1")
        self.manager.register_test("test2", [Dependency("test1")])
        
        # Dependency not marked as completed
        
        # Resolve dependencies
        can_execute, statuses = self.manager.resolve_dependencies("test2")
        
        # Should not be executable with pending dependencies
        self.assertFalse(can_execute)
        self.assertEqual(statuses["test1"], DependencyStatus.PENDING)
    
    def test_resolve_dependencies_failed_hard(self):
        """Test resolving dependencies when a hard dependency failed."""
        # Set up tests with hard dependency
        self.manager.register_test("test1")
        self.manager.register_test("test2", [Dependency("test1", dependency_type=DependencyType.HARD)])
        
        # Mark dependency as failed
        self.manager.update_test_status("test1", "failed")
        
        # Resolve dependencies
        can_execute, statuses = self.manager.resolve_dependencies("test2")
        
        # Should not be executable with failed hard dependency
        self.assertFalse(can_execute)
        self.assertEqual(statuses["test1"], DependencyStatus.FAILED)
    
    def test_resolve_dependencies_failed_soft(self):
        """Test resolving dependencies when a soft dependency failed."""
        # Set up tests with soft dependency
        self.manager.register_test("test1")
        self.manager.register_test("test2", [Dependency("test1", dependency_type=DependencyType.SOFT)])
        
        # Mark dependency as failed
        self.manager.update_test_status("test1", "failed")
        
        # Resolve dependencies
        can_execute, statuses = self.manager.resolve_dependencies("test2")
        
        # Should be executable with failed soft dependency
        self.assertTrue(can_execute)
        self.assertEqual(statuses["test1"], DependencyStatus.SATISFIED)
    
    def test_resolve_dependencies_optional(self):
        """Test resolving dependencies when an optional dependency is pending."""
        # Set up tests with optional dependency
        self.manager.register_test("test1")
        self.manager.register_test("test2", [Dependency("test1", dependency_type=DependencyType.OPTIONAL)])
        
        # Do not update dependency status (pending)
        
        # Resolve dependencies
        can_execute, statuses = self.manager.resolve_dependencies("test2")
        
        # Should be executable with pending optional dependency
        self.assertTrue(can_execute)
        self.assertEqual(statuses["test1"], DependencyStatus.IGNORED)
    
    def test_resolve_dependencies_group(self):
        """Test resolving dependencies on groups."""
        # Set up tests with group dependency
        self.manager.register_test("test1", [], ["group1"])
        self.manager.register_test("test2", [], ["group1"])
        self.manager.register_test("test3", [Dependency("group1", is_group=True)])
        
        # Mark group tests as completed
        self.manager.update_test_status("test1", "completed")
        self.manager.update_test_status("test2", "completed")
        
        # Resolve dependencies
        can_execute, statuses = self.manager.resolve_dependencies("test3")
        
        # Should be executable with all group dependencies satisfied
        self.assertTrue(can_execute)
        self.assertEqual(statuses["group:group1"], DependencyStatus.SATISFIED)
    
    def test_resolve_dependencies_group_pending(self):
        """Test resolving dependencies when group tests are pending."""
        # Set up tests with group dependency
        self.manager.register_test("test1", [], ["group1"])
        self.manager.register_test("test2", [], ["group1"])
        self.manager.register_test("test3", [Dependency("group1", is_group=True)])
        
        # Mark one group test as completed, one pending
        self.manager.update_test_status("test1", "completed")
        
        # Resolve dependencies
        can_execute, statuses = self.manager.resolve_dependencies("test3")
        
        # Should not be executable with pending group dependencies
        self.assertFalse(can_execute)
        
        # Should have individual status for pending group tests
        has_pending_status = False
        for key, status in statuses.items():
            if "group:group1:test2" in key and status == DependencyStatus.PENDING:
                has_pending_status = True
                break
        self.assertTrue(has_pending_status)
    
    def test_get_execution_order(self):
        """Test generating a valid execution order."""
        # Create a dependency chain
        self.manager.register_test("test1")
        self.manager.register_test("test2", [Dependency("test1")])
        self.manager.register_test("test3", [Dependency("test2")])
        self.manager.register_test("test4", [Dependency("test2")])
        self.manager.register_test("test5")  # No dependencies
        
        # Validate dependencies (required for execution order)
        is_valid, _ = self.manager.validate_dependencies()
        self.assertTrue(is_valid)
        
        # Get execution order
        order = self.manager.get_execution_order()
        
        # Verify order respects dependencies
        self.assertEqual(len(order), 5)
        
        # test1 must come before test2
        idx_test1 = order.index("test1")
        idx_test2 = order.index("test2")
        self.assertLess(idx_test1, idx_test2)
        
        # test2 must come before test3 and test4
        idx_test3 = order.index("test3")
        idx_test4 = order.index("test4")
        self.assertLess(idx_test2, idx_test3)
        self.assertLess(idx_test2, idx_test4)
        
        # test5 should be in the list (no dependencies)
        self.assertIn("test5", order)
    
    def test_get_execution_order_with_cycles(self):
        """Test generating an execution order with cycles raises an error."""
        # Create a dependency cycle
        self.manager.register_test("test1", [Dependency("test3")])
        self.manager.register_test("test2", [Dependency("test1")])
        self.manager.register_test("test3", [Dependency("test2")])
        
        # Attempt to get execution order
        with self.assertRaises(ValueError):
            self.manager.get_execution_order()
    
    def test_get_parallel_execution_groups(self):
        """Test generating parallel execution groups."""
        # Create a dependency graph with multiple levels
        self.manager.register_test("test1")
        self.manager.register_test("test2")
        self.manager.register_test("test3", [Dependency("test1")])
        self.manager.register_test("test4", [Dependency("test1")])
        self.manager.register_test("test5", [Dependency("test2")])
        self.manager.register_test("test6", [Dependency("test3"), Dependency("test5")])
        
        # Validate dependencies (required for parallel groups)
        is_valid, _ = self.manager.validate_dependencies()
        self.assertTrue(is_valid)
        
        # Get parallel execution groups
        groups = self.manager.get_parallel_execution_groups()
        
        # Should have 3 levels
        self.assertEqual(len(groups), 3)
        
        # Level 0: test1, test2 (no dependencies)
        self.assertEqual(set(groups[0]), {"test1", "test2"})
        
        # Level 1: test3, test4, test5 (depend on level 0)
        self.assertEqual(set(groups[1]), {"test3", "test4", "test5"})
        
        # Level 2: test6 (depends on level 1)
        self.assertEqual(groups[2], ["test6"])
    
    def test_get_ready_tests(self):
        """Test getting tests that are ready to execute."""
        # Create tests with dependencies
        self.manager.register_test("test1")
        self.manager.register_test("test2", [Dependency("test1")])
        self.manager.register_test("test3", [Dependency("test2")])
        self.manager.register_test("test4")  # No dependencies
        
        # Mark test1 as completed
        self.manager.update_test_status("test1", "completed")
        
        # Mark test4 as running
        self.manager.update_test_status("test4", "running")
        
        # Get ready tests
        ready = self.manager.get_ready_tests()
        
        # Only test2 should be ready (test1 completed, test3 waiting on test2, test4 running)
        self.assertEqual(ready, ["test2"])
    
    def test_prune_satisfied_dependencies(self):
        """Test pruning satisfied dependencies."""
        # Create tests with dependencies
        self.manager.register_test("test1")
        self.manager.register_test("test2")
        self.manager.register_test("test3", [
            Dependency("test1"),
            Dependency("test2")
        ])
        
        # Mark test1 as completed
        self.manager.update_test_status("test1", "completed")
        
        # Prune satisfied dependencies
        self.manager.prune_satisfied_dependencies("test3")
        
        # Only the dependency on test2 should remain
        dependencies = self.manager.test_dependencies["test3"].dependencies
        self.assertEqual(len(dependencies), 1)
        self.assertEqual(dependencies[0].dependency_id, "test2")
        
        # Graph should be updated
        self.assertFalse(self.manager.graph.has_edge("test1", "test3"))
        self.assertTrue(self.manager.graph.has_edge("test2", "test3"))
    
    def test_export_import_dependency_data(self):
        """Test exporting and importing dependency data."""
        # Create some test data
        self.manager.register_test("test1")
        self.manager.register_test("test2", [Dependency("test1")], ["group1"])
        self.manager.register_test("test3", [
            Dependency("test1"),
            Dependency("test2"),
            Dependency("group1", is_group=True)
        ], ["group2"])
        
        # Update some statuses
        self.manager.update_test_status("test1", "completed")
        self.manager.update_test_status("test2", "running")
        
        # Export data
        exported_data = self.manager.export_dependency_data()
        
        # Create a new manager
        new_manager = TestDependencyManager()
        
        # Import data
        new_manager.import_dependency_data(exported_data)
        
        # Verify tests were imported
        self.assertEqual(len(new_manager.test_dependencies), 3)
        self.assertIn("test1", new_manager.test_dependencies)
        self.assertIn("test2", new_manager.test_dependencies)
        self.assertIn("test3", new_manager.test_dependencies)
        
        # Verify dependencies were imported
        test3_deps = new_manager.test_dependencies["test3"].dependencies
        self.assertEqual(len(test3_deps), 3)
        
        # Verify groups were imported
        self.assertEqual(len(new_manager.groups), 2)
        self.assertIn("group1", new_manager.groups)
        self.assertIn("group2", new_manager.groups)
        
        # Verify statuses were imported
        self.assertEqual(new_manager.get_test_status("test1")["status"], "completed")
        self.assertEqual(new_manager.get_test_status("test2")["status"], "running")
    
    def test_dependency_graph_json(self):
        """Test generating a JSON representation of the dependency graph."""
        # Create some test data
        self.manager.register_test("test1")
        self.manager.register_test("test2", [Dependency("test1")], ["group1"])
        self.manager.register_test("test3", [
            Dependency("test1"),
            Dependency("test2")
        ], ["group2"])
        
        # Update a status
        self.manager.update_test_status("test1", "completed")
        
        # Get graph JSON
        graph_json = self.manager.get_dependency_graph_json()
        
        # Verify structure
        self.assertIn("nodes", graph_json)
        self.assertIn("links", graph_json)
        self.assertIn("groups", graph_json)
        
        # Verify nodes
        self.assertEqual(len(graph_json["nodes"]), 3)
        
        # Verify links
        self.assertEqual(len(graph_json["links"]), 3)  # test1->test2, test1->test3, test2->test3
        
        # Verify groups
        self.assertEqual(len(graph_json["groups"]), 2)
        self.assertIn("group1", graph_json["groups"])
        self.assertIn("group2", graph_json["groups"])


if __name__ == '__main__':
    unittest.main()