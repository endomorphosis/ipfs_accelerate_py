#!/usr/bin/env python3
"""
Unit tests for the Execution Orchestrator component.

This module contains comprehensive tests for the ExecutionOrchestrator class
to ensure proper execution of tests in a parallel and distributed manner.
"""

import unittest
import sys
import os
import logging
import anyio
import time
from unittest.mock import patch, MagicMock
from typing import Dict, List, Set, Optional, Tuple, Any, Union

# Add the parent directory to the path to import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from execution_orchestrator import (
    ExecutionOrchestrator, ExecutionStrategy, ExecutionStatus,
    ExecutionContext, ExecutionGroup
)
from test_dependency_manager import (
    TestDependencyManager, Dependency, DependencyType
)


class ExecutionOrchestratorTests(unittest.TestCase):
    """Unit tests for the ExecutionOrchestrator class."""
    
    def setUp(self):
        """Set up a test orchestrator before each test."""
        # Disable logging for tests
        logging.disable(logging.CRITICAL)
        
        # Create a test dependency manager
        self.dependency_manager = TestDependencyManager()
        
        # Register some tests with dependencies
        self.dependency_manager.register_test("test1", [], ["group1"])
        self.dependency_manager.register_test("test2", [Dependency("test1")], ["group1"])
        self.dependency_manager.register_test("test3", [
            Dependency("test1"), 
            Dependency("test2")
        ], ["group2"])
        self.dependency_manager.register_test("test4", [
            Dependency("group1", is_group=True)
        ], ["group2"])
        self.dependency_manager.register_test("test5", [
            Dependency("test3"), 
            Dependency("test4", dependency_type=DependencyType.SOFT)
        ])
        
        # Create a fresh orchestrator for each test
        self.orchestrator = ExecutionOrchestrator(
            dependency_manager=self.dependency_manager,
            max_workers=2,
            strategy=ExecutionStrategy.SIMPLE
        )
    
    def tearDown(self):
        """Clean up after each test."""
        # Re-enable logging
        logging.disable(logging.NOTSET)
    
    def test_orchestrator_initialization(self):
        """Test orchestrator initialization."""
        self.assertEqual(self.orchestrator.max_workers, 2)
        self.assertEqual(self.orchestrator.strategy, ExecutionStrategy.SIMPLE)
        self.assertIs(self.orchestrator.dependency_manager, self.dependency_manager)
        self.assertEqual(len(self.orchestrator.execution_groups), 0)
        self.assertEqual(len(self.orchestrator.execution_contexts), 0)
        self.assertEqual(len(self.orchestrator.active_workers), 0)
        self.assertEqual(len(self.orchestrator.execution_queue), 0)
    
    def test_create_execution_groups(self):
        """Test creation of execution groups."""
        # Create execution groups
        self.orchestrator.create_execution_groups()
        
        # Verify groups were created
        self.assertGreater(len(self.orchestrator.execution_groups), 0)
        
        # Verify each test has an execution context
        self.assertEqual(len(self.orchestrator.execution_contexts), 5)
        
        # Verify first group is ready
        self.assertTrue(self.orchestrator.execution_groups["group_0"].is_ready)
        
        # Verify group dependencies are set correctly
        for group_id, group in self.orchestrator.execution_groups.items():
            level = group.metadata.get("level", 0)
            if level > 0:
                self.assertGreater(len(group.dependencies), 0)
    
    def test_adjust_max_workers(self):
        """Test adjustment of max workers."""
        # Create orchestrator with different strategies
        for strategy in ExecutionStrategy:
            orchestrator = ExecutionOrchestrator(
                dependency_manager=self.dependency_manager,
                max_workers=4,
                strategy=strategy
            )
            
            # Adjust max workers
            original_max_workers = orchestrator.max_workers
            orchestrator.adjust_max_workers()
            
            # Verify max workers was adjusted or remains the same
            self.assertGreaterEqual(orchestrator.max_workers, 1)
            
            # For resource-aware strategy, the result depends on system load
            if strategy != ExecutionStrategy.RESOURCE_AWARE:
                self.assertNotEqual(orchestrator.max_workers, 0)
    
    def test_update_group_status(self):
        """Test updating group status."""
        # Create execution groups
        self.orchestrator.create_execution_groups()
        
        # Verify first group is ready
        self.assertTrue(self.orchestrator.execution_groups["group_0"].is_ready)
        
        # Mark tests in first group as completed
        for test_id in self.orchestrator.execution_groups["group_0"].test_ids:
            execution_context = self.orchestrator.execution_contexts[test_id]
            execution_context.status = ExecutionStatus.COMPLETED
            self.dependency_manager.update_test_status(test_id, "completed")
        
        # Update group status
        self.orchestrator.update_group_status()
        
        # Verify first group is marked as completed
        self.assertTrue(self.orchestrator.execution_groups["group_0"].is_completed)
        
        # Verify second group is now ready
        if "group_1" in self.orchestrator.execution_groups:
            self.assertTrue(self.orchestrator.execution_groups["group_1"].is_ready)
    
    def test_select_tests_for_execution(self):
        """Test selection of tests for execution."""
        # Create execution groups
        self.orchestrator.create_execution_groups()
        
        # Verify that tests from first group are in the execution queue
        first_group_tests = self.orchestrator.execution_groups["group_0"].test_ids
        for test_id in first_group_tests:
            self.assertIn(test_id, self.orchestrator.execution_queue)
        
        # Select tests for execution
        selected_tests = self.orchestrator.select_tests_for_execution()
        
        # Verify some tests were selected
        self.assertLessEqual(len(selected_tests), self.orchestrator.max_workers)
        
        # Verify selected tests are no longer in the queue
        for test_id in selected_tests:
            self.assertNotIn(test_id, self.orchestrator.execution_queue)
    
    @patch('execution_orchestrator.ExecutionOrchestrator.execute_test')
    def test_async_execute_tests(self, mock_execute):
        """Test asynchronous execution of tests."""
        # Create execution groups
        self.orchestrator.create_execution_groups()
        
        # Select tests for execution
        selected_tests = self.orchestrator.select_tests_for_execution()
        
        # Execute tests asynchronously
        futures = self.orchestrator.async_execute_tests(selected_tests)
        
        # Verify futures were created
        self.assertEqual(len(futures), len(selected_tests))
        
        # Verify execute_test was called for each test
        self.assertEqual(mock_execute.call_count, len(selected_tests))
    
    @patch('time.sleep')
    @patch('random.uniform')
    @patch('random.random')
    def test_execute_test(self, mock_random, mock_uniform, mock_sleep):
        """Test execution of a single test."""
        # Mock random functions
        mock_uniform.return_value = 0.1  # Fast execution time
        mock_random.return_value = 0.95  # Success (> 0.9)
        
        # Create execution context
        test_id = "test1"
        worker_id = "worker1"
        self.orchestrator.execution_contexts[test_id] = ExecutionContext(test_id=test_id)
        
        # Execute test
        self.orchestrator.execute_test(test_id, worker_id)
        
        # Verify context was updated
        context = self.orchestrator.execution_contexts[test_id]
        self.assertEqual(context.worker_id, worker_id)
        self.assertEqual(context.status, ExecutionStatus.COMPLETED)
        self.assertIsNotNone(context.start_time)
        self.assertIsNotNone(context.end_time)
        self.assertIsNotNone(context.result)
        self.assertIsNone(context.error)
        self.assertEqual(context.progress, 1.0)
        
        # Verify dependency manager was updated
        status = self.dependency_manager.get_test_status(test_id)
        self.assertEqual(status["status"], "completed")
    
    @patch('time.sleep')
    @patch('random.uniform')
    @patch('random.random')
    def test_execute_test_failure(self, mock_random, mock_uniform, mock_sleep):
        """Test execution of a failing test."""
        # Mock random functions
        mock_uniform.return_value = 0.1  # Fast execution time
        mock_random.return_value = 0.8  # Failure (< 0.9)
        
        # Create execution context
        test_id = "test1"
        worker_id = "worker1"
        self.orchestrator.execution_contexts[test_id] = ExecutionContext(test_id=test_id)
        
        # Execute test
        self.orchestrator.execute_test(test_id, worker_id)
        
        # Verify context was updated
        context = self.orchestrator.execution_contexts[test_id]
        self.assertEqual(context.worker_id, worker_id)
        self.assertEqual(context.status, ExecutionStatus.FAILED)
        self.assertIsNotNone(context.start_time)
        self.assertIsNotNone(context.end_time)
        self.assertIsNone(context.result)
        self.assertIsNotNone(context.error)
        self.assertEqual(context.progress, 1.0)
        
        # Verify dependency manager was updated
        status = self.dependency_manager.get_test_status(test_id)
        self.assertEqual(status["status"], "failed")
    
    @patch('execution_orchestrator.ExecutionOrchestrator.execute_test')
    def test_hooks(self, mock_execute):
        """Test pre and post execution hooks."""
        # Create mocks for hooks
        pre_hook = MagicMock()
        post_hook = MagicMock()
        test_start_hook = MagicMock()
        test_end_hook = MagicMock()
        
        # Set hooks
        self.orchestrator.set_pre_execution_hook(pre_hook)
        self.orchestrator.set_post_execution_hook(post_hook)
        self.orchestrator.set_test_start_hook(test_start_hook)
        self.orchestrator.set_test_end_hook(test_end_hook)
        
        # Wrap the execute_test method to call hooks
        original_execute_test = self.orchestrator.execute_test
        
        def execute_test_wrapper(test_id, worker_id):
            # Call original method
            context = self.orchestrator.execution_contexts[test_id]
            context.status = ExecutionStatus.COMPLETED
            
            # Call test hooks
            self.orchestrator.test_start_hook(test_id, worker_id)
            self.orchestrator.test_end_hook(test_id, ExecutionStatus.COMPLETED, None, None)
        
        mock_execute.side_effect = execute_test_wrapper
        
        # Execute a simple test run
        self.orchestrator.create_execution_groups()
        test_id = self.orchestrator.execution_groups["group_0"].test_ids[0]
        self.orchestrator.execute_test(test_id, "worker1")
        
        # Execute all tests with limited concurrency
        with patch('concurrent.futures.ThreadPoolExecutor'):
            self.orchestrator.execute_all_tests()
        
        # Verify hooks were called
        pre_hook.assert_called_once()
        post_hook.assert_called_once()
        self.assertGreater(test_start_hook.call_count, 0)
        self.assertGreater(test_end_hook.call_count, 0)
    
    @patch('concurrent.futures.ThreadPoolExecutor')
    @patch('time.sleep')
    def test_execute_all_tests(self, mock_sleep, mock_executor):
        """Test execution of all tests."""
        # Setup mock executor
        mock_executor_instance = MagicMock()
        mock_executor.return_value = mock_executor_instance
        mock_executor_instance.submit.side_effect = lambda fn, *args, **kwargs: MagicMock()
        
        # Patch update_group_status to mark all groups as completed after first call
        original_update_group_status = self.orchestrator.update_group_status
        
        def mock_update_group_status():
            original_update_group_status()
            
            # Mark all groups as completed
            for group in self.orchestrator.execution_groups.values():
                group.is_completed = True
            
            # Mark all tests as completed
            for context in self.orchestrator.execution_contexts.values():
                if context.status not in [
                    ExecutionStatus.COMPLETED,
                    ExecutionStatus.FAILED,
                    ExecutionStatus.SKIPPED,
                ]:
                    context.status = ExecutionStatus.COMPLETED
        
        with patch.object(self.orchestrator, 'update_group_status', side_effect=mock_update_group_status):
            # Execute all tests
            results = self.orchestrator.execute_all_tests()
        
        # Verify results structure
        self.assertIn('metrics', results)
        self.assertIn('test_results', results)
        self.assertIn('execution_groups', results)
        self.assertIn('start_time', results)
        self.assertIn('end_time', results)
        self.assertIsNotNone(results['total_time'])
    
    def test_visualize_execution_plan(self):
        """Test visualization of execution plan."""
        # Create execution groups
        self.orchestrator.create_execution_groups()
        
        # Test text format
        text_plan = self.orchestrator.visualize_execution_plan(output_format="text")
        self.assertIsInstance(text_plan, str)
        self.assertIn("Execution Plan", text_plan)
        
        # Test JSON format
        json_plan = self.orchestrator.visualize_execution_plan(output_format="json")
        self.assertIsInstance(json_plan, str)
        self.assertIn("strategy", json_plan)
        self.assertIn("groups", json_plan)
        
        # Test Mermaid format
        mermaid_plan = self.orchestrator.visualize_execution_plan(output_format="mermaid")
        self.assertIsInstance(mermaid_plan, str)
        self.assertIn("graph TD", mermaid_plan)
        
        # Test invalid format
        with self.assertRaises(ValueError):
            self.orchestrator.visualize_execution_plan(output_format="invalid")
    
    @patch('execution_orchestrator.asyncio.create_task')
    @patch('execution_orchestrator.asyncio.wait')
    async def test_execute_all_tests_async(self, mock_wait, mock_create_task):
        """Test asynchronous execution of all tests."""
        # Setup mock asyncio functions
        mock_task = MagicMock()
        mock_create_task.return_value = mock_task
        
        # Mock asyncio.wait to return done tasks
        mock_wait.side_effect = lambda tasks, **kwargs: (set([mock_task]), set())
        
        # Patch update_group_status to mark all groups as completed after first call
        original_update_group_status = self.orchestrator.update_group_status
        
        def mock_update_group_status():
            original_update_group_status()
            
            # Mark all groups as completed
            for group in self.orchestrator.execution_groups.values():
                group.is_completed = True
            
            # Mark all tests as completed
            for context in self.orchestrator.execution_contexts.values():
                if context.status == ExecutionStatus.PENDING:
                    context.status = ExecutionStatus.COMPLETED
        
        with patch.object(self.orchestrator, 'update_group_status', side_effect=mock_update_group_status):
            # Execute all tests asynchronously
            results = await self.orchestrator.execute_all_tests_async()
        
        # Verify results structure
        self.assertIn('metrics', results)
        self.assertIn('test_results', results)
        self.assertIn('execution_groups', results)
        self.assertIn('start_time', results)
        self.assertIn('end_time', results)
    
    @patch('asyncio.sleep')
    @patch('random.uniform')
    @patch('random.random')
    async def test_execute_test_async(self, mock_random, mock_uniform, mock_sleep):
        """Test asynchronous execution of a single test."""
        # Mock random functions
        mock_uniform.return_value = 0.1  # Fast execution time
        mock_random.return_value = 0.95  # Success (> 0.9)
        
        # Create execution context
        test_id = "test1"
        worker_id = "worker1"
        self.orchestrator.execution_contexts[test_id] = ExecutionContext(test_id=test_id)
        
        # Execute test asynchronously
        await self.orchestrator.execute_test_async(test_id, worker_id)
        
        # Verify context was updated
        context = self.orchestrator.execution_contexts[test_id]
        self.assertEqual(context.worker_id, worker_id)
        self.assertEqual(context.status, ExecutionStatus.COMPLETED)
        self.assertIsNotNone(context.start_time)
        self.assertIsNotNone(context.end_time)
        self.assertIsNotNone(context.result)
        self.assertIsNone(context.error)
        self.assertEqual(context.progress, 1.0)
        
        # Verify dependency manager was updated
        status = self.dependency_manager.get_test_status(test_id)
        self.assertEqual(status["status"], "completed")
    
    @patch('asyncio.sleep')
    @patch('random.uniform')
    @patch('random.random')
    async def test_execute_test_async_failure(self, mock_random, mock_uniform, mock_sleep):
        """Test asynchronous execution of a failing test."""
        # Mock random functions
        mock_uniform.return_value = 0.1  # Fast execution time
        mock_random.return_value = 0.8  # Failure (< 0.9)
        
        # Create execution context
        test_id = "test1"
        worker_id = "worker1"
        self.orchestrator.execution_contexts[test_id] = ExecutionContext(test_id=test_id)
        
        # Execute test asynchronously
        await self.orchestrator.execute_test_async(test_id, worker_id)
        
        # Verify context was updated
        context = self.orchestrator.execution_contexts[test_id]
        self.assertEqual(context.worker_id, worker_id)
        self.assertEqual(context.status, ExecutionStatus.FAILED)
        self.assertIsNotNone(context.start_time)
        self.assertIsNotNone(context.end_time)
        self.assertIsNone(context.result)
        self.assertIsNotNone(context.error)
        self.assertEqual(context.progress, 1.0)
        
        # Verify dependency manager was updated
        status = self.dependency_manager.get_test_status(test_id)
        self.assertEqual(status["status"], "failed")
    
    @patch('asyncio.sleep')
    async def test_execute_test_async_cancellation(self, mock_sleep):
        """Test cancellation of asynchronous test execution."""
        # Mock sleep to raise CancelledError
        mock_sleep.side_effect = asyncio.CancelledError()
        
        # Create execution context
        test_id = "test1"
        worker_id = "worker1"
        self.orchestrator.execution_contexts[test_id] = ExecutionContext(test_id=test_id)
        
        # Execute test asynchronously (should raise CancelledError)
        with self.assertRaises(asyncio.CancelledError):
            await self.orchestrator.execute_test_async(test_id, worker_id)
        
        # Verify context was updated
        context = self.orchestrator.execution_contexts[test_id]
        self.assertEqual(context.worker_id, worker_id)
        self.assertEqual(context.status, ExecutionStatus.FAILED)
        self.assertIsNotNone(context.error)
        
        # Verify dependency manager was updated
        status = self.dependency_manager.get_test_status(test_id)
        self.assertEqual(status["status"], "failed")


# Run only the synchronous tests by default
if __name__ == '__main__':
    unittest.main()