"""
Test for distributed testing coordinator in IPFS Accelerate.

This test verifies the functionality of the distributed testing coordinator,
which manages distributed test execution across multiple worker nodes.
"""

import os
import sys
import pytest
import logging
import time
import json
import threading
import tempfile
import socket
import queue
from typing import Dict, List, Any, Optional, Tuple, Set

# Import common utilities
from common.hardware_detection import detect_hardware

# Mock classes for distributed testing
class MockCoordinator:
    """Mock implementation of distributed testing coordinator."""
    
    def __init__(self, port=8765):
        self.port = port
        self.workers = {}
        self.tasks = queue.Queue()
        self.results = queue.Queue()
        self.running = False
        self.server_thread = None
        self.task_id_counter = 0
        self.worker_id_counter = 0
    
    def start(self):
        """Start the coordinator."""
        self.running = True
        self.server_thread = threading.Thread(target=self._run_server)
        self.server_thread.daemon = True
        self.server_thread.start()
        return self.server_thread
    
    def stop(self):
        """Stop the coordinator."""
        self.running = False
        if self.server_thread:
            self.server_thread.join(timeout=1.0)
    
    def _run_server(self):
        """Run the coordinator server."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
                server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                server.bind(('localhost', self.port))
                server.listen()
                server.settimeout(0.5)
                
                while self.running:
                    try:
                        conn, addr = server.accept()
                        self._handle_client(conn, addr)
                    except socket.timeout:
                        continue
        except Exception as e:
            logging.error(f"Coordinator server error: {e}")
    
    def _handle_client(self, conn, addr):
        """Handle a client connection."""
        try:
            with conn:
                data = conn.recv(4096)
                if data:
                    message = json.loads(data.decode('utf-8'))
                    response = self._process_message(message, addr)
                    conn.sendall(json.dumps(response).encode('utf-8'))
        except Exception as e:
            logging.error(f"Error handling client {addr}: {e}")
    
    def _process_message(self, message, addr):
        """Process a message from a client."""
        msg_type = message.get('type')
        
        if msg_type == 'register_worker':
            worker_id = f"worker_{self.worker_id_counter}"
            self.worker_id_counter += 1
            
            self.workers[worker_id] = {
                'address': addr,
                'capabilities': message.get('capabilities', {}),
                'status': 'idle',
                'last_seen': time.time()
            }
            
            return {
                'type': 'register_response',
                'worker_id': worker_id,
                'success': True
            }
        
        elif msg_type == 'get_task':
            worker_id = message.get('worker_id')
            if worker_id not in self.workers:
                return {'type': 'error', 'message': 'Unknown worker'}
            
            try:
                task = self.tasks.get_nowait()
                self.workers[worker_id]['status'] = 'busy'
                return {
                    'type': 'task',
                    'task_id': task['task_id'],
                    'test_file': task['test_file'],
                    'parameters': task.get('parameters', {})
                }
            except queue.Empty:
                return {'type': 'no_task'}
        
        elif msg_type == 'submit_result':
            worker_id = message.get('worker_id')
            if worker_id not in self.workers:
                return {'type': 'error', 'message': 'Unknown worker'}
            
            self.workers[worker_id]['status'] = 'idle'
            self.results.put({
                'task_id': message.get('task_id'),
                'worker_id': worker_id,
                'success': message.get('success', False),
                'output': message.get('output', ''),
                'error': message.get('error', None)
            })
            
            return {'type': 'result_accepted'}
        
        elif msg_type == 'heartbeat':
            worker_id = message.get('worker_id')
            if worker_id in self.workers:
                self.workers[worker_id]['last_seen'] = time.time()
            
            return {'type': 'heartbeat_response'}
        
        else:
            return {'type': 'error', 'message': 'Unknown message type'}
    
    def add_task(self, test_file, parameters=None):
        """Add a task to the queue."""
        task_id = f"task_{self.task_id_counter}"
        self.task_id_counter += 1
        
        task = {
            'task_id': task_id,
            'test_file': test_file,
            'parameters': parameters or {}
        }
        
        self.tasks.put(task)
        return task_id
    
    def get_result(self, timeout=None):
        """Get a result from the queue."""
        try:
            return self.results.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_worker_count(self):
        """Get the number of registered workers."""
        return len(self.workers)
    
    def get_idle_worker_count(self):
        """Get the number of idle workers."""
        return sum(1 for worker in self.workers.values() if worker['status'] == 'idle')
    
    def get_busy_worker_count(self):
        """Get the number of busy workers."""
        return sum(1 for worker in self.workers.values() if worker['status'] == 'busy')
    
    def get_worker_status(self, worker_id):
        """Get a worker's status."""
        if worker_id in self.workers:
            return self.workers[worker_id]['status']
        return None

class MockWorker:
    """Mock implementation of distributed testing worker."""
    
    def __init__(self, coordinator_address, capabilities=None):
        self.coordinator_address = coordinator_address
        self.capabilities = capabilities or {
            'hardware': ['cpu'],
            'browsers': []
        }
        self.worker_id = None
        self.running = False
        self.worker_thread = None
        self.tasks_processed = 0
        self.task_results = []
    
    def connect(self):
        """Connect to the coordinator and register."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
                host, port = self.coordinator_address.split(':')
                client.connect((host, int(port)))
                
                message = {
                    'type': 'register_worker',
                    'capabilities': self.capabilities
                }
                
                client.sendall(json.dumps(message).encode('utf-8'))
                data = client.recv(4096)
                response = json.loads(data.decode('utf-8'))
                
                if response.get('success'):
                    self.worker_id = response.get('worker_id')
                    return True
                
                return False
        except Exception as e:
            logging.error(f"Worker connection error: {e}")
            return False
    
    def start(self):
        """Start the worker."""
        if not self.worker_id:
            if not self.connect():
                return False
        
        self.running = True
        self.worker_thread = threading.Thread(target=self._run_worker)
        self.worker_thread.daemon = True
        self.worker_thread.start()
        return True
    
    def stop(self):
        """Stop the worker."""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=1.0)
    
    def _run_worker(self):
        """Run the worker thread."""
        while self.running:
            try:
                # Get a task
                task = self._get_task()
                if task:
                    # Process the task
                    result = self._process_task(task)
                    # Submit the result
                    self._submit_result(task, result)
            except Exception as e:
                logging.error(f"Worker error: {e}")
            
            # Sleep a bit to avoid hammering the coordinator
            time.sleep(0.1)
    
    def _get_task(self):
        """Get a task from the coordinator."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
                host, port = self.coordinator_address.split(':')
                client.connect((host, int(port)))
                
                message = {
                    'type': 'get_task',
                    'worker_id': self.worker_id
                }
                
                client.sendall(json.dumps(message).encode('utf-8'))
                data = client.recv(4096)
                response = json.loads(data.decode('utf-8'))
                
                if response.get('type') == 'task':
                    return {
                        'task_id': response.get('task_id'),
                        'test_file': response.get('test_file'),
                        'parameters': response.get('parameters', {})
                    }
                
                return None
        except Exception as e:
            logging.error(f"Worker get_task error: {e}")
            return None
    
    def _process_task(self, task):
        """Process a task."""
        # Simulate task processing
        time.sleep(0.2)  # Simulate work
        
        # 90% success rate
        import random
        success = random.random() < 0.9
        
        self.tasks_processed += 1
        
        result = {
            'success': success,
            'output': f"Processed {task['test_file']} with parameters {task['parameters']}",
            'error': None if success else "Simulated task failure"
        }
        
        self.task_results.append({
            'task_id': task['task_id'],
            'result': result
        })
        
        return result
    
    def _submit_result(self, task, result):
        """Submit a result to the coordinator."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
                host, port = self.coordinator_address.split(':')
                client.connect((host, int(port)))
                
                message = {
                    'type': 'submit_result',
                    'worker_id': self.worker_id,
                    'task_id': task['task_id'],
                    'success': result['success'],
                    'output': result['output'],
                    'error': result['error']
                }
                
                client.sendall(json.dumps(message).encode('utf-8'))
                data = client.recv(4096)
                response = json.loads(data.decode('utf-8'))
                
                return response.get('type') == 'result_accepted'
        except Exception as e:
            logging.error(f"Worker submit_result error: {e}")
            return False
    
    def send_heartbeat(self):
        """Send a heartbeat to the coordinator."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
                host, port = self.coordinator_address.split(':')
                client.connect((host, int(port)))
                
                message = {
                    'type': 'heartbeat',
                    'worker_id': self.worker_id
                }
                
                client.sendall(json.dumps(message).encode('utf-8'))
                data = client.recv(4096)
                response = json.loads(data.decode('utf-8'))
                
                return response.get('type') == 'heartbeat_response'
        except Exception as e:
            logging.error(f"Worker heartbeat error: {e}")
            return False

# Test fixtures
@pytest.fixture
def coordinator():
    """Create a mock coordinator for testing."""
    # Use a random port to avoid conflicts
    import random
    port = random.randint(10000, 20000)
    
    coordinator = MockCoordinator(port=port)
    coordinator.start()
    
    yield coordinator
    
    coordinator.stop()

@pytest.fixture
def coordinator_address(coordinator):
    """Get the coordinator address."""
    return f"localhost:{coordinator.port}"

@pytest.fixture
def worker(coordinator_address):
    """Create a mock worker for testing."""
    worker = MockWorker(coordinator_address)
    
    yield worker
    
    worker.stop()

@pytest.fixture
def multi_workers(coordinator_address):
    """Create multiple mock workers for testing."""
    workers = []
    
    # Create workers with different capabilities
    workers.append(MockWorker(coordinator_address, capabilities={
        'hardware': ['cpu'],
        'browsers': []
    }))
    
    workers.append(MockWorker(coordinator_address, capabilities={
        'hardware': ['cpu', 'cuda'],
        'browsers': []
    }))
    
    workers.append(MockWorker(coordinator_address, capabilities={
        'hardware': ['cpu'],
        'browsers': ['chrome']
    }))
    
    # Start all workers
    for worker in workers:
        worker.start()
    
    yield workers
    
    # Stop all workers
    for worker in workers:
        worker.stop()

@pytest.mark.integration
@pytest.mark.distributed
class TestDistributedCoordinator:
    """
    Tests for distributed testing coordinator.
    
    These tests verify that the coordinator can manage distributed
    test execution across multiple worker nodes.
    """
    
    def test_coordinator_start(self, coordinator):
        """Test that the coordinator starts successfully."""
        assert coordinator.running
        assert coordinator.server_thread.is_alive()
    
    def test_worker_registration(self, coordinator, worker):
        """Test worker registration with the coordinator."""
        # Connect and register the worker
        success = worker.connect()
        assert success
        assert worker.worker_id is not None
        
        # Verify worker is registered with the coordinator
        assert coordinator.get_worker_count() == 1
        assert coordinator.get_idle_worker_count() == 1
        assert coordinator.get_busy_worker_count() == 0
    
    def test_task_distribution(self, coordinator, worker):
        """Test task distribution to workers."""
        # Connect the worker
        worker.connect()
        
        # Add some tasks
        task_ids = []
        for i in range(5):
            task_id = coordinator.add_task(f"test_{i}.py")
            task_ids.append(task_id)
        
        # Start the worker
        worker.start()
        
        # Wait for worker to process tasks
        max_wait = 10  # seconds
        start_time = time.time()
        while worker.tasks_processed < 5:
            if time.time() - start_time > max_wait:
                pytest.fail(f"Worker processed only {worker.tasks_processed} tasks in {max_wait} seconds")
            time.sleep(0.1)
        
        # All tasks should be processed
        assert worker.tasks_processed == 5
        
        # Check results
        results_count = 0
        while not coordinator.results.empty():
            result = coordinator.get_result()
            assert result is not None
            assert result['task_id'] in task_ids
            results_count += 1
        
        assert results_count == 5
    
    def test_multiple_workers(self, coordinator, multi_workers):
        """Test task distribution among multiple workers."""
        # Add some tasks
        for i in range(10):
            coordinator.add_task(f"test_{i}.py")
        
        # Wait for workers to process tasks
        max_wait = 10  # seconds
        start_time = time.time()
        
        while sum(w.tasks_processed for w in multi_workers) < 10:
            if time.time() - start_time > max_wait:
                processed = sum(w.tasks_processed for w in multi_workers)
                pytest.fail(f"Workers processed only {processed} tasks in {max_wait} seconds")
            time.sleep(0.1)
        
        # All tasks should be processed
        assert sum(w.tasks_processed for w in multi_workers) == 10
        
        # Each worker should have processed at least one task
        for worker in multi_workers:
            assert worker.tasks_processed > 0
        
        # Check results
        results_count = 0
        while not coordinator.results.empty():
            result = coordinator.get_result()
            assert result is not None
            results_count += 1
        
        assert results_count == 10
    
    def test_worker_heartbeat(self, coordinator, worker):
        """Test worker heartbeat mechanism."""
        # Connect the worker
        worker.connect()
        
        # Send a heartbeat
        response = worker.send_heartbeat()
        assert response
        
        # Check last seen time
        assert worker.worker_id in coordinator.workers
        assert coordinator.workers[worker.worker_id]['last_seen'] > 0
        
        # Send another heartbeat after a delay
        initial_time = coordinator.workers[worker.worker_id]['last_seen']
        time.sleep(0.2)
        worker.send_heartbeat()
        
        # Last seen time should be updated
        assert coordinator.workers[worker.worker_id]['last_seen'] > initial_time
    
    def test_capability_based_task_distribution(self, coordinator):
        """Test task distribution based on worker capabilities."""
        # Create workers with different capabilities
        cpu_worker = MockWorker(f"localhost:{coordinator.port}", capabilities={
            'hardware': ['cpu'],
            'browsers': []
        })
        gpu_worker = MockWorker(f"localhost:{coordinator.port}", capabilities={
            'hardware': ['cpu', 'cuda'],
            'browsers': []
        })
        
        # Connect and start workers
        cpu_worker.connect()
        gpu_worker.connect()
        cpu_worker.start()
        gpu_worker.start()
        
        # Add tasks with specific hardware requirements
        cpu_tasks = []
        for i in range(3):
            task_id = coordinator.add_task(f"cpu_test_{i}.py", 
                                          {'required_hardware': 'cpu'})
            cpu_tasks.append(task_id)
        
        gpu_tasks = []
        for i in range(3):
            task_id = coordinator.add_task(f"gpu_test_{i}.py", 
                                          {'required_hardware': 'cuda'})
            gpu_tasks.append(task_id)
        
        # Wait for workers to process tasks
        max_wait = 10  # seconds
        start_time = time.time()
        
        while (cpu_worker.tasks_processed + gpu_worker.tasks_processed) < 6:
            if time.time() - start_time > max_wait:
                processed = cpu_worker.tasks_processed + gpu_worker.tasks_processed
                pytest.fail(f"Workers processed only {processed} tasks in {max_wait} seconds")
            time.sleep(0.1)
        
        # Stop workers
        cpu_worker.stop()
        gpu_worker.stop()
        
        # All tasks should be processed
        assert cpu_worker.tasks_processed + gpu_worker.tasks_processed == 6
        
        # GPU worker should process more tasks (both CPU and GPU tasks)
        assert gpu_worker.tasks_processed >= 3
        
        # Check results to ensure all tasks were processed
        results_count = 0
        while not coordinator.results.empty():
            result = coordinator.get_result()
            assert result is not None
            results_count += 1
        
        assert results_count == 6
    
    def test_fault_tolerance(self, coordinator):
        """Test fault tolerance when a worker fails."""
        # Create workers
        worker1 = MockWorker(f"localhost:{coordinator.port}")
        worker2 = MockWorker(f"localhost:{coordinator.port}")
        
        # Connect and start workers
        worker1.connect()
        worker2.connect()
        worker1.start()
        
        # Add tasks
        for i in range(5):
            coordinator.add_task(f"test_{i}.py")
        
        # Let worker1 process some tasks
        time.sleep(1)
        
        # Stop worker1 to simulate failure
        worker1.stop()
        
        # Start worker2 to take over
        worker2.start()
        
        # Wait for all tasks to be processed
        max_wait = 10  # seconds
        start_time = time.time()
        
        while not coordinator.tasks.empty():
            if time.time() - start_time > max_wait:
                pytest.fail(f"Not all tasks were processed in {max_wait} seconds")
            time.sleep(0.1)
        
        # Wait a bit for results to be submitted
        time.sleep(1)
        
        # All tasks should have been processed by the two workers
        assert worker1.tasks_processed + worker2.tasks_processed == 5
        
        # Check results
        results_count = 0
        while not coordinator.results.empty():
            result = coordinator.get_result()
            assert result is not None
            results_count += 1
        
        assert results_count == 5
        
        # Stop worker2
        worker2.stop()
    
    @pytest.mark.skip(reason="This test is just an example of high availability testing")
    def test_coordinator_high_availability(self):
        """Test coordinator high availability features."""
        # This would test advanced features like:
        # - Leader election
        # - State replication
        # - Automatic failover
        # - Split brain prevention
        pass