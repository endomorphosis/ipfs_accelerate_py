#!/usr/bin/env python3
"""
Test scenarios for coordinator failover in the Distributed Testing Framework.
Focuses on high-availability cluster behavior during leader failure and recovery.
"""

import anyio
import os
import sys
import unittest
import tempfile
import time
import json
import logging
import subprocess
import signal
from unittest.mock import patch, MagicMock, AsyncMock

import pytest

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from integration_mode import real_integration_enabled, integration_opt_in_message

# This suite spawns multiple coordinator processes, binds local ports, and
# requires timing-sensitive leader election/state replication. Treat it as
# REAL integration rather than CI-safe SIMULATED.
if not real_integration_enabled():
    pytest.skip(integration_opt_in_message(), allow_module_level=True)

pytest.importorskip("aiohttp")

from coordinator_redundancy import RedundancyManager, NodeRole
from coordinator import DistributedTestingCoordinator

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FailoverSimulator:
    """Simulates a cluster of coordinator nodes with failure scenarios."""
    
    def __init__(self, node_count=3, base_port=8080):
        self.node_count = node_count
        self.base_port = base_port
        self.processes = []
        self.temp_dirs = [tempfile.mkdtemp() for _ in range(node_count)]
        self.db_paths = [os.path.join(temp_dir, "coordinator.duckdb") for temp_dir in self.temp_dirs]

        # Subprocesses are launched via `python -m distributed_testing.coordinator`.
        # That module is located under the repository's `test/` directory, so we
        # ensure the subprocess working directory / PYTHONPATH includes it.
        self._test_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir)
        )

    def _subprocess_env(self):
        test_root = self._test_root
        pythonpath = os.environ.get("PYTHONPATH", "")
        if pythonpath:
            pythonpath = f"{test_root}{os.pathsep}{pythonpath}"
        else:
            pythonpath = test_root
        env = dict(os.environ)
        env["PYTHONPATH"] = pythonpath
        env.setdefault("PYTHONUNBUFFERED", "1")
        return env
        
    async def start_cluster(self):
        """Start a cluster of coordinator nodes."""
        for i in range(self.node_count):
            node_id = f"node-{i+1}"
            port = self.base_port + i
            peers = ",".join([f"localhost:{self.base_port+j}" for j in range(self.node_count) if j != i])
            
            # Create command for starting a coordinator node
            cmd = [
                sys.executable,
                "-m", "distributed_testing.coordinator",
                "--id", node_id,
                "--port", str(port),
                "--db-path", self.db_paths[i],
                "--data-dir", self.temp_dirs[i],
                "--enable-redundancy",
                "--peers", peers,
                "--log-level", "INFO"
            ]
            
            # Start the process
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                cwd=self._test_root,
                env=self._subprocess_env(),
            )
            self.processes.append(process)
            
        # Wait for cluster to stabilize
        await anyio.sleep(5)
        
    def kill_node(self, node_index):
        """Kill a specific node to simulate failure."""
        if 0 <= node_index < len(self.processes):
            process = self.processes[node_index]
            process.send_signal(signal.SIGTERM)
            process.wait()
            logger.info(f"Killed node {node_index+1}")
        
    def restart_node(self, node_index):
        """Restart a previously killed node."""
        if 0 <= node_index < len(self.processes):
            node_id = f"node-{node_index+1}"
            port = self.base_port + node_index
            peers = ",".join([f"localhost:{self.base_port+j}" for j in range(self.node_count) if j != node_index])
            
            # Create command for starting a coordinator node
            cmd = [
                sys.executable,
                "-m", "distributed_testing.coordinator",
                "--id", node_id,
                "--port", str(port),
                "--db-path", self.db_paths[node_index],
                "--data-dir", self.temp_dirs[node_index],
                "--enable-redundancy",
                "--peers", peers,
                "--log-level", "INFO"
            ]
            
            # Start the process
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                cwd=self._test_root,
                env=self._subprocess_env(),
            )
            self.processes[node_index] = process
            logger.info(f"Restarted node {node_index+1}")
            
    async def stop_cluster(self):
        """Stop all nodes in the cluster."""
        for i, process in enumerate(self.processes):
            if process.poll() is None:  # Only if still running
                process.send_signal(signal.SIGTERM)
                process.wait()
                logger.info(f"Stopped node {i+1}")
                
        # Clean up temporary directories
        for temp_dir in self.temp_dirs:
            try:
                import shutil
                shutil.rmtree(temp_dir)
            except Exception as e:
                logger.warning(f"Error cleaning up temp dir: {e}")
                
    async def get_cluster_status(self):
        """Get the status of each node in the cluster."""
        status = []
        
        for i in range(self.node_count):
            port = self.base_port + i
            url = f"http://localhost:{port}/api/status"
            
            try:
                import aiohttp
                async with aiohttp.ClientSession() as session:
                    async with session.get(url) as response:
                        if response.status == 200:
                            data = await response.json()
                            status.append({
                                "node_id": f"node-{i+1}",
                                "port": port,
                                "role": data.get("role"),
                                "term": data.get("term"),
                                "leader": data.get("current_leader"),
                                "status": "running"
                            })
                        else:
                            status.append({
                                "node_id": f"node-{i+1}",
                                "port": port,
                                "status": "error",
                                "error": f"HTTP {response.status}"
                            })
            except Exception as e:
                status.append({
                    "node_id": f"node-{i+1}",
                    "port": port,
                    "status": "unreachable",
                    "error": str(e)
                })
                
        return status
    
    async def find_leader(self):
        """Find the current leader in the cluster."""
        status = await self.get_cluster_status()
        for node in status:
            if node.get("status") == "running" and node.get("role") == "LEADER":
                return node
        return None
        
    async def add_worker(self, worker_id, host="worker-host", port=9000):
        """Add a worker to the cluster through any available node."""
        for i in range(self.node_count):
            node_port = self.base_port + i
            url = f"http://localhost:{node_port}/api/workers/register"
            
            try:
                import aiohttp
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, json={
                        "worker_id": worker_id,
                        "host": host,
                        "port": port
                    }) as response:
                        if response.status == 200:
                            return await response.json()
            except Exception:
                continue
                
        raise Exception("Failed to add worker - no coordinator nodes available")
        
    async def get_workers(self):
        """Get the list of workers from any available node."""
        for i in range(self.node_count):
            node_port = self.base_port + i
            url = f"http://localhost:{node_port}/api/workers"
            
            try:
                import aiohttp
                async with aiohttp.ClientSession() as session:
                    async with session.get(url) as response:
                        if response.status == 200:
                            return await response.json()
            except Exception:
                continue
                
        raise Exception("Failed to get workers - no coordinator nodes available")


class TestCoordinatorFailover(unittest.TestCase):
    """Test failover scenarios for coordinator redundancy implementation."""
    
    @classmethod
    def setUpClass(cls):
        """Set up the test environment once for all tests."""
        # Ensure aiohttp is available
        try:
            import aiohttp
        except ImportError:
            raise unittest.SkipTest("aiohttp is required for these tests")
            
        # Create event loop for async tests
        import asyncio
        cls.event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(cls.event_loop)
        
    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests."""
        cls.event_loop.close()
        
    def setUp(self):
        """Set up before each test."""
        self.simulator = FailoverSimulator(node_count=3, base_port=18080)
        
    def tearDown(self):
        """Clean up after each test."""
        # Stop the cluster
        self.event_loop.run_until_complete(self.simulator.stop_cluster())
        
    def test_basic_failover(self):
        """Test basic leader failover when the leader node fails."""
        # Start the cluster
        self.event_loop.run_until_complete(self.simulator.start_cluster())
        
        # Wait for leader election
        leader = self.event_loop.run_until_complete(self.simulator.find_leader())
        self.assertIsNotNone(leader, "Leader should be elected")
        leader_index = int(leader["node_id"].split("-")[1]) - 1
        
        # Add a worker to verify state replication
        worker_result = self.event_loop.run_until_complete(
            self.simulator.add_worker("worker-1", "test-host", 9001)
        )
        self.assertTrue(worker_result.get("success"), "Adding worker should succeed")
        
        # Get initial workers list
        workers_before = self.event_loop.run_until_complete(self.simulator.get_workers())
        self.assertIn("worker-1", workers_before, "Worker should be in the list")
        
        # Kill the leader
        self.simulator.kill_node(leader_index)
        
        # Wait for new leader election
        self.event_loop.run_until_complete(anyio.sleep(10))
        
        # Find the new leader
        new_leader = self.event_loop.run_until_complete(self.simulator.find_leader())
        self.assertIsNotNone(new_leader, "New leader should be elected")
        self.assertNotEqual(new_leader["node_id"], leader["node_id"], 
                          "New leader should be different from old leader")
        
        # Verify state consistency by getting workers list
        workers_after = self.event_loop.run_until_complete(self.simulator.get_workers())
        self.assertIn("worker-1", workers_after, 
                    "Worker should still be in the list after failover")
        
    def test_leader_rejoin(self):
        """Test behavior when a failed leader rejoins the cluster."""
        # Start the cluster
        self.event_loop.run_until_complete(self.simulator.start_cluster())
        
        # Wait for leader election
        leader = self.event_loop.run_until_complete(self.simulator.find_leader())
        self.assertIsNotNone(leader, "Leader should be elected")
        leader_index = int(leader["node_id"].split("-")[1]) - 1
        
        # Add some state
        self.event_loop.run_until_complete(
            self.simulator.add_worker("worker-1", "test-host", 9001)
        )
        
        # Kill the leader
        self.simulator.kill_node(leader_index)
        
        # Wait for new leader election
        self.event_loop.run_until_complete(anyio.sleep(10))
        
        # Find the new leader
        new_leader = self.event_loop.run_until_complete(self.simulator.find_leader())
        self.assertIsNotNone(new_leader, "New leader should be elected")
        
        # Add more state with new leader
        self.event_loop.run_until_complete(
            self.simulator.add_worker("worker-2", "test-host-2", 9002)
        )
        
        # Restart the old leader
        self.simulator.restart_node(leader_index)
        
        # Wait for the node to sync
        self.event_loop.run_until_complete(anyio.sleep(5))
        
        # Get cluster status
        status = self.event_loop.run_until_complete(self.simulator.get_cluster_status())
        
        # Find the restarted node
        restarted_node = next((node for node in status 
                             if node["node_id"] == f"node-{leader_index+1}" 
                             and node["status"] == "running"), None)
        
        self.assertIsNotNone(restarted_node, "Restarted node should be running")
        self.assertEqual(restarted_node["role"], "FOLLOWER", 
                       "Restarted node should be a follower")
        
        # Verify state consistency by getting workers list from the restarted node
        for i in range(3):
            if i == leader_index:
                node_port = self.simulator.base_port + i
                url = f"http://localhost:{node_port}/api/workers"
                
                try:
                    import aiohttp
                    response_data = self.event_loop.run_until_complete(
                        self._get_request(url)
                    )
                    self.assertIn("worker-1", response_data,
                                "worker-1 should be in restarted node's state")
                    self.assertIn("worker-2", response_data,
                                "worker-2 should be in restarted node's state")
                except Exception as e:
                    self.fail(f"Failed to get workers from restarted node: {e}")
                
    def test_majority_failure(self):
        """Test behavior when majority of nodes fail (should lose availability)."""
        # Start the cluster
        self.event_loop.run_until_complete(self.simulator.start_cluster())
        
        # Wait for leader election
        leader = self.event_loop.run_until_complete(self.simulator.find_leader())
        self.assertIsNotNone(leader, "Leader should be elected")
        
        # Add a worker
        self.event_loop.run_until_complete(
            self.simulator.add_worker("worker-1", "test-host", 9001)
        )
        
        # Kill majority of nodes (2 out of 3)
        for i in range(2):
            self.simulator.kill_node(i)
            
        # Wait a moment
        self.event_loop.run_until_complete(anyio.sleep(5))
        
        # Try to add another worker (should fail - no quorum)
        with self.assertRaises(Exception):
            self.event_loop.run_until_complete(
                self.simulator.add_worker("worker-2", "test-host-2", 9002)
            )
            
        # Restart one node to restore quorum
        self.simulator.restart_node(0)
        
        # Wait for cluster to stabilize
        self.event_loop.run_until_complete(anyio.sleep(10))
        
        # Should have a leader now
        leader = self.event_loop.run_until_complete(self.simulator.find_leader())
        self.assertIsNotNone(leader, "Leader should be elected after quorum restored")
        
        # Should be able to add workers again
        try:
            result = self.event_loop.run_until_complete(
                self.simulator.add_worker("worker-3", "test-host-3", 9003)
            )
            self.assertTrue(result.get("success"), "Adding worker should succeed after quorum")
        except Exception as e:
            self.fail(f"Failed to add worker after quorum restored: {e}")
            
    def test_term_advancement(self):
        """Test term advancement during sequential leader failures."""
        # Start the cluster
        self.event_loop.run_until_complete(self.simulator.start_cluster())
        
        # Wait for initial leader election
        leader1 = self.event_loop.run_until_complete(self.simulator.find_leader())
        self.assertIsNotNone(leader1, "Leader should be elected")
        leader1_index = int(leader1["node_id"].split("-")[1]) - 1
        term1 = leader1["term"]
        
        # Kill the leader
        self.simulator.kill_node(leader1_index)
        
        # Wait for new leader election
        self.event_loop.run_until_complete(anyio.sleep(10))
        
        # Find the new leader
        leader2 = self.event_loop.run_until_complete(self.simulator.find_leader())
        self.assertIsNotNone(leader2, "New leader should be elected")
        leader2_index = int(leader2["node_id"].split("-")[1]) - 1
        term2 = leader2["term"]
        
        # Term should have advanced
        self.assertGreater(term2, term1, "Term should advance after leader failure")
        
        # Kill the second leader
        self.simulator.kill_node(leader2_index)
        
        # Wait for third leader election
        self.event_loop.run_until_complete(anyio.sleep(10))
        
        # Find the third leader
        leader3 = self.event_loop.run_until_complete(self.simulator.find_leader())
        self.assertIsNotNone(leader3, "Third leader should be elected")
        term3 = leader3["term"]
        
        # Term should have advanced again
        self.assertGreater(term3, term2, "Term should advance after second leader failure")
        
    async def _get_request(self, url):
        """Helper method to make GET requests."""
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status != 200:
                    raise Exception(f"HTTP request failed with status {response.status}")
                return await response.json()
                

if __name__ == "__main__":
    unittest.main()