#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Unit tests for the Cloud Provider Manager component of the Dynamic Resource Management system.
"""

import unittest
import json
import os
import sys
import tempfile
from unittest.mock import MagicMock, patch, mock_open

# Add parent directory to path to import modules correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cloud_provider_integration import CloudProviderManager, CloudProviderBase


class MockAWSProvider(CloudProviderBase):
    """Mock AWS cloud provider for testing."""
    
    def __init__(self, config):
        """Initialize provider with configuration."""
        super().__init__("aws", config)
        self.instances = {}
        self.next_instance_id = 1
    
    def create_worker(self, resources=None, worker_type=None):
        """Create a worker instance."""
        instance_id = f"i-{self.next_instance_id:08x}"
        self.next_instance_id += 1
        self.instances[instance_id] = {
            "status": "running",
            "resources": resources,
            "worker_type": worker_type,
            "instance_type": self.config["instance_types"].get(
                worker_type, self.config["instance_types"].get("default", "t3.medium")
            )
        }
        return {
            "worker_id": instance_id,
            "status": "running",
            "provider": "aws",
            "endpoint": f"http://{instance_id}.example.com:8080"
        }
    
    def terminate_worker(self, worker_id):
        """Terminate a worker instance."""
        if worker_id in self.instances:
            self.instances[worker_id]["status"] = "terminated"
            return True
        return False
    
    def get_worker_status(self, worker_id):
        """Get the status of a worker instance."""
        if worker_id in self.instances:
            return {
                "worker_id": worker_id,
                "status": self.instances[worker_id]["status"],
                "provider": "aws"
            }
        return None
    
    def get_available_resources(self):
        """Get available resources on this provider."""
        return {
            "instance_types": list(self.config["instance_types"].values()),
            "regions": [self.config["region"]],
            "max_instances": 10
        }


class MockGCPProvider(CloudProviderBase):
    """Mock GCP cloud provider for testing."""
    
    def __init__(self, config):
        """Initialize provider with configuration."""
        super().__init__("gcp", config)
        self.instances = {}
        self.next_instance_id = 1
    
    def create_worker(self, resources=None, worker_type=None):
        """Create a worker instance."""
        instance_id = f"gcp-{self.next_instance_id:08x}"
        self.next_instance_id += 1
        self.instances[instance_id] = {
            "status": "running",
            "resources": resources,
            "worker_type": worker_type,
            "machine_type": self.config["machine_types"].get(
                worker_type, self.config["machine_types"].get("default", "n1-standard-2")
            )
        }
        return {
            "worker_id": instance_id,
            "status": "running",
            "provider": "gcp",
            "endpoint": f"http://{instance_id}.example.com:8080"
        }
    
    def terminate_worker(self, worker_id):
        """Terminate a worker instance."""
        if worker_id in self.instances:
            self.instances[worker_id]["status"] = "terminated"
            return True
        return False
    
    def get_worker_status(self, worker_id):
        """Get the status of a worker instance."""
        if worker_id in self.instances:
            return {
                "worker_id": worker_id,
                "status": self.instances[worker_id]["status"],
                "provider": "gcp"
            }
        return None
    
    def get_available_resources(self):
        """Get available resources on this provider."""
        return {
            "machine_types": list(self.config["machine_types"].values()),
            "zones": [self.config["zone"]],
            "max_instances": 8
        }


class MockDockerProvider(CloudProviderBase):
    """Mock Docker local provider for testing."""
    
    def __init__(self, config):
        """Initialize provider with configuration."""
        super().__init__("docker_local", config)
        self.containers = {}
        self.next_container_id = 1
    
    def create_worker(self, resources=None, worker_type=None):
        """Create a worker container."""
        container_id = f"container-{self.next_container_id:08x}"
        self.next_container_id += 1
        self.containers[container_id] = {
            "status": "running",
            "resources": resources,
            "worker_type": worker_type,
            "image": self.config["image"]
        }
        return {
            "worker_id": container_id,
            "status": "running",
            "provider": "docker_local",
            "endpoint": f"http://localhost:{8080 + self.next_container_id - 1}"
        }
    
    def terminate_worker(self, worker_id):
        """Terminate a worker container."""
        if worker_id in self.containers:
            self.containers[worker_id]["status"] = "stopped"
            return True
        return False
    
    def get_worker_status(self, worker_id):
        """Get the status of a worker container."""
        if worker_id in self.containers:
            return {
                "worker_id": worker_id,
                "status": self.containers[worker_id]["status"],
                "provider": "docker_local"
            }
        return None
    
    def get_available_resources(self):
        """Get available resources on this provider."""
        return {
            "cpu_limit": self.config["cpu_limit"],
            "memory_limit": self.config["memory_limit"],
            "max_containers": 5
        }


class TestCloudProviderManager(unittest.TestCase):
    """Test suite for CloudProviderManager class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Sample provider configuration
        self.config = {
            "aws": {
                "region": "us-west-2",
                "instance_types": {
                    "cpu": "c5.xlarge",
                    "gpu": "g4dn.xlarge",
                    "default": "t3.medium"
                },
                "credentials": {
                    "access_key_id": "mock-access-key",
                    "secret_access_key": "mock-secret-key"
                },
                "spot_instance_enabled": True
            },
            "gcp": {
                "project": "mock-project",
                "zone": "us-central1-a",
                "machine_types": {
                    "cpu": "n2-standard-4",
                    "gpu": "n1-standard-4",
                    "default": "n1-standard-2"
                },
                "credentials_file": "/path/to/credentials.json",
                "preemptible_enabled": True
            },
            "docker_local": {
                "image": "ipfs-accelerate-worker:latest",
                "cpu_limit": 4,
                "memory_limit": "16g",
                "network": "host"
            }
        }
        
        # Write config to temp file
        self.config_path = os.path.join(self.temp_dir.name, "cloud_config.json")
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f)
        
        # Initialize manager with mock providers
        self.manager = CloudProviderManager(self.config_path)
        
        # Register mock providers
        self.manager.providers["aws"] = MockAWSProvider(self.config["aws"])
        self.manager.providers["gcp"] = MockGCPProvider(self.config["gcp"])
        self.manager.providers["docker_local"] = MockDockerProvider(self.config["docker_local"])

    def tearDown(self):
        """Tear down test fixtures."""
        self.temp_dir.cleanup()

    def test_init_with_config_path(self):
        """Test initialization with config path."""
        manager = CloudProviderManager(self.config_path)
        self.assertEqual(manager.config_path, self.config_path)
        self.assertIsNotNone(manager.providers)
        self.assertEqual(len(manager.providers), 0)  # No providers registered yet

    def test_init_with_config_dict(self):
        """Test initialization with config dict."""
        manager = CloudProviderManager(config=self.config)
        self.assertIsNone(manager.config_path)
        self.assertEqual(manager.config, self.config)
        self.assertIsNotNone(manager.providers)
        self.assertEqual(len(manager.providers), 0)  # No providers registered yet

    def test_load_config(self):
        """Test loading configuration from file."""
        manager = CloudProviderManager()
        manager.load_config(self.config_path)
        
        self.assertEqual(manager.config_path, self.config_path)
        self.assertEqual(manager.config["aws"]["region"], "us-west-2")
        self.assertEqual(manager.config["gcp"]["project"], "mock-project")
        self.assertEqual(manager.config["docker_local"]["image"], "ipfs-accelerate-worker:latest")

    def test_add_provider(self):
        """Test adding a provider."""
        # Create a fresh manager
        manager = CloudProviderManager(config=self.config)
        
        # Add providers
        manager.add_provider("aws", MockAWSProvider(self.config["aws"]))
        manager.add_provider("gcp", MockGCPProvider(self.config["gcp"]))
        
        # Check providers are registered
        self.assertIn("aws", manager.providers)
        self.assertIn("gcp", manager.providers)
        self.assertIsInstance(manager.providers["aws"], MockAWSProvider)
        self.assertIsInstance(manager.providers["gcp"], MockGCPProvider)

    def test_create_worker(self):
        """Test creating a worker."""
        # Create a worker on AWS
        result = self.manager.create_worker(
            provider="aws",
            resources={"cpu_cores": 4, "memory_mb": 16384},
            worker_type="cpu"
        )
        
        self.assertIsNotNone(result)
        self.assertIn("worker_id", result)
        self.assertEqual(result["status"], "running")
        self.assertEqual(result["provider"], "aws")
        
        # Create a worker on GCP
        result = self.manager.create_worker(
            provider="gcp",
            resources={"cpu_cores": 8, "memory_mb": 32768, "gpu_memory_mb": 16384},
            worker_type="gpu"
        )
        
        self.assertIsNotNone(result)
        self.assertIn("worker_id", result)
        self.assertEqual(result["status"], "running")
        self.assertEqual(result["provider"], "gcp")
        
        # Create a worker on Docker local
        result = self.manager.create_worker(
            provider="docker_local",
            resources={"cpu_cores": 2, "memory_mb": 8192},
            worker_type="cpu"
        )
        
        self.assertIsNotNone(result)
        self.assertIn("worker_id", result)
        self.assertEqual(result["status"], "running")
        self.assertEqual(result["provider"], "docker_local")
        
        # Try to create a worker on non-existent provider
        result = self.manager.create_worker(
            provider="nonexistent",
            resources={"cpu_cores": 4, "memory_mb": 8192}
        )
        
        self.assertIsNone(result)

    def test_terminate_worker(self):
        """Test terminating a worker."""
        # Create workers
        aws_result = self.manager.create_worker(provider="aws")
        gcp_result = self.manager.create_worker(provider="gcp")
        docker_result = self.manager.create_worker(provider="docker_local")
        
        # Terminate AWS worker
        result = self.manager.terminate_worker(
            provider="aws",
            worker_id=aws_result["worker_id"]
        )
        self.assertTrue(result)
        
        # Terminate GCP worker
        result = self.manager.terminate_worker(
            provider="gcp",
            worker_id=gcp_result["worker_id"]
        )
        self.assertTrue(result)
        
        # Terminate Docker worker
        result = self.manager.terminate_worker(
            provider="docker_local",
            worker_id=docker_result["worker_id"]
        )
        self.assertTrue(result)
        
        # Try to terminate non-existent worker
        result = self.manager.terminate_worker(
            provider="aws",
            worker_id="nonexistent"
        )
        self.assertFalse(result)
        
        # Try to terminate worker on non-existent provider
        result = self.manager.terminate_worker(
            provider="nonexistent",
            worker_id=aws_result["worker_id"]
        )
        self.assertFalse(result)

    def test_get_worker_status(self):
        """Test getting worker status."""
        # Create workers
        aws_result = self.manager.create_worker(provider="aws")
        gcp_result = self.manager.create_worker(provider="gcp")
        
        # Get AWS worker status
        status = self.manager.get_worker_status(
            provider="aws",
            worker_id=aws_result["worker_id"]
        )
        self.assertIsNotNone(status)
        self.assertEqual(status["worker_id"], aws_result["worker_id"])
        self.assertEqual(status["status"], "running")
        self.assertEqual(status["provider"], "aws")
        
        # Get GCP worker status
        status = self.manager.get_worker_status(
            provider="gcp",
            worker_id=gcp_result["worker_id"]
        )
        self.assertIsNotNone(status)
        self.assertEqual(status["worker_id"], gcp_result["worker_id"])
        self.assertEqual(status["status"], "running")
        self.assertEqual(status["provider"], "gcp")
        
        # Try to get non-existent worker status
        status = self.manager.get_worker_status(
            provider="aws",
            worker_id="nonexistent"
        )
        self.assertIsNone(status)
        
        # Try to get worker status from non-existent provider
        status = self.manager.get_worker_status(
            provider="nonexistent",
            worker_id=aws_result["worker_id"]
        )
        self.assertIsNone(status)

    def test_get_available_resources(self):
        """Test getting available resources."""
        # Get AWS resources
        resources = self.manager.get_available_resources(provider="aws")
        self.assertIsNotNone(resources)
        self.assertIn("instance_types", resources)
        self.assertIn("regions", resources)
        self.assertEqual(resources["regions"][0], "us-west-2")
        
        # Get GCP resources
        resources = self.manager.get_available_resources(provider="gcp")
        self.assertIsNotNone(resources)
        self.assertIn("machine_types", resources)
        self.assertIn("zones", resources)
        self.assertEqual(resources["zones"][0], "us-central1-a")
        
        # Get Docker resources
        resources = self.manager.get_available_resources(provider="docker_local")
        self.assertIsNotNone(resources)
        self.assertIn("cpu_limit", resources)
        self.assertIn("memory_limit", resources)
        self.assertEqual(resources["cpu_limit"], 4)
        
        # Try to get resources from non-existent provider
        resources = self.manager.get_available_resources(provider="nonexistent")
        self.assertIsNone(resources)

    def test_get_preferred_provider(self):
        """Test getting preferred provider based on requirements."""
        # Create a resource manager with mock providers and resource data
        manager = self.manager
        
        # Prefer AWS for GPU resources
        preferred = manager.get_preferred_provider(
            requirements={"gpu": True, "min_memory_gb": 16}
        )
        self.assertEqual(preferred, "aws")
        
        # Prefer Docker for local development
        preferred = manager.get_preferred_provider(
            requirements={"local": True}
        )
        self.assertEqual(preferred, "docker_local")
        
        # With no specific requirements, should return the first provider
        preferred = manager.get_preferred_provider(requirements={})
        self.assertEqual(preferred, "aws")  # First provider in our setup

    def test_get_all_providers(self):
        """Test getting all providers."""
        providers = self.manager.get_all_providers()
        self.assertEqual(len(providers), 3)
        self.assertIn("aws", providers)
        self.assertIn("gcp", providers)
        self.assertIn("docker_local", providers)


if __name__ == '__main__':
    unittest.main()