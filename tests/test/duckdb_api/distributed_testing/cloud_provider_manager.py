#!/usr/bin/env python3
"""
Distributed Testing Framework - Cloud Provider Manager

This module implements the CloudProviderManager for the distributed testing framework,
which handles the provisioning, management, and termination of worker nodes across
various cloud providers.

Core responsibilities:
- Provisioning worker nodes based on resource requirements
- Scaling worker nodes up/down based on demand
- Managing worker lifecycle (creation, monitoring, termination)
- Cost optimization through spot/preemptible instances
- Multi-cloud integration (AWS, GCP, Docker)

Usage:
    # Import and initialize
    from duckdb_api.distributed_testing.cloud_provider_manager import CloudProviderManager
    
    # Create manager with config file
    manager = CloudProviderManager(config_path="/path/to/cloud_config.json")
    
    # Alternative: Create manager with config dict
    manager = CloudProviderManager(config={
        "aws": {
            "region": "us-west-2",
            "instance_types": {
                "cpu": "c5.xlarge",
                "gpu": "g4dn.xlarge",
                "default": "t3.medium"
            },
            "credentials": {
                "access_key_id": "${AWS_ACCESS_KEY_ID}",
                "secret_access_key": "${AWS_SECRET_ACCESS_KEY}"
            },
            "spot_instance_enabled": true
        },
        "docker_local": {
            "image": "ipfs-accelerate-worker:latest",
            "cpu_limit": 4,
            "memory_limit": "16g"
        }
    })
    
    # Add provider
    manager.add_provider("aws", aws_provider_instance)
    
    # Create worker with specific resource requirements
    worker = manager.create_worker(
        provider="aws",
        resources={"cpu_cores": 4, "memory_gb": 16},
        worker_type="cpu"
    )
    
    # Get worker status
    status = manager.get_worker_status("aws", worker["worker_id"])
    
    # Terminate worker when done
    manager.terminate_worker("aws", worker["worker_id"])
    
    # Get provider resources
    resources = manager.get_available_resources("aws")
    
    # Get preferred provider for workload
    provider = manager.get_preferred_provider(
        requirements={"gpu": True, "min_memory_gb": 32}
    )
"""

import os
import sys
import json
import time
import uuid
import logging
import threading
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set
from pathlib import Path
import importlib

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("cloud_provider_manager")

# Add parent directory to path to import modules from parent
parent_dir = str(Path(__file__).parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Try to import the cloud provider integration module
try:
    from cloud_provider_integration import (
        AWSCloudProvider, 
        GCPCloudProvider, 
        DockerLocalProvider
    )
    CLOUD_INTEGRATION_AVAILABLE = True
except ImportError:
    logger.warning("Cloud provider integration module not available. Using minimal implementation.")
    CLOUD_INTEGRATION_AVAILABLE = False

# Cloud provider constants
DEFAULT_WORKER_PARAMS = {
    "cpu": {
        "cpu_cores": 4,
        "memory_gb": 16,
        "gpu_count": 0
    },
    "gpu": {
        "cpu_cores": 8,
        "memory_gb": 32,
        "gpu_count": 1,
        "gpu_type": "nvidia-t4"
    },
    "memory": {
        "cpu_cores": 8,
        "memory_gb": 64,
        "gpu_count": 0
    },
    "default": {
        "cpu_cores": 2,
        "memory_gb": 8,
        "gpu_count": 0
    }
}


class CloudProviderManager:
    """
    Cloud Provider Manager for the distributed testing framework.
    
    Handles the provisioning, management, and termination of worker nodes across
    various cloud providers.
    """
    
    def __init__(self, config_path: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Cloud Provider Manager.
        
        Args:
            config_path: Path to cloud provider configuration file
            config: Direct configuration dictionary (alternative to config_path)
        """
        self.config_path = config_path
        self.config = config
        self.providers = {}  # provider_name -> provider instance
        
        # If config_path is provided, load config from file
        if config_path and not config:
            self.load_config(config_path)
        
        logger.info("Cloud Provider Manager initialized")
    
    def load_config(self, config_path: str) -> None:
        """
        Load configuration from file.
        
        Args:
            config_path: Path to configuration file
        """
        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
            
            self.config_path = config_path
            logger.info(f"Loaded configuration from {config_path}")
            
            # Initialize providers if cloud integration is available
            if CLOUD_INTEGRATION_AVAILABLE:
                self._initialize_providers()
        
        except Exception as e:
            logger.error(f"Error loading configuration from {config_path}: {e}")
            logger.debug(traceback.format_exc())
            self.config = {}
    
    def _initialize_providers(self) -> None:
        """Initialize cloud providers from configuration."""
        if not self.config:
            logger.warning("No configuration available to initialize providers")
            return
        
        try:
            # Initialize AWS provider if configured
            if "aws" in self.config and self.config["aws"].get("enabled", True):
                aws_config = self.config["aws"]
                region = aws_config.get("region", "us-east-1")
                
                try:
                    aws_provider = AWSCloudProvider(
                        region=region,
                        profile_name=aws_config.get("profile_name"),
                        access_key_id=aws_config.get("credentials", {}).get("access_key_id"),
                        secret_access_key=aws_config.get("credentials", {}).get("secret_access_key")
                    )
                    self.add_provider("aws", aws_provider)
                    logger.info(f"Initialized AWS provider for region {region}")
                except Exception as e:
                    logger.error(f"Failed to initialize AWS provider: {e}")
            
            # Initialize GCP provider if configured
            if "gcp" in self.config and self.config["gcp"].get("enabled", True):
                gcp_config = self.config["gcp"]
                project_id = gcp_config.get("project")
                zone = gcp_config.get("zone", "us-central1-a")
                
                if project_id:
                    try:
                        gcp_provider = GCPCloudProvider(
                            region=zone,
                            project_id=project_id,
                            credentials_file=gcp_config.get("credentials_file")
                        )
                        self.add_provider("gcp", gcp_provider)
                        logger.info(f"Initialized GCP provider for project {project_id}, zone {zone}")
                    except Exception as e:
                        logger.error(f"Failed to initialize GCP provider: {e}")
                else:
                    logger.warning("GCP provider requires project_id to be configured")
            
            # Initialize Docker local provider if configured
            if "docker_local" in self.config and self.config["docker_local"].get("enabled", True):
                docker_config = self.config["docker_local"]
                
                try:
                    docker_provider = DockerLocalProvider(
                        region="local",
                        docker_host=docker_config.get("docker_host"),
                        network_name=docker_config.get("network", "bridge")
                    )
                    self.add_provider("docker_local", docker_provider)
                    logger.info("Initialized Docker local provider")
                except Exception as e:
                    logger.error(f"Failed to initialize Docker provider: {e}")
        
        except Exception as e:
            logger.error(f"Error initializing providers: {e}")
            logger.debug(traceback.format_exc())
    
    def add_provider(self, provider_name: str, provider_instance: Any) -> None:
        """
        Add a cloud provider to the manager.
        
        Args:
            provider_name: Name of the provider
            provider_instance: Provider instance
        """
        self.providers[provider_name] = provider_instance
        logger.info(f"Added provider: {provider_name}")
    
    def create_worker(self, provider: str, resources: Optional[Dict[str, Any]] = None, 
                     worker_type: Optional[str] = None, coordinator_url: Optional[str] = None,
                     api_key: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Create a worker node on the specified provider.
        
        Args:
            provider: Provider name
            resources: Resource requirements
            worker_type: Type of worker (cpu, gpu, memory)
            coordinator_url: URL for the worker to connect to
            api_key: API key for worker authentication
        
        Returns:
            dict: Worker information or None on failure
        """
        try:
            if provider not in self.providers:
                logger.error(f"Provider not found: {provider}")
                return None
            
            provider_instance = self.providers[provider]
            
            # Prepare worker configuration
            worker_config = {}
            
            # Get provider-specific configuration
            if self.config and provider in self.config:
                provider_config = self.config[provider]
                
                # Include coordinator URL and API key in environment variables
                environment = provider_config.get("environment", {}).copy()
                if coordinator_url:
                    environment["COORDINATOR_URL"] = coordinator_url
                if api_key:
                    environment["WORKER_API_KEY"] = api_key
                
                # Create worker configuration
                worker_config = {
                    "worker_name": f"worker-{provider}-{uuid.uuid4().hex[:8]}",
                    "environment": environment,
                    "coordinator_url": coordinator_url,
                    "api_key": api_key
                }
                
                # Add resource requirements
                if worker_type and worker_type in DEFAULT_WORKER_PARAMS:
                    worker_params = DEFAULT_WORKER_PARAMS[worker_type]
                    worker_config["cpu_cores"] = worker_params["cpu_cores"]
                    worker_config["memory_gb"] = worker_params["memory_gb"]
                    worker_config["gpu_count"] = worker_params["gpu_count"]
                    if "gpu_type" in worker_params:
                        worker_config["gpu_type"] = worker_params["gpu_type"]
                
                # Override with provided resources
                if resources:
                    if "cpu_cores" in resources:
                        worker_config["cpu_cores"] = resources["cpu_cores"]
                    if "memory_mb" in resources:
                        worker_config["memory_gb"] = resources["memory_mb"] / 1024  # Convert MB to GB
                    if "memory_gb" in resources:
                        worker_config["memory_gb"] = resources["memory_gb"]
                    if "gpu_count" in resources:
                        worker_config["gpu_count"] = resources["gpu_count"]
                    if "gpu_type" in resources:
                        worker_config["gpu_type"] = resources["gpu_type"]
                    if "gpu_memory_mb" in resources and resources["gpu_memory_mb"] > 0:
                        worker_config["gpu_count"] = max(1, worker_config.get("gpu_count", 1))
            
            # Create worker
            result = provider_instance.create_worker(worker_config)
            
            # Add metadata
            if result:
                result["provider"] = provider
                result["creation_time"] = datetime.now().isoformat()
            
            logger.info(f"Created worker on {provider}: {result.get('worker_id') if result else 'failed'}")
            return result
        
        except Exception as e:
            logger.error(f"Error creating worker on {provider}: {e}")
            logger.debug(traceback.format_exc())
            return None
    
    def terminate_worker(self, provider: str, worker_id: str) -> bool:
        """
        Terminate a worker node.
        
        Args:
            provider: Provider name
            worker_id: Worker ID to terminate
        
        Returns:
            bool: Success status
        """
        try:
            if provider not in self.providers:
                logger.error(f"Provider not found: {provider}")
                return False
            
            provider_instance = self.providers[provider]
            result = provider_instance.terminate_worker(worker_id)
            
            logger.info(f"Terminated worker {worker_id} on {provider}: {'success' if result else 'failed'}")
            return result
        
        except Exception as e:
            logger.error(f"Error terminating worker {worker_id} on {provider}: {e}")
            logger.debug(traceback.format_exc())
            return False
    
    def get_worker_status(self, provider: str, worker_id: str) -> Optional[Dict[str, Any]]:
        """
        Get worker node status.
        
        Args:
            provider: Provider name
            worker_id: Worker ID to check
        
        Returns:
            dict: Worker status or None on failure
        """
        try:
            if provider not in self.providers:
                logger.error(f"Provider not found: {provider}")
                return None
            
            provider_instance = self.providers[provider]
            status = provider_instance.get_worker_status(worker_id)
            
            if status:
                status["provider"] = provider
            
            return status
        
        except Exception as e:
            logger.error(f"Error getting status for worker {worker_id} on {provider}: {e}")
            logger.debug(traceback.format_exc())
            return None
    
    def get_available_resources(self, provider: str) -> Optional[Dict[str, Any]]:
        """
        Get available resources on a provider.
        
        Args:
            provider: Provider name
        
        Returns:
            dict: Available resource information or None on failure
        """
        try:
            if provider not in self.providers:
                logger.error(f"Provider not found: {provider}")
                return None
            
            provider_instance = self.providers[provider]
            resources = provider_instance.get_available_resources()
            
            if resources:
                resources["provider"] = provider
            
            return resources
        
        except Exception as e:
            logger.error(f"Error getting available resources on {provider}: {e}")
            logger.debug(traceback.format_exc())
            return None
    
    def get_preferred_provider(self, requirements: Dict[str, Any]) -> Optional[str]:
        """
        Get the preferred provider based on requirements.
        
        Args:
            requirements: Dictionary of requirements
                gpu: Whether GPU is required
                min_cpu_cores: Minimum CPU cores required
                min_memory_gb: Minimum memory (GB) required
                local: Whether local execution is preferred
        
        Returns:
            str: Provider name or None if no suitable provider
        """
        try:
            if not self.providers:
                logger.warning("No providers available")
                return None
            
            # If local execution is preferred, use Docker if available
            if requirements.get("local", False) and "docker_local" in self.providers:
                return "docker_local"
            
            # If GPU is required, prioritize providers with GPU support
            if requirements.get("gpu", False):
                # Check AWS first for GPU support (typically better GPU options)
                if "aws" in self.providers:
                    aws_resources = self.get_available_resources("aws")
                    if aws_resources:
                        return "aws"
                
                # Then check GCP
                if "gcp" in self.providers:
                    gcp_resources = self.get_available_resources("gcp")
                    if gcp_resources:
                        return "gcp"
            
            # For high memory requirements, check GCP first
            min_memory_gb = requirements.get("min_memory_gb", 0)
            if min_memory_gb > 64:
                if "gcp" in self.providers:
                    return "gcp"
                if "aws" in self.providers:
                    return "aws"
            
            # For high CPU requirements, check AWS first
            min_cpu_cores = requirements.get("min_cpu_cores", 0)
            if min_cpu_cores > 32:
                if "aws" in self.providers:
                    return "aws"
                if "gcp" in self.providers:
                    return "gcp"
            
            # If no specific requirement, return the first available provider
            return next(iter(self.providers))
        
        except Exception as e:
            logger.error(f"Error determining preferred provider: {e}")
            logger.debug(traceback.format_exc())
            return next(iter(self.providers)) if self.providers else None
    
    def get_all_providers(self) -> Dict[str, Any]:
        """
        Get all registered providers.
        
        Returns:
            dict: Dictionary of provider name to provider instance
        """
        return self.providers


# Main function for testing
if __name__ == "__main__":
    """Run standalone test of the Cloud Provider Manager."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Cloud Provider Manager")
    parser.add_argument("--config", help="Path to cloud provider configuration file")
    parser.add_argument("--provider", default="docker_local", help="Provider to test")
    parser.add_argument("--action", choices=["create", "terminate", "status", "resources"],
                       default="resources", help="Action to perform")
    parser.add_argument("--worker-id", help="Worker ID for terminate/status actions")
    
    args = parser.parse_args()
    
    # Create manager
    manager = CloudProviderManager(config_path=args.config)
    
    # If no config file provided, create a minimal configuration for testing
    if not args.config:
        test_config = {
            "docker_local": {
                "enabled": True,
                "image": "ipfs-accelerate-worker:latest",
                "cpu_limit": 4,
                "memory_limit": "16g",
                "network": "host"
            }
        }
        manager.config = test_config
    
    # Initialize providers
    if CLOUD_INTEGRATION_AVAILABLE:
        manager._initialize_providers()
    else:
        # For testing, create a simple mock provider
        class MockProvider:
            def create_worker(self, config=None):
                worker_id = f"mock-worker-{uuid.uuid4().hex[:8]}"
                return {"worker_id": worker_id, "status": "running", "endpoint": f"http://localhost:8080"}
            
            def terminate_worker(self, worker_id):
                return True
            
            def get_worker_status(self, worker_id):
                return {"worker_id": worker_id, "status": "running"}
            
            def get_available_resources(self):
                return {"cpu_cores": 8, "memory_gb": 32, "max_workers": 5}
        
        manager.add_provider(args.provider, MockProvider())
    
    # Perform the requested action
    if args.action == "create":
        result = manager.create_worker(
            provider=args.provider,
            resources={"cpu_cores": 4, "memory_gb": 8},
            coordinator_url="http://localhost:8080",
            api_key="test-key"
        )
        print(f"Created worker: {json.dumps(result, indent=2)}")
    
    elif args.action == "terminate":
        if not args.worker_id:
            print("Error: --worker-id is required for terminate action")
            sys.exit(1)
        
        result = manager.terminate_worker(args.provider, args.worker_id)
        print(f"Terminated worker {args.worker_id}: {result}")
    
    elif args.action == "status":
        if not args.worker_id:
            print("Error: --worker-id is required for status action")
            sys.exit(1)
        
        status = manager.get_worker_status(args.provider, args.worker_id)
        print(f"Worker status: {json.dumps(status, indent=2)}")
    
    elif args.action == "resources":
        resources = manager.get_available_resources(args.provider)
        print(f"Available resources: {json.dumps(resources, indent=2)}")