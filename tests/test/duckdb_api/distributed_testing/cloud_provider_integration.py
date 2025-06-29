#!/usr/bin/env python3
"""
Distributed Testing Framework - Cloud Provider Integration

This module handles integration with cloud providers for dynamic worker management.
It enables the creation, monitoring, and termination of ephemeral worker nodes on
cloud platforms like AWS, GCP, and Azure.

Core responsibilities:
- Worker node provisioning on cloud platforms
- Resource configuration (CPU, memory, GPU)
- Cost optimization with spot/preemptible instances
- Worker monitoring and lifecycle management
- Cleanup of terminated resources

Usage:
    # Import and initialize
    from duckdb_api.distributed_testing.cloud_provider_integration import CloudProviderManager
    
    # Create cloud manager for AWS
    cloud_mgr = CloudProviderManager(provider="aws", region="us-west-2")
    
    # Create worker node
    worker_id = cloud_mgr.create_worker({
        "cpu_cores": 4,
        "memory_gb": 16,
        "gpu_type": "nvidia-t4",
        "gpu_count": 1,
        "coordinator_url": "http://coordinator-host:8080",
        "api_key": "worker-api-key"
    })
    
    # Check worker status
    status = cloud_mgr.get_worker_status(worker_id)
    
    # Terminate worker
    cloud_mgr.terminate_worker(worker_id)
"""

import os
import sys
import json
import time
import uuid
import logging
import threading
import traceback
import subprocess
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set
from pathlib import Path
from abc import ABC, abstractmethod

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("cloud_provider_integration")

# Add parent directory to path to import modules from parent
parent_dir = str(Path(__file__).parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Try to import optional cloud provider SDKs
# AWS
try:
    import boto3
    AWS_AVAILABLE = True
except ImportError:
    logger.warning("boto3 not available. AWS integration disabled.")
    AWS_AVAILABLE = False

# GCP
try:
    from google.cloud import compute_v1
    GCP_AVAILABLE = True
except ImportError:
    logger.warning("google.cloud.compute_v1 not available. GCP integration disabled.")
    GCP_AVAILABLE = False

# Azure
try:
    from azure.identity import DefaultAzureCredential
    from azure.mgmt.compute import ComputeManagementClient
    AZURE_AVAILABLE = True
except ImportError:
    logger.warning("azure.mgmt.compute not available. Azure integration disabled.")
    AZURE_AVAILABLE = False

# Docker
try:
    import docker
    DOCKER_AVAILABLE = True
except ImportError:
    logger.warning("docker not available. Local Docker integration disabled.")
    DOCKER_AVAILABLE = False


class CloudProviderBase(ABC):
    """Base class for cloud provider implementations."""
    
    def __init__(self, region: str, **kwargs):
        """
        Initialize the cloud provider.
        
        Args:
            region: Cloud provider region
            **kwargs: Provider-specific arguments
        """
        self.region = region
        self.workers = {}  # worker_id -> worker details
    
    @abstractmethod
    def create_worker(self, config: Dict[str, Any]) -> str:
        """
        Create a worker node in the cloud.
        
        Args:
            config: Worker configuration
        
        Returns:
            str: Worker ID
        """
        pass
    
    @abstractmethod
    def terminate_worker(self, worker_id: str) -> bool:
        """
        Terminate a worker node.
        
        Args:
            worker_id: Worker ID to terminate
        
        Returns:
            bool: Success status
        """
        pass
    
    @abstractmethod
    def get_worker_status(self, worker_id: str) -> Dict[str, Any]:
        """
        Get worker node status.
        
        Args:
            worker_id: Worker ID to check
        
        Returns:
            dict: Worker status information
        """
        pass
    
    @abstractmethod
    def list_workers(self) -> Dict[str, Dict[str, Any]]:
        """
        List all managed worker nodes.
        
        Returns:
            dict: Dictionary of worker ID to worker details
        """
        pass
    
    @abstractmethod
    def get_available_resources(self) -> Dict[str, Any]:
        """
        Get available cloud resources.
        
        Returns:
            dict: Available resource information
        """
        pass
    
    def get_base_worker_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get base worker configuration with defaults.
        
        Args:
            config: User-provided configuration
        
        Returns:
            dict: Complete configuration with defaults
        """
        # Set default values
        worker_config = {
            "cpu_cores": 2,
            "memory_gb": 8,
            "disk_gb": 50,
            "gpu_type": None,
            "gpu_count": 0,
            "use_spot": True,
            "max_price": None,
            "worker_name": f"worker-{uuid.uuid4().hex[:8]}",
            "tags": {"managed_by": "distributed_testing_framework"},
            "coordinator_url": None,
            "api_key": None,
            "worker_script": None,
            "worker_zip": None,
            "worker_docker_image": "ipfs_accelerate_worker:latest",
            "worker_type": "docker" if DOCKER_AVAILABLE else "vm",
            "environment": {},
        }
        
        # Update with user-provided values
        worker_config.update(config)
        
        # Ensure required fields
        if not worker_config["coordinator_url"]:
            logger.error("Coordinator URL is required")
            raise ValueError("Coordinator URL is required")
        
        if not worker_config["api_key"]:
            logger.error("API key is required")
            raise ValueError("API key is required")
        
        # Add basic environment variables
        worker_config["environment"].update({
            "COORDINATOR_URL": worker_config["coordinator_url"],
            "WORKER_API_KEY": worker_config["api_key"],
            "WORKER_ID": worker_config["worker_name"],
        })
        
        return worker_config


class AWSCloudProvider(CloudProviderBase):
    """AWS cloud provider implementation."""
    
    def __init__(self, region: str, **kwargs):
        """
        Initialize AWS cloud provider.
        
        Args:
            region: AWS region
            **kwargs: Additional AWS-specific arguments
                profile_name: AWS profile name
                access_key_id: AWS access key ID
                secret_access_key: AWS secret access key
        """
        super().__init__(region, **kwargs)
        
        if not AWS_AVAILABLE:
            logger.error("boto3 not available. AWS integration disabled.")
            raise ImportError("boto3 not available. AWS integration disabled.")
        
        # Initialize AWS clients
        profile_name = kwargs.get("profile_name")
        access_key_id = kwargs.get("access_key_id")
        secret_access_key = kwargs.get("secret_access_key")
        
        session_kwargs = {}
        if profile_name:
            session_kwargs["profile_name"] = profile_name
        if access_key_id and secret_access_key:
            session_kwargs["aws_access_key_id"] = access_key_id
            session_kwargs["aws_secret_access_key"] = secret_access_key
        
        self.session = boto3.Session(region_name=region, **session_kwargs)
        self.ec2 = self.session.client("ec2")
        
        # Cache available resource info
        self._instance_types = None
        
        logger.info(f"AWS cloud provider initialized for region {region}")
    
    def create_worker(self, config: Dict[str, Any]) -> str:
        """
        Create a worker node in AWS EC2.
        
        Args:
            config: Worker configuration
                cpu_cores: Number of CPU cores
                memory_gb: Memory in GB
                disk_gb: Disk size in GB
                gpu_type: GPU type (e.g., "nvidia-t4")
                gpu_count: Number of GPUs
                use_spot: Whether to use spot instances
                max_price: Maximum spot price
                worker_name: Worker name
                tags: Instance tags
                coordinator_url: Coordinator URL
                api_key: API key for worker authentication
                worker_script: Custom worker script
                worker_zip: ZIP archive containing worker code
                subnet_id: Subnet ID
                security_group_ids: Security group IDs
                ami_id: Amazon Machine Image ID
        
        Returns:
            str: Worker ID (EC2 instance ID)
        """
        worker_config = self.get_base_worker_config(config)
        
        try:
            # Select appropriate instance type based on requirements
            instance_type = self._select_instance_type(
                cpu_cores=worker_config["cpu_cores"],
                memory_gb=worker_config["memory_gb"],
                gpu_type=worker_config["gpu_type"],
                gpu_count=worker_config["gpu_count"]
            )
            
            # Prepare user data for worker setup
            user_data = self._generate_user_data(worker_config)
            
            # Prepare tags
            tags = [
                {"Key": "Name", "Value": worker_config["worker_name"]},
                {"Key": "ManagedBy", "Value": "DistributedTestingFramework"},
            ]
            for key, value in worker_config.get("tags", {}).items():
                tags.append({"Key": key, "Value": str(value)})
            
            # Prepare network interfaces
            network_interfaces = []
            if worker_config.get("subnet_id") and worker_config.get("security_group_ids"):
                network_interfaces.append({
                    "AssociatePublicIpAddress": True,
                    "DeviceIndex": 0,
                    "SubnetId": worker_config["subnet_id"],
                    "Groups": worker_config["security_group_ids"]
                })
            
            # Common launch parameters
            launch_params = {
                "ImageId": worker_config.get("ami_id", self._get_default_ami()),
                "InstanceType": instance_type,
                "MinCount": 1,
                "MaxCount": 1,
                "UserData": user_data,
                "TagSpecifications": [
                    {
                        "ResourceType": "instance",
                        "Tags": tags
                    }
                ],
                "BlockDeviceMappings": [
                    {
                        "DeviceName": "/dev/sda1",
                        "Ebs": {
                            "VolumeSize": worker_config["disk_gb"],
                            "VolumeType": "gp3",
                            "DeleteOnTermination": True
                        }
                    }
                ]
            }
            
            # Add network interfaces if specified
            if network_interfaces:
                launch_params["NetworkInterfaces"] = network_interfaces
            
            # Launch as spot or on-demand instance
            if worker_config["use_spot"]:
                spot_params = {
                    "InstanceMarketOptions": {
                        "MarketType": "spot",
                        "SpotOptions": {
                            "SpotInstanceType": "one-time",
                            "InstanceInterruptionBehavior": "terminate"
                        }
                    }
                }
                
                if worker_config["max_price"]:
                    spot_params["InstanceMarketOptions"]["SpotOptions"]["MaxPrice"] = str(worker_config["max_price"])
                
                launch_params.update(spot_params)
                
                logger.info(f"Launching spot instance {worker_config['worker_name']} with type {instance_type}")
            else:
                logger.info(f"Launching on-demand instance {worker_config['worker_name']} with type {instance_type}")
            
            # Launch instance
            response = self.ec2.run_instances(**launch_params)
            
            # Get instance ID
            instance_id = response["Instances"][0]["InstanceId"]
            
            # Store worker info
            self.workers[instance_id] = {
                "instance_id": instance_id,
                "worker_name": worker_config["worker_name"],
                "instance_type": instance_type,
                "launch_time": datetime.now().isoformat(),
                "state": "pending",
                "config": worker_config,
            }
            
            logger.info(f"Launched worker {worker_config['worker_name']} with ID {instance_id}")
            
            return instance_id
        
        except Exception as e:
            logger.error(f"Error creating AWS worker: {e}")
            logger.debug(traceback.format_exc())
            raise
    
    def terminate_worker(self, worker_id: str) -> bool:
        """
        Terminate an EC2 instance.
        
        Args:
            worker_id: EC2 instance ID
        
        Returns:
            bool: Success status
        """
        try:
            response = self.ec2.terminate_instances(InstanceIds=[worker_id])
            
            # Update worker state
            if worker_id in self.workers:
                self.workers[worker_id]["state"] = "terminating"
            
            logger.info(f"Terminated worker with ID {worker_id}")
            return True
        
        except Exception as e:
            logger.error(f"Error terminating worker {worker_id}: {e}")
            logger.debug(traceback.format_exc())
            return False
    
    def get_worker_status(self, worker_id: str) -> Dict[str, Any]:
        """
        Get EC2 instance status.
        
        Args:
            worker_id: EC2 instance ID
        
        Returns:
            dict: Worker status information
        """
        try:
            response = self.ec2.describe_instances(InstanceIds=[worker_id])
            
            if not response["Reservations"] or not response["Reservations"][0]["Instances"]:
                logger.warning(f"Worker {worker_id} not found")
                return {"state": "not_found"}
            
            instance = response["Reservations"][0]["Instances"][0]
            
            status = {
                "instance_id": worker_id,
                "state": instance["State"]["Name"],
                "launch_time": instance["LaunchTime"].isoformat(),
                "instance_type": instance["InstanceType"],
                "public_ip": instance.get("PublicIpAddress"),
                "private_ip": instance.get("PrivateIpAddress"),
            }
            
            # Update cached worker info
            if worker_id in self.workers:
                self.workers[worker_id]["state"] = status["state"]
                self.workers[worker_id]["public_ip"] = status["public_ip"]
                self.workers[worker_id]["private_ip"] = status["private_ip"]
            
            return status
        
        except Exception as e:
            logger.error(f"Error getting worker status for {worker_id}: {e}")
            logger.debug(traceback.format_exc())
            return {"state": "error", "error": str(e)}
    
    def list_workers(self) -> Dict[str, Dict[str, Any]]:
        """
        List all managed worker nodes.
        
        Returns:
            dict: Dictionary of worker ID to worker details
        """
        # Refresh worker states
        try:
            # Get all instance IDs of our workers
            instance_ids = list(self.workers.keys())
            
            if not instance_ids:
                return {}
            
            # Batch describe instances (100 at a time to respect AWS limits)
            batch_size = 100
            for i in range(0, len(instance_ids), batch_size):
                batch_ids = instance_ids[i:i+batch_size]
                response = self.ec2.describe_instances(InstanceIds=batch_ids)
                
                # Process each reservation and instance
                for reservation in response["Reservations"]:
                    for instance in reservation["Instances"]:
                        instance_id = instance["InstanceId"]
                        
                        if instance_id in self.workers:
                            self.workers[instance_id]["state"] = instance["State"]["Name"]
                            self.workers[instance_id]["public_ip"] = instance.get("PublicIpAddress")
                            self.workers[instance_id]["private_ip"] = instance.get("PrivateIpAddress")
        
        except Exception as e:
            logger.error(f"Error listing workers: {e}")
            logger.debug(traceback.format_exc())
        
        return self.workers
    
    def get_available_resources(self) -> Dict[str, Any]:
        """
        Get available EC2 instance types and their capabilities.
        
        Returns:
            dict: Available instance types and their capabilities
        """
        if self._instance_types:
            return self._instance_types
        
        try:
            response = self.ec2.describe_instance_types()
            
            instance_types = {}
            for instance in response["InstanceTypes"]:
                instance_type = instance["InstanceType"]
                instance_types[instance_type] = {
                    "vcpus": instance["VCpuInfo"]["DefaultVCpus"],
                    "memory_gb": instance["MemoryInfo"]["SizeInMiB"] / 1024,
                    "gpu_info": None,
                    "network_performance": instance.get("NetworkInfo", {}).get("NetworkPerformance"),
                }
                
                # Get accelerator (GPU) info if available
                if "GpuInfo" in instance:
                    gpu_info = instance["GpuInfo"]
                    instance_types[instance_type]["gpu_info"] = {
                        "gpus": gpu_info["Gpus"][0]["Count"],
                        "gpu_type": gpu_info["Gpus"][0]["Name"],
                        "gpu_memory_gb": gpu_info["Gpus"][0]["MemoryInfo"]["SizeInMiB"] / 1024,
                    }
            
            self._instance_types = instance_types
            return instance_types
        
        except Exception as e:
            logger.error(f"Error getting available resources: {e}")
            logger.debug(traceback.format_exc())
            return {}
    
    def _select_instance_type(self, cpu_cores: int, memory_gb: int, 
                             gpu_type: Optional[str] = None, gpu_count: int = 0) -> str:
        """
        Select an appropriate EC2 instance type based on requirements.
        
        Args:
            cpu_cores: Number of CPU cores required
            memory_gb: Memory in GB required
            gpu_type: GPU type required (e.g., "nvidia-t4")
            gpu_count: Number of GPUs required
        
        Returns:
            str: Selected EC2 instance type
        """
        # Get available instance types
        instance_types = self.get_available_resources()
        
        # Define scoring function for instance types
        def score_instance_type(instance_type_info):
            # Higher score = better match
            score = 0
            
            # CPU score - prefer slightly more than required
            cpu_ratio = instance_type_info["vcpus"] / cpu_cores
            if cpu_ratio < 1:
                # Not enough CPUs
                return -1
            elif cpu_ratio <= 1.5:
                # Close match
                score += 100
            elif cpu_ratio <= 2:
                # Slight overprovisioning
                score += 80
            else:
                # Too much overprovisioning
                score += 50 / cpu_ratio
            
            # Memory score - prefer slightly more than required
            memory_ratio = instance_type_info["memory_gb"] / memory_gb
            if memory_ratio < 1:
                # Not enough memory
                return -1
            elif memory_ratio <= 1.25:
                # Close match
                score += 100
            elif memory_ratio <= 2:
                # Slight overprovisioning
                score += 80
            else:
                # Too much overprovisioning
                score += 50 / memory_ratio
            
            # GPU score
            if gpu_count > 0:
                if not instance_type_info["gpu_info"]:
                    # No GPUs available
                    return -1
                
                gpu_info = instance_type_info["gpu_info"]
                
                if gpu_info["gpus"] < gpu_count:
                    # Not enough GPUs
                    return -1
                
                if gpu_type and gpu_type.lower() not in gpu_info["gpu_type"].lower():
                    # Wrong GPU type
                    return -1
                
                # Good GPU match
                score += 100
            elif instance_type_info["gpu_info"]:
                # GPUs available but not needed
                score -= 50
            
            return score
        
        # Score and filter instance types
        scored_instances = []
        for instance_type, info in instance_types.items():
            score = score_instance_type(info)
            if score > 0:
                scored_instances.append((instance_type, score))
        
        if not scored_instances:
            # No instances match criteria, use default fallback based on GPU requirements
            if gpu_count > 0:
                return "p3.2xlarge"  # Default GPU instance
            else:
                return "c5.2xlarge"  # Default CPU instance
        
        # Sort by score (descending)
        scored_instances.sort(key=lambda x: x[1], reverse=True)
        
        # Return best match
        return scored_instances[0][0]
    
    def _generate_user_data(self, config: Dict[str, Any]) -> str:
        """
        Generate user data script for EC2 instance.
        
        Args:
            config: Worker configuration
        
        Returns:
            str: Base64-encoded user data script
        """
        script = """#!/bin/bash
set -e

echo "Starting worker setup..."

# Install dependencies
apt-get update
apt-get install -y python3 python3-pip git unzip

# Set up environment variables
"""
        
        # Add environment variables
        for key, value in config["environment"].items():
            script += f'export {key}="{value}"\n'
        
        # Add coordinator and API key environment variables
        script += f'export COORDINATOR_URL="{config["coordinator_url"]}"\n'
        script += f'export WORKER_API_KEY="{config["api_key"]}"\n'
        script += f'export WORKER_ID="{config["worker_name"]}"\n'
        
        # Clone repository or download worker code
        script += """
# Clone the repository (if not using worker_zip)
git clone https://github.com/example/ipfs_accelerate_py.git
cd ipfs_accelerate_py

# Install requirements
pip3 install -r requirements.txt

# Start the worker
python3 -m duckdb_api.distributed_testing.worker --coordinator $COORDINATOR_URL --api-key $WORKER_API_KEY
"""
        
        # Return the script
        import base64
        return base64.b64encode(script.encode()).decode()
    
    def _get_default_ami(self) -> str:
        """
        Get the latest Ubuntu 22.04 AMI ID for the current region.
        
        Returns:
            str: AMI ID
        """
        try:
            response = self.ec2.describe_images(
                Owners=["099720109477"],  # Canonical
                Filters=[
                    {"Name": "name", "Values": ["ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*"]},
                    {"Name": "state", "Values": ["available"]}
                ]
            )
            
            # Sort by creation date (newest first)
            images = sorted(response["Images"], key=lambda x: x["CreationDate"], reverse=True)
            
            if images:
                return images[0]["ImageId"]
            
            # Fallback to region-specific AMIs
            ami_map = {
                "us-east-1": "ami-0261755bbcb8c4a84",
                "us-east-2": "ami-0430580de6244e02e",
                "us-west-1": "ami-04d1dcfb793f6fa37",
                "us-west-2": "ami-0c65adc9a5c1b5d7c",
                "eu-west-1": "ami-0905a3c97561e0b69",
                "eu-central-1": "ami-0faab6bdbac9486fb",
            }
            
            return ami_map.get(self.region, "ami-0261755bbcb8c4a84")  # Default to us-east-1
        
        except Exception as e:
            logger.error(f"Error getting default AMI: {e}")
            logger.debug(traceback.format_exc())
            
            # Return a default AMI for Ubuntu 22.04 in the specified region
            ami_map = {
                "us-east-1": "ami-0261755bbcb8c4a84",
                "us-east-2": "ami-0430580de6244e02e",
                "us-west-1": "ami-04d1dcfb793f6fa37",
                "us-west-2": "ami-0c65adc9a5c1b5d7c",
                "eu-west-1": "ami-0905a3c97561e0b69",
                "eu-central-1": "ami-0faab6bdbac9486fb",
            }
            
            return ami_map.get(self.region, "ami-0261755bbcb8c4a84")  # Default to us-east-1


class GCPCloudProvider(CloudProviderBase):
    """GCP cloud provider implementation."""
    
    def __init__(self, region: str, **kwargs):
        """
        Initialize GCP cloud provider.
        
        Args:
            region: GCP region
            **kwargs: Additional GCP-specific arguments
                project_id: GCP project ID
                credentials_file: Path to service account credentials file
        """
        super().__init__(region, **kwargs)
        
        if not GCP_AVAILABLE:
            logger.error("google.cloud.compute_v1 not available. GCP integration disabled.")
            raise ImportError("google.cloud.compute_v1 not available. GCP integration disabled.")
        
        # Initialize GCP clients
        self.project_id = kwargs.get("project_id")
        if not self.project_id:
            raise ValueError("project_id is required for GCP integration")
        
        credentials_file = kwargs.get("credentials_file")
        if credentials_file:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_file
        
        # Parsing region and zone
        # GCP regions are like "us-central1" and zones are like "us-central1-a"
        if "-" in region and region.count("-") == 1:
            # Region format
            self.region = region
            self.zone = f"{region}-a"  # Default to first zone
        elif "-" in region and region.count("-") == 2:
            # Zone format
            self.zone = region
            self.region = region.rsplit("-", 1)[0]
        else:
            raise ValueError(f"Invalid GCP region/zone format: {region}")
        
        # Initialize clients
        self.instance_client = compute_v1.InstancesClient()
        self.image_client = compute_v1.ImagesClient()
        self.machine_types_client = compute_v1.MachineTypesClient()
        self.accelerator_types_client = compute_v1.AcceleratorTypesClient()
        
        # Cache available resource info
        self._machine_types = None
        self._accelerator_types = None
        
        logger.info(f"GCP cloud provider initialized for project {self.project_id}, region {self.region}, zone {self.zone}")
    
    def create_worker(self, config: Dict[str, Any]) -> str:
        """
        Create a worker node in GCP Compute Engine.
        
        Args:
            config: Worker configuration
                cpu_cores: Number of CPU cores
                memory_gb: Memory in GB
                disk_gb: Disk size in GB
                gpu_type: GPU type (e.g., "nvidia-tesla-t4")
                gpu_count: Number of GPUs
                use_spot: Whether to use spot instances
                max_price: Maximum spot price (not used in GCP, use "use_spot" instead)
                worker_name: Worker name
                tags: Instance tags
                coordinator_url: Coordinator URL
                api_key: API key for worker authentication
                worker_script: Custom worker script
                worker_zip: ZIP archive containing worker code
                network: Network name
                subnetwork: Subnetwork name
                service_account: Service account email
        
        Returns:
            str: Worker ID (GCP instance name)
        """
        worker_config = self.get_base_worker_config(config)
        
        try:
            # Select appropriate machine type based on requirements
            machine_type = self._select_machine_type(
                cpu_cores=worker_config["cpu_cores"],
                memory_gb=worker_config["memory_gb"]
            )
            
            # Prepare instance properties
            instance = {
                "name": worker_config["worker_name"],
                "machine_type": f"projects/{self.project_id}/zones/{self.zone}/machineTypes/{machine_type}",
                "disks": [
                    {
                        "boot": True,
                        "auto_delete": True,
                        "initialize_params": {
                            "source_image": self._get_default_image(),
                            "disk_size_gb": worker_config["disk_gb"],
                            "disk_type": f"projects/{self.project_id}/zones/{self.zone}/diskTypes/pd-ssd"
                        }
                    }
                ],
                "network_interfaces": [
                    {
                        "access_configs": [
                            {
                                "name": "External NAT",
                                "type": "ONE_TO_ONE_NAT"
                            }
                        ]
                    }
                ],
                "metadata": {
                    "items": [
                        {
                            "key": "startup-script",
                            "value": self._generate_startup_script(worker_config)
                        }
                    ]
                },
                "labels": {
                    "managed_by": "distributed_testing_framework"
                },
                "scheduling": {
                    "provisioning_model": "SPOT" if worker_config["use_spot"] else "STANDARD",
                    "instance_termination_action": "STOP" if worker_config["use_spot"] else None
                }
            }
            
            # Add network and subnetwork if specified
            if worker_config.get("network") or worker_config.get("subnetwork"):
                network_interface = instance["network_interfaces"][0]
                
                if worker_config.get("network"):
                    network_interface["network"] = f"projects/{self.project_id}/global/networks/{worker_config['network']}"
                
                if worker_config.get("subnetwork"):
                    network_interface["subnetwork"] = f"projects/{self.project_id}/regions/{self.region}/subnetworks/{worker_config['subnetwork']}"
            
            # Add service account if specified
            if worker_config.get("service_account"):
                instance["service_accounts"] = [
                    {
                        "email": worker_config["service_account"],
                        "scopes": ["https://www.googleapis.com/auth/cloud-platform"]
                    }
                ]
            
            # Add GPUs if requested
            if worker_config["gpu_count"] > 0 and worker_config["gpu_type"]:
                # Map common GPU type names to GCP accelerator types
                gpu_type_map = {
                    "nvidia-t4": "nvidia-tesla-t4",
                    "nvidia-v100": "nvidia-tesla-v100",
                    "nvidia-p100": "nvidia-tesla-p100",
                    "nvidia-k80": "nvidia-tesla-k80",
                    "nvidia-a100": "nvidia-tesla-a100",
                }
                
                gcp_gpu_type = gpu_type_map.get(worker_config["gpu_type"], worker_config["gpu_type"])
                
                instance["guest_accelerators"] = [
                    {
                        "accelerator_count": worker_config["gpu_count"],
                        "accelerator_type": f"projects/{self.project_id}/zones/{self.zone}/acceleratorTypes/{gcp_gpu_type}"
                    }
                ]
                
                # When using GPUs, we need to install drivers
                instance["metadata"]["items"].append({
                    "key": "install-nvidia-driver",
                    "value": "True"
                })
                
                # GPUs require certain scheduling options
                instance["scheduling"]["on_host_maintenance"] = "TERMINATE"
            
            # Add custom labels
            for key, value in worker_config.get("tags", {}).items():
                # GCP label keys must be lowercase letters, numbers, underscores, or hyphens
                # They must start with a lowercase letter, and be <= 63 characters
                label_key = key.lower().replace(" ", "_")[:63]
                label_value = str(value).lower().replace(" ", "_")[:63]
                instance["labels"][label_key] = label_value
            
            # Create the instance
            operation = self.instance_client.insert(
                project=self.project_id,
                zone=self.zone,
                instance_resource=instance
            )
            
            # Wait for the operation to complete
            while not operation.status == compute_v1.Operation.Status.DONE:
                operation = self.instance_client.get_global_operation(
                    project=self.project_id,
                    operation=operation.name
                )
                time.sleep(1)
            
            # Store worker info
            worker_id = worker_config["worker_name"]
            self.workers[worker_id] = {
                "instance_name": worker_id,
                "machine_type": machine_type,
                "launch_time": datetime.now().isoformat(),
                "state": "PROVISIONING",
                "config": worker_config,
            }
            
            logger.info(f"Launched worker {worker_id} with machine type {machine_type}")
            
            return worker_id
        
        except Exception as e:
            logger.error(f"Error creating GCP worker: {e}")
            logger.debug(traceback.format_exc())
            raise
    
    def terminate_worker(self, worker_id: str) -> bool:
        """
        Terminate a GCP Compute Engine instance.
        
        Args:
            worker_id: Instance name
        
        Returns:
            bool: Success status
        """
        try:
            operation = self.instance_client.delete(
                project=self.project_id,
                zone=self.zone,
                instance=worker_id
            )
            
            # Wait for the operation to complete
            while not operation.status == compute_v1.Operation.Status.DONE:
                operation = self.instance_client.get_global_operation(
                    project=self.project_id,
                    operation=operation.name
                )
                time.sleep(1)
            
            # Update worker state
            if worker_id in self.workers:
                self.workers[worker_id]["state"] = "TERMINATED"
            
            logger.info(f"Terminated worker {worker_id}")
            return True
        
        except Exception as e:
            logger.error(f"Error terminating worker {worker_id}: {e}")
            logger.debug(traceback.format_exc())
            return False
    
    def get_worker_status(self, worker_id: str) -> Dict[str, Any]:
        """
        Get GCP Compute Engine instance status.
        
        Args:
            worker_id: Instance name
        
        Returns:
            dict: Worker status information
        """
        try:
            instance = self.instance_client.get(
                project=self.project_id,
                zone=self.zone,
                instance=worker_id
            )
            
            # Extract network information
            public_ip = None
            private_ip = None
            for interface in instance.network_interfaces:
                if interface.network_ip:
                    private_ip = interface.network_ip
                
                if interface.access_configs:
                    for config in interface.access_configs:
                        if config.nat_ip:
                            public_ip = config.nat_ip
                            break
            
            status = {
                "instance_name": worker_id,
                "state": instance.status,
                "machine_type": instance.machine_type.split("/")[-1],
                "public_ip": public_ip,
                "private_ip": private_ip,
                "creation_timestamp": instance.creation_timestamp,
            }
            
            # Update cached worker info
            if worker_id in self.workers:
                self.workers[worker_id]["state"] = status["state"]
                self.workers[worker_id]["public_ip"] = status["public_ip"]
                self.workers[worker_id]["private_ip"] = status["private_ip"]
            
            return status
        
        except Exception as e:
            logger.error(f"Error getting worker status for {worker_id}: {e}")
            logger.debug(traceback.format_exc())
            return {"state": "error", "error": str(e)}
    
    def list_workers(self) -> Dict[str, Dict[str, Any]]:
        """
        List all managed worker nodes.
        
        Returns:
            dict: Dictionary of worker ID to worker details
        """
        try:
            # Get all instances with our labels
            instances = self.instance_client.list(
                project=self.project_id,
                zone=self.zone,
                filter="labels.managed_by=distributed_testing_framework"
            )
            
            # Process each instance
            for instance in instances:
                worker_id = instance.name
                
                # Add to workers dict if not already present
                if worker_id not in self.workers:
                    self.workers[worker_id] = {
                        "instance_name": worker_id,
                        "machine_type": instance.machine_type.split("/")[-1],
                        "launch_time": instance.creation_timestamp,
                        "state": instance.status,
                    }
                else:
                    # Update existing worker info
                    self.workers[worker_id]["state"] = instance.status
                    
                    # Extract network information
                    public_ip = None
                    private_ip = None
                    for interface in instance.network_interfaces:
                        if interface.network_ip:
                            private_ip = interface.network_ip
                        
                        if interface.access_configs:
                            for config in interface.access_configs:
                                if config.nat_ip:
                                    public_ip = config.nat_ip
                                    break
                    
                    self.workers[worker_id]["public_ip"] = public_ip
                    self.workers[worker_id]["private_ip"] = private_ip
        
        except Exception as e:
            logger.error(f"Error listing workers: {e}")
            logger.debug(traceback.format_exc())
        
        return self.workers
    
    def get_available_resources(self) -> Dict[str, Any]:
        """
        Get available GCP machine types and accelerators.
        
        Returns:
            dict: Available resource information
        """
        result = {
            "machine_types": self._get_machine_types(),
            "accelerator_types": self._get_accelerator_types(),
        }
        
        return result
    
    def _get_machine_types(self) -> Dict[str, Any]:
        """
        Get available GCP machine types in the current zone.
        
        Returns:
            dict: Machine types and their capabilities
        """
        if self._machine_types:
            return self._machine_types
        
        try:
            machine_types = {}
            
            # List machine types
            response = self.machine_types_client.list(
                project=self.project_id,
                zone=self.zone
            )
            
            for machine_type in response:
                machine_types[machine_type.name] = {
                    "vcpus": machine_type.guest_cpus,
                    "memory_gb": machine_type.memory_mb / 1024,
                    "description": machine_type.description,
                }
            
            self._machine_types = machine_types
            return machine_types
        
        except Exception as e:
            logger.error(f"Error getting machine types: {e}")
            logger.debug(traceback.format_exc())
            return {}
    
    def _get_accelerator_types(self) -> Dict[str, Any]:
        """
        Get available GCP accelerator types in the current zone.
        
        Returns:
            dict: Accelerator types and their capabilities
        """
        if self._accelerator_types:
            return self._accelerator_types
        
        try:
            accelerator_types = {}
            
            # List accelerator types
            response = self.accelerator_types_client.list(
                project=self.project_id,
                zone=self.zone
            )
            
            for accelerator_type in response:
                accelerator_types[accelerator_type.name] = {
                    "max_count": accelerator_type.maximum_cards_per_instance,
                    "description": accelerator_type.description,
                }
            
            self._accelerator_types = accelerator_types
            return accelerator_types
        
        except Exception as e:
            logger.error(f"Error getting accelerator types: {e}")
            logger.debug(traceback.format_exc())
            return {}
    
    def _select_machine_type(self, cpu_cores: int, memory_gb: int) -> str:
        """
        Select an appropriate GCP machine type based on requirements.
        
        Args:
            cpu_cores: Number of CPU cores required
            memory_gb: Memory in GB required
        
        Returns:
            str: Selected GCP machine type
        """
        machine_types = self._get_machine_types()
        
        # Define scoring function for machine types
        def score_machine_type(machine_type_info):
            # Higher score = better match
            score = 0
            
            # CPU score - prefer slightly more than required
            cpu_ratio = machine_type_info["vcpus"] / cpu_cores
            if cpu_ratio < 1:
                # Not enough CPUs
                return -1
            elif cpu_ratio <= 1.5:
                # Close match
                score += 100
            elif cpu_ratio <= 2:
                # Slight overprovisioning
                score += 80
            else:
                # Too much overprovisioning
                score += 50 / cpu_ratio
            
            # Memory score - prefer slightly more than required
            memory_ratio = machine_type_info["memory_gb"] / memory_gb
            if memory_ratio < 1:
                # Not enough memory
                return -1
            elif memory_ratio <= 1.25:
                # Close match
                score += 100
            elif memory_ratio <= 2:
                # Slight overprovisioning
                score += 80
            else:
                # Too much overprovisioning
                score += 50 / memory_ratio
            
            return score
        
        # Score and filter machine types
        scored_machines = []
        for machine_type, info in machine_types.items():
            score = score_machine_type(info)
            if score > 0:
                scored_machines.append((machine_type, score))
        
        if not scored_machines:
            # No machines match criteria, use a default based on requirements
            if cpu_cores <= 2 and memory_gb <= 4:
                return "n1-standard-2"
            elif cpu_cores <= 4 and memory_gb <= 16:
                return "n1-standard-4"
            elif cpu_cores <= 8 and memory_gb <= 32:
                return "n1-standard-8"
            else:
                return "n1-standard-16"
        
        # Sort by score (descending)
        scored_machines.sort(key=lambda x: x[1], reverse=True)
        
        # Return best match
        return scored_machines[0][0]
    
    def _generate_startup_script(self, config: Dict[str, Any]) -> str:
        """
        Generate startup script for GCP instance.
        
        Args:
            config: Worker configuration
        
        Returns:
            str: Startup script
        """
        script = """#!/bin/bash
set -e

echo "Starting worker setup..."

# Install dependencies
apt-get update
apt-get install -y python3 python3-pip git unzip

# Set up environment variables
"""
        
        # Add environment variables
        for key, value in config["environment"].items():
            script += f'export {key}="{value}"\n'
        
        # Add coordinator and API key environment variables
        script += f'export COORDINATOR_URL="{config["coordinator_url"]}"\n'
        script += f'export WORKER_API_KEY="{config["api_key"]}"\n'
        script += f'export WORKER_ID="{config["worker_name"]}"\n'
        
        # Clone repository or download worker code
        script += """
# Clone the repository (if not using worker_zip)
git clone https://github.com/example/ipfs_accelerate_py.git
cd ipfs_accelerate_py

# Install requirements
pip3 install -r requirements.txt

# Start the worker
python3 -m duckdb_api.distributed_testing.worker --coordinator $COORDINATOR_URL --api-key $WORKER_API_KEY
"""
        
        # Return the script
        return script
    
    def _get_default_image(self) -> str:
        """
        Get the latest Ubuntu 22.04 image for GCP.
        
        Returns:
            str: Image URL
        """
        try:
            # List Ubuntu images from the ubuntu-os-cloud project
            response = self.image_client.list(project="ubuntu-os-cloud")
            
            # Filter for Ubuntu 22.04 (Jammy Jellyfish)
            ubuntu_images = [img for img in response if "ubuntu-2204" in img.name]
            
            # Sort by creation time (newest first)
            ubuntu_images.sort(key=lambda img: img.creation_timestamp, reverse=True)
            
            if ubuntu_images:
                return f"projects/ubuntu-os-cloud/global/images/{ubuntu_images[0].name}"
            
            # Fallback to a known image
            return "projects/ubuntu-os-cloud/global/images/ubuntu-2204-jammy-v20230421"
        
        except Exception as e:
            logger.error(f"Error getting default image: {e}")
            logger.debug(traceback.format_exc())
            
            # Return a default Ubuntu 22.04 image
            return "projects/ubuntu-os-cloud/global/images/ubuntu-2204-jammy-v20230421"


class DockerLocalProvider(CloudProviderBase):
    """Docker local provider implementation."""
    
    def __init__(self, region: str = "local", **kwargs):
        """
        Initialize Docker local provider.
        
        Args:
            region: Ignored for Docker (use "local")
            **kwargs: Additional Docker-specific arguments
                docker_host: Docker host URL (optional)
                network_name: Docker network name (optional)
        """
        super().__init__(region, **kwargs)
        
        if not DOCKER_AVAILABLE:
            logger.error("docker not available. Docker integration disabled.")
            raise ImportError("docker not available. Docker integration disabled.")
        
        # Initialize Docker client
        docker_host = kwargs.get("docker_host")
        self.client = docker.from_env() if not docker_host else docker.DockerClient(base_url=docker_host)
        
        # Docker-specific options
        self.network_name = kwargs.get("network_name", "bridge")
        
        logger.info(f"Docker local provider initialized")
    
    def create_worker(self, config: Dict[str, Any]) -> str:
        """
        Create a worker container using Docker.
        
        Args:
            config: Worker configuration
                worker_name: Worker name/container name
                worker_docker_image: Docker image for the worker
                coordinator_url: Coordinator URL
                api_key: API key for worker authentication
                environment: Environment variables for the container
                cpu_cores: Number of CPU cores (maps to Docker --cpus)
                memory_gb: Memory in GB (maps to Docker --memory)
                gpu_count: Number of GPUs (maps to all GPUs with --gpus all or specific IDs)
        
        Returns:
            str: Worker ID (Docker container ID)
        """
        worker_config = self.get_base_worker_config(config)
        
        try:
            # Prepare environment variables
            environment = worker_config["environment"].copy()
            environment["COORDINATOR_URL"] = worker_config["coordinator_url"]
            environment["WORKER_API_KEY"] = worker_config["api_key"]
            environment["WORKER_ID"] = worker_config["worker_name"]
            
            # Prepare resource limits
            cpu_limit = worker_config["cpu_cores"]
            memory_limit = f"{int(worker_config['memory_gb'] * 1024)}m"  # Convert GB to MB
            
            # Prepare GPU settings
            device_requests = []
            if worker_config["gpu_count"] > 0:
                if worker_config["gpu_count"] >= 1:
                    # Request all GPUs
                    device_requests.append(
                        docker.types.DeviceRequest(count=-1, capabilities=[['gpu']])
                    )
            
            # Create and start the container
            container = self.client.containers.run(
                image=worker_config["worker_docker_image"],
                name=worker_config["worker_name"],
                environment=environment,
                network=self.network_name,
                detach=True,
                cpu_quota=int(cpu_limit * 100000),
                cpu_period=100000,
                mem_limit=memory_limit,
                device_requests=device_requests,
                restart_policy={"Name": "unless-stopped"},
                labels={
                    "managed_by": "distributed_testing_framework",
                    **worker_config.get("tags", {})
                }
            )
            
            # Store worker info
            worker_id = container.id
            self.workers[worker_id] = {
                "container_id": worker_id,
                "worker_name": worker_config["worker_name"],
                "launch_time": datetime.now().isoformat(),
                "state": "running",
                "config": worker_config,
            }
            
            logger.info(f"Created Docker worker {worker_config['worker_name']} with ID {worker_id}")
            
            return worker_id
        
        except Exception as e:
            logger.error(f"Error creating Docker worker: {e}")
            logger.debug(traceback.format_exc())
            raise
    
    def terminate_worker(self, worker_id: str) -> bool:
        """
        Terminate a Docker container.
        
        Args:
            worker_id: Container ID
        
        Returns:
            bool: Success status
        """
        try:
            container = self.client.containers.get(worker_id)
            container.stop()
            container.remove()
            
            # Update worker state
            if worker_id in self.workers:
                self.workers[worker_id]["state"] = "terminated"
            
            logger.info(f"Terminated Docker worker with ID {worker_id}")
            return True
        
        except Exception as e:
            logger.error(f"Error terminating Docker worker {worker_id}: {e}")
            logger.debug(traceback.format_exc())
            return False
    
    def get_worker_status(self, worker_id: str) -> Dict[str, Any]:
        """
        Get Docker container status.
        
        Args:
            worker_id: Container ID
        
        Returns:
            dict: Worker status information
        """
        try:
            container = self.client.containers.get(worker_id)
            
            status = {
                "container_id": worker_id,
                "state": container.status,
                "image": container.image.tags[0] if container.image.tags else container.image.id,
                "name": container.name,
                "created": container.attrs["Created"],
                "ip_address": self._get_container_ip(container),
            }
            
            # Update cached worker info
            if worker_id in self.workers:
                self.workers[worker_id]["state"] = status["state"]
                self.workers[worker_id]["ip_address"] = status["ip_address"]
            
            return status
        
        except Exception as e:
            logger.error(f"Error getting Docker worker status for {worker_id}: {e}")
            logger.debug(traceback.format_exc())
            return {"state": "error", "error": str(e)}
    
    def list_workers(self) -> Dict[str, Dict[str, Any]]:
        """
        List all managed Docker containers.
        
        Returns:
            dict: Dictionary of worker ID to worker details
        """
        try:
            # Get all containers with our label
            containers = self.client.containers.list(
                all=True,
                filters={"label": "managed_by=distributed_testing_framework"}
            )
            
            # Process each container
            for container in containers:
                worker_id = container.id
                
                # Add to workers dict if not already present
                if worker_id not in self.workers:
                    self.workers[worker_id] = {
                        "container_id": worker_id,
                        "worker_name": container.name,
                        "launch_time": container.attrs["Created"],
                        "state": container.status,
                        "ip_address": self._get_container_ip(container),
                    }
                else:
                    # Update existing worker info
                    self.workers[worker_id]["state"] = container.status
                    self.workers[worker_id]["ip_address"] = self._get_container_ip(container)
        
        except Exception as e:
            logger.error(f"Error listing Docker workers: {e}")
            logger.debug(traceback.format_exc())
        
        return self.workers
    
    def get_available_resources(self) -> Dict[str, Any]:
        """
        Get available Docker resources.
        
        Returns:
            dict: Available resource information
        """
        try:
            # Get Docker info
            info = self.client.info()
            
            resources = {
                "cpu_cores": info["NCPU"],
                "memory_gb": info["MemTotal"] / (1024 * 1024 * 1024),  # Convert bytes to GB
                "docker_version": info["ServerVersion"],
                "containers_running": info["ContainersRunning"],
                "containers_total": info["Containers"],
                "images": len(self.client.images.list()),
                "operating_system": info["OperatingSystem"],
                "architecture": info["Architecture"],
            }
            
            # Check if NVIDIA GPU support is available
            try:
                gpu_info = self.client.containers.run(
                    "nvidia/cuda:11.0-base",
                    "nvidia-smi --query-gpu=name,memory.total,memory.free,temperature.gpu --format=csv,noheader",
                    remove=True
                )
                
                # Parse GPU info
                gpu_data = []
                for line in gpu_info.decode().strip().split("\n"):
                    parts = line.split(", ")
                    if len(parts) >= 3:
                        gpu_data.append({
                            "name": parts[0],
                            "memory_total": parts[1],
                            "memory_free": parts[2],
                            "temperature": parts[3] if len(parts) > 3 else "N/A"
                        })
                
                resources["gpus"] = gpu_data
                resources["gpu_available"] = len(gpu_data) > 0
            
            except Exception as e:
                logger.warning(f"NVIDIA GPU detection failed: {e}")
                resources["gpu_available"] = False
            
            return resources
        
        except Exception as e:
            logger.error(f"Error getting Docker resources: {e}")
            logger.debug(traceback.format_exc())
            return {}
    
    def _get_container_ip(self, container) -> Optional[str]:
        """
        Get the IP address of a container.
        
        Args:
            container: Docker container object
        
        Returns:
            str: IP address or None
        """
        try:
            # Refresh container to get latest network info
            container.reload()
            
            networks = container.attrs["NetworkSettings"]["Networks"]
            
            # Try to get IP from the specified network first
            if self.network_name in networks:
                return networks[self.network_name]["IPAddress"]
            
            # Otherwise, get the first available IP
            for network in networks.values():
                if network["IPAddress"]:
                    return network["IPAddress"]
            
            return None
        
        except Exception as e:
            logger.error(f"Error getting container IP: {e}")
            logger.debug(traceback.format_exc())
            return None


class CloudProviderManager:
    """Manager for cloud providers."""
    
    def __init__(self, provider: str, region: str, **kwargs):
        """
        Initialize cloud provider manager.
        
        Args:
            provider: Cloud provider type ("aws", "gcp", "azure", "docker")
            region: Cloud provider region
            **kwargs: Provider-specific arguments
        """
        self.provider_type = provider.lower()
        self.region = region
        
        # Initialize the appropriate cloud provider
        if self.provider_type == "aws":
            if not AWS_AVAILABLE:
                raise ImportError("boto3 not available. AWS integration disabled.")
            self.provider = AWSCloudProvider(region, **kwargs)
        
        elif self.provider_type == "gcp":
            if not GCP_AVAILABLE:
                raise ImportError("google.cloud.compute_v1 not available. GCP integration disabled.")
            self.provider = GCPCloudProvider(region, **kwargs)
        
        elif self.provider_type == "azure":
            if not AZURE_AVAILABLE:
                raise ImportError("azure.mgmt.compute not available. Azure integration disabled.")
            raise NotImplementedError("Azure cloud provider not yet implemented")
        
        elif self.provider_type == "docker":
            if not DOCKER_AVAILABLE:
                raise ImportError("docker not available. Docker integration disabled.")
            self.provider = DockerLocalProvider(region, **kwargs)
        
        else:
            raise ValueError(f"Unsupported cloud provider: {provider}")
        
        logger.info(f"Initialized {provider} cloud provider manager for region {region}")
    
    def create_worker(self, config: Dict[str, Any]) -> str:
        """
        Create a worker node.
        
        Args:
            config: Worker configuration
        
        Returns:
            str: Worker ID
        """
        return self.provider.create_worker(config)
    
    def terminate_worker(self, worker_id: str) -> bool:
        """
        Terminate a worker node.
        
        Args:
            worker_id: Worker ID
        
        Returns:
            bool: Success status
        """
        return self.provider.terminate_worker(worker_id)
    
    def get_worker_status(self, worker_id: str) -> Dict[str, Any]:
        """
        Get worker node status.
        
        Args:
            worker_id: Worker ID
        
        Returns:
            dict: Worker status information
        """
        return self.provider.get_worker_status(worker_id)
    
    def list_workers(self) -> Dict[str, Dict[str, Any]]:
        """
        List all managed worker nodes.
        
        Returns:
            dict: Dictionary of worker ID to worker details
        """
        return self.provider.list_workers()
    
    def get_available_resources(self) -> Dict[str, Any]:
        """
        Get available cloud resources.
        
        Returns:
            dict: Available resource information
        """
        return self.provider.get_available_resources()


# Main function for testing
if __name__ == "__main__":
    """Run standalone test of the Cloud Provider Integration."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Cloud Provider Integration")
    parser.add_argument("--provider", choices=["aws", "gcp", "docker"], default="docker", help="Cloud provider to test")
    parser.add_argument("--region", default="us-west-2", help="Cloud region (for AWS/GCP)")
    parser.add_argument("--action", choices=["create", "status", "list", "terminate"], required=True, help="Action to perform")
    parser.add_argument("--worker-id", help="Worker ID for status/terminate actions")
    
    args = parser.parse_args()
    
    # Initialize cloud provider manager
    if args.provider == "docker":
        manager = CloudProviderManager(args.provider, "local")
    else:
        manager = CloudProviderManager(args.provider, args.region)
    
    # Perform the requested action
    if args.action == "create":
        config = {
            "worker_name": f"test-worker-{uuid.uuid4().hex[:8]}",
            "cpu_cores": 2,
            "memory_gb": 4,
            "coordinator_url": "http://localhost:8080",
            "api_key": "test-api-key",
        }
        
        worker_id = manager.create_worker(config)
        print(f"Created worker: {worker_id}")
    
    elif args.action == "status":
        if not args.worker_id:
            print("Error: worker-id is required for status action")
            sys.exit(1)
        
        status = manager.get_worker_status(args.worker_id)
        print(f"Worker status: {json.dumps(status, indent=2)}")
    
    elif args.action == "list":
        workers = manager.list_workers()
        print(f"Workers: {json.dumps(workers, indent=2)}")
    
    elif args.action == "terminate":
        if not args.worker_id:
            print("Error: worker-id is required for terminate action")
            sys.exit(1)
        
        result = manager.terminate_worker(args.worker_id)
        print(f"Terminated worker {args.worker_id}: {result}")