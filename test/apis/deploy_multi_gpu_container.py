#!/usr/bin/env python3
"""
Multi-GPU Container Deployment Script

This script automates the deployment of ML models in containers with multi-GPU support.
It handles:
1. Hardware detection and optimal GPU selection
2. Container runtime configuration with proper GPU arguments
3. Environment variable setup for multi-GPU training and inference
4. Container lifecycle management (deployment, monitoring, shutdown)

Usage:
  python deploy_multi_gpu_container.py --model <hf_model_id> --image <docker_image> [--devices cuda:0 cuda:1]
"""

import os
import sys
import time
import json
import argparse
import subprocess
import logging
from typing import Dict, List, Optional, Union, Any, Tuple

# Add repository root to path to import modules
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'ipfs_accelerate_py'))

# Import utility modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.device_mapper import DeviceMapper
from utils.multi_gpu_utils import get_container_gpu_config, detect_optimal_device_configuration

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Deploy containers with multi-GPU support")
    
    # Model and container configuration
    parser.add_argument("--model", type=str, required=True,
                       help="Hugging Face model ID or local path to deploy")
    parser.add_argument("--image", type=str, default="huggingface/text-generation-inference:latest",
                       help="Docker image to use for deployment")
    parser.add_argument("--api-type", type=str, default="tgi",
                       choices=["tgi", "tei", "vllm", "ollama"],
                       help="Type of API to deploy")
    
    # Hardware configuration
    parser.add_argument("--devices", type=str, nargs="+",
                       help="Specific devices to use (e.g., cuda:0 cuda:1)")
    parser.add_argument("--auto-select", action="store_true",
                       help="Automatically select optimal devices")
    
    # Container networking
    parser.add_argument("--port", type=int, default=8080,
                       help="Port to expose the container API on")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                       help="Host interface to bind to")
    
    # Advanced options
    parser.add_argument("--container-name", type=str,
                       help="Custom container name (default: auto-generated)")
    parser.add_argument("--env", type=str, nargs="+",
                       help="Additional environment variables (KEY=VALUE)")
    parser.add_argument("--volumes", type=str, nargs="+",
                       help="Volume mounts (SOURCE:TARGET)")
    parser.add_argument("--strategy", type=str, default="auto",
                       choices=["auto", "tensor-parallel", "pipeline-parallel", "zero"],
                       help="Parallelism strategy for multi-GPU deployment")
    
    # Execution modes
    parser.add_argument("--dry-run", action="store_true",
                       help="Print container command without executing")
    parser.add_argument("--detect-only", action="store_true",
                       help="Only run device detection and exit")
    
    return parser.parse_args()

def deploy_container(
    model_id: str,
    image: str,
    api_type: str = "tgi",
    devices: Optional[List[str]] = None,
    port: int = 8080,
    host: str = "0.0.0.0",
    container_name: Optional[str] = None,
    env_vars: Optional[List[str]] = None,
    volumes: Optional[List[str]] = None,
    strategy: str = "auto",
    dry_run: bool = False
) -> Tuple[bool, str]:
    """
    Deploy a container with multi-GPU support.
    
    Args:
        model_id: Hugging Face model ID or local path
        image: Docker image to use
        api_type: Type of API to deploy (tgi, tei, vllm, ollama)
        devices: List of devices to use
        port: Port to expose
        host: Host interface to bind to
        container_name: Custom container name
        env_vars: Additional environment variables
        volumes: Volume mounts
        strategy: Parallelism strategy
        dry_run: If True, print command without executing
        
    Returns:
        Tuple of (success, container_id or error_message)
    """
    # Get container GPU configuration
    container_config = get_container_gpu_config(devices)
    
    # Generate a container name if not provided
    if not container_name:
        model_name = model_id.split("/")[-1] if "/" in model_id else model_id
        api_suffix = api_type.lower()
        container_name = f"{model_name}-{api_suffix}-{int(time.time())}"
    
    # Prepare environment variables
    env_list = container_config["environment"]
    
    # Add model ID and API-specific environment variables
    if api_type.lower() == "tgi":
        env_list["MODEL_ID"] = model_id
        env_list["MAX_INPUT_LENGTH"] = "2048"
        env_list["MAX_TOTAL_TOKENS"] = "4096"
        env_list["TRUST_REMOTE_CODE"] = "true"
    elif api_type.lower() == "tei":
        env_list["MODEL_ID"] = model_id
        env_list["TRUST_REMOTE_CODE"] = "true"
    elif api_type.lower() == "vllm":
        env_list["MODEL"] = model_id
        env_list["TENSOR_PARALLEL_SIZE"] = str(len(container_config["devices"]) if container_config["devices"] else 1)
    elif api_type.lower() == "ollama":
        env_list["OLLAMA_MODEL"] = model_id
    
    # Add user-provided environment variables
    if env_vars:
        for env_var in env_vars:
            if "=" in env_var:
                key, value = env_var.split("=", 1)
                env_list[key] = value
    
    # Prepare volume mounts
    volume_args = []
    if volumes:
        for volume in volumes:
            volume_args.extend(["-v", volume])
    
    # Prepare environment variable arguments
    env_args = []
    for key, value in env_list.items():
        env_args.extend(["-e", f"{key}={value}"])
    
    # Prepare port mapping
    port_mapping = f"{host}:{port}:80"
    
    # Build docker command
    cmd = [
        "docker", "run", "-d",
        "--name", container_name,
        "-p", port_mapping,
        "--shm-size", "1g"  # Shared memory for multi-GPU communication
    ]
    
    # Add GPU arguments if available
    if container_config["gpu_arg"]:
        cmd.extend(container_config["gpu_arg"].split())
    
    # Add environment variables
    cmd.extend(env_args)
    
    # Add volume mounts
    cmd.extend(volume_args)
    
    # Add image name
    cmd.append(image)
    
    # Log the command
    logger.info(f"Prepared container deployment command: {' '.join(cmd)}")
    
    # If dry run, just return the command
    if dry_run:
        return True, " ".join(cmd)
    
    # Execute command
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        container_id = result.stdout.strip()
        logger.info(f"Container deployed successfully with ID: {container_id}")
        
        # Wait a moment for the container to start
        time.sleep(2)
        
        # Check container status
        status_cmd = ["docker", "inspect", "--format", "{{.State.Status}}", container_id]
        status_result = subprocess.run(status_cmd, capture_output=True, text=True)
        
        if status_result.stdout.strip() != "running":
            # Get container logs to determine the issue
            logs_cmd = ["docker", "logs", container_id]
            logs_result = subprocess.run(logs_cmd, capture_output=True, text=True)
            error_msg = f"Container is not running. Status: {status_result.stdout.strip()}\nLogs: {logs_result.stdout}"
            logger.error(error_msg)
            return False, error_msg
        
        return True, container_id
    
    except subprocess.CalledProcessError as e:
        error_msg = f"Failed to deploy container: {e.stderr}"
        logger.error(error_msg)
        return False, error_msg
    
    except Exception as e:
        error_msg = f"Error deploying container: {str(e)}"
        logger.error(error_msg)
        return False, error_msg

def monitor_container(container_id: str, timeout: int = 60) -> Dict[str, Any]:
    """
    Monitor a deployed container and wait for it to be ready.
    
    Args:
        container_id: Container ID or name
        timeout: Timeout in seconds
        
    Returns:
        Status information dictionary
    """
    logger.info(f"Monitoring container {container_id}...")
    start_time = time.time()
    status = {"status": "unknown", "ready": False, "logs": ""}
    
    while time.time() - start_time < timeout:
        try:
            # Check container status
            status_cmd = ["docker", "inspect", "--format", "{{.State.Status}}", container_id]
            status_result = subprocess.run(status_cmd, capture_output=True, text=True)
            
            # Get container logs
            logs_cmd = ["docker", "logs", container_id]
            logs_result = subprocess.run(logs_cmd, capture_output=True, text=True)
            
            status["status"] = status_result.stdout.strip()
            status["logs"] = logs_result.stdout
            
            # Check if service is ready based on logs
            if "Model loaded" in logs_result.stdout or "Server started" in logs_result.stdout:
                status["ready"] = True
                logger.info(f"Container {container_id} is ready")
                break
            
            # If container stopped or has an error, break
            if status["status"] not in ["running", "starting"]:
                logger.error(f"Container {container_id} has status: {status['status']}")
                break
            
            # Wait before next check
            time.sleep(5)
            
        except Exception as e:
            logger.error(f"Error monitoring container: {str(e)}")
            status["error"] = str(e)
            break
    
    if not status["ready"] and status["status"] == "running":
        logger.warning(f"Container {container_id} is running but might not be fully initialized after {timeout}s")
        status["ready"] = "unknown"
    
    return status

def shutdown_container(container_id: str) -> bool:
    """
    Stop and remove a container.
    
    Args:
        container_id: Container ID or name
        
    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Shutting down container {container_id}...")
    
    try:
        # Stop container
        stop_cmd = ["docker", "stop", container_id]
        subprocess.run(stop_cmd, capture_output=True, text=True, check=True)
        
        # Remove container
        rm_cmd = ["docker", "rm", container_id]
        subprocess.run(rm_cmd, capture_output=True, text=True, check=True)
        
        logger.info(f"Container {container_id} stopped and removed")
        return True
    
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to shutdown container: {e.stderr}")
        return False
    
    except Exception as e:
        logger.error(f"Error shutting down container: {str(e)}")
        return False

def get_container_status(container_id: str) -> Dict[str, Any]:
    """
    Get detailed status information for a container.
    
    Args:
        container_id: Container ID or name
        
    Returns:
        Status information dictionary
    """
    status = {"id": container_id, "running": False}
    
    try:
        # Get container status
        inspect_cmd = ["docker", "inspect", container_id]
        result = subprocess.run(inspect_cmd, capture_output=True, text=True, check=True)
        container_info = json.loads(result.stdout)[0]
        
        # Extract status information
        status["running"] = container_info["State"]["Running"]
        status["status"] = container_info["State"]["Status"]
        status["started_at"] = container_info["State"]["StartedAt"]
        
        # Get exposed ports
        status["ports"] = container_info["NetworkSettings"]["Ports"]
        
        # Get GPU information
        if "Nvidia" in container_info.get("HostConfig", {}).get("DeviceRequests", [{}])[0].get("Driver", ""):
            status["gpu_enabled"] = True
        else:
            status["gpu_enabled"] = False
        
        # Get environment variables
        status["environment"] = container_info["Config"]["Env"]
        
        return status
    
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to get container status: {e.stderr}")
        return {"id": container_id, "error": e.stderr}
    
    except Exception as e:
        logger.error(f"Error getting container status: {str(e)}")
        return {"id": container_id, "error": str(e)}

def main():
    """Main entry point"""
    args = parse_args()
    
    # Set up device mapper
    mapper = DeviceMapper()
    
    # Run hardware detection and print information
    hardware = mapper.device_info
    logger.info(f"Detected hardware: {json.dumps(hardware, indent=2)}")
    
    # If detect-only mode, exit
    if args.detect_only:
        logger.info("Device detection completed. Exiting.")
        sys.exit(0)
    
    # Get optimal device configuration if auto-select is enabled
    if args.auto_select and not args.devices:
        recommendations = detect_optimal_device_configuration(args.model)
        logger.info(f"Recommended configuration: {json.dumps(recommendations, indent=2)}")
        
        # Use recommended devices
        if recommendations["recommended_approach"] == "multi_gpu":
            # Collect all available GPUs
            devices = []
            for device_type in ["cuda", "rocm"]:
                if hardware[device_type]["count"] > 0:
                    for i in range(hardware[device_type]["count"]):
                        devices.append(f"{device_type}:{i}")
            
            logger.info(f"Auto-selected devices: {devices}")
            args.devices = devices
        elif recommendations["recommended_approach"] == "single_gpu":
            # Use the single GPU that was recommended
            device = recommendations["recommendations"]["single_gpu"]["device"]
            logger.info(f"Auto-selected device: {device}")
            args.devices = [device]
        else:
            # Fall back to CPU (no devices)
            logger.info("Auto-selected CPU (no GPU devices)")
            args.devices = None
    
    # Deploy the container
    success, result = deploy_container(
        model_id=args.model,
        image=args.image,
        api_type=args.api_type,
        devices=args.devices,
        port=args.port,
        host=args.host,
        container_name=args.container_name,
        env_vars=args.env,
        volumes=args.volumes,
        strategy=args.strategy,
        dry_run=args.dry_run
    )
    
    # If dry run, just print the command and exit
    if args.dry_run:
        print(f"Dry run - Docker command:\n{result}")
        sys.exit(0)
    
    # If deployment failed, exit
    if not success:
        logger.error(f"Container deployment failed: {result}")
        sys.exit(1)
    
    # Monitor the container
    container_id = result
    status = monitor_container(container_id)
    
    # Print status and connection information
    if status["ready"]:
        logger.info(f"Container {container_id} is ready")
        detailed_status = get_container_status(container_id)
        
        # Print connection information
        print("\n" + "=" * 60)
        print(f"Container {container_id} deployed successfully")
        print(f"API Type: {args.api_type.upper()}")
        print(f"Model: {args.model}")
        print(f"API URL: http://{args.host if args.host != '0.0.0.0' else 'localhost'}:{args.port}")
        print(f"Health URL: http://{args.host if args.host != '0.0.0.0' else 'localhost'}:{args.port}/health")
        print("=" * 60 + "\n")
        
        # Print example curl commands
        if args.api_type.lower() == "tgi":
            print("Example usage:")
            print(f'curl -X POST "http://localhost:{args.port}/generate" \\')
            print('  -H "Content-Type: application/json" \\')
            print('  -d \'{"inputs": "Once upon a time", "parameters": {"max_new_tokens": 20}}\'')
        elif args.api_type.lower() == "tei":
            print("Example usage:")
            print(f'curl -X POST "http://localhost:{args.port}/generate_text" \\')
            print('  -H "Content-Type: application/json" \\')
            print('  -d \'{"text": "Once upon a time", "max_length": 50}\'')
        elif args.api_type.lower() == "vllm":
            print("Example usage:")
            print(f'curl -X POST "http://localhost:{args.port}/generate" \\')
            print('  -H "Content-Type: application/json" \\')
            print('  -d \'{"prompt": "Once upon a time", "max_tokens": 20}\'')
        print("\n")
        
        # Print GPU information
        if detailed_status.get("gpu_enabled", False):
            print("GPU Configuration:")
            print(f"Devices: {args.devices}")
            if args.api_type.lower() == "tgi" or args.api_type.lower() == "tei":
                print(f"Model Shards: {detailed_status.get('environment', [])}")
            elif args.api_type.lower() == "vllm":
                print(f"Tensor Parallel Size: {len(args.devices) if args.devices else 1}")
            print("\n")
        
        print("To stop the container:")
        print(f"  docker stop {container_id}")
        print(f"  docker rm {container_id}")
        print("\n")
    else:
        logger.error(f"Container {container_id} initialization failed or timed out")
        print("\nContainer logs:")
        print(status["logs"])
        sys.exit(1)

if __name__ == "__main__":
    main()