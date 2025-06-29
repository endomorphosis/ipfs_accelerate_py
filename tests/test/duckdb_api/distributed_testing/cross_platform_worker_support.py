#!/usr/bin/env python3
"""
Distributed Testing Framework - Cross-Platform Worker Support

This module provides cross-platform support for worker deployment and management
across different operating systems and container environments. It handles platform-specific
differences in hardware detection, resource management, and worker deployment.

Key features:
- Platform detection and abstraction
- Platform-specific hardware detection
- Environment-aware deployment scripts
- Container support for consistent environments
- Resource monitoring tailored to platform capabilities
"""

import os
import sys
import json
import uuid
import platform
import logging
import subprocess
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("cross_platform_worker")

# Add parent directory to path to import modules from parent
parent_dir = str(Path(__file__).parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


class CrossPlatformWorkerSupport:
    """Provides cross-platform support for worker deployment and management."""
    
    def __init__(self, config_path=None):
        """Initialize the cross-platform worker support.
        
        Args:
            config_path: Path to configuration file (optional)
        """
        self.platform_handlers = {
            "linux": LinuxPlatformHandler(),
            "windows": WindowsPlatformHandler(),
            "darwin": MacOSPlatformHandler(),
            "container": ContainerPlatformHandler()
        }
        
        self.current_platform = self._detect_platform()
        self.handler = self.platform_handlers.get(self.current_platform)
        
        # Load configuration
        self.config = {}
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    self.config = json.load(f)
            except json.JSONDecodeError as e:
                logger.error(f"Error loading configuration: {e}")
                
        logger.info(f"Initialized cross-platform support for {self.current_platform}")
    
    def _detect_platform(self):
        """Detect the current platform.
        
        Returns:
            str: Platform identifier ("linux", "windows", "darwin", or "container")
        """
        # Check if running in a container
        if os.environ.get("CONTAINER_ENV") or os.path.exists("/.dockerenv"):
            return "container"
            
        # Detect OS platform
        system = platform.system().lower()
        if system in self.platform_handlers:
            return system
            
        # Default to Linux if unknown
        logger.warning(f"Unknown platform: {system}, defaulting to Linux")
        return "linux"
    
    def get_worker_command(self, config):
        """Get the platform-specific command to start a worker.
        
        Args:
            config: Worker configuration dictionary
            
        Returns:
            str: Command to start a worker
        """
        if not self.handler:
            raise ValueError(f"Unsupported platform: {self.current_platform}")
            
        return self.handler.get_worker_command(config)
    
    def create_deployment_script(self, config, output_path):
        """Create a platform-specific deployment script.
        
        Args:
            config: Worker configuration dictionary
            output_path: Path to save the deployment script
            
        Returns:
            str: Path to the created script
        """
        if not self.handler:
            raise ValueError(f"Unsupported platform: {self.current_platform}")
            
        return self.handler.create_deployment_script(config, output_path)
    
    def install_dependencies(self, dependencies=None):
        """Install platform-specific dependencies.
        
        Args:
            dependencies: List of dependencies to install (if None, install defaults)
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.handler:
            raise ValueError(f"Unsupported platform: {self.current_platform}")
            
        return self.handler.install_dependencies(dependencies)
    
    def detect_hardware(self):
        """Detect hardware capabilities in a platform-specific way.
        
        Returns:
            Dict: Hardware capabilities
        """
        if not self.handler:
            raise ValueError(f"Unsupported platform: {self.current_platform}")
            
        return self.handler.detect_hardware()
    
    def get_startup_script(self, coordinator_url, api_key, worker_id=None):
        """Generate a platform-specific worker startup script.
        
        Args:
            coordinator_url: URL of the coordinator
            api_key: API key for authentication
            worker_id: Worker ID (if None, generate a new one)
            
        Returns:
            str: Worker startup script content
        """
        if not self.handler:
            raise ValueError(f"Unsupported platform: {self.current_platform}")
            
        if not worker_id:
            worker_id = f"worker_{uuid.uuid4().hex[:8]}"
            
        return self.handler.get_startup_script(coordinator_url, api_key, worker_id)
    
    def convert_path_for_platform(self, path):
        """Convert a path to the correct format for the current platform.
        
        Args:
            path: Path to convert
            
        Returns:
            str: Converted path
        """
        if not self.handler:
            raise ValueError(f"Unsupported platform: {self.current_platform}")
            
        return self.handler.convert_path(path)
    
    def get_platform_info(self):
        """Get information about the current platform.
        
        Returns:
            Dict: Platform information
        """
        info = {
            "platform": self.current_platform,
            "python_version": platform.python_version(),
            "system": platform.system(),
            "release": platform.release(),
            "architecture": platform.machine(),
        }
        
        # Add platform-specific info if handler is available
        if self.handler:
            info.update(self.handler.get_platform_info())
            
        return info


class PlatformHandler:
    """Base class for platform-specific handlers."""
    
    def get_worker_command(self, config):
        """Get the command to start a worker.
        
        Args:
            config: Worker configuration dictionary
            
        Returns:
            str: Command to start a worker
        """
        raise NotImplementedError()
    
    def create_deployment_script(self, config, output_path):
        """Create a deployment script.
        
        Args:
            config: Worker configuration dictionary
            output_path: Path to save the deployment script
            
        Returns:
            str: Path to the created script
        """
        raise NotImplementedError()
    
    def install_dependencies(self, dependencies=None):
        """Install dependencies.
        
        Args:
            dependencies: List of dependencies to install
            
        Returns:
            bool: True if successful, False otherwise
        """
        raise NotImplementedError()
    
    def detect_hardware(self):
        """Detect hardware capabilities.
        
        Returns:
            Dict: Hardware capabilities
        """
        raise NotImplementedError()
    
    def get_startup_script(self, coordinator_url, api_key, worker_id):
        """Generate a worker startup script.
        
        Args:
            coordinator_url: URL of the coordinator
            api_key: API key for authentication
            worker_id: Worker ID
            
        Returns:
            str: Worker startup script content
        """
        raise NotImplementedError()
    
    def convert_path(self, path):
        """Convert a path to the correct format for this platform.
        
        Args:
            path: Path to convert
            
        Returns:
            str: Converted path
        """
        raise NotImplementedError()
    
    def get_platform_info(self):
        """Get platform-specific information.
        
        Returns:
            Dict: Platform information
        """
        raise NotImplementedError()


class LinuxPlatformHandler(PlatformHandler):
    """Linux-specific platform handler."""
    
    def get_worker_command(self, config):
        """Get the command to start a worker on Linux.
        
        Args:
            config: Worker configuration dictionary
            
        Returns:
            str: Command to start a worker
        """
        cmd = [
            "python3", 
            "run_worker_client.py",
            "--coordinator", config.get("coordinator_url", "http://localhost:8080"),
            "--api-key", config.get("api_key", "default_key")
        ]
        
        if "worker_id" in config:
            cmd.extend(["--worker-id", config["worker_id"]])
            
        if config.get("log_to_file"):
            cmd.extend(["--log-file", f"worker_{config.get('worker_id', 'unknown')}.log"])
            
        return " ".join(cmd)
    
    def create_deployment_script(self, config, output_path):
        """Create a Linux deployment script.
        
        Args:
            config: Worker configuration dictionary
            output_path: Path to save the deployment script
            
        Returns:
            str: Path to the created script
        """
        script_content = """#!/bin/bash
# Worker deployment script for Linux

# Configuration
COORDINATOR_URL="{coordinator_url}"
API_KEY="{api_key}"
WORKER_ID="{worker_id}"
LOG_FILE="worker_$WORKER_ID.log"

# Ensure dependencies are installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is required but not installed. Attempting to install..."
    apt-get update && apt-get install -y python3 python3-pip || \\
    yum install -y python3 python3-pip || \\
    dnf install -y python3 python3-pip
fi

# Install Python dependencies
pip3 install -r requirements.txt

# Start the worker
python3 run_worker_client.py \\
    --coordinator "$COORDINATOR_URL" \\
    --api-key "$API_KEY" \\
    --worker-id "$WORKER_ID" \\
    --log-file "$LOG_FILE"
""".format(
            coordinator_url=config.get("coordinator_url", "http://localhost:8080"),
            api_key=config.get("api_key", "default_key"),
            worker_id=config.get("worker_id", "worker_" + str(uuid.uuid4())[:8])
        )
        
        with open(output_path, "w") as f:
            f.write(script_content)
            
        os.chmod(output_path, 0o755)  # Make executable
        return output_path
    
    def install_dependencies(self, dependencies=None):
        """Install dependencies on Linux.
        
        Args:
            dependencies: List of dependencies to install
            
        Returns:
            bool: True if successful, False otherwise
        """
        if dependencies is None:
            dependencies = ["websockets", "psutil", "pyjwt", "duckdb", "numpy"]
            
        try:
            subprocess.run([sys.executable, "-m", "pip", "install"] + dependencies, check=True)
            logger.info(f"Installed dependencies: {', '.join(dependencies)}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install dependencies: {e}")
            return False
    
    def detect_hardware(self):
        """Detect hardware capabilities on Linux.
        
        Returns:
            Dict: Hardware capabilities
        """
        hardware_info = {
            "platform": "linux",
            "cpu": self._detect_linux_cpu(),
            "memory": self._detect_linux_memory(),
            "gpu": self._detect_linux_gpu(),
            "disk": self._detect_linux_disk()
        }
        return hardware_info
    
    def _detect_linux_cpu(self):
        """Detect CPU information on Linux.
        
        Returns:
            Dict: CPU information
        """
        cpu_info = {
            "cores": os.cpu_count(),
            "model": "Unknown"
        }
        
        try:
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if line.startswith("model name"):
                        cpu_info["model"] = line.split(":", 1)[1].strip()
                        break
        except Exception as e:
            logger.warning(f"Failed to get detailed CPU info: {e}")
            
        return cpu_info
    
    def _detect_linux_memory(self):
        """Detect memory information on Linux.
        
        Returns:
            Dict: Memory information
        """
        memory_info = {
            "total_gb": 0
        }
        
        try:
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if line.startswith("MemTotal"):
                        mem_kb = int(line.split()[1])
                        memory_info["total_gb"] = round(mem_kb / (1024 * 1024), 2)
                        break
        except Exception as e:
            logger.warning(f"Failed to get memory info: {e}")
            
        return memory_info
    
    def _detect_linux_gpu(self):
        """Detect GPU information on Linux.
        
        Returns:
            Dict: GPU information
        """
        gpu_info = {
            "count": 0,
            "devices": []
        }
        
        # Try nvidia-smi for NVIDIA GPUs
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False
            )
            
            if result.returncode == 0:
                lines = result.stdout.strip().split("\n")
                for i, line in enumerate(lines):
                    if not line.strip():
                        continue
                    parts = line.split(",")
                    if len(parts) >= 2:
                        gpu_info["devices"].append({
                            "id": i,
                            "name": parts[0].strip(),
                            "memory": parts[1].strip(),
                            "type": "cuda"
                        })
                gpu_info["count"] = len(gpu_info["devices"])
        except Exception as e:
            logger.debug(f"Failed to detect NVIDIA GPUs: {e}")
            
        # Try rocm-smi for AMD GPUs
        if gpu_info["count"] == 0:
            try:
                result = subprocess.run(
                    ["rocm-smi", "--showproductname"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=False
                )
                
                if result.returncode == 0:
                    lines = result.stdout.strip().split("\n")
                    gpu_names = []
                    for line in lines:
                        if "GPU[" in line and ":" in line:
                            name = line.split(":", 1)[1].strip()
                            gpu_names.append(name)
                    
                    for i, name in enumerate(gpu_names):
                        gpu_info["devices"].append({
                            "id": i,
                            "name": name,
                            "type": "rocm"
                        })
                    gpu_info["count"] = len(gpu_info["devices"])
            except Exception as e:
                logger.debug(f"Failed to detect AMD GPUs: {e}")
                
        return gpu_info
    
    def _detect_linux_disk(self):
        """Detect disk information on Linux.
        
        Returns:
            Dict: Disk information
        """
        disk_info = {
            "total_gb": 0,
            "free_gb": 0
        }
        
        try:
            disk_usage = shutil.disk_usage(".")
            disk_info["total_gb"] = round(disk_usage.total / (1024**3), 2)
            disk_info["free_gb"] = round(disk_usage.free / (1024**3), 2)
        except Exception as e:
            logger.warning(f"Failed to get disk info: {e}")
            
        return disk_info
    
    def get_startup_script(self, coordinator_url, api_key, worker_id):
        """Generate a Linux worker startup script.
        
        Args:
            coordinator_url: URL of the coordinator
            api_key: API key for authentication
            worker_id: Worker ID
            
        Returns:
            str: Worker startup script content
        """
        script = f"""#!/bin/bash
# Auto-generated worker startup script for Linux

# Set up environment variables
export WORKER_ID="{worker_id}"
export COORDINATOR_URL="{coordinator_url}"
export API_KEY="{api_key}"

# Create log directory
mkdir -p logs

# Start worker in the background with proper error handling
nohup python3 run_worker_client.py \\
    --coordinator "$COORDINATOR_URL" \\
    --api-key "$API_KEY" \\
    --worker-id "$WORKER_ID" \\
    --log-file "logs/$WORKER_ID.log" \\
    > "logs/$WORKER_ID.out" 2> "logs/$WORKER_ID.err" &

# Save PID for management
echo $! > "logs/$WORKER_ID.pid"

echo "Worker $WORKER_ID started, connecting to $COORDINATOR_URL"
echo "Worker PID: $!"
echo "Logs available in logs/$WORKER_ID.log"
"""
        return script
    
    def convert_path(self, path):
        """Convert a path to the correct format for Linux.
        
        Args:
            path: Path to convert
            
        Returns:
            str: Converted path
        """
        # Linux paths are the standard
        return str(Path(path))
    
    def get_platform_info(self):
        """Get Linux-specific information.
        
        Returns:
            Dict: Platform information
        """
        info = {}
        
        # Get Linux distribution
        try:
            dist_info = platform.freedesktop_os_release()
            info["distribution"] = dist_info.get("PRETTY_NAME", "Unknown")
        except:
            try:
                # Fallback for older Python versions
                import distro
                info["distribution"] = distro.name(pretty=True)
            except:
                info["distribution"] = "Unknown Linux"
        
        # Get kernel version
        info["kernel"] = platform.release()
        
        return info


class WindowsPlatformHandler(PlatformHandler):
    """Windows-specific platform handler."""
    
    def get_worker_command(self, config):
        """Get the command to start a worker on Windows.
        
        Args:
            config: Worker configuration dictionary
            
        Returns:
            str: Command to start a worker
        """
        cmd = [
            "python", 
            "run_worker_client.py",
            "--coordinator", config.get("coordinator_url", "http://localhost:8080"),
            "--api-key", config.get("api_key", "default_key")
        ]
        
        if "worker_id" in config:
            cmd.extend(["--worker-id", config["worker_id"]])
            
        if config.get("log_to_file"):
            cmd.extend(["--log-file", f"worker_{config.get('worker_id', 'unknown')}.log"])
            
        return " ".join(cmd)
    
    def create_deployment_script(self, config, output_path):
        """Create a Windows deployment script.
        
        Args:
            config: Worker configuration dictionary
            output_path: Path to save the deployment script
            
        Returns:
            str: Path to the created script
        """
        script_content = """@echo off
:: Worker deployment script for Windows

:: Configuration
set COORDINATOR_URL={coordinator_url}
set API_KEY={api_key}
set WORKER_ID={worker_id}
set LOG_FILE=worker_%WORKER_ID%.log

:: Ensure dependencies are installed
python -m pip install -r requirements.txt

:: Start the worker
python run_worker_client.py ^
    --coordinator "%COORDINATOR_URL%" ^
    --api-key "%API_KEY%" ^
    --worker-id "%WORKER_ID%" ^
    --log-file "%LOG_FILE%"
""".format(
            coordinator_url=config.get("coordinator_url", "http://localhost:8080"),
            api_key=config.get("api_key", "default_key"),
            worker_id=config.get("worker_id", "worker_" + str(uuid.uuid4())[:8])
        )
        
        # Use .bat extension for Windows
        if not output_path.endswith(".bat"):
            output_path = output_path + ".bat"
            
        with open(output_path, "w") as f:
            f.write(script_content)
            
        return output_path
    
    def install_dependencies(self, dependencies=None):
        """Install dependencies on Windows.
        
        Args:
            dependencies: List of dependencies to install
            
        Returns:
            bool: True if successful, False otherwise
        """
        if dependencies is None:
            dependencies = ["websockets", "psutil", "pyjwt", "duckdb", "numpy"]
            
        try:
            subprocess.run([sys.executable, "-m", "pip", "install"] + dependencies, check=True)
            logger.info(f"Installed dependencies: {', '.join(dependencies)}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install dependencies: {e}")
            return False
    
    def detect_hardware(self):
        """Detect hardware capabilities on Windows.
        
        Returns:
            Dict: Hardware capabilities
        """
        hardware_info = {
            "platform": "windows",
            "cpu": self._detect_windows_cpu(),
            "memory": self._detect_windows_memory(),
            "gpu": self._detect_windows_gpu(),
            "disk": self._detect_windows_disk()
        }
        return hardware_info
    
    def _detect_windows_cpu(self):
        """Detect CPU information on Windows.
        
        Returns:
            Dict: CPU information
        """
        cpu_info = {
            "cores": os.cpu_count(),
            "model": "Unknown"
        }
        
        try:
            import wmi
            c = wmi.WMI()
            for processor in c.Win32_Processor():
                cpu_info["model"] = processor.Name
                break
        except:
            # Fallback if WMI isn't available
            try:
                import subprocess
                result = subprocess.run("wmic cpu get name", shell=True, capture_output=True, text=True)
                lines = result.stdout.strip().split('\n')
                if len(lines) >= 2:
                    cpu_info["model"] = lines[1].strip()
            except:
                logger.warning("Failed to get detailed CPU info on Windows")
            
        return cpu_info
    
    def _detect_windows_memory(self):
        """Detect memory information on Windows.
        
        Returns:
            Dict: Memory information
        """
        memory_info = {
            "total_gb": 0
        }
        
        try:
            import psutil
            memory_info["total_gb"] = round(psutil.virtual_memory().total / (1024**3), 2)
        except ImportError:
            logger.warning("psutil not available for memory detection on Windows")
            
        return memory_info
    
    def _detect_windows_gpu(self):
        """Detect GPU information on Windows.
        
        Returns:
            Dict: GPU information
        """
        gpu_info = {
            "count": 0,
            "devices": []
        }
        
        # Try nvidia-smi for NVIDIA GPUs
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False
            )
            
            if result.returncode == 0:
                lines = result.stdout.strip().split("\n")
                for i, line in enumerate(lines):
                    if not line.strip():
                        continue
                    parts = line.split(",")
                    if len(parts) >= 2:
                        gpu_info["devices"].append({
                            "id": i,
                            "name": parts[0].strip(),
                            "memory": parts[1].strip(),
                            "type": "cuda"
                        })
                gpu_info["count"] = len(gpu_info["devices"])
        except Exception as e:
            logger.debug(f"Failed to detect NVIDIA GPUs: {e}")
            
        # If no NVIDIA GPUs, try to get generic GPU info
        if gpu_info["count"] == 0:
            try:
                import wmi
                c = wmi.WMI()
                for gpu in c.Win32_VideoController():
                    gpu_info["devices"].append({
                        "id": len(gpu_info["devices"]),
                        "name": gpu.Name,
                        "type": "generic"
                    })
                gpu_info["count"] = len(gpu_info["devices"])
            except Exception as e:
                logger.debug(f"Failed to detect generic GPUs: {e}")
                
        return gpu_info
    
    def _detect_windows_disk(self):
        """Detect disk information on Windows.
        
        Returns:
            Dict: Disk information
        """
        disk_info = {
            "total_gb": 0,
            "free_gb": 0
        }
        
        try:
            disk_usage = shutil.disk_usage(".")
            disk_info["total_gb"] = round(disk_usage.total / (1024**3), 2)
            disk_info["free_gb"] = round(disk_usage.free / (1024**3), 2)
        except Exception as e:
            logger.warning(f"Failed to get disk info: {e}")
            
        return disk_info
    
    def get_startup_script(self, coordinator_url, api_key, worker_id):
        """Generate a Windows worker startup script.
        
        Args:
            coordinator_url: URL of the coordinator
            api_key: API key for authentication
            worker_id: Worker ID
            
        Returns:
            str: Worker startup script content
        """
        script = f"""@echo off
:: Auto-generated worker startup script for Windows

:: Set environment variables
set WORKER_ID={worker_id}
set COORDINATOR_URL={coordinator_url}
set API_KEY={api_key}

:: Create log directory
if not exist logs mkdir logs

:: Start worker in the background
start /b python run_worker_client.py ^
    --coordinator "%COORDINATOR_URL%" ^
    --api-key "%API_KEY%" ^
    --worker-id "%WORKER_ID%" ^
    --log-file "logs\\%WORKER_ID%.log" > "logs\\%WORKER_ID%.out" 2> "logs\\%WORKER_ID%.err"

:: Save PID (not standard on Windows, but attempt it)
for /f "tokens=2" %%a in ('tasklist /fi "imagename eq python.exe" /fo list ^| find "PID:"') do (
    echo %%a > "logs\\%WORKER_ID%.pid"
    echo Worker started with PID: %%a
)

echo Worker %WORKER_ID% started, connecting to %COORDINATOR_URL%
echo Logs available in logs\\%WORKER_ID%.log
"""
        return script
    
    def convert_path(self, path):
        """Convert a path to the correct format for Windows.
        
        Args:
            path: Path to convert
            
        Returns:
            str: Converted path
        """
        # Convert forward slashes to backslashes
        return str(Path(path)).replace("/", "\\")
    
    def get_platform_info(self):
        """Get Windows-specific information.
        
        Returns:
            Dict: Platform information
        """
        info = {}
        
        # Get Windows version
        info["version"] = platform.version()
        info["edition"] = platform.win32_edition() if hasattr(platform, "win32_edition") else "Unknown"
        info["windows_version"] = platform.win32_ver()
        
        return info


class MacOSPlatformHandler(PlatformHandler):
    """macOS-specific platform handler."""
    
    def get_worker_command(self, config):
        """Get the command to start a worker on macOS.
        
        Args:
            config: Worker configuration dictionary
            
        Returns:
            str: Command to start a worker
        """
        cmd = [
            "python3", 
            "run_worker_client.py",
            "--coordinator", config.get("coordinator_url", "http://localhost:8080"),
            "--api-key", config.get("api_key", "default_key")
        ]
        
        if "worker_id" in config:
            cmd.extend(["--worker-id", config["worker_id"]])
            
        if config.get("log_to_file"):
            cmd.extend(["--log-file", f"worker_{config.get('worker_id', 'unknown')}.log"])
            
        return " ".join(cmd)
    
    def create_deployment_script(self, config, output_path):
        """Create a macOS deployment script.
        
        Args:
            config: Worker configuration dictionary
            output_path: Path to save the deployment script
            
        Returns:
            str: Path to the created script
        """
        script_content = """#!/bin/bash
# Worker deployment script for macOS

# Configuration
COORDINATOR_URL="{coordinator_url}"
API_KEY="{api_key}"
WORKER_ID="{worker_id}"
LOG_FILE="worker_$WORKER_ID.log"

# Ensure dependencies are installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is required but not installed. Please install it from python.org or using Homebrew."
    exit 1
fi

# Install Python dependencies
pip3 install -r requirements.txt

# Start the worker
python3 run_worker_client.py \\
    --coordinator "$COORDINATOR_URL" \\
    --api-key "$API_KEY" \\
    --worker-id "$WORKER_ID" \\
    --log-file "$LOG_FILE"
""".format(
            coordinator_url=config.get("coordinator_url", "http://localhost:8080"),
            api_key=config.get("api_key", "default_key"),
            worker_id=config.get("worker_id", "worker_" + str(uuid.uuid4())[:8])
        )
        
        with open(output_path, "w") as f:
            f.write(script_content)
            
        os.chmod(output_path, 0o755)  # Make executable
        return output_path
    
    def install_dependencies(self, dependencies=None):
        """Install dependencies on macOS.
        
        Args:
            dependencies: List of dependencies to install
            
        Returns:
            bool: True if successful, False otherwise
        """
        if dependencies is None:
            dependencies = ["websockets", "psutil", "pyjwt", "duckdb", "numpy"]
            
        try:
            subprocess.run([sys.executable, "-m", "pip", "install"] + dependencies, check=True)
            logger.info(f"Installed dependencies: {', '.join(dependencies)}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install dependencies: {e}")
            return False
    
    def detect_hardware(self):
        """Detect hardware capabilities on macOS.
        
        Returns:
            Dict: Hardware capabilities
        """
        hardware_info = {
            "platform": "darwin",
            "cpu": self._detect_macos_cpu(),
            "memory": self._detect_macos_memory(),
            "gpu": self._detect_macos_gpu(),
            "disk": self._detect_macos_disk()
        }
        return hardware_info
    
    def _detect_macos_cpu(self):
        """Detect CPU information on macOS.
        
        Returns:
            Dict: CPU information
        """
        cpu_info = {
            "cores": os.cpu_count(),
            "model": "Unknown"
        }
        
        try:
            # Get CPU model
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False
            )
            
            if result.returncode == 0:
                cpu_info["model"] = result.stdout.strip()
                
            # Check for Apple Silicon
            result = subprocess.run(
                ["sysctl", "-n", "hw.optional.arm64"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False
            )
            
            if result.returncode == 0 and result.stdout.strip() == "1":
                cpu_info["architecture"] = "arm64"
                cpu_info["apple_silicon"] = True
            else:
                cpu_info["architecture"] = "x86_64"
                cpu_info["apple_silicon"] = False
                
        except Exception as e:
            logger.warning(f"Failed to get detailed CPU info: {e}")
            
        return cpu_info
    
    def _detect_macos_memory(self):
        """Detect memory information on macOS.
        
        Returns:
            Dict: Memory information
        """
        memory_info = {
            "total_gb": 0
        }
        
        try:
            # Get memory size
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False
            )
            
            if result.returncode == 0:
                memory_bytes = int(result.stdout.strip())
                memory_info["total_gb"] = round(memory_bytes / (1024**3), 2)
        except Exception as e:
            logger.warning(f"Failed to get memory info: {e}")
            
        return memory_info
    
    def _detect_macos_gpu(self):
        """Detect GPU information on macOS.
        
        Returns:
            Dict: GPU information
        """
        gpu_info = {
            "count": 0,
            "devices": []
        }
        
        try:
            # Get GPU information using system_profiler
            result = subprocess.run(
                ["system_profiler", "SPDisplaysDataType"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False
            )
            
            if result.returncode == 0:
                lines = result.stdout.strip().split("\n")
                current_gpu = None
                
                for line in lines:
                    line = line.strip()
                    
                    if "Chipset Model:" in line:
                        if current_gpu:
                            gpu_info["devices"].append(current_gpu)
                            
                        current_gpu = {
                            "id": len(gpu_info["devices"]),
                            "name": line.split(":", 1)[1].strip(),
                            "type": "generic"
                        }
                        
                        # Check for known GPU types
                        name_lower = current_gpu["name"].lower()
                        if "nvidia" in name_lower:
                            current_gpu["type"] = "cuda"
                        elif "amd" in name_lower or "radeon" in name_lower:
                            current_gpu["type"] = "metal"
                        elif "intel" in name_lower:
                            current_gpu["type"] = "metal"
                        elif "apple" in name_lower:
                            current_gpu["type"] = "metal"
                            current_gpu["apple_silicon"] = True
                    
                    elif current_gpu and "VRAM" in line:
                        memory_str = line.split(":", 1)[1].strip()
                        current_gpu["memory"] = memory_str
                
                if current_gpu:
                    gpu_info["devices"].append(current_gpu)
                    
                gpu_info["count"] = len(gpu_info["devices"])
        except Exception as e:
            logger.warning(f"Failed to get GPU info: {e}")
            
        return gpu_info
    
    def _detect_macos_disk(self):
        """Detect disk information on macOS.
        
        Returns:
            Dict: Disk information
        """
        disk_info = {
            "total_gb": 0,
            "free_gb": 0
        }
        
        try:
            disk_usage = shutil.disk_usage(".")
            disk_info["total_gb"] = round(disk_usage.total / (1024**3), 2)
            disk_info["free_gb"] = round(disk_usage.free / (1024**3), 2)
        except Exception as e:
            logger.warning(f"Failed to get disk info: {e}")
            
        return disk_info
    
    def get_startup_script(self, coordinator_url, api_key, worker_id):
        """Generate a macOS worker startup script.
        
        Args:
            coordinator_url: URL of the coordinator
            api_key: API key for authentication
            worker_id: Worker ID
            
        Returns:
            str: Worker startup script content
        """
        script = f"""#!/bin/bash
# Auto-generated worker startup script for macOS

# Set up environment variables
export WORKER_ID="{worker_id}"
export COORDINATOR_URL="{coordinator_url}"
export API_KEY="{api_key}"

# Create log directory
mkdir -p logs

# Start worker in the background
nohup python3 run_worker_client.py \\
    --coordinator "$COORDINATOR_URL" \\
    --api-key "$API_KEY" \\
    --worker-id "$WORKER_ID" \\
    --log-file "logs/$WORKER_ID.log" \\
    > "logs/$WORKER_ID.out" 2> "logs/$WORKER_ID.err" &

# Save PID for management
echo $! > "logs/$WORKER_ID.pid"

echo "Worker $WORKER_ID started, connecting to $COORDINATOR_URL"
echo "Worker PID: $!"
echo "Logs available in logs/$WORKER_ID.log"
"""
        return script
    
    def convert_path(self, path):
        """Convert a path to the correct format for macOS.
        
        Args:
            path: Path to convert
            
        Returns:
            str: Converted path
        """
        # macOS paths use forward slashes like Linux
        return str(Path(path))
    
    def get_platform_info(self):
        """Get macOS-specific information.
        
        Returns:
            Dict: Platform information
        """
        info = {}
        
        # Get macOS version
        try:
            result = subprocess.run(
                ["sw_vers"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False
            )
            
            if result.returncode == 0:
                lines = result.stdout.strip().split("\n")
                for line in lines:
                    if "ProductVersion" in line:
                        info["version"] = line.split(":", 1)[1].strip()
                    elif "BuildVersion" in line:
                        info["build"] = line.split(":", 1)[1].strip()
        except Exception as e:
            logger.warning(f"Failed to get detailed macOS info: {e}")
            
        return info


class ContainerPlatformHandler(PlatformHandler):
    """Container-specific platform handler (Docker, Kubernetes, etc.)."""
    
    def get_worker_command(self, config):
        """Get the command to start a worker in a container.
        
        Args:
            config: Worker configuration dictionary
            
        Returns:
            str: Command to start a worker
        """
        cmd = [
            "python", 
            "run_worker_client.py",
            "--coordinator", config.get("coordinator_url", "http://coordinator:8080"),
            "--api-key", config.get("api_key", "default_key")
        ]
        
        if "worker_id" in config:
            cmd.extend(["--worker-id", config["worker_id"]])
            
        # Container-specific settings
        cmd.extend(["--container-mode"])
        
        if config.get("log_to_file"):
            cmd.extend(["--log-file", "/logs/worker.log"])
            
        return " ".join(cmd)
    
    def create_deployment_script(self, config, output_path):
        """Create a container deployment script (Docker Compose).
        
        Args:
            config: Worker configuration dictionary
            output_path: Path to save the deployment script
            
        Returns:
            str: Path to the created script
        """
        compose_content = """version: '3'

services:
  worker:
    build:
      context: .
      dockerfile: Dockerfile.worker
    environment:
      - COORDINATOR_URL={coordinator_url}
      - API_KEY={api_key}
      - WORKER_ID={worker_id}
      - CONTAINER_ENV=1
    volumes:
      - ./logs:/app/logs
    restart: unless-stopped
    networks:
      - testing_network

networks:
  testing_network:
    driver: bridge
""".format(
            coordinator_url=config.get("coordinator_url", "http://coordinator:8080"),
            api_key=config.get("api_key", "default_key"),
            worker_id=config.get("worker_id", "worker_" + str(uuid.uuid4())[:8])
        )
        
        # Create docker-compose.yml
        with open(output_path, "w") as f:
            f.write(compose_content)
            
        # Create Dockerfile.worker
        dockerfile_path = os.path.join(os.path.dirname(output_path), "Dockerfile.worker")
        dockerfile_content = """FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy worker code
COPY . .

# Create log directory
RUN mkdir -p logs

# Entry point
CMD ["python", "run_worker_client.py", "--coordinator", "${COORDINATOR_URL}", "--api-key", "${API_KEY}", "--worker-id", "${WORKER_ID}", "--log-file", "/app/logs/worker.log", "--container-mode"]
"""
        
        with open(dockerfile_path, "w") as f:
            f.write(dockerfile_content)
            
        return output_path
    
    def install_dependencies(self, dependencies=None):
        """Install dependencies in a container.
        
        Args:
            dependencies: List of dependencies to install
            
        Returns:
            bool: True if successful, False otherwise
        """
        if dependencies is None:
            dependencies = ["websockets", "psutil", "pyjwt", "duckdb", "numpy"]
            
        try:
            subprocess.run([sys.executable, "-m", "pip", "install"] + dependencies, check=True)
            logger.info(f"Installed dependencies: {', '.join(dependencies)}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install dependencies: {e}")
            return False
    
    def detect_hardware(self):
        """Detect hardware capabilities in a container.
        
        Returns:
            Dict: Hardware capabilities
        """
        # Detect underlying system (likely Linux)
        linux_handler = LinuxPlatformHandler()
        hardware_info = linux_handler.detect_hardware()
        
        # Add container-specific information
        hardware_info["platform"] = "container"
        hardware_info["container"] = self._detect_container_info()
        
        return hardware_info
    
    def _detect_container_info(self):
        """Detect container-specific information.
        
        Returns:
            Dict: Container information
        """
        container_info = {
            "type": "unknown"
        }
        
        # Check for Docker
        if os.path.exists("/.dockerenv"):
            container_info["type"] = "docker"
            
            # Get container ID
            try:
                with open("/proc/self/cgroup", "r") as f:
                    for line in f:
                        if "docker" in line:
                            container_info["id"] = line.split("/")[-1].strip()
                            break
            except:
                pass
        
        # Check for Kubernetes
        if os.environ.get("KUBERNETES_SERVICE_HOST"):
            container_info["type"] = "kubernetes"
            container_info["pod_name"] = os.environ.get("HOSTNAME")
            container_info["namespace"] = os.environ.get("KUBERNETES_NAMESPACE")
            
        # Check resource limits
        try:
            # CPU limits (if cgroups v1)
            cpu_quota_path = "/sys/fs/cgroup/cpu/cpu.cfs_quota_us"
            cpu_period_path = "/sys/fs/cgroup/cpu/cpu.cfs_period_us"
            
            if os.path.exists(cpu_quota_path) and os.path.exists(cpu_period_path):
                with open(cpu_quota_path, "r") as f:
                    cpu_quota = int(f.read().strip())
                with open(cpu_period_path, "r") as f:
                    cpu_period = int(f.read().strip())
                
                if cpu_quota > 0:
                    container_info["cpu_limit"] = cpu_quota / cpu_period
            
            # Memory limits
            memory_limit_path = "/sys/fs/cgroup/memory/memory.limit_in_bytes"
            if os.path.exists(memory_limit_path):
                with open(memory_limit_path, "r") as f:
                    memory_limit = int(f.read().strip())
                container_info["memory_limit_gb"] = round(memory_limit / (1024**3), 2)
        except:
            pass
            
        return container_info
    
    def get_startup_script(self, coordinator_url, api_key, worker_id):
        """Generate a container worker startup script.
        
        Args:
            coordinator_url: URL of the coordinator
            api_key: API key for authentication
            worker_id: Worker ID
            
        Returns:
            str: Worker startup script content
        """
        script = f"""#!/bin/bash
# Auto-generated worker startup script for containers

# Set up environment variables
export WORKER_ID="{worker_id}"
export COORDINATOR_URL="{coordinator_url}"
export API_KEY="{api_key}"
export CONTAINER_ENV=1

# Create log directory
mkdir -p /app/logs

# Start worker (in containers, we typically run in foreground)
python run_worker_client.py \\
    --coordinator "$COORDINATOR_URL" \\
    --api-key "$API_KEY" \\
    --worker-id "$WORKER_ID" \\
    --log-file "/app/logs/$WORKER_ID.log" \\
    --container-mode
"""
        return script
    
    def convert_path(self, path):
        """Convert a path to the correct format for containers.
        
        Args:
            path: Path to convert
            
        Returns:
            str: Converted path
        """
        # Container paths use Linux conventions
        return str(Path(path))
    
    def get_platform_info(self):
        """Get container-specific information.
        
        Returns:
            Dict: Platform information
        """
        info = {
            "container_type": "unknown"
        }
        
        # Check for Docker
        if os.path.exists("/.dockerenv"):
            info["container_type"] = "docker"
        
        # Check for Kubernetes
        if os.environ.get("KUBERNETES_SERVICE_HOST"):
            info["container_type"] = "kubernetes"
            info["pod_name"] = os.environ.get("HOSTNAME")
            info["namespace"] = os.environ.get("KUBERNETES_NAMESPACE")
            
        return info


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Cross-Platform Worker Support")
    parser.add_argument("--detect-platform", action="store_true", help="Detect platform")
    parser.add_argument("--detect-hardware", action="store_true", help="Detect hardware")
    parser.add_argument("--create-script", action="store_true", help="Create deployment script")
    parser.add_argument("--coordinator-url", default="http://localhost:8080", help="Coordinator URL")
    parser.add_argument("--api-key", default="test_key", help="API key")
    parser.add_argument("--worker-id", help="Worker ID")
    parser.add_argument("--output", help="Output file path")
    
    args = parser.parse_args()
    
    # Initialize cross-platform support
    platform_support = CrossPlatformWorkerSupport()
    
    if args.detect_platform:
        platform_info = platform_support.get_platform_info()
        print(f"Detected platform: {platform_support.current_platform}")
        print(json.dumps(platform_info, indent=2))
        
    if args.detect_hardware:
        hardware_info = platform_support.detect_hardware()
        print(f"Hardware information:")
        print(json.dumps(hardware_info, indent=2))
        
    if args.create_script:
        if not args.output:
            parser.error("--output is required when using --create-script")
            
        config = {
            "coordinator_url": args.coordinator_url,
            "api_key": args.api_key,
            "worker_id": args.worker_id
        }
        
        output_path = platform_support.create_deployment_script(config, args.output)
        print(f"Created deployment script at {output_path}")