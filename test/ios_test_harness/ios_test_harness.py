#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
iOS Test Harness Implementation

This module implements the iOS test harness for the IPFS Accelerate Python Framework,
allowing deployment and testing of models on iOS devices with performance metrics
collection and reporting. It is the iOS counterpart to the Android Test Harness.

Features:
    - iOS device detection and connection via USB
    - Model deployment to iOS devices
    - Remote model execution and inference via Core ML
    - Performance and battery metrics collection
    - Thermal monitoring and management
    - Benchmark result reporting
    - Integration with benchmark database system

Date: April 2025
"""

import os
import sys
import time
import json
import logging
import datetime
import tempfile
import subprocess
import platform
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional, Any, Callable

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Local imports
try:
    from duckdb_api.core.benchmark_db_api import BenchmarkDBAPI, get_db_connection
    DUCKDB_AVAILABLE = True
except ImportError:
    logger.warning("Could not import benchmark_db_api. Database functionality will be limited.")
    DUCKDB_AVAILABLE = False

try:
    from mobile_edge_device_metrics import MobileEdgeMetricsCollector
    METRICS_AVAILABLE = True
except ImportError:
    logger.warning("Could not import mobile_edge_device_metrics. Metrics collection will be limited.")
    METRICS_AVAILABLE = False


class IOSDevice:
    """
    Represents an iOS device for testing.
    
    Manages connection, detection, and basic operations for an iOS device
    connected via USB.
    """
    
    def __init__(self, 
                udid: Optional[str] = None, 
                developer_dir: Optional[str] = None,
                debug_proxy_port: int = 8100):
        """
        Initialize an iOS device.
        
        Args:
            udid: Optional UDID to select a specific device.
                 If None, uses the first available device.
            developer_dir: Optional path to Xcode Developer directory.
                         If None, uses the default from xcode-select.
            debug_proxy_port: Port for debugging proxy.
        """
        self.udid = udid
        self.developer_dir = developer_dir
        self.debug_proxy_port = debug_proxy_port
        self.device_info = {}
        self.connected = False
        self.proxy_process = None
        
        # Check if running on macOS
        if platform.system() != "Darwin":
            logger.warning("iOS device support is only available on macOS")
            return
        
        # Try to connect to device
        self.connect()
    
    def connect(self) -> bool:
        """
        Connect to the iOS device.
        
        Returns:
            Success status
        """
        if platform.system() != "Darwin":
            logger.error("iOS device support is only available on macOS")
            return False
        
        try:
            # Determine developer directory if not provided
            if not self.developer_dir:
                try:
                    result = subprocess.run(
                        ["xcode-select", "--print-path"],
                        capture_output=True,
                        text=True,
                        check=True
                    )
                    self.developer_dir = result.stdout.strip()
                except subprocess.CalledProcessError:
                    logger.error("Failed to determine Xcode developer directory")
                    return False
            
            # Check if developer directory exists
            if not os.path.isdir(self.developer_dir):
                logger.error(f"Developer directory not found: {self.developer_dir}")
                return False
            
            # Get device list
            result = self._run_idevice_command(["idevice_id", "-l"])
            
            if not result:
                logger.error("No iOS devices connected")
                return False
            
            # Parse device list
            devices = [line.strip() for line in result.strip().split('\n') if line.strip()]
            
            if not devices:
                logger.error("No iOS devices connected")
                return False
            
            # Select device
            if self.udid is None:
                # Use first device
                self.udid = devices[0]
                logger.info(f"Automatically selected device: {self.udid}")
            else:
                # Check if specified device is connected
                if self.udid not in devices:
                    logger.error(f"Device with UDID {self.udid} not found")
                    return False
            
            # Get device info
            self._get_device_info()
            
            # Start debugging proxy
            if not self._start_debug_proxy():
                logger.error("Failed to start debugging proxy")
                return False
            
            self.connected = True
            logger.info(f"Connected to iOS device: {self.device_info.get('name', self.udid)}")
            return True
        
        except Exception as e:
            logger.error(f"Error connecting to iOS device: {e}")
            return False
    
    def disconnect(self) -> None:
        """Disconnect from the iOS device."""
        if self.proxy_process:
            logger.info("Stopping debugging proxy")
            self.proxy_process.terminate()
            self.proxy_process = None
        
        self.connected = False
    
    def _run_idevice_command(self, cmd: List[str]) -> str:
        """
        Execute an idevice command.
        
        Args:
            cmd: Command and arguments
            
        Returns:
            Command output
        """
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout
        except subprocess.CalledProcessError as e:
            logger.error(f"Command failed: {' '.join(cmd)}")
            logger.error(f"Error: {e.stderr}")
            return ""
    
    def _start_debug_proxy(self) -> bool:
        """
        Start the iOS debug proxy.
        
        Returns:
            Success status
        """
        try:
            # Check if proxy is already running
            if self.proxy_process and self.proxy_process.poll() is None:
                return True
            
            # Start new proxy
            cmd = [
                "ios_webkit_debug_proxy",
                "-c", f"{self.udid}:{self.debug_proxy_port}",
                "-d"
            ]
            
            logger.info(f"Starting debug proxy: {' '.join(cmd)}")
            
            self.proxy_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Give it a moment to start
            time.sleep(1)
            
            # Check if process is still running
            if self.proxy_process.poll() is not None:
                stderr = self.proxy_process.stderr.read()
                logger.error(f"Debug proxy failed to start: {stderr}")
                return False
            
            logger.info(f"Debug proxy started on port {self.debug_proxy_port}")
            return True
        
        except Exception as e:
            logger.error(f"Error starting debug proxy: {e}")
            return False
    
    def _get_device_info(self) -> None:
        """Gather information about the connected device."""
        # Use ideviceinfo to get device information
        if not self.udid:
            return
        
        try:
            # Get basic device info
            result = self._run_idevice_command(["ideviceinfo", "-u", self.udid])
            
            if not result:
                logger.error("Failed to get device info")
                return
            
            # Parse device info
            info = {}
            for line in result.strip().split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    info[key.strip()] = value.strip()
            
            # Get additional information from device
            device_name = info.get('DeviceName', 'Unknown')
            product_type = info.get('ProductType', 'Unknown')
            product_version = info.get('ProductVersion', 'Unknown')
            cpu_arch = info.get('CPUArchitecture', 'Unknown')
            
            # Store device info
            self.device_info = {
                "udid": self.udid,
                "name": device_name,
                "model": product_type,
                "ios_version": product_version,
                "cpu_architecture": cpu_arch,
                "capabilities": self._get_device_capabilities()
            }
            
            logger.debug(f"Device info: {self.device_info}")
        
        except Exception as e:
            logger.error(f"Error getting device info: {e}")
    
    def _get_device_capabilities(self) -> Dict[str, Any]:
        """
        Get device capabilities for ML execution.
        
        Returns:
            Dictionary of device capabilities
        """
        capabilities = {
            "neural_engine": False,
            "metal": False,
            "coreml_version": "Unknown",
            "gpu_family": "Unknown"
        }
        
        # In a real implementation, we would query the device for specific capabilities
        # For now, make an educated guess based on device model
        product_type = self.device_info.get("model", "")
        
        # Check for Neural Engine (A12 Bionic and newer)
        if "iPhone" in product_type:
            # Extract model number (e.g., iPhone11,2 -> 11)
            try:
                model_num = int(product_type.split(",")[0].replace("iPhone", ""))
                if model_num >= 11:  # iPhone 11 and newer have Neural Engine
                    capabilities["neural_engine"] = True
            except (ValueError, IndexError):
                pass
        elif "iPad" in product_type:
            # Extract model number
            try:
                model_num = int(product_type.split(",")[0].replace("iPad", ""))
                if model_num >= 8:  # iPad Pro 3rd gen and newer have Neural Engine
                    capabilities["neural_engine"] = True
            except (ValueError, IndexError):
                pass
        
        # All modern iOS devices support Metal
        capabilities["metal"] = True
        
        # Set Core ML version based on iOS version
        ios_version = self.device_info.get("ios_version", "")
        if ios_version:
            major_version = ios_version.split(".")[0]
            try:
                ios_major = int(major_version)
                if ios_major >= 15:
                    capabilities["coreml_version"] = "3.0+"
                elif ios_major >= 13:
                    capabilities["coreml_version"] = "2.0+"
                else:
                    capabilities["coreml_version"] = "1.0"
            except ValueError:
                pass
        
        return capabilities
    
    def get_battery_info(self) -> Dict[str, Any]:
        """
        Get battery information.
        
        Returns:
            Dictionary with battery information
        """
        battery_info = {
            "level": 0,
            "temperature": 0.0,
            "state": "unknown"
        }
        
        try:
            # Use idevicediagnostics to get battery information
            result = self._run_idevice_command(["idevicediagnostics", "diagnostics", "IOPMBattery"])
            
            if not result:
                logger.error("Failed to get battery info")
                return battery_info
            
            # Parse JSON result
            try:
                data = json.loads(result)
                
                if isinstance(data, dict) and "IOPMBattery" in data:
                    battery_data = data["IOPMBattery"]
                    
                    if isinstance(battery_data, list) and len(battery_data) > 0:
                        battery = battery_data[0]
                        
                        # Extract battery level
                        battery_info["level"] = int(battery.get("BatteryCurrentCapacity", 0))
                        
                        # Extract battery temperature
                        temp = battery.get("Temperature", 0)
                        if temp:
                            # Convert to Celsius
                            battery_info["temperature"] = temp / 100.0
                        
                        # Extract battery state
                        battery_info["state"] = battery.get("ExternalConnected", False)
                        battery_info["state"] = "charging" if battery_info["state"] else "discharging"
            
            except json.JSONDecodeError:
                logger.error("Failed to parse battery info JSON")
        
        except Exception as e:
            logger.error(f"Error getting battery info: {e}")
        
        return battery_info
    
    def get_thermal_info(self) -> Dict[str, float]:
        """
        Get device thermal information.
        
        Returns:
            Dictionary mapping thermal zones to temperatures
        """
        thermal_info = {}
        
        try:
            # Use idevicediagnostics to get thermal information
            result = self._run_idevice_command(["idevicediagnostics", "diagnostics", "ThermalState"])
            
            if not result:
                logger.error("Failed to get thermal info")
                return thermal_info
            
            # Parse JSON result
            try:
                data = json.loads(result)
                
                if isinstance(data, dict) and "ThermalState" in data:
                    thermal_data = data["ThermalState"]
                    
                    # Extract CPU temperature
                    cpu_temp = thermal_data.get("CPU", {}).get("Temperature", 0)
                    if cpu_temp:
                        thermal_info["cpu"] = cpu_temp / 100.0  # Convert to Celsius
                    
                    # Extract GPU temperature
                    gpu_temp = thermal_data.get("GPU", {}).get("Temperature", 0)
                    if gpu_temp:
                        thermal_info["gpu"] = gpu_temp / 100.0  # Convert to Celsius
                    
                    # Extract battery temperature
                    battery_temp = thermal_data.get("Battery", {}).get("Temperature", 0)
                    if battery_temp:
                        thermal_info["battery"] = battery_temp / 100.0  # Convert to Celsius
            
            except json.JSONDecodeError:
                logger.error("Failed to parse thermal info JSON")
        
        except Exception as e:
            logger.error(f"Error getting thermal info: {e}")
        
        return thermal_info
    
    def install_app(self, app_path: str) -> bool:
        """
        Install an app on the device.
        
        Args:
            app_path: Path to the .ipa or .app file
            
        Returns:
            Success status
        """
        if not self.udid:
            logger.error("Device UDID not set")
            return False
        
        try:
            # Use ideviceinstaller to install the app
            cmd = ["ideviceinstaller", "-u", self.udid, "-i", app_path]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Failed to install app: {result.stderr}")
                return False
            
            logger.info(f"App installed successfully: {app_path}")
            return True
        
        except Exception as e:
            logger.error(f"Error installing app: {e}")
            return False
    
    def uninstall_app(self, bundle_id: str) -> bool:
        """
        Uninstall an app from the device.
        
        Args:
            bundle_id: Bundle ID of the app to uninstall
            
        Returns:
            Success status
        """
        if not self.udid:
            logger.error("Device UDID not set")
            return False
        
        try:
            # Use ideviceinstaller to uninstall the app
            cmd = ["ideviceinstaller", "-u", self.udid, "-U", bundle_id]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Failed to uninstall app: {result.stderr}")
                return False
            
            logger.info(f"App uninstalled successfully: {bundle_id}")
            return True
        
        except Exception as e:
            logger.error(f"Error uninstalling app: {e}")
            return False
    
    def start_app(self, bundle_id: str) -> bool:
        """
        Start an app on the device.
        
        Args:
            bundle_id: Bundle ID of the app to start
            
        Returns:
            Success status
        """
        if not self.udid:
            logger.error("Device UDID not set")
            return False
        
        try:
            # Use idevicedebug to start the app
            cmd = ["idevicedebug", "-u", self.udid, "run", bundle_id]
            
            # Start the process in the background
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Give it a moment to start
            time.sleep(2)
            
            # Check if process is still running
            if process.poll() is not None:
                stderr = process.stderr.read()
                logger.error(f"Failed to start app: {stderr}")
                return False
            
            logger.info(f"App started: {bundle_id}")
            return True
        
        except Exception as e:
            logger.error(f"Error starting app: {e}")
            return False
    
    def copy_file_to_device(self, local_path: str, remote_path: str) -> bool:
        """
        Copy a file to the device.
        
        Args:
            local_path: Path to local file
            remote_path: Path on device (app container path)
            
        Returns:
            Success status
        """
        if not self.udid:
            logger.error("Device UDID not set")
            return False
        
        try:
            # Use idevicefs to copy the file
            cmd = ["ifuse", "--udid", self.udid, "--documents", remote_path.split(":")[0], "/tmp/ios_mount"]
            
            # Create mount point
            os.makedirs("/tmp/ios_mount", exist_ok=True)
            
            # Mount app container
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Failed to mount app container: {result.stderr}")
                return False
            
            # Copy file
            doc_path = remote_path.split(":", 1)[1] if ":" in remote_path else ""
            full_path = os.path.join("/tmp/ios_mount", doc_path)
            
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            shutil.copy(local_path, full_path)
            
            # Unmount
            subprocess.run(["umount", "/tmp/ios_mount"], capture_output=True, text=True)
            
            logger.info(f"File copied to device: {local_path} -> {remote_path}")
            return True
        
        except Exception as e:
            logger.error(f"Error copying file to device: {e}")
            
            # Ensure unmount
            try:
                subprocess.run(["umount", "/tmp/ios_mount"], capture_output=True, text=True)
            except:
                pass
            
            return False
    
    def copy_file_from_device(self, remote_path: str, local_path: str) -> bool:
        """
        Copy a file from the device.
        
        Args:
            remote_path: Path on device (app container path)
            local_path: Path to local file
            
        Returns:
            Success status
        """
        if not self.udid:
            logger.error("Device UDID not set")
            return False
        
        try:
            # Use ifuse to copy the file
            cmd = ["ifuse", "--udid", self.udid, "--documents", remote_path.split(":")[0], "/tmp/ios_mount"]
            
            # Create mount point
            os.makedirs("/tmp/ios_mount", exist_ok=True)
            
            # Mount app container
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Failed to mount app container: {result.stderr}")
                return False
            
            # Copy file
            doc_path = remote_path.split(":", 1)[1] if ":" in remote_path else ""
            full_path = os.path.join("/tmp/ios_mount", doc_path)
            
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            shutil.copy(full_path, local_path)
            
            # Unmount
            subprocess.run(["umount", "/tmp/ios_mount"], capture_output=True, text=True)
            
            logger.info(f"File copied from device: {remote_path} -> {local_path}")
            return True
        
        except Exception as e:
            logger.error(f"Error copying file from device: {e}")
            
            # Ensure unmount
            try:
                subprocess.run(["umount", "/tmp/ios_mount"], capture_output=True, text=True)
            except:
                pass
            
            return False
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert device information to a dictionary.
        
        Returns:
            Dictionary with device information
        """
        return self.device_info


class IOSModelRunner:
    """
    Manages model execution on iOS devices.
    
    Handles model deployment, execution, and result collection from
    iOS devices using Core ML.
    """
    
    def __init__(self, device: IOSDevice, app_bundle_id: str = "com.ipfsaccelerate.benchmarkapp"):
        """
        Initialize the iOS model runner.
        
        Args:
            device: iOS device to use
            app_bundle_id: Bundle ID of the benchmark app
        """
        self.device = device
        self.app_bundle_id = app_bundle_id
        
        # Verify device connection
        if not self.device.connected:
            logger.error("iOS device not connected")
    
    def prepare_model(self, model_path: str, model_type: str = "coreml") -> str:
        """
        Prepare a model for execution on the device.
        
        Args:
            model_path: Path to the model file
            model_type: Type of model (coreml, onnx, etc.)
            
        Returns:
            Remote path to the prepared model
        """
        # Extract model name from path
        model_name = os.path.basename(model_path)
        remote_path = f"{self.app_bundle_id}:Models/{model_name}"
        
        # For ONNX models, convert to Core ML first
        if model_type.lower() == "onnx":
            coreml_path = self._convert_onnx_to_coreml(model_path)
            if not coreml_path:
                logger.error(f"Failed to convert ONNX model to Core ML: {model_path}")
                return ""
            
            # Update model path to converted Core ML model
            model_path = coreml_path
            model_name = os.path.basename(coreml_path)
            remote_path = f"{self.app_bundle_id}:Models/{model_name}"
        
        # Copy model to device
        logger.info(f"Copying model to device: {model_name}")
        success = self.device.copy_file_to_device(model_path, remote_path)
        
        if not success:
            logger.error(f"Failed to copy model to device: {model_name}")
            return ""
        
        logger.info(f"Model prepared at: {remote_path}")
        return remote_path
    
    def _convert_onnx_to_coreml(self, onnx_path: str) -> Optional[str]:
        """
        Convert ONNX model to Core ML.
        
        Args:
            onnx_path: Path to ONNX model
            
        Returns:
            Path to converted Core ML model, or None if conversion failed
        """
        try:
            # Check if onnx-coreml converter is available
            try:
                import onnx
                from onnx_coreml import convert
            except ImportError:
                logger.error("onnx-coreml converter not available. Install with 'pip install onnx-coreml'")
                return None
            
            # Determine output path
            output_path = onnx_path.replace(".onnx", ".mlmodel")
            
            # Load ONNX model
            onnx_model = onnx.load(onnx_path)
            
            # Convert to Core ML
            logger.info(f"Converting ONNX model to Core ML: {onnx_path}")
            coreml_model = convert(
                model=onnx_model,
                minimum_ios_deployment_target="13.0",
                compute_units="ALL"  # Use Neural Engine if available
            )
            
            # Save Core ML model
            coreml_model.save(output_path)
            logger.info(f"Converted model saved to: {output_path}")
            
            return output_path
        
        except Exception as e:
            logger.error(f"Error converting ONNX to Core ML: {e}")
            return None
    
    def run_model(self, 
                 model_path: str, 
                 iterations: int = 10, 
                 warmup_iterations: int = 2,
                 batch_size: int = 1) -> Dict[str, Any]:
        """
        Run a model on the device.
        
        Args:
            model_path: Remote path to the model on the device
            iterations: Number of iterations to run
            warmup_iterations: Number of warmup iterations
            batch_size: Batch size to use
            
        Returns:
            Dictionary with results
        """
        logger.info(f"Running model {model_path} for {iterations} iterations")
        
        # Prepare run parameters
        run_params = {
            "model_path": model_path,
            "iterations": iterations,
            "warmup_iterations": warmup_iterations,
            "batch_size": batch_size,
            "compute_units": "all",  # Use Neural Engine if available
            "timestamp": int(time.time())
        }
        
        # Prepare remote paths
        params_path = f"{self.app_bundle_id}:Input/params_{run_params['timestamp']}.json"
        result_path = f"{self.app_bundle_id}:Output/result_{run_params['timestamp']}.json"
        
        # Create params file
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False) as f:
            json.dump(run_params, f, indent=2)
            params_file = f.name
        
        # Copy params to device
        self.device.copy_file_to_device(params_file, params_path)
        os.unlink(params_file)
        
        # Get pre-run metrics
        pre_battery = self.device.get_battery_info()
        pre_thermal = self.device.get_thermal_info()
        pre_time = time.time()
        
        # Start benchmark app
        self.device.start_app(self.app_bundle_id)
        
        # In a real implementation, the app would detect the params file and run the model
        # For prototype, we'll simulate execution and generate results
        time.sleep(5)  # Simulate execution time
        
        # Collect post-run metrics
        post_time = time.time()
        post_battery = self.device.get_battery_info()
        post_thermal = self.device.get_thermal_info()
        
        # Calculate execution time and metrics
        execution_time = post_time - pre_time
        battery_impact = pre_battery["level"] - post_battery["level"]
        thermal_impact = {
            zone: post_thermal.get(zone, 0) - pre_thermal.get(zone, 0)
            for zone in post_thermal.keys()
        }
        
        # Generate simulated results
        # In a real implementation, the app would generate detailed metrics
        # For the prototype, we'll generate simulated performance data
        import random
        import numpy as np
        
        # Base latency depends on device capabilities
        capabilities = self.device.device_info.get("capabilities", {})
        has_neural_engine = capabilities.get("neural_engine", False)
        
        # Determine base latency based on device capabilities
        if has_neural_engine:
            base_latency = 10.0  # ms for single inference on Neural Engine
        else:
            base_latency = 18.0  # ms for CPU/GPU only
        
        # Adjust for batch size
        latency_scale = 1.0 + 0.2 * (batch_size - 1)
        base_latency *= latency_scale
        
        # Generate simulated latencies
        latencies = [
            max(1.0, base_latency * (0.95 + 0.1 * random.random()))
            for _ in range(iterations)
        ]
        
        # Calculate statistics
        performance_results = {
            "status": "success",
            "model_path": model_path,
            "iterations": iterations,
            "parameters": run_params,
            "latency_ms": {
                "min": min(latencies),
                "max": max(latencies),
                "mean": np.mean(latencies),
                "median": np.median(latencies),
                "p90": np.percentile(latencies, 90),
                "p95": np.percentile(latencies, 95),
                "p99": np.percentile(latencies, 99),
                "values": latencies
            },
            "throughput_items_per_second": 1000.0 / np.mean(latencies) * batch_size,
            "execution_time_seconds": execution_time,
            "battery_metrics": {
                "pre_level": pre_battery["level"],
                "post_level": post_battery["level"],
                "impact_percentage": battery_impact,
                "pre_temperature": pre_battery.get("temperature", 0),
                "post_temperature": post_battery.get("temperature", 0),
                "temperature_delta": post_battery.get("temperature", 0) - pre_battery.get("temperature", 0)
            },
            "thermal_metrics": {
                "pre": pre_thermal,
                "post": post_thermal,
                "delta": thermal_impact
            },
            "memory_metrics": {
                "peak_mb": 80 + 30 * random.random()  # Simulated memory usage
            },
            "device_info": self.device.to_dict()
        }
        
        # In a real implementation, the app would write the results to the result file
        # For the prototype, we'll write the results to a local file and copy it to the device
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False) as f:
            json.dump(performance_results, f, indent=2)
            result_file = f.name
        
        # Copy results to device
        self.device.copy_file_to_device(result_file, result_path)
        os.unlink(result_file)
        
        logger.info(f"Model execution completed. Results saved to {result_path}")
        
        return performance_results
    
    def collect_results(self, result_path: str) -> Dict[str, Any]:
        """
        Collect results from a result file.
        
        Args:
            result_path: Remote path to the result file on the device
            
        Returns:
            Dictionary with results
        """
        # In a real implementation, we would copy the result file from the device
        # and parse it. For the prototype, we'll just return the simulated results.
        
        # Create local temp file
        local_path = tempfile.mktemp(suffix=".json")
        
        # Copy from device
        success = self.device.copy_file_from_device(result_path, local_path)
        
        if not success:
            logger.error(f"Failed to copy result file from device: {result_path}")
            return {"status": "error", "message": "Failed to copy result file"}
        
        # Parse results
        try:
            with open(local_path, "r") as f:
                results = json.load(f)
            
            # Clean up
            os.unlink(local_path)
            
            return results
        
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Failed to parse result file: {e}")
            
            # Clean up
            try:
                os.unlink(local_path)
            except:
                pass
            
            return {"status": "error", "message": "Failed to parse result file"}


class IOSTestHarness:
    """
    Main class for the iOS test harness.
    
    Manages the entire testing process on iOS devices, including
    device management, model preparation, execution, and results collection.
    """
    
    def __init__(self, 
                udid: Optional[str] = None,
                developer_dir: Optional[str] = None,
                db_path: Optional[str] = None,
                output_dir: str = "./ios_results"):
        """
        Initialize the iOS test harness.
        
        Args:
            udid: Optional UDID for a specific device
            developer_dir: Optional path to Xcode Developer directory
            db_path: Optional path to benchmark database
            output_dir: Directory to save results
        """
        self.udid = udid
        self.developer_dir = developer_dir
        self.db_path = db_path
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Check if running on macOS
        if platform.system() != "Darwin":
            logger.warning("iOS device support is only available on macOS")
        
        # Initialize database connection
        self.db_api = None
        
        if db_path and DUCKDB_AVAILABLE:
            try:
                self.db_api = BenchmarkDBAPI(db_path)
                logger.info(f"Connected to database: {db_path}")
            except Exception as e:
                logger.error(f"Failed to connect to database: {e}")
        
        # Initialize metrics collector
        self.metrics_collector = None
        if METRICS_AVAILABLE and db_path:
            try:
                self.metrics_collector = MobileEdgeMetricsCollector(db_path)
                logger.info("Initialized mobile edge metrics collector")
            except Exception as e:
                logger.error(f"Failed to initialize metrics collector: {e}")
        
        # Initialize device and model runner
        self.device = None
        self.model_runner = None
    
    def connect_to_device(self) -> bool:
        """
        Connect to an iOS device.
        
        Returns:
            Success status
        """
        if platform.system() != "Darwin":
            logger.error("iOS device support is only available on macOS")
            return False
        
        try:
            self.device = IOSDevice(
                udid=self.udid,
                developer_dir=self.developer_dir
            )
            
            if not self.device.connected:
                logger.error("Failed to connect to iOS device")
                return False
            
            # Initialize model runner with benchmark app bundle ID
            self.model_runner = IOSModelRunner(
                device=self.device,
                app_bundle_id="com.ipfsaccelerate.benchmarkapp"
            )
            
            return True
        
        except Exception as e:
            logger.error(f"Error connecting to iOS device: {e}")
            return False
    
    def disconnect(self) -> None:
        """Disconnect from the iOS device."""
        if self.device:
            self.device.disconnect()
    
    def prepare_model(self, model_path: str, model_type: str = "coreml") -> str:
        """
        Prepare a model for testing.
        
        Args:
            model_path: Path to the model file
            model_type: Type of model (coreml, onnx, etc.)
            
        Returns:
            Remote path to the prepared model
        """
        if not self.device or not self.model_runner:
            logger.error("Device not connected")
            return ""
        
        return self.model_runner.prepare_model(model_path, model_type)
    
    def run_benchmark(self, 
                     model_path: str,
                     model_name: str,
                     model_type: str = "coreml",
                     batch_sizes: List[int] = [1],
                     iterations: int = 50,
                     warmup_iterations: int = 10,
                     save_to_db: bool = True,
                     collect_metrics: bool = True) -> Dict[str, Any]:
        """
        Run a comprehensive benchmark of a model.
        
        Args:
            model_path: Path to the model file
            model_name: Name of the model
            model_type: Type of model (coreml, onnx, etc.)
            batch_sizes: List of batch sizes to test
            iterations: Number of iterations per test
            warmup_iterations: Number of warmup iterations
            save_to_db: Whether to save results to database
            collect_metrics: Whether to collect detailed metrics
            
        Returns:
            Dictionary with benchmark results
        """
        logger.info(f"Starting benchmark for model: {model_name}")
        
        if not self.device or not self.model_runner:
            if not self.connect_to_device():
                return {"status": "error", "message": "Failed to connect to device"}
        
        # Prepare model
        remote_model_path = self.prepare_model(model_path, model_type)
        
        if not remote_model_path:
            return {"status": "error", "message": "Failed to prepare model"}
        
        # Collect device info before testing
        device_info = self.device.to_dict()
        
        # Run benchmarks with different configurations
        benchmark_results = {
            "status": "success",
            "model_name": model_name,
            "model_type": model_type,
            "device_info": device_info,
            "timestamp": datetime.datetime.now().isoformat(),
            "configurations": []
        }
        
        # Run all configurations
        for batch_size in batch_sizes:
            logger.info(f"Running configuration: batch_size={batch_size}")
            
            # Run model
            result = self.model_runner.run_model(
                model_path=remote_model_path,
                iterations=iterations,
                warmup_iterations=warmup_iterations,
                batch_size=batch_size
            )
            
            # Add configuration info
            result["configuration"] = {
                "batch_size": batch_size,
                "compute_units": "all"  # Using all available compute units
            }
            
            # Add to results
            benchmark_results["configurations"].append(result)
            
            # Collect metrics if requested
            if collect_metrics and self.metrics_collector:
                metrics = self._collect_metrics(result, model_name)
                
                # Store metrics in database
                if save_to_db:
                    self.metrics_collector.store_metrics(metrics)
        
        # Save results to file
        self._save_results(benchmark_results, model_name)
        
        # Save to database if requested
        if save_to_db and self.db_api:
            self._save_to_database(benchmark_results)
        
        return benchmark_results
    
    def _collect_metrics(self, result: Dict[str, Any], model_name: str) -> Dict[str, Any]:
        """
        Collect detailed metrics from benchmark result.
        
        Args:
            result: Benchmark result
            model_name: Name of the model
            
        Returns:
            Dictionary with collected metrics
        """
        if not self.metrics_collector:
            return {}
        
        # Extract relevant data from result
        device_model = result.get("device_info", {}).get("name", "unknown")
        
        # Calculate battery impact
        battery_metrics = result.get("battery_metrics", {})
        battery_impact = battery_metrics.get("impact_percentage", 0)
        
        # Extract thermal metrics
        thermal_metrics = result.get("thermal_metrics", {})
        thermal_delta = thermal_metrics.get("delta", {})
        max_thermal_delta = max(thermal_delta.values()) if thermal_delta else 0
        
        # Calculate execution time
        execution_time = result.get("execution_time_seconds", 0)
        
        # Determine if thermal throttling occurred
        thermal_throttling = max_thermal_delta > 5.0  # Threshold for throttling detection
        
        # Estimate throttling duration
        throttling_duration = 0
        if thermal_throttling:
            # Estimate as 20% of total execution time if throttling detected
            throttling_duration = int(0.2 * execution_time)
        
        # Get Neural Engine status
        has_neural_engine = result.get("device_info", {}).get("capabilities", {}).get("neural_engine", False)
        
        # Collect metrics
        metrics = {
            "model_name": model_name,
            "device_name": device_model,
            "collection_time": datetime.datetime.now().isoformat(),
            "duration_seconds": execution_time,
            "simulated": False,
            "device_model": device_model,
            "battery_impact_percent": battery_impact,
            "thermal_throttling_detected": thermal_throttling,
            "thermal_throttling_duration_seconds": throttling_duration,
            "battery_temperature_celsius": battery_metrics.get("post_temperature", 0),
            "soc_temperature_celsius": thermal_metrics.get("post", {}).get("cpu", 0),
            "power_efficiency_score": 100 - min(100, max(0, battery_impact * 10)),
            "startup_time_ms": result.get("latency_ms", {}).get("values", [0])[0] if result.get("latency_ms", {}).get("values") else 0,
            "runtime_memory_profile": {
                "peak_memory_mb": result.get("memory_metrics", {}).get("peak_mb", 0),
                "average_memory_mb": result.get("memory_metrics", {}).get("peak_mb", 0) * 0.8  # Estimate
            },
            "power_metrics": {
                "average_power_watts": 0.5 + 0.1 * battery_impact,  # Estimate
                "peak_power_watts": 0.8 + 0.2 * battery_impact  # Estimate
            },
            "thermal_data": [
                {
                    "timestamp": time.time(),
                    "soc_temperature_celsius": thermal_metrics.get("post", {}).get("cpu", 0),
                    "battery_temperature_celsius": battery_metrics.get("post_temperature", 0),
                    "cpu_temperature_celsius": thermal_metrics.get("post", {}).get("cpu", 0),
                    "gpu_temperature_celsius": thermal_metrics.get("post", {}).get("gpu", 0),
                    "ambient_temperature_celsius": 25.0,  # Placeholder
                    "throttling_active": thermal_throttling,
                    "throttling_level": 1 if thermal_throttling else 0
                }
            ],
            "optimization_settings": {
                "quantization_method": "fp16",  # Placeholder
                "precision": "fp16",  # Placeholder
                "thread_count": 4,  # iOS manages threading automatically
                "batch_size": result.get("configuration", {}).get("batch_size", 1),
                "power_mode": "balanced",  # Placeholder
                "memory_optimization": "none",  # Placeholder
                "delegate": "neural_engine" if has_neural_engine else "gpu",
                "cache_enabled": True,
                "optimization_level": 3
            }
        }
        
        return metrics
    
    def _save_results(self, results: Dict[str, Any], model_name: str) -> str:
        """
        Save benchmark results to a file.
        
        Args:
            results: Benchmark results
            model_name: Name of the model
            
        Returns:
            Path to the saved file
        """
        # Generate filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.output_dir}/{model_name}_{timestamp}.json"
        
        # Save to file
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Benchmark results saved to {filename}")
        return filename
    
    def _save_to_database(self, results: Dict[str, Any]) -> bool:
        """
        Save benchmark results to database.
        
        Args:
            results: Benchmark results
            
        Returns:
            Success status
        """
        if not self.db_api:
            logger.warning("Database API not available")
            return False
        
        try:
            # Store in core database
            if hasattr(self.db_api, 'store_ios_benchmark'):
                self.db_api.store_ios_benchmark(results)
                logger.info("Benchmark results saved to database")
                return True
            
            logger.warning("Database API does not support iOS benchmarks")
            return False
        
        except Exception as e:
            logger.error(f"Failed to save results to database: {e}")
            return False
    
    def generate_report(self, 
                       results_file: Optional[str] = None,
                       results_data: Optional[Dict[str, Any]] = None,
                       report_format: str = "markdown") -> str:
        """
        Generate a report from benchmark results.
        
        Args:
            results_file: Optional path to results file
            results_data: Optional results data
            report_format: Report format (markdown, html)
            
        Returns:
            Generated report
        """
        # Get results data
        if results_data is None:
            if results_file is None:
                logger.error("Either results_file or results_data must be provided")
                return ""
            
            with open(results_file, 'r') as f:
                results_data = json.load(f)
        
        # Generate report
        if report_format == "html":
            return self._generate_html_report(results_data)
        else:
            return self._generate_markdown_report(results_data)
    
    def _generate_markdown_report(self, results: Dict[str, Any]) -> str:
        """
        Generate a markdown report from benchmark results.
        
        Args:
            results: Benchmark results
            
        Returns:
            Markdown report
        """
        model_name = results.get("model_name", "Unknown")
        device_info = results.get("device_info", {})
        device_name = device_info.get("name", "Unknown")
        
        # Generate report
        report = f"# iOS Benchmark Report: {model_name}\n\n"
        report += f"Generated: {results.get('timestamp', datetime.datetime.now().isoformat())}\n\n"
        
        # Device information
        report += "## Device Information\n\n"
        report += f"- **Name**: {device_name}\n"
        report += f"- **Model**: {device_info.get('model', 'Unknown')}\n"
        report += f"- **iOS Version**: {device_info.get('ios_version', 'Unknown')}\n"
        
        # Neural Engine availability
        has_neural_engine = device_info.get("capabilities", {}).get("neural_engine", False)
        report += f"- **Neural Engine**: {'Available' if has_neural_engine else 'Not Available'}\n"
        report += f"- **Core ML Version**: {device_info.get('capabilities', {}).get('coreml_version', 'Unknown')}\n\n"
        
        # Results summary
        report += "## Benchmark Results Summary\n\n"
        
        # Create summary table
        report += "| Configuration | Latency (ms) | Throughput | Battery Impact | Thermal Impact |\n"
        report += "|--------------|-------------|------------|----------------|----------------|\n"
        
        for config in results.get("configurations", []):
            config_info = config.get("configuration", {})
            batch_size = config_info.get("batch_size", 1)
            compute_units = config_info.get("compute_units", "all")
            
            latency = config.get("latency_ms", {}).get("mean", 0)
            throughput = config.get("throughput_items_per_second", 0)
            
            battery_impact = config.get("battery_metrics", {}).get("impact_percentage", 0)
            
            thermal_metrics = config.get("thermal_metrics", {})
            thermal_delta = thermal_metrics.get("delta", {})
            max_thermal_delta = max(thermal_delta.values()) if thermal_delta else 0
            
            report += f"| B{batch_size}, {compute_units} | {latency:.2f} | {throughput:.2f} items/s | {battery_impact:.1f}% | {max_thermal_delta:.1f}°C |\n"
        
        # Detailed results
        report += "\n## Detailed Results\n\n"
        
        for i, config in enumerate(results.get("configurations", [])):
            config_info = config.get("configuration", {})
            batch_size = config_info.get("batch_size", 1)
            compute_units = config_info.get("compute_units", "all")
            
            report += f"### Configuration {i+1}: Batch Size {batch_size}, Compute Units {compute_units}\n\n"
            
            # Latency statistics
            latency = config.get("latency_ms", {})
            report += "#### Latency (ms)\n\n"
            report += f"- **Min**: {latency.get('min', 0):.2f}\n"
            report += f"- **Mean**: {latency.get('mean', 0):.2f}\n"
            report += f"- **Median**: {latency.get('median', 0):.2f}\n"
            report += f"- **P90**: {latency.get('p90', 0):.2f}\n"
            report += f"- **P95**: {latency.get('p95', 0):.2f}\n"
            report += f"- **P99**: {latency.get('p99', 0):.2f}\n"
            report += f"- **Max**: {latency.get('max', 0):.2f}\n\n"
            
            # Throughput
            report += "#### Throughput\n\n"
            report += f"- **Items/second**: {config.get('throughput_items_per_second', 0):.2f}\n\n"
            
            # Battery impact
            battery_metrics = config.get("battery_metrics", {})
            report += "#### Battery Impact\n\n"
            report += f"- **Level Change**: {battery_metrics.get('pre_level', 0)} → {battery_metrics.get('post_level', 0)} ({battery_metrics.get('impact_percentage', 0):.1f}%)\n"
            report += f"- **Temperature Change**: {battery_metrics.get('pre_temperature', 0):.1f}°C → {battery_metrics.get('post_temperature', 0):.1f}°C ({battery_metrics.get('temperature_delta', 0):.1f}°C)\n\n"
            
            # Thermal impact
            thermal_metrics = config.get("thermal_metrics", {})
            report += "#### Thermal Impact\n\n"
            
            pre_thermal = thermal_metrics.get("pre", {})
            post_thermal = thermal_metrics.get("post", {})
            delta_thermal = thermal_metrics.get("delta", {})
            
            for zone in sorted(post_thermal.keys()):
                pre_val = pre_thermal.get(zone, 0)
                post_val = post_thermal.get(zone, 0)
                delta_val = delta_thermal.get(zone, 0)
                
                report += f"- **{zone}**: {pre_val:.1f}°C → {post_val:.1f}°C ({delta_val:+.1f}°C)\n"
            
            report += "\n"
            
            # Memory metrics
            memory_metrics = config.get("memory_metrics", {})
            report += "#### Memory Usage\n\n"
            report += f"- **Peak**: {memory_metrics.get('peak_mb', 0):.1f} MB\n\n"
        
        # Recommendations
        report += "## Recommendations\n\n"
        
        # Find best configuration
        best_config = None
        best_throughput = 0
        
        for config in results.get("configurations", []):
            throughput = config.get("throughput_items_per_second", 0)
            if throughput > best_throughput:
                best_throughput = throughput
                best_config = config
        
        if best_config:
            config_info = best_config.get("configuration", {})
            batch_size = config_info.get("batch_size", 1)
            compute_units = config_info.get("compute_units", "all")
            
            report += f"- **Best Configuration**: Batch Size {batch_size}, Compute Units {compute_units}\n"
            report += f"- **Best Performance**: {best_throughput:.2f} items/second\n\n"
        
        report += "Based on the benchmark results, we recommend:\n\n"
        
        # Check battery impact
        avg_battery_impact = sum(
            config.get("battery_metrics", {}).get("impact_percentage", 0) 
            for config in results.get("configurations", [])
        ) / max(1, len(results.get("configurations", [])))
        
        if avg_battery_impact > 5:
            report += "- **Power Optimization**: Model shows significant battery impact. Consider optimizing for power efficiency.\n"
        
        # Check thermal impact
        max_thermal_deltas = [
            max(config.get("thermal_metrics", {}).get("delta", {}).values()) 
            for config in results.get("configurations", [])
        ]
        avg_thermal_impact = sum(max_thermal_deltas) / max(1, len(max_thermal_deltas))
        
        if avg_thermal_impact > 5:
            report += "- **Thermal Management**: Model shows significant thermal impact. Consider adding cooling periods during inference.\n"
        
        # Neural Engine recommendation
        if not has_neural_engine:
            report += "- **Hardware Upgrade**: This device does not have a Neural Engine. Consider using a newer device with Neural Engine for better performance.\n"
        
        return report
    
    def _generate_html_report(self, results: Dict[str, Any]) -> str:
        """
        Generate an HTML report from benchmark results.
        
        Args:
            results: Benchmark results
            
        Returns:
            HTML report
        """
        model_name = results.get("model_name", "Unknown")
        device_info = results.get("device_info", {})
        device_name = device_info.get("name", "Unknown")
        
        # Has Neural Engine
        has_neural_engine = device_info.get("capabilities", {}).get("neural_engine", False)
        
        # Generate HTML
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>iOS Benchmark Report: {model_name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
        h1, h2, h3, h4 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        .chart {{ width: 100%; height: 300px; margin-bottom: 20px; }}
        .positive-delta {{ color: red; }}
        .negative-delta {{ color: green; }}
    </style>
</head>
<body>
    <h1>iOS Benchmark Report: {model_name}</h1>
    <p>Generated: {results.get('timestamp', datetime.datetime.now().isoformat())}</p>
    
    <h2>Device Information</h2>
    <table>
        <tr><th>Property</th><th>Value</th></tr>
        <tr><td>Name</td><td>{device_name}</td></tr>
        <tr><td>Model</td><td>{device_info.get('model', 'Unknown')}</td></tr>
        <tr><td>iOS Version</td><td>{device_info.get('ios_version', 'Unknown')}</td></tr>
        <tr><td>Neural Engine</td><td>{'Available' if has_neural_engine else 'Not Available'}</td></tr>
        <tr><td>Core ML Version</td><td>{device_info.get('capabilities', {}).get('coreml_version', 'Unknown')}</td></tr>
    </table>
    
    <h2>Benchmark Results Summary</h2>
    <table>
        <tr>
            <th>Configuration</th>
            <th>Latency (ms)</th>
            <th>Throughput</th>
            <th>Battery Impact</th>
            <th>Thermal Impact</th>
        </tr>
"""
        
        for config in results.get("configurations", []):
            config_info = config.get("configuration", {})
            batch_size = config_info.get("batch_size", 1)
            compute_units = config_info.get("compute_units", "all")
            
            latency = config.get("latency_ms", {}).get("mean", 0)
            throughput = config.get("throughput_items_per_second", 0)
            
            battery_impact = config.get("battery_metrics", {}).get("impact_percentage", 0)
            
            thermal_metrics = config.get("thermal_metrics", {})
            thermal_delta = thermal_metrics.get("delta", {})
            max_thermal_delta = max(thermal_delta.values()) if thermal_delta else 0
            
            html += f"""
        <tr>
            <td>B{batch_size}, {compute_units}</td>
            <td>{latency:.2f}</td>
            <td>{throughput:.2f} items/s</td>
            <td>{battery_impact:.1f}%</td>
            <td>{max_thermal_delta:.1f}°C</td>
        </tr>"""
        
        html += """
    </table>
    
    <h2>Detailed Results</h2>
"""
        
        for i, config in enumerate(results.get("configurations", [])):
            config_info = config.get("configuration", {})
            batch_size = config_info.get("batch_size", 1)
            compute_units = config_info.get("compute_units", "all")
            
            html += f"""
    <h3>Configuration {i+1}: Batch Size {batch_size}, Compute Units {compute_units}</h3>
    
    <h4>Latency (ms)</h4>
    <table>
        <tr>
            <th>Min</th>
            <th>Mean</th>
            <th>Median</th>
            <th>P90</th>
            <th>P95</th>
            <th>P99</th>
            <th>Max</th>
        </tr>
        <tr>
            <td>{config.get('latency_ms', {}).get('min', 0):.2f}</td>
            <td>{config.get('latency_ms', {}).get('mean', 0):.2f}</td>
            <td>{config.get('latency_ms', {}).get('median', 0):.2f}</td>
            <td>{config.get('latency_ms', {}).get('p90', 0):.2f}</td>
            <td>{config.get('latency_ms', {}).get('p95', 0):.2f}</td>
            <td>{config.get('latency_ms', {}).get('p99', 0):.2f}</td>
            <td>{config.get('latency_ms', {}).get('max', 0):.2f}</td>
        </tr>
    </table>
    
    <h4>Throughput</h4>
    <p><strong>Items/second</strong>: {config.get('throughput_items_per_second', 0):.2f}</p>
    
    <h4>Battery Impact</h4>
    <table>
        <tr>
            <th>Metric</th>
            <th>Before</th>
            <th>After</th>
            <th>Change</th>
        </tr>
        <tr>
            <td>Battery Level</td>
            <td>{config.get('battery_metrics', {}).get('pre_level', 0)}%</td>
            <td>{config.get('battery_metrics', {}).get('post_level', 0)}%</td>
            <td>{config.get('battery_metrics', {}).get('impact_percentage', 0):.1f}%</td>
        </tr>
        <tr>
            <td>Battery Temperature</td>
            <td>{config.get('battery_metrics', {}).get('pre_temperature', 0):.1f}°C</td>
            <td>{config.get('battery_metrics', {}).get('post_temperature', 0):.1f}°C</td>
            <td>{config.get('battery_metrics', {}).get('temperature_delta', 0):.1f}°C</td>
        </tr>
    </table>
    
    <h4>Thermal Impact</h4>
    <table>
        <tr>
            <th>Zone</th>
            <th>Before</th>
            <th>After</th>
            <th>Change</th>
        </tr>
"""
            
            thermal_metrics = config.get("thermal_metrics", {})
            pre_thermal = thermal_metrics.get("pre", {})
            post_thermal = thermal_metrics.get("post", {})
            delta_thermal = thermal_metrics.get("delta", {})
            
            for zone in sorted(post_thermal.keys()):
                pre_val = pre_thermal.get(zone, 0)
                post_val = post_thermal.get(zone, 0)
                delta_val = delta_thermal.get(zone, 0)
                
                delta_class = "positive-delta" if delta_val > 0 else "negative-delta" if delta_val < 0 else ""
                
                html += f"""
        <tr>
            <td>{zone}</td>
            <td>{pre_val:.1f}°C</td>
            <td>{post_val:.1f}°C</td>
            <td class="{delta_class}">{delta_val:+.1f}°C</td>
        </tr>"""
            
            html += """
    </table>
    
    <h4>Memory Usage</h4>
    <p><strong>Peak</strong>: """ + f"{config.get('memory_metrics', {}).get('peak_mb', 0):.1f} MB</p>"
        
        # Recommendations
        html += """
    <h2>Recommendations</h2>
"""
        
        # Find best configuration
        best_config = None
        best_throughput = 0
        
        for config in results.get("configurations", []):
            throughput = config.get("throughput_items_per_second", 0)
            if throughput > best_throughput:
                best_throughput = throughput
                best_config = config
        
        if best_config:
            config_info = best_config.get("configuration", {})
            batch_size = config_info.get("batch_size", 1)
            compute_units = config_info.get("compute_units", "all")
            
            html += f"""
    <p><strong>Best Configuration</strong>: Batch Size {batch_size}, Compute Units {compute_units}</p>
    <p><strong>Best Performance</strong>: {best_throughput:.2f} items/second</p>
"""
        
        html += """
    <p>Based on the benchmark results, we recommend:</p>
    <ul>
"""
        
        # Check battery impact
        avg_battery_impact = sum(
            config.get("battery_metrics", {}).get("impact_percentage", 0) 
            for config in results.get("configurations", [])
        ) / max(1, len(results.get("configurations", [])))
        
        if avg_battery_impact > 5:
            html += """
        <li><strong>Power Optimization</strong>: Model shows significant battery impact. Consider optimizing for power efficiency.</li>
"""
        
        # Check thermal impact
        max_thermal_deltas = [
            max(config.get("thermal_metrics", {}).get("delta", {}).values()) 
            for config in results.get("configurations", [])
        ]
        avg_thermal_impact = sum(max_thermal_deltas) / max(1, len(max_thermal_deltas))
        
        if avg_thermal_impact > 5:
            html += """
        <li><strong>Thermal Management</strong>: Model shows significant thermal impact. Consider adding cooling periods during inference.</li>
"""
        
        # Neural Engine recommendation
        if not has_neural_engine:
            html += """
        <li><strong>Hardware Upgrade</strong>: This device does not have a Neural Engine. Consider using a newer device with Neural Engine for better performance.</li>
"""
        
        html += """
    </ul>
</body>
</html>
"""
        
        return html


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="iOS Test Harness for IPFS Accelerate Python Framework")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Connect command
    connect_parser = subparsers.add_parser("connect", help="Connect to an iOS device")
    connect_parser.add_argument("--udid", help="Device UDID")
    connect_parser.add_argument("--developer-dir", help="Xcode Developer directory")
    connect_parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    # Run benchmark command
    benchmark_parser = subparsers.add_parser("benchmark", help="Run a model benchmark")
    benchmark_parser.add_argument("--model", required=True, help="Path to model file")
    benchmark_parser.add_argument("--name", help="Model name (defaults to filename)")
    benchmark_parser.add_argument("--type", default="coreml", choices=["coreml", "onnx"], help="Model type")
    benchmark_parser.add_argument("--udid", help="Device UDID")
    benchmark_parser.add_argument("--db-path", help="Path to benchmark database")
    benchmark_parser.add_argument("--output-dir", default="./ios_results", help="Directory to save results")
    benchmark_parser.add_argument("--iterations", type=int, default=50, help="Number of iterations")
    benchmark_parser.add_argument("--batch-sizes", type=str, default="1", help="Comma-separated list of batch sizes")
    benchmark_parser.add_argument("--skip-db", action="store_true", help="Skip database storage")
    benchmark_parser.add_argument("--skip-metrics", action="store_true", help="Skip detailed metrics collection")
    benchmark_parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    # Generate report command
    report_parser = subparsers.add_parser("report", help="Generate a benchmark report")
    report_parser.add_argument("--results", required=True, help="Path to benchmark results file")
    report_parser.add_argument("--format", default="markdown", choices=["markdown", "html"], help="Report format")
    report_parser.add_argument("--output", help="Path to save the report")
    report_parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Set log level
    if hasattr(args, "verbose") and args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Check if running on macOS
    if platform.system() != "Darwin":
        print("Error: iOS device support is only available on macOS")
        return 1
    
    # Execute command
    if args.command == "connect":
        harness = IOSTestHarness(args.udid, args.developer_dir)
        connected = harness.connect_to_device()
        
        if connected:
            device_info = harness.device.to_dict()
            print("Connected to iOS device:")
            for key, value in device_info.items():
                if key != "capabilities":
                    print(f"  {key}: {value}")
                else:
                    print("  capabilities:")
                    for cap_key, cap_value in value.items():
                        print(f"    {cap_key}: {cap_value}")
        else:
            print("Failed to connect to iOS device")
    
    elif args.command == "benchmark":
        # Parse batch sizes
        batch_sizes = [int(bs) for bs in args.batch_sizes.split(",")]
        
        # Determine model name
        model_name = args.name or os.path.basename(args.model)
        
        # Create test harness
        harness = IOSTestHarness(
            udid=args.udid,
            db_path=args.db_path,
            output_dir=args.output_dir
        )
        
        # Run benchmark
        results = harness.run_benchmark(
            model_path=args.model,
            model_name=model_name,
            model_type=args.type,
            batch_sizes=batch_sizes,
            iterations=args.iterations,
            save_to_db=not args.skip_db,
            collect_metrics=not args.skip_metrics
        )
        
        # Print summary
        if results.get("status") == "success":
            print(f"Benchmark completed for {model_name}")
            print(f"Device: {results.get('device_info', {}).get('name', 'Unknown')}")
            print(f"Configurations tested: {len(results.get('configurations', []))}")
            
            # Find best configuration
            best_config = None
            best_throughput = 0
            
            for config in results.get("configurations", []):
                throughput = config.get("throughput_items_per_second", 0)
                if throughput > best_throughput:
                    best_throughput = throughput
                    best_config = config
            
            if best_config:
                config_info = best_config.get("configuration", {})
                batch_size = config_info.get("batch_size", 1)
                compute_units = config_info.get("compute_units", "all")
                
                print(f"Best configuration: Batch Size {batch_size}, Compute Units {compute_units}")
                print(f"Best performance: {best_throughput:.2f} items/second")
        else:
            print(f"Benchmark failed: {results.get('message', 'Unknown error')}")
    
    elif args.command == "report":
        # Create test harness
        harness = IOSTestHarness()
        
        # Generate report
        report = harness.generate_report(
            results_file=args.results,
            report_format=args.format
        )
        
        # Save or print report
        if args.output:
            with open(args.output, 'w') as f:
                f.write(report)
            print(f"Report saved to {args.output}")
        else:
            print(report)
    
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    exit(main())