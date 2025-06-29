#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mobile CI Runners Setup Utility

This script helps set up and manage CI runners for mobile testing in the IPFS Accelerate
Python Framework. It handles runner registration, device connectivity verification,
and environment configuration.

Usage:
    python setup_mobile_ci_runners.py [--action {check,register,configure,verify}]
    [--platform {android,ios,all}] [--device-id DEVICE_ID]

Examples:
    # Check current environment
    python setup_mobile_ci_runners.py --action check --platform all
    
    # Register a new runner
    python setup_mobile_ci_runners.py --action register --platform android
    
    # Configure environment for a specific device
    python setup_mobile_ci_runners.py --action configure --platform android --device-id emulator-5554

Date: May 2025
"""

import os
import sys
import json
import time
import argparse
import logging
import subprocess
import platform
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MobileCIRunnerSetup:
    """
    Sets up and manages CI runners for mobile testing.
    """
    
    def __init__(self, 
                 platform_type: str = "all",
                 device_id: Optional[str] = None,
                 verbose: bool = False):
        """
        Initialize the CI runner setup utility.
        
        Args:
            platform_type: Type of platform (android, ios, all)
            device_id: Optional device ID for specific device operations
            verbose: Enable verbose logging
        """
        self.platform_type = platform_type.lower()
        self.device_id = device_id
        self.verbose = verbose
        
        # Set log level
        if verbose:
            logger.setLevel(logging.DEBUG)
        
        # Determine current OS
        self.system = platform.system()
        logger.info(f"Detected operating system: {self.system}")
        
        # Check if platform type is valid
        if self.platform_type not in ["android", "ios", "all"]:
            logger.error(f"Invalid platform type: {platform_type}")
            logger.error("Platform type must be one of: android, ios, all")
            raise ValueError(f"Invalid platform type: {platform_type}")
    
    def check_environment(self) -> Dict[str, Any]:
        """
        Check the current environment for mobile testing.
        
        Returns:
            Dictionary with environment status
        """
        environment_status = {
            "system": self.system,
            "python_version": platform.python_version(),
            "platforms": {}
        }
        
        # Define platforms to check
        platforms_to_check = []
        if self.platform_type == "all":
            platforms_to_check = ["android", "ios"]
        else:
            platforms_to_check = [self.platform_type]
        
        # Check each platform
        for platform_type in platforms_to_check:
            if platform_type == "android":
                environment_status["platforms"]["android"] = self._check_android_environment()
            elif platform_type == "ios":
                environment_status["platforms"]["ios"] = self._check_ios_environment()
        
        return environment_status
    
    def _check_android_environment(self) -> Dict[str, Any]:
        """
        Check the Android development environment.
        
        Returns:
            Dictionary with Android environment status
        """
        android_status = {
            "status": "unknown",
            "java_installed": False,
            "android_sdk_installed": False,
            "adb_installed": False,
            "emulator_installed": False,
            "sdk_path": None,
            "devices": []
        }
        
        # Check Java
        try:
            java_output = subprocess.run(
                ["java", "-version"],
                capture_output=True,
                text=True,
                check=False
            )
            android_status["java_installed"] = java_output.returncode == 0
            if android_status["java_installed"]:
                java_version = java_output.stderr.split("\n")[0]
                android_status["java_version"] = java_version
        except FileNotFoundError:
            android_status["java_installed"] = False
        
        # Check Android SDK
        android_home = os.environ.get("ANDROID_HOME", os.environ.get("ANDROID_SDK_ROOT"))
        if android_home and os.path.exists(android_home):
            android_status["android_sdk_installed"] = True
            android_status["sdk_path"] = android_home
        
        # Check ADB
        try:
            adb_output = subprocess.run(
                ["adb", "version"],
                capture_output=True,
                text=True,
                check=False
            )
            android_status["adb_installed"] = adb_output.returncode == 0
            if android_status["adb_installed"]:
                adb_version = adb_output.stdout.split("\n")[0]
                android_status["adb_version"] = adb_version
                
                # Get connected devices
                devices = self._get_android_devices()
                android_status["devices"] = devices
        except FileNotFoundError:
            android_status["adb_installed"] = False
        
        # Check emulator
        try:
            emulator_output = subprocess.run(
                ["emulator", "-list-avds"],
                capture_output=True,
                text=True,
                check=False
            )
            android_status["emulator_installed"] = emulator_output.returncode == 0
            if android_status["emulator_installed"]:
                avds = [avd.strip() for avd in emulator_output.stdout.strip().split("\n") if avd.strip()]
                android_status["available_avds"] = avds
        except FileNotFoundError:
            android_status["emulator_installed"] = False
        
        # Determine overall status
        if (android_status["java_installed"] and 
            android_status["android_sdk_installed"] and 
            android_status["adb_installed"]):
            android_status["status"] = "ready"
        else:
            android_status["status"] = "missing_dependencies"
        
        return android_status
    
    def _check_ios_environment(self) -> Dict[str, Any]:
        """
        Check the iOS development environment.
        
        Returns:
            Dictionary with iOS environment status
        """
        ios_status = {
            "status": "unknown",
            "macos": self.system == "Darwin",
            "xcode_installed": False,
            "xcrun_installed": False,
            "simulator_installed": False,
            "xcode_path": None,
            "devices": []
        }
        
        # iOS development is only supported on macOS
        if not ios_status["macos"]:
            ios_status["status"] = "unsupported_os"
            return ios_status
        
        # Check Xcode
        try:
            xcode_output = subprocess.run(
                ["xcode-select", "--print-path"],
                capture_output=True,
                text=True,
                check=False
            )
            ios_status["xcode_installed"] = xcode_output.returncode == 0
            if ios_status["xcode_installed"]:
                ios_status["xcode_path"] = xcode_output.stdout.strip()
        except FileNotFoundError:
            ios_status["xcode_installed"] = False
        
        # Check xcrun
        try:
            xcrun_output = subprocess.run(
                ["xcrun", "--version"],
                capture_output=True,
                text=True,
                check=False
            )
            ios_status["xcrun_installed"] = xcrun_output.returncode == 0
            if ios_status["xcrun_installed"]:
                xcrun_version = xcrun_output.stdout.strip()
                ios_status["xcrun_version"] = xcrun_version
                
                # Get connected devices
                devices = self._get_ios_devices()
                ios_status["devices"] = devices
                
                # Check simulator
                simctl_output = subprocess.run(
                    ["xcrun", "simctl", "list"],
                    capture_output=True,
                    text=True,
                    check=False
                )
                ios_status["simulator_installed"] = simctl_output.returncode == 0
                if ios_status["simulator_installed"]:
                    ios_status["simulators"] = self._parse_ios_simulators(simctl_output.stdout)
        except FileNotFoundError:
            ios_status["xcrun_installed"] = False
        
        # Determine overall status
        if (ios_status["macos"] and 
            ios_status["xcode_installed"] and 
            ios_status["xcrun_installed"]):
            ios_status["status"] = "ready"
        else:
            ios_status["status"] = "missing_dependencies"
        
        return ios_status
    
    def _get_android_devices(self) -> List[Dict[str, Any]]:
        """
        Get a list of connected Android devices.
        
        Returns:
            List of connected device information
        """
        devices = []
        
        try:
            # Get device list
            output = subprocess.run(
                ["adb", "devices", "-l"],
                capture_output=True,
                text=True,
                check=True
            )
            
            # Parse device list
            lines = output.stdout.strip().split("\n")
            for line in lines[1:]:  # Skip header line
                if line.strip():
                    parts = line.split()
                    if len(parts) >= 2:
                        device_id = parts[0]
                        device_info = {
                            "id": device_id,
                            "status": parts[1]
                        }
                        
                        # Get additional device info if device is connected
                        if device_info["status"] == "device":
                            try:
                                # Get device model
                                model_output = subprocess.run(
                                    ["adb", "-s", device_id, "shell", "getprop", "ro.product.model"],
                                    capture_output=True,
                                    text=True,
                                    check=True
                                )
                                device_info["model"] = model_output.stdout.strip()
                                
                                # Get Android version
                                version_output = subprocess.run(
                                    ["adb", "-s", device_id, "shell", "getprop", "ro.build.version.release"],
                                    capture_output=True,
                                    text=True,
                                    check=True
                                )
                                device_info["android_version"] = version_output.stdout.strip()
                                
                                # Check if emulator
                                device_info["emulator"] = "emulator" in device_id
                            except subprocess.SubprocessError:
                                pass
                        
                        devices.append(device_info)
        
        except (subprocess.SubprocessError, FileNotFoundError) as e:
            logger.error(f"Error getting Android devices: {e}")
        
        return devices
    
    def _get_ios_devices(self) -> List[Dict[str, Any]]:
        """
        Get a list of connected iOS devices.
        
        Returns:
            List of connected device information
        """
        devices = []
        
        # iOS devices can only be queried on macOS
        if self.system != "Darwin":
            return devices
        
        try:
            # Get device list using xcrun
            output = subprocess.run(
                ["xcrun", "xctrace", "list", "devices"],
                capture_output=True,
                text=True,
                check=True
            )
            
            # Parse device list
            lines = output.stdout.strip().split("\n")
            for line in lines:
                if "(" in line and ")" in line and "Simulator" not in line:
                    try:
                        # Parse line like "iPhone 12 (123456789) (iOS 15.0)"
                        device_name = line.split("(")[0].strip()
                        device_id = line.split("(")[1].split(")")[0].strip()
                        ios_version = line.split("(")[2].split(")")[0].strip().replace("iOS ", "")
                        
                        device_info = {
                            "id": device_id,
                            "name": device_name,
                            "ios_version": ios_version,
                            "status": "connected",
                            "simulator": False
                        }
                        
                        devices.append(device_info)
                    except (IndexError, ValueError):
                        pass
        
        except (subprocess.SubprocessError, FileNotFoundError) as e:
            logger.error(f"Error getting iOS devices: {e}")
        
        return devices
    
    def _parse_ios_simulators(self, simctl_output: str) -> List[Dict[str, Any]]:
        """
        Parse iOS simulator list from simctl output.
        
        Args:
            simctl_output: Output from simctl list command
            
        Returns:
            List of simulator information
        """
        simulators = []
        
        try:
            lines = simctl_output.strip().split("\n")
            current_device_set = None
            current_runtime = None
            
            for line in lines:
                line = line.strip()
                
                # Skip empty lines
                if not line:
                    continue
                
                # Detect device set
                if line.startswith("Device sets"):
                    current_device_set = line.split("from")[1].strip()
                
                # Detect runtime
                elif line.startswith("-- iOS ") and "(" in line and ")" in line:
                    current_runtime = line.replace("-- ", "").strip()
                
                # Detect simulator
                elif " (" in line and ")" in line and not line.startswith("--"):
                    try:
                        # Parse line like "iPhone 12 (00000000-0000-0000-0000-000000000000) (Shutdown)"
                        simulator_name = line.split(" (")[0].strip()
                        simulator_id = line.split(" (")[1].split(")")[0].strip()
                        simulator_state = line.split("(")[-1].split(")")[0].strip()
                        
                        simulator_info = {
                            "id": simulator_id,
                            "name": simulator_name,
                            "state": simulator_state,
                            "runtime": current_runtime,
                            "device_set": current_device_set,
                            "simulator": True
                        }
                        
                        simulators.append(simulator_info)
                    except (IndexError, ValueError):
                        pass
        
        except Exception as e:
            logger.error(f"Error parsing iOS simulators: {e}")
        
        return simulators
    
    def register_runner(self) -> bool:
        """
        Register a new CI runner.
        
        Returns:
            Success status
        """
        logger.info(f"Registering CI runner for {self.platform_type} platform")
        
        # Check environment first
        environment = self.check_environment()
        platforms = environment.get("platforms", {})
        
        if self.platform_type == "all":
            # Check if all platforms are ready
            all_ready = True
            for platform_type, platform_status in platforms.items():
                if platform_status.get("status") != "ready":
                    logger.error(f"Platform {platform_type} is not ready: {platform_status.get('status')}")
                    all_ready = False
            
            if not all_ready:
                logger.error("Not all platforms are ready for registration")
                return False
            
            # Register all platforms
            success = True
            for platform_type in platforms.keys():
                platform_success = self._register_platform_runner(platform_type)
                success = success and platform_success
            
            return success
        
        else:
            # Check if platform is ready
            platform_status = platforms.get(self.platform_type, {})
            if platform_status.get("status") != "ready":
                logger.error(f"Platform {self.platform_type} is not ready: {platform_status.get('status')}")
                return False
            
            # Register the specific platform
            return self._register_platform_runner(self.platform_type)
    
    def _register_platform_runner(self, platform_type: str) -> bool:
        """
        Register a CI runner for a specific platform.
        
        Args:
            platform_type: Type of platform (android, ios)
            
        Returns:
            Success status
        """
        logger.info(f"Registering runner for platform: {platform_type}")
        
        # Registration requires GitHub token and URL
        token = os.environ.get("GITHUB_TOKEN")
        url = os.environ.get("GITHUB_REPOSITORY")
        
        if not token or not url:
            logger.error("Missing required environment variables: GITHUB_TOKEN, GITHUB_REPOSITORY")
            logger.error("Please set these variables before registering runners")
            return False
        
        # Determine runner name and labels
        runner_name = f"{platform_type}-{platform.node()}"
        labels = f"{platform_type},mobile"
        
        if platform_type == "android":
            # Add Android-specific labels
            devices = self._get_android_devices()
            if devices:
                for device in devices:
                    if device.get("status") == "device":
                        if device.get("emulator", False):
                            labels += ",emulator"
                        else:
                            labels += ",physical"
                        
                        model = device.get("model", "").replace(" ", "-").lower()
                        if model:
                            labels += f",{model}"
                        
                        android_version = device.get("android_version", "").replace(".", "")
                        if android_version:
                            labels += f",android{android_version}"
        
        elif platform_type == "ios":
            # Add iOS-specific labels
            if self.system != "Darwin":
                logger.error("iOS runners can only be registered on macOS")
                return False
            
            # Add Xcode version label
            try:
                xcode_version_output = subprocess.run(
                    ["xcodebuild", "-version"],
                    capture_output=True,
                    text=True,
                    check=True
                )
                xcode_version = xcode_version_output.stdout.split("\n")[0].split(" ")[1].strip()
                labels += f",xcode{xcode_version.replace('.', '')}"
            except (subprocess.SubprocessError, IndexError):
                pass
            
            devices = self._get_ios_devices()
            if devices:
                for device in devices:
                    if device.get("status") == "connected":
                        labels += ",physical"
                        
                        name = device.get("name", "").replace(" ", "-").lower()
                        if name:
                            labels += f",{name}"
                        
                        ios_version = device.get("ios_version", "").replace(".", "")
                        if ios_version:
                            labels += f",ios{ios_version}"
            
            # Check for simulators
            try:
                simctl_output = subprocess.run(
                    ["xcrun", "simctl", "list"],
                    capture_output=True,
                    text=True,
                    check=True
                )
                if "Simulator" in simctl_output.stdout:
                    labels += ",simulator"
            except subprocess.SubprocessError:
                pass
        
        logger.info(f"Runner name: {runner_name}")
        logger.info(f"Runner labels: {labels}")
        
        # TODO: Implement actual GitHub runner registration using the Actions API
        # For now, display instructions to manually register the runner
        logger.info("\nTo manually register this runner:")
        logger.info("1. Go to your repository settings")
        logger.info("2. Navigate to Actions > Runners")
        logger.info("3. Click 'New self-hosted runner'")
        logger.info("4. Select the appropriate OS")
        logger.info("5. Follow the instructions to download and configure the runner")
        logger.info(f"6. Use the following labels when configuring: {labels}")
        
        return True
    
    def configure_environment(self) -> bool:
        """
        Configure the environment for mobile testing.
        
        Returns:
            Success status
        """
        logger.info(f"Configuring environment for {self.platform_type} platform")
        
        # Check environment first
        environment = self.check_environment()
        platforms = environment.get("platforms", {})
        
        if self.platform_type == "all":
            # Configure all platforms
            success = True
            for platform_type in platforms.keys():
                platform_success = self._configure_platform_environment(platform_type)
                success = success and platform_success
            
            return success
        
        else:
            # Configure the specific platform
            return self._configure_platform_environment(self.platform_type)
    
    def _configure_platform_environment(self, platform_type: str) -> bool:
        """
        Configure the environment for a specific platform.
        
        Args:
            platform_type: Type of platform (android, ios)
            
        Returns:
            Success status
        """
        logger.info(f"Configuring environment for platform: {platform_type}")
        
        if platform_type == "android":
            return self._configure_android_environment()
        elif platform_type == "ios":
            return self._configure_ios_environment()
        else:
            logger.error(f"Unsupported platform type: {platform_type}")
            return False
    
    def _configure_android_environment(self) -> bool:
        """
        Configure the Android environment.
        
        Returns:
            Success status
        """
        # Check if Android SDK is already configured
        android_home = os.environ.get("ANDROID_HOME", os.environ.get("ANDROID_SDK_ROOT"))
        if android_home and os.path.exists(android_home):
            logger.info(f"Android SDK already configured at: {android_home}")
            
            # Check for required dependencies
            missing_deps = []
            
            # Check Java
            try:
                subprocess.run(["java", "-version"], capture_output=True, check=True)
            except (subprocess.SubprocessError, FileNotFoundError):
                missing_deps.append("java")
            
            # Check ADB
            try:
                subprocess.run(["adb", "version"], capture_output=True, check=True)
            except (subprocess.SubprocessError, FileNotFoundError):
                missing_deps.append("adb")
            
            # Install missing dependencies
            if missing_deps:
                logger.warning(f"Missing dependencies: {', '.join(missing_deps)}")
                logger.info("Attempting to install missing dependencies...")
                
                if self.system == "Linux":
                    try:
                        # Install on Linux
                        subprocess.run(
                            ["sudo", "apt-get", "update"],
                            check=True
                        )
                        subprocess.run(
                            ["sudo", "apt-get", "install", "-y", "openjdk-11-jdk", "adb"],
                            check=True
                        )
                        logger.info("Dependencies installed successfully")
                    except subprocess.SubprocessError as e:
                        logger.error(f"Error installing dependencies: {e}")
                        return False
                
                elif self.system == "Darwin":
                    try:
                        # Check if Homebrew is installed
                        subprocess.run(["brew", "--version"], capture_output=True, check=True)
                        
                        # Install on macOS
                        subprocess.run(
                            ["brew", "install", "openjdk@11", "android-platform-tools"],
                            check=True
                        )
                        logger.info("Dependencies installed successfully")
                    except (subprocess.SubprocessError, FileNotFoundError):
                        logger.error("Homebrew is required to install dependencies on macOS")
                        logger.error("Please install Homebrew from https://brew.sh/")
                        return False
                
                else:
                    logger.error(f"Unsupported operating system: {self.system}")
                    logger.error("Please install the required dependencies manually")
                    return False
        
        else:
            logger.error("Android SDK not found")
            logger.error("Please install Android SDK and set ANDROID_HOME environment variable")
            return False
        
        # Check for connected devices
        devices = self._get_android_devices()
        if not devices:
            logger.warning("No Android devices connected")
            logger.warning("Please connect a device or start an emulator")
        
        # If a specific device ID is provided, check if it's connected
        if self.device_id:
            found = False
            for device in devices:
                if device.get("id") == self.device_id:
                    found = True
                    logger.info(f"Device found: {device}")
                    if device.get("status") != "device":
                        logger.warning(f"Device status is not 'device': {device.get('status')}")
                    break
            
            if not found:
                logger.error(f"Device with ID '{self.device_id}' not found")
                return False
        
        # Install test dependencies
        try:
            logger.info("Installing test dependencies...")
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
                check=True
            )
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r", "test/android_test_harness/requirements.txt"],
                check=True
            )
            logger.info("Test dependencies installed successfully")
        except subprocess.SubprocessError as e:
            logger.error(f"Error installing test dependencies: {e}")
            return False
        
        # Create necessary directories
        os.makedirs("models", exist_ok=True)
        os.makedirs("benchmark_results", exist_ok=True)
        os.makedirs("android_benchmark_results", exist_ok=True)
        
        # Download test models
        try:
            logger.info("Downloading test models...")
            subprocess.run(
                [sys.executable, "test/android_test_harness/download_test_models.py"],
                check=True
            )
            logger.info("Test models downloaded successfully")
        except subprocess.SubprocessError as e:
            logger.error(f"Error downloading test models: {e}")
            return False
        
        logger.info("Android environment configured successfully")
        return True
    
    def _configure_ios_environment(self) -> bool:
        """
        Configure the iOS environment.
        
        Returns:
            Success status
        """
        # iOS environment can only be configured on macOS
        if self.system != "Darwin":
            logger.error("iOS environment can only be configured on macOS")
            return False
        
        # Check if Xcode is installed
        try:
            xcode_path = subprocess.run(
                ["xcode-select", "--print-path"],
                capture_output=True,
                text=True,
                check=True
            ).stdout.strip()
            logger.info(f"Xcode found at: {xcode_path}")
        except subprocess.SubprocessError:
            logger.error("Xcode not found")
            logger.error("Please install Xcode from the App Store")
            return False
        
        # Check for connected devices
        devices = self._get_ios_devices()
        simulators = []
        
        try:
            simctl_output = subprocess.run(
                ["xcrun", "simctl", "list"],
                capture_output=True,
                text=True,
                check=True
            )
            simulators = self._parse_ios_simulators(simctl_output.stdout)
        except subprocess.SubprocessError:
            pass
        
        if not devices and not simulators:
            logger.warning("No iOS devices or simulators found")
            logger.warning("Please connect a device or create a simulator")
        
        # If a specific device ID is provided, check if it's connected
        if self.device_id:
            found_device = False
            found_simulator = False
            
            # Check physical devices
            for device in devices:
                if device.get("id") == self.device_id:
                    found_device = True
                    logger.info(f"Device found: {device}")
                    break
            
            # Check simulators
            for simulator in simulators:
                if simulator.get("id") == self.device_id:
                    found_simulator = True
                    logger.info(f"Simulator found: {simulator}")
                    
                    # Check if simulator is booted
                    if simulator.get("state") != "Booted":
                        logger.info(f"Booting simulator: {self.device_id}")
                        try:
                            subprocess.run(
                                ["xcrun", "simctl", "boot", self.device_id],
                                check=True
                            )
                            logger.info("Simulator booted successfully")
                        except subprocess.SubprocessError as e:
                            logger.error(f"Error booting simulator: {e}")
                            return False
                    break
            
            if not found_device and not found_simulator:
                logger.error(f"Device or simulator with ID '{self.device_id}' not found")
                return False
        
        # Install test dependencies
        try:
            logger.info("Installing test dependencies...")
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
                check=True
            )
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r", "test/ios_test_harness/requirements.txt"],
                check=True
            )
            logger.info("Test dependencies installed successfully")
        except subprocess.SubprocessError as e:
            logger.error(f"Error installing test dependencies: {e}")
            return False
        
        # Create necessary directories
        os.makedirs("models", exist_ok=True)
        os.makedirs("benchmark_results", exist_ok=True)
        os.makedirs("ios_benchmark_results", exist_ok=True)
        
        # Install coremltools if needed
        try:
            logger.info("Installing coremltools...")
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "coremltools>=6.0"],
                check=True
            )
            logger.info("coremltools installed successfully")
        except subprocess.SubprocessError as e:
            logger.error(f"Error installing coremltools: {e}")
            return False
        
        # Download test models
        try:
            logger.info("Downloading test models...")
            subprocess.run(
                [sys.executable, "test/ios_test_harness/download_test_models.py", "--install-deps"],
                check=True
            )
            logger.info("Test models downloaded successfully")
        except subprocess.SubprocessError as e:
            logger.error(f"Error downloading test models: {e}")
            return False
        
        logger.info("iOS environment configured successfully")
        return True
    
    def verify_device_connectivity(self) -> bool:
        """
        Verify device connectivity for mobile testing.
        
        Returns:
            Success status
        """
        logger.info(f"Verifying device connectivity for {self.platform_type} platform")
        
        if self.platform_type == "all":
            # Verify all platforms
            android_success = self._verify_android_connectivity()
            ios_success = self._verify_ios_connectivity()
            return android_success and ios_success
        
        elif self.platform_type == "android":
            return self._verify_android_connectivity()
        
        elif self.platform_type == "ios":
            return self._verify_ios_connectivity()
        
        else:
            logger.error(f"Unsupported platform type: {self.platform_type}")
            return False
    
    def _verify_android_connectivity(self) -> bool:
        """
        Verify Android device connectivity.
        
        Returns:
            Success status
        """
        # Get connected devices
        devices = self._get_android_devices()
        
        if not devices:
            logger.error("No Android devices connected")
            return False
        
        # If device ID is specified, test that specific device
        if self.device_id:
            found = False
            for device in devices:
                if device.get("id") == self.device_id:
                    found = True
                    logger.info(f"Testing device: {device}")
                    
                    if device.get("status") != "device":
                        logger.error(f"Device status is not 'device': {device.get('status')}")
                        logger.error("Device is not ready for testing")
                        return False
                    
                    return self._test_android_device(device.get("id"))
            
            if not found:
                logger.error(f"Device with ID '{self.device_id}' not found")
                return False
        
        # Otherwise, test all connected devices
        success = True
        for device in devices:
            if device.get("status") == "device":
                logger.info(f"Testing device: {device}")
                device_success = self._test_android_device(device.get("id"))
                success = success and device_success
        
        return success
    
    def _test_android_device(self, device_id: str) -> bool:
        """
        Test an Android device connectivity.
        
        Args:
            device_id: Device ID to test
            
        Returns:
            Success status
        """
        try:
            # Test ADB connection
            subprocess.run(
                ["adb", "-s", device_id, "shell", "echo", "Hello from device"],
                capture_output=True,
                check=True
            )
            
            # Test model loading (simplified test)
            logger.info("Running simplified benchmark test...")
            subprocess.run(
                [
                    sys.executable,
                    "test/android_test_harness/run_ci_benchmarks.py",
                    "--device-id", device_id,
                    "--output-db", "benchmark_results/verify_test.duckdb",
                    "--timeout", "60",
                    "--verbose"
                ],
                check=True
            )
            
            logger.info(f"Device {device_id} connectivity verified successfully")
            return True
        
        except subprocess.SubprocessError as e:
            logger.error(f"Error testing device {device_id}: {e}")
            return False
    
    def _verify_ios_connectivity(self) -> bool:
        """
        Verify iOS device connectivity.
        
        Returns:
            Success status
        """
        # iOS connectivity can only be verified on macOS
        if self.system != "Darwin":
            logger.error("iOS connectivity can only be verified on macOS")
            return False
        
        # Get connected devices and simulators
        devices = self._get_ios_devices()
        
        try:
            simctl_output = subprocess.run(
                ["xcrun", "simctl", "list"],
                capture_output=True,
                text=True,
                check=True
            )
            simulators = self._parse_ios_simulators(simctl_output.stdout)
        except subprocess.SubprocessError:
            simulators = []
        
        if not devices and not simulators:
            logger.error("No iOS devices or simulators found")
            return False
        
        # If device ID is specified, test that specific device or simulator
        if self.device_id:
            # Check if device ID refers to a physical device
            found_device = False
            for device in devices:
                if device.get("id") == self.device_id:
                    found_device = True
                    logger.info(f"Testing device: {device}")
                    return self._test_ios_device(device.get("id"), False)
            
            # Check if device ID refers to a simulator
            found_simulator = False
            for simulator in simulators:
                if simulator.get("id") == self.device_id:
                    found_simulator = True
                    logger.info(f"Testing simulator: {simulator}")
                    
                    # Boot simulator if not already booted
                    if simulator.get("state") != "Booted":
                        logger.info(f"Booting simulator: {self.device_id}")
                        try:
                            subprocess.run(
                                ["xcrun", "simctl", "boot", self.device_id],
                                check=True
                            )
                            logger.info("Simulator booted successfully")
                        except subprocess.SubprocessError as e:
                            logger.error(f"Error booting simulator: {e}")
                            return False
                    
                    return self._test_ios_device(simulator.get("id"), True)
            
            if not found_device and not found_simulator:
                logger.error(f"Device or simulator with ID '{self.device_id}' not found")
                return False
        
        # Otherwise, test a connected device or simulator
        if devices:
            # Test first connected device
            device = devices[0]
            logger.info(f"Testing device: {device}")
            return self._test_ios_device(device.get("id"), False)
        
        elif simulators:
            # Test first simulator
            for simulator in simulators:
                if simulator.get("state") == "Booted":
                    logger.info(f"Testing booted simulator: {simulator}")
                    return self._test_ios_device(simulator.get("id"), True)
            
            # No booted simulator found, boot the first one
            simulator = simulators[0]
            logger.info(f"Booting and testing simulator: {simulator}")
            
            try:
                subprocess.run(
                    ["xcrun", "simctl", "boot", simulator.get("id")],
                    check=True
                )
                logger.info("Simulator booted successfully")
                return self._test_ios_device(simulator.get("id"), True)
            
            except subprocess.SubprocessError as e:
                logger.error(f"Error booting simulator: {e}")
                return False
        
        logger.error("No suitable iOS device or simulator found")
        return False
    
    def _test_ios_device(self, device_id: str, is_simulator: bool) -> bool:
        """
        Test an iOS device connectivity.
        
        Args:
            device_id: Device ID to test
            is_simulator: Whether the device is a simulator
            
        Returns:
            Success status
        """
        try:
            # For real devices, test connection using xctrace
            if not is_simulator:
                subprocess.run(
                    ["xcrun", "xctrace", "list", "devices"],
                    capture_output=True,
                    check=True
                )
            
            # Test model loading (simplified test)
            logger.info("Running simplified benchmark test...")
            command = [
                sys.executable,
                "test/ios_test_harness/run_ci_benchmarks.py",
                "--device-id", device_id,
                "--output-db", "benchmark_results/verify_test.duckdb",
                "--timeout", "60",
                "--verbose"
            ]
            
            if is_simulator:
                command.append("--simulator")
            
            subprocess.run(command, check=True)
            
            logger.info(f"{'Simulator' if is_simulator else 'Device'} {device_id} connectivity verified successfully")
            return True
        
        except subprocess.SubprocessError as e:
            logger.error(f"Error testing {'simulator' if is_simulator else 'device'} {device_id}: {e}")
            return False
    
    def run(self, action: str) -> bool:
        """
        Run the specified action.
        
        Args:
            action: Action to perform (check, register, configure, verify)
            
        Returns:
            Success status
        """
        if action == "check":
            environment = self.check_environment()
            print(json.dumps(environment, indent=2))
            
            # Check if environment is ready
            platforms = environment.get("platforms", {})
            all_ready = True
            
            for platform_type, platform_status in platforms.items():
                if platform_status.get("status") != "ready":
                    logger.warning(f"Platform {platform_type} is not ready: {platform_status.get('status')}")
                    all_ready = False
            
            return all_ready
        
        elif action == "register":
            return self.register_runner()
        
        elif action == "configure":
            return self.configure_environment()
        
        elif action == "verify":
            return self.verify_device_connectivity()
        
        else:
            logger.error(f"Invalid action: {action}")
            logger.error("Action must be one of: check, register, configure, verify")
            return False


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Mobile CI Runners Setup Utility")
    
    parser.add_argument(
        "--action",
        choices=["check", "register", "configure", "verify"],
        default="check",
        help="Action to perform"
    )
    parser.add_argument(
        "--platform",
        choices=["android", "ios", "all"],
        default="all",
        help="Platform to target"
    )
    parser.add_argument(
        "--device-id",
        help="Device ID for specific device operations"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    try:
        # Create setup utility
        setup = MobileCIRunnerSetup(
            platform_type=args.platform,
            device_id=args.device_id,
            verbose=args.verbose
        )
        
        # Run the specified action
        success = setup.run(args.action)
        
        # Return exit code based on success
        return 0 if success else 1
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())