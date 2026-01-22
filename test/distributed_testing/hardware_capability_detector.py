#!/usr/bin/env python3
"""
Hardware Capability Detector for Distributed Testing Framework

This module provides comprehensive detection of hardware capabilities on worker nodes.
It integrates with the existing enhanced_hardware_capability.py system but provides
specialized functions for the distributed testing framework's needs, including:

1. Automated hardware detection on worker nodes
2. Database integration for capability storage
3. Hardware fingerprinting for unique identification
4. WebGPU/WebNN detection with browser automation support
5. DuckDB integration for optimization

Usage:
    detector = HardwareCapabilityDetector()
    capabilities = detector.detect_all_capabilities()
    detector.store_capabilities(capabilities)
"""

import os
import sys
import platform
import json
import logging
import subprocess
import uuid
import hashlib
import socket
import time
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path

import psutil
import duckdb

# Add parent directory to path for imports
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import from enhanced hardware capability module
try:
    # Try both import paths for flexibility
    try:
        from distributed_testing.enhanced_hardware_capability import (
            HardwareCapabilityDetector as BaseHardwareCapabilityDetector,
            HardwareType, HardwareVendor, PrecisionType, CapabilityScore,
            HardwareCapability, WorkerHardwareCapabilities
        )
    except ImportError:
        from test.distributed_testing.enhanced_hardware_capability import (
            HardwareCapabilityDetector as BaseHardwareCapabilityDetector,
            HardwareType, HardwareVendor, PrecisionType, CapabilityScore,
            HardwareCapability, WorkerHardwareCapabilities
        )
except ImportError:
    logging.error("Failed to import enhanced_hardware_capability. Using fallback implementation.")
    # Define minimal classes if import fails
    class HardwareType(Enum):
        CPU = "cpu"
        GPU = "gpu"
        TPU = "tpu"
        NPU = "npu"
        WEBGPU = "webgpu"
        WEBNN = "webnn"
        OTHER = "other"

    class HardwareVendor(Enum):
        INTEL = "intel"
        AMD = "amd"
        NVIDIA = "nvidia"
        APPLE = "apple"
        QUALCOMM = "qualcomm"
        UNKNOWN = "unknown"

    class PrecisionType(Enum):
        FP32 = "fp32"
        FP16 = "fp16"
        INT8 = "int8"
        INT4 = "int4"

    class CapabilityScore(Enum):
        EXCELLENT = 5
        GOOD = 4
        AVERAGE = 3
        BASIC = 2
        MINIMAL = 1
        UNKNOWN = 0

    @dataclass
    class HardwareCapability:
        hardware_type: HardwareType
        vendor: HardwareVendor = HardwareVendor.UNKNOWN
        model: str = "Unknown"
        version: Optional[str] = None
        driver_version: Optional[str] = None
        compute_units: Optional[int] = None
        cores: Optional[int] = None
        memory_gb: Optional[float] = None
        supported_precisions: List[PrecisionType] = field(default_factory=list)
        capabilities: Dict[str, Any] = field(default_factory=dict)
        scores: Dict[str, CapabilityScore] = field(default_factory=dict)
        
    @dataclass
    class WorkerHardwareCapabilities:
        worker_id: str
        os_type: str
        os_version: str
        hostname: str
        cpu_count: int
        total_memory_gb: float
        hardware_capabilities: List[HardwareCapability] = field(default_factory=list)
        last_updated: Optional[float] = None

    class BaseHardwareCapabilityDetector:
        """Fallback base detector class"""
        def __init__(self, worker_id=None):
            self.worker_id = worker_id or self._generate_worker_id()
            
        def _generate_worker_id(self):
            return f"worker_{uuid.uuid4().hex[:8]}"
        
        def detect_all_capabilities(self):
            # Minimal implementation
            return WorkerHardwareCapabilities(
                worker_id=self.worker_id,
                os_type=platform.system(),
                os_version=platform.version(),
                hostname=socket.gethostname(),
                cpu_count=psutil.cpu_count(logical=False),
                total_memory_gb=psutil.virtual_memory().total / (1024**3),
                hardware_capabilities=[],
                last_updated=time.time()
            )

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("hardware_capability_detector")


class HardwareCapabilityDetector(BaseHardwareCapabilityDetector):
    """
    Enhanced hardware capability detector for distributed testing framework.
    
    This class extends the base HardwareCapabilityDetector with:
    1. Database integration
    2. Fingerprinting for hardware identification
    3. WebGPU/WebNN detection with browser support
    4. Advanced browser capability detection
    5. Performance profiling
    6. Database-based storage and retrieval
    """
    
    def __init__(
        self, 
        worker_id: Optional[str] = None,
        db_path: Optional[str] = None,
        enable_browser_detection: bool = False,
        browser_executable_path: Optional[str] = None,
    ):
        """
        Initialize the hardware capability detector.
        
        Args:
            worker_id: Optional worker ID (will be auto-generated if not provided)
            db_path: Path to DuckDB database for storing results
            enable_browser_detection: Whether to enable browser-based detection
            browser_executable_path: Path to browser executable for automated detection
        """
        super().__init__(worker_id)
        
        self.db_path = db_path
        self.db_connection = None
        self.enable_browser_detection = enable_browser_detection
        self.browser_executable_path = browser_executable_path
        
        # Initialize database connection if path provided
        if db_path:
            self._init_database()
    
    def _init_database(self):
        """Initialize database connection and create tables if needed."""
        try:
            # Connect to database
            self.db_connection = duckdb.connect(self.db_path)
            
            # Create tables if they don't exist
            self._create_tables()
            
            logger.info(f"Database connection established to {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to initialize database: {str(e)}")
            self.db_connection = None
    
    def _create_tables(self):
        """Create necessary tables in the database."""
        if not self.db_connection:
            return
        
        try:
            # Create worker_hardware table
            self.db_connection.execute("""
                CREATE TABLE IF NOT EXISTS worker_hardware (
                    id INTEGER PRIMARY KEY,
                    worker_id VARCHAR,
                    hostname VARCHAR,
                    os_type VARCHAR,
                    os_version VARCHAR,
                    cpu_count INTEGER,
                    total_memory_gb FLOAT,
                    fingerprint VARCHAR,
                    last_updated TIMESTAMP,
                    metadata JSON
                )
            """)
            
            # Create hardware_capabilities table
            self.db_connection.execute("""
                CREATE TABLE IF NOT EXISTS hardware_capabilities (
                    id INTEGER PRIMARY KEY,
                    worker_id VARCHAR,
                    hardware_type VARCHAR,
                    vendor VARCHAR,
                    model VARCHAR,
                    version VARCHAR,
                    driver_version VARCHAR,
                    compute_units INTEGER,
                    cores INTEGER,
                    memory_gb FLOAT,
                    supported_precisions JSON,
                    capabilities JSON,
                    scores JSON,
                    last_updated TIMESTAMP
                )
            """)
            
            # Create hardware_performance table
            self.db_connection.execute("""
                CREATE TABLE IF NOT EXISTS hardware_performance (
                    id INTEGER PRIMARY KEY,
                    hardware_capability_id INTEGER,
                    benchmark_type VARCHAR,
                    metric_name VARCHAR,
                    metric_value FLOAT,
                    units VARCHAR,
                    run_date TIMESTAMP,
                    metadata JSON,
                    FOREIGN KEY (hardware_capability_id) REFERENCES hardware_capabilities(id)
                )
            """)
            
            logger.info("Database tables created/verified")
        except Exception as e:
            logger.error(f"Failed to create database tables: {str(e)}")
    
    def generate_hardware_fingerprint(self, capabilities: WorkerHardwareCapabilities) -> str:
        """
        Generate a unique fingerprint for the hardware configuration.
        
        Args:
            capabilities: Hardware capabilities to fingerprint
            
        Returns:
            Unique hardware fingerprint string
        """
        # Create a dictionary with essential hardware info
        fingerprint_data = {
            "hostname": capabilities.hostname,
            "os_type": capabilities.os_type,
            "os_version": capabilities.os_version,
            "cpu_count": capabilities.cpu_count,
            "total_memory_gb": round(capabilities.total_memory_gb, 2),
            "hardware": []
        }
        
        # Add each hardware component
        for hw in capabilities.hardware_capabilities:
            hw_info = {
                "type": hw.hardware_type.value,
                "vendor": hw.vendor.value,
                "model": hw.model,
                "memory_gb": hw.memory_gb
            }
            fingerprint_data["hardware"].append(hw_info)
        
        # Sort to ensure consistency
        fingerprint_data["hardware"].sort(key=lambda x: (x["type"], x["vendor"], x["model"]))
        
        # Create fingerprint
        fingerprint_json = json.dumps(fingerprint_data, sort_keys=True)
        fingerprint = hashlib.sha256(fingerprint_json.encode()).hexdigest()
        
        return fingerprint
    
    def detect_webgpu_capabilities(self) -> Optional[HardwareCapability]:
        """
        Detect WebGPU capabilities with browser automation support.
        
        Returns:
            HardwareCapability for WebGPU or None if not available
        """
        # Skip browser detection if disabled
        if not self.enable_browser_detection:
            logger.info("WebGPU detection skipped (browser detection disabled)")
            return None
        
        try:
            # Check for Selenium and browser driver
            import selenium
            from selenium import webdriver
            
            # Choose browser driver based on availability
            browser = None
            
            # Check for custom executable path
            if self.browser_executable_path:
                if "chrome" in self.browser_executable_path.lower():
                    browser = "chrome"
                elif "firefox" in self.browser_executable_path.lower():
                    browser = "firefox"
                elif "edge" in self.browser_executable_path.lower() or "msedge" in self.browser_executable_path.lower():
                    browser = "edge"
            
            # If no browser specified, try to detect
            if not browser:
                # Try Chrome first
                try:
                    from selenium.webdriver.chrome.service import Service as ChromeService
                    from webdriver_manager.chrome import ChromeDriverManager
                    
                    # Use WebDriver Manager to automatically download the appropriate driver
                    chrome_service = ChromeService(ChromeDriverManager().install())
                    driver = webdriver.Chrome(service=chrome_service)
                    browser = "chrome"
                    logger.info("Using Chrome for WebGPU detection")
                except Exception as e:
                    logger.warning(f"Chrome WebDriver not available: {str(e)}")
                    
                    # Try Firefox next
                    try:
                        from selenium.webdriver.firefox.service import Service as FirefoxService
                        from webdriver_manager.firefox import GeckoDriverManager
                        
                        firefox_service = FirefoxService(GeckoDriverManager().install())
                        driver = webdriver.Firefox(service=firefox_service)
                        browser = "firefox"
                        logger.info("Using Firefox for WebGPU detection")
                    except Exception as e:
                        logger.warning(f"Firefox WebDriver not available: {str(e)}")
                        
                        # Try Edge as last resort
                        try:
                            from selenium.webdriver.edge.service import Service as EdgeService
                            from webdriver_manager.microsoft import EdgeChromiumDriverManager
                            
                            edge_service = EdgeService(EdgeChromiumDriverManager().install())
                            driver = webdriver.Edge(service=edge_service)
                            browser = "edge"
                            logger.info("Using Edge for WebGPU detection")
                        except Exception as e:
                            logger.warning(f"Edge WebDriver not available: {str(e)}")
                            logger.error("No supported browser found for WebGPU detection")
                            return None
            
            # If browser was specified by executable path, initialize it
            if browser and not 'driver' in locals():
                if browser == "chrome":
                    from selenium.webdriver.chrome.service import Service as ChromeService
                    from selenium.webdriver.chrome.options import Options as ChromeOptions
                    
                    chrome_options = ChromeOptions()
                    chrome_options.binary_location = self.browser_executable_path
                    chrome_service = ChromeService()
                    driver = webdriver.Chrome(service=chrome_service, options=chrome_options)
                
                elif browser == "firefox":
                    from selenium.webdriver.firefox.service import Service as FirefoxService
                    from selenium.webdriver.firefox.options import Options as FirefoxOptions
                    
                    firefox_options = FirefoxOptions()
                    firefox_options.binary_location = self.browser_executable_path
                    firefox_service = FirefoxService()
                    driver = webdriver.Firefox(service=firefox_service, options=firefox_options)
                
                elif browser == "edge":
                    from selenium.webdriver.edge.service import Service as EdgeService
                    from selenium.webdriver.edge.options import Options as EdgeOptions
                    
                    edge_options = EdgeOptions()
                    edge_options.binary_location = self.browser_executable_path
                    edge_service = EdgeService()
                    driver = webdriver.Edge(service=edge_service, options=edge_options)
            
            # Create WebGPU detection script
            webgpu_detection_script = """
            // Function to detect WebGPU capabilities
            async function detectWebGPU() {
                const results = {
                    isAvailable: false,
                    adapterInfo: null,
                    features: [],
                    limits: null,
                    error: null
                };
                
                try {
                    // Check if WebGPU is supported
                    if (!navigator.gpu) {
                        results.error = "WebGPU not supported in this browser";
                        return results;
                    }
                    
                    // Request adapter
                    const adapter = await navigator.gpu.requestAdapter();
                    if (!adapter) {
                        results.error = "WebGPU adapter not available";
                        return results;
                    }
                    
                    // Get adapter info
                    results.isAvailable = true;
                    results.adapterInfo = await adapter.requestAdapterInfo();
                    
                    // Get supported features
                    results.features = Array.from(adapter.features).map(feature => feature.toString());
                    
                    // Get limits
                    results.limits = {};
                    for (const limit in adapter.limits) {
                        results.limits[limit] = adapter.limits[limit];
                    }
                    
                    // Make a test device to confirm it works
                    const device = await adapter.requestDevice();
                    if (device) {
                        results.deviceCreated = true;
                    }
                    
                } catch (e) {
                    results.error = e.toString();
                }
                
                return results;
            }
            
            // Run detection and return promise
            return detectWebGPU();
            """
            
            # Set up a specific URL to bypass security restrictions on local file access
            driver.get("https://webgpureport.org/")
            
            # Wait for the page to load
            from selenium.webdriver.support.ui import WebDriverWait
            from selenium.webdriver.support import expected_conditions as EC
            from selenium.webdriver.common.by import By
            
            # Wait for page to be ready
            wait = WebDriverWait(driver, 10)
            wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
            
            # Execute WebGPU detection script
            detection_result = driver.execute_async_script(f"""
                var callback = arguments[arguments.length - 1];
                {webgpu_detection_script}
                    .then(result => callback(result))
                    .catch(error => callback({{ error: error.toString() }}));
            """)
            
            # Close the browser
            driver.quit()
            
            # Parse detection results
            if detection_result.get('isAvailable', False):
                # WebGPU is available
                adapter_info = detection_result.get('adapterInfo', {})
                features = detection_result.get('features', [])
                limits = detection_result.get('limits', {})
                
                # Extract vendor information from adapter info
                vendor = HardwareVendor.UNKNOWN
                vendor_str = adapter_info.get('vendor', '').lower()
                
                if 'nvidia' in vendor_str:
                    vendor = HardwareVendor.NVIDIA
                elif 'amd' in vendor_str or 'ati' in vendor_str:
                    vendor = HardwareVendor.AMD
                elif 'intel' in vendor_str:
                    vendor = HardwareVendor.INTEL
                elif 'apple' in vendor_str:
                    vendor = HardwareVendor.APPLE
                
                # Create capability object
                gpu_capability = HardwareCapability(
                    hardware_type=HardwareType.WEBGPU,
                    vendor=vendor,
                    model=adapter_info.get('description', f"{browser.capitalize()} WebGPU"),
                    version=adapter_info.get('architecture', None),
                    driver_version=adapter_info.get('driver', None),
                    memory_gb=limits.get('maxBufferSize', 0) / (1024 ** 3) if limits else None,
                    supported_precisions=[
                        PrecisionType.FP32,
                        PrecisionType.FP16 if 'float16' in features else None,
                        PrecisionType.INT8 if 'texture-compression-bc' in features else None
                    ],
                    capabilities={
                        'browser': browser,
                        'features': features,
                        'limits': limits,
                        'adapter_info': adapter_info
                    }
                )
                
                # Filter None values from supported precisions
                gpu_capability.supported_precisions = [p for p in gpu_capability.supported_precisions if p is not None]
                
                logger.info(f"Detected WebGPU capability in {browser}: {adapter_info.get('description', 'Unknown')}")
                return gpu_capability
            else:
                # WebGPU not available
                logger.info(f"WebGPU not available in {browser}: {detection_result.get('error', 'Unknown error')}")
                return None
        
        except ImportError:
            logger.warning("Selenium not installed, cannot perform browser-based WebGPU detection")
            return None
        
        except Exception as e:
            logger.error(f"Error during WebGPU detection: {str(e)}")
            return None
    
    def detect_webnn_capabilities(self) -> Optional[HardwareCapability]:
        """
        Detect WebNN capabilities with browser automation support.
        
        Returns:
            HardwareCapability for WebNN or None if not available
        """
        # Skip browser detection if disabled
        if not self.enable_browser_detection:
            logger.info("WebNN detection skipped (browser detection disabled)")
            return None
        
        try:
            # Reuse browser detection logic from WebGPU function
            import selenium
            from selenium import webdriver
            
            # Choose browser driver based on availability
            browser = None
            
            # Check for custom executable path
            if self.browser_executable_path:
                if "chrome" in self.browser_executable_path.lower():
                    browser = "chrome"
                elif "edge" in self.browser_executable_path.lower() or "msedge" in self.browser_executable_path.lower():
                    browser = "edge"
                elif "firefox" in self.browser_executable_path.lower():
                    browser = "firefox"
            
            # If no browser specified, try to detect
            if not browser:
                # Try Edge first (best WebNN support)
                try:
                    from selenium.webdriver.edge.service import Service as EdgeService
                    from webdriver_manager.microsoft import EdgeChromiumDriverManager
                    
                    edge_service = EdgeService(EdgeChromiumDriverManager().install())
                    driver = webdriver.Edge(service=edge_service)
                    browser = "edge"
                    logger.info("Using Edge for WebNN detection")
                except Exception as e:
                    logger.warning(f"Edge WebDriver not available: {str(e)}")
                    
                    # Try Chrome next
                    try:
                        from selenium.webdriver.chrome.service import Service as ChromeService
                        from webdriver_manager.chrome import ChromeDriverManager
                        
                        chrome_service = ChromeService(ChromeDriverManager().install())
                        driver = webdriver.Chrome(service=chrome_service)
                        browser = "chrome"
                        logger.info("Using Chrome for WebNN detection")
                    except Exception as e:
                        logger.warning(f"Chrome WebDriver not available: {str(e)}")
                        
                        # Try Firefox as last resort
                        try:
                            from selenium.webdriver.firefox.service import Service as FirefoxService
                            from webdriver_manager.firefox import GeckoDriverManager
                            
                            firefox_service = FirefoxService(GeckoDriverManager().install())
                            driver = webdriver.Firefox(service=firefox_service)
                            browser = "firefox"
                            logger.info("Using Firefox for WebNN detection")
                        except Exception as e:
                            logger.warning(f"Firefox WebDriver not available: {str(e)}")
                            logger.error("No supported browser found for WebNN detection")
                            return None
            
            # If browser was specified by executable path, initialize it
            if browser and not 'driver' in locals():
                if browser == "chrome":
                    from selenium.webdriver.chrome.service import Service as ChromeService
                    from selenium.webdriver.chrome.options import Options as ChromeOptions
                    
                    chrome_options = ChromeOptions()
                    chrome_options.binary_location = self.browser_executable_path
                    chrome_service = ChromeService()
                    driver = webdriver.Chrome(service=chrome_service, options=chrome_options)
                
                elif browser == "firefox":
                    from selenium.webdriver.firefox.service import Service as FirefoxService
                    from selenium.webdriver.firefox.options import Options as FirefoxOptions
                    
                    firefox_options = FirefoxOptions()
                    firefox_options.binary_location = self.browser_executable_path
                    firefox_service = FirefoxService()
                    driver = webdriver.Firefox(service=firefox_service, options=firefox_options)
                
                elif browser == "edge":
                    from selenium.webdriver.edge.service import Service as EdgeService
                    from selenium.webdriver.edge.options import Options as EdgeOptions
                    
                    edge_options = EdgeOptions()
                    edge_options.binary_location = self.browser_executable_path
                    edge_service = EdgeService()
                    driver = webdriver.Edge(service=edge_service, options=edge_options)
            
            # Create WebNN detection script
            webnn_detection_script = """
            // Function to detect WebNN capabilities
            async function detectWebNN() {
                const results = {
                    isAvailable: false,
                    device: null,
                    supportedOperations: [],
                    error: null
                };
                
                try {
                    // Check if WebNN is supported
                    if (!('ml' in navigator)) {
                        results.error = "WebNN not supported in this browser";
                        return results;
                    }
                    
                    // List available devices
                    results.devices = [];
                    
                    // Check if CPU is available
                    try {
                        const cpuContext = await navigator.ml.createContext({ deviceType: 'cpu' });
                        if (cpuContext) {
                            results.devices.push({
                                type: 'cpu',
                                available: true
                            });
                        }
                    } catch (e) {
                        results.devices.push({
                            type: 'cpu',
                            available: false,
                            error: e.toString()
                        });
                    }
                    
                    // Check if GPU is available
                    try {
                        const gpuContext = await navigator.ml.createContext({ deviceType: 'gpu' });
                        if (gpuContext) {
                            results.devices.push({
                                type: 'gpu',
                                available: true
                            });
                        }
                    } catch (e) {
                        results.devices.push({
                            type: 'gpu',
                            available: false,
                            error: e.toString()
                        });
                    }
                    
                    // Set availability based on at least one device being available
                    results.isAvailable = results.devices.some(device => device.available);
                    
                    // Test basic operations to see what's supported
                    if (results.isAvailable) {
                        try {
                            const context = await navigator.ml.createContext({
                                deviceType: results.devices.find(d => d.available)?.type || 'cpu'
                            });
                            
                            // Test basic operations
                            const opTests = {
                                'add': false,
                                'sub': false,
                                'mul': false,
                                'matmul': false,
                                'conv2d': false,
                                'relu': false,
                                'softmax': false,
                                'pool2d': false
                            };
                            
                            // Create tensors for testing
                            const builder = new MLGraphBuilder(context);
                            const a = builder.input('a', {dataType: 'float32', dimensions: [1, 3]});
                            const b = builder.input('b', {dataType: 'float32', dimensions: [1, 3]});
                            
                            try { builder.add(a, b); opTests.add = true; } catch (e) {}
                            try { builder.sub(a, b); opTests.sub = true; } catch (e) {}
                            try { builder.mul(a, b); opTests.mul = true; } catch (e) {}
                            
                            // Add successful operations to results
                            results.supportedOperations = Object.entries(opTests)
                                .filter(([_, supported]) => supported)
                                .map(([op, _]) => op);
                        } catch (e) {
                            results.operationTestError = e.toString();
                        }
                    }
                    
                } catch (e) {
                    results.error = e.toString();
                }
                
                return results;
            }
            
            // Run detection and return promise
            return detectWebNN();
            """
            
            # Set up a specific URL for detection
            driver.get("https://webnn.dev/")
            
            # Wait for the page to load
            from selenium.webdriver.support.ui import WebDriverWait
            from selenium.webdriver.support import expected_conditions as EC
            from selenium.webdriver.common.by import By
            
            # Wait for page to be ready
            wait = WebDriverWait(driver, 10)
            wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
            
            # Execute WebNN detection script
            detection_result = driver.execute_async_script(f"""
                var callback = arguments[arguments.length - 1];
                {webnn_detection_script}
                    .then(result => callback(result))
                    .catch(error => callback({{ error: error.toString() }}));
            """)
            
            # Close the browser
            driver.quit()
            
            # Parse detection results
            if detection_result.get('isAvailable', False):
                # WebNN is available
                devices = detection_result.get('devices', [])
                supported_operations = detection_result.get('supportedOperations', [])
                
                # Create capability object
                webnn_capability = HardwareCapability(
                    hardware_type=HardwareType.WEBNN,
                    vendor=HardwareVendor.UNKNOWN,  # Not directly available
                    model=f"{browser.capitalize()} WebNN",
                    version=None,  # Not directly available
                    supported_precisions=[
                        PrecisionType.FP32,  # Always supported
                        PrecisionType.FP16  # May be supported
                    ],
                    capabilities={
                        'browser': browser,
                        'supported_devices': devices,
                        'supported_operations': supported_operations
                    }
                )
                
                logger.info(f"Detected WebNN capability in {browser} with {len(supported_operations)} supported operations")
                return webnn_capability
            else:
                # WebNN not available
                logger.info(f"WebNN not available in {browser}: {detection_result.get('error', 'Unknown error')}")
                return None
        
        except ImportError:
            logger.warning("Selenium not installed, cannot perform browser-based WebNN detection")
            return None
        
        except Exception as e:
            logger.error(f"Error during WebNN detection: {str(e)}")
            return None
    
    def store_capabilities(self, capabilities: WorkerHardwareCapabilities) -> bool:
        """
        Store hardware capabilities in the database.
        
        Args:
            capabilities: Hardware capabilities to store
            
        Returns:
            True if stored successfully, False otherwise
        """
        if not self.db_connection:
            logger.warning("No database connection, cannot store capabilities")
            return False
        
        try:
            # Generate hardware fingerprint
            fingerprint = self.generate_hardware_fingerprint(capabilities)
            
            # Store worker hardware information
            self.db_connection.execute("""
                INSERT INTO worker_hardware (
                    worker_id, hostname, os_type, os_version, 
                    cpu_count, total_memory_gb, fingerprint, last_updated, metadata
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                capabilities.worker_id,
                capabilities.hostname,
                capabilities.os_type,
                capabilities.os_version,
                capabilities.cpu_count,
                capabilities.total_memory_gb,
                fingerprint,
                datetime.now(),
                json.dumps(getattr(capabilities, 'metadata', {}))
            ])
            
            # Store each hardware capability
            for hw in capabilities.hardware_capabilities:
                # Convert enums to strings
                hardware_type = hw.hardware_type.value if isinstance(hw.hardware_type, Enum) else hw.hardware_type
                vendor = hw.vendor.value if isinstance(hw.vendor, Enum) else hw.vendor
                
                # Convert supported precisions to list of strings
                supported_precisions = [p.value if isinstance(p, Enum) else p for p in hw.supported_precisions]
                
                # Convert scores dictionary
                scores = {k: v.value if isinstance(v, Enum) else v for k, v in hw.scores.items()}
                
                # Insert hardware capability
                self.db_connection.execute("""
                    INSERT INTO hardware_capabilities (
                        worker_id, hardware_type, vendor, model, version, driver_version,
                        compute_units, cores, memory_gb, supported_precisions, capabilities, scores,
                        last_updated
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    capabilities.worker_id,
                    hardware_type,
                    vendor,
                    hw.model,
                    hw.version,
                    hw.driver_version,
                    hw.compute_units,
                    hw.cores,
                    hw.memory_gb,
                    json.dumps(supported_precisions),
                    json.dumps(hw.capabilities),
                    json.dumps(scores),
                    datetime.now()
                ])
            
            logger.info(f"Stored hardware capabilities for worker {capabilities.worker_id} in database")
            return True
        
        except Exception as e:
            logger.error(f"Failed to store capabilities in database: {str(e)}")
            return False
    
    def get_worker_capabilities(self, worker_id: str) -> Optional[WorkerHardwareCapabilities]:
        """
        Retrieve worker capabilities from the database.
        
        Args:
            worker_id: Worker ID to retrieve capabilities for
            
        Returns:
            WorkerHardwareCapabilities or None if not found
        """
        if not self.db_connection:
            logger.warning("No database connection, cannot retrieve capabilities")
            return None
        
        try:
            # Get worker hardware information
            worker_result = self.db_connection.execute("""
                SELECT 
                    worker_id, hostname, os_type, os_version, 
                    cpu_count, total_memory_gb, fingerprint, last_updated, metadata
                FROM worker_hardware
                WHERE worker_id = ?
                ORDER BY last_updated DESC
                LIMIT 1
            """, [worker_id]).fetchone()
            
            if not worker_result:
                logger.warning(f"No hardware information found for worker {worker_id}")
                return None
            
            # Get hardware capabilities
            hw_results = self.db_connection.execute("""
                SELECT 
                    hardware_type, vendor, model, version, driver_version,
                    compute_units, cores, memory_gb, supported_precisions, capabilities, scores
                FROM hardware_capabilities
                WHERE worker_id = ?
                ORDER BY last_updated DESC
            """, [worker_id]).fetchall()
            
            # Create worker capabilities object
            capabilities = WorkerHardwareCapabilities(
                worker_id=worker_result[0],
                hostname=worker_result[1],
                os_type=worker_result[2],
                os_version=worker_result[3],
                cpu_count=worker_result[4],
                total_memory_gb=worker_result[5],
                hardware_capabilities=[],
                last_updated=worker_result[7].timestamp() if worker_result[7] else None
            )
            
            # Add metadata if available
            if worker_result[8]:
                try:
                    capabilities.metadata = json.loads(worker_result[8])
                except json.JSONDecodeError:
                    pass
            
            # Process hardware capabilities
            for hw_result in hw_results:
                # Convert hardware type and vendor to enums
                try:
                    hardware_type = HardwareType(hw_result[0])
                except (ValueError, TypeError):
                    hardware_type = HardwareType.OTHER
                
                try:
                    vendor = HardwareVendor(hw_result[1])
                except (ValueError, TypeError):
                    vendor = HardwareVendor.UNKNOWN
                
                # Convert supported precisions
                supported_precisions = []
                if hw_result[8]:
                    try:
                        precision_strings = json.loads(hw_result[8])
                        for p_str in precision_strings:
                            try:
                                supported_precisions.append(PrecisionType(p_str))
                            except (ValueError, TypeError):
                                pass
                    except json.JSONDecodeError:
                        pass
                
                # Convert capabilities and scores
                capabilities_dict = {}
                if hw_result[9]:
                    try:
                        capabilities_dict = json.loads(hw_result[9])
                    except json.JSONDecodeError:
                        pass
                
                scores_dict = {}
                if hw_result[10]:
                    try:
                        scores_json = json.loads(hw_result[10])
                        for score_type, score_value in scores_json.items():
                            try:
                                scores_dict[score_type] = CapabilityScore(score_value)
                            except (ValueError, TypeError):
                                scores_dict[score_type] = CapabilityScore.UNKNOWN
                    except json.JSONDecodeError:
                        pass
                
                # Create hardware capability object
                hw_capability = HardwareCapability(
                    hardware_type=hardware_type,
                    vendor=vendor,
                    model=hw_result[2],
                    version=hw_result[3],
                    driver_version=hw_result[4],
                    compute_units=hw_result[5],
                    cores=hw_result[6],
                    memory_gb=hw_result[7],
                    supported_precisions=supported_precisions,
                    capabilities=capabilities_dict,
                    scores=scores_dict
                )
                
                # Add to capabilities list
                capabilities.hardware_capabilities.append(hw_capability)
            
            logger.info(f"Retrieved hardware capabilities for worker {worker_id} with {len(capabilities.hardware_capabilities)} hardware components")
            return capabilities
        
        except Exception as e:
            logger.error(f"Failed to retrieve capabilities from database: {str(e)}")
            return None
    
    def get_workers_by_hardware_type(self, hardware_type: Union[HardwareType, str]) -> List[str]:
        """
        Get worker IDs that have a specific hardware type.
        
        Args:
            hardware_type: Hardware type to search for
            
        Returns:
            List of worker IDs with the specified hardware type
        """
        if not self.db_connection:
            logger.warning("No database connection, cannot search workers by hardware type")
            return []
        
        try:
            # Convert hardware type to string if it's an enum
            hw_type_str = hardware_type.value if isinstance(hardware_type, Enum) else hardware_type
            
            # Query database
            results = self.db_connection.execute("""
                SELECT DISTINCT worker_id
                FROM hardware_capabilities
                WHERE hardware_type = ?
            """, [hw_type_str]).fetchall()
            
            # Extract worker IDs
            worker_ids = [row[0] for row in results]
            
            logger.info(f"Found {len(worker_ids)} workers with hardware type {hw_type_str}")
            return worker_ids
        
        except Exception as e:
            logger.error(f"Failed to search workers by hardware type: {str(e)}")
            return []
    
    def find_compatible_workers(self, 
                              hardware_requirements: Dict[str, Any],
                              min_memory_gb: Optional[float] = None,
                              preferred_hardware_types: Optional[List[Union[HardwareType, str]]] = None) -> List[str]:
        """
        Find workers that are compatible with the given hardware requirements.
        
        Args:
            hardware_requirements: Dictionary of hardware requirements
            min_memory_gb: Minimum memory requirement in GB
            preferred_hardware_types: List of preferred hardware types in order of preference
            
        Returns:
            List of compatible worker IDs
        """
        if not self.db_connection:
            logger.warning("No database connection, cannot find compatible workers")
            return []
        
        try:
            # Base query to join worker_hardware and hardware_capabilities
            query = """
                SELECT DISTINCT h.worker_id, w.hostname
                FROM hardware_capabilities h
                JOIN worker_hardware w ON h.worker_id = w.worker_id
                WHERE 1=1
            """
            
            params = []
            
            # Add hardware type filter if specified
            if 'hardware_type' in hardware_requirements:
                hw_type = hardware_requirements['hardware_type']
                hw_type_str = hw_type.value if isinstance(hw_type, Enum) else hw_type
                query += " AND h.hardware_type = ?"
                params.append(hw_type_str)
            
            # Add vendor filter if specified
            if 'vendor' in hardware_requirements:
                vendor = hardware_requirements['vendor']
                vendor_str = vendor.value if isinstance(vendor, Enum) else vendor
                query += " AND h.vendor = ?"
                params.append(vendor_str)
            
            # Add memory filter if specified
            if min_memory_gb is not None:
                query += " AND h.memory_gb >= ?"
                params.append(min_memory_gb)
            
            # Execute query
            results = self.db_connection.execute(query, params).fetchall()
            
            # Create worker ID list
            worker_ids = [row[0] for row in results]
            
            # Sort by preferred hardware types if specified
            if preferred_hardware_types and worker_ids:
                # Convert preferred hardware types to strings
                preferred_hw_strs = []
                for hw_type in preferred_hardware_types:
                    if isinstance(hw_type, Enum):
                        preferred_hw_strs.append(hw_type.value)
                    else:
                        preferred_hw_strs.append(hw_type)
                
                # Group workers by hardware type
                workers_by_hw_type = {}
                for worker_id in worker_ids:
                    worker_hw_results = self.db_connection.execute("""
                        SELECT hardware_type
                        FROM hardware_capabilities
                        WHERE worker_id = ?
                    """, [worker_id]).fetchall()
                    
                    for hw_type in [row[0] for row in worker_hw_results]:
                        if hw_type not in workers_by_hw_type:
                            workers_by_hw_type[hw_type] = []
                        workers_by_hw_type[hw_type].append(worker_id)
                
                # Sort workers by preferred hardware types
                sorted_worker_ids = []
                for hw_type in preferred_hw_strs:
                    if hw_type in workers_by_hw_type:
                        sorted_worker_ids.extend(workers_by_hw_type[hw_type])
                
                # Add any remaining workers that weren't in the preferred list
                for worker_id in worker_ids:
                    if worker_id not in sorted_worker_ids:
                        sorted_worker_ids.append(worker_id)
                
                worker_ids = sorted_worker_ids
            
            logger.info(f"Found {len(worker_ids)} compatible workers for the given requirements")
            return worker_ids
        
        except Exception as e:
            logger.error(f"Failed to find compatible workers: {str(e)}")
            return []
    
    def perform_hardware_profiling(self, 
                                worker_id: str,
                                hardware_type: Union[HardwareType, str],
                                benchmark_type: str = "basic") -> Dict[str, Any]:
        """
        Perform hardware profiling benchmarks.
        
        Args:
            worker_id: Worker ID to profile
            hardware_type: Hardware type to profile
            benchmark_type: Type of benchmark to perform ("basic", "compute", "memory", "full")
            
        Returns:
            Dictionary with benchmark results
        """
        # For now, this is a stub implementation
        # In a real implementation, this would execute various benchmarks on the worker
        
        logger.info(f"Hardware profiling not fully implemented - would profile {hardware_type} on worker {worker_id}")
        
        # Mock benchmark results
        return {
            "worker_id": worker_id,
            "hardware_type": hardware_type.value if isinstance(hardware_type, Enum) else hardware_type,
            "benchmark_type": benchmark_type,
            "timestamp": datetime.now().isoformat(),
            "metrics": {
                "compute_score": random.uniform(100, 1000),
                "memory_bandwidth_gbps": random.uniform(10, 100),
                "latency_ms": random.uniform(1, 10)
            }
        }
    
    def detect_all_capabilities_with_browsers(self) -> WorkerHardwareCapabilities:
        """
        Detect all hardware capabilities including browser-based capabilities.
        This is an extended version of detect_all_capabilities that includes
        browser-based detection of WebGPU and WebNN.
        
        Returns:
            WorkerHardwareCapabilities with all detected capabilities
        """
        # Start with basic detection
        capabilities = self.detect_all_capabilities()
        
        # Add browser-specific capabilities if enabled
        if self.enable_browser_detection:
            # Detect WebGPU
            webgpu_capability = self.detect_webgpu_capabilities()
            if webgpu_capability:
                capabilities.hardware_capabilities.append(webgpu_capability)
            
            # Detect WebNN
            webnn_capability = self.detect_webnn_capabilities()
            if webnn_capability:
                capabilities.hardware_capabilities.append(webnn_capability)
        
        return capabilities


def main():
    """Main function for standalone execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Hardware Capability Detector for Distributed Testing Framework")
    parser.add_argument("--worker-id", help="Worker ID (default: auto-generated)")
    parser.add_argument("--db-path", help="Path to DuckDB database for storing results")
    parser.add_argument("--enable-browser-detection", action="store_true", help="Enable browser-based WebGPU/WebNN detection")
    parser.add_argument("--browser-path", help="Path to browser executable for automated detection")
    parser.add_argument("--detect-only", action="store_true", help="Only detect capabilities, don't store in database")
    parser.add_argument("--output-json", help="Path to output JSON file for capabilities")
    parser.add_argument("--search-workers", help="Search for workers with specific hardware type")
    parser.add_argument("--find-compatible", help="Find workers compatible with specific requirements (json string)")
    parser.add_argument("--profile-hardware", help="Perform hardware profiling for specific worker and hardware type (format: worker_id:hardware_type)")
    
    args = parser.parse_args()
    
    # Create detector
    detector = HardwareCapabilityDetector(
        worker_id=args.worker_id,
        db_path=args.db_path,
        enable_browser_detection=args.enable_browser_detection,
        browser_executable_path=args.browser_path
    )
    
    if args.search_workers:
        # Search for workers with specific hardware type
        worker_ids = detector.get_workers_by_hardware_type(args.search_workers)
        print(f"Found {len(worker_ids)} workers with hardware type {args.search_workers}:")
        for worker_id in worker_ids:
            print(f"  - {worker_id}")
    
    elif args.find_compatible:
        # Find compatible workers
        try:
            requirements = json.loads(args.find_compatible)
            min_memory_gb = requirements.pop("min_memory_gb", None)
            preferred_hardware_types = requirements.pop("preferred_hardware_types", None)
            
            worker_ids = detector.find_compatible_workers(
                requirements, min_memory_gb, preferred_hardware_types
            )
            
            print(f"Found {len(worker_ids)} compatible workers:")
            for worker_id in worker_ids:
                print(f"  - {worker_id}")
        
        except json.JSONDecodeError:
            print("Error: Invalid JSON for compatibility requirements")
    
    elif args.profile_hardware:
        # Perform hardware profiling
        try:
            worker_id, hardware_type = args.profile_hardware.split(":")
            results = detector.perform_hardware_profiling(worker_id, hardware_type)
            print(f"Profiling results: {results}")
        
        except ValueError:
            print("Error: Invalid format for profile-hardware parameter (use worker_id:hardware_type)")
    
    else:
        # Default: detect capabilities
        method = "detect_all_capabilities_with_browsers" if args.enable_browser_detection else "detect_all_capabilities"
        capabilities = getattr(detector, method)()
        
        # Store in database if requested
        if not args.detect_only and args.db_path:
            detector.store_capabilities(capabilities)
        
        # Output capabilities info
        print(f"\nWorker ID: {capabilities.worker_id}")
        print(f"Hostname: {capabilities.hostname}")
        print(f"OS: {capabilities.os_type} {capabilities.os_version}")
        print(f"CPU Count: {capabilities.cpu_count}")
        print(f"Total Memory: {capabilities.total_memory_gb:.2f} GB")
        print(f"Detected {len(capabilities.hardware_capabilities)} hardware capabilities")
        
        # Output each hardware capability
        for idx, hw in enumerate(capabilities.hardware_capabilities):
            hw_type = hw.hardware_type.name if isinstance(hw.hardware_type, Enum) else hw.hardware_type
            vendor = hw.vendor.name if isinstance(hw.vendor, Enum) else hw.vendor
            
            print(f"\n  Capability {idx+1}: {hw_type} - {hw.model}")
            print(f"    Vendor: {vendor}")
            if hw.memory_gb:
                print(f"    Memory: {hw.memory_gb:.2f} GB")
            
            # Print precision support
            precisions = [p.name if isinstance(p, Enum) else p for p in hw.supported_precisions]
            if precisions:
                print(f"    Supported Precisions: {', '.join(precisions)}")
            
            # Print scores
            if hw.scores:
                print("    Scores:")
                for score_type, score in hw.scores.items():
                    score_name = score.name if isinstance(score, Enum) else score
                    print(f"      {score_type}: {score_name}")
            
            # Print additional capability details
            if "browser" in hw.capabilities:
                print(f"    Browser: {hw.capabilities['browser']}")
        
        # Output to JSON file if requested
        if args.output_json:
            try:
                # Convert capabilities to dictionary for JSON serialization
                capabilities_dict = {
                    "worker_id": capabilities.worker_id,
                    "hostname": capabilities.hostname,
                    "os_type": capabilities.os_type,
                    "os_version": capabilities.os_version,
                    "cpu_count": capabilities.cpu_count,
                    "total_memory_gb": capabilities.total_memory_gb,
                    "hardware_capabilities": [],
                    "last_updated": datetime.now().isoformat()
                }
                
                # Convert hardware capabilities
                for hw in capabilities.hardware_capabilities:
                    hw_type = hw.hardware_type.value if isinstance(hw.hardware_type, Enum) else hw.hardware_type
                    vendor = hw.vendor.value if isinstance(hw.vendor, Enum) else hw.vendor
                    
                    # Convert precisions
                    precisions = [p.value if isinstance(p, Enum) else p for p in hw.supported_precisions]
                    
                    # Convert scores
                    scores = {k: v.value if isinstance(v, Enum) else v for k, v in hw.scores.items()}
                    
                    # Create hardware capability dict
                    hw_dict = {
                        "hardware_type": hw_type,
                        "vendor": vendor,
                        "model": hw.model,
                        "version": hw.version,
                        "driver_version": hw.driver_version,
                        "compute_units": hw.compute_units,
                        "cores": hw.cores,
                        "memory_gb": hw.memory_gb,
                        "supported_precisions": precisions,
                        "capabilities": hw.capabilities,
                        "scores": scores
                    }
                    
                    capabilities_dict["hardware_capabilities"].append(hw_dict)
                
                # Write to JSON file
                with open(args.output_json, 'w') as f:
                    json.dump(capabilities_dict, f, indent=2)
                
                print(f"\nCapabilities written to {args.output_json}")
                
            except Exception as e:
                print(f"\nError writing to JSON file: {str(e)}")


if __name__ == "__main__":
    main()