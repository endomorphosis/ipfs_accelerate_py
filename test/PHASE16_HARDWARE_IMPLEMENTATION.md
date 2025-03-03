# Phase 16 Hardware Integration Implementation Guide

*Last Updated: March 2, 2025*

This document provides details on the hardware integration implementation for Phase 16 of the IPFS Accelerate Python Framework. It focuses on ensuring that all test generators, benchmark tools, and skill generators correctly support multiple hardware platforms including CPU, CUDA, OpenVINO, MPS (Apple Silicon), ROCm (AMD), WebNN, and WebGPU.

## Current Status and Implementation

We have implemented comprehensive hardware platform support through:

1. Updated hardware compatibility maps in the test generators
2. Fixed hardware integration issues in key model test files
3. Enhanced WebNN and WebGPU support with proper simulation modes
4. Added proper AMD precision handling for ROCm hardware
5. Improved test method integration for all hardware platforms

### Key Components

The implementation consists of the following key components:

1. **Updated Hardware Compatibility Map**: Refined mapping in `merged_test_generator.py` with accurate "REAL" vs "SIMULATION" designations for all platforms
2. **Hardware Integration Fixer**: New tool in `fix_hardware_integration.py` to identify and fix hardware integration issues
3. **Automated Fix Script**: Runner in `run_key_model_fixes.sh` to apply fixes to all key model test files
4. **Hardware Templates**: Model-specific WebNN and WebGPU implementation templates based on modality (text, vision, audio, multimodal)

## Platform Support Matrix

The following table shows the current hardware platform support status for key model families:

| Model Family | CPU | CUDA | OpenVINO | MPS | ROCm | WebNN | WebGPU |
|--------------|-----|------|----------|-----|------|-------|--------|
| BERT | REAL | REAL | REAL | REAL | REAL | REAL | REAL |
| T5 | REAL | REAL | REAL | REAL | REAL | REAL | REAL |
| LLaMA | REAL | REAL | REAL | REAL | REAL | SIMULATION | SIMULATION |
| CLIP | REAL | REAL | REAL | REAL | REAL | REAL | REAL |
| ViT | REAL | REAL | REAL | REAL | REAL | REAL | REAL |
| CLAP | REAL | REAL | REAL | REAL | REAL | SIMULATION | SIMULATION |
| Whisper | REAL | REAL | REAL | REAL | REAL | SIMULATION | SIMULATION |
| Wav2Vec2 | REAL | REAL | REAL | REAL | REAL | SIMULATION | SIMULATION |
| LLaVA | REAL | REAL | SIMULATION | SIMULATION | SIMULATION | SIMULATION | SIMULATION |
| LLaVA-Next | REAL | REAL | SIMULATION | SIMULATION | SIMULATION | SIMULATION | SIMULATION |
| XCLIP | REAL | REAL | REAL | REAL | REAL | SIMULATION | SIMULATION |
| Qwen2/3 | REAL | REAL | SIMULATION | SIMULATION | SIMULATION | SIMULATION | SIMULATION |
| DETR | REAL | REAL | REAL | REAL | REAL | SIMULATION | SIMULATION |

**Legend:**
- **REAL**: Fully implemented with real hardware execution
- **SIMULATION**: Implemented with CPU-based simulation for consistent API behavior

## Implementation Details

### 1. Hardware Compatibility Map

The hardware compatibility map in `merged_test_generator.py` was updated to correctly reflect the actual implementation status of each hardware platform for each model family. This ensures that test generators and skill generators produce code with appropriate hardware support.

Key changes:
- Corrected WebNN and WebGPU status for LLMs and audio models from "REAL" to "SIMULATION"
- Updated multimodal models (LLaVA, LLaVA-Next) with "SIMULATION" for non-CUDA platforms
- Set proper implementation status for Qwen2/3 models

### 2. Hardware Integration Fixer

The `fix_hardware_integration.py` script was created to automatically analyze and fix hardware integration issues in test files. It addresses:

- Missing hardware platform methods (`init_<platform>` and `test_with_<platform>`)
- Improper integration of test methods in the `run_tests` method
- Indentation issues with hardware-related methods
- Missing asyncio imports when needed
- AMD precision handling in ROCm implementations
- WebNN and WebGPU simulation implementation

The script generates appropriate templates based on model type (text, vision, audio, multimodal) and integrates them into the existing test files.

### 3. Automated Fix Script

The `run_key_model_fixes.sh` script automates the process of applying hardware integration fixes to all key model test files. It:

1. Analyzes all key model test files for hardware integration issues
2. Applies fixes to each model individually
3. Performs a final analysis to verify that issues were resolved
4. Saves detailed reports for each step

### 4. WebNN and WebGPU Implementation

For WebNN and WebGPU platforms, we implemented:

- Proper detection of WebNN and WebGPU support in browser environments
- CPU-based simulation for non-browser environments
- Modality-specific implementations for text, vision, and audio models
- Appropriate metadata to match transformers.js output structure
- Integration with the test workflow

## How to Apply Hardware Fixes

To apply hardware fixes to model test files:

1. Run the automated fix script:
   ```bash
   ./run_key_model_fixes.sh
   ```

2. Review the resulting changes in the test files.

3. Verify the implementation with comprehensive hardware tests:
   ```bash
   ./run_comprehensive_hardware_tests.sh
   ```

4. To fix specific models:
   ```bash
   python fix_hardware_integration.py --specific-models bert,t5
   ```

5. To analyze without fixing:
   ```bash
   python fix_hardware_integration.py --all-key-models --analyze-only
   ```

## Integration with Generators

The hardware compatibility mapping and platform support is integrated with:

1. **Merged Test Generator**: Test files generated with `merged_test_generator.py` will include appropriate hardware platform support based on the model type.

2. **Skillset Generator**: The `integrated_skillset_generator.py` uses the hardware compatibility mapping to generate skillset implementations with correct hardware support.

3. **Template System**: The template inheritance system properly inherits hardware-specific implementations from parent templates.

## Web Platform Implementation Plan

### 1. Enhanced Browser Automation and Testing

To improve web platform testing capabilities, we will implement the following enhancements:

1. **Selenium/Playwright Integration for Advanced Browser Control**
   - Integration with Selenium and Playwright testing frameworks
   - Full DOM access and interaction for comprehensive testing
   - Visual regression testing for UI components
   - Network traffic monitoring and manipulation

2. **Headless Browser Testing for CI/CD Environments**
   - Automated testing without visible browser windows
   - Integration with GitHub Actions and other CI systems
   - Container-optimized test execution
   - Parallel test execution for multiple browsers

3. **Cross-Browser Test Result Comparison**
   - Automated performance comparison across browser vendors
   - Compatibility matrices for features and optimizations
   - Visual results comparison for rendering differences
   - Standardized metrics collection across browsers

4. **Browser Extension Context Testing**
   - Testing models within extension execution environments
   - Permission and context isolation validation
   - Content script and background worker interoperation
   - Extension-specific optimizations and constraints

5. **Mobile Browser Emulation Support**
   - Mobile-specific testing for responsive applications
   - Touch event simulation and interaction testing
   - Performance profiling under mobile constraints
   - Device emulation for various screen sizes and capabilities

6. **Multi-Browser Testing in Parallel**
   - Simultaneous testing across multiple browsers
   - Consolidated reporting and metric comparison
   - Optimized test distribution and resource management
   - Load balancing for efficient resource utilization

### 2. DuckDB Integration for Results Storage and Analysis

To improve data management and analysis capabilities, we'll replace JSON-based storage with DuckDB:

1. **Direct DuckDB Interfaces for Test Results**
   - Store all web platform test results directly in DuckDB tables
   - Replace JSON file storage with structured database schema
   - Implement real-time metrics collection during testing
   - Create specialized schema for web platform performance data

2. **Query-Based Analysis Tools**
   - Develop SQL-based analysis tooling for test results
   - Create standardized queries for common performance metrics
   - Implement comparative analysis across different browser vendors
   - Build trend analysis capabilities for detecting regressions

3. **Web Platform Testing Database Schema**
   - Design schema with browser-specific metrics and capabilities
   - Track feature support across different browsers
   - Implement schema versioning for evolving test requirements
   - Create specialized tables for platform-specific metrics

4. **Performance Dashboard Integration**
   - Build integration with existing performance dashboards
   - Create web platform specific visualizations
   - Implement real-time data updates during test runs
   - Add drill-down capabilities for detailed platform analysis

#### Technical Implementation Details

The implementation will be based on these key components:

```python
# Core DuckDB Integration Structure
class WebPlatformDuckDBStore:
    """DuckDB storage handler for web platform test results"""
    
    def __init__(self, db_path="./benchmark_db.duckdb"):
        self.db_path = db_path
        self.conn = self._initialize_connection()
        self._create_schema_if_needed()
    
    def _initialize_connection(self):
        """Initialize DuckDB connection with proper settings"""
        import duckdb
        conn = duckdb.connect(self.db_path)
        conn.execute("SET enable_external_access=true")
        conn.execute("PRAGMA memory_limit='4GB'")
        return conn
    
    def _create_schema_if_needed(self):
        """Create the web platform testing schema if it doesn't exist"""
        # Core tables for web platform test results
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS web_platform_tests (
            test_id INTEGER PRIMARY KEY,
            run_id INTEGER,
            timestamp TIMESTAMP DEFAULT current_timestamp,
            model_name VARCHAR,
            model_type VARCHAR,
            platform VARCHAR,
            browser VARCHAR,
            browser_version VARCHAR,
            device_info VARCHAR,
            implementation_type VARCHAR,
            is_simulation BOOLEAN,
            test_duration_ms FLOAT,
            success BOOLEAN,
            error_message VARCHAR
        )
        """)
        
        # Performance metrics table with flexible structure
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS web_platform_performance (
            metric_id INTEGER PRIMARY KEY,
            test_id INTEGER,
            metric_name VARCHAR,
            metric_value FLOAT,
            metric_unit VARCHAR,
            FOREIGN KEY (test_id) REFERENCES web_platform_tests(test_id)
        )
        """)
        
        # Browser capabilities table
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS browser_capabilities (
            capability_id INTEGER PRIMARY KEY,
            browser VARCHAR,
            browser_version VARCHAR,
            feature VARCHAR,
            is_supported BOOLEAN,
            support_level VARCHAR,
            detection_method VARCHAR,
            verification_date TIMESTAMP DEFAULT current_timestamp
        )
        """)
        
        # Feature usage metrics
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS feature_usage (
            feature_id INTEGER PRIMARY KEY,
            test_id INTEGER,
            feature_name VARCHAR,
            feature_enabled BOOLEAN,
            performance_impact_ms FLOAT,
            memory_impact_mb FLOAT,
            FOREIGN KEY (test_id) REFERENCES web_platform_tests(test_id)
        )
        """)
    
    def store_test_result(self, result_data):
        """Store a web platform test result directly in DuckDB"""
        # Insert the core test data
        test_id = self.conn.execute("""
        INSERT INTO web_platform_tests (
            run_id, model_name, model_type, platform, browser, 
            browser_version, device_info, implementation_type, 
            is_simulation, test_duration_ms, success, error_message
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        RETURNING test_id
        """, (
            result_data.get("run_id", None),
            result_data.get("model_name", ""),
            result_data.get("model_type", ""),
            result_data.get("platform", ""),
            result_data.get("browser", ""),
            result_data.get("browser_version", ""),
            result_data.get("device_info", ""),
            result_data.get("implementation_type", ""),
            result_data.get("is_simulation", True),
            result_data.get("test_duration_ms", 0.0),
            result_data.get("success", False),
            result_data.get("error_message", "")
        )).fetchone()[0]
        
        # Insert performance metrics
        if "metrics" in result_data and isinstance(result_data["metrics"], dict):
            for metric_name, metric_value in result_data["metrics"].items():
                if isinstance(metric_value, (int, float)):
                    self.conn.execute("""
                    INSERT INTO web_platform_performance (
                        test_id, metric_name, metric_value, metric_unit
                    ) VALUES (?, ?, ?, ?)
                    """, (
                        test_id,
                        metric_name,
                        float(metric_value),
                        result_data.get("metric_units", {}).get(metric_name, "ms")
                    ))
        
        # Insert feature usage data
        if "features" in result_data and isinstance(result_data["features"], dict):
            for feature_name, feature_data in result_data["features"].items():
                if isinstance(feature_data, dict):
                    self.conn.execute("""
                    INSERT INTO feature_usage (
                        test_id, feature_name, feature_enabled, 
                        performance_impact_ms, memory_impact_mb
                    ) VALUES (?, ?, ?, ?, ?)
                    """, (
                        test_id,
                        feature_name,
                        feature_data.get("enabled", False),
                        feature_data.get("performance_impact_ms", 0.0),
                        feature_data.get("memory_impact_mb", 0.0)
                    ))
        
        return test_id
    
    def get_test_results(self, filters=None, limit=100):
        """Query test results with optional filtering"""
        query = "SELECT * FROM web_platform_tests"
        params = []
        
        if filters:
            where_clauses = []
            for key, value in filters.items():
                if key in ["model_name", "platform", "browser", "implementation_type"]:
                    where_clauses.append(f"{key} = ?")
                    params.append(value)
            
            if where_clauses:
                query += " WHERE " + " AND ".join(where_clauses)
        
        query += f" ORDER BY timestamp DESC LIMIT {limit}"
        
        return self.conn.execute(query, params).fetchall()
    
    def generate_performance_report(self, platform=None, browser=None, 
                                  model_type=None, output_format="dict"):
        """Generate a comprehensive performance report"""
        filters = []
        params = []
        
        if platform:
            filters.append("t.platform = ?")
            params.append(platform)
        
        if browser:
            filters.append("t.browser = ?")
            params.append(browser)
        
        if model_type:
            filters.append("t.model_type = ?")
            params.append(model_type)
        
        where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""
        
        query = f"""
        SELECT 
            t.model_name,
            t.platform,
            t.browser,
            t.implementation_type,
            t.is_simulation,
            AVG(t.test_duration_ms) as avg_duration_ms,
            MIN(t.test_duration_ms) as min_duration_ms,
            MAX(t.test_duration_ms) as max_duration_ms,
            COUNT(*) as test_count,
            SUM(CASE WHEN t.success THEN 1 ELSE 0 END) as success_count,
            SUM(CASE WHEN t.success THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as success_rate
        FROM 
            web_platform_tests t
        {where_clause}
        GROUP BY 
            t.model_name, t.platform, t.browser, t.implementation_type, t.is_simulation
        ORDER BY 
            t.model_name, t.platform, t.browser
        """
        
        results = self.conn.execute(query, params).fetchall()
        
        if output_format == "dict":
            column_names = ["model_name", "platform", "browser", "implementation_type", 
                          "is_simulation", "avg_duration_ms", "min_duration_ms", 
                          "max_duration_ms", "test_count", "success_count", "success_rate"]
            
            return [dict(zip(column_names, row)) for row in results]
        else:
            return results
    
    def export_to_parquet(self, output_path="./web_platform_results.parquet"):
        """Export web platform test results to Parquet for external analysis"""
        query = """
        SELECT 
            t.*,
            p.metric_name,
            p.metric_value,
            p.metric_unit,
            f.feature_name,
            f.feature_enabled,
            f.performance_impact_ms,
            f.memory_impact_mb
        FROM 
            web_platform_tests t
        LEFT JOIN 
            web_platform_performance p ON t.test_id = p.test_id
        LEFT JOIN 
            feature_usage f ON t.test_id = f.test_id
        """
        
        self.conn.execute(f"COPY ({query}) TO '{output_path}' (FORMAT PARQUET)")
        return output_path
    
    def close(self):
        """Close the DuckDB connection"""
        if self.conn:
            self.conn.close()
            self.conn = None
```

This schema allows for:

1. **Structured Test Data**: Storing core test information with proper typing
2. **Flexible Metrics**: Storing performance metrics with proper numerical handling
3. **Feature Tracking**: Tracking which features were enabled during tests
4. **Browser Capabilities**: Recording which browsers support which features
5. **Direct Parquet Export**: Exporting data for use with other tools

The integration will be used in place of JSON storage throughout the web platform testing system.

### 3. NPU (Neural Processing Unit) Hardware Integration

To support emerging NPU hardware and parallel execution scenarios:

1. **NPU Detection and Support**
   - Add NPU capability detection to hardware_detection module
   - Support Intel NPUs, Qualcomm NPUs, and Apple Neural Engine
   - Implement feature detection for NPU-specific optimizations
   - Create NPU-specific simulation mode for testing without hardware

2. **ResourcePool NPU Support**
   - Extend ResourcePool to manage NPU resources
   - Implement NPU-specific memory management
   - Add NPU to hardware priority lists for compatible models
   - Create NPU-specific model optimization pathways

3. **Multi-Hardware Simultaneous Inference**
   - Enable simultaneous model execution across CPU, GPU, and NPU
   - Create multi-endpoint handler system for parallel execution
   - Implement task distribution based on hardware capabilities
   - Develop result aggregation and comparison logic

4. **Multi-Hardware Endpoint Handler**
   - Extend endpoint handler paradigm to support multiple hardware devices
   - Create specialized handlers for CPU+GPU+NPU combinations
   - Implement dynamic hardware selection based on workload
   - Add fallback mechanisms for graceful degradation

5. **Benchmark Tools for Multi-Hardware Execution**
   - Create benchmarking tools for multi-hardware scenarios
   - Measure throughput, latency, and efficiency metrics
   - Compare different hardware allocation strategies
   - Implement DuckDB-based result storage for analysis

#### Technical Implementation Details

The NPU integration will consist of these key components:

```python
# Enhanced Hardware Detection with NPU Support
class HardwareDetectionWithNPU:
    """Enhanced hardware detection module with NPU support"""
    
    @staticmethod
    def detect_available_hardware():
        """Detect all available hardware including NPUs"""
        result = {
            "cpu": {"available": True, "count": os.cpu_count()},
            "cuda": {"available": False},
            "rocm": {"available": False},
            "mps": {"available": False},
            "npu": {"available": False},
            "npu_type": None,
            "openvino": {"available": False},
            "webnn": {"available": False},
            "webgpu": {"available": False}
        }
        
        # Standard PyTorch device detection
        try:
            import torch
            # CUDA (NVIDIA)
            if torch.cuda.is_available():
                result["cuda"] = {
                    "available": True,
                    "device_count": torch.cuda.device_count(),
                    "device_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                    "cuda_version": torch.version.cuda
                }
            
            # ROCm (AMD)
            if hasattr(torch.version, 'hip') and torch.version.hip is not None:
                result["rocm"] = {
                    "available": True,
                    "version": torch.version.hip
                }
            
            # MPS (Apple Silicon)
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                result["mps"] = {
                    "available": True,
                    "device": "mps"
                }
        except ImportError:
            pass
        
        # NPU detection
        # 1. Intel NPU
        result.update(HardwareDetectionWithNPU._detect_intel_npu())
        
        # 2. Apple Neural Engine
        if not result["npu"]["available"]:
            result.update(HardwareDetectionWithNPU._detect_apple_neural_engine())
        
        # 3. Qualcomm NPU
        if not result["npu"]["available"]:
            result.update(HardwareDetectionWithNPU._detect_qualcomm_npu())
        
        # 4. Google TPU
        if not result["npu"]["available"]:
            result.update(HardwareDetectionWithNPU._detect_google_tpu())
        
        # OpenVINO detection
        try:
            import importlib.util
            if importlib.util.find_spec("openvino") is not None:
                import openvino as ov
                ie = ov.Core()
                available_devices = ie.available_devices
                result["openvino"] = {
                    "available": True,
                    "devices": available_devices,
                    "version": ov.__version__
                }
        except ImportError:
            pass
        
        return result
    
    @staticmethod
    def _detect_intel_npu():
        """Detect Intel NPU (Neural Compute Stick, Arc GPUs with XMX, etc)"""
        result = {"npu": {"available": False}}
        
        try:
            # Check for OpenVINO NPU support
            import importlib.util
            if importlib.util.find_spec("openvino") is not None:
                import openvino as ov
                ie = ov.Core()
                devices = ie.available_devices
                
                # Check for Intel NPU devices
                npu_devices = [d for d in devices if "HDDL" in d or "MYRIAD" in d or "GPU.1" in d]
                if npu_devices:
                    result["npu"] = {
                        "available": True,
                        "type": "intel",
                        "devices": npu_devices,
                        "provider": "openvino"
                    }
                    result["npu_type"] = "intel"
        except Exception:
            pass
        
        return result
    
    @staticmethod
    def _detect_apple_neural_engine():
        """Detect Apple Neural Engine"""
        result = {"npu": {"available": False}}
        
        try:
            import platform
            if platform.system() == "Darwin" and platform.machine() == "arm64":
                # Check for CoreML
                import importlib.util
                if importlib.util.find_spec("coremltools") is not None:
                    # Apple Silicon with Neural Engine support
                    result["npu"] = {
                        "available": True,
                        "type": "apple_neural_engine",
                        "provider": "coreml"
                    }
                    result["npu_type"] = "apple_neural_engine"
        except Exception:
            pass
        
        return result
    
    @staticmethod
    def _detect_qualcomm_npu():
        """Detect Qualcomm NPU (Hexagon DSP, etc)"""
        result = {"npu": {"available": False}}
        
        try:
            import platform
            if platform.system() == "Linux":
                # Check for SNPE (Snapdragon Neural Processing Engine)
                import importlib.util
                if importlib.util.find_spec("snpe") is not None:
                    result["npu"] = {
                        "available": True,
                        "type": "qualcomm_hexagon",
                        "provider": "snpe"
                    }
                    result["npu_type"] = "qualcomm_hexagon"
        except Exception:
            pass
        
        return result
    
    @staticmethod
    def _detect_google_tpu():
        """Detect Google TPU"""
        result = {"npu": {"available": False}}
        
        try:
            # Check for TensorFlow TPU support
            import importlib.util
            if importlib.util.find_spec("tensorflow") is not None:
                import tensorflow as tf
                tpu_devices = tf.config.list_logical_devices('TPU')
                if tpu_devices:
                    result["npu"] = {
                        "available": True,
                        "type": "google_tpu",
                        "devices": [d.name for d in tpu_devices],
                        "provider": "tensorflow"
                    }
                    result["npu_type"] = "google_tpu"
        except Exception:
            pass
        
        return result
```

The multi-hardware endpoint handler system will build on this with:

```python
# Multi-Hardware Endpoint Handler System
class MultiHardwareEndpointHandler:
    """Handler for running models across multiple hardware devices in parallel"""
    
    def __init__(self, model_name, tokenizer=None, devices=None, 
                weights=None, db_store=None):
        """
        Initialize the multi-hardware endpoint handler
        
        Args:
            model_name: Name of the model to load
            tokenizer: Optional tokenizer to use
            devices: List of devices to use ["cpu", "cuda", "npu", etc]
            weights: Optional weights for load balancing across devices
            db_store: Optional DuckDB storage instance for results
        """
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.devices = devices or ["cpu", "cuda", "npu"]
        self.weights = weights or {"cpu": 0.2, "cuda": 0.5, "npu": 0.3}
        self.db_store = db_store
        
        # Filter to available devices
        self._detect_and_filter_devices()
        
        # Initialize endpoint handlers for each device
        self.endpoint_handlers = {}
        self._initialize_endpoints()
        
        # Create resource pool for sharing tokenizers, etc.
        from resource_pool import get_global_resource_pool
        self.resource_pool = get_global_resource_pool()
        
        # Statistics tracking
        self.stats = {
            "requests": 0,
            "successes": 0,
            "failures": 0,
            "device_usage": {d: 0 for d in self.devices},
            "latencies": {d: [] for d in self.devices}
        }
    
    def _detect_and_filter_devices(self):
        """Detect available hardware and filter devices list"""
        hardware_info = HardwareDetectionWithNPU.detect_available_hardware()
        
        # Filter to available devices
        available_devices = []
        for device in self.devices:
            if device == "npu":
                if hardware_info["npu"]["available"]:
                    available_devices.append(device)
            elif device in hardware_info and hardware_info[device]["available"]:
                available_devices.append(device)
        
        # Update device list
        self.devices = available_devices
        
        # If no devices available, fall back to CPU
        if not self.devices:
            self.devices = ["cpu"]
            self.weights = {"cpu": 1.0}
    
    def _initialize_endpoints(self):
        """Initialize endpoint handlers for each device"""
        for device in self.devices:
            try:
                if device == "cpu":
                    from endpoint_handlers import CPUEndpointHandler
                    self.endpoint_handlers[device] = CPUEndpointHandler(
                        model_name=self.model_name,
                        tokenizer=self.tokenizer
                    )
                elif device == "cuda":
                    from endpoint_handlers import CUDAEndpointHandler
                    self.endpoint_handlers[device] = CUDAEndpointHandler(
                        model_name=self.model_name,
                        tokenizer=self.tokenizer
                    )
                elif device == "npu":
                    from endpoint_handlers import NPUEndpointHandler
                    self.endpoint_handlers[device] = NPUEndpointHandler(
                        model_name=self.model_name,
                        tokenizer=self.tokenizer
                    )
                elif device == "mps":
                    from endpoint_handlers import MPSEndpointHandler
                    self.endpoint_handlers[device] = MPSEndpointHandler(
                        model_name=self.model_name,
                        tokenizer=self.tokenizer
                    )
                elif device == "rocm":
                    from endpoint_handlers import ROCmEndpointHandler
                    self.endpoint_handlers[device] = ROCmEndpointHandler(
                        model_name=self.model_name,
                        tokenizer=self.tokenizer
                    )
            except Exception as e:
                print(f"Error initializing {device} endpoint: {e}")
    
    def __call__(self, inputs, strategy="parallel"):
        """
        Process inputs using the multi-hardware strategy
        
        Args:
            inputs: Model inputs
            strategy: Execution strategy ("parallel", "fastest", "balanced")
            
        Returns:
            Model outputs from the best or aggregated device results
        """
        self.stats["requests"] += 1
        
        if not self.endpoint_handlers:
            raise ValueError("No endpoint handlers available")
        
        if strategy == "parallel":
            # Run on all devices simultaneously and use the fastest result
            return self._execute_parallel(inputs)
        elif strategy == "fastest":
            # Use the historically fastest device
            return self._execute_fastest(inputs)
        elif strategy == "balanced":
            # Load balance across devices according to weights
            return self._execute_balanced(inputs)
        else:
            # Default to parallel
            return self._execute_parallel(inputs)
    
    def _execute_parallel(self, inputs):
        """Execute on all devices in parallel and return fastest result"""
        import concurrent.futures
        import time
        
        results = {}
        errors = {}
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.devices)) as executor:
            # Submit tasks for each device
            futures = {}
            for device in self.endpoint_handlers:
                endpoint = self.endpoint_handlers[device]
                futures[executor.submit(self._timed_inference, endpoint, inputs, device)] = device
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(futures):
                device = futures[future]
                try:
                    success, latency, result = future.result()
                    if success:
                        results[device] = (latency, result)
                        self.stats["successes"] += 1
                        self.stats["device_usage"][device] += 1
                        self.stats["latencies"][device].append(latency)
                    else:
                        errors[device] = result
                        self.stats["failures"] += 1
                except Exception as e:
                    errors[device] = str(e)
                    self.stats["failures"] += 1
        
        # Return fastest successful result
        if results:
            # Find device with minimum latency
            fastest_device = min(results.keys(), key=lambda d: results[d][0])
            result = results[fastest_device][1]
            
            # Add metadata about execution
            if isinstance(result, dict):
                result["multi_hardware_metadata"] = {
                    "strategy": "parallel",
                    "selected_device": fastest_device,
                    "all_devices": list(self.devices),
                    "all_latencies": {d: results[d][0] for d in results},
                    "errors": errors
                }
            
            # Store results in DuckDB if available
            if self.db_store:
                self._store_result(result, fastest_device, errors, strategy="parallel")
            
            return result
        
        # No successful results
        error_msg = "; ".join([f"{d}: {e}" for d, e in errors.items()])
        raise RuntimeError(f"All devices failed: {error_msg}")
    
    def _execute_fastest(self, inputs):
        """Execute on historically fastest device with fallback"""
        # Find device with lowest average latency
        devices_with_stats = [d for d in self.devices if len(self.stats["latencies"][d]) > 0]
        
        if not devices_with_stats:
            # No history yet, use first available device
            target_device = self.devices[0]
        else:
            # Calculate average latency for each device
            avg_latencies = {
                d: sum(self.stats["latencies"][d]) / len(self.stats["latencies"][d])
                for d in devices_with_stats
            }
            # Select device with lowest average latency
            target_device = min(avg_latencies, key=avg_latencies.get)
        
        # Try primary device
        try:
            endpoint = self.endpoint_handlers[target_device]
            success, latency, result = self._timed_inference(endpoint, inputs, target_device)
            
            if success:
                self.stats["successes"] += 1
                self.stats["device_usage"][target_device] += 1
                self.stats["latencies"][target_device].append(latency)
                
                # Add metadata
                if isinstance(result, dict):
                    result["multi_hardware_metadata"] = {
                        "strategy": "fastest",
                        "selected_device": target_device,
                        "latency": latency,
                        "fallback_used": False
                    }
                
                # Store results in DuckDB if available
                if self.db_store:
                    self._store_result(result, target_device, {}, strategy="fastest")
                
                return result
        except Exception as e:
            # Primary device failed, try fallbacks
            primary_error = str(e)
            
            # Try other devices in order of historical performance
            other_devices = [d for d in self.devices if d != target_device]
            other_devices.sort(key=lambda d: sum(self.stats["latencies"][d]) / max(1, len(self.stats["latencies"][d])) if self.stats["latencies"][d] else float('inf'))
            
            for device in other_devices:
                try:
                    endpoint = self.endpoint_handlers[device]
                    success, latency, result = self._timed_inference(endpoint, inputs, device)
                    
                    if success:
                        self.stats["successes"] += 1
                        self.stats["device_usage"][device] += 1
                        self.stats["latencies"][device].append(latency)
                        
                        # Add metadata
                        if isinstance(result, dict):
                            result["multi_hardware_metadata"] = {
                                "strategy": "fastest",
                                "selected_device": device,
                                "primary_device": target_device,
                                "primary_error": primary_error,
                                "latency": latency,
                                "fallback_used": True
                            }
                        
                        # Store results in DuckDB if available
                        if self.db_store:
                            self._store_result(result, device, {target_device: primary_error}, strategy="fastest_fallback")
                        
                        return result
                except Exception as fallback_e:
                    # Continue to next fallback device
                    continue
            
            # All devices failed
            self.stats["failures"] += 1
            raise RuntimeError(f"All devices failed. Primary error: {primary_error}")
    
    def _execute_balanced(self, inputs):
        """Load balance across devices according to weights"""
        import random
        
        # Normalize weights to sum to 1.0
        total_weight = sum(self.weights[d] for d in self.devices if d in self.weights)
        normalized_weights = {d: self.weights.get(d, 0) / total_weight for d in self.devices}
        
        # Create weighted distribution
        devices = list(self.devices)
        weights = [normalized_weights.get(d, 0) for d in devices]
        
        # Select a device based on weights
        selected_device = random.choices(devices, weights=weights, k=1)[0]
        
        # Execute on selected device
        try:
            endpoint = self.endpoint_handlers[selected_device]
            success, latency, result = self._timed_inference(endpoint, inputs, selected_device)
            
            if success:
                self.stats["successes"] += 1
                self.stats["device_usage"][selected_device] += 1
                self.stats["latencies"][selected_device].append(latency)
                
                # Add metadata
                if isinstance(result, dict):
                    result["multi_hardware_metadata"] = {
                        "strategy": "balanced",
                        "selected_device": selected_device,
                        "weights": self.weights,
                        "latency": latency
                    }
                
                # Store results in DuckDB if available
                if self.db_store:
                    self._store_result(result, selected_device, {}, strategy="balanced")
                
                return result
            else:
                # Selected device failed, try others in weight order
                error = result
                other_devices = [d for d in devices if d != selected_device]
                other_devices.sort(key=lambda d: normalized_weights.get(d, 0), reverse=True)
                
                for device in other_devices:
                    try:
                        endpoint = self.endpoint_handlers[device]
                        success, latency, result = self._timed_inference(endpoint, inputs, device)
                        
                        if success:
                            self.stats["successes"] += 1
                            self.stats["device_usage"][device] += 1
                            self.stats["latencies"][device].append(latency)
                            
                            # Update weights to reflect this failure
                            self._update_weights(selected_device, False)
                            self._update_weights(device, True)
                            
                            # Add metadata
                            if isinstance(result, dict):
                                result["multi_hardware_metadata"] = {
                                    "strategy": "balanced",
                                    "selected_device": device,
                                    "primary_device": selected_device,
                                    "primary_error": error,
                                    "latency": latency,
                                    "fallback_used": True
                                }
                            
                            # Store results in DuckDB if available
                            if self.db_store:
                                self._store_result(result, device, {selected_device: error}, strategy="balanced_fallback")
                            
                            return result
                    except Exception:
                        # Continue to next device
                        continue
                
                # All devices failed
                self.stats["failures"] += 1
                raise RuntimeError(f"All devices failed. Primary error: {error}")
        except Exception as e:
            self.stats["failures"] += 1
            raise e
    
    def _timed_inference(self, endpoint, inputs, device):
        """Run inference with timing and error handling"""
        import time
        
        start_time = time.time()
        try:
            result = endpoint(inputs)
            end_time = time.time()
            latency = (end_time - start_time) * 1000  # ms
            
            # Check for valid result
            if result is None:
                return False, 0, "Empty result"
            
            return True, latency, result
        except Exception as e:
            end_time = time.time()
            latency = (end_time - start_time) * 1000  # ms
            return False, latency, str(e)
    
    def _update_weights(self, device, success):
        """Update weights based on success/failure"""
        if device not in self.weights:
            return
            
        if success:
            # Increase weight for successful device
            self.weights[device] *= 1.1
        else:
            # Decrease weight for failed device
            self.weights[device] *= 0.8
            
        # Normalize weights
        total = sum(self.weights.values())
        for d in self.weights:
            self.weights[d] /= total
    
    def _store_result(self, result, selected_device, errors, strategy):
        """Store results in DuckDB if available"""
        if not self.db_store:
            return
            
        try:
            # Extract metadata
            latency = 0
            if isinstance(result, dict) and "multi_hardware_metadata" in result:
                metadata = result["multi_hardware_metadata"]
                latency = metadata.get("latency", 0)
            
            # Prepare data for storage
            data = {
                "model_name": self.model_name,
                "device": selected_device,
                "strategy": strategy,
                "success": True,
                "latency_ms": latency,
                "errors": errors,
                "timestamp": time.time()
            }
            
            # Store in DuckDB
            self.db_store.store_multi_hardware_result(data)
        except Exception as e:
            print(f"Error storing result in DuckDB: {e}")
    
    def get_stats(self):
        """Get usage statistics"""
        return {
            "requests": self.stats["requests"],
            "successes": self.stats["successes"],
            "failures": self.stats["failures"],
            "success_rate": self.stats["successes"] / max(1, self.stats["requests"]) * 100,
            "device_usage": self.stats["device_usage"],
            "device_usage_percent": {
                d: (self.stats["device_usage"][d] / max(1, self.stats["requests"]) * 100)
                for d in self.stats["device_usage"]
            },
            "average_latency": {
                d: (sum(self.stats["latencies"][d]) / max(1, len(self.stats["latencies"][d])))
                for d in self.stats["latencies"]
            },
            "current_weights": self.weights
        }
```

This implementation provides:

1. **Dynamic Hardware Detection**: Automatic detection of CPU, GPU, and NPU devices
2. **Flexible Execution Strategies**: Parallel, fastest, and balanced load execution options 
3. **Intelligent Fallback**: Automatic fallback to available devices if primary fails
4. **Performance Tracking**: Comprehensive statistics on hardware performance
5. **Result Metadata**: Detailed execution information included with results
6. **DuckDB Integration**: Direct storage of multi-hardware execution results

### 4. Cross-Platform Testing Integration 

To complete the web platform implementation plan, we need to bring together browser automation, DuckDB integration, and NPU support:

1. **Unified Testing System**
   - Implement cross-platform testing that integrates all hardware types
   - Create a consistent API for accessing browser-based and hardware-based acceleration
   - Unify metrics collection across all platforms for consistent reporting
   - Build comprehensive test suites covering all supported hardware combinations

2. **Web-Specific NPU Integration**
   - Add browser-based NPU detection using WebNN and WebGPU APIs
   - Create specialized test paths for browsers with Neural Engine access
   - Test on Apple Silicon devices with Neural Engine via WebNN
   - Integrate with Chrome's experimental WebNN NPU backend

3. **Comprehensive Results Dashboard**
   - Create an integrated dashboard showing results across all platforms
   - Use DuckDB as the unified storage backend for all test results
   - Implement interactive visualizations for hardware comparisons
   - Provide drill-down views of individual test configurations

4. **Regression Testing System**
   - Implement automatic regression detection across all platforms
   - Track performance changes across browsers and hardware generations
   - Create alert system for significant performance regressions
   - Store historical benchmarks for long-term trend analysis

```python
# Example of unified test runner with NPU/Browser integration
class UnifiedTestRunner:
    """Test runner that can execute tests across any combination of hardware and browsers"""
    
    def __init__(self, db_path="./benchmark_db.duckdb"):
        """Initialize the unified test runner"""
        # Set up DuckDB storage
        from web_platform_duckdb import WebPlatformDuckDBStore
        self.db_store = WebPlatformDuckDBStore(db_path=db_path)
        
        # Hardware detection
        from hardware_detection_with_npu import HardwareDetectionWithNPU
        self.hardware_info = HardwareDetectionWithNPU.detect_available_hardware()
        
        # Browser automation setup
        from fixed_web_platform.browser_automation import setup_browser_automation
        self.browser_setup = {}
        
        # Track available hardware
        self.available_hardware = self._get_available_hardware()
        
    def _get_available_hardware(self):
        """Get list of all available hardware devices"""
        hardware = []
        
        # Add standard hardware
        if self.hardware_info["cpu"]["available"]:
            hardware.append("cpu")
        if self.hardware_info["cuda"]["available"]:
            hardware.append("cuda")
        if self.hardware_info["rocm"]["available"]:
            hardware.append("rocm")
        if self.hardware_info["mps"]["available"]:
            hardware.append("mps")
        if self.hardware_info["npu"]["available"]:
            hardware.append("npu")
        if self.hardware_info["openvino"]["available"]:
            hardware.append("openvino")
            
        # Check for browser platforms
        for browser in ["chrome", "edge", "firefox"]:
            browser_path = self._check_browser_availability(browser)
            if browser_path:
                hardware.append(f"webnn-{browser}")
                hardware.append(f"webgpu-{browser}")
                
        return hardware
    
    def _check_browser_availability(self, browser):
        """Check if a specific browser is available"""
        # Import browser automation module
        from fixed_web_platform.browser_automation import find_browser_executable
        return find_browser_executable(browser)
    
    def run_test(self, model_name, test_config):
        """
        Run a test with the specified model and configuration
        
        Args:
            model_name: Name of the model to test
            test_config: Configuration dictionary with test parameters
                - hardware: List of hardware devices to test on
                - browser: Browser to use for web platform tests
                - strategy: Strategy for multi-hardware testing
                - execution_mode: "sequential", "parallel", or "matrix"
                
        Returns:
            Dictionary with test results
        """
        results = {}
        
        # Validate hardware selection against available hardware
        selected_hardware = test_config.get("hardware", [])
        if not selected_hardware:
            selected_hardware = ["cpu"]  # Default to CPU
            
        # Filter to available hardware
        selected_hardware = [h for h in selected_hardware if h in self.available_hardware]
        
        # Determine execution mode
        execution_mode = test_config.get("execution_mode", "sequential")
        
        if execution_mode == "sequential":
            # Test on each hardware sequentially
            for hardware in selected_hardware:
                result = self._run_single_hardware_test(model_name, hardware, test_config)
                results[hardware] = result
                
        elif execution_mode == "parallel":
            # Test on all hardware in parallel
            results = self._run_parallel_hardware_tests(model_name, selected_hardware, test_config)
            
        elif execution_mode == "matrix":
            # Run a matrix of tests with different configurations
            results = self._run_matrix_tests(model_name, selected_hardware, test_config)
            
        # Store results in DuckDB
        self._store_results(model_name, results, test_config)
        
        return results
    
    def _run_single_hardware_test(self, model_name, hardware, test_config):
        """Run a test on a single hardware device"""
        # Check if this is a web platform test
        if hardware.startswith("webnn-") or hardware.startswith("webgpu-"):
            # Extract platform and browser
            parts = hardware.split("-")
            platform = parts[0]
            browser = parts[1] if len(parts) > 1 else "chrome"
            
            # Run browser-based test
            return self._run_browser_test(model_name, platform, browser, test_config)
            
        elif hardware == "npu":
            # Run NPU test
            return self._run_npu_test(model_name, test_config)
            
        else:
            # Run standard hardware test
            return self._run_standard_hardware_test(model_name, hardware, test_config)
    
    def _run_browser_test(self, model_name, platform, browser, test_config):
        """Run a test using browser automation"""
        # Set up browser automation
        from fixed_web_platform.browser_automation import setup_browser_automation
        
        # Get additional features configuration
        compute_shaders = test_config.get("compute_shaders", False)
        precompile_shaders = test_config.get("precompile_shaders", False)
        parallel_loading = test_config.get("parallel_loading", False)
        
        # Set up browser automation
        browser_config = setup_browser_automation(
            platform=platform,
            browser_preference=browser,
            modality=test_config.get("modality", "text"),
            model_name=model_name,
            compute_shaders=compute_shaders,
            precompile_shaders=precompile_shaders,
            parallel_loading=parallel_loading
        )
        
        # Run the test in the browser
        from fixed_web_platform.browser_automation import run_browser_test
        result = run_browser_test(browser_config, timeout_seconds=test_config.get("timeout", 60))
        
        # Convert to standard result format
        return {
            "success": result.get("success", False),
            "implementation_type": result.get("implementation_type", f"REAL_{platform.upper()}"),
            "browser": browser,
            "platform": platform,
            "latency_ms": result.get("latency_ms", 0),
            "error": result.get("error", "")
        }
    
    def _run_npu_test(self, model_name, test_config):
        """Run a test on NPU hardware"""
        # Create NPU endpoint handler
        from endpoint_handlers import NPUEndpointHandler
        
        try:
            # Create endpoint handler
            endpoint = NPUEndpointHandler(model_name=model_name)
            
            # Run test
            import time
            start_time = time.time()
            result = endpoint(test_config.get("inputs", {}))
            end_time = time.time()
            
            # Calculate latency
            latency_ms = (end_time - start_time) * 1000
            
            # Return formatted result
            return {
                "success": True,
                "implementation_type": "REAL_NPU",
                "npu_type": self.hardware_info["npu_type"],
                "latency_ms": latency_ms,
                "result": result
            }
        except Exception as e:
            return {
                "success": False,
                "implementation_type": "REAL_NPU",
                "npu_type": self.hardware_info["npu_type"],
                "error": str(e)
            }
    
    def _run_standard_hardware_test(self, model_name, hardware, test_config):
        """Run a test on standard hardware (CPU, CUDA, etc)"""
        # Import appropriate endpoint handler
        handler_name = f"{hardware.upper()}EndpointHandler"
        try:
            from endpoint_handlers import get_endpoint_handler
            endpoint_class = get_endpoint_handler(hardware)
            
            # Create endpoint handler
            endpoint = endpoint_class(model_name=model_name)
            
            # Run test
            import time
            start_time = time.time()
            result = endpoint(test_config.get("inputs", {}))
            end_time = time.time()
            
            # Calculate latency
            latency_ms = (end_time - start_time) * 1000
            
            # Return formatted result
            return {
                "success": True,
                "implementation_type": f"REAL_{hardware.upper()}",
                "hardware": hardware,
                "latency_ms": latency_ms,
                "result": result
            }
        except Exception as e:
            return {
                "success": False,
                "implementation_type": f"REAL_{hardware.upper()}",
                "hardware": hardware,
                "error": str(e)
            }
    
    def _run_parallel_hardware_tests(self, model_name, hardware_list, test_config):
        """Run tests on multiple hardware devices in parallel"""
        import concurrent.futures
        
        results = {}
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(hardware_list)) as executor:
            # Submit tasks for each hardware
            futures = {}
            for hardware in hardware_list:
                futures[executor.submit(self._run_single_hardware_test, model_name, hardware, test_config)] = hardware
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(futures):
                hardware = futures[future]
                try:
                    result = future.result()
                    results[hardware] = result
                except Exception as e:
                    results[hardware] = {
                        "success": False,
                        "error": str(e)
                    }
        
        return results
    
    def _run_matrix_tests(self, model_name, hardware_list, test_config):
        """Run a matrix of tests with different configurations"""
        matrix_results = {}
        
        # Get matrix parameters
        batch_sizes = test_config.get("batch_sizes", [1])
        sequence_lengths = test_config.get("sequence_lengths", [128])
        
        # Run tests for each combination
        for hardware in hardware_list:
            hardware_results = {}
            
            for batch_size in batch_sizes:
                for seq_length in sequence_lengths:
                    # Create test configuration
                    config = test_config.copy()
                    config["batch_size"] = batch_size
                    config["sequence_length"] = seq_length
                    
                    # Run test
                    result = self._run_single_hardware_test(model_name, hardware, config)
                    
                    # Store result
                    key = f"b{batch_size}_s{seq_length}"
                    hardware_results[key] = result
            
            matrix_results[hardware] = hardware_results
        
        return matrix_results
    
    def _store_results(self, model_name, results, test_config):
        """Store test results in DuckDB"""
        # Flatten results for storage
        flattened_results = []
        
        # Process results based on structure
        if isinstance(results, dict):
            for hardware, result in results.items():
                if isinstance(result, dict) and "success" in result:
                    # Direct hardware result
                    entry = {
                        "model_name": model_name,
                        "hardware": hardware,
                        **result
                    }
                    flattened_results.append(entry)
                elif isinstance(result, dict):
                    # Matrix result
                    for config, config_result in result.items():
                        entry = {
                            "model_name": model_name,
                            "hardware": hardware,
                            "config": config,
                            **config_result
                        }
                        flattened_results.append(entry)
        
        # Store each result in DuckDB
        for entry in flattened_results:
            self.db_store.store_test_result(entry)
```

### Implementation Timeline

| Phase | Features | Target Date |
|-------|----------|-------------|
| 1 | Enhanced Selenium/Playwright Integration | Q2 2025 |
| 2 | Headless Browser Testing for CI/CD | Q2 2025 |
| 3 | DuckDB Integration for Test Results | Q2 2025 |
| 4 | NPU Detection and Support | Q2 2025 |
| 5 | Cross-Browser Test Result Comparison | Q3 2025 |
| 6 | Browser Extension Context Testing | Q3 2025 |
| 7 | ResourcePool NPU Support | Q3 2025 |
| 8 | Multi-Hardware Simultaneous Inference | Q3 2025 |
| 9 | Mobile Browser Emulation Support | Q4 2025 |
| 10 | Multi-Hardware Endpoint Handler | Q4 2025 |
| 11 | Multi-Browser Testing in Parallel | Q4 2025 |
| 12 | Benchmark Tools for Multi-Hardware Execution | Q4 2025 |

## Next Steps

1. **Validation**: Run comprehensive hardware tests on all key models to verify implementation
2. **Performance Benchmarking**: Benchmark performance across hardware platforms
3. **Documentation**: Update hardware compatibility matrix in `CLAUDE.md`
4. **Web Platform Integration**: Implement the enhanced browser automation plan

## Conclusion

Phase 16 hardware integration implementation ensures that all model test files and generators correctly support multiple hardware platforms. The automated fix tools help maintain consistent hardware support across the codebase and facilitate the addition of new hardware platforms in the future.

For detailed hardware benchmarking information, refer to the [Hardware Benchmarking Guide](HARDWARE_BENCHMARKING_GUIDE.md).