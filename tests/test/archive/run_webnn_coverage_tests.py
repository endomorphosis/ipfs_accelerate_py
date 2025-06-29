#!/usr/bin/env python3
"""
WebNN Comprehensive Test Coverage Tool

This script runs a complete set of tests to verify WebNN and WebGPU functionality
across browsers, models, and hardware platforms. It generates a comprehensive
report showing test coverage status and performance benchmarks.

Usage:
    python run_webnn_coverage_tests.py --all
    python run_webnn_coverage_tests.py --browser edge --models bert,t5
"""

import os
import sys
import json
import time
import argparse
import subprocess
import concurrent.futures
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Define constants
SUPPORTED_BROWSERS = ["chrome", "edge", "firefox", "safari"]
SUPPORTED_MODELS = [
    "prajjwal1/bert-tiny",
    "t5-small",
    "google/vit-base-patch16-224",
    "whisper-tiny",
    "openai/clip-vit-base-patch32"
]
AUDIO_MODELS = ["whisper-tiny", "facebook/wav2vec2-base"]
MULTIMODAL_MODELS = ["openai/clip-vit-base-patch32"]
DEFAULT_BATCH_SIZES = [1, 4]
MAX_CONCURRENT_TESTS = 2  # Don't run too many browser tests in parallel

class TestCoverageRunner:
    """Runs comprehensive WebNN and WebGPU test coverage."""
    
    def __init__(self, 
                 browsers: List[str] = None,
                 models: List[str] = None,
                 batch_sizes: List[int] = None,
                 output_dir: str = "./webnn_coverage_results",
                 db_path: Optional[str] = None,
                 timeout: int = 600,
                 parallel: bool = True,
                 max_concurrent: int = 2,
                 enable_compute_shaders: bool = False,
                 enable_parallel_loading: bool = False,
                 enable_shader_precompile: bool = False):
        """Initialize the test coverage runner.
        
        Args:
            browsers: List of browsers to test. Defaults to ["edge", "chrome"].
            models: List of models to test. Defaults to a small set of representative models.
            batch_sizes: List of batch sizes to test. Defaults to [1, 4].
            output_dir: Directory to store test results. Defaults to "./webnn_coverage_results".
            db_path: Path to the benchmark database. If not provided, uses environment variable.
            timeout: Timeout in seconds for each test. Defaults to 600.
            parallel: Whether to run tests in parallel. Defaults to True.
            max_concurrent: Maximum number of concurrent tests. Defaults to 2.
            enable_compute_shaders: Whether to enable compute shader optimizations. Defaults to False.
            enable_parallel_loading: Whether to enable parallel loading optimizations. Defaults to False.
            enable_shader_precompile: Whether to enable shader precompilation. Defaults to False.
        """
        self.browsers = browsers or ["edge", "chrome"]
        self.models = models or ["prajjwal1/bert-tiny", "whisper-tiny", "openai/clip-vit-base-patch32"]
        self.batch_sizes = batch_sizes or DEFAULT_BATCH_SIZES
        self.output_dir = output_dir
        self.db_path = db_path or os.environ.get("BENCHMARK_DB_PATH")
        self.timeout = timeout
        self.parallel = parallel
        self.max_concurrent = max_concurrent
        
        # Optimization flags
        self.enable_compute_shaders = enable_compute_shaders
        self.enable_parallel_loading = enable_parallel_loading
        self.enable_shader_precompile = enable_shader_precompile
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Get system information
        system_info = {
            "platform": platform.system(),
            "platform_version": platform.version(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "timestamp": time.time(),
            "date": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Track test results
        self.results = {
            "timestamp": time.time(),
            "system_info": system_info,
            "test_config": {
                "browsers": self.browsers,
                "models": self.models,
                "batch_sizes": self.batch_sizes,
                "optimizations": {
                    "compute_shaders": self.enable_compute_shaders,
                    "parallel_loading": self.enable_parallel_loading,
                    "shader_precompile": self.enable_shader_precompile
                }
            },
            "browser_capabilities": {},
            "webnn_benchmarks": {},
            "webgpu_benchmarks": {},
            "cross_browser_comparisons": {},
            "optimization_tests": {},
            "coverage_summary": {}
        }
        
    def run_browser_capability_test(self, browser: str) -> Dict:
        """Run browser capability check for a specific browser.
        
        Args:
            browser: Browser to test.
            
        Returns:
            Dictionary with test results.
        """
        print(f"[{browser}] Running browser capability check...")
        
        output_file = os.path.join(self.output_dir, f"{browser}_capabilities.json")
        
        cmd = [
            "./run_browser_capability_check.sh",
            f"--browser={browser}",
            f"--output={output_file}"
        ]
        
        try:
            output = subprocess.check_output(cmd, timeout=self.timeout, stderr=subprocess.STDOUT)
            output_str = output.decode('utf-8')
            
            # Parse output to get capabilities
            webnn_available = "WebNN: Available" in output_str
            webgpu_available = "WebGPU: Available" in output_str
            hw_acceleration = "Hardware Acceleration: Enabled" in output_str
            
            # Load detailed results from JSON file if available
            detailed_results = {}
            if os.path.exists(output_file):
                try:
                    with open(output_file, 'r') as f:
                        detailed_results = json.load(f)
                except json.JSONDecodeError:
                    pass
            
            results = {
                "browser": browser,
                "webnn_available": webnn_available,
                "webgpu_available": webgpu_available,
                "hardware_acceleration": hw_acceleration,
                "details": detailed_results,
                "error": None
            }
            
            print(f"[{browser}] Capability check completed: WebNN={webnn_available}, WebGPU={webgpu_available}, HW Accel={hw_acceleration}")
            
            return results
            
        except subprocess.CalledProcessError as e:
            print(f"[{browser}] Error running capability check: {e}")
            error_output = e.output.decode('utf-8') if e.output else "No output"
            return {
                "browser": browser,
                "webnn_available": False,
                "webgpu_available": False,
                "hardware_acceleration": False,
                "error": str(e),
                "error_output": error_output
            }
        except subprocess.TimeoutExpired:
            print(f"[{browser}] Capability check timed out after {self.timeout}s")
            return {
                "browser": browser,
                "webnn_available": False,
                "webgpu_available": False,
                "hardware_acceleration": False,
                "error": f"Timeout after {self.timeout}s"
            }
            
    def run_webnn_benchmark(self, browser: str, model: str, batch_size: int = 1) -> Dict:
        """Run WebNN benchmark for a specific browser, model, and batch size.
        
        Args:
            browser: Browser to test.
            model: Model to test.
            batch_size: Batch size to test.
            
        Returns:
            Dictionary with benchmark results.
        """
        print(f"[{browser}] Running WebNN benchmark for {model} (batch_size={batch_size})...")
        
        output_file = os.path.join(self.output_dir, f"{browser}_{model.replace('/', '_')}_{batch_size}_webnn.json")
        
        cmd = [
            "./run_webnn_benchmark.sh",
            f"--browser={browser}",
            f"--model={model}",
            f"--batch-size={batch_size}",
            f"--output-dir={self.output_dir}"
        ]
        
        try:
            output = subprocess.check_output(cmd, timeout=self.timeout, stderr=subprocess.STDOUT)
            output_str = output.decode('utf-8')
            
            # Parse output to get benchmark results
            # Look for patterns like "CPU Time: 120.45 ms" and "WebNN Time: 42.33 ms"
            cpu_time = None
            webnn_time = None
            speedup = None
            simulation = None
            
            for line in output_str.split('\n'):
                if "CPU Time:" in line:
                    try:
                        cpu_time = float(line.split("CPU Time:")[1].strip().split()[0])
                    except (ValueError, IndexError):
                        pass
                elif "WebNN Time:" in line:
                    try:
                        webnn_time = float(line.split("WebNN Time:")[1].strip().split()[0])
                    except (ValueError, IndexError):
                        pass
                elif "Speedup:" in line:
                    try:
                        speedup = float(line.split("Speedup:")[1].strip().split('x')[0])
                    except (ValueError, IndexError):
                        pass
                elif "Simulation:" in line:
                    simulation = "True" in line
            
            # Look for success/status code indicators
            webnn_status = "not_supported"
            if "SUCCESS: WebNN is enabled and using real hardware acceleration" in output_str:
                webnn_status = "real_hardware"
            elif "WARNING: WebNN is supported but using simulation mode" in output_str:
                webnn_status = "simulation"
            
            # Try to load detailed results from output file
            detailed_results = {}
            result_files = [f for f in os.listdir(self.output_dir) if f.startswith(f"webnn_benchmark_{browser}_{model.replace('/', '_')}")]
            if result_files:
                result_file = os.path.join(self.output_dir, sorted(result_files)[-1])  # Get the most recent file
                try:
                    with open(result_file, 'r') as f:
                        detailed_results = json.load(f)
                except (json.JSONDecodeError, FileNotFoundError):
                    pass
            
            results = {
                "browser": browser,
                "model": model,
                "batch_size": batch_size,
                "cpu_time": cpu_time,
                "webnn_time": webnn_time,
                "speedup": speedup,
                "simulation": simulation,
                "status": webnn_status,
                "details": detailed_results,
                "error": None
            }
            
            print(f"[{browser}] WebNN benchmark completed for {model}: CPU={cpu_time}ms, WebNN={webnn_time}ms, Speedup={speedup}x")
            
            return results
            
        except subprocess.CalledProcessError as e:
            print(f"[{browser}] Error running WebNN benchmark for {model}: {e}")
            error_output = e.output.decode('utf-8') if e.output else "No output"
            return {
                "browser": browser,
                "model": model,
                "batch_size": batch_size,
                "status": "error",
                "error": str(e),
                "error_output": error_output
            }
        except subprocess.TimeoutExpired:
            print(f"[{browser}] WebNN benchmark for {model} timed out after {self.timeout}s")
            return {
                "browser": browser,
                "model": model,
                "batch_size": batch_size,
                "status": "timeout",
                "error": f"Timeout after {self.timeout}s"
            }
            
    def run_webgpu_browser_comparison(self, model: str, browser: str = None) -> Dict:
        """Run WebGPU browser comparison test for a specific model.
        
        Args:
            model: Model to test.
            browser: Specific browser to test, or None to test all browsers.
            
        Returns:
            Dictionary with comparison results.
        """
        print(f"Running WebGPU browser comparison for {model}...")
        
        output_file = os.path.join(self.output_dir, f"{model.replace('/', '_')}_webgpu_comparison.json")
        
        # Determine if this is an audio model that benefits from Firefox compute shaders
        is_audio = any(audio_model in model for audio_model in ["whisper", "wav2vec2", "clap"])
        is_multimodal = any(mm_model in model for mm_model in ["clip", "llava"])
        
        # Base command
        cmd = [
            "python", "test_webgpu_browsers_comparison.py"
        ]
        
        # Add browser if specified
        if browser:
            cmd.extend([f"--browser={browser}"])
        else:
            cmd.extend(["--all-browsers"])
            
        # Add model
        cmd.extend([f"--model={model}"])
        
        # Add optimization flags based on model type
        if is_audio:
            cmd.extend(["--enable-compute-shaders", "--enable-shader-precompile"])
        elif is_multimodal:
            cmd.extend(["--enable-parallel-loading", "--enable-shader-precompile"])
        else:
            cmd.extend(["--enable-shader-precompile"])
            
        # Add output file
        cmd.extend([f"--report-output={output_file}"])
        
        try:
            output = subprocess.check_output(cmd, timeout=self.timeout, stderr=subprocess.STDOUT)
            output_str = output.decode('utf-8')
            
            # Load detailed results from output file
            detailed_results = {}
            if os.path.exists(output_file):
                try:
                    with open(output_file, 'r') as f:
                        detailed_results = json.load(f)
                except (json.JSONDecodeError, FileNotFoundError):
                    pass
                    
            # Extract key performance metrics from the output
            firefox_perf = {}
            chrome_perf = {}
            edge_perf = {}
            
            # Parse browser comparison info from output
            if is_audio and "Firefox is" in output_str and "faster than Chrome" in output_str:
                # Extract percentage
                try:
                    perf_line = [line for line in output_str.split('\n') if "Firefox is" in line and "faster than Chrome" in line][0]
                    percentage = float(perf_line.split("Firefox is")[1].split("%")[0].strip())
                    firefox_perf["vs_chrome_pct"] = percentage
                except (IndexError, ValueError):
                    pass
            
            results = {
                "model": model,
                "is_audio": is_audio,
                "is_multimodal": is_multimodal,
                "firefox_performance": firefox_perf,
                "chrome_performance": chrome_perf,
                "edge_performance": edge_perf,
                "details": detailed_results,
                "error": None
            }
            
            print(f"WebGPU browser comparison completed for {model}")
            
            return results
            
        except subprocess.CalledProcessError as e:
            print(f"Error running WebGPU browser comparison for {model}: {e}")
            error_output = e.output.decode('utf-8') if e.output else "No output"
            return {
                "model": model,
                "error": str(e),
                "error_output": error_output
            }
        except subprocess.TimeoutExpired:
            print(f"WebGPU browser comparison for {model} timed out after {self.timeout}s")
            return {
                "model": model,
                "error": f"Timeout after {self.timeout}s"
            }
    
    def run_optimization_test(self, optimization: str, model: str, browser: str = "firefox") -> Dict:
        """Run a specific optimization test.
        
        Args:
            optimization: Optimization to test (compute_shaders, parallel_loading, shader_precompile).
            model: Model to test.
            browser: Browser to test.
            
        Returns:
            Dictionary with optimization test results.
        """
        print(f"[{browser}] Running {optimization} optimization test for {model}...")
        
        # Map optimization to flag
        opt_flag_map = {
            "compute_shaders": "--enable-compute-shaders",
            "parallel_loading": "--enable-parallel-loading",
            "shader_precompile": "--enable-shader-precompile"
        }
        
        opt_flag = opt_flag_map.get(optimization)
        if not opt_flag:
            return {
                "browser": browser,
                "model": model,
                "optimization": optimization,
                "error": f"Unknown optimization: {optimization}"
            }
        
        # Run the optimization test
        cmd = [
            "python", "test_web_platform_optimizations.py",
            f"--browser={browser}",
            f"--model={model}",
            opt_flag
        ]
        
        try:
            output = subprocess.check_output(cmd, timeout=self.timeout, stderr=subprocess.STDOUT)
            output_str = output.decode('utf-8')
            
            # Parse optimization improvement from output
            improvement_pct = None
            baseline_time = None
            optimized_time = None
            
            if "Improvement:" in output_str:
                try:
                    improvement_line = [line for line in output_str.split('\n') if "Improvement:" in line][0]
                    improvement_pct = float(improvement_line.split("Improvement:")[1].split("%")[0].strip())
                except (IndexError, ValueError):
                    pass
                    
            if "Baseline:" in output_str:
                try:
                    baseline_line = [line for line in output_str.split('\n') if "Baseline:" in line][0]
                    baseline_time = float(baseline_line.split("Baseline:")[1].split("ms")[0].strip())
                except (IndexError, ValueError):
                    pass
                    
            if "Optimized:" in output_str:
                try:
                    optimized_line = [line for line in output_str.split('\n') if "Optimized:" in line][0]
                    optimized_time = float(optimized_line.split("Optimized:")[1].split("ms")[0].strip())
                except (IndexError, ValueError):
                    pass
            
            results = {
                "browser": browser,
                "model": model,
                "optimization": optimization,
                "improvement_pct": improvement_pct,
                "baseline_time": baseline_time,
                "optimized_time": optimized_time,
                "error": None
            }
            
            print(f"[{browser}] {optimization} optimization test completed for {model}: Improvement={improvement_pct}%")
            
            return results
            
        except subprocess.CalledProcessError as e:
            print(f"[{browser}] Error running {optimization} optimization test for {model}: {e}")
            error_output = e.output.decode('utf-8') if e.output else "No output"
            return {
                "browser": browser,
                "model": model,
                "optimization": optimization,
                "error": str(e),
                "error_output": error_output
            }
        except subprocess.TimeoutExpired:
            print(f"[{browser}] {optimization} optimization test for {model} timed out after {self.timeout}s")
            return {
                "browser": browser,
                "model": model,
                "optimization": optimization,
                "error": f"Timeout after {self.timeout}s"
            }
    
    def run_all_tests(self) -> Dict:
        """Run all WebNN and WebGPU tests and collect results.
        
        Returns:
            Dictionary with all test results.
        """
        start_time = time.time()
        print(f"Starting comprehensive WebNN/WebGPU test coverage run at {time.ctime()}")
        print(f"Testing browsers: {', '.join(self.browsers)}")
        print(f"Testing models: {', '.join(self.models)}")
        print(f"Testing batch sizes: {', '.join(map(str, self.batch_sizes))}")
        print(f"Results will be stored in: {self.output_dir}")
        print("-" * 80)
        
        # Step 1: Run browser capability checks
        for browser in self.browsers:
            self.results["browser_capabilities"][browser] = self.run_browser_capability_test(browser)
        
        # Step 2: Run WebNN benchmarks
        if self.parallel:
            # Run tests in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_concurrent) as executor:
                # Create a list of all benchmark tasks
                benchmark_tasks = []
                
                for browser in self.browsers:
                    # Skip browsers without WebNN support
                    if not self.results["browser_capabilities"][browser].get("webnn_available", False):
                        print(f"Skipping WebNN benchmarks for {browser} as WebNN is not available")
                        continue
                    
                    # Prioritize different models for different browsers for better coverage
                    if browser == "edge":
                        # Edge has best WebNN support - test all models
                        browser_models = self.models
                    elif browser == "chrome":
                        # Chrome has good WebNN support too
                        browser_models = self.models
                    elif browser == "firefox":
                        # Firefox doesn't fully support WebNN, focus on WebGPU with audio models
                        browser_models = [m for m in self.models if m in AUDIO_MODELS]
                    else:
                        # Other browsers - test a subset
                        browser_models = self.models[:1] if self.models else []
                    
                    for model in browser_models:
                        for batch_size in self.batch_sizes:
                            benchmark_tasks.append(
                                executor.submit(
                                    self.run_webnn_benchmark,
                                    browser=browser,
                                    model=model,
                                    batch_size=batch_size
                                )
                            )
                
                # Process results as they complete
                for task in concurrent.futures.as_completed(benchmark_tasks):
                    result = task.result()
                    browser = result["browser"]
                    model = result["model"]
                    batch_size = result["batch_size"]
                    
                    if browser not in self.results["webnn_benchmarks"]:
                        self.results["webnn_benchmarks"][browser] = {}
                        
                    if model not in self.results["webnn_benchmarks"][browser]:
                        self.results["webnn_benchmarks"][browser][model] = {}
                        
                    self.results["webnn_benchmarks"][browser][model][str(batch_size)] = result
        else:
            # Run tests sequentially
            for browser in self.browsers:
                # Skip browsers without WebNN support
                if not self.results["browser_capabilities"][browser].get("webnn_available", False):
                    print(f"Skipping WebNN benchmarks for {browser} as WebNN is not available")
                    continue
                    
                self.results["webnn_benchmarks"][browser] = {}
                
                for model in self.models:
                    self.results["webnn_benchmarks"][browser][model] = {}
                    
                    for batch_size in self.batch_sizes:
                        self.results["webnn_benchmarks"][browser][model][str(batch_size)] = \
                            self.run_webnn_benchmark(browser=browser, model=model, batch_size=batch_size)
        
        # Step 3: Run WebGPU browser comparisons
        # This is best done sequentially due to the complexity
        for model in self.models:
            self.results["cross_browser_comparisons"][model] = self.run_webgpu_browser_comparison(model=model)
        
        # Step 4: Run optimization tests
        if self.parallel:
            # Run tests in parallel with optimizations enabled based on flags
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_concurrent) as executor:
                # Create a list of all optimization tasks
                optimization_tasks = []
                
                # Compute shader optimization (best on Firefox for audio models)
                if self.enable_compute_shaders:
                    audio_models = [m for m in self.models if any(audio in m for audio in ["whisper", "wav2vec2", "clap"])]
                    for model in audio_models:
                        optimization_tasks.append(
                            executor.submit(
                                self.run_optimization_test,
                                optimization="compute_shaders",
                                model=model,
                                browser="firefox"
                            )
                        )
                    
                    # Also test Chrome for comparison if we have multiple models
                    if len(audio_models) > 1:
                        optimization_tasks.append(
                            executor.submit(
                                self.run_optimization_test,
                                optimization="compute_shaders",
                                model=audio_models[0],  # Just use the first model
                                browser="chrome"
                            )
                        )
                
                # Parallel loading optimization (for multimodal models)
                if self.enable_parallel_loading:
                    multimodal_models = [m for m in self.models if any(mm in m for mm in ["clip", "llava"])]
                    for model in multimodal_models:
                        optimization_tasks.append(
                            executor.submit(
                                self.run_optimization_test,
                                optimization="parallel_loading",
                                model=model,
                                browser="chrome"
                            )
                        )
                        
                        # Also test Edge for comparison if we have multiple models
                        if len(multimodal_models) > 1:
                            optimization_tasks.append(
                                executor.submit(
                                    self.run_optimization_test,
                                    optimization="parallel_loading",
                                    model=multimodal_models[0],  # Just use the first model
                                    browser="edge"
                                )
                            )
                
                # Shader precompilation (for all models)
                if self.enable_shader_precompile:
                    # Pick a subset of models to avoid too many tests
                    test_models = self.models[:3] if len(self.models) > 3 else self.models
                    
                    for browser in ["chrome", "firefox", "edge"]:
                        # Skip browsers that aren't in our test list
                        if browser not in self.browsers:
                            continue
                            
                        # Test one model per browser
                        if test_models:
                            optimization_tasks.append(
                                executor.submit(
                                    self.run_optimization_test,
                                    optimization="shader_precompile",
                                    model=test_models[0],
                                    browser=browser
                                )
                            )
                
                # Process results as they complete
                for task in concurrent.futures.as_completed(optimization_tasks):
                    result = task.result()
                    optimization = result["optimization"]
                    model = result["model"]
                    
                    if optimization not in self.results["optimization_tests"]:
                        self.results["optimization_tests"][optimization] = {}
                        
                    self.results["optimization_tests"][optimization][model] = result
        else:
            # Run tests sequentially
            self.results["optimization_tests"] = {
                "compute_shaders": {},
                "parallel_loading": {},
                "shader_precompile": {}
            }
            
            # Compute shader optimization (best on Firefox for audio models)
            for model in [m for m in self.models if any(audio in m for audio in ["whisper", "wav2vec2", "clap"])]:
                self.results["optimization_tests"]["compute_shaders"][model] = \
                    self.run_optimization_test(optimization="compute_shaders", model=model, browser="firefox")
            
            # Parallel loading optimization (for multimodal models)
            for model in [m for m in self.models if any(mm in m for mm in ["clip", "llava"])]:
                self.results["optimization_tests"]["parallel_loading"][model] = \
                    self.run_optimization_test(optimization="parallel_loading", model=model, browser="chrome")
            
            # Shader precompilation (for all models)
            for model in self.models:
                self.results["optimization_tests"]["shader_precompile"][model] = \
                    self.run_optimization_test(optimization="shader_precompile", model=model, browser="chrome")
        
        # Step 5: Generate coverage summary
        self.generate_coverage_summary()
        
        # Save complete results to file
        output_file = os.path.join(self.output_dir, f"webnn_coverage_results_{int(time.time())}.json")
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"Complete results saved to {output_file}")
        
        # Generate report
        report_file = os.path.join(self.output_dir, f"webnn_coverage_report_{int(time.time())}.md")
        report = self.generate_markdown_report()
        with open(report_file, 'w') as f:
            f.write(report)
        print(f"Report saved to {report_file}")
        
        end_time = time.time()
        duration = end_time - start_time
        print(f"Test coverage run completed in {duration:.1f} seconds")
        
        return self.results
    
    def generate_coverage_summary(self) -> None:
        """Generate a summary of test coverage.
        
        Updates the results dictionary with coverage statistics.
        """
        summary = {
            "browsers_tested": len(self.browsers),
            "models_tested": len(self.models),
            "total_tests_run": 0,
            "successful_tests": 0,
            "failed_tests": 0,
            "browser_support": {},
            "model_coverage": {},
            "optimization_impact": {}
        }
        
        # Count tests and track success/failure
        test_count = 0
        success_count = 0
        failure_count = 0
        
        # Browser capability tests
        test_count += len(self.results["browser_capabilities"])
        for browser, result in self.results["browser_capabilities"].items():
            if not result.get("error"):
                success_count += 1
            else:
                failure_count += 1
        
        # WebNN benchmark tests
        for browser, models in self.results["webnn_benchmarks"].items():
            for model, batch_results in models.items():
                for batch_size, result in batch_results.items():
                    test_count += 1
                    if not result.get("error"):
                        success_count += 1
                    else:
                        failure_count += 1
        
        # WebGPU browser comparison tests
        test_count += len(self.results["cross_browser_comparisons"])
        for model, result in self.results["cross_browser_comparisons"].items():
            if not result.get("error"):
                success_count += 1
            else:
                failure_count += 1
        
        # Optimization tests
        for optimization, models in self.results["optimization_tests"].items():
            for model, result in models.items():
                test_count += 1
                if not result.get("error"):
                    success_count += 1
                else:
                    failure_count += 1
        
        summary["total_tests_run"] = test_count
        summary["successful_tests"] = success_count
        summary["failed_tests"] = failure_count
        
        # Browser support summary
        for browser in self.browsers:
            capabilities = self.results["browser_capabilities"].get(browser, {})
            summary["browser_support"][browser] = {
                "webnn_available": capabilities.get("webnn_available", False),
                "webgpu_available": capabilities.get("webgpu_available", False),
                "hardware_acceleration": capabilities.get("hardware_acceleration", False)
            }
        
        # Model coverage summary
        for model in self.models:
            model_coverage = {
                "browsers_tested": [],
                "best_browser": None,
                "best_speedup": 0,
                "optimizations": {}
            }
            
            # Track browsers tested
            for browser in self.browsers:
                if browser in self.results["webnn_benchmarks"] and model in self.results["webnn_benchmarks"][browser]:
                    model_coverage["browsers_tested"].append(browser)
                    
                    # Check for best speedup
                    for batch_size, result in self.results["webnn_benchmarks"][browser][model].items():
                        speedup = result.get("speedup", 0) or 0
                        if speedup > model_coverage["best_speedup"]:
                            model_coverage["best_speedup"] = speedup
                            model_coverage["best_browser"] = browser
            
            # Track optimizations
            for optimization in ["compute_shaders", "parallel_loading", "shader_precompile"]:
                if optimization in self.results["optimization_tests"] and model in self.results["optimization_tests"][optimization]:
                    result = self.results["optimization_tests"][optimization][model]
                    model_coverage["optimizations"][optimization] = {
                        "tested": True,
                        "improvement_pct": result.get("improvement_pct")
                    }
                else:
                    model_coverage["optimizations"][optimization] = {
                        "tested": False
                    }
            
            summary["model_coverage"][model] = model_coverage
        
        # Optimization impact summary
        for optimization in ["compute_shaders", "parallel_loading", "shader_precompile"]:
            if optimization in self.results["optimization_tests"]:
                improvements = []
                for model, result in self.results["optimization_tests"][optimization].items():
                    improvement_pct = result.get("improvement_pct")
                    if improvement_pct is not None:
                        improvements.append(improvement_pct)
                
                if improvements:
                    summary["optimization_impact"][optimization] = {
                        "models_tested": len(self.results["optimization_tests"][optimization]),
                        "avg_improvement_pct": sum(improvements) / len(improvements),
                        "max_improvement_pct": max(improvements),
                        "min_improvement_pct": min(improvements)
                    }
        
        self.results["coverage_summary"] = summary
    
    def generate_markdown_report(self) -> str:
        """Generate a Markdown report of test results.
        
        Returns:
            Markdown report as a string.
        """
        report = "# WebNN and WebGPU Test Coverage Report\n\n"
        report += f"Generated on: {time.ctime(self.results['timestamp'])}\n\n"
        
        # Add coverage summary
        summary = self.results["coverage_summary"]
        report += "## Coverage Summary\n\n"
        report += f"- Browsers tested: {summary['browsers_tested']}\n"
        report += f"- Models tested: {summary['models_tested']}\n"
        report += f"- Total tests run: {summary['total_tests_run']}\n"
        report += f"- Successful tests: {summary['successful_tests']} ({summary['successful_tests'] / summary['total_tests_run'] * 100:.1f}%)\n"
        report += f"- Failed tests: {summary['failed_tests']} ({summary['failed_tests'] / summary['total_tests_run'] * 100:.1f}%)\n\n"
        
        # Browser support
        report += "## Browser Support\n\n"
        report += "| Browser | WebNN Available | WebGPU Available | Hardware Acceleration |\n"
        report += "|---------|----------------|------------------|----------------------|\n"
        
        for browser, support in summary["browser_support"].items():
            webnn = "✅" if support["webnn_available"] else "❌"
            webgpu = "✅" if support["webgpu_available"] else "❌"
            hw_accel = "✅" if support["hardware_acceleration"] else "❌"
            
            report += f"| {browser} | {webnn} | {webgpu} | {hw_accel} |\n"
        
        report += "\n"
        
        # WebNN performance by browser
        report += "## WebNN Performance\n\n"
        
        for browser in self.browsers:
            if browser not in self.results["webnn_benchmarks"]:
                continue
                
            report += f"### {browser.title()}\n\n"
            report += "| Model | Batch Size | CPU Time (ms) | WebNN Time (ms) | Speedup | Simulation |\n"
            report += "|-------|------------|--------------|-----------------|---------|------------|\n"
            
            for model in self.models:
                if model not in self.results["webnn_benchmarks"][browser]:
                    continue
                    
                for batch_size in self.batch_sizes:
                    batch_str = str(batch_size)
                    if batch_str not in self.results["webnn_benchmarks"][browser][model]:
                        continue
                        
                    result = self.results["webnn_benchmarks"][browser][model][batch_str]
                    
                    if result.get("error"):
                        report += f"| {model} | {batch_size} | Error: {result['error']} | - | - | - |\n"
                    else:
                        cpu_time = result.get("cpu_time", "N/A")
                        webnn_time = result.get("webnn_time", "N/A")
                        speedup = result.get("speedup", "N/A")
                        simulation = "Yes" if result.get("simulation", True) else "No"
                        
                        report += f"| {model} | {batch_size} | {cpu_time} | {webnn_time} | {speedup}x | {simulation} |\n"
            
            report += "\n"
        
        # Cross-browser WebGPU comparison
        report += "## Cross-Browser WebGPU Comparison\n\n"
        
        for model in self.models:
            if model not in self.results["cross_browser_comparisons"]:
                continue
                
            result = self.results["cross_browser_comparisons"][model]
            
            if result.get("error"):
                report += f"### {model}\n\n"
                report += f"Error: {result['error']}\n\n"
            else:
                report += f"### {model}\n\n"
                
                # Check for Firefox vs Chrome comparison for audio models
                if result.get("is_audio") and "vs_chrome_pct" in result.get("firefox_performance", {}):
                    pct = result["firefox_performance"]["vs_chrome_pct"]
                    report += f"Firefox is **{pct:.1f}%** faster than Chrome for this audio model\n\n"
        
        # Optimization impact
        report += "## Optimization Impact\n\n"
        
        # Compute shader optimization
        if "compute_shaders" in summary.get("optimization_impact", {}):
            opt_impact = summary["optimization_impact"]["compute_shaders"]
            report += "### WebGPU Compute Shader Optimization (Audio Models)\n\n"
            report += f"- Models tested: {opt_impact['models_tested']}\n"
            report += f"- Average improvement: {opt_impact['avg_improvement_pct']:.1f}%\n"
            report += f"- Maximum improvement: {opt_impact['max_improvement_pct']:.1f}%\n"
            report += f"- Minimum improvement: {opt_impact['min_improvement_pct']:.1f}%\n\n"
            
            report += "| Model | Improvement | Baseline (ms) | Optimized (ms) |\n"
            report += "|-------|-------------|--------------|---------------|\n"
            
            for model, result in self.results["optimization_tests"]["compute_shaders"].items():
                if result.get("error"):
                    report += f"| {model} | Error: {result['error']} | - | - |\n"
                else:
                    improvement = result.get("improvement_pct", "N/A")
                    baseline = result.get("baseline_time", "N/A")
                    optimized = result.get("optimized_time", "N/A")
                    
                    report += f"| {model} | {improvement}% | {baseline} | {optimized} |\n"
            
            report += "\n"
        
        # Parallel loading optimization
        if "parallel_loading" in summary.get("optimization_impact", {}):
            opt_impact = summary["optimization_impact"]["parallel_loading"]
            report += "### WebGPU Parallel Loading Optimization (Multimodal Models)\n\n"
            report += f"- Models tested: {opt_impact['models_tested']}\n"
            report += f"- Average improvement: {opt_impact['avg_improvement_pct']:.1f}%\n"
            report += f"- Maximum improvement: {opt_impact['max_improvement_pct']:.1f}%\n"
            report += f"- Minimum improvement: {opt_impact['min_improvement_pct']:.1f}%\n\n"
            
            report += "| Model | Improvement | Baseline (ms) | Optimized (ms) |\n"
            report += "|-------|-------------|--------------|---------------|\n"
            
            for model, result in self.results["optimization_tests"]["parallel_loading"].items():
                if result.get("error"):
                    report += f"| {model} | Error: {result['error']} | - | - |\n"
                else:
                    improvement = result.get("improvement_pct", "N/A")
                    baseline = result.get("baseline_time", "N/A")
                    optimized = result.get("optimized_time", "N/A")
                    
                    report += f"| {model} | {improvement}% | {baseline} | {optimized} |\n"
            
            report += "\n"
        
        # Shader precompilation optimization
        if "shader_precompile" in summary.get("optimization_impact", {}):
            opt_impact = summary["optimization_impact"]["shader_precompile"]
            report += "### WebGPU Shader Precompilation Optimization\n\n"
            report += f"- Models tested: {opt_impact['models_tested']}\n"
            report += f"- Average improvement: {opt_impact['avg_improvement_pct']:.1f}%\n"
            report += f"- Maximum improvement: {opt_impact['max_improvement_pct']:.1f}%\n"
            report += f"- Minimum improvement: {opt_impact['min_improvement_pct']:.1f}%\n\n"
            
            report += "| Model | Improvement | Baseline (ms) | Optimized (ms) |\n"
            report += "|-------|-------------|--------------|---------------|\n"
            
            for model, result in self.results["optimization_tests"]["shader_precompile"].items():
                if result.get("error"):
                    report += f"| {model} | Error: {result['error']} | - | - |\n"
                else:
                    improvement = result.get("improvement_pct", "N/A")
                    baseline = result.get("baseline_time", "N/A")
                    optimized = result.get("optimized_time", "N/A")
                    
                    report += f"| {model} | {improvement}% | {baseline} | {optimized} |\n"
            
            report += "\n"
        
        # Model coverage summary
        report += "## Model Coverage Summary\n\n"
        report += "| Model | Browsers Tested | Best Browser | Best Speedup | Optimizations Tested |\n"
        report += "|-------|----------------|-------------|-------------|---------------------|\n"
        
        for model, coverage in summary["model_coverage"].items():
            browsers_tested = ", ".join(coverage["browsers_tested"]) or "None"
            best_browser = coverage.get("best_browser", "N/A")
            best_speedup = f"{coverage.get('best_speedup', 0):.2f}x" if coverage.get("best_speedup", 0) > 0 else "N/A"
            
            optimizations_tested = []
            for opt, opt_result in coverage.get("optimizations", {}).items():
                if opt_result.get("tested", False):
                    optimizations_tested.append(opt)
            
            opt_str = ", ".join(optimizations_tested) or "None"
            
            report += f"| {model} | {browsers_tested} | {best_browser} | {best_speedup} | {opt_str} |\n"
        
        report += "\n"
        
        # Recommendations
        report += "## Recommendations\n\n"
        
        # Browser recommendations
        report += "### Browser Recommendations\n\n"
        
        edge_webnn = summary["browser_support"].get("edge", {}).get("webnn_available", False)
        chrome_webnn = summary["browser_support"].get("chrome", {}).get("webnn_available", False)
        firefox_webgpu = summary["browser_support"].get("firefox", {}).get("webgpu_available", False)
        
        if edge_webnn:
            report += "- For WebNN acceleration, Microsoft Edge provides the best and most complete implementation\n"
        elif chrome_webnn:
            report += "- For WebNN acceleration, Google Chrome provides good support\n"
        
        if firefox_webgpu and "compute_shaders" in summary.get("optimization_impact", {}):
            report += "- For audio models (Whisper, Wav2Vec2, CLAP), Firefox with compute shader optimization provides superior performance\n"
        
        # Model type recommendations
        report += "\n### Model Type Recommendations\n\n"
        
        report += "- **Text Models** (BERT, T5): Enable shader precompilation for faster first inference\n"
        report += "- **Vision Models** (ViT, ResNet): Enable shader precompilation for faster first inference\n"
        report += "- **Audio Models** (Whisper, Wav2Vec2): Use Firefox with compute shader optimization for best performance\n"
        report += "- **Multimodal Models** (CLIP, LLaVA): Enable parallel loading to reduce initialization time\n"
        
        # Optimization combinations
        report += "\n### Optimization Combinations\n\n"
        
        report += "- For audio models in Firefox:\n"
        report += "  ```bash\n"
        report += "  ./run_web_platform_tests.sh --firefox --model whisper-tiny --enable-compute-shaders --enable-shader-precompile\n"
        report += "  ```\n\n"
        
        report += "- For multimodal models in Chrome/Edge:\n"
        report += "  ```bash\n"
        report += "  ./run_web_platform_tests.sh --browser chrome --model clip --enable-parallel-loading --enable-shader-precompile\n"
        report += "  ```\n\n"
        
        report += "- For text/vision models in any browser:\n"
        report += "  ```bash\n"
        report += "  ./run_web_platform_tests.sh --browser edge --model bert --enable-shader-precompile\n"
        report += "  ```\n"
        
        return report

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="WebNN Comprehensive Test Coverage Tool")
    
    # Browser options
    browser_group = parser.add_argument_group('Browser options')
    browser_group.add_argument("--browser", type=str, help="Browser to test (chrome, edge, firefox, safari)")
    browser_group.add_argument("--browsers", type=str, nargs='+', help="List of browsers to test")
    browser_group.add_argument("--all-browsers", action="store_true", help="Test all supported browsers")
    
    # Model options
    model_group = parser.add_argument_group('Model options')
    model_group.add_argument("--model", type=str, help="Model to test")
    model_group.add_argument("--models", type=str, nargs='+', help="List of models to test")
    model_group.add_argument("--all-models", action="store_true", help="Test all supported models")
    model_group.add_argument("--audio-models-only", action="store_true", help="Test only audio models (good for testing Firefox optimization)")
    model_group.add_argument("--multimodal-models-only", action="store_true", help="Test only multimodal models (good for testing parallel loading)")
    model_group.add_argument("--batch-sizes", type=int, nargs='+', help="List of batch sizes to test")
    
    # Optimization options
    optimization_group = parser.add_argument_group('Optimization options')
    optimization_group.add_argument("--compute-shaders", action="store_true", help="Enable compute shader optimizations")
    optimization_group.add_argument("--parallel-loading", action="store_true", help="Enable parallel loading optimizations")
    optimization_group.add_argument("--shader-precompile", action="store_true", help="Enable shader precompilation")
    optimization_group.add_argument("--all-optimizations", action="store_true", help="Enable all optimizations")
    
    # Test scope options
    scope_group = parser.add_argument_group('Test scope options')
    scope_group.add_argument("--all", action="store_true", help="Test all browsers, models and optimizations")
    scope_group.add_argument("--quick", action="store_true", help="Run a quick test with minimal configuration")
    scope_group.add_argument("--firefox-audio-only", action="store_true", help="Test only Firefox with audio models (compute shader optimization)")
    scope_group.add_argument("--capabilities-only", action="store_true", help="Only check browser capabilities without benchmarks")
    
    # Output options
    output_group = parser.add_argument_group('Output options')
    output_group.add_argument("--output-dir", type=str, default="./webnn_coverage_results", help="Directory to store test results")
    output_group.add_argument("--db-path", type=str, help="Path to benchmark database")
    output_group.add_argument("--report-format", type=str, choices=["markdown", "html"], default="markdown", help="Format of the generated report")
    output_group.add_argument("--report-file", type=str, help="File to save the report to (default: webnn_coverage_report_{timestamp}.{format})")
    
    # Execution options
    execution_group = parser.add_argument_group('Execution options')
    execution_group.add_argument("--timeout", type=int, default=600, help="Timeout in seconds for each test")
    execution_group.add_argument("--sequential", action="store_true", help="Run tests sequentially")
    execution_group.add_argument("--max-concurrent", type=int, default=2, help="Maximum number of concurrent tests to run")
    
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()
    
    # Special test configurations
    if args.quick:
        # Quick test with minimal configuration
        browsers = ["edge"]  # Best WebNN support
        models = ["prajjwal1/bert-tiny"]  # Small, fast model
        batch_sizes = [1]
        enable_optimizations = False
        capabilities_only = False
        max_concurrent = 1
        
    elif args.firefox_audio_only:
        # Test Firefox with audio models for compute shader optimization
        browsers = ["firefox"]
        models = AUDIO_MODELS
        batch_sizes = [1]
        enable_optimizations = True
        capabilities_only = False
        max_concurrent = 1
        
    elif args.capabilities_only:
        # Only check browser capabilities
        browsers = args.all_browsers and SUPPORTED_BROWSERS or (
            args.browsers or ([args.browser] if args.browser else ["chrome", "edge", "firefox"])
        )
        models = []
        batch_sizes = []
        enable_optimizations = False
        capabilities_only = True
        max_concurrent = len(browsers)
        
    else:
        # Standard test configuration based on arguments
        
        # Determine browsers to test
        if args.all or args.all_browsers:
            browsers = SUPPORTED_BROWSERS
        elif args.browsers:
            browsers = args.browsers
        elif args.browser:
            browsers = [args.browser]
        else:
            browsers = ["edge", "chrome"]  # Default browsers
        
        # Determine models to test
        if args.all or args.all_models:
            models = SUPPORTED_MODELS
        elif args.audio_models_only:
            models = AUDIO_MODELS
        elif args.multimodal_models_only:
            models = MULTIMODAL_MODELS
        elif args.models:
            models = args.models
        elif args.model:
            models = [args.model]
        else:
            # Default to a mix of model types
            models = ["prajjwal1/bert-tiny", "whisper-tiny", "openai/clip-vit-base-patch32"]
        
        batch_sizes = args.batch_sizes
        enable_optimizations = args.all_optimizations or args.compute_shaders or args.parallel_loading or args.shader_precompile
        capabilities_only = False
        max_concurrent = args.max_concurrent
    
    # Create test coverage runner
    runner = TestCoverageRunner(
        browsers=browsers,
        models=models,
        batch_sizes=batch_sizes,
        output_dir=args.output_dir,
        db_path=args.db_path,
        timeout=args.timeout,
        parallel=not args.sequential,
        max_concurrent=max_concurrent,
        enable_compute_shaders=args.compute_shaders or args.all_optimizations,
        enable_parallel_loading=args.parallel_loading or args.all_optimizations,
        enable_shader_precompile=args.shader_precompile or args.all_optimizations
    )
    
    # Run tests
    if capabilities_only:
        # Only run browser capability checks
        for browser in browsers:
            capability_results = runner.run_browser_capability_test(browser)
            runner.results["browser_capabilities"][browser] = capability_results
        
        # Generate coverage summary even with just capabilities
        runner.generate_coverage_summary()
    else:
        # Run full test suite
        runner.run_all_tests()
    
    # Generate and save report
    report_format = args.report_format or "markdown"
    if report_format == "markdown":
        report = runner.generate_markdown_report()
        extension = "md"
    else:
        report = runner._generate_html_report()
        extension = "html"
    
    # Save report to file
    if args.report_file:
        report_path = args.report_file
    else:
        timestamp = int(time.time())
        report_path = os.path.join(args.output_dir, f"webnn_coverage_report_{timestamp}.{extension}")
    
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"Test coverage report saved to: {report_path}")

if __name__ == "__main__":
    main()