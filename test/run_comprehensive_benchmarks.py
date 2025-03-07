#!/usr/bin/env python3
"""
Run Comprehensive Benchmarks

This script runs the comprehensive benchmarks for all available hardware platforms,
including CPU, CUDA, ROCm, MPS, OpenVINO, QNN, WebNN, and WebGPU. It executes benchmarks 
for a subset of models to make progress on item #9 from NEXT_STEPS.md.

Enhanced in April 2025 to add web testing environment support for WebNN and WebGPU benchmarks,
with features like compute shader optimization, parallel loading, and shader precompilation.

Usage:
    python run_comprehensive_benchmarks.py
    python run_comprehensive_benchmarks.py --models bert,t5,vit
    python run_comprehensive_benchmarks.py --hardware cpu,cuda
    python run_comprehensive_benchmarks.py --batch-sizes 1,4,16
    python run_comprehensive_benchmarks.py --force-hardware rocm,webgpu
    python run_comprehensive_benchmarks.py --report-format markdown
    
    # Web Platform Testing (April 2025)
    python run_comprehensive_benchmarks.py --setup-web-testing --browser chrome
    python run_comprehensive_benchmarks.py --web-compute-shaders --models whisper,wav2vec2
    python run_comprehensive_benchmarks.py --web-parallel-loading --models clip,llava
    python run_comprehensive_benchmarks.py --web-shader-precompile --models bert,vit
    python run_comprehensive_benchmarks.py --web-all-optimizations --models all
"""

import os
import sys
import subprocess
import logging
import argparse
import json
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("comprehensive_benchmarks_run.log")
    ]
)
logger = logging.getLogger(__name__)

# Define subset of models to benchmark
DEFAULT_MODELS = ["bert", "t5", "vit", "whisper"]

# Define all supported hardware platforms
ALL_HARDWARE = ["cpu", "cuda", "rocm", "mps", "openvino", "qnn", "webnn", "webgpu"]

# Define commonly available hardware platforms
DEFAULT_HARDWARE = ["cpu", "cuda"]

# Define web platform audio models
AUDIO_MODELS = ["whisper", "wav2vec2", "clap"]

# Define web platform multimodal models
MULTIMODAL_MODELS = ["clip", "llava", "llava_next", "xclip"]

def detect_available_hardware(try_advanced_detection=True, check_web_browsers=True):
    """
    Detect available hardware platforms.
    
    Args:
        try_advanced_detection: Whether to try using the advanced hardware detection module
        check_web_browsers: Whether to check for web browser availability for WebNN and WebGPU
        
    Returns:
        dict: Dictionary mapping hardware platform to availability status
    """
    available_hardware = {"cpu": True}  # CPU is always available
    
    # Try to use the advanced hardware detection if available
    if try_advanced_detection:
        try:
            # First try centralized hardware detection
            from centralized_hardware_detection.hardware_detection import detect_hardware_capabilities
            capabilities = detect_hardware_capabilities()
            logger.info("Using centralized hardware detection system")
            
            # Map capabilities to hardware availability
            available_hardware.update({
                "cuda": capabilities.get("cuda", {}).get("available", False),
                "rocm": capabilities.get("rocm", {}).get("available", False),
                "mps": capabilities.get("mps", {}).get("available", False),
                "openvino": capabilities.get("openvino", {}).get("available", False),
                "qnn": capabilities.get("qnn", {}).get("available", False),
                "webnn": capabilities.get("webnn", {}).get("available", False),
                "webgpu": capabilities.get("webgpu", {}).get("available", False)
            })
            
            # Log detected hardware
            for hw, available in available_hardware.items():
                if available:
                    logger.info(f"Detected {hw.upper()} as available")
            
            return available_hardware
        except ImportError:
            logger.info("Centralized hardware detection not available, falling back to basic detection")
    
    # Fallback to basic detection
    try:
        import torch
        if torch.cuda.is_available():
            available_hardware["cuda"] = True
            logger.info(f"CUDA is available with {torch.cuda.device_count()} devices")
        else:
            available_hardware["cuda"] = False
        
        # Check for MPS (Apple Silicon)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            available_hardware["mps"] = True
            logger.info("MPS (Apple Silicon) is available")
        else:
            available_hardware["mps"] = False
    except ImportError:
        logger.warning("PyTorch not available, assuming only CPU is available")
        available_hardware["cuda"] = False
        available_hardware["mps"] = False
    
    # Check for OpenVINO
    try:
        import openvino
        available_hardware["openvino"] = True
        logger.info(f"OpenVINO is available (version {openvino.__version__})")
    except ImportError:
        available_hardware["openvino"] = False
    
    # Check for ROCm via environment variable
    if os.environ.get("ROCM_HOME"):
        available_hardware["rocm"] = True
        logger.info("ROCm is available")
    else:
        available_hardware["rocm"] = False
    
    # Other platforms are less likely to be available by default
    available_hardware["qnn"] = False
    
    # Check for web browser availability if requested
    if check_web_browsers:
        webnn_available, webgpu_available = detect_web_browsers()
        available_hardware["webnn"] = webnn_available
        available_hardware["webgpu"] = webgpu_available
    else:
        available_hardware["webnn"] = False
        available_hardware["webgpu"] = False
    
    return available_hardware

def detect_web_browsers():
    """
    Detect available web browsers for WebNN and WebGPU platforms.
    
    Returns:
        Tuple[bool, bool]: (webnn_available, webgpu_available)
    """
    webnn_available = False
    webgpu_available = False
    
    try:
        # Try to import browser_automation module
        browser_automation_path = None
        possible_paths = [
            "fixed_web_platform.browser_automation",
            "fixed_web_platform/browser_automation.py",
            str(Path(__file__).parent / "fixed_web_platform" / "browser_automation.py")
        ]
        
        for path in possible_paths:
            try:
                if path.endswith('.py'):
                    # Try to import from file path
                    import importlib.util
                    spec = importlib.util.spec_from_file_location("browser_automation", path)
                    if spec and spec.loader:
                        browser_automation = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(browser_automation)
                        browser_automation_path = path
                        break
                else:
                    # Try to import from module path
                    browser_automation = __import__(path, fromlist=[''])
                    browser_automation_path = path
                    break
            except (ImportError, ModuleNotFoundError):
                continue
        
        if browser_automation_path:
            logger.info(f"Using browser automation module from: {browser_automation_path}")
            
            # Check for Edge browser (WebNN)
            edge_path = browser_automation.find_browser_executable("edge")
            if edge_path:
                logger.info(f"Found Edge browser at: {edge_path}")
                webnn_available = True
            
            # Check for Chrome browser (WebGPU)
            chrome_path = browser_automation.find_browser_executable("chrome")
            if chrome_path:
                logger.info(f"Found Chrome browser at: {chrome_path}")
                webgpu_available = True
            
            # Firefox can also support WebGPU
            if not webgpu_available:
                firefox_path = browser_automation.find_browser_executable("firefox")
                if firefox_path:
                    logger.info(f"Found Firefox browser at: {firefox_path}")
                    webgpu_available = True
        else:
            # Fallback to basic browser detection if module not found
            logger.warning("Browser automation module not found, using basic browser detection")
            
            # Check for browsers using 'which' command on Linux/macOS or checking paths on Windows
            if os.name == 'nt':  # Windows
                edge_paths = [
                    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"
                ]
                chrome_paths = [
                    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"
                ]
                firefox_paths = [
                    r"C:\Program Files\Mozilla Firefox\firefox.exe",
                    r"C:\Program Files (x86)\Mozilla Firefox\firefox.exe"
                ]
                
                for path in edge_paths:
                    if os.path.exists(path):
                        webnn_available = True
                        logger.info(f"Found Edge browser at: {path}")
                        break
                
                for path in chrome_paths + firefox_paths:
                    if os.path.exists(path):
                        webgpu_available = True
                        logger.info(f"Found WebGPU-capable browser at: {path}")
                        break
            else:  # Linux/macOS
                try:
                    # Check for browsers using 'which' command
                    for browser, platform in [("microsoft-edge", "webnn"), 
                                              ("google-chrome", "webgpu"),
                                              ("firefox", "webgpu")]:
                        try:
                            result = subprocess.run(["which", browser], 
                                                  stdout=subprocess.PIPE, 
                                                  stderr=subprocess.PIPE,
                                                  text=True)
                            if result.returncode == 0 and result.stdout.strip():
                                logger.info(f"Found {browser} at: {result.stdout.strip()}")
                                
                                if platform == "webnn":
                                    webnn_available = True
                                elif platform == "webgpu":
                                    webgpu_available = True
                        except subprocess.SubprocessError:
                            continue
                except Exception as e:
                    logger.warning(f"Error detecting browsers: {e}")
    
    except Exception as e:
        logger.error(f"Error during web browser detection: {e}")
    
    logger.info(f"Web browser detection results - WebNN: {'Available' if webnn_available else 'Not Available'}, "
               f"WebGPU: {'Available' if webgpu_available else 'Not Available'}")
    
    return webnn_available, webgpu_available

def run_benchmarks(models=None, hardware=None, batch_sizes=None, small_models=True, 
                  db_path=None, output_dir="./benchmark_results", timeout=None,
                  report_format="html", force_hardware=None):
    """
    Run comprehensive benchmarks for specified models and hardware.
    
    Args:
        models: List of models to benchmark (default: DEFAULT_MODELS)
        hardware: List of hardware platforms to benchmark (default: detect_available_hardware())
        batch_sizes: List of batch sizes to test (default: [1, 2, 4, 8, 16])
        small_models: Use smaller model variants for quicker testing
        db_path: Path to benchmark database
        output_dir: Directory to save results
        timeout: Timeout in seconds for each benchmark
        report_format: Output format for the report (html, markdown, json)
        force_hardware: List of hardware platforms to force even if not detected as available
        
    Returns:
        bool: True if benchmarks completed successfully
    """
    models = models or DEFAULT_MODELS
    batch_sizes = batch_sizes or [1, 2, 4, 8, 16]
    
    # Detect available hardware
    available_hardware_dict = detect_available_hardware()
    available_hardware = [hw for hw, available in available_hardware_dict.items() if available]
    
    # Determine hardware to benchmark
    if hardware:
        # User specified hardware
        hardware_to_benchmark = hardware
    else:
        # Use available hardware by default
        hardware_to_benchmark = available_hardware
    
    # Force specified hardware platforms if requested
    if force_hardware:
        for hw in force_hardware:
            if hw not in hardware_to_benchmark:
                hardware_to_benchmark.append(hw)
                logger.warning(f"Forcing benchmark on {hw} even though it may not be available")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Get database path
    if not db_path:
        db_path = os.environ.get("BENCHMARK_DB_PATH", "./benchmark_db.duckdb")
    
    # Convert batch sizes to string
    batch_sizes_str = ",".join(str(bs) for bs in batch_sizes)
    
    # Prepare command
    cmd = [
        sys.executable,
        "execute_comprehensive_benchmarks.py",
        "--run-all",
        "--small-models" if small_models else "",
        "--db-path", db_path,
        "--output-dir", str(output_dir),
        "--batch-sizes", batch_sizes_str,
        "--report-format", report_format,
    ]
    
    # Add model and hardware arguments
    cmd.append("--model")
    cmd.extend(models)
    
    cmd.append("--hardware")
    cmd.extend(hardware_to_benchmark)
    
    # Add timeout if specified
    if timeout:
        cmd.extend(["--timeout", str(timeout)])
    
    # Remove empty arguments
    cmd = [arg for arg in cmd if arg]
    
    # Convert to string for logging
    cmd_str = " ".join(cmd)
    logger.info(f"Running benchmark command: {cmd_str}")
    
    # Track benchmark status
    benchmark_status = {
        "timestamp": datetime.now().isoformat(),
        "models": models,
        "hardware": hardware_to_benchmark,
        "batch_sizes": batch_sizes,
        "small_models": small_models,
        "db_path": db_path,
        "output_dir": str(output_dir),
        "command": cmd_str,
        "status": "running",
        "start_time": datetime.now().isoformat()
    }
    
    # Save initial status
    status_file = output_dir / f"benchmark_status_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(status_file, 'w') as f:
        json.dump(benchmark_status, f, indent=2)
    
    # Execute the command
    try:
        start_time = datetime.now()
        logger.info(f"Starting benchmarks at {start_time}")
        
        # Run subprocess with real-time output
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # Collect output for status file
        output_lines = []
        
        # Print output in real-time
        for line in process.stdout:
            print(line, end='')
            logger.info(line.strip())
            output_lines.append(line.strip())
        
        # Wait for process to complete
        return_code = process.wait()
        
        end_time = datetime.now()
        duration = end_time - start_time
        duration_seconds = duration.total_seconds()
        
        # Update benchmark status
        benchmark_status.update({
            "status": "completed" if return_code == 0 else "failed",
            "return_code": return_code,
            "end_time": end_time.isoformat(),
            "duration_seconds": duration_seconds,
            "output_summary": output_lines[-20:] if len(output_lines) > 20 else output_lines
        })
        
        # Save updated status
        with open(status_file, 'w') as f:
            json.dump(benchmark_status, f, indent=2)
        
        if return_code == 0:
            logger.info(f"Benchmarks completed successfully in {duration}")
            
            # Generate the report
            report_output = output_dir / f"benchmark_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{report_format.lower()}"
            report_cmd = [
                sys.executable,
                "benchmark_timing_report.py",
                "--generate",
                "--db-path", db_path,
                "--format", report_format,
                "--output", str(report_output)
            ]
            
            logger.info(f"Generating report: {' '.join(report_cmd)}")
            try:
                report_process = subprocess.run(report_cmd, check=True, capture_output=True, text=True)
                
                # Update benchmark status with report info
                benchmark_status.update({
                    "report_path": str(report_output),
                    "report_generated": True
                })
                
                # Create a symlink to the latest report
                latest_report = output_dir / f"benchmark_report_latest.{report_format.lower()}"
                if latest_report.exists():
                    try:
                        latest_report.unlink()
                    except:
                        logger.warning(f"Could not remove existing symlink: {latest_report}")
                try:
                    latest_report.symlink_to(report_output.name)
                    logger.info(f"Created symlink to latest report: {latest_report}")
                except Exception as e:
                    logger.warning(f"Could not create symlink to latest report: {str(e)}")
                
                logger.info(f"Report generated successfully: {report_output}")
                print(f"\nBenchmark report generated: {report_output}")
                
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to generate report: {e.stderr}")
                benchmark_status.update({
                    "report_generated": False,
                    "report_error": e.stderr
                })
            
            # Save final status
            with open(status_file, 'w') as f:
                json.dump(benchmark_status, f, indent=2)
            
            # Also save to a "latest" file for easy access
            latest_status = output_dir / "benchmark_status_latest.json"
            with open(latest_status, 'w') as f:
                json.dump(benchmark_status, f, indent=2)
            
            return True
        else:
            logger.error(f"Benchmarks failed with return code {return_code}")
            
            # Save final status
            with open(status_file, 'w') as f:
                json.dump(benchmark_status, f, indent=2)
            
            return False
        
    except Exception as e:
        logger.error(f"Error running benchmarks: {str(e)}")
        
        # Update benchmark status with error
        benchmark_status.update({
            "status": "error",
            "error": str(e),
            "end_time": datetime.now().isoformat()
        })
        
        # Save error status
        with open(status_file, 'w') as f:
            json.dump(benchmark_status, f, indent=2)
        
        return False

def setup_web_testing_environment(browser=None, output_dir="./web_testing_env", 
                              debug=False) -> Dict[str, Any]:
    """
    Set up a web testing environment for WebNN and WebGPU platforms.
    
    Args:
        browser: Preferred browser ('edge', 'chrome', 'firefox') or None for auto-select
        output_dir: Directory to save web testing environment files
        debug: Enable debug mode
        
    Returns:
        dict: Dictionary with web testing environment details
    """
    logger.info(f"Setting up web testing environment in {output_dir}")
    
    result = {
        "status": "error",
        "web_testing_dir": output_dir,
        "webnn_available": False,
        "webgpu_available": False,
        "browsers": {},
        "env_vars": {}
    }
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Try to import browser_automation
        browser_automation = None
        try:
            # Try different import methods
            try:
                from fixed_web_platform.browser_automation import (
                    find_browser_executable,
                    get_browser_args,
                    create_test_html
                )
                browser_automation = True
                result["browser_automation_source"] = "fixed_web_platform.browser_automation"
            except ImportError:
                # Try to load from path
                module_path = Path(__file__).parent / "fixed_web_platform" / "browser_automation.py"
                if module_path.exists():
                    import importlib.util
                    spec = importlib.util.spec_from_file_location("browser_automation", module_path)
                    if spec and spec.loader:
                        browser_automation_module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(browser_automation_module)
                        find_browser_executable = browser_automation_module.find_browser_executable
                        get_browser_args = browser_automation_module.get_browser_args
                        create_test_html = browser_automation_module.create_test_html
                        browser_automation = True
                        result["browser_automation_source"] = str(module_path)
        except Exception as e:
            logger.error(f"Error importing browser automation: {e}")
            browser_automation = False
            
        if not browser_automation:
            logger.error("Browser automation module not found, cannot set up web testing environment")
            result["error"] = "Browser automation module not found"
            return result
            
        # Detect browsers - start with the specified browser if provided
        browsers_to_check = []
        if browser:
            browsers_to_check.append(browser)
            
        # Add other browsers to check
        if "edge" not in browsers_to_check:
            browsers_to_check.append("edge")
        if "chrome" not in browsers_to_check:
            browsers_to_check.append("chrome")
        if "firefox" not in browsers_to_check:
            browsers_to_check.append("firefox")
            
        # Check all browsers
        for browser_name in browsers_to_check:
            browser_path = find_browser_executable(browser_name)
            if browser_path:
                logger.info(f"Found {browser_name} at {browser_path}")
                result["browsers"][browser_name] = {
                    "path": browser_path,
                    "args": {
                        "webnn": get_browser_args("webnn", browser_name),
                        "webgpu": get_browser_args("webgpu", browser_name)
                    }
                }
                
                # Update availability
                if browser_name in ["edge", "chrome"] and not result["webnn_available"]:
                    result["webnn_available"] = True
                if browser_name in ["chrome", "edge", "firefox"] and not result["webgpu_available"]:
                    result["webgpu_available"] = True
        
        if not result["browsers"]:
            logger.error("No compatible browsers found for web testing")
            result["error"] = "No compatible browsers found"
            return result
            
        # Create test HTML files
        webnn_test_file = os.path.join(output_dir, "webnn_test.html")
        webgpu_test_file = os.path.join(output_dir, "webgpu_test.html")
        
        # Create basic test page for WebNN
        with open(webnn_test_file, 'w') as f:
            html = create_test_html("webnn", "text", "bert-base-uncased")
            if html:
                f.write(html)
                result["webnn_test_file"] = webnn_test_file
                
        # Create basic test page for WebGPU
        with open(webgpu_test_file, 'w') as f:
            html = create_test_html("webgpu", "text", "bert-base-uncased")
            if html:
                f.write(html)
                result["webgpu_test_file"] = webgpu_test_file
                
        # Create configuration file
        config_file = os.path.join(output_dir, "web_testing_config.json")
        with open(config_file, 'w') as f:
            config = {
                "webnn_available": result["webnn_available"],
                "webgpu_available": result["webgpu_available"],
                "browsers": result["browsers"],
                "test_files": {
                    "webnn": result["webnn_test_file"] if "webnn_test_file" in result else None,
                    "webgpu": result["webgpu_test_file"] if "webgpu_test_file" in result else None
                },
                "created_at": datetime.now().isoformat()
            }
            json.dump(config, f, indent=2)
            
        result["config_file"] = config_file
        
        # Create env vars for other tools
        result["env_vars"] = {
            "WEB_TESTING_DIR": output_dir,
            "WEB_TESTING_CONFIG": config_file,
            "WEBNN_AVAILABLE": "1" if result["webnn_available"] else "0",
            "WEBGPU_AVAILABLE": "1" if result["webgpu_available"] else "0"
        }
        
        # Create README file with instructions
        readme_file = os.path.join(output_dir, "README.md")
        with open(readme_file, 'w') as f:
            f.write(f"""# Web Testing Environment

Created: {datetime.now().isoformat()}

## Available Browsers
{', '.join(result['browsers'].keys())}

## Environment Status
- WebNN: {'✅ Available' if result['webnn_available'] else '❌ Not Available'}
- WebGPU: {'✅ Available' if result['webgpu_available'] else '❌ Not Available'}

## Usage with run_comprehensive_benchmarks.py
```bash
# For WebNN benchmarks
python run_comprehensive_benchmarks.py --models bert,t5 --hardware webnn --web-testing-dir {output_dir}

# For WebGPU benchmarks
python run_comprehensive_benchmarks.py --models bert,vit --hardware webgpu --web-testing-dir {output_dir}

# For WebGPU with compute shader optimization (audio models)
python run_comprehensive_benchmarks.py --models whisper,wav2vec2 --hardware webgpu --web-compute-shaders --web-testing-dir {output_dir}

# For WebGPU with parallel loading (multimodal models)
python run_comprehensive_benchmarks.py --models clip,llava --hardware webgpu --web-parallel-loading --web-testing-dir {output_dir}

# For WebGPU with shader precompilation (all models)
python run_comprehensive_benchmarks.py --models bert,vit --hardware webgpu --web-shader-precompile --web-testing-dir {output_dir}

# For all optimizations
python run_comprehensive_benchmarks.py --models all --hardware webgpu --web-all-optimizations --web-testing-dir {output_dir}
```

## Browser-Specific Features
- Edge: Best for WebNN
- Chrome: Best for general WebGPU
- Firefox: Best for WebGPU audio models (with compute shaders)
""")
            
        result["readme_file"] = readme_file
        result["status"] = "success"
        
        logger.info(f"Web testing environment set up successfully in {output_dir}")
        logger.info(f"WebNN: {'Available' if result['webnn_available'] else 'Not Available'}")
        logger.info(f"WebGPU: {'Available' if result['webgpu_available'] else 'Not Available'}")
        
        return result
    except Exception as e:
        logger.error(f"Error setting up web testing environment: {e}")
        result["error"] = str(e)
        return result

def run_web_platform_benchmarks(models, platform, web_testing_dir=None, browser=None,
                               compute_shaders=False, parallel_loading=False, 
                               shader_precompile=False, small_models=True, 
                               db_path=None, output_dir="./benchmark_results"):
    """
    Run benchmarks for WebNN or WebGPU platforms.
    
    Args:
        models: List of models to benchmark
        platform: 'webnn' or 'webgpu'
        web_testing_dir: Directory with web testing environment
        browser: Preferred browser ('edge', 'chrome', 'firefox') or None for auto-select
        compute_shaders: Enable compute shader optimization (for audio models)
        parallel_loading: Enable parallel model loading (for multimodal models)
        shader_precompile: Enable shader precompilation (for faster startup)
        small_models: Use smaller model variants when available
        db_path: Path to benchmark database
        output_dir: Directory to save results
        
    Returns:
        bool: True if benchmarks completed successfully
    """
    logger.info(f"Running {platform} benchmarks for {len(models)} models")
    
    # First, check if run_web_platform_tests_with_db.py exists
    web_test_script = Path(__file__).parent / "run_web_platform_tests_with_db.py"
    if not web_test_script.exists():
        logger.error(f"Web platform test script not found: {web_test_script}")
        return False
    
    # Build command
    cmd = [
        sys.executable,
        str(web_test_script)
    ]
    
    # Add models
    if 'all' in models:
        cmd.append("--all-models")
    else:
        cmd.append("--models")
        cmd.extend(models)
    
    # Add platform
    if platform == "webnn":
        cmd.append("--run-webnn")
    elif platform == "webgpu":
        cmd.append("--run-webgpu")
    
    # Add optimization flags
    if compute_shaders:
        cmd.append("--compute-shaders")
    if parallel_loading:
        cmd.append("--parallel-loading")
    if shader_precompile:
        cmd.append("--shader-precompile")
    
    # Add other options
    if small_models:
        cmd.append("--small-models")
    
    # Add database path
    if db_path:
        cmd.extend(["--db-path", db_path])
    
    # Add results directory
    results_dir = os.path.join(output_dir, f"{platform}_results")
    cmd.extend(["--results-dir", results_dir])
    
    # Add headless mode
    cmd.append("--headless")
    
    # Set environment variables
    env = os.environ.copy()
    
    # Add web testing environment variables if provided
    if web_testing_dir:
        env["WEB_TESTING_DIR"] = web_testing_dir
        
        # Check for config file
        config_file = os.path.join(web_testing_dir, "web_testing_config.json")
        if os.path.exists(config_file):
            env["WEB_TESTING_CONFIG"] = config_file
            
            # Load the config
            with open(config_file, 'r') as f:
                try:
                    config = json.load(f)
                    
                    # Set platform availability
                    if platform == "webnn":
                        env["WEBNN_AVAILABLE"] = "1" if config.get("webnn_available", False) else "0"
                    elif platform == "webgpu":
                        env["WEBGPU_AVAILABLE"] = "1" if config.get("webgpu_available", False) else "0"
                    
                    # Set browser if not specified
                    if not browser and "browsers" in config:
                        # For WebNN prefer Edge, then Chrome
                        if platform == "webnn":
                            if "edge" in config["browsers"]:
                                browser = "edge"
                            elif "chrome" in config["browsers"]:
                                browser = "chrome"
                        # For WebGPU prefer Chrome, then Edge, then Firefox
                        elif platform == "webgpu":
                            if "chrome" in config["browsers"]:
                                browser = "chrome"
                            elif "edge" in config["browsers"]:
                                browser = "edge"
                            elif "firefox" in config["browsers"]:
                                browser = "firefox"
                except json.JSONDecodeError:
                    logger.warning(f"Failed to load web testing config from {config_file}")
    
    # Set platform-specific environment variables
    if platform == "webnn":
        env["WEBNN_ENABLED"] = "1"
    elif platform == "webgpu":
        env["WEBGPU_ENABLED"] = "1"
        
        # Set optimization environment variables
        if compute_shaders:
            env["WEBGPU_COMPUTE_SHADERS"] = "1"
        if parallel_loading:
            env["WEB_PARALLEL_LOADING"] = "1"
        if shader_precompile:
            env["WEBGPU_SHADER_PRECOMPILE"] = "1"
    
    # Set browser environment variable if specified
    if browser:
        env["PREFERRED_BROWSER"] = browser
        env["BROWSER"] = browser
    
    # Log the command
    logger.info(f"Running command: {' '.join(cmd)}")
    
    # Execute the command
    try:
        process = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        
        # Read and log output in real-time
        for line in process.stdout:
            print(line, end='')
            logger.info(line.strip())
        
        # Wait for process to complete
        return_code = process.wait()
        
        if return_code == 0:
            logger.info(f"{platform} benchmarks completed successfully")
            return True
        else:
            logger.error(f"{platform} benchmarks failed with return code {return_code}")
            return False
    except Exception as e:
        logger.error(f"Error running {platform} benchmarks: {e}")
        return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Run comprehensive benchmarks")
    
    # Core arguments
    parser.add_argument("--models", help="Comma-separated list of models to benchmark (default: bert,t5,vit,whisper)")
    parser.add_argument("--hardware", help="Comma-separated list of hardware platforms to benchmark (default: auto-detect)")
    parser.add_argument("--force-hardware", help="Comma-separated list of hardware platforms to force benchmarking on, even if not detected")
    parser.add_argument("--no-small-models", action="store_true", help="Use full-sized models instead of smaller variants")
    parser.add_argument("--batch-sizes", default="1,2,4,8,16", help="Comma-separated list of batch sizes to test")
    
    # Configuration options
    parser.add_argument("--db-path", help="Path to benchmark database (default: env var BENCHMARK_DB_PATH or ./benchmark_db.duckdb)")
    parser.add_argument("--output-dir", default="./benchmark_results", help="Directory to save results")
    parser.add_argument("--timeout", type=int, help="Timeout in seconds for each benchmark (default: 600)")
    parser.add_argument("--report-format", choices=["html", "markdown", "json"], default="html", help="Output format for the report")
    
    # Web platform testing arguments (April 2025)
    web_group = parser.add_argument_group("Web Platform Testing (April 2025)")
    web_group.add_argument("--setup-web-testing", action="store_true", help="Set up web testing environment for WebNN and WebGPU")
    web_group.add_argument("--browser", choices=["edge", "chrome", "firefox"], help="Preferred browser for web platform testing")
    web_group.add_argument("--web-testing-dir", default="./web_testing_env", help="Directory for web testing environment")
    web_group.add_argument("--web-compute-shaders", action="store_true", help="Enable compute shader optimization for audio models")
    web_group.add_argument("--web-parallel-loading", action="store_true", help="Enable parallel loading for multimodal models")
    web_group.add_argument("--web-shader-precompile", action="store_true", help="Enable shader precompilation for faster startup")
    web_group.add_argument("--web-all-optimizations", action="store_true", help="Enable all web platform optimizations")
    
    # Advanced options
    parser.add_argument("--skip-report", action="store_true", help="Skip generating the report after benchmarks complete")
    parser.add_argument("--skip-hardware-detection", action="store_true", help="Skip hardware detection and use specified hardware only")
    parser.add_argument("--list-available-hardware", action="store_true", help="List available hardware platforms and exit")
    parser.add_argument("--all-hardware", action="store_true", help="Run benchmarks on all supported hardware platforms (may use simulation)")
    
    args = parser.parse_args()
    
    # Setup web testing environment if requested
    if args.setup_web_testing:
        web_env = setup_web_testing_environment(
            browser=args.browser,
            output_dir=args.web_testing_dir,
            debug=True
        )
        
        if web_env["status"] == "success":
            print("\nWeb testing environment set up successfully:")
            print(f"Directory: {web_env['web_testing_dir']}")
            print(f"WebNN: {'Available' if web_env['webnn_available'] else 'Not Available'}")
            print(f"WebGPU: {'Available' if web_env['webgpu_available'] else 'Not Available'}")
            print(f"Browsers: {', '.join(web_env['browsers'].keys())}")
            print(f"\nSee README file for usage instructions: {web_env['readme_file']}")
            return 0
        else:
            print("\nFailed to set up web testing environment:")
            print(f"Error: {web_env.get('error', 'Unknown error')}")
            return 1
    
    # List available hardware if requested
    if args.list_available_hardware:
        available_hardware_dict = detect_available_hardware()
        print("\nAvailable Hardware Platforms:")
        for hw in ALL_HARDWARE:
            status = "✅ AVAILABLE" if available_hardware_dict.get(hw, False) else "❌ NOT AVAILABLE"
            print(f"  - {hw.upper()}: {status}")
        return 0
    
    # Process models argument
    if args.models:
        models = args.models.split(",")
    else:
        models = DEFAULT_MODELS
    
    # Enable all optimizations if requested
    if args.web_all_optimizations:
        args.web_compute_shaders = True
        args.web_parallel_loading = True
        args.web_shader_precompile = True
    
    # Process hardware argument
    if args.all_hardware:
        # Use all hardware platforms
        hardware = ALL_HARDWARE
        logger.info("Using all supported hardware platforms (may use simulation for unavailable hardware)")
    elif args.hardware:
        # Use specified hardware
        hardware = args.hardware.split(",")
    elif args.skip_hardware_detection:
        # Skip detection and use default hardware
        hardware = DEFAULT_HARDWARE
        logger.info("Skipping hardware detection and using default hardware")
    else:
        # Auto-detect hardware
        available_hardware_dict = detect_available_hardware()
        hardware = [hw for hw, available in available_hardware_dict.items() if available]
        logger.info(f"Auto-detected hardware: {', '.join(hardware)}")
    
    # Process force hardware argument
    force_hardware = args.force_hardware.split(",") if args.force_hardware else None
    
    # Process batch sizes
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    
    # Check if we're only running web platform benchmarks
    web_only = all(hw in ["webnn", "webgpu"] for hw in hardware)
    
    # If running web platform benchmarks, make sure optimizations are applied to appropriate models
    if "webgpu" in hardware:
        # If compute shaders are enabled, make sure we have audio models
        if args.web_compute_shaders:
            if not any(model in AUDIO_MODELS for model in models) and 'all' not in models:
                logger.warning("Compute shader optimization is enabled but no audio models specified")
                logger.info(f"Adding audio models: {', '.join(AUDIO_MODELS[:1])}")
                models.extend(AUDIO_MODELS[:1])  # Add at least whisper
        
        # If parallel loading is enabled, make sure we have multimodal models
        if args.web_parallel_loading:
            if not any(model in MULTIMODAL_MODELS for model in models) and 'all' not in models:
                logger.warning("Parallel loading optimization is enabled but no multimodal models specified")
                logger.info(f"Adding multimodal models: {', '.join(MULTIMODAL_MODELS[:1])}")
                models.extend(MULTIMODAL_MODELS[:1])  # Add at least clip
    
    logger.info(f"Running benchmarks for models: {', '.join(models)}")
    logger.info(f"Using hardware platforms: {', '.join(hardware)}")
    logger.info(f"Using batch sizes: {', '.join(map(str, batch_sizes))}")
    logger.info(f"Using {'small' if not args.no_small_models else 'full-sized'} models")
    
    # Check if we need to run special web platform benchmarks
    web_platforms = []
    if "webnn" in hardware:
        web_platforms.append("webnn")
    if "webgpu" in hardware:
        web_platforms.append("webgpu")
    
    # If only running web platforms, use the web platform benchmark runner
    if web_only and web_platforms:
        all_success = True
        for platform in web_platforms:
            platform_models = models.copy()
            
            # Filter models for platform-specific optimizations
            if platform == "webgpu" and args.web_compute_shaders and 'all' not in models:
                # For compute shaders, prioritize audio models
                platform_models = [m for m in models if m in AUDIO_MODELS] or platform_models
            if platform == "webgpu" and args.web_parallel_loading and 'all' not in models:
                # For parallel loading, prioritize multimodal models
                platform_models = [m for m in models if m in MULTIMODAL_MODELS] or platform_models
            
            success = run_web_platform_benchmarks(
                models=platform_models,
                platform=platform,
                web_testing_dir=args.web_testing_dir,
                browser=args.browser,
                compute_shaders=args.web_compute_shaders,
                parallel_loading=args.web_parallel_loading,
                shader_precompile=args.web_shader_precompile,
                small_models=not args.no_small_models,
                db_path=args.db_path,
                output_dir=args.output_dir
            )
            
            all_success = all_success and success
        
        if all_success:
            print("\nWeb platform benchmarks completed successfully!")
        else:
            print("\nSome web platform benchmarks failed. Check logs for details.")
        
        return 0 if all_success else 1
    
    # Otherwise, run regular benchmarks
    success = run_benchmarks(
        models=models,
        hardware=hardware,
        batch_sizes=batch_sizes,
        small_models=not args.no_small_models,
        db_path=args.db_path,
        output_dir=args.output_dir,
        timeout=args.timeout,
        report_format=args.report_format,
        force_hardware=force_hardware
    )
    
    if success:
        print("\nBenchmarks completed successfully!")
    else:
        print("\nBenchmarks failed. Check logs for details.")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())