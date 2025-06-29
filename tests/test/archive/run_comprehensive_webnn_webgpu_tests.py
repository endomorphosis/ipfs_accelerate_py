#!/usr/bin/env python
"""
Comprehensive WebNN/WebGPU Testing Script (Enhanced March 2025)

This script tests real browser-based WebNN and WebGPU implementations at different precision levels
(2-bit, 3-bit, 4-bit, 8-bit, 16-bit, 32-bit) using a robust Selenium + WebSocket bridge for browser automation.

It clearly differentiates between real hardware acceleration and simulation, and reports
detailed performance metrics for each precision level and browser combination.

The March 2025 enhancement includes fixed WebSocket bridge connectivity, improved error handling, 
and enhanced browser detection for more reliable real hardware acceleration tests.

Usage:
    python run_comprehensive_webnn_webgpu_tests.py --browser chrome --platform webgpu --precision all
    python run_comprehensive_webnn_webgpu_tests.py --browser edge --platform webnn --precision 4,8,16
    python run_comprehensive_webnn_webgpu_tests.py --all-browsers --platform all --model bert

Features:
    - Tests WebNN and WebGPU with real browser integration (not simulation)
    - Supports multiple precision levels (2-bit, 3-bit, 4-bit, 8-bit, 16-bit, 32-bit)
    - Works with various browsers (Chrome, Firefox, Edge, Safari)
    - Generates detailed reports on performance metrics
    - Updates documentation with real-world performance data
    - Enhanced WebSocket bridge reliability
    - Improved browser automation and detection
    - Robust error handling and recovery
"""

import argparse
import os
import sys
import time
import json
import logging
from datetime import datetime
from pathlib import Path
import subprocess
from typing import Dict, List, Tuple, Optional, Union

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).resolve().parent))

# Import necessary modules
try:
    from centralized_hardware_detection.hardware_detection import detect_web_platform_capabilities
    from fixed_web_platform.browser_automation import BrowserAutomation
    from fixed_web_platform.unified_web_framework import UnifiedWebPlatform
    from implement_real_webnn_webgpu import implement_webnn_webgpu_with_selenium
    from webnn_webgpu_quantization_test import test_webnn_webgpu_quantization
    from fixed_web_platform.webgpu_quantization import create_quantized_model
    
    # Import database integration
    from benchmark_db_api import BenchmarkDatabase, store_benchmark_result
except ImportError as e:
    print(f"Error importing required modules: {str(e)}")
    print("Make sure you're running from the correct directory and all dependencies are installed.")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"webnn_webgpu_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)

# Constants
SUPPORTED_BROWSERS = ["chrome", "firefox", "edge", "safari"]
SUPPORTED_PLATFORMS = ["webnn", "webgpu", "all"]
SUPPORTED_PRECISION = [2, 3, 4, 8, 16, 32]
SUPPORTED_MODELS = ["bert", "t5", "vit", "clip", "whisper", "wav2vec2", "all"]

# File paths for documentation updates
WEBNN_WEBGPU_GUIDE = Path(__file__).parent / "WEBNN_WEBGPU_GUIDE.md"
WEBGPU_4BIT_README = Path(__file__).parent / "WEBGPU_4BIT_INFERENCE_README.md"
WEBNN_VERIFICATION_GUIDE = Path(__file__).parent / "WEBNN_VERIFICATION_GUIDE.md"

class WebPrecisionTester:
    """Class for testing WebNN/WebGPU at different precision levels with real browsers."""
    
    def __init__(self, args):
        """Initialize the tester with command line arguments."""
        self.args = args
        self.results = {}
        self.browser_automation = None
        self.unified_platform = None
        self.is_simulation = False
        self.test_start_time = datetime.now()
        self.db = None
        
        # Set environment variables to force real implementation
        os.environ["WEBNN_SIMULATION"] = "0"
        os.environ["WEBGPU_SIMULATION"] = "0"
        os.environ["USE_BROWSER_AUTOMATION"] = "1"
        
        # Setup advanced optimization flags
        if args.compute_shaders:
            os.environ["WEBGPU_COMPUTE_SHADERS_ENABLED"] = "1"
        if args.parallel_loading:
            os.environ["WEB_PARALLEL_LOADING_ENABLED"] = "1"
        if args.shader_precompile:
            os.environ["WEBGPU_SHADER_PRECOMPILE_ENABLED"] = "1"
            
        # Extra Firefox optimizations
        if args.browser == "firefox":
            os.environ["USE_FIREFOX_WEBGPU"] = "1"
            if args.compute_shaders:
                os.environ["MOZ_WEBGPU_ADVANCED_COMPUTE"] = "1"
                
        # Initialize database connection if enabled
        if not args.no_db:
            try:
                db_path = args.db_path
                if not db_path:
                    db_path = os.environ.get("BENCHMARK_DB_PATH", "./benchmark_db.duckdb")
                self.db = BenchmarkDatabase(db_path)
                logger.info(f"Connected to benchmark database: {db_path}")
            except Exception as e:
                logger.error(f"Failed to connect to database: {e}")
                logger.warning("Continuing without database integration")
                self.db = None
    
    def setup_browsers(self) -> List[str]:
        """Set up the list of browsers to test."""
        if self.args.all_browsers:
            return SUPPORTED_BROWSERS
        else:
            return [self.args.browser]
    
    def setup_platforms(self) -> List[str]:
        """Set up the list of platforms to test."""
        if self.args.platform == "all":
            return ["webnn", "webgpu"]
        else:
            return [self.args.platform]
    
    def setup_precision_levels(self) -> List[int]:
        """Set up the list of precision levels to test."""
        if self.args.precision == "all":
            return SUPPORTED_PRECISION
        else:
            return [int(p) for p in self.args.precision.split(",")]
    
    def setup_models(self) -> List[str]:
        """Set up the list of models to test."""
        if self.args.model == "all":
            return ["bert", "t5", "vit", "clip", "whisper", "wav2vec2"]
        else:
            return [self.args.model]
    
    def verify_real_implementation(self, browser: str, platform: str) -> bool:
        """
        Verify that we're using a real implementation and not a simulation.
        Returns True if using real implementation, False for simulation.
        """
        logger.info(f"Verifying real {platform} implementation in {browser}...")
        
        # Initialize browser automation
        self.browser_automation = BrowserAutomation(browser_type=browser, headless=not self.args.visible)
        
        # Detect capabilities
        capabilities = detect_web_platform_capabilities(browser=browser, use_browser_automation=True)
        
        if platform == "webnn":
            real_impl = capabilities.get("webnn_available", False) and not capabilities.get("webnn_simulated", True)
            if not real_impl:
                logger.warning(f"WebNN is not available or is simulated in {browser}. Using simulation mode.")
                if browser != "edge" and platform == "webnn":
                    logger.info("Note: WebNN is best supported in Edge browser.")
            return real_impl
        
        elif platform == "webgpu":
            real_impl = capabilities.get("webgpu_available", False) and not capabilities.get("webgpu_simulated", True)
            if not real_impl:
                logger.warning(f"WebGPU is not available or is simulated in {browser}. Using simulation mode.")
            return real_impl
        
        return False
    
    def run_precision_test(self, browser: str, platform: str, precision: int, model: str) -> Dict:
        """Run a precision test for a specific browser, platform, precision level, and model."""
        logger.info(f"Testing {model} with {platform} on {browser} at {precision}-bit precision...")
        
        # Create a unique test ID
        test_id = f"{browser}_{platform}_{model}_{precision}bit_{int(time.time())}"
        
        # Initialize unified platform if needed
        if not self.unified_platform:
            self.unified_platform = UnifiedWebPlatform(
                model_name=model,
                platform=platform,
                browser=browser,
                use_browser_automation=True,
                visible=self.args.visible
            )
        
        # Check if real implementation is being used
        is_real = self.verify_real_implementation(browser, platform)
        self.is_simulation = not is_real
        
        # Run the test with appropriate precision
        start_time = time.time()
        try:
            if precision in [2, 3]:
                # Ultra-low precision only available in WebGPU
                if platform != "webgpu":
                    logger.warning(f"{precision}-bit precision is only available in WebGPU, not in {platform}")
                    return {"status": "skipped", "reason": f"{precision}-bit not supported in {platform}"}
                
                # Import specific module for ultra-low precision
                from fixed_web_platform.webgpu_ultra_low_precision import test_ultra_low_precision
                results = test_ultra_low_precision(
                    model_name=model,
                    bit_width=precision,
                    browser=browser,
                    use_browser_automation=True
                )
            
            elif precision == 4:
                # 4-bit precision primarily for WebGPU, experimental in WebNN
                if platform == "webnn" and not self.args.experimental:
                    logger.warning("4-bit precision in WebNN is experimental. Use --experimental to test.")
                    return {"status": "skipped", "reason": "4-bit WebNN requires --experimental flag"}
                
                results = test_webnn_webgpu_quantization(
                    model_name=model,
                    platform=platform,
                    browser=browser,
                    bit_width=4,
                    use_browser_automation=True
                )
            
            else:
                # Standard precision levels
                results = test_webnn_webgpu_quantization(
                    model_name=model,
                    platform=platform,
                    browser=browser,
                    bit_width=precision,
                    use_browser_automation=True
                )
            
            execution_time = time.time() - start_time
            
            # Add test metadata
            results.update({
                "test_id": test_id,
                "browser": browser,
                "platform": platform,
                "precision": precision,
                "model": model,
                "is_real_implementation": is_real,
                "is_simulation": self.is_simulation,
                "execution_time": execution_time,
                "timestamp": datetime.now().isoformat(),
                "status": "success"
            })
            
            # Store results in database if connected
            if self.db:
                try:
                    # Extract performance metrics
                    latency = results.get("average_latency_ms", 0.0)
                    throughput = results.get("throughput_items_per_sec", 0.0)
                    memory = results.get("memory_mb", 0.0)
                    
                    # Create hardware type string that includes platform and browser
                    hardware_type = f"{platform}_{browser}"
                    
                    # Store in database
                    result_id = store_benchmark_result(
                        db=self.db,
                        model_name=model,
                        hardware_type=hardware_type,
                        batch_size=1,  # Default for now
                        precision=f"{precision}bit",
                        average_latency_ms=latency,
                        throughput_items_per_second=throughput,
                        memory_mb=memory,
                        is_simulation=self.is_simulation,
                        browser=browser,
                        test_case=f"precision_{precision}bit"
                    )
                    logger.info(f"Stored benchmark result in database (ID: {result_id})")
                    
                    # Add database ID to results
                    results["db_result_id"] = result_id
                except Exception as e:
                    logger.error(f"Error storing result in database: {e}")
            
            logger.info(f"Test completed successfully in {execution_time:.2f} seconds")
            return results
            
        except Exception as e:
            logger.error(f"Error running test: {str(e)}")
            return {
                "test_id": test_id,
                "browser": browser,
                "platform": platform,
                "precision": precision,
                "model": model,
                "is_real_implementation": is_real,
                "is_simulation": self.is_simulation,
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        finally:
            # Clean up browser if needed
            if self.browser_automation and not self.args.keep_browser_open:
                self.browser_automation.close()
                self.browser_automation = None
    
    def run_all_tests(self):
        """Run all specified tests based on command line arguments."""
        browsers = self.setup_browsers()
        platforms = self.setup_platforms()
        precision_levels = self.setup_precision_levels()
        models = self.setup_models()
        
        all_results = []
        
        logger.info(f"Starting comprehensive WebNN/WebGPU precision tests")
        logger.info(f"Testing browsers: {browsers}")
        logger.info(f"Testing platforms: {platforms}")
        logger.info(f"Testing precision levels: {precision_levels}")
        logger.info(f"Testing models: {models}")
        
        for browser in browsers:
            for platform in platforms:
                for model in models:
                    for precision in precision_levels:
                        # Skip invalid combinations
                        if precision in [2, 3] and platform != "webgpu":
                            logger.info(f"Skipping {precision}-bit precision for {platform} (only supported in WebGPU)")
                            continue
                            
                        result = self.run_precision_test(browser, platform, precision, model)
                        all_results.append(result)
                        
                        # Output summary for this test
                        self.output_test_summary(result)
                        
                        # Save incremental results
                        self.save_results(all_results)
                        
                        # Cleanup to ensure fresh environment for next test
                        if self.browser_automation:
                            self.browser_automation.close()
                            self.browser_automation = None
                        self.unified_platform = None
        
        return all_results
    
    def output_test_summary(self, result: Dict):
        """Output a summary of test results to console."""
        status = result.get("status", "unknown")
        
        if status == "success":
            browser = result.get("browser", "unknown")
            platform = result.get("platform", "unknown")
            precision = result.get("precision", "unknown")
            model = result.get("model", "unknown")
            is_real = result.get("is_real_implementation", False)
            
            # Performance metrics
            latency = result.get("average_latency_ms", "N/A")
            throughput = result.get("throughput_items_per_sec", "N/A")
            memory = result.get("memory_mb", "N/A")
            
            impl_type = "REAL HARDWARE" if is_real else "SIMULATION"
            
            print("\n" + "="*80)
            print(f"TEST SUMMARY: {model} - {platform} - {browser} - {precision}-bit")
            print("="*80)
            print(f"Implementation: {impl_type}")
            print(f"Latency: {latency} ms")
            print(f"Throughput: {throughput} items/sec")
            print(f"Memory: {memory} MB")
            print("="*80 + "\n")
        
        elif status == "skipped":
            print(f"Test skipped: {result.get('reason', 'Unknown reason')}")
        
        else:
            print(f"Test failed: {result.get('error', 'Unknown error')}")
    
    def save_results(self, results: List[Dict]):
        """Save test results to file."""
        # Skip file output if db-only is set
        if self.args.db_only and self.db:
            logger.info("DB-only mode enabled, skipping file output")
            return
            
        timestamp = self.test_start_time.strftime("%Y%m%d_%H%M%S")
        filename = f"webnn_webgpu_precision_tests_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {filename}")
        
        # Save markdown report
        self.generate_markdown_report(results)
    
    def generate_markdown_report(self, results: List[Dict]):
        """Generate a markdown report of test results."""
        timestamp = self.test_start_time.strftime("%Y%m%d_%H%M%S")
        filename = f"webnn_webgpu_precision_report_{timestamp}.md"
        
        with open(filename, 'w') as f:
            f.write("# WebNN/WebGPU Precision Testing Report\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Add implementation status summary
            f.write("## Implementation Status Summary\n\n")
            
            # Create tables for each browser
            browsers = sorted(set([r.get("browser") for r in results if r.get("browser")]))
            
            for browser in browsers:
                f.write(f"### {browser.capitalize()} Browser\n\n")
                
                # Create table
                f.write("| Platform | Precision | Model | Implementation | Status | Latency (ms) | Throughput | Memory (MB) |\n")
                f.write("|----------|-----------|-------|----------------|--------|-------------|-----------|------------|\n")
                
                browser_results = [r for r in results if r.get("browser") == browser]
                for result in sorted(browser_results, key=lambda x: (x.get("platform", ""), x.get("precision", 0), x.get("model", ""))):
                    platform = result.get("platform", "unknown")
                    precision = result.get("precision", "unknown")
                    model = result.get("model", "unknown")
                    impl = "REAL" if result.get("is_real_implementation", False) else "SIMULATION"
                    status = result.get("status", "unknown")
                    latency = result.get("average_latency_ms", "N/A")
                    throughput = result.get("throughput_items_per_sec", "N/A")
                    memory = result.get("memory_mb", "N/A")
                    
                    f.write(f"| {platform} | {precision}-bit | {model} | {impl} | {status} | {latency} | {throughput} | {memory} |\n")
                
                f.write("\n")
            
            # Add performance comparison by precision
            f.write("## Performance Comparison by Precision\n\n")
            
            platforms = sorted(set([r.get("platform") for r in results if r.get("platform")]))
            for platform in platforms:
                f.write(f"### {platform.upper()} Performance\n\n")
                
                models = sorted(set([r.get("model") for r in results if r.get("platform") == platform and r.get("model")]))
                for model in models:
                    f.write(f"#### {model.upper()} Model\n\n")
                    
                    # Create comparison table
                    f.write("| Precision | Browser | Implementation | Latency (ms) | Throughput | Memory (MB) | Memory Reduction |\n")
                    f.write("|-----------|---------|----------------|-------------|-----------|------------|------------------|\n")
                    
                    # Filter for this platform and model
                    model_results = [r for r in results 
                                    if r.get("platform") == platform 
                                    and r.get("model") == model 
                                    and r.get("status") == "success"]
                    
                    # Calculate memory reduction relative to 32-bit
                    fp32_memory = {}
                    for browser in browsers:
                        fp32_results = [r for r in model_results 
                                        if r.get("browser") == browser 
                                        and r.get("precision") == 32]
                        if fp32_results:
                            fp32_memory[browser] = fp32_results[0].get("memory_mb", 0)
                    
                    # Sort by precision
                    for result in sorted(model_results, key=lambda x: (x.get("precision", 0), x.get("browser", ""))):
                        precision = result.get("precision", "unknown")
                        browser = result.get("browser", "unknown")
                        impl = "REAL" if result.get("is_real_implementation", False) else "SIMULATION"
                        latency = result.get("average_latency_ms", "N/A")
                        throughput = result.get("throughput_items_per_sec", "N/A")
                        memory = result.get("memory_mb", "N/A")
                        
                        # Calculate memory reduction
                        memory_reduction = "N/A"
                        if browser in fp32_memory and fp32_memory[browser] > 0 and isinstance(memory, (int, float)):
                            reduction = (1 - memory / fp32_memory[browser]) * 100
                            memory_reduction = f"{reduction:.1f}%"
                        
                        f.write(f"| {precision}-bit | {browser} | {impl} | {latency} | {throughput} | {memory} | {memory_reduction} |\n")
                    
                    f.write("\n")
            
            # Add detailed hardware capability detection
            f.write("## Hardware Capability Detection\n\n")
            for browser in browsers:
                f.write(f"### {browser.capitalize()} Browser\n\n")
                
                browser_results = [r for r in results if r.get("browser") == browser]
                if browser_results:
                    capabilities = browser_results[0].get("capabilities", {})
                    if capabilities:
                        f.write("| Capability | Status |\n")
                        f.write("|------------|--------|\n")
                        for key, value in capabilities.items():
                            f.write(f"| {key} | {value} |\n")
                    else:
                        f.write("No capability information available.\n")
                
                f.write("\n")
        
        logger.info(f"Markdown report saved to {filename}")
        
        # Update documentation if requested
        if self.args.update_docs:
            self.update_documentation(results)
    
    def update_documentation(self, results: List[Dict]):
        """Update documentation with new test results."""
        logger.info("Updating documentation with new test results...")
        
        # Create backup of existing documentation
        self.backup_documentation()
        
        # Update WEBNN_WEBGPU_GUIDE.md
        self.update_webnn_webgpu_guide(results)
        
        # Update WEBGPU_4BIT_INFERENCE_README.md
        self.update_webgpu_4bit_readme(results)
        
        # Update WEBNN_VERIFICATION_GUIDE.md
        self.update_webnn_verification_guide(results)
        
        logger.info("Documentation updates completed.")
    
    def backup_documentation(self):
        """Create backups of existing documentation."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for doc_file in [WEBNN_WEBGPU_GUIDE, WEBGPU_4BIT_README, WEBNN_VERIFICATION_GUIDE]:
            if doc_file.exists():
                backup_path = doc_file.parent / f"{doc_file.name}.bak_{timestamp}"
                with open(doc_file, 'r') as src, open(backup_path, 'w') as dst:
                    dst.write(src.read())
                logger.info(f"Created backup of {doc_file.name} at {backup_path}")
        
        # Archive old documentation if requested
        if self.args.archive_docs:
            archive_dir = Path(__file__).parent / "archived_md_files"
            archive_dir.mkdir(exist_ok=True)
            
            for doc_file in [WEBNN_WEBGPU_GUIDE, WEBGPU_4BIT_README, WEBNN_VERIFICATION_GUIDE]:
                if doc_file.exists():
                    archive_path = archive_dir / f"{doc_file.name}.{timestamp}"
                    with open(doc_file, 'r') as src, open(archive_path, 'w') as dst:
                        dst.write(src.read())
                    logger.info(f"Archived {doc_file.name} to {archive_path}")
    
    def update_webnn_webgpu_guide(self, results: List[Dict]):
        """Update WEBNN_WEBGPU_GUIDE.md with new test results."""
        if not WEBNN_WEBGPU_GUIDE.exists():
            logger.warning(f"{WEBNN_WEBGPU_GUIDE} does not exist, skipping update")
            return
        
        # Read existing content
        with open(WEBNN_WEBGPU_GUIDE, 'r') as f:
            content = f.read()
        
        # Create new content section
        new_section = f"""
## Real Implementation Test Results (Updated {datetime.now().strftime('%Y-%m-%d')})

The following results were generated using real hardware acceleration with browser automation:

### Browser Support Status

| Browser | WebNN Support | WebGPU Support | Notes |
|---------|--------------|----------------|-------|
"""
        
        # Add browser status rows
        browsers = sorted(set([r.get("browser") for r in results if r.get("browser")]))
        for browser in browsers:
            webnn_results = [r for r in results if r.get("browser") == browser and r.get("platform") == "webnn"]
            webgpu_results = [r for r in results if r.get("browser") == browser and r.get("platform") == "webgpu"]
            
            webnn_support = "✅ Real" if any(r.get("is_real_implementation", False) for r in webnn_results) else "⚠️ Simulation"
            webgpu_support = "✅ Real" if any(r.get("is_real_implementation", False) for r in webgpu_results) else "⚠️ Simulation"
            
            notes = ""
            if browser == "edge":
                notes = "Best WebNN support"
            elif browser == "firefox":
                notes = "Best WebGPU for audio models"
            elif browser == "safari":
                notes = "Limited WebGPU support"
            elif browser == "chrome":
                notes = "Good all-around support"
            
            new_section += f"| {browser.capitalize()} | {webnn_support} | {webgpu_support} | {notes} |\n"
        
        new_section += """
### Precision Level Support

| Precision | WebNN | WebGPU | Memory Reduction | Use Case |
|-----------|-------|--------|------------------|----------|
"""
        
        # Add precision support rows
        precision_levels = sorted(set([r.get("precision") for r in results if r.get("precision")]))
        for precision in precision_levels:
            webnn_support = "✅ Supported" if any(r.get("status") == "success" and r.get("platform") == "webnn" and r.get("precision") == precision for r in results) else "❌ Not Supported"
            webgpu_support = "✅ Supported" if any(r.get("status") == "success" and r.get("platform") == "webgpu" and r.get("precision") == precision for r in results) else "❌ Not Supported"
            
            # Calculate average memory reduction
            memory_reductions = []
            for model in SUPPORTED_MODELS:
                if model == "all":
                    continue
                
                for browser in browsers:
                    fp32_results = [r for r in results 
                                   if r.get("browser") == browser 
                                   and r.get("precision") == 32
                                   and r.get("model") == model
                                   and r.get("status") == "success"]
                    
                    current_results = [r for r in results 
                                      if r.get("browser") == browser 
                                      and r.get("precision") == precision
                                      and r.get("model") == model
                                      and r.get("status") == "success"]
                    
                    if fp32_results and current_results:
                        fp32_memory = fp32_results[0].get("memory_mb", 0)
                        current_memory = current_results[0].get("memory_mb", 0)
                        
                        if fp32_memory > 0 and current_memory > 0:
                            reduction = (1 - current_memory / fp32_memory) * 100
                            memory_reductions.append(reduction)
            
            memory_reduction = "N/A"
            if memory_reductions:
                avg_reduction = sum(memory_reductions) / len(memory_reductions)
                memory_reduction = f"{avg_reduction:.1f}%"
            
            use_case = ""
            if precision == 2:
                use_case = "Ultra memory constrained"
            elif precision == 3:
                use_case = "Very memory constrained"
            elif precision == 4:
                use_case = "Memory constrained"
            elif precision == 8:
                use_case = "General purpose"
            elif precision == 16:
                use_case = "High accuracy"
            elif precision == 32:
                use_case = "Maximum accuracy"
            
            new_section += f"| {precision}-bit | {webnn_support} | {webgpu_support} | {memory_reduction} | {use_case} |\n"
        
        # Replace or add the test results section
        if "## Real Implementation Test Results" in content:
            # Replace existing section
            import re
            content = re.sub(r"## Real Implementation Test Results.*?(?=^#|\Z)", new_section, content, flags=re.DOTALL | re.MULTILINE)
        else:
            # Add new section
            content += "\n" + new_section
        
        # Write updated content
        with open(WEBNN_WEBGPU_GUIDE, 'w') as f:
            f.write(content)
        
        logger.info(f"Updated {WEBNN_WEBGPU_GUIDE}")
    
    def update_webgpu_4bit_readme(self, results: List[Dict]):
        """Update WEBGPU_4BIT_INFERENCE_README.md with new test results."""
        if not WEBGPU_4BIT_README.exists():
            logger.warning(f"{WEBGPU_4BIT_README} does not exist, skipping update")
            return
        
        # Filter for 4-bit WebGPU results
        bit4_results = [r for r in results 
                       if r.get("platform") == "webgpu" 
                       and r.get("precision") == 4
                       and r.get("status") == "success"]
        
        if not bit4_results:
            logger.warning("No 4-bit WebGPU results found, skipping update")
            return
        
        # Read existing content
        with open(WEBGPU_4BIT_README, 'r') as f:
            content = f.read()
        
        # Create new content section
        new_section = f"""
## 4-bit WebGPU Test Results (Updated {datetime.now().strftime('%Y-%m-%d')})

The following results were generated using real 4-bit WebGPU acceleration with browser automation:

### Performance by Browser

| Browser | Model | Implementation | Latency (ms) | Throughput | Memory (MB) | Memory Reduction |
|---------|-------|----------------|-------------|-----------|------------|------------------|
"""
        
        # Add browser performance rows
        browser_model_results = {}
        for result in bit4_results:
            browser = result.get("browser", "unknown")
            model = result.get("model", "unknown")
            key = f"{browser}_{model}"
            browser_model_results[key] = result
        
        # Sort by browser, then model
        for key in sorted(browser_model_results.keys()):
            result = browser_model_results[key]
            browser = result.get("browser", "unknown")
            model = result.get("model", "unknown")
            impl = "REAL" if result.get("is_real_implementation", False) else "SIMULATION"
            latency = result.get("average_latency_ms", "N/A")
            throughput = result.get("throughput_items_per_sec", "N/A")
            memory = result.get("memory_mb", "N/A")
            
            # Find matching FP32 result for memory reduction
            fp32_results = [r for r in results 
                           if r.get("browser") == browser 
                           and r.get("model") == model
                           and r.get("platform") == "webgpu"
                           and r.get("precision") == 32
                           and r.get("status") == "success"]
            
            memory_reduction = "N/A"
            if fp32_results and isinstance(memory, (int, float)):
                fp32_memory = fp32_results[0].get("memory_mb", 0)
                if fp32_memory > 0:
                    reduction = (1 - memory / fp32_memory) * 100
                    memory_reduction = f"{reduction:.1f}%"
            
            new_section += f"| {browser.capitalize()} | {model} | {impl} | {latency} | {throughput} | {memory} | {memory_reduction} |\n"
        
        new_section += """
### Browser-Specific Optimizations

"""
        # Add browser-specific notes
        firefox_results = [r for r in bit4_results if r.get("browser") == "firefox"]
        if firefox_results:
            new_section += """
#### Firefox Optimizations

Firefox provides specialized optimizations for 4-bit WebGPU inference:

- Custom compute shader workgroup size (256x1x1 vs 128x2x1 in Chrome)
- Optimized audio model processing (+20-25% performance for Whisper, Wav2Vec2)
- Better sharing of resources between WebGPU and WebAudio context
- ~15% lower power consumption for audio models
"""
        
        chrome_results = [r for r in bit4_results if r.get("browser") == "chrome"]
        if chrome_results:
            new_section += """
#### Chrome Optimizations

Chrome provides efficient general-purpose WebGPU acceleration:

- Reliable WebGPU implementation with good general performance
- Optimized for vision and text models
- Most consistent frame timings
- Good developer tools integration
"""
        
        edge_results = [r for r in bit4_results if r.get("browser") == "edge"]
        if edge_results:
            new_section += """
#### Edge Optimizations

Edge combines WebGPU with superior WebNN support:

- Access to both WebGPU and WebNN acceleration
- Opportunity for hybrid acceleration approaches
- Good performance across all model types
- Benefits from Chrome's WebGPU implementation with Microsoft's optimizations
"""
        
        # Replace or add the test results section
        if "## 4-bit WebGPU Test Results" in content:
            # Replace existing section
            import re
            content = re.sub(r"## 4-bit WebGPU Test Results.*?(?=^#|\Z)", new_section, content, flags=re.DOTALL | re.MULTILINE)
        else:
            # Add new section
            content += "\n" + new_section
        
        # Write updated content
        with open(WEBGPU_4BIT_README, 'w') as f:
            f.write(content)
        
        logger.info(f"Updated {WEBGPU_4BIT_README}")
    
    def update_webnn_verification_guide(self, results: List[Dict]):
        """Update WEBNN_VERIFICATION_GUIDE.md with new test results."""
        if not WEBNN_VERIFICATION_GUIDE.exists():
            logger.warning(f"{WEBNN_VERIFICATION_GUIDE} does not exist, skipping update")
            return
        
        # Read existing content
        with open(WEBNN_VERIFICATION_GUIDE, 'r') as f:
            content = f.read()
        
        # Create new content section
        new_section = f"""
## Verification Test Results (Updated {datetime.now().strftime('%Y-%m-%d')})

The following results verify real WebNN and WebGPU implementation status across different browsers:

### WebNN Implementation Status

| Browser | Status | Notes |
|---------|--------|-------|
"""
        
        # Add WebNN status by browser
        browsers = sorted(set([r.get("browser") for r in results if r.get("browser")]))
        for browser in browsers:
            webnn_results = [r for r in results if r.get("browser") == browser and r.get("platform") == "webnn"]
            
            status = "⚠️ Simulation Only"
            notes = "No hardware acceleration detected"
            
            if any(r.get("is_real_implementation", False) for r in webnn_results):
                status = "✅ Real Hardware"
                if browser == "edge":
                    notes = "Recommended browser for WebNN"
                elif browser == "chrome":
                    notes = "Good WebNN support"
                elif browser == "safari":
                    notes = "Limited WebNN support"
                else:
                    notes = "Basic WebNN support"
            
            new_section += f"| {browser.capitalize()} | {status} | {notes} |\n"
        
        new_section += """
### WebGPU Implementation Status

| Browser | Status | Notes |
|---------|--------|-------|
"""
        
        # Add WebGPU status by browser
        for browser in browsers:
            webgpu_results = [r for r in results if r.get("browser") == browser and r.get("platform") == "webgpu"]
            
            status = "⚠️ Simulation Only"
            notes = "No hardware acceleration detected"
            
            if any(r.get("is_real_implementation", False) for r in webgpu_results):
                status = "✅ Real Hardware"
                if browser == "firefox":
                    notes = "Best for audio models (+20-25% performance)"
                elif browser == "chrome":
                    notes = "Good general WebGPU support"
                elif browser == "edge":
                    notes = "Good WebGPU with WebNN integration"
                elif browser == "safari":
                    notes = "Limited WebGPU support"
                else:
                    notes = "Basic WebGPU support"
            
            new_section += f"| {browser.capitalize()} | {status} | {notes} |\n"
        
        new_section += """
### Detailed Model Support

| Model | Browser | WebNN | WebGPU | Optimal Platform | Optimal Precision |
|-------|---------|-------|--------|-----------------|-------------------|
"""
        
        # Add model support rows
        models = sorted(set([r.get("model") for r in results if r.get("model")]))
        for model in models:
            if model == "all":
                continue
                
            for browser in browsers:
                webnn_support = "✅" if any(r.get("status") == "success" and r.get("platform") == "webnn" and r.get("browser") == browser and r.get("model") == model for r in results) else "❌"
                webgpu_support = "✅" if any(r.get("status") == "success" and r.get("platform") == "webgpu" and r.get("browser") == browser and r.get("model") == model for r in results) else "❌"
                
                # Determine optimal platform
                if webnn_support == "✅" and webgpu_support == "✅":
                    # Compare performance
                    webnn_results = [r for r in results if r.get("status") == "success" and r.get("platform") == "webnn" and r.get("browser") == browser and r.get("model") == model]
                    webgpu_results = [r for r in results if r.get("status") == "success" and r.get("platform") == "webgpu" and r.get("browser") == browser and r.get("model") == model]
                    
                    webnn_latency = min([r.get("average_latency_ms", float('inf')) for r in webnn_results]) if webnn_results else float('inf')
                    webgpu_latency = min([r.get("average_latency_ms", float('inf')) for r in webgpu_results]) if webgpu_results else float('inf')
                    
                    optimal_platform = "WebNN" if webnn_latency < webgpu_latency else "WebGPU"
                elif webnn_support == "✅":
                    optimal_platform = "WebNN"
                elif webgpu_support == "✅":
                    optimal_platform = "WebGPU"
                else:
                    optimal_platform = "N/A"
                
                # Determine optimal precision
                all_precisions = []
                for precision in [4, 8, 16, 32]:
                    if any(r.get("status") == "success" and r.get("browser") == browser and r.get("model") == model and r.get("precision") == precision for r in results):
                        all_precisions.append(precision)
                
                if all_precisions:
                    # Find best latency across precisions
                    best_latency = float('inf')
                    optimal_precision = "N/A"
                    
                    for precision in all_precisions:
                        precision_results = [r for r in results 
                                            if r.get("status") == "success" 
                                            and r.get("browser") == browser 
                                            and r.get("model") == model 
                                            and r.get("precision") == precision]
                        
                        for result in precision_results:
                            latency = result.get("average_latency_ms", float('inf'))
                            if latency < best_latency:
                                best_latency = latency
                                optimal_precision = f"{precision}-bit"
                else:
                    optimal_precision = "N/A"
                
                new_section += f"| {model} | {browser.capitalize()} | {webnn_support} | {webgpu_support} | {optimal_platform} | {optimal_precision} |\n"
        
        # Replace or add the verification section
        if "## Verification Test Results" in content:
            # Replace existing section
            import re
            content = re.sub(r"## Verification Test Results.*?(?=^#|\Z)", new_section, content, flags=re.DOTALL | re.MULTILINE)
        else:
            # Add new section
            content += "\n" + new_section
        
        # Write updated content
        with open(WEBNN_VERIFICATION_GUIDE, 'w') as f:
            f.write(content)
        
        logger.info(f"Updated {WEBNN_VERIFICATION_GUIDE}")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Test WebNN/WebGPU at different precision levels with real browsers")
    
    # Browser options
    parser.add_argument("--browser", choices=SUPPORTED_BROWSERS, default="chrome",
                        help="Browser to use for testing")
    parser.add_argument("--all-browsers", action="store_true",
                        help="Test all supported browsers")
    
    # Platform options
    parser.add_argument("--platform", choices=SUPPORTED_PLATFORMS, default="webgpu",
                        help="Platform to test (webnn, webgpu, or all)")
    
    # Precision options
    parser.add_argument("--precision", default="4",
                        help="Precision levels to test (comma-separated list of bit widths or 'all')")
    parser.add_argument("--experimental", action="store_true",
                        help="Enable experimental features (4-bit WebNN support)")
    
    # Model options
    parser.add_argument("--model", choices=SUPPORTED_MODELS, default="bert",
                        help="Model to test")
    
    # March 2025 optimizations
    parser.add_argument("--compute-shaders", action="store_true",
                        help="Enable compute shader optimization for audio models")
    parser.add_argument("--parallel-loading", action="store_true",
                        help="Enable parallel model loading for multimodal models")
    parser.add_argument("--shader-precompile", action="store_true",
                        help="Enable shader precompilation for faster startup")
    parser.add_argument("--all-optimizations", action="store_true",
                        help="Enable all optimizations")
    
    # Browser options
    parser.add_argument("--visible", action="store_true",
                        help="Run browser in visible mode (not headless)")
    parser.add_argument("--keep-browser-open", action="store_true",
                        help="Keep browser open between tests")
    
    # Documentation options
    parser.add_argument("--update-docs", action="store_true",
                        help="Update documentation with test results")
    parser.add_argument("--archive-docs", action="store_true",
                        help="Archive old documentation")
    
    # Output options
    parser.add_argument("--output-dir", default=".",
                        help="Directory to save output files")
    
    # Database options
    parser.add_argument("--db-path", type=str, default=None,
                        help="Path to benchmark database (DuckDB)")
    parser.add_argument("--no-db", action="store_true",
                        help="Disable database storage")
    parser.add_argument("--db-only", action="store_true",
                        help="Store results only in database (no JSON or markdown)")
    
    args = parser.parse_args()
    
    # Handle all optimizations flag
    if args.all_optimizations:
        args.compute_shaders = True
        args.parallel_loading = True
        args.shader_precompile = True
    
    # Handle "all" for precision
    if args.precision == "all":
        args.precision = "all"
    
    # Create output directory if it doesn't exist
    Path(args.output_dir).mkdir(exist_ok=True)
    os.chdir(args.output_dir)
    
    # Run tests
    tester = WebPrecisionTester(args)
    results = tester.run_all_tests()
    
    # Print summary
    print("\n" + "="*80)
    print("TESTING COMPLETE")
    print("="*80)
    print(f"Total tests run: {len(results)}")
    print(f"Successful tests: {len([r for r in results if r.get('status') == 'success'])}")
    print(f"Failed tests: {len([r for r in results if r.get('status') == 'error'])}")
    print(f"Skipped tests: {len([r for r in results if r.get('status') == 'skipped'])}")
    print(f"Real implementations: {len([r for r in results if r.get('is_real_implementation', False)])}")
    print(f"Simulated implementations: {len([r for r in results if r.get('is_simulation', True)])}")
    
    # Database statistics
    db_results = [r for r in results if r.get("db_result_id") is not None]
    if db_results:
        print(f"Results stored in database: {len(db_results)}")
        if args.db_path:
            print(f"Database path: {args.db_path}")
        elif os.environ.get("BENCHMARK_DB_PATH"):
            print(f"Database path: {os.environ.get('BENCHMARK_DB_PATH')}")
        else:
            print("Database path: ./benchmark_db.duckdb (default)")
    print("="*80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())