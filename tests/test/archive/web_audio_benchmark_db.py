#!/usr/bin/env python3
"""
Web Audio Benchmark Database Integration for IPFS Accelerate Python.

This module integrates the Web Audio Platform Testing module with the benchmark
database system, allowing web audio test results to be stored, analyzed and compared
within the comprehensive benchmarking system.
"""

import os
import sys
import json
import time
import logging
import argparse
import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple

# Always deprecate JSON output in favor of DuckDB
DEPRECATE_JSON_OUTPUT = os.environ.get("DEPRECATE_JSON_OUTPUT", "1").lower() in ("1", "true", "yes")


# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the parent directory to sys.path to import modules correctly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class WebAudioBenchmarkDB:
    """Database integration for web audio platform tests."""
    
    def __init__(self, 
                database_path: str = "./benchmark_db.duckdb",
                results_dir: str = "./web_audio_platform_results",
                debug: bool = False):
        """
        Initialize the web audio benchmark database integration.
        
        Args:
            database_path: Path to the benchmark database
            results_dir: Path to the web audio test results directory
            debug: Enable debug logging
        """
        self.database_path = database_path
        self.results_dir = Path(results_dir)
        
        # Set debug logging if requested
        if debug:
            logger.setLevel(logging.DEBUG)
        
        # Import benchmark database modules if available
        try:
            sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))
            from benchmark_db_api import BenchmarkDBAPI
            self.db_api = BenchmarkDBAPI(database_path=database_path)
            self.db_available = True
            logger.info(f"Connected to benchmark database: {database_path}")
        except ImportError as e:
            logger.warning(f"Benchmark database API not available: {e}")
            self.db_api = None
            self.db_available = False
    
    def import_test_results(self, results_file: Optional[str] = None, 
                          all_files: bool = False) -> Dict[str, Any]:
        """
        Import web audio test results into the benchmark database.
        
        Args:
            results_file: Path to a specific results file to import
            all_files: Import all results files in the results directory
            
        Returns:
            Dictionary with import statistics
        """
        if not self.db_available:
            logger.error("Benchmark database not available")
            return {"success": False, "error": "Benchmark database not available"}
        
        stats = {
            "success": False,
            "files_processed": 0,
            "records_imported": 0,
            "errors": [],
            "imported_files": []
        }
        
        try:
            # Process a single file
            if results_file:
                file_path = Path(results_file)
                if not file_path.exists():
                    return {"success": False, "error": f"File not found: {results_file}"}
                
                result = self._import_single_file(file_path)
                stats["files_processed"] = 1
                stats["records_imported"] = result.get("records_imported", 0)
                stats["errors"] = result.get("errors", [])
                stats["imported_files"] = [str(file_path)]
                stats["success"] = len(result.get("errors", [])) == 0
                
            # Process all files in the results directory
            elif all_files:
                files_processed = 0
                records_imported = 0
                all_errors = []
                imported_files = []
                
                # Look for web audio test result files
                for file_path in self.results_dir.glob("*_tests_*.json"):
                    result = self._import_single_file(file_path)
                    files_processed += 1
                    records_imported += result.get("records_imported", 0)
                    all_errors.extend(result.get("errors", []))
                    
                    if result.get("success", False):
                        imported_files.append(str(file_path))
                
                stats["files_processed"] = files_processed
                stats["records_imported"] = records_imported
                stats["errors"] = all_errors
                stats["imported_files"] = imported_files
                stats["success"] = len(all_errors) == 0
            
            else:
                return {"success": False, "error": "No results file specified and all_files not set"}
            
            return stats
            
        except Exception as e:
            logger.error(f"Error importing test results: {e}")
            return {"success": False, "error": str(e)}
    
    def _import_single_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Import a single test results file into the benchmark database.
        
        Args:
            file_path: Path to the results file
            
        Returns:
            Dictionary with import statistics
        """
        result = {
            "success": False,
            "records_imported": 0,
            "errors": []
        }
        
        try:
            # Load the results file
            with open(file_path, 'r') as f:
# Try database first, fall back to JSON if necessary
try:
    from benchmark_db_api import BenchmarkDBAPI
    db_api = BenchmarkDBAPI(db_path=os.environ.get("BENCHMARK_DB_PATH", "./benchmark_db.duckdb"))
    test_data = db_api.get_benchmark_results()
    logger.info("Successfully loaded results from database")
except Exception as e:
    logger.warning(f"Error reading from database, falling back to JSON: {e}")
                    test_data = json.load(f)

            
            # Process based on file format
            if "model_type" in test_data:
                # Single model test results
                result = self._process_single_model_results(test_data, str(file_path))
            elif "results" in test_data and isinstance(test_data["results"], list):
                # Multiple model test results
                records_imported = 0
                all_errors = []
                
                for model_result in test_data["results"]:
                    if "model_type" in model_result and "results" in model_result:
                        sub_result = self._process_single_model_results(
                            model_result["results"], 
                            str(file_path)
                        )
                        records_imported += sub_result.get("records_imported", 0)
                        all_errors.extend(sub_result.get("errors", []))
                
                result["records_imported"] = records_imported
                result["errors"] = all_errors
                result["success"] = len(all_errors) == 0
            else:
                result["errors"].append(f"Unknown result format in {file_path}")
                
            return result
            
        except json.JSONDecodeError as e:
            return {
                "success": False,
                "records_imported": 0,
                "errors": [f"JSON parsing error in {file_path}: {e}"]
            }
        except Exception as e:
            return {
                "success": False,
                "records_imported": 0,
                "errors": [f"Error processing {file_path}: {e}"]
            }
    
    def _process_single_model_results(self, test_data: Dict[str, Any], file_source: str) -> Dict[str, Any]:
        """
        Process test results for a single model and store in the database.
        
        Args:
            test_data: Dictionary containing the test results
            file_source: Path to the source file
            
        Returns:
            Dictionary with import statistics
        """
        result = {
            "success": False,
            "records_imported": 0,
            "errors": []
        }
        
        try:
            # Extract model information
            model_type = test_data.get("model_type")
            model_name = test_data.get("model_name")
            browser = test_data.get("browser")
            timestamp = test_data.get("timestamp")
            
            if not model_type or not model_name or not browser:
                result["errors"].append(f"Missing required fields in test data: {file_source}")
                return result
            
            # Make timestamp a proper datetime if it's a string
            if isinstance(timestamp, str):
                try:
                    timestamp = datetime.datetime.fromisoformat(timestamp)
                except ValueError:
                    timestamp = datetime.datetime.now()
            
            # Store test data in the database
            tests = test_data.get("tests", [])
            records_imported = 0
            
            for test in tests:
                test_type = test.get("test_type")
                status = test.get("status")
                
                if test_type and status:
                    try:
                        # Prepare metrics based on test_type
                        metrics = {}
                        
                        # Extract metrics if available
                        if "results" in test and isinstance(test["results"], dict):
                            test_results = test["results"]
                            
                            # Common metrics
                            if "execution_time" in test_results:
                                metrics["execution_time"] = test_results["execution_time"]
                            
                            # WebGPU specific metrics
                            if test_type == "webgpu" and "webgpu_metrics" in test_results:
                                webgpu_metrics = test_results["webgpu_metrics"]
                                for key, value in webgpu_metrics.items():
                                    metrics[f"webgpu_{key}"] = value
                            
                            # WebNN specific metrics
                            if test_type == "webnn" and "webnn_metrics" in test_results:
                                webnn_metrics = test_results["webnn_metrics"]
                                for key, value in webnn_metrics.items():
                                    metrics[f"webnn_{key}"] = value
                            
                            # Model-specific metrics
                            if model_type == "whisper" and "whisper_metrics" in test_results:
                                whisper_metrics = test_results["whisper_metrics"]
                                for key, value in whisper_metrics.items():
                                    metrics[f"whisper_{key}"] = value
                            
                            if model_type == "wav2vec2" and "wav2vec2_metrics" in test_results:
                                wav2vec2_metrics = test_results["wav2vec2_metrics"]
                                for key, value in wav2vec2_metrics.items():
                                    metrics[f"wav2vec2_{key}"] = value
                            
                            if model_type == "clap" and "clap_metrics" in test_results:
                                clap_metrics = test_results["clap_metrics"]
                                for key, value in clap_metrics.items():
                                    metrics[f"clap_{key}"] = value
                        
                        # Store in database
                        self.db_api.store_web_platform_result(
                            model_name=model_name,
                            model_type=model_type,
                            browser=browser,
                            platform=test_type,
                            status=status,
                            metrics=metrics,
                            timestamp=timestamp,
                            source_file=file_source
                        )
                        
                        records_imported += 1
                    except Exception as e:
                        result["errors"].append(f"Error storing test result: {e}")
            
            result["records_imported"] = records_imported
            result["success"] = len(result["errors"]) == 0
            return result
            
        except Exception as e:
            result["errors"].append(f"Error processing test data: {e}")
            return result
    
    def query_web_audio_results(self, query_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Query web audio test results from the benchmark database.
        
        Args:
            query_params: Dictionary with query parameters
            
        Returns:
            Dictionary with query results
        """
        if not self.db_available:
            logger.error("Benchmark database not available")
            return {"success": False, "error": "Benchmark database not available"}
        
        try:
            # Default query parameters
            params = {
                "model_type": None,
                "model_name": None,
                "browser": None,
                "platform": None,
                "status": None,
                "start_date": None,
                "end_date": None,
                "limit": 100
            }
            
            # Update with provided parameters
            if query_params:
                params.update(query_params)
            
            # Query the database
            results = self.db_api.query_web_platform_results(**params)
            return {"success": True, "results": results}
            
        except Exception as e:
            logger.error(f"Error querying web audio results: {e}")
            return {"success": False, "error": str(e)}
    
    def generate_web_audio_report(self, query_params: Dict[str, Any] = None, 
                               output_file: str = None) -> Dict[str, Any]:
        """
        Generate a report on web audio test results.
        
        Args:
            query_params: Dictionary with query parameters
            output_file: Path to the output file
            
        Returns:
            Dictionary with report generation results
        """
        if not self.db_available:
            logger.error("Benchmark database not available")
            return {"success": False, "error": "Benchmark database not available"}
        
        try:
            # Query results
            query_result = self.query_web_audio_results(query_params)
            if not query_result["success"]:
                return query_result
            
            results = query_result["results"]
            
            # Generate default output file name if not provided
            if not output_file:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = f"web_audio_report_{timestamp}.md"
            
            # Generate report
            with open(output_file, 'w') as f:
                # Write header
                f.write("# Web Audio Platform Benchmark Report\n\n")
                f.write(f"Generated: {datetime.datetime.now().isoformat()}\n\n")
                
                # Summary statistics
                f.write("## Summary\n\n")
                total_tests = len(results)
                successful_tests = sum(1 for r in results if r.get("status") == "successful")
                webnn_tests = sum(1 for r in results if r.get("platform") == "webnn")
                webgpu_tests = sum(1 for r in results if r.get("platform") == "webgpu")
                
                f.write(f"- Total Tests: {total_tests}\n")
                f.write(f"- Successful Tests: {successful_tests} ({successful_tests/total_tests*100:.1f}% if total_tests > 0 else 0}%)\n")
                f.write(f"- WebNN Tests: {webnn_tests}\n")
                f.write(f"- WebGPU Tests: {webgpu_tests}\n\n")
                
                # Browser statistics
                browsers = {}
                for r in results:
                    browser = r.get("browser")
                    if browser:
                        browsers[browser] = browsers.get(browser, 0) + 1
                
                f.write("### Browser Distribution\n\n")
                for browser, count in browsers.items():
                    f.write(f"- {browser}: {count} tests\n")
                f.write("\n")
                
                # Model type statistics
                model_types = {}
                for r in results:
                    model_type = r.get("model_type")
                    if model_type:
                        model_types[model_type] = model_types.get(model_type, 0) + 1
                
                f.write("### Model Type Distribution\n\n")
                for model_type, count in model_types.items():
                    f.write(f"- {model_type}: {count} tests\n")
                f.write("\n")
                
                # Create comparison tables by model type
                f.write("## Performance by Model Type\n\n")
                
                # Group results by model type
                grouped_results = {}
                for r in results:
                    model_type = r.get("model_type")
                    if model_type:
                        if model_type not in grouped_results:
                            grouped_results[model_type] = []
                        grouped_results[model_type].append(r)
                
                # Generate tables for each model type
                for model_type, type_results in grouped_results.items():
                    f.write(f"### {model_type.upper()}\n\n")
                    f.write("| Model | Browser | Platform | Status | Execution Time (ms) | Notes |\n")
                    f.write("|-------|---------|----------|--------|---------------------|-------|\n")
                    
                    for r in type_results:
                        model_name = r.get("model_name", "Unknown")
                        browser = r.get("browser", "Unknown")
                        platform = r.get("platform", "Unknown")
                        status = r.get("status", "Unknown")
                        
                        # Extract execution time from metrics
                        metrics = r.get("metrics", {})
                        execution_time = metrics.get("execution_time", "N/A")
                        if execution_time != "N/A":
                            execution_time = f"{float(execution_time) * 1000:.2f}"  # Convert to ms
                        
                        # Additional notes
                        notes = ""
                        if status != "successful":
                            notes = r.get("error_message", "")
                        
                        f.write(f"| {model_name} | {browser} | {platform} | {status} | {execution_time} | {notes} |\n")
                    
                    f.write("\n")
                
                # Performance comparison between WebNN and WebGPU
                f.write("## WebNN vs WebGPU Performance\n\n")
                f.write("| Model | Browser | WebNN Time (ms) | WebGPU Time (ms) | Speedup Factor | Faster Platform |\n")
                f.write("|-------|---------|-----------------|------------------|----------------|----------------|\n")
                
                # Group results by model and browser
                comparison_results = {}
                for r in results:
                    if r.get("status") != "successful":
                        continue
                    
                    model_name = r.get("model_name")
                    browser = r.get("browser")
                    platform = r.get("platform")
                    
                    if model_name and browser and platform:
                        key = f"{model_name}|{browser}"
                        if key not in comparison_results:
                            comparison_results[key] = {}
                        
                        metrics = r.get("metrics", {})
                        execution_time = metrics.get("execution_time")
                        if execution_time:
                            comparison_results[key][platform] = float(execution_time) * 1000  # ms
                
                # Generate comparison rows
                for key, platforms in comparison_results.items():
                    if "webnn" in platforms and "webgpu" in platforms:
                        model_name, browser = key.split("|")
                        webnn_time = platforms["webnn"]
                        webgpu_time = platforms["webgpu"]
                        
                        if webnn_time > 0 and webgpu_time > 0:
                            speedup = webnn_time / webgpu_time
                            faster = "WebGPU" if speedup > 1 else "WebNN"
                            
                            f.write(f"| {model_name} | {browser} | {webnn_time:.2f} | {webgpu_time:.2f} | {speedup:.2f}x | {faster} |\n")
                
                f.write("\n")
                
                # Recommendations
                f.write("## Recommendations\n\n")
                
                # Determine overall faster platform
                webnn_times = []
                webgpu_times = []
                
                for platforms in comparison_results.values():
                    if "webnn" in platforms:
                        webnn_times.append(platforms["webnn"])
                    if "webgpu" in platforms:
                        webgpu_times.append(platforms["webgpu"])
                
                if webnn_times and webgpu_times:
                    avg_webnn = sum(webnn_times) / len(webnn_times)
                    avg_webgpu = sum(webgpu_times) / len(webgpu_times)
                    
                    if avg_webnn < avg_webgpu:
                        f.write("- **Overall Recommendation**: WebNN appears to be faster on average for audio models\n")
                    else:
                        f.write("- **Overall Recommendation**: WebGPU appears to be faster on average for audio models\n")
                
                # Model-specific recommendations
                f.write("\n### Model-Specific Recommendations\n\n")
                
                for model_type in grouped_results:
                    f.write(f"#### {model_type.upper()}\n\n")
                    
                    # Calculate average performance by platform for this model type
                    platform_times = {"webnn": [], "webgpu": []}
                    
                    for r in grouped_results[model_type]:
                        if r.get("status") != "successful":
                            continue
                        
                        platform = r.get("platform")
                        metrics = r.get("metrics", {})
                        execution_time = metrics.get("execution_time")
                        
                        if platform and execution_time:
                            platform_times[platform].append(float(execution_time) * 1000)  # ms
                    
                    # Generate recommendation
                    if platform_times["webnn"] and platform_times["webgpu"]:
                        avg_webnn = sum(platform_times["webnn"]) / len(platform_times["webnn"])
                        avg_webgpu = sum(platform_times["webgpu"]) / len(platform_times["webgpu"])
                        
                        if avg_webnn < avg_webgpu:
                            f.write(f"- Recommended platform: **WebNN** (avg: {avg_webnn:.2f}ms vs {avg_webgpu:.2f}ms)\n")
                        else:
                            f.write(f"- Recommended platform: **WebGPU** (avg: {avg_webgpu:.2f}ms vs {avg_webnn:.2f}ms)\n")
                    else:
                        f.write("- Insufficient data for recommendation\n")
                    
                    f.write("\n")
                
                # Implementation recommendations
                f.write("### Implementation Recommendations\n\n")
                f.write("1. **Model Size Optimization**: Optimize audio models for web deployment\n")
                f.write("2. **Web-Native Audio Processing**: Implement audio preprocessing directly in web backends\n")
                f.write("3. **Streaming Support**: Add support for streaming audio input\n")
                f.write("4. **Specialized Web Variants**: Create specialized variants of audio models for WebNN/WebGPU\n")
                f.write("5. **Progressive Loading**: Implement progressive model loading for better UX\n")
            
            logger.info(f"Generated web audio report: {output_file}")
            return {"success": True, "report_file": output_file}
            
        except Exception as e:
            logger.error(f"Error generating web audio report: {e}")
            return {"success": False, "error": str(e)}
    
    def compare_web_audio_platforms(self, model_types: List[str] = None, 
                                 browsers: List[str] = None,
                                 output_file: str = None) -> Dict[str, Any]:
        """
        Compare WebNN and WebGPU performance for audio models.
        
        Args:
            model_types: List of model types to compare (whisper, wav2vec2, clap)
            browsers: List of browsers to include
            output_file: Path to the output file
            
        Returns:
            Dictionary with comparison results
        """
        if not self.db_available:
            logger.error("Benchmark database not available")
            return {"success": False, "error": "Benchmark database not available"}
        
        try:
            # Build query parameters
            query_params = {
                "status": "successful",
                "limit": 1000  # Get a larger sample for comparison
            }
            
            if model_types:
                query_params["model_type"] = model_types
            
            if browsers:
                query_params["browser"] = browsers
            
            # Query results
            query_result = self.query_web_audio_results(query_params)
            if not query_result["success"]:
                return query_result
            
            results = query_result["results"]
            
            # Generate default output file name if not provided
            if not output_file:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = f"web_audio_platform_comparison_{timestamp}.md"
            
            # Process results for comparison
            comparison_data = self._process_platform_comparison(results)
            
            # Generate comparison report
            with open(output_file, 'w') as f:
                # Write header
                f.write("# Web Audio Platform Comparison\n\n")
                f.write(f"Generated: {datetime.datetime.now().isoformat()}\n\n")
                
                # Overall statistics
                f.write("## Overall Comparison\n\n")
                
                # Compute average speedup
                speedups = []
                
                for key, data in comparison_data["comparisons"].items():
                    if "speedup" in data:
                        speedups.append(data["speedup"])
                
                avg_speedup = sum(speedups) / len(speedups) if speedups else 1.0
                faster_platform = "WebGPU" if avg_speedup > 1.0 else "WebNN"
                
                f.write(f"- **Tests Analyzed**: {comparison_data['total_comparisons']}\n")
                f.write(f"- **Average Speedup**: {avg_speedup:.2f}x\n")
                f.write(f"- **Faster Platform Overall**: {faster_platform}\n\n")
                
                # Platform comparison table
                f.write("## Detailed Comparison\n\n")
                f.write("| Model | Browser | WebNN Time (ms) | WebGPU Time (ms) | Speedup | Faster Platform |\n")
                f.write("|-------|---------|-----------------|------------------|---------|----------------|\n")
                
                for key, data in comparison_data["comparisons"].items():
                    model_name, browser = key.split("|")
                    
                    webnn_time = data.get("webnn_time", "N/A")
                    if webnn_time != "N/A":
                        webnn_time = f"{webnn_time:.2f}"
                    
                    webgpu_time = data.get("webgpu_time", "N/A")
                    if webgpu_time != "N/A":
                        webgpu_time = f"{webgpu_time:.2f}"
                    
                    speedup = data.get("speedup", "N/A")
                    if speedup != "N/A":
                        speedup = f"{speedup:.2f}x"
                    
                    faster = data.get("faster", "N/A")
                    
                    f.write(f"| {model_name} | {browser} | {webnn_time} | {webgpu_time} | {speedup} | {faster} |\n")
                
                f.write("\n")
                
                # Model type comparison
                f.write("## Model Type Comparison\n\n")
                f.write("| Model Type | Avg WebNN Time (ms) | Avg WebGPU Time (ms) | Avg Speedup | Better Platform |\n")
                f.write("|------------|---------------------|----------------------|-------------|----------------|\n")
                
                for model_type, data in comparison_data["model_types"].items():
                    avg_webnn = data.get("avg_webnn_time", "N/A")
                    if avg_webnn != "N/A":
                        avg_webnn = f"{avg_webnn:.2f}"
                    
                    avg_webgpu = data.get("avg_webgpu_time", "N/A")
                    if avg_webgpu != "N/A":
                        avg_webgpu = f"{avg_webgpu:.2f}"
                    
                    avg_speedup = data.get("avg_speedup", "N/A")
                    if avg_speedup != "N/A":
                        avg_speedup = f"{avg_speedup:.2f}x"
                    
                    better_platform = data.get("better_platform", "N/A")
                    
                    f.write(f"| {model_type} | {avg_webnn} | {avg_webgpu} | {avg_speedup} | {better_platform} |\n")
                
                f.write("\n")
                
                # Browser comparison
                f.write("## Browser Comparison\n\n")
                f.write("| Browser | Avg WebNN Time (ms) | Avg WebGPU Time (ms) | Avg Speedup | Better Platform |\n")
                f.write("|---------|---------------------|----------------------|-------------|----------------|\n")
                
                for browser, data in comparison_data["browsers"].items():
                    avg_webnn = data.get("avg_webnn_time", "N/A")
                    if avg_webnn != "N/A":
                        avg_webnn = f"{avg_webnn:.2f}"
                    
                    avg_webgpu = data.get("avg_webgpu_time", "N/A")
                    if avg_webgpu != "N/A":
                        avg_webgpu = f"{avg_webgpu:.2f}"
                    
                    avg_speedup = data.get("avg_speedup", "N/A")
                    if avg_speedup != "N/A":
                        avg_speedup = f"{avg_speedup:.2f}x"
                    
                    better_platform = data.get("better_platform", "N/A")
                    
                    f.write(f"| {browser} | {avg_webnn} | {avg_webgpu} | {avg_speedup} | {better_platform} |\n")
                
                f.write("\n")
                
                # Recommendations
                f.write("## Recommendations\n\n")
                
                # Overall recommendation
                f.write("### Overall Recommendation\n\n")
                f.write(f"Based on test results, **{faster_platform}** appears to be the better platform for audio models on the web, with an average speedup of {avg_speedup:.2f}x.\n\n")
                
                # Model-specific recommendations
                f.write("### Model-Specific Recommendations\n\n")
                
                for model_type, data in comparison_data["model_types"].items():
                    better_platform = data.get("better_platform", "N/A")
                    avg_speedup = data.get("avg_speedup", "N/A")
                    
                    if better_platform != "N/A" and avg_speedup != "N/A":
                        f.write(f"- **{model_type}**: Use **{better_platform}** (speedup: {avg_speedup:.2f}x)\n")
                
                f.write("\n")
                
                # Browser-specific recommendations
                f.write("### Browser-Specific Recommendations\n\n")
                
                for browser, data in comparison_data["browsers"].items():
                    better_platform = data.get("better_platform", "N/A")
                    avg_speedup = data.get("avg_speedup", "N/A")
                    
                    if better_platform != "N/A" and avg_speedup != "N/A":
                        f.write(f"- **{browser}**: Use **{better_platform}** (speedup: {avg_speedup:.2f}x)\n")
            
            logger.info(f"Generated web audio platform comparison: {output_file}")
            return {
                "success": True, 
                "report_file": output_file,
                "comparison_data": comparison_data
            }
            
        except Exception as e:
            logger.error(f"Error comparing web audio platforms: {e}")
            return {"success": False, "error": str(e)}
    
    def _process_platform_comparison(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process test results for platform comparison.
        
        Args:
            results: List of test result dictionaries
            
        Returns:
            Dictionary with comparison data
        """
        # Initialize comparison data
        comparison_data = {
            "comparisons": {},
            "model_types": {},
            "browsers": {},
            "total_comparisons": 0
        }
        
        # Group results by model and browser
        grouped_results = {}
        
        for r in results:
            if r.get("status") != "successful":
                continue
            
            model_name = r.get("model_name")
            model_type = r.get("model_type")
            browser = r.get("browser")
            platform = r.get("platform")
            
            if model_name and browser and platform and model_type:
                # Store by model|browser combination
                key = f"{model_name}|{browser}"
                if key not in grouped_results:
                    grouped_results[key] = {
                        "model_name": model_name,
                        "model_type": model_type,
                        "browser": browser,
                        "platforms": {}
                    }
                
                metrics = r.get("metrics", {})
                execution_time = metrics.get("execution_time")
                
                if execution_time:
                    grouped_results[key]["platforms"][platform] = float(execution_time) * 1000  # ms
        
        # Process comparison data
        for key, data in grouped_results.items():
            model_name = data["model_name"]
            model_type = data["model_type"]
            browser = data["browser"]
            platforms = data["platforms"]
            
            # Initialize model type data if not exists
            if model_type not in comparison_data["model_types"]:
                comparison_data["model_types"][model_type] = {
                    "webnn_times": [],
                    "webgpu_times": [],
                    "speedups": []
                }
            
            # Initialize browser data if not exists
            if browser not in comparison_data["browsers"]:
                comparison_data["browsers"][browser] = {
                    "webnn_times": [],
                    "webgpu_times": [],
                    "speedups": []
                }
            
            # Skip if not both platforms available
            if "webnn" not in platforms or "webgpu" not in platforms:
                continue
            
            webnn_time = platforms["webnn"]
            webgpu_time = platforms["webgpu"]
            
            # Skip invalid times
            if webnn_time <= 0 or webgpu_time <= 0:
                continue
            
            # Calculate speedup (WebNN/WebGPU)
            # > 1.0 means WebGPU is faster, < 1.0 means WebNN is faster
            speedup = webnn_time / webgpu_time
            faster = "WebGPU" if speedup > 1.0 else "WebNN"
            
            # Store comparison data
            comparison_data["comparisons"][key] = {
                "webnn_time": webnn_time,
                "webgpu_time": webgpu_time,
                "speedup": speedup,
                "faster": faster
            }
            
            # Update model type data
            comparison_data["model_types"][model_type]["webnn_times"].append(webnn_time)
            comparison_data["model_types"][model_type]["webgpu_times"].append(webgpu_time)
            comparison_data["model_types"][model_type]["speedups"].append(speedup)
            
            # Update browser data
            comparison_data["browsers"][browser]["webnn_times"].append(webnn_time)
            comparison_data["browsers"][browser]["webgpu_times"].append(webgpu_time)
            comparison_data["browsers"][browser]["speedups"].append(speedup)
            
            # Increment total comparisons
            comparison_data["total_comparisons"] += 1
        
        # Calculate averages for model types
        for model_type, data in comparison_data["model_types"].items():
            webnn_times = data.get("webnn_times", [])
            webgpu_times = data.get("webgpu_times", [])
            speedups = data.get("speedups", [])
            
            if webnn_times and webgpu_times and speedups:
                avg_webnn_time = sum(webnn_times) / len(webnn_times)
                avg_webgpu_time = sum(webgpu_times) / len(webgpu_times)
                avg_speedup = sum(speedups) / len(speedups)
                
                data["avg_webnn_time"] = avg_webnn_time
                data["avg_webgpu_time"] = avg_webgpu_time
                data["avg_speedup"] = avg_speedup
                data["better_platform"] = "WebGPU" if avg_speedup > 1.0 else "WebNN"
        
        # Calculate averages for browsers
        for browser, data in comparison_data["browsers"].items():
            webnn_times = data.get("webnn_times", [])
            webgpu_times = data.get("webgpu_times", [])
            speedups = data.get("speedups", [])
            
            if webnn_times and webgpu_times and speedups:
                avg_webnn_time = sum(webnn_times) / len(webnn_times)
                avg_webgpu_time = sum(webgpu_times) / len(webgpu_times)
                avg_speedup = sum(speedups) / len(speedups)
                
                data["avg_webnn_time"] = avg_webnn_time
                data["avg_webgpu_time"] = avg_webgpu_time
                data["avg_speedup"] = avg_speedup
                data["better_platform"] = "WebGPU" if avg_speedup > 1.0 else "WebNN"
        
        return comparison_data

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Web Audio Benchmark Database Integration")
    
    # Basic options
    parser.add_argument("--database", default="./benchmark_db.duckdb",
                      help="Path to the benchmark database")
    parser.add_argument("--results-dir", default="./web_audio_platform_results",
                      help="Directory for web audio test results")
    parser.add_argument("--debug", action="store_true",
                      help="Enable debug logging")
    
    # Import options
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--import-file", help="Import a specific results file")
    group.add_argument("--import-all", action="store_true", help="Import all results files")
    
    # Query and report options
    parser.add_argument("--generate-report", action="store_true",
                      help="Generate a report from benchmark database")
    parser.add_argument("--output-file", help="Path to the output report file")
    parser.add_argument("--compare-platforms", action="store_true",
                      help="Compare WebNN and WebGPU performance")
    
    # Filter options
    parser.add_argument("--model-types", nargs="+", 
                      help="Filter by model types (whisper, wav2vec2, clap)")
    parser.add_argument("--browsers", nargs="+",
                      help="Filter by browsers (chrome, firefox, safari, edge)")
    parser.add_argument("--start-date", help="Filter by start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", help="Filter by end date (YYYY-MM-DD)")
    
    return parser.parse_args()

def main():
    """Main entry point for the script."""
    args = parse_arguments()
    
    # Create database integration
    db_integration = WebAudioBenchmarkDB(
        database_path=args.database,
        results_dir=args.results_dir,
        debug=args.debug
    )
    
    # Import results if requested
    if args.import_file:
        logger.info(f"Importing results file: {args.import_file}")
        result = db_integration.import_test_results(results_file=args.import_file)
        
        if result["success"]:
            logger.info(f"Imported {result['records_imported']} records successfully")
        else:
            logger.error(f"Import failed: {result.get('error', 'Unknown error')}")
            for error in result.get("errors", []):
                logger.error(f"- {error}")
    
    elif args.import_all:
        logger.info("Importing all results files")
        result = db_integration.import_test_results(all_files=True)
        
        if result["success"]:
            logger.info(f"Imported {result['records_imported']} records from {result['files_processed']} files")
        else:
            logger.error(f"Import failed: {result.get('error', 'Unknown error')}")
            for error in result.get("errors", [])[:10]:  # Show at most 10 errors
                logger.error(f"- {error}")
            
            if len(result.get("errors", [])) > 10:
                logger.error(f"... and {len(result.get('errors', [])) - 10} more errors")
    
    # Generate report if requested
    if args.generate_report:
        logger.info("Generating web audio report")
        
        # Build query parameters
        query_params = {}
        
        if args.model_types:
            query_params["model_type"] = args.model_types
        
        if args.browsers:
            query_params["browser"] = args.browsers
        
        if args.start_date:
            query_params["start_date"] = args.start_date
        
        if args.end_date:
            query_params["end_date"] = args.end_date
        
        result = db_integration.generate_web_audio_report(
            query_params=query_params,
            output_file=args.output_file
        )
        
        if result["success"]:
            logger.info(f"Generated report: {result['report_file']}")
        else:
            logger.error(f"Report generation failed: {result.get('error', 'Unknown error')}")
    
    # Compare platforms if requested
    if args.compare_platforms:
        logger.info("Comparing web audio platforms")
        
        result = db_integration.compare_web_audio_platforms(
            model_types=args.model_types,
            browsers=args.browsers,
            output_file=args.output_file
        )
        
        if result["success"]:
            logger.info(f"Generated platform comparison: {result['report_file']}")
            
            # Show brief summary
            if "comparison_data" in result:
                data = result["comparison_data"]
                total = data.get("total_comparisons", 0)
                
                speedups = []
                for key, comp in data.get("comparisons", {}).items():
                    if "speedup" in comp:
                        speedups.append(comp["speedup"])
                
                avg_speedup = sum(speedups) / len(speedups) if speedups else 1.0
                faster = "WebGPU" if avg_speedup > 1.0 else "WebNN"
                
                logger.info(f"Analyzed {total} comparisons")
                logger.info(f"Average speedup: {avg_speedup:.2f}x")
                logger.info(f"Faster platform overall: {faster}")
                
        else:
            logger.error(f"Platform comparison failed: {result.get('error', 'Unknown error')}")
    
    # If no action specified, show help
    if not (args.import_file or args.import_all or args.generate_report or args.compare_platforms):
        import sys
        parser = argparse.ArgumentParser(description="Web Audio Benchmark Database Integration")
        parser.print_help()
        sys.exit(1)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())