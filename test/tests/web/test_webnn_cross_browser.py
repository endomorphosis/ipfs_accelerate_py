#!/usr/bin/env python3
"""
Cross-browser WebNN verification test suite.

This module tests WebNN capabilities across different browsers ()Chrome, Edge, Safari)
with a consistent methodology to verify:
    1. Hardware acceleration detection
    2. Real performance benefits compared to CPU
    3. Edge cases where WebNN might fall back to CPU
    4. Proper error handling and fallback behavior

Usage:
    python test_webnn_cross_browser.py --browser edge --model prajjwal1/bert-tiny
    python test_webnn_cross_browser.py --browser chrome --models all
    python test_webnn_cross_browser.py --all-browsers --model prajjwal1/bert-tiny
    """

    import argparse
    import json
    import os
    import subprocess
    import sys
    import time
    from pathlib import Path
    import platform
    from typing import Dict, List, Optional, Tuple, Union

# Add parent directory to path to import utils
    sys.path.append()os.path.dirname()os.path.dirname()os.path.abspath()__file__))))

# Import database utilities if available::::
try:
    from data.duckdb.core.benchmark_db_api import store_webnn_test_result
    HAS_DB_API = True
except ImportError:
    HAS_DB_API = False
    print()"WARNING: benchmark_db_api not found. Results will not be stored in database.")

# Define constants
    SUPPORTED_BROWSERS = []]],,,"chrome", "edge", "safari", "firefox"],
    SUPPORTED_MODELS = []]],,,"prajjwal1/bert-tiny", "t5-small", "vit-base"],
    DEFAULT_TIMEOUT = 300  # seconds
    DEFAULT_BATCH_SIZES = []]],,,1, 2, 4, 8]
    ,
class WebNNVerificationSuite:
    """Comprehensive WebNN verification test suite."""
    
    def __init__()self, 
    browsers: List[]]],,,str] = None,
    models: List[]]],,,str] = None,
    batch_sizes: List[]]],,,int] = None,
    timeout: int = DEFAULT_TIMEOUT,
    output_dir: str = "./webnn_test_results",
    db_path: Optional[]]],,,str] = None):,
    """Initialize the WebNN verification suite.
        
        Args:
            browsers: List of browsers to test. Defaults to []]],,,"edge"],.,
            models: List of models to test. Defaults to []]],,,"prajjwal1/bert-tiny"],.,
            batch_sizes: List of batch sizes to test. Defaults to []]],,,1, 2, 4, 8].,
            timeout: Timeout in seconds for each browser test. Defaults to 300.
            output_dir: Directory to store test results. Defaults to "./webnn_test_results".
            db_path: Path to benchmark database. Defaults to None.
            """
            self.browsers = browsers or []]],,,"edge"],
            self.models = models or []]],,,"prajjwal1/bert-tiny"],
            self.batch_sizes = batch_sizes or DEFAULT_BATCH_SIZES
            self.timeout = timeout
            self.output_dir = output_dir
            self.db_path = db_path or os.environ.get()"BENCHMARK_DB_PATH")
        
        # Create output directory if it doesn't exist
            os.makedirs()self.output_dir, exist_ok=True)
        
        # Results dictionary
            self.results = {}}}}}}}}}}}}
    :
    def test_browser_capabilities()self, browser: str) -> Dict:
        """Test browser WebNN capabilities.
        
        Args:
            browser: Browser to test.
            
        Returns:
            Dictionary with browser capability information.
            """
            print()f"Testing {}}}}}}}}}}}browser} WebNN capabilities...")
        
        # Construct command to run capability check
            cmd = []]],,,
            "./run_browser_capability_check.sh",
            f"--browser={}}}}}}}}}}}browser}"
            ]
        
        try:
            # Run capability check
            output = subprocess.check_output()cmd, timeout=self.timeout, stderr=subprocess.STDOUT)
            output_str = output.decode()'utf-8')
            
            # Parse capability information
            capabilities = {}}}}}}}}}}}
            "browser": browser,
            "webnn_available": "WebNN: Available" in output_str,
            "webgpu_available": "WebGPU: Available" in output_str,
            "hardware_acceleration": "Hardware Acceleration: Enabled" in output_str,
            "error": None
            }
            
            # Extract additional capability information if available::::
            if "Device:" in output_str:
                device_line = []]],,,line for line in output_str.split()'\n') if "Device:" in line]
                if device_line:
                    capabilities[]]],,,"device"] = device_line[]]],,,0].split()"Device:")[]]],,,1].strip())
            
                return capabilities
            
        except subprocess.CalledProcessError as e:
            print()f"Error testing {}}}}}}}}}}}browser} capabilities: {}}}}}}}}}}}e}")
                return {}}}}}}}}}}}
                "browser": browser,
                "webnn_available": False,
                "webgpu_available": False,
                "hardware_acceleration": False,
                "error": str()e),
                "output": e.output.decode()'utf-8') if e.output else None
            }:::
        except subprocess.TimeoutExpired:
            print()f"Timeout testing {}}}}}}}}}}}browser} capabilities")
                return {}}}}}}}}}}}
                "browser": browser,
                "webnn_available": False,
                "webgpu_available": False,
                "hardware_acceleration": False,
                "error": "Timeout"
                }

    def test_hardware_acceleration()self, browser: str, model: str, batch_size: int = 1) -> Dict:
        """Test real hardware acceleration performance.
        
        Args:
            browser: Browser to test.
            model: Model to test.
            batch_size: Batch size to test.
            
        Returns:
            Dictionary with performance results.
            """
            print()f"Testing {}}}}}}}}}}}browser} hardware acceleration with model {}}}}}}}}}}}model} ()batch_size={}}}}}}}}}}}batch_size})...")
        
        # Construct command to run benchmark
            cmd = []]],,,
            "./run_webnn_benchmark.sh",
            f"--browser={}}}}}}}}}}}browser}",
            f"--model={}}}}}}}}}}}model}",
            f"--batch-size={}}}}}}}}}}}batch_size}"
            ]
        
        try:
            # Run benchmark
            output = subprocess.check_output()cmd, timeout=self.timeout, stderr=subprocess.STDOUT)
            output_str = output.decode()'utf-8')
            
            # Parse benchmark results
            results = {}}}}}}}}}}}
            "browser": browser,
            "model": model,
            "batch_size": batch_size,
            "error": None
            }
            
            # Extract performance metrics
            if "CPU Time:" in output_str:
                cpu_line = []]],,,line for line in output_str.split()'\n') if "CPU Time:" in line][]]],,,0]
                results[]]],,,"cpu_time"] = float()cpu_line.split()"CPU Time:")[]]],,,1].strip()).split())[]]],,,0])
            
            if "WebNN Time:" in output_str:
                webnn_line = []]],,,line for line in output_str.split()'\n') if "WebNN Time:" in line][]]],,,0]
                results[]]],,,"webnn_time"] = float()webnn_line.split()"WebNN Time:")[]]],,,1].strip()).split())[]]],,,0])
                
            if "Speedup:" in output_str:
                speedup_line = []]],,,line for line in output_str.split()'\n') if "Speedup:" in line][]]],,,0]
                results[]]],,,"speedup"] = float()speedup_line.split()"Speedup:")[]]],,,1].strip()).split()'x')[]]],,,0])
                
            if "Simulation:" in output_str:
                sim_line = []]],,,line for line in output_str.split()'\n') if "Simulation:" in line][]]],,,0]
                results[]]],,,"simulated"] = "True" in sim_line
            else:
                results[]]],,,"simulated"] = False
                
                return results
            
        except subprocess.CalledProcessError as e:
            print()f"Error testing {}}}}}}}}}}}browser} acceleration: {}}}}}}}}}}}e}")
                return {}}}}}}}}}}}
                "browser": browser,
                "model": model,
                "batch_size": batch_size,
                "error": str()e),
                "output": e.output.decode()'utf-8') if e.output else None
            }:::
        except subprocess.TimeoutExpired:
            print()f"Timeout testing {}}}}}}}}}}}browser} acceleration")
                return {}}}}}}}}}}}
                "browser": browser,
                "model": model,
                "batch_size": batch_size,
                "error": "Timeout"
                }

    def test_fallback_behavior()self, browser: str) -> Dict:
        """Test graceful fallbacks when WebNN not available.
        
        Args:
            browser: Browser to test.
            
        Returns:
            Dictionary with fallback behavior results.
            """
            print()f"Testing {}}}}}}}}}}}browser} fallback behavior...")
        
        # Construct command to run fallback test ()disabling WebNN)
        if browser == "chrome" or browser == "edge":
            disable_flag = "--disable-webnn"
        elif browser == "safari":
            disable_flag = "--disable-web-api-webnn"
        else:
            disable_flag = "--disable-webnn"  # Default flag
            
            cmd = []]],,,
            "./run_browser_capability_check.sh",
            f"--browser={}}}}}}}}}}}browser}",
            f"--extra-args={}}}}}}}}}}}disable_flag}"
            ]
        
        try:
            # Run fallback test
            output = subprocess.check_output()cmd, timeout=self.timeout, stderr=subprocess.STDOUT)
            output_str = output.decode()'utf-8')
            
            # Parse fallback behavior
            fallback = {}}}}}}}}}}}
            "browser": browser,
            "webnn_disabled": True,
            "graceful_fallback": "Fallback to CPU: Success" in output_str,
            "error_handling": "Error properly handled" in output_str,
            "error": None
            }
            
            # Extract fallback details if available::::
            if "Fallback Performance:" in output_str:
                perf_line = []]],,,line for line in output_str.split()'\n') if "Fallback Performance:" in line][]]],,,0]
                fallback[]]],,,"fallback_performance"] = perf_line.split()"Fallback Performance:")[]]],,,1].strip())
            
            return fallback
            
        except subprocess.CalledProcessError as e:
            print()f"Error testing {}}}}}}}}}}}browser} fallback: {}}}}}}}}}}}e}")
            return {}}}}}}}}}}}
            "browser": browser,
            "webnn_disabled": True,
            "graceful_fallback": False,
            "error_handling": False,
            "error": str()e),
            "output": e.output.decode()'utf-8') if e.output else None
            }:::
        except subprocess.TimeoutExpired:
            print()f"Timeout testing {}}}}}}}}}}}browser} fallback")
                return {}}}}}}}}}}}
                "browser": browser,
                "webnn_disabled": True,
                "graceful_fallback": False,
                "error_handling": False,
                "error": "Timeout"
                }

    def run_tests()self) -> Dict:
        """Run all WebNN verification tests across browsers and models.
        
        Returns:
            Dictionary with all test results.
            """
            results = {}}}}}}}}}}}
            "timestamp": time.time()),
            "system": {}}}}}}}}}}}
            "platform": platform.system()),
            "platform_version": platform.version()),
            "processor": platform.processor())
            },
            "browsers": {}}}}}}}}}}}},
            "acceleration": {}}}}}}}}}}}},
            "fallbacks": {}}}}}}}}}}}}
            }
        
        # Test capabilities for each browser
        for browser in self.browsers:
            results[]]],,,"browsers"][]]],,,browser] = self.test_browser_capabilities()browser)
            
        # Test acceleration for each browser and model combination
        for browser in self.browsers:
            if not results[]]],,,"browsers"][]]],,,browser].get()"webnn_available", False):
                print()f"Skipping acceleration tests for {}}}}}}}}}}}browser} as WebNN is not available")
            continue
                
            if browser not in results[]]],,,"acceleration"]:
                results[]]],,,"acceleration"][]]],,,browser] = {}}}}}}}}}}}}
                
            for model in self.models:
                results[]]],,,"acceleration"][]]],,,browser][]]],,,model] = {}}}}}}}}}}}}
                
                for batch_size in self.batch_sizes:
                    results[]]],,,"acceleration"][]]],,,browser][]]],,,model][]]],,,str()batch_size)] = \
                    self.test_hardware_acceleration()browser, model, batch_size)
        
        # Test fallback behavior for each browser
        for browser in self.browsers:
            results[]]],,,"fallbacks"][]]],,,browser] = self.test_fallback_behavior()browser)
        
        # Save results to file
            output_file = os.path.join()self.output_dir, f"webnn_cross_browser_{}}}}}}}}}}}int()time.time()))}.json")
        with open()output_file, 'w') as f:
            json.dump()results, f, indent=2)
            print()f"Results saved to {}}}}}}}}}}}output_file}")
        
        # Store results in database if available::::
        if HAS_DB_API and self.db_path:
            try:
                store_webnn_test_result()results, self.db_path)
                print()f"Results stored in database at {}}}}}}}}}}}self.db_path}")
            except Exception as e:
                print()f"Error storing results in database: {}}}}}}}}}}}e}")
        
                self.results = results
                return results
        
    def generate_report()self, output_format: str = "markdown") -> str:
        """Generate a report from test results.
        
        Args:
            output_format: Format for the report. Supports "markdown" or "html".
            
        Returns:
            Report string in the specified format.
            """
        if not self.results:
            return "No test results available. Run tests first."
            
        if output_format == "markdown":
            return self._generate_markdown_report())
        elif output_format == "html":
            return self._generate_html_report())
        else:
            return f"Unsupported output format: {}}}}}}}}}}}output_format}"
    
    def _generate_markdown_report()self) -> str:
        """Generate a markdown report from test results.
        
        Returns:
            Markdown report string.
            """
            report = "# WebNN Cross-Browser Verification Report\n\n"
        
        # System information
            report += "## System Information\n\n"
            report += f"- Platform: {}}}}}}}}}}}self.results[]]],,,'system'][]]],,,'platform']}\n"
            report += f"- Platform Version: {}}}}}}}}}}}self.results[]]],,,'system'][]]],,,'platform_version']}\n"
            report += f"- Processor: {}}}}}}}}}}}self.results[]]],,,'system'][]]],,,'processor']}\n\n"
        
        # Browser capabilities
            report += "## Browser WebNN Capabilities\n\n"
            report += "| Browser | WebNN Available | WebGPU Available | Hardware Acceleration | Device |\n"
            report += "|---------|----------------|------------------|----------------------|--------|\n"
        
        for browser, capabilities in self.results[]]],,,"browsers"].items()):
            webnn = "✅" if capabilities.get()"webnn_available", False) else "❌"
            webgpu = "✅" if capabilities.get()"webgpu_available", False) else "❌"
            hw_accel = "✅" if capabilities.get()"hardware_acceleration", False) else "❌"
            device = capabilities.get()"device", "N/A")
            
            report += f"| {}}}}}}}}}}}browser} | {}}}}}}}}}}}webnn} | {}}}}}}}}}}}webgpu} | {}}}}}}}}}}}hw_accel} | {}}}}}}}}}}}device} |\n"
        
            report += "\n"
        
        # Acceleration performance
            report += "## Hardware Acceleration Performance\n\n"
        :
        for browser, models in self.results[]]],,,"acceleration"].items()):
            report += f"### {}}}}}}}}}}}browser.title())}\n\n"
            
            for model, batch_results in models.items()):
                report += f"#### Model: {}}}}}}}}}}}model}\n\n"
                report += "| Batch Size | CPU Time ()ms) | WebNN Time ()ms) | Speedup | Simulated |\n"
                report += "|------------|--------------|----------------|---------|----------|\n"
                
                for batch_size, results in batch_results.items()):
                    cpu_time = results.get()"cpu_time", "N/A")
                    webnn_time = results.get()"webnn_time", "N/A")
                    speedup = results.get()"speedup", "N/A")
                    simulated = "Yes" if results.get()"simulated", False) else "No"
                    :
                    if results.get()"error"):
                        report += f"| {}}}}}}}}}}}batch_size} | Error: {}}}}}}}}}}}results[]]],,,'error']} | - | - | - |\n"
                    else:
                        report += f"| {}}}}}}}}}}}batch_size} | {}}}}}}}}}}}cpu_time} | {}}}}}}}}}}}webnn_time} | {}}}}}}}}}}}speedup}x | {}}}}}}}}}}}simulated} |\n"
                
                        report += "\n"
        
        # Fallback behavior
                        report += "## Fallback Behavior\n\n"
                        report += "| Browser | Graceful Fallback | Error Handling | Notes |\n"
                        report += "|---------|-------------------|----------------|-------|\n"
        
        for browser, fallback in self.results[]]],,,"fallbacks"].items()):
            graceful = "✅" if fallback.get()"graceful_fallback", False) else "❌"
            error_handling = "✅" if fallback.get()"error_handling", False) else "❌"
            notes = fallback.get()"fallback_performance", "N/A")
            :
            if fallback.get()"error"):
                notes = f"Error: {}}}}}}}}}}}fallback[]]],,,'error']}"
                
                report += f"| {}}}}}}}}}}}browser} | {}}}}}}}}}}}graceful} | {}}}}}}}}}}}error_handling} | {}}}}}}}}}}}notes} |\n"
        
                return report
    
    def _generate_html_report()self) -> str:
        """Generate an HTML report from test results.
        
        Returns:
            HTML report string.
            """
        # Basic HTML report - this could be enhanced with charts and styling
            html = """<!DOCTYPE html>
            <html>
            <head>
            <title>WebNN Cross-Browser Verification Report</title>
            <style>
            body {}}}}}}}}}}} font-family: Arial, sans-serif; margin: 20px; }
            h1 {}}}}}}}}}}} color: #333; }
            h2 {}}}}}}}}}}} color: #444; margin-top: 20px; }
            h3 {}}}}}}}}}}} color: #555; }
            table {}}}}}}}}}}} border-collapse: collapse; width: 100%; margin-bottom: 20px; }
            th, td {}}}}}}}}}}} border: 1px solid #ddd; padding: 8px; text-align: left; }
            th {}}}}}}}}}}} background-color: #f2f2f2; }
            tr:nth-child()even) {}}}}}}}}}}} background-color: #f9f9f9; }
            .success {}}}}}}}}}}} color: green; }
            .failure {}}}}}}}}}}} color: red; }
            </style>
            </head>
            <body>
            <h1>WebNN Cross-Browser Verification Report</h1>
            """
        
        # System information
            html += "<h2>System Information</h2>"
            html += "<ul>"
            html += f"<li><strong>Platform:</strong> {}}}}}}}}}}}self.results[]]],,,'system'][]]],,,'platform']}</li>"
            html += f"<li><strong>Platform Version:</strong> {}}}}}}}}}}}self.results[]]],,,'system'][]]],,,'platform_version']}</li>"
            html += f"<li><strong>Processor:</strong> {}}}}}}}}}}}self.results[]]],,,'system'][]]],,,'processor']}</li>"
            html += "</ul>"
        
        # Browser capabilities
            html += "<h2>Browser WebNN Capabilities</h2>"
            html += "<table>"
            html += "<tr><th>Browser</th><th>WebNN Available</th><th>WebGPU Available</th><th>Hardware Acceleration</th><th>Device</th></tr>"
        
        for browser, capabilities in self.results[]]],,,"browsers"].items()):
            webnn = '<span class="success">✓</span>' if capabilities.get()"webnn_available", False) else '<span class="failure">✗</span>'
            webgpu = '<span class="success">✓</span>' if capabilities.get()"webgpu_available", False) else '<span class="failure">✗</span>'
            hw_accel = '<span class="success">✓</span>' if capabilities.get()"hardware_acceleration", False) else '<span class="failure">✗</span>'
            device = capabilities.get()"device", "N/A")
            
            html += f"<tr><td>{}}}}}}}}}}}browser}</td><td>{}}}}}}}}}}}webnn}</td><td>{}}}}}}}}}}}webgpu}</td><td>{}}}}}}}}}}}hw_accel}</td><td>{}}}}}}}}}}}device}</td></tr>"
        
            html += "</table>"
        
        # Acceleration performance
            html += "<h2>Hardware Acceleration Performance</h2>"
        :
        for browser, models in self.results[]]],,,"acceleration"].items()):
            html += f"<h3>{}}}}}}}}}}}browser.title())}</h3>"
            
            for model, batch_results in models.items()):
                html += f"<h4>Model: {}}}}}}}}}}}model}</h4>"
                html += "<table>"
                html += "<tr><th>Batch Size</th><th>CPU Time ()ms)</th><th>WebNN Time ()ms)</th><th>Speedup</th><th>Simulated</th></tr>"
                
                for batch_size, results in batch_results.items()):
                    if results.get()"error"):
                        html += f"<tr><td>{}}}}}}}}}}}batch_size}</td><td colspan='4'>Error: {}}}}}}}}}}}results[]]],,,'error']}</td></tr>"
                    else:
                        cpu_time = results.get()"cpu_time", "N/A")
                        webnn_time = results.get()"webnn_time", "N/A")
                        speedup = results.get()"speedup", "N/A")
                        simulated = "Yes" if results.get()"simulated", False) else "No"
                        :    sim_class = "failure" if results.get()"simulated", False) else "success"
                        
                        html += f"<tr><td>{}}}}}}}}}}}batch_size}</td><td>{}}}}}}}}}}}cpu_time}</td><td>{}}}}}}}}}}}webnn_time}</td><td>{}}}}}}}}}}}speedup}x</td><td class='{}}}}}}}}}}}sim_class}'>{}}}}}}}}}}}simulated}</td></tr>"
                
                        html += "</table>"
        
        # Fallback behavior
                        html += "<h2>Fallback Behavior</h2>"
                        html += "<table>"
                        html += "<tr><th>Browser</th><th>Graceful Fallback</th><th>Error Handling</th><th>Notes</th></tr>"
        
        for browser, fallback in self.results[]]],,,"fallbacks"].items()):
            graceful = '<span class="success">✓</span>' if fallback.get()"graceful_fallback", False) else '<span class="failure">✗</span>'
            error_handling = '<span class="success">✓</span>' if fallback.get()"error_handling", False) else '<span class="failure">✗</span>'
            notes = fallback.get()"fallback_performance", "N/A")
            :
            if fallback.get()"error"):
                notes = f"Error: {}}}}}}}}}}}fallback[]]],,,'error']}"
                
                html += f"<tr><td>{}}}}}}}}}}}browser}</td><td>{}}}}}}}}}}}graceful}</td><td>{}}}}}}}}}}}error_handling}</td><td>{}}}}}}}}}}}notes}</td></tr>"
        
                html += "</table>"
                html += "</body></html>"
        
                return html

def parse_args()):
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()description="Cross-browser WebNN verification test suite.")
    parser.add_argument()"--browser", type=str, help="Browser to test ()chrome, edge, safari, firefox)")
    parser.add_argument()"--model", type=str, help="Model to test")
    parser.add_argument()"--models", type=str, nargs='+', help="List of models to test")
    parser.add_argument()"--all-browsers", action="store_true", help="Test all supported browsers")
    parser.add_argument()"--batch-sizes", type=int, nargs='+', help="List of batch sizes to test")
    parser.add_argument()"--timeout", type=int, default=DEFAULT_TIMEOUT, help="Timeout in seconds for each test")
    parser.add_argument()"--output-dir", type=str, default="./webnn_test_results", help="Directory to store test results")
    parser.add_argument()"--db-path", type=str, help="Path to benchmark database")
    parser.add_argument()"--report", type=str, choices=[]]],,,"markdown", "html"], help="Generate report in specified format")
    parser.add_argument()"--report-output", type=str, help="Path to save the report")
    
                return parser.parse_args())

def main()):
    """Main function."""
    args = parse_args())
    
    # Determine browsers to test
    if args.all_browsers:
        browsers = SUPPORTED_BROWSERS
    elif args.browser:
        browsers = []]],,,args.browser]
    else:
        browsers = []]],,,"edge"],  # Default to Edge as it has the best WebNN support
    
    # Determine models to test
    if args.models and args.models[]]],,,0] == "all":
        models = SUPPORTED_MODELS
    elif args.models:
        models = args.models
    elif args.model:
        models = []]],,,args.model]
    else:
        models = []]],,,"prajjwal1/bert-tiny"],  # Default model
    
    # Create and run the test suite
        suite = WebNNVerificationSuite()
        browsers=browsers,
        models=models,
        batch_sizes=args.batch_sizes,
        timeout=args.timeout,
        output_dir=args.output_dir,
        db_path=args.db_path
        )
    
        suite.run_tests())
    
    # Generate report if requested:
    if args.report:
        report = suite.generate_report()args.report)
        
        if args.report_output:
            with open()args.report_output, 'w') as f:
                f.write()report)
                print()f"Report saved to {}}}}}}}}}}}args.report_output}")
        else:
            print()report)

if __name__ == "__main__":
    main())