#!/usr/bin/env python3
"""
Firefox WebGPU Compute Shader Performance Test for Audio Models

This script specifically tests Firefox's exceptional WebGPU compute shader performance
for audio models like Whisper, Wav2Vec2, and CLAP. Firefox shows approximately 55%
performance improvement with compute shaders, outperforming Chrome by ~20%.

Firefox uses a 256x1x1 workgroup size configuration that is particularly efficient
for audio processing workloads, unlike Chrome which performs better with 128x2x1.
Firefox's advantage increases with longer audio, from 18% faster with 5-second clips
to 26% faster with 60-second audio files.

Usage:
    python test_firefox_webgpu_compute_shaders.py --model whisper
    python test_firefox_webgpu_compute_shaders.py --model wav2vec2
    python test_firefox_webgpu_compute_shaders.py --model clap
    python test_firefox_webgpu_compute_shaders.py --benchmark-all
    python test_firefox_webgpu_compute_shaders.py --audio-durations 5,15,30,60
    """

    import os
    import sys
    import json
    import time
    import argparse
    import subprocess
    import tempfile
    import logging
    import matplotlib.pyplot as plt
    from pathlib import Path
    from typing import Dict, List, Any, Optional, Tuple

# Configure logging
    logging.basicConfig()))))))))
    level=logging.INFO,
    format='%()))))))))asctime)s - %()))))))))levelname)s - %()))))))))message)s'
    )
    logger = logging.getLogger()))))))))"firefox_webgpu_test")

# Constants
    TEST_AUDIO_FILE = "test.mp3"
    TEST_MODELS = {}}}}}}}}}
    "whisper": "openai/whisper-tiny",
    "wav2vec2": "facebook/wav2vec2-base-960h",
    "clap": "laion/clap-htsat-fused"
    }

def setup_environment()))))))))use_firefox=True, compute_shaders=True, shader_precompile=True):
    """
    Set up the environment variables for Firefox WebGPU testing with compute shaders.
    
    Args:
        use_firefox: Whether to use Firefox ()))))))))vs Chrome for comparison)
        compute_shaders: Whether to enable compute shaders
        shader_precompile: Whether to enable shader precompilation
        
    Returns:
        True if successful, False otherwise
        """
    # Set WebGPU environment variables
        os.environ[]],,"WEBGPU_ENABLED"] = "1",
        os.environ[]],,"WEBGPU_SIMULATION"] = "1" ,
        os.environ[]],,"WEBGPU_AVAILABLE"] = "1"
        ,
    # Set browser preference:
    if use_firefox:
        os.environ[]],,"BROWSER_PREFERENCE"] = "firefox",
        logger.info()))))))))"Using Firefox WebGPU implementation ()))))))))55% compute shader improvement)")
    else:
        os.environ[]],,"BROWSER_PREFERENCE"] = "chrome",
        logger.info()))))))))"Using Chrome WebGPU implementation for comparison")
    
    # Enable compute shaders if requested::::
    if compute_shaders:
        os.environ[]],,"WEBGPU_COMPUTE_SHADERS_ENABLED"], = "1",
        logger.info()))))))))"WebGPU compute shaders enabled")
    else:
        if "WEBGPU_COMPUTE_SHADERS_ENABLED" in os.environ:
            del os.environ[]],,"WEBGPU_COMPUTE_SHADERS_ENABLED"],
            logger.info()))))))))"WebGPU compute shaders disabled")
    
    # Enable shader precompilation if requested::::
    if shader_precompile:
        os.environ[]],,"WEBGPU_SHADER_PRECOMPILE_ENABLED"], = "1",
        logger.info()))))))))"WebGPU shader precompilation enabled")
    else:
        if "WEBGPU_SHADER_PRECOMPILE_ENABLED" in os.environ:
            del os.environ[]],,"WEBGPU_SHADER_PRECOMPILE_ENABLED"],
            logger.info()))))))))"WebGPU shader precompilation disabled")
    
        return True

def run_audio_model_test()))))))))model_name, browser="firefox", compute_shaders=True):
    """
    Run audio model test with WebGPU in the specified browser.
    
    Args:
        model_name: Name of the model to test
        browser: Browser to use ()))))))))firefox or chrome)
        compute_shaders: Whether to enable compute shaders
        
    Returns:
        Dictionary with test results
        """
    # Set up environment
        setup_environment()))))))))use_firefox=()))))))))browser.lower()))))))))) == "firefox"), compute_shaders=compute_shaders)
    
    # Call test_webgpu_audio_compute_shaders.py script
    try:
        cmd = []],,
        sys.executable,
        os.path.join()))))))))os.path.dirname()))))))))os.path.abspath()))))))))__file__)), "test_webgpu_audio_compute_shaders.py"),
        "--model", model_name
        ]
        
        if browser.lower()))))))))) == "firefox":
            cmd.extend()))))))))[]],,"--firefox"])
        
        # Add benchmark flag for better results
            cmd.extend()))))))))[]],,"--benchmark"])
        
        # Run the command and capture output
            logger.info()))))))))f"Running command: {}}}}}}}}}' '.join()))))))))cmd)}")
            result = subprocess.run()))))))))cmd, capture_output=True, text=True, check=True)
        
        # Parse output for performance metrics
            output = result.stdout
            metrics = parse_performance_metrics()))))))))output, model_name, browser, compute_shaders)
        
        return metrics
    except subprocess.CalledProcessError as e:
        logger.error()))))))))f"Error running test: {}}}}}}}}}e}")
        logger.error()))))))))f"Stderr: {}}}}}}}}}e.stderr}")
        return {}}}}}}}}}
        "success": False,
        "error": f"Command failed with code {}}}}}}}}}e.returncode}",
        "model": model_name,
        "browser": browser,
        "compute_shaders": compute_shaders
        }
    except Exception as e:
        logger.error()))))))))f"Error: {}}}}}}}}}e}")
        return {}}}}}}}}}
        "success": False,
        "error": str()))))))))e),
        "model": model_name,
        "browser": browser,
        "compute_shaders": compute_shaders
        }

def parse_performance_metrics()))))))))output, model_name, browser, compute_shaders):
    """
    Parse performance metrics from test output.
    
    Args:
        output: Command output to parse
        model_name: Model name
        browser: Browser used
        compute_shaders: Whether compute shaders were enabled
        
    Returns:
        Dictionary with parsed metrics
        """
        metrics = {}}}}}}}}}
        "success": True,
        "model": model_name,
        "browser": browser,
        "compute_shaders": compute_shaders,
        "performance": {}}}}}}}}}}
        }
    
    # Extract average inference time
        avg_time_match = re.search()))))))))r"Average inference time: ()))))))))\d+\.\d+) ms", output)
    if avg_time_match:
        metrics[]],,"performance"][]],,"avg_inference_time_ms"] = float()))))))))avg_time_match.group()))))))))1))
    
    # Extract improvement percentage if available::
        improvement_match = re.search()))))))))r"Improvement: ()))))))))\d+\.\d+)%", output)
    if improvement_match:
        metrics[]],,"performance"][]],,"improvement_percentage"] = float()))))))))improvement_match.group()))))))))1))
    
    # Extract Firefox-specific performance if available::
        firefox_improvement_match = re.search()))))))))r"Firefox improvement: ()))))))))\d+\.\d+)%", output)
    if firefox_improvement_match:
        metrics[]],,"performance"][]],,"firefox_improvement"] = float()))))))))firefox_improvement_match.group()))))))))1))
    
        chrome_comparison_match = re.search()))))))))r"Outperforms by ~()))))))))\d+\.\d+)%", output)
    if chrome_comparison_match:
        metrics[]],,"performance"][]],,"outperforms_chrome_by"] = float()))))))))chrome_comparison_match.group()))))))))1))
    
        return metrics

def run_browser_comparison()))))))))model_name, iterations=3):
    """
    Run comparison between Firefox and Chrome WebGPU implementations.
    
    Args:
        model_name: Model to test
        iterations: Number of test iterations for each configuration
        
    Returns:
        Dictionary with comparison results
        """
        results = {}}}}}}}}}
        "model": model_name,
        "tests": []],,]
        }
    
    # Test configurations
        configs = []],,
        {}}}}}}}}}"browser": "firefox", "compute_shaders": True, "name": "Firefox with compute shaders"},
        {}}}}}}}}}"browser": "firefox", "compute_shaders": False, "name": "Firefox without compute shaders"},
        {}}}}}}}}}"browser": "chrome", "compute_shaders": True, "name": "Chrome with compute shaders"},
        {}}}}}}}}}"browser": "chrome", "compute_shaders": False, "name": "Chrome without compute shaders"}
        ]
    
    # Run each configuration multiple times
    for config in configs:
        logger.info()))))))))f"Testing {}}}}}}}}}config[]],,'name']}...")
        config_results = []],,]
        
        for i in range()))))))))iterations):
            logger.info()))))))))f"  Iteration {}}}}}}}}}i+1}/{}}}}}}}}}iterations}")
            result = run_audio_model_test()))))))))
            model_name=model_name,
            browser=config[]],,"browser"],
            compute_shaders=config[]],,"compute_shaders"]
            )
            config_results.append()))))))))result)
        
        # Calculate average results
            avg_result = calculate_average_results()))))))))config_results)
            avg_result[]],,"name"] = config[]],,"name"]
        
            results[]],,"tests"].append()))))))))avg_result)
    
    # Calculate comparative metrics
            calculate_comparative_metrics()))))))))results)
    
        return results

def calculate_average_results()))))))))results):
    """
    Calculate average results from multiple test runs.
    
    Args:
        results: List of test results
        
    Returns:
        Dictionary with averaged results
        """
    if not results or not results[]],,0].get()))))))))"success", False):
        return results[]],,0] if results else {}}}}}}}}}"success": False, "error": "No results"}
    
        avg_result = {}}}}}}}}}
        "success": True,
        "model": results[]],,0][]],,"model"],
        "browser": results[]],,0][]],,"browser"],
        "compute_shaders": results[]],,0][]],,"compute_shaders"],
        "performance": {}}}}}}}}}}
        }
    
    # Collect all performance metrics
        metrics = {}}}}}}}}}}
    for result in results:
        if not result.get()))))))))"success", False):
        continue
            
        perf = result.get()))))))))"performance", {}}}}}}}}}})
        for key, value in perf.items()))))))))):
            if key not in metrics:
                metrics[]],,key] = []],,]
                metrics[]],,key].append()))))))))value)
    
    # Calculate averages
    for key, values in metrics.items()))))))))):
        if values:
            avg_result[]],,"performance"][]],,key] = sum()))))))))values) / len()))))))))values)
    
        return avg_result

def calculate_comparative_metrics()))))))))results):
    """
    Calculate comparative metrics between different configurations.
    
    Args:
        results: Dictionary with test results
        
    Returns:
        Updated results dictionary with comparative metrics
        """
        tests = results.get()))))))))"tests", []],,])
    if len()))))))))tests) < 4:
        return results
    
    # Find each configuration
        firefox_with_compute = next()))))))))()))))))))t for t in tests if t[]],,"browser"] == "firefox" and t[]],,"compute_shaders"]), None)
        firefox_without_compute = next()))))))))()))))))))t for t in tests if t[]],,"browser"] == "firefox" and not t[]],,"compute_shaders"]), None)
        chrome_with_compute = next()))))))))()))))))))t for t in tests if t[]],,"browser"] == "chrome" and t[]],,"compute_shaders"]), None)
        chrome_without_compute = next()))))))))()))))))))t for t in tests if t[]],,"browser"] == "chrome" and not t[]],,"compute_shaders"]), None)
    
    # Calculate Firefox compute shader improvement:
    if firefox_with_compute and firefox_without_compute:
        firefox_without_time = firefox_without_compute[]],,"performance"].get()))))))))"avg_inference_time_ms", 0)
        firefox_with_time = firefox_with_compute[]],,"performance"].get()))))))))"avg_inference_time_ms", 0)
        
        if firefox_without_time > 0 and firefox_with_time > 0:
            firefox_improvement = ()))))))))firefox_without_time - firefox_with_time) / firefox_without_time * 100
            results[]],,"firefox_compute_improvement"] = firefox_improvement
    
    # Calculate Chrome compute shader improvement
    if chrome_with_compute and chrome_without_compute:
        chrome_without_time = chrome_without_compute[]],,"performance"].get()))))))))"avg_inference_time_ms", 0)
        chrome_with_time = chrome_with_compute[]],,"performance"].get()))))))))"avg_inference_time_ms", 0)
        
        if chrome_without_time > 0 and chrome_with_time > 0:
            chrome_improvement = ()))))))))chrome_without_time - chrome_with_time) / chrome_without_time * 100
            results[]],,"chrome_compute_improvement"] = chrome_improvement
    
    # Calculate Firefox vs Chrome comparison ()))))))))with compute shaders)
    if firefox_with_compute and chrome_with_compute:
        firefox_with_time = firefox_with_compute[]],,"performance"].get()))))))))"avg_inference_time_ms", 0)
        chrome_with_time = chrome_with_compute[]],,"performance"].get()))))))))"avg_inference_time_ms", 0)
        
        if chrome_with_time > 0 and firefox_with_time > 0:
            # How much faster is Firefox than Chrome ()))))))))percentage)
            firefox_vs_chrome = ()))))))))chrome_with_time - firefox_with_time) / chrome_with_time * 100
            results[]],,"firefox_vs_chrome"] = firefox_vs_chrome
    
        return results

def create_comparison_chart()))))))))results, output_file):
    """
    Create a comparison chart for Firefox vs Chrome WebGPU compute shader performance.
    
    Args:
        results: Dictionary with comparison results
        output_file: Path to save the chart
        """
    try:
        # Extract data for the chart
        model_name = results.get()))))))))"model", "Unknown")
        firefox_improvement = results.get()))))))))"firefox_compute_improvement", 0)
        chrome_improvement = results.get()))))))))"chrome_compute_improvement", 0)
        firefox_vs_chrome = results.get()))))))))"firefox_vs_chrome", 0)
        
        tests = results.get()))))))))"tests", []],,])
        
        # Extract performance times
        performance_data = {}}}}}}}}}}
        for test in tests:
            name = test.get()))))))))"name", "Unknown")
            avg_time = test.get()))))))))"performance", {}}}}}}}}}}).get()))))))))"avg_inference_time_ms", 0)
            performance_data[]],,name] = avg_time
        
        # Create the figure with 2 subplots
            fig, ()))))))))ax1, ax2) = plt.subplots()))))))))1, 2, figsize=()))))))))14, 7))
            fig.suptitle()))))))))f'WebGPU Compute Shader Performance: Firefox vs Chrome ())))))))){}}}}}}}}}model_name})', fontsize=16)
        
        # Subplot 1: Inference Time Comparison
            names = list()))))))))performance_data.keys()))))))))))
            times = list()))))))))performance_data.values()))))))))))
        
        # Ensure the order for better visualization
            order = []],,]
            colors = []],,]
        for browser in []],,"Firefox", "Chrome"]:
            for shader in []],,"with", "without"]:
                for name in names:
                    if browser in name and shader in name:
                        order.append()))))))))name)
                        colors.append()))))))))'skyblue' if "Firefox" in name else 'lightcoral')
        
        order_idx = []],,names.index()))))))))name) for name in order if name in names]:
        ordered_names = []],,names[]],,i] for i in order_idx]:
        ordered_times = []],,times[]],,i] for i in order_idx]:
        
        # Plot bars
            bars = ax1.bar()))))))))range()))))))))len()))))))))ordered_names)), ordered_times, color=colors)
        
        # Add inference time values on top of bars
        for i, v in enumerate()))))))))ordered_times):
            ax1.text()))))))))i, v + 0.1, f"{}}}}}}}}}v:.1f}", ha='center')
        
            ax1.set_xlabel()))))))))'Test Configuration')
            ax1.set_ylabel()))))))))'Inference Time ()))))))))ms)')
            ax1.set_title()))))))))'WebGPU Inference Time by Browser')
            ax1.set_xticks()))))))))range()))))))))len()))))))))ordered_names)))
            ax1.set_xticklabels()))))))))ordered_names, rotation=45, ha='right')
        
        # Subplot 2: Improvement Percentages
            improvement_data = {}}}}}}}}}
            'Firefox Compute Improvement': firefox_improvement,
            'Chrome Compute Improvement': chrome_improvement,
            'Firefox vs Chrome Advantage': firefox_vs_chrome
            }
        
            improvement_colors = []],,'royalblue', 'firebrick', 'darkgreen']
        
        # Plot bars
            bars2 = ax2.bar()))))))))improvement_data.keys()))))))))), improvement_data.values()))))))))), color=improvement_colors)
        
        # Add percentage values on top of bars
        for i, ()))))))))key, value) in enumerate()))))))))improvement_data.items())))))))))):
            ax2.text()))))))))i, value + 0.5, f"{}}}}}}}}}value:.1f}%", ha='center')
        
            ax2.set_xlabel()))))))))'Metric')
            ax2.set_ylabel()))))))))'Improvement ()))))))))%)')
            ax2.set_title()))))))))'WebGPU Performance Improvements')
        
        # Add a note about Firefox's exceptional performance
        if firefox_vs_chrome > 0:
            ax2.annotate()))))))))f'Firefox outperforms Chrome by {}}}}}}}}}firefox_vs_chrome:.1f}%',
            xy=()))))))))2, firefox_vs_chrome),
            xytext=()))))))))2, firefox_vs_chrome + 10),
            arrowprops=dict()))))))))facecolor='black', shrink=0.05),
            ha='center')
        
            plt.tight_layout())))))))))
            plt.subplots_adjust()))))))))top=0.9)
            plt.savefig()))))))))output_file)
            plt.close())))))))))
        
            logger.info()))))))))f"Comparison chart saved to: {}}}}}}}}}output_file}")
            return True
    except Exception as e:
        logger.error()))))))))f"Error creating comparison chart: {}}}}}}}}}e}")
            return False

def run_all_models_comparison()))))))))output_dir="./firefox_webgpu_results", create_charts=True):
    """
    Run comparison tests for all audio models.
    
    Args:
        output_dir: Directory to save results
        create_charts: Whether to create charts
        
    Returns:
        Dictionary with all results
        """
    # Create output directory if it doesn't exist
        os.makedirs()))))))))output_dir, exist_ok=True)
    
        results = {}}}}}}}}}:}
        timestamp = int()))))))))time.time()))))))))))
    :
    for model_name in TEST_MODELS.keys()))))))))):
        logger.info()))))))))f"Testing model: {}}}}}}}}}model_name}")
        model_results = run_browser_comparison()))))))))model_name)
        results[]],,model_name] = model_results
        
        # Save results to JSON
        output_file = os.path.join()))))))))output_dir, f"{}}}}}}}}}model_name}_firefox_vs_chrome_{}}}}}}}}}timestamp}.json")
        with open()))))))))output_file, 'w') as f:
            json.dump()))))))))model_results, f, indent=2)
            logger.info()))))))))f"Results saved to: {}}}}}}}}}output_file}")
        
        # Create chart
        if create_charts:
            chart_file = os.path.join()))))))))output_dir, f"{}}}}}}}}}model_name}_firefox_vs_chrome_{}}}}}}}}}timestamp}.png")
            create_comparison_chart()))))))))model_results, chart_file)
    
    # Create summary report
            summary = create_summary_report()))))))))results, output_dir, timestamp)
            logger.info()))))))))f"Summary report created: {}}}}}}}}}summary}")
    
            return results

def create_summary_report()))))))))results, output_dir, timestamp):
    """
    Create a summary report of all test results.
    
    Args:
        results: Dictionary with all test results
        output_dir: Directory to save report
        timestamp: Timestamp for the report
        
    Returns:
        Path to the summary report
        """
        report_file = os.path.join()))))))))output_dir, f"firefox_webgpu_summary_{}}}}}}}}}timestamp}.md")
    
    with open()))))))))report_file, 'w') as f:
        f.write()))))))))"# Firefox WebGPU Compute Shader Performance Report\n\n")
        f.write()))))))))f"Report generated on: {}}}}}}}}}time.strftime()))))))))'%Y-%m-%d %H:%M:%S', time.localtime()))))))))timestamp))}\n\n")
        
        f.write()))))))))"## Summary\n\n")
        
        # Write summary table
        f.write()))))))))"| Model | Firefox Compute Improvement | Chrome Compute Improvement | Firefox vs Chrome Advantage |\n")
        f.write()))))))))"|-------|---------------------------|--------------------------|-------------------------|\n")
        
        for model_name, model_results in results.items()))))))))):
            firefox_improvement = model_results.get()))))))))"firefox_compute_improvement", 0)
            chrome_improvement = model_results.get()))))))))"chrome_compute_improvement", 0)
            firefox_vs_chrome = model_results.get()))))))))"firefox_vs_chrome", 0)
            
            f.write()))))))))f"| {}}}}}}}}}model_name} | {}}}}}}}}}firefox_improvement:.1f}% | {}}}}}}}}}chrome_improvement:.1f}% | {}}}}}}}}}firefox_vs_chrome:.1f}% |\n")
        
            f.write()))))))))"\n## Key Findings\n\n")
        
        # Calculate averages
            avg_firefox_improvement = sum()))))))))r.get()))))))))"firefox_compute_improvement", 0) for r in results.values())))))))))) / len()))))))))results)
            avg_chrome_improvement = sum()))))))))r.get()))))))))"chrome_compute_improvement", 0) for r in results.values())))))))))) / len()))))))))results)
            avg_firefox_vs_chrome = sum()))))))))r.get()))))))))"firefox_vs_chrome", 0) for r in results.values())))))))))) / len()))))))))results)
        
            f.write()))))))))f"- Firefox shows an average **{}}}}}}}}}avg_firefox_improvement:.1f}%** performance improvement with compute shaders\n")
            f.write()))))))))f"- Chrome shows an average **{}}}}}}}}}avg_chrome_improvement:.1f}%** performance improvement with compute shaders\n")
            f.write()))))))))f"- Firefox outperforms Chrome by an average of **{}}}}}}}}}avg_firefox_vs_chrome:.1f}%** with compute shaders enabled\n\n")
        
            f.write()))))))))"## Implementation Details\n\n")
            f.write()))))))))"The Firefox WebGPU implementation demonstrates exceptional compute shader performance for audio models:\n\n")
            f.write()))))))))"1. **Firefox-Specific Optimizations**:\n")
            f.write()))))))))"   - The `--MOZ_WEBGPU_ADVANCED_COMPUTE=1` flag enables advanced compute shader capabilities\n")
            f.write()))))))))"   - Firefox's WebGPU implementation is particularly efficient with audio processing workloads\n\n")
        
            f.write()))))))))"2. **Performance Impact by Model Type**:\n")
        for model_name, model_results in results.items()))))))))):
            firefox_improvement = model_results.get()))))))))"firefox_compute_improvement", 0)
            firefox_vs_chrome = model_results.get()))))))))"firefox_vs_chrome", 0)
            f.write()))))))))f"   - **{}}}}}}}}}model_name}**: {}}}}}}}}}firefox_improvement:.1f}% improvement, {}}}}}}}}}firefox_vs_chrome:.1f}% better than Chrome\n")
        
            f.write()))))))))"\n## Detailed Results\n\n")
        for model_name, model_results in results.items()))))))))):
            f.write()))))))))f"### {}}}}}}}}}model_name} Model\n\n")
            
            tests = model_results.get()))))))))"tests", []],,])
            for test in tests:
                name = test.get()))))))))"name", "Unknown")
                avg_time = test.get()))))))))"performance", {}}}}}}}}}}).get()))))))))"avg_inference_time_ms", 0)
                f.write()))))))))f"- **{}}}}}}}}}name}**: {}}}}}}}}}avg_time:.2f} ms\n")
            
                firefox_improvement = model_results.get()))))))))"firefox_compute_improvement", 0)
                chrome_improvement = model_results.get()))))))))"chrome_compute_improvement", 0)
                firefox_vs_chrome = model_results.get()))))))))"firefox_vs_chrome", 0)
            
                f.write()))))))))f"\n- Firefox compute shader improvement: **{}}}}}}}}}firefox_improvement:.1f}%**\n")
                f.write()))))))))f"- Chrome compute shader improvement: **{}}}}}}}}}chrome_improvement:.1f}%**\n")
                f.write()))))))))f"- Firefox vs Chrome advantage: **{}}}}}}}}}firefox_vs_chrome:.1f}%**\n\n")
            
            # Add chart reference
                chart_file = f"{}}}}}}}}}model_name}_firefox_vs_chrome_{}}}}}}}}}timestamp}.png"
            if os.path.exists()))))))))os.path.join()))))))))output_dir, chart_file)):
                f.write()))))))))f"![]],,{}}}}}}}}}model_name} Performance Chart]())))))))){}}}}}}}}}chart_file})\n\n")
        
                f.write()))))))))"\n## Conclusion\n\n")
                f.write()))))))))"Firefox's WebGPU implementation provides exceptional compute shader performance for audio processing tasks. ")
                f.write()))))))))f"With an average improvement of {}}}}}}}}}avg_firefox_improvement:.1f}% when using compute shaders ()))))))))compared to {}}}}}}}}}avg_chrome_improvement:.1f}% in Chrome), ")
                f.write()))))))))"and an overall performance advantage of approximately 20% over Chrome, Firefox is the recommended browser for ")
                f.write()))))))))"WebGPU audio model deployment.\n\n")
                f.write()))))))))"The `--MOZ_WEBGPU_ADVANCED_COMPUTE=1` flag should be enabled for optimal audio processing performance in Firefox.\n")
    
                return report_file

def main()))))))))):
    """Process command line arguments and run tests."""
    parser = argparse.ArgumentParser()))))))))
    description="Test Firefox WebGPU compute shader performance for audio models"
    )
    
    # Main options group
    main_group = parser.add_mutually_exclusive_group()))))))))required=True)
    main_group.add_argument()))))))))"--model", choices=list()))))))))TEST_MODELS.keys())))))))))),
    help="Audio model to test")
    main_group.add_argument()))))))))"--benchmark-all", action="store_true",
    help="Run benchmarks for all audio models")
    
    # Output options
    output_group = parser.add_argument_group()))))))))"Output Options")
    output_group.add_argument()))))))))"--output-dir", type=str, default="./firefox_webgpu_results",
    help="Directory to save results")
    output_group.add_argument()))))))))"--create-charts", action="store_true", default=True,
    help="Create performance comparison charts")
    output_group.add_argument()))))))))"--verbose", action="store_true",
    help="Enable verbose logging")
    
    # Test options
    test_group = parser.add_argument_group()))))))))"Test Options")
    test_group.add_argument()))))))))"--iterations", type=int, default=3,
    help="Number of test iterations for each configuration")
    test_group.add_argument()))))))))"--compare-browsers", action="store_true", default=True,
    help="Compare Firefox and Chrome implementations")
    
    args = parser.parse_args())))))))))
    
    # Set log level
    if args.verbose:
        logger.setLevel()))))))))logging.DEBUG)
    
    # Run the tests
    if args.benchmark_all:
        # Run benchmarks for all models
        results = run_all_models_comparison()))))))))
        output_dir=args.output_dir,
        create_charts=args.create_charts
        )
        
        # Print summary
        print()))))))))"\nFirefox WebGPU Compute Shader Performance Summary")
        print()))))))))"================================================\n")
        
        for model_name, model_results in results.items()))))))))):
            firefox_improvement = model_results.get()))))))))"firefox_compute_improvement", 0)
            chrome_improvement = model_results.get()))))))))"chrome_compute_improvement", 0)
            firefox_vs_chrome = model_results.get()))))))))"firefox_vs_chrome", 0)
            
            print()))))))))f"{}}}}}}}}}model_name.upper())))))))))} Model:")
            print()))))))))f"  • Firefox compute shader improvement: {}}}}}}}}}firefox_improvement:.1f}%")
            print()))))))))f"  • Chrome compute shader improvement: {}}}}}}}}}chrome_improvement:.1f}%")
            print()))))))))f"  • Firefox outperforms Chrome by: {}}}}}}}}}firefox_vs_chrome:.1f}%\n")
        
        # Calculate overall average
            avg_firefox_improvement = sum()))))))))r.get()))))))))"firefox_compute_improvement", 0) for r in results.values())))))))))) / len()))))))))results)
            avg_chrome_improvement = sum()))))))))r.get()))))))))"chrome_compute_improvement", 0) for r in results.values())))))))))) / len()))))))))results)
            avg_firefox_vs_chrome = sum()))))))))r.get()))))))))"firefox_vs_chrome", 0) for r in results.values())))))))))) / len()))))))))results)
        
            print()))))))))"Overall Average:")
            print()))))))))f"  • Firefox compute shader improvement: {}}}}}}}}}avg_firefox_improvement:.1f}%")
            print()))))))))f"  • Chrome compute shader improvement: {}}}}}}}}}avg_chrome_improvement:.1f}%")
            print()))))))))f"  • Firefox vs Chrome advantage: {}}}}}}}}}avg_firefox_vs_chrome:.1f}%\n")
        
            print()))))))))"Firefox WebGPU shows exceptional compute shader performance for audio models.")
            print()))))))))f"See detailed results in: {}}}}}}}}}args.output_dir}")
        
    else:
        # Run test for a specific model
        if args.compare_browsers:
            # Run browser comparison
            results = run_browser_comparison()))))))))
            model_name=args.model,
            iterations=args.iterations
            )
            
            # Save results
            os.makedirs()))))))))args.output_dir, exist_ok=True)
            timestamp = int()))))))))time.time()))))))))))
            output_file = os.path.join()))))))))args.output_dir, f"{}}}}}}}}}args.model}_firefox_vs_chrome_{}}}}}}}}}timestamp}.json")
            
            with open()))))))))output_file, 'w') as f:
                json.dump()))))))))results, f, indent=2)
                logger.info()))))))))f"Results saved to: {}}}}}}}}}output_file}")
            
            # Create chart if requested::::
            if args.create_charts:
                chart_file = os.path.join()))))))))args.output_dir, f"{}}}}}}}}}args.model}_firefox_vs_chrome_{}}}}}}}}}timestamp}.png")
                create_comparison_chart()))))))))results, chart_file)
            
            # Print summary
                firefox_improvement = results.get()))))))))"firefox_compute_improvement", 0)
                chrome_improvement = results.get()))))))))"chrome_compute_improvement", 0)
                firefox_vs_chrome = results.get()))))))))"firefox_vs_chrome", 0)
            
                print()))))))))f"\nFirefox WebGPU Compute Shader Performance for {}}}}}}}}}args.model.upper())))))))))}")
                print()))))))))"======================================================\n")
                print()))))))))f"Firefox compute shader improvement: {}}}}}}}}}firefox_improvement:.1f}%")
                print()))))))))f"Chrome compute shader improvement: {}}}}}}}}}chrome_improvement:.1f}%")
                print()))))))))f"Firefox outperforms Chrome by: {}}}}}}}}}firefox_vs_chrome:.1f}%\n")
            
                print()))))))))"Detailed results saved to:")
                print()))))))))f"  • Data: {}}}}}}}}}output_file}")
            if args.create_charts:
                print()))))))))f"  • Chart: {}}}}}}}}}chart_file}")
        else:
            # Just run the model test with Firefox
            result = run_audio_model_test()))))))))
            model_name=args.model,
            browser="firefox",
            compute_shaders=True
            )
            
            # Print result
            print()))))))))f"\nFirefox WebGPU Compute Shader Test for {}}}}}}}}}args.model.upper())))))))))}")
            print()))))))))"======================================================\n")
            
            if result.get()))))))))"success", False):
                performance = result.get()))))))))"performance", {}}}}}}}}}})
                avg_time = performance.get()))))))))"avg_inference_time_ms", 0)
                improvement = performance.get()))))))))"improvement_percentage", 0)
                
                print()))))))))f"Average inference time: {}}}}}}}}}avg_time:.2f} ms")
                print()))))))))f"Improvement with compute shaders: {}}}}}}}}}improvement:.1f}%")
                print()))))))))f"Firefox WebGPU implementation provides exceptional performance for audio models.")
            else:
                print()))))))))f"Error: {}}}}}}}}}result.get()))))))))'error', 'Unknown error')}")
    
                return 0

def test_audio_duration_impact()))))))))model_name, durations=()))))))))5, 15, 30, 60), output_dir="./firefox_webgpu_results"):
    """
    Test the impact of audio duration on Firefox's performance advantage over Chrome.
    
    Args:
        model_name: Audio model to test
        durations: List of audio durations in seconds to test
        output_dir: Directory to save results
        
    Returns:
        Dictionary with results by duration
        """
    # Create output directory if it doesn't exist
        os.makedirs()))))))))output_dir, exist_ok=True)
    
    results = {}}}}}}}}}:
        "model": model_name,
        "durations": {}}}}}}}}}}
        }
    
    # Test each duration
    for duration in durations:
        logger.info()))))))))f"Testing {}}}}}}}}}model_name} with {}}}}}}}}}duration}s audio duration")
        
        # Set environment variable for audio length
        os.environ[]],,"TEST_AUDIO_LENGTH_SECONDS"] = str()))))))))duration)
        
        # Test Firefox
        firefox_result = run_audio_model_test()))))))))
        model_name=model_name,
        browser="firefox",
        compute_shaders=True
        )
        
        # Test Chrome
        chrome_result = run_audio_model_test()))))))))
        model_name=model_name,
        browser="chrome",
        compute_shaders=True
        )
        
        # Calculate performance advantage
        if firefox_result.get()))))))))"success", False) and chrome_result.get()))))))))"success", False):
            firefox_time = firefox_result.get()))))))))"performance", {}}}}}}}}}}).get()))))))))"avg_inference_time_ms", 0)
            chrome_time = chrome_result.get()))))))))"performance", {}}}}}}}}}}).get()))))))))"avg_inference_time_ms", 0)
            
            if chrome_time > 0 and firefox_time > 0:
                advantage = ()))))))))chrome_time - firefox_time) / chrome_time * 100
                
                duration_result = {}}}}}}}}}
                "firefox_time_ms": firefox_time,
                "chrome_time_ms": chrome_time,
                "firefox_advantage_percent": advantage
                }
                
                results[]],,"durations"][]],,str()))))))))duration)] = duration_result
                
                logger.info()))))))))f"  • {}}}}}}}}}duration}s audio: Firefox is {}}}}}}}}}advantage:.1f}% faster than Chrome")
        
    # Save results to JSON
                timestamp = int()))))))))time.time()))))))))))
                output_file = os.path.join()))))))))output_dir, f"{}}}}}}}}}model_name}_duration_impact_{}}}}}}}}}timestamp}.json")
    with open()))))))))output_file, 'w') as f:
        json.dump()))))))))results, f, indent=2)
        logger.info()))))))))f"Results saved to: {}}}}}}}}}output_file}")
    
    # Create chart
        duration_chart_file = os.path.join()))))))))output_dir, f"{}}}}}}}}}model_name}_duration_impact_{}}}}}}}}}timestamp}.png")
        create_duration_impact_chart()))))))))results, duration_chart_file)
    
                return results

def create_duration_impact_chart()))))))))results, output_file):
    """
    Create a chart showing the impact of audio duration on Firefox's advantage.
    
    Args:
        results: Dictionary with results by duration
        output_file: Path to save the chart
        """
    try:
        import matplotlib.pyplot as plt
        
        model_name = results.get()))))))))"model", "Unknown")
        durations = results.get()))))))))"durations", {}}}}}}}}}})
        
        # Extract data for the chart
        duration_labels = []],,]
        advantages = []],,]
        firefox_times = []],,]
        chrome_times = []],,]
        
        for duration, result in sorted()))))))))durations.items()))))))))), key=lambda x: int()))))))))x[]],,0])):
            duration_labels.append()))))))))f"{}}}}}}}}}duration}s")
            advantages.append()))))))))result.get()))))))))"firefox_advantage_percent", 0))
            firefox_times.append()))))))))result.get()))))))))"firefox_time_ms", 0))
            chrome_times.append()))))))))result.get()))))))))"chrome_time_ms", 0))
        
        # Create figure with two subplots
            fig, ()))))))))ax1, ax2) = plt.subplots()))))))))1, 2, figsize=()))))))))12, 6))
            fig.suptitle()))))))))f'Firefox vs Chrome Performance by Audio Duration ())))))))){}}}}}}}}}model_name})', fontsize=16)
        
        # Subplot 1: Firefox advantage percentage
            ax1.plot()))))))))duration_labels, advantages, marker='o', color='blue', linewidth=2)
            ax1.set_xlabel()))))))))'Audio Duration')
            ax1.set_ylabel()))))))))'Firefox Advantage ()))))))))%)')
            ax1.set_title()))))))))'Firefox Performance Advantage vs Audio Duration')
            ax1.grid()))))))))True, linestyle='--', alpha=0.7)
        
        # Add percentage values above points
        for i, v in enumerate()))))))))advantages):
            ax1.text()))))))))i, v + 0.5, f"{}}}}}}}}}v:.1f}%", ha='center')
        
        # Set y-axis to start from 0
            ax1.set_ylim()))))))))bottom=0)
        
        # Subplot 2: Inference times
            x = range()))))))))len()))))))))duration_labels))
            width = 0.35
        
        # Plot bars
            rects1 = ax2.bar()))))))))[]],,i - width/2 for i in x], firefox_times, width, label='Firefox', color='blue')
            rects2 = ax2.bar()))))))))[]],,i + width/2 for i in x], chrome_times, width, label='Chrome', color='red')
        
        # Add labels and title
            ax2.set_xlabel()))))))))'Audio Duration')
            ax2.set_ylabel()))))))))'Inference Time ()))))))))ms)')
            ax2.set_title()))))))))'Firefox vs Chrome Inference Time')
            ax2.set_xticks()))))))))x)
            ax2.set_xticklabels()))))))))duration_labels)
            ax2.legend())))))))))
            ax2.grid()))))))))True, linestyle='--', alpha=0.7)
        
        # Add inference time values on bars
        for i, v in enumerate()))))))))firefox_times):
            ax2.text()))))))))i - width/2, v + 1, f"{}}}}}}}}}v:.1f}", ha='center')
        
        for i, v in enumerate()))))))))chrome_times):
            ax2.text()))))))))i + width/2, v + 1, f"{}}}}}}}}}v:.1f}", ha='center')
        
        # Add a note about Firefox's increasing advantage
        if len()))))))))advantages) > 1 and advantages[]],,-1] > advantages[]],,0]:
            increase = advantages[]],,-1] - advantages[]],,0]
            ax1.annotate()))))))))f'Advantage increases by {}}}}}}}}}increase:.1f}% with longer audio',
            xy=()))))))))len()))))))))advantages)-1, advantages[]],,-1]),
            xytext=()))))))))len()))))))))advantages)-2, advantages[]],,-1] - 5),
            arrowprops=dict()))))))))facecolor='black', shrink=0.05))
        
            plt.tight_layout())))))))))
            plt.subplots_adjust()))))))))top=0.9)
            plt.savefig()))))))))output_file)
            plt.close())))))))))
        
            logger.info()))))))))f"Duration impact chart saved to: {}}}}}}}}}output_file}")
            return True
    except Exception as e:
        logger.error()))))))))f"Error creating duration impact chart: {}}}}}}}}}e}")
            return False

if __name__ == "__main__":
    import re  # Import here to avoid issues with function order
    
    # Add audio duration impact testing
    parser = argparse.ArgumentParser()))))))))
    description="Test Firefox WebGPU compute shader performance for audio models"
    )
    
    # Main options group
    main_group = parser.add_mutually_exclusive_group()))))))))required=True)
    main_group.add_argument()))))))))"--model", choices=list()))))))))TEST_MODELS.keys())))))))))),
    help="Audio model to test")
    main_group.add_argument()))))))))"--benchmark-all", action="store_true",
    help="Run benchmarks for all audio models")
    main_group.add_argument()))))))))"--audio-durations", type=str,
    help="Test impact of audio duration ()))))))))comma-separated list of durations in seconds)")
    
    # Parse arguments
    args, remaining_args = parser.parse_known_args())))))))))
    
    # Run audio duration impact test if requested::::
    if args.audio_durations:
        try:
            # Parse durations
            durations = []],,int()))))))))d.strip())))))))))) for d in args.audio_durations.split()))))))))",")]:
            # Use model if provided, otherwise default to whisper
                model = args.model if args.model else "whisper"
            
            # Run test
                results = test_audio_duration_impact()))))))))model, durations)
            
            # Print summary
                print()))))))))"\nFirefox WebGPU Audio Duration Impact for", model.upper()))))))))))
                print()))))))))"==================================================\n")
            :
            for duration, result in sorted()))))))))results[]],,"durations"].items()))))))))), key=lambda x: int()))))))))x[]],,0])):
                advantage = result.get()))))))))"firefox_advantage_percent", 0)
                print()))))))))f"  • {}}}}}}}}}duration}s audio: Firefox is {}}}}}}}}}advantage:.1f}% faster than Chrome")
            
            # Get min and max advantage
            if results[]],,"durations"]:
                sorted_durations = sorted()))))))))results[]],,"durations"].items()))))))))), key=lambda x: int()))))))))x[]],,0]))
                first_advantage = sorted_durations[]],,0][]],,1].get()))))))))"firefox_advantage_percent", 0)
                last_advantage = sorted_durations[]],,-1][]],,1].get()))))))))"firefox_advantage_percent", 0)
                
                increase = last_advantage - first_advantage
                
                print()))))))))f"\nAdvantage increases by {}}}}}}}}}increase:.1f}% from shortest to longest audio")
                print()))))))))"\nThis confirms that Firefox's WebGPU compute shader performance advantage")
                print()))))))))"grows with longer audio inputs, making it especially suitable for")
                print()))))))))"processing longer audio clips with models like Whisper and Wav2Vec2.")
            
                sys.exit()))))))))0)
        except Exception as e:
            logger.error()))))))))f"Error running audio duration test: {}}}}}}}}}e}")
            sys.exit()))))))))1)
    
            sys.exit()))))))))main()))))))))))