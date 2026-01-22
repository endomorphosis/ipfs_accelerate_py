#!/usr/bin/env python3
"""
WebGPU Cross-Browser Comparison Test Suite.

This module compares WebGPU performance across different browsers ()))))))))))))))Chrome, Firefox, Edge, Safari)
with a focus on browser-specific optimizations:

    1. Firefox audio compute shader optimizations for audio models ()))))))))))))))Whisper, Wav2Vec2, CLAP)
    2. Parallel loading optimization for multimodal models ()))))))))))))))CLIP, LLaVA)
    3. Shader precompilation for faster startup across browsers
    4. Combined optimizations impact on different model types

Usage:
    python test_webgpu_browsers_comparison.py --browser firefox --model whisper-tiny
    python test_webgpu_browsers_comparison.py --all-browsers --model whisper-tiny
    python test_webgpu_browsers_comparison.py --all-browsers --all-optimizations --model clip
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
    sys.path.append()))))))))))))))os.path.dirname()))))))))))))))os.path.dirname()))))))))))))))os.path.abspath()))))))))))))))__file__))))

# Import database utilities if available:::
try:
    from duckdb_api.core.benchmark_db_api import store_webgpu_comparison_result
    HAS_DB_API = True
except ImportError:
    HAS_DB_API = False
    print()))))))))))))))"WARNING: benchmark_db_api not found. Results will not be stored in database.")

# Define constants
    SUPPORTED_BROWSERS = []],,"chrome", "firefox", "edge", "safari"],
    AUDIO_MODELS = []],,"whisper-tiny", "wav2vec2-base", "LAION-AI/clap-htsat-fused"],
    MULTIMODAL_MODELS = []],,"openai/clip-vit-base-patch32", "llava-hf/llava-1.5-7b-hf"],
    TEXT_VISION_MODELS = []],,"bert-base-uncased", "google/vit-base-patch16-224"],
    DEFAULT_TIMEOUT = 600  # seconds
    DEFAULT_BATCH_SIZES = []],,1, 4]
    ,
class WebGPUBrowserComparison:
    """WebGPU performance comparison across browsers with optimization focus."""
    
    def __init__()))))))))))))))self, 
    browsers: List[]],,str] = None,
    models: List[]],,str] = None,
    batch_sizes: List[]],,int] = None,
    enable_compute_shaders: bool = False,
    enable_parallel_loading: bool = False,
    enable_shader_precompile: bool = False,
    all_optimizations: bool = False,
    timeout: int = DEFAULT_TIMEOUT,
    output_dir: str = "./webgpu_browser_comparison_results",
    db_path: Optional[]],,str] = None):,
    """Initialize the WebGPU browser comparison suite.
        
        Args:
            browsers: List of browsers to test. Defaults to []],,"chrome", "firefox"],.,
            models: List of models to test. Defaults to []],,"whisper-tiny"],.,
            batch_sizes: List of batch sizes to test. Defaults to []],,1, 4].,
            enable_compute_shaders: Enable compute shader optimizations. Defaults to False.
            enable_parallel_loading: Enable parallel loading optimizations. Defaults to False.
            enable_shader_precompile: Enable shader precompilation. Defaults to False.
            all_optimizations: Enable all optimizations. Defaults to False.
            timeout: Timeout in seconds for each browser test. Defaults to 600.
            output_dir: Directory to store test results. Defaults to "./webgpu_browser_comparison_results".
            db_path: Path to benchmark database. Defaults to None.
            """
            self.browsers = browsers or []],,"chrome", "firefox"],
            self.models = models or []],,"whisper-tiny"],
            self.batch_sizes = batch_sizes or DEFAULT_BATCH_SIZES
            self.timeout = timeout
            self.output_dir = output_dir
            self.db_path = db_path or os.environ.get()))))))))))))))"BENCHMARK_DB_PATH")
        
        # Optimization flags
            self.enable_compute_shaders = enable_compute_shaders or all_optimizations
            self.enable_parallel_loading = enable_parallel_loading or all_optimizations
            self.enable_shader_precompile = enable_shader_precompile or all_optimizations
            self.all_optimizations = all_optimizations
        
        # Create output directory if it doesn't exist
            os.makedirs()))))))))))))))self.output_dir, exist_ok=True)
        
        # Results dictionary
            self.results = {}}}}}}}}}}}}}}}}}}}}}}}}
    :
    def test_browser_webgpu_capabilities()))))))))))))))self, browser: str) -> Dict:
        """Test browser WebGPU capabilities.
        
        Args:
            browser: Browser to test.
            
        Returns:
            Dictionary with browser capability information.
            """
            print()))))))))))))))f"Testing {}}}}}}}}}}}}}}}}}}}}}}}browser} WebGPU capabilities...")
        
        # Construct command to run capability check
            cmd = []],,
            "./run_browser_capability_check.sh",
            f"--browser={}}}}}}}}}}}}}}}}}}}}}}}browser}",
            "--webgpu-details"
            ]
        
        try:
            # Run capability check
            output = subprocess.check_output()))))))))))))))cmd, timeout=self.timeout, stderr=subprocess.STDOUT)
            output_str = output.decode()))))))))))))))'utf-8')
            
            # Parse capability information
            capabilities = {}}}}}}}}}}}}}}}}}}}}}}}
            "browser": browser,
            "webgpu_available": "WebGPU: Available" in output_str,
            "hardware_acceleration": "Hardware Acceleration: Enabled" in output_str,
            "error": None
            }
            
            # Extract additional WebGPU capability information if available:::
            if "GPU Device:" in output_str:
                device_line = []],,line for line in output_str.split()))))))))))))))'\n') if "GPU Device:" in line]
                if device_line:
                    capabilities[]],,"gpu_device"] = device_line[]],,0].split()))))))))))))))"GPU Device:")[]],,1].strip())))))))))))))))
            
            if "Adapter Info:" in output_str:
                adapter_lines = []],,]
                capture = False
                for line in output_str.split()))))))))))))))'\n'):
                    if "Adapter Info:" in line:
                        capture = True
                    continue
                    if capture and line.strip()))))))))))))))) and not line.startswith()))))))))))))))"---"):
                        adapter_lines.append()))))))))))))))line.strip()))))))))))))))))
                    if capture and "---" in line:
                        capture = False
                
                if adapter_lines:
                    capabilities[]],,"adapter_info"] = "\n".join()))))))))))))))adapter_lines)
            
            # Extract feature support information
            if "Supported Features:" in output_str:
                feature_lines = []],,]
                capture = False
                for line in output_str.split()))))))))))))))'\n'):
                    if "Supported Features:" in line:
                        capture = True
                    continue
                    if capture and line.strip()))))))))))))))) and not line.startswith()))))))))))))))"---"):
                        feature_lines.append()))))))))))))))line.strip()))))))))))))))))
                    if capture and "---" in line:
                        capture = False
                
                if feature_lines:
                    capabilities[]],,"supported_features"] = feature_lines
            
                        return capabilities
            
        except subprocess.CalledProcessError as e:
            print()))))))))))))))f"Error testing {}}}}}}}}}}}}}}}}}}}}}}}browser} capabilities: {}}}}}}}}}}}}}}}}}}}}}}}e}")
                        return {}}}}}}}}}}}}}}}}}}}}}}}
                        "browser": browser,
                        "webgpu_available": False,
                        "hardware_acceleration": False,
                        "error": str()))))))))))))))e),
                        "output": e.output.decode()))))))))))))))'utf-8') if e.output else None
            }::
        except subprocess.TimeoutExpired:
            print()))))))))))))))f"Timeout testing {}}}}}}}}}}}}}}}}}}}}}}}browser} capabilities")
                return {}}}}}}}}}}}}}}}}}}}}}}}
                "browser": browser,
                "webgpu_available": False,
                "hardware_acceleration": False,
                "error": "Timeout"
                }

                def test_webgpu_performance()))))))))))))))self, browser: str, model: str, batch_size: int = 1,
                              optimizations: Optional[]],,Dict[]],,str, bool]] = None) -> Dict:
                                  """Test WebGPU performance with specific optimizations.
        
        Args:
            browser: Browser to test.
            model: Model to test.
            batch_size: Batch size to test.
            optimizations: Dictionary of optimization flags to override default settings.
            
        Returns:
            Dictionary with performance results.
            """
        # Determine which optimizations to enable
            compute_shaders = optimizations.get()))))))))))))))"compute_shaders", self.enable_compute_shaders) if optimizations else self.enable_compute_shaders
            parallel_loading = optimizations.get()))))))))))))))"parallel_loading", self.enable_parallel_loading) if optimizations else self.enable_parallel_loading
            shader_precompile = optimizations.get()))))))))))))))"shader_precompile", self.enable_shader_precompile) if optimizations else self.enable_shader_precompile
        
        # Log information about test configuration
        opt_str = []],,]:
        if compute_shaders:
            opt_str.append()))))))))))))))"compute shaders")
        if parallel_loading:
            opt_str.append()))))))))))))))"parallel loading")
        if shader_precompile:
            opt_str.append()))))))))))))))"shader precompilation")
        
            opt_text = ", ".join()))))))))))))))opt_str) if opt_str else "no optimizations"
            print()))))))))))))))f"Testing {}}}}}}}}}}}}}}}}}}}}}}}browser} WebGPU performance with model {}}}}}}}}}}}}}}}}}}}}}}}model} ()))))))))))))))batch_size={}}}}}}}}}}}}}}}}}}}}}}}batch_size}, {}}}}}}}}}}}}}}}}}}}}}}}opt_text})...")
        
        # Construct command to run benchmark
            cmd = []],,
            "./run_web_platform_tests.sh",
            f"--browser={}}}}}}}}}}}}}}}}}}}}}}}browser}",
            f"--model={}}}}}}}}}}}}}}}}}}}}}}}model}",
            f"--batch-size={}}}}}}}}}}}}}}}}}}}}}}}batch_size}",
            "--platform=webgpu",
            "--performance-details"
            ]
        
        # Add optimization flags:
        if compute_shaders:
            cmd.append()))))))))))))))"--enable-compute-shaders")
        if parallel_loading:
            cmd.append()))))))))))))))"--enable-parallel-loading")
        if shader_precompile:
            cmd.append()))))))))))))))"--enable-shader-precompile")
            
        try:
            # Run benchmark
            output = subprocess.check_output()))))))))))))))cmd, timeout=self.timeout, stderr=subprocess.STDOUT)
            output_str = output.decode()))))))))))))))'utf-8')
            
            # Parse benchmark results
            results = {}}}}}}}}}}}}}}}}}}}}}}}
            "browser": browser,
            "model": model,
            "batch_size": batch_size,
            "optimizations": {}}}}}}}}}}}}}}}}}}}}}}}
            "compute_shaders": compute_shaders,
            "parallel_loading": parallel_loading,
            "shader_precompile": shader_precompile
            },
            "error": None
            }
            
            # Extract performance metrics
            if "Inference Time:" in output_str:
                inference_line = []],,line for line in output_str.split()))))))))))))))'\n') if "Inference Time:" in line][]],,0]
                results[]],,"inference_time_ms"] = float()))))))))))))))inference_line.split()))))))))))))))"Inference Time:")[]],,1].strip()))))))))))))))).split())))))))))))))))[]],,0])
            
            if "Loading Time:" in output_str:
                loading_line = []],,line for line in output_str.split()))))))))))))))'\n') if "Loading Time:" in line][]],,0]
                results[]],,"loading_time_ms"] = float()))))))))))))))loading_line.split()))))))))))))))"Loading Time:")[]],,1].strip()))))))))))))))).split())))))))))))))))[]],,0])
            
            if "First Inference Time:" in output_str:
                first_line = []],,line for line in output_str.split()))))))))))))))'\n') if "First Inference Time:" in line][]],,0]
                results[]],,"first_inference_time_ms"] = float()))))))))))))))first_line.split()))))))))))))))"First Inference Time:")[]],,1].strip()))))))))))))))).split())))))))))))))))[]],,0])
            
            if "Shader Compilation Time:" in output_str:
                shader_line = []],,line for line in output_str.split()))))))))))))))'\n') if "Shader Compilation Time:" in line][]],,0]
                results[]],,"shader_compilation_time_ms"] = float()))))))))))))))shader_line.split()))))))))))))))"Shader Compilation Time:")[]],,1].strip()))))))))))))))).split())))))))))))))))[]],,0])
            
            if "Memory Usage:" in output_str:
                memory_line = []],,line for line in output_str.split()))))))))))))))'\n') if "Memory Usage:" in line][]],,0]
                results[]],,"memory_usage_mb"] = float()))))))))))))))memory_line.split()))))))))))))))"Memory Usage:")[]],,1].strip()))))))))))))))).split())))))))))))))))[]],,0])
            
            if "Throughput:" in output_str:
                throughput_line = []],,line for line in output_str.split()))))))))))))))'\n') if "Throughput:" in line][]],,0]
                results[]],,"throughput_items_per_sec"] = float()))))))))))))))throughput_line.split()))))))))))))))"Throughput:")[]],,1].strip()))))))))))))))).split())))))))))))))))[]],,0])
            
            if "Simulation:" in output_str:
                sim_line = []],,line for line in output_str.split()))))))))))))))'\n') if "Simulation:" in line][]],,0]
                results[]],,"simulated"] = "True" in sim_line
            else:
                results[]],,"simulated"] = False
                
                return results
            
        except subprocess.CalledProcessError as e:
            print()))))))))))))))f"Error testing {}}}}}}}}}}}}}}}}}}}}}}}browser} WebGPU performance: {}}}}}}}}}}}}}}}}}}}}}}}e}")
                return {}}}}}}}}}}}}}}}}}}}}}}}
                "browser": browser,
                "model": model,
                "batch_size": batch_size,
                "optimizations": {}}}}}}}}}}}}}}}}}}}}}}}
                "compute_shaders": compute_shaders,
                "parallel_loading": parallel_loading,
                "shader_precompile": shader_precompile
                },
                "error": str()))))))))))))))e),
                "output": e.output.decode()))))))))))))))'utf-8') if e.output else None
            }::
        except subprocess.TimeoutExpired:
            print()))))))))))))))f"Timeout testing {}}}}}}}}}}}}}}}}}}}}}}}browser} WebGPU performance")
                return {}}}}}}}}}}}}}}}}}}}}}}}
                "browser": browser,
                "model": model,
                "batch_size": batch_size,
                "optimizations": {}}}}}}}}}}}}}}}}}}}}}}}
                "compute_shaders": compute_shaders,
                "parallel_loading": parallel_loading,
                "shader_precompile": shader_precompile
                },
                "error": "Timeout"
                }

    def test_audio_optimization()))))))))))))))self, browser: str, model: str, batch_size: int = 1) -> Dict:
        """Test audio model optimization ()))))))))))))))especially Firefox compute shader optimization).
        
        Args:
            browser: Browser to test.
            model: Audio model to test.
            batch_size: Batch size to test.
            
        Returns:
            Dictionary with audio optimization results.
            """
            print()))))))))))))))f"Testing {}}}}}}}}}}}}}}}}}}}}}}}browser} audio optimization with model {}}}}}}}}}}}}}}}}}}}}}}}model} ()))))))))))))))batch_size={}}}}}}}}}}}}}}}}}}}}}}}batch_size})...")
        
        # Test with compute shader optimization enabled
            results_with_opt = self.test_webgpu_performance()))))))))))))))
            browser=browser,
            model=model,
            batch_size=batch_size,
            optimizations={}}}}}}}}}}}}}}}}}}}}}}}"compute_shaders": True, "shader_precompile": True}
            )
        
        # Test with compute shader optimization disabled
            results_without_opt = self.test_webgpu_performance()))))))))))))))
            browser=browser,
            model=model,
            batch_size=batch_size,
            optimizations={}}}}}}}}}}}}}}}}}}}}}}}"compute_shaders": False, "shader_precompile": True}
            )
        
        # Calculate improvement
            improvement = {}}}}}}}}}}}}}}}}}}}}}}}}
        
            if ()))))))))))))))"inference_time_ms" in results_with_opt and
            "inference_time_ms" in results_without_opt and
            not results_with_opt.get()))))))))))))))"error") and:
            not results_without_opt.get()))))))))))))))"error")):
            
                time_with_opt = results_with_opt[]],,"inference_time_ms"]
                time_without_opt = results_without_opt[]],,"inference_time_ms"]
            
            if time_without_opt > 0:
                speedup = time_without_opt / time_with_opt
                improvement_pct = ()))))))))))))))time_without_opt - time_with_opt) / time_without_opt * 100
                
                improvement = {}}}}}}}}}}}}}}}}}}}}}}}
                "inference_time_speedup": speedup,
                "inference_time_improvement_pct": improvement_pct,
                "baseline_time_ms": time_without_opt,
                "optimized_time_ms": time_with_opt
                }
        
                return {}}}}}}}}}}}}}}}}}}}}}}}
                "browser": browser,
                "model": model,
                "batch_size": batch_size,
                "with_optimization": results_with_opt,
                "without_optimization": results_without_opt,
                "improvement": improvement
                }

    def test_multimodal_loading()))))))))))))))self, browser: str, model: str) -> Dict:
        """Test parallel loading optimization for multimodal models.
        
        Args:
            browser: Browser to test.
            model: Multimodal model to test.
            
        Returns:
            Dictionary with parallel loading optimization results.
            """
            print()))))))))))))))f"Testing {}}}}}}}}}}}}}}}}}}}}}}}browser} parallel loading with model {}}}}}}}}}}}}}}}}}}}}}}}model}...")
        
        # Test with parallel loading optimization enabled
            results_with_opt = self.test_webgpu_performance()))))))))))))))
            browser=browser,
            model=model,
            batch_size=1,
            optimizations={}}}}}}}}}}}}}}}}}}}}}}}"parallel_loading": True, "shader_precompile": True}
            )
        
        # Test with parallel loading optimization disabled
            results_without_opt = self.test_webgpu_performance()))))))))))))))
            browser=browser,
            model=model,
            batch_size=1,
            optimizations={}}}}}}}}}}}}}}}}}}}}}}}"parallel_loading": False, "shader_precompile": True}
            )
        
        # Calculate improvement
            improvement = {}}}}}}}}}}}}}}}}}}}}}}}}
        
            if ()))))))))))))))"loading_time_ms" in results_with_opt and
            "loading_time_ms" in results_without_opt and
            not results_with_opt.get()))))))))))))))"error") and:
            not results_without_opt.get()))))))))))))))"error")):
            
                time_with_opt = results_with_opt[]],,"loading_time_ms"]
                time_without_opt = results_without_opt[]],,"loading_time_ms"]
            
            if time_without_opt > 0:
                speedup = time_without_opt / time_with_opt
                improvement_pct = ()))))))))))))))time_without_opt - time_with_opt) / time_without_opt * 100
                
                improvement = {}}}}}}}}}}}}}}}}}}}}}}}
                "loading_time_speedup": speedup,
                "loading_time_improvement_pct": improvement_pct,
                "baseline_time_ms": time_without_opt,
                "optimized_time_ms": time_with_opt
                }
        
                return {}}}}}}}}}}}}}}}}}}}}}}}
                "browser": browser,
                "model": model,
                "with_optimization": results_with_opt,
                "without_optimization": results_without_opt,
                "improvement": improvement
                }

    def test_shader_precompilation()))))))))))))))self, browser: str, model: str) -> Dict:
        """Test shader precompilation optimization.
        
        Args:
            browser: Browser to test.
            model: Model to test.
            
        Returns:
            Dictionary with shader precompilation optimization results.
            """
            print()))))))))))))))f"Testing {}}}}}}}}}}}}}}}}}}}}}}}browser} shader precompilation with model {}}}}}}}}}}}}}}}}}}}}}}}model}...")
        
        # Test with shader precompilation enabled
            results_with_opt = self.test_webgpu_performance()))))))))))))))
            browser=browser,
            model=model,
            batch_size=1,
            optimizations={}}}}}}}}}}}}}}}}}}}}}}}"shader_precompile": True}
            )
        
        # Test with shader precompilation disabled
            results_without_opt = self.test_webgpu_performance()))))))))))))))
            browser=browser,
            model=model,
            batch_size=1,
            optimizations={}}}}}}}}}}}}}}}}}}}}}}}"shader_precompile": False}
            )
        
        # Calculate improvement
            improvement = {}}}}}}}}}}}}}}}}}}}}}}}}
        
            if ()))))))))))))))"first_inference_time_ms" in results_with_opt and
            "first_inference_time_ms" in results_without_opt and
            not results_with_opt.get()))))))))))))))"error") and:
            not results_without_opt.get()))))))))))))))"error")):
            
                time_with_opt = results_with_opt[]],,"first_inference_time_ms"]
                time_without_opt = results_without_opt[]],,"first_inference_time_ms"]
            
            if time_without_opt > 0:
                speedup = time_without_opt / time_with_opt
                improvement_pct = ()))))))))))))))time_without_opt - time_with_opt) / time_without_opt * 100
                
                improvement = {}}}}}}}}}}}}}}}}}}}}}}}
                "first_inference_speedup": speedup,
                "first_inference_improvement_pct": improvement_pct,
                "baseline_time_ms": time_without_opt,
                "optimized_time_ms": time_with_opt
                }
        
        # Extract shader compilation specific metrics if available:::
                if ()))))))))))))))"shader_compilation_time_ms" in results_with_opt and
            "shader_compilation_time_ms" in results_without_opt):
            
                shader_time_with_opt = results_with_opt.get()))))))))))))))"shader_compilation_time_ms", 0)
                shader_time_without_opt = results_without_opt.get()))))))))))))))"shader_compilation_time_ms", 0)
            
            if shader_time_without_opt > 0:
                shader_speedup = shader_time_without_opt / max()))))))))))))))shader_time_with_opt, 0.1)  # Avoid division by zero
                shader_improvement_pct = ()))))))))))))))shader_time_without_opt - shader_time_with_opt) / shader_time_without_opt * 100
                
                improvement[]],,"shader_compilation_speedup"] = shader_speedup
                improvement[]],,"shader_compilation_improvement_pct"] = shader_improvement_pct
                improvement[]],,"baseline_shader_time_ms"] = shader_time_without_opt
                improvement[]],,"optimized_shader_time_ms"] = shader_time_with_opt
        
                return {}}}}}}}}}}}}}}}}}}}}}}}
                "browser": browser,
                "model": model,
                "with_optimization": results_with_opt,
                "without_optimization": results_without_opt,
                "improvement": improvement
                }

                def test_browser_comparison()))))))))))))))self, model: str, batch_size: int = 1,
                              optimization_set: Optional[]],,Dict[]],,str, bool]] = None) -> Dict:
                                  """Test WebGPU performance across browsers with the same model and optimizations.
        
        Args:
            model: Model to test.
            batch_size: Batch size to test.
            optimization_set: Dictionary of optimization flags.
            
        Returns:
            Dictionary with cross-browser comparison results.
            """
        # Default to all optimizations enabled if None provided:
        if optimization_set is None:
            optimization_set = {}}}}}}}}}}}}}}}}}}}}}}}
            "compute_shaders": self.enable_compute_shaders,
            "parallel_loading": self.enable_parallel_loading,
            "shader_precompile": self.enable_shader_precompile
            }
            
        # Convert optimization set to string for display
            opt_str = []],,]
        if optimization_set.get()))))))))))))))"compute_shaders"):
            opt_str.append()))))))))))))))"compute shaders")
        if optimization_set.get()))))))))))))))"parallel_loading"):
            opt_str.append()))))))))))))))"parallel loading")
        if optimization_set.get()))))))))))))))"shader_precompile"):
            opt_str.append()))))))))))))))"shader precompilation")
        
            opt_text = ", ".join()))))))))))))))opt_str) if opt_str else "no optimizations"
            print()))))))))))))))f"Running cross-browser comparison for model {}}}}}}}}}}}}}}}}}}}}}}}model} ()))))))))))))))batch_size={}}}}}}}}}}}}}}}}}}}}}}}batch_size}, {}}}}}}}}}}}}}}}}}}}}}}}opt_text})...")
        
        results = {}}}}}}}}}}}}}}}}}}}}}}}:
            "model": model,
            "batch_size": batch_size,
            "optimizations": optimization_set,
            "browsers": {}}}}}}}}}}}}}}}}}}}}}}}}
            }
        
        # Test each browser with the same configuration
        for browser in self.browsers:
            browser_result = self.test_webgpu_performance()))))))))))))))
            browser=browser,
            model=model,
            batch_size=batch_size,
            optimizations=optimization_set
            )
            
            results[]],,"browsers"][]],,browser] = browser_result
        
        # Find best performing browser
            best_browser = None
            best_time = float()))))))))))))))'inf')
        
        for browser, result in results[]],,"browsers"].items()))))))))))))))):
            if result.get()))))))))))))))"error"):
            continue
                
            inference_time = result.get()))))))))))))))"inference_time_ms")
            if inference_time and inference_time < best_time:
                best_time = inference_time
                best_browser = browser
        
        # Calculate relative performance compared to best browser
        if best_browser:
            results[]],,"best_browser"] = best_browser
            results[]],,"best_inference_time_ms"] = best_time
            
            for browser, result in results[]],,"browsers"].items()))))))))))))))):
                if result.get()))))))))))))))"error") or "inference_time_ms" not in result:
                continue
                    
                relative_performance = best_time / result[]],,"inference_time_ms"]
                result[]],,"relative_performance"] = relative_performance
                result[]],,"performance_vs_best_pct"] = ()))))))))))))))relative_performance - 1) * 100
        
            return results

    def run_tests()))))))))))))))self) -> Dict:
        """Run all WebGPU browser comparison tests.
        
        Returns:
            Dictionary with all test results.
            """
            results = {}}}}}}}}}}}}}}}}}}}}}}}
            "timestamp": time.time()))))))))))))))),
            "system": {}}}}}}}}}}}}}}}}}}}}}}}
            "platform": platform.system()))))))))))))))),
            "platform_version": platform.version()))))))))))))))),
            "processor": platform.processor())))))))))))))))
            },
            "browsers": {}}}}}}}}}}}}}}}}}}}}}}}},
            "audio_optimization": {}}}}}}}}}}}}}}}}}}}}}}}},
            "multimodal_loading": {}}}}}}}}}}}}}}}}}}}}}}}},
            "shader_precompilation": {}}}}}}}}}}}}}}}}}}}}}}}},
            "browser_comparison": {}}}}}}}}}}}}}}}}}}}}}}}}
            }
        
        # Test capabilities for each browser
        for browser in self.browsers:
            results[]],,"browsers"][]],,browser] = self.test_browser_webgpu_capabilities()))))))))))))))browser)
        
        # Test audio optimization ()))))))))))))))Firefox compute shader performance)
        for model in []],,m for m in self.models if m in AUDIO_MODELS]:
            results[]],,"audio_optimization"][]],,model] = {}}}}}}}}}}}}}}}}}}}}}}}}
            
            for browser in self.browsers:
                # Skip if browser doesn't support WebGPU:::
                if not results[]],,"browsers"][]],,browser].get()))))))))))))))"webgpu_available", False):
                    print()))))))))))))))f"Skipping audio optimization tests for {}}}}}}}}}}}}}}}}}}}}}}}browser} as WebGPU is not available")
                continue
                    
                for batch_size in self.batch_sizes:
                    key = f"{}}}}}}}}}}}}}}}}}}}}}}}browser}_batch{}}}}}}}}}}}}}}}}}}}}}}}batch_size}"
                    results[]],,"audio_optimization"][]],,model][]],,key] = self.test_audio_optimization()))))))))))))))
                    browser=browser,
                    model=model,
                    batch_size=batch_size
                    )
        
        # Test multimodal loading optimization
        for model in []],,m for m in self.models if m in MULTIMODAL_MODELS]:
            results[]],,"multimodal_loading"][]],,model] = {}}}}}}}}}}}}}}}}}}}}}}}}
            
            for browser in self.browsers:
                # Skip if browser doesn't support WebGPU:::
                if not results[]],,"browsers"][]],,browser].get()))))))))))))))"webgpu_available", False):
                    print()))))))))))))))f"Skipping multimodal loading tests for {}}}}}}}}}}}}}}}}}}}}}}}browser} as WebGPU is not available")
                continue
                    
                results[]],,"multimodal_loading"][]],,model][]],,browser] = self.test_multimodal_loading()))))))))))))))
                browser=browser,
                model=model
                )
        
        # Test shader precompilation optimization
        for model in self.models:
            results[]],,"shader_precompilation"][]],,model] = {}}}}}}}}}}}}}}}}}}}}}}}}
            
            for browser in self.browsers:
                # Skip if browser doesn't support WebGPU:::
                if not results[]],,"browsers"][]],,browser].get()))))))))))))))"webgpu_available", False):
                    print()))))))))))))))f"Skipping shader precompilation tests for {}}}}}}}}}}}}}}}}}}}}}}}browser} as WebGPU is not available")
                continue
                    
                results[]],,"shader_precompilation"][]],,model][]],,browser] = self.test_shader_precompilation()))))))))))))))
                browser=browser,
                model=model
                )
        
        # Test browser comparison with different optimization combinations
        for model in self.models:
            results[]],,"browser_comparison"][]],,model] = {}}}}}}}}}}}}}}}}}}}}}}}}
            
            # No optimizations
            key = "no_optimizations"
            results[]],,"browser_comparison"][]],,model][]],,key] = self.test_browser_comparison()))))))))))))))
            model=model,
            batch_size=self.batch_sizes[]],,0] if self.batch_sizes else 1,
                optimization_set={}}}}}}}}}}}}}}}}}}}}}}}::
                    "compute_shaders": False,
                    "parallel_loading": False,
                    "shader_precompile": False
                    }
                    )
            
            # All optimizations
                    key = "all_optimizations"
                    results[]],,"browser_comparison"][]],,model][]],,key] = self.test_browser_comparison()))))))))))))))
                    model=model,
                    batch_size=self.batch_sizes[]],,0] if self.batch_sizes else 1,
                optimization_set={}}}}}}}}}}}}}}}}}}}}}}}::
                    "compute_shaders": True,
                    "parallel_loading": True,
                    "shader_precompile": True
                    }
                    )
            
            # Compute shaders only ()))))))))))))))for audio models)
            if model in AUDIO_MODELS:
                key = "compute_shaders_only"
                results[]],,"browser_comparison"][]],,model][]],,key] = self.test_browser_comparison()))))))))))))))
                model=model,
                batch_size=self.batch_sizes[]],,0] if self.batch_sizes else 1,
                    optimization_set={}}}}}}}}}}}}}}}}}}}}}}}::
                        "compute_shaders": True,
                        "parallel_loading": False,
                        "shader_precompile": False
                        }
                        )
            
            # Parallel loading only ()))))))))))))))for multimodal models)
            if model in MULTIMODAL_MODELS:
                key = "parallel_loading_only"
                results[]],,"browser_comparison"][]],,model][]],,key] = self.test_browser_comparison()))))))))))))))
                model=model,
                batch_size=self.batch_sizes[]],,0] if self.batch_sizes else 1,
                    optimization_set={}}}}}}}}}}}}}}}}}}}}}}}::
                        "compute_shaders": False,
                        "parallel_loading": True,
                        "shader_precompile": False
                        }
                        )
        
        # Save results to file
                        output_file = os.path.join()))))))))))))))self.output_dir, f"webgpu_browser_comparison_{}}}}}}}}}}}}}}}}}}}}}}}int()))))))))))))))time.time()))))))))))))))))}.json")
        with open()))))))))))))))output_file, 'w') as f:
            json.dump()))))))))))))))results, f, indent=2)
            print()))))))))))))))f"Results saved to {}}}}}}}}}}}}}}}}}}}}}}}output_file}")
        
        # Store results in database if available:::
        if HAS_DB_API and self.db_path:
            try:
                store_webgpu_comparison_result()))))))))))))))results, self.db_path)
                print()))))))))))))))f"Results stored in database at {}}}}}}}}}}}}}}}}}}}}}}}self.db_path}")
            except Exception as e:
                print()))))))))))))))f"Error storing results in database: {}}}}}}}}}}}}}}}}}}}}}}}e}")
        
                self.results = results
                return results
        
    def generate_report()))))))))))))))self, output_format: str = "markdown") -> str:
        """Generate a report from test results.
        
        Args:
            output_format: Format for the report. Supports "markdown" or "html".
            
        Returns:
            Report string in the specified format.
            """
        if not self.results:
            return "No test results available. Run tests first."
            
        if output_format == "markdown":
            return self._generate_markdown_report())))))))))))))))
        elif output_format == "html":
            return self._generate_html_report())))))))))))))))
        else:
            return f"Unsupported output format: {}}}}}}}}}}}}}}}}}}}}}}}output_format}"
    
    def _generate_markdown_report()))))))))))))))self) -> str:
        """Generate a markdown report from test results.
        
        Returns:
            Markdown report string.
            """
            report = "# WebGPU Cross-Browser Comparison Report\n\n"
        
        # System information
            report += "## System Information\n\n"
            report += f"- Platform: {}}}}}}}}}}}}}}}}}}}}}}}self.results[]],,'system'][]],,'platform']}\n"
            report += f"- Platform Version: {}}}}}}}}}}}}}}}}}}}}}}}self.results[]],,'system'][]],,'platform_version']}\n"
            report += f"- Processor: {}}}}}}}}}}}}}}}}}}}}}}}self.results[]],,'system'][]],,'processor']}\n"
            report += f"- Timestamp: {}}}}}}}}}}}}}}}}}}}}}}}time.ctime()))))))))))))))self.results[]],,'timestamp'])}\n\n"
        
        # Browser capabilities
            report += "## Browser WebGPU Capabilities\n\n"
            report += "| Browser | WebGPU Available | Hardware Acceleration | GPU Device |\n"
            report += "|---------|-----------------|----------------------|------------|\n"
        
        for browser, capabilities in self.results[]],,"browsers"].items()))))))))))))))):
            webgpu = "✅" if capabilities.get()))))))))))))))"webgpu_available", False) else "❌"
            hw_accel = "✅" if capabilities.get()))))))))))))))"hardware_acceleration", False) else "❌"
            device = capabilities.get()))))))))))))))"gpu_device", "N/A")
            
            report += f"| {}}}}}}}}}}}}}}}}}}}}}}}browser} | {}}}}}}}}}}}}}}}}}}}}}}}webgpu} | {}}}}}}}}}}}}}}}}}}}}}}}hw_accel} | {}}}}}}}}}}}}}}}}}}}}}}}device} |\n"
        
            report += "\n"
        
        # Browser comparison summary
            report += "## Cross-Browser Performance Summary\n\n"
        :
        for model, comparisons in self.results[]],,"browser_comparison"].items()))))))))))))))):
            report += f"### Model: {}}}}}}}}}}}}}}}}}}}}}}}model}\n\n"
            
            # No optimizations vs All optimizations
            if "no_optimizations" in comparisons and "all_optimizations" in comparisons:
                no_opt = comparisons[]],,"no_optimizations"]
                all_opt = comparisons[]],,"all_optimizations"]
                
                report += "#### No Optimizations vs All Optimizations\n\n"
                report += "| Browser | No Optimizations Time ()))))))))))))))ms) | All Optimizations Time ()))))))))))))))ms) | Improvement ()))))))))))))))%) | Simulated |\n"
                report += "|---------|----------------------------|------------------------------|----------------|----------|\n"
                
                for browser in self.browsers:
                    if browser not in no_opt[]],,"browsers"] or browser not in all_opt[]],,"browsers"]:
                    continue
                    
                    no_opt_time = no_opt[]],,"browsers"][]],,browser].get()))))))))))))))"inference_time_ms", "N/A")
                    all_opt_time = all_opt[]],,"browsers"][]],,browser].get()))))))))))))))"inference_time_ms", "N/A")
                    
                    improvement = "N/A"
                    if isinstance()))))))))))))))no_opt_time, ()))))))))))))))int, float)) and isinstance()))))))))))))))all_opt_time, ()))))))))))))))int, float)) and no_opt_time > 0:
                        improvement = f"{}}}}}}}}}}}}}}}}}}}}}}}()))))))))))))))()))))))))))))))no_opt_time - all_opt_time) / no_opt_time * 100):.1f}"
                    
                        simulated = "Yes" if all_opt[]],,"browsers"][]],,browser].get()))))))))))))))"simulated", False) else "No"
                    :
                    if all_opt[]],,"browsers"][]],,browser].get()))))))))))))))"error") or no_opt[]],,"browsers"][]],,browser].get()))))))))))))))"error"):
                        report += f"| {}}}}}}}}}}}}}}}}}}}}}}}browser} | Error | Error | - | - |\n"
                    else:
                        report += f"| {}}}}}}}}}}}}}}}}}}}}}}}browser} | {}}}}}}}}}}}}}}}}}}}}}}}no_opt_time} | {}}}}}}}}}}}}}}}}}}}}}}}all_opt_time} | {}}}}}}}}}}}}}}}}}}}}}}}improvement} | {}}}}}}}}}}}}}}}}}}}}}}}simulated} |\n"
                
                # Best browser
                if all_opt.get()))))))))))))))"best_browser"):
                    report += f"\n**Best browser with all optimizations:** {}}}}}}}}}}}}}}}}}}}}}}}all_opt[]],,'best_browser']}\n\n"
            
            # Firefox vs Chrome for audio models
            if model in AUDIO_MODELS and "all_optimizations" in comparisons:
                all_opt = comparisons[]],,"all_optimizations"]
                
                firefox_time = None
                chrome_time = None
                
                if "firefox" in all_opt[]],,"browsers"] and "chrome" in all_opt[]],,"browsers"]:
                    firefox_time = all_opt[]],,"browsers"][]],,"firefox"].get()))))))))))))))"inference_time_ms")
                    chrome_time = all_opt[]],,"browsers"][]],,"chrome"].get()))))))))))))))"inference_time_ms")
                    
                    if firefox_time and chrome_time and not all_opt[]],,"browsers"][]],,"firefox"].get()))))))))))))))"error") and not all_opt[]],,"browsers"][]],,"chrome"].get()))))))))))))))"error"):
                        report += "#### Firefox vs Chrome Audio Model Performance\n\n"
                        
                        if firefox_time < chrome_time:
                            improvement = ()))))))))))))))chrome_time - firefox_time) / chrome_time * 100
                            report += f"Firefox is **{}}}}}}}}}}}}}}}}}}}}}}}improvement:.1f}%** faster than Chrome for audio model {}}}}}}}}}}}}}}}}}}}}}}}model}\n\n"
                        else:
                            improvement = ()))))))))))))))firefox_time - chrome_time) / firefox_time * 100
                            report += f"Chrome is **{}}}}}}}}}}}}}}}}}}}}}}}improvement:.1f}%** faster than Firefox for audio model {}}}}}}}}}}}}}}}}}}}}}}}model}\n\n"
        
        # Audio optimization results
        if self.results[]],,"audio_optimization"]:
            report += "## Audio Model Compute Shader Optimization\n\n"
            
            for model, browser_results in self.results[]],,"audio_optimization"].items()))))))))))))))):
                report += f"### Model: {}}}}}}}}}}}}}}}}}}}}}}}model}\n\n"
                report += "| Browser | Batch Size | Without Optimization ()))))))))))))))ms) | With Optimization ()))))))))))))))ms) | Improvement ()))))))))))))))%) | Simulated |\n"
                report += "|---------|------------|---------------------------|------------------------|----------------|----------|\n"
                
                for key, results in browser_results.items()))))))))))))))):
                    browser, batch = key.split()))))))))))))))"_batch")
                    
                    without_opt = results[]],,"without_optimization"].get()))))))))))))))"inference_time_ms", "N/A")
                    with_opt = results[]],,"with_optimization"].get()))))))))))))))"inference_time_ms", "N/A")
                    
                    improvement = "N/A"
                    if "improvement" in results and "inference_time_improvement_pct" in results[]],,"improvement"]:
                        improvement = f"{}}}}}}}}}}}}}}}}}}}}}}}results[]],,'improvement'][]],,'inference_time_improvement_pct']:.1f}"
                    
                        simulated = "Yes" if results[]],,"with_optimization"].get()))))))))))))))"simulated", False) else "No"
                    :::
                    if results[]],,"with_optimization"].get()))))))))))))))"error") or results[]],,"without_optimization"].get()))))))))))))))"error"):
                        report += f"| {}}}}}}}}}}}}}}}}}}}}}}}browser} | {}}}}}}}}}}}}}}}}}}}}}}}batch} | Error | Error | - | - |\n"
                    else:
                        report += f"| {}}}}}}}}}}}}}}}}}}}}}}}browser} | {}}}}}}}}}}}}}}}}}}}}}}}batch} | {}}}}}}}}}}}}}}}}}}}}}}}without_opt} | {}}}}}}}}}}}}}}}}}}}}}}}with_opt} | {}}}}}}}}}}}}}}}}}}}}}}}improvement} | {}}}}}}}}}}}}}}}}}}}}}}}simulated} |\n"
                
                        report += "\n"
        
        # Multimodal loading optimization results
        if self.results[]],,"multimodal_loading"]:
            report += "## Multimodal Model Parallel Loading Optimization\n\n"
            
            for model, browser_results in self.results[]],,"multimodal_loading"].items()))))))))))))))):
                report += f"### Model: {}}}}}}}}}}}}}}}}}}}}}}}model}\n\n"
                report += "| Browser | Without Optimization ()))))))))))))))ms) | With Optimization ()))))))))))))))ms) | Improvement ()))))))))))))))%) | Simulated |\n"
                report += "|---------|---------------------------|------------------------|----------------|----------|\n"
                
                for browser, results in browser_results.items()))))))))))))))):
                    without_opt = results[]],,"without_optimization"].get()))))))))))))))"loading_time_ms", "N/A")
                    with_opt = results[]],,"with_optimization"].get()))))))))))))))"loading_time_ms", "N/A")
                    
                    improvement = "N/A"
                    if "improvement" in results and "loading_time_improvement_pct" in results[]],,"improvement"]:
                        improvement = f"{}}}}}}}}}}}}}}}}}}}}}}}results[]],,'improvement'][]],,'loading_time_improvement_pct']:.1f}"
                    
                        simulated = "Yes" if results[]],,"with_optimization"].get()))))))))))))))"simulated", False) else "No"
                    :::
                    if results[]],,"with_optimization"].get()))))))))))))))"error") or results[]],,"without_optimization"].get()))))))))))))))"error"):
                        report += f"| {}}}}}}}}}}}}}}}}}}}}}}}browser} | Error | Error | - | - |\n"
                    else:
                        report += f"| {}}}}}}}}}}}}}}}}}}}}}}}browser} | {}}}}}}}}}}}}}}}}}}}}}}}without_opt} | {}}}}}}}}}}}}}}}}}}}}}}}with_opt} | {}}}}}}}}}}}}}}}}}}}}}}}improvement} | {}}}}}}}}}}}}}}}}}}}}}}}simulated} |\n"
                
                        report += "\n"
        
        # Shader precompilation optimization results
        if self.results[]],,"shader_precompilation"]:
            report += "## Shader Precompilation Optimization\n\n"
            
            for model, browser_results in self.results[]],,"shader_precompilation"].items()))))))))))))))):
                report += f"### Model: {}}}}}}}}}}}}}}}}}}}}}}}model}\n\n"
                report += "| Browser | First Inference Without ()))))))))))))))ms) | First Inference With ()))))))))))))))ms) | Improvement ()))))))))))))))%) | Simulated |\n"
                report += "|---------|------------------------------|---------------------------|----------------|----------|\n"
                
                for browser, results in browser_results.items()))))))))))))))):
                    without_opt = results[]],,"without_optimization"].get()))))))))))))))"first_inference_time_ms", "N/A")
                    with_opt = results[]],,"with_optimization"].get()))))))))))))))"first_inference_time_ms", "N/A")
                    
                    improvement = "N/A"
                    if "improvement" in results and "first_inference_improvement_pct" in results[]],,"improvement"]:
                        improvement = f"{}}}}}}}}}}}}}}}}}}}}}}}results[]],,'improvement'][]],,'first_inference_improvement_pct']:.1f}"
                    
                        simulated = "Yes" if results[]],,"with_optimization"].get()))))))))))))))"simulated", False) else "No"
                    :::
                    if results[]],,"with_optimization"].get()))))))))))))))"error") or results[]],,"without_optimization"].get()))))))))))))))"error"):
                        report += f"| {}}}}}}}}}}}}}}}}}}}}}}}browser} | Error | Error | - | - |\n"
                    else:
                        report += f"| {}}}}}}}}}}}}}}}}}}}}}}}browser} | {}}}}}}}}}}}}}}}}}}}}}}}without_opt} | {}}}}}}}}}}}}}}}}}}}}}}}with_opt} | {}}}}}}}}}}}}}}}}}}}}}}}improvement} | {}}}}}}}}}}}}}}}}}}}}}}}simulated} |\n"
                
                        report += "\n"
        
        # Recommendations
                        report += "## Browser-Specific Recommendations\n\n"
        
        # Audio model recommendations
                        report += "### Audio Models ()))))))))))))))Whisper, Wav2Vec2, CLAP)\n\n"
        
                        firefox_better = False
        for model in AUDIO_MODELS:
            if ()))))))))))))))model in self.results[]],,"audio_optimization"] and 
                "firefox_batch1" in self.results[]],,"audio_optimization"][]],,model] and ::
                "chrome_batch1" in self.results[]],,"audio_optimization"][]],,model]):
                
                    firefox_results = self.results[]],,"audio_optimization"][]],,model][]],,"firefox_batch1"]
                    chrome_results = self.results[]],,"audio_optimization"][]],,model][]],,"chrome_batch1"]
                
                    if ()))))))))))))))not firefox_results[]],,"with_optimization"].get()))))))))))))))"error") and
                    not chrome_results[]],,"with_optimization"].get()))))))))))))))"error") and
                    "inference_time_ms" in firefox_results[]],,"with_optimization"] and::
                    "inference_time_ms" in chrome_results[]],,"with_optimization"]):
                    
                        firefox_time = firefox_results[]],,"with_optimization"][]],,"inference_time_ms"]
                        chrome_time = chrome_results[]],,"with_optimization"][]],,"inference_time_ms"]
                    
                    if firefox_time < chrome_time:
                        firefox_better = True
                        diff_pct = ()))))))))))))))chrome_time - firefox_time) / chrome_time * 100
                        report += f"- For {}}}}}}}}}}}}}}}}}}}}}}}model}, Firefox is {}}}}}}}}}}}}}}}}}}}}}}}diff_pct:.1f}% faster than Chrome\n"
        
        if firefox_better:
            report += "- **Recommendation:** For audio models like Whisper and Wav2Vec2, Firefox with compute shader optimization is the preferred browser\n"
            report += "- Enable compute shader optimization when running audio models\n"
        else:
            report += "- No clear browser preference for audio models was detected in this test run\n"
        
            report += "\n"
        
        # Multimodal model recommendations
            report += "### Multimodal Models ()))))))))))))))CLIP, LLaVA)\n\n"
        
            best_browser = None
            best_improvement = 0
        
        for model in MULTIMODAL_MODELS:
            if model in self.results[]],,"multimodal_loading"]:
                for browser, results in self.results[]],,"multimodal_loading"][]],,model].items()))))))))))))))):
                    if ()))))))))))))))"improvement" in results and ::
                        "loading_time_improvement_pct" in results[]],,"improvement"] and:
                        results[]],,"improvement"][]],,"loading_time_improvement_pct"] > best_improvement):
                        
                            best_improvement = results[]],,"improvement"][]],,"loading_time_improvement_pct"]
                            best_browser = browser
        
        if best_browser:
            report += f"- **Recommendation:** For multimodal models, {}}}}}}}}}}}}}}}}}}}}}}}best_browser} with parallel loading enabled shows the best performance\n"
            report += f"- Parallel loading reduces model loading time by up to {}}}}}}}}}}}}}}}}}}}}}}}best_improvement:.1f}%\n"
        else:
            report += "- No clear browser preference for multimodal models was detected in this test run\n"
        
            report += "\n"
        
        # Text/Vision model recommendations
            report += "### Text and Vision Models ()))))))))))))))BERT, ViT)\n\n"
        
            best_browser = None
            best_improvement = 0
        
        for model in TEXT_VISION_MODELS:
            if model in self.results[]],,"shader_precompilation"]:
                for browser, results in self.results[]],,"shader_precompilation"][]],,model].items()))))))))))))))):
                    if ()))))))))))))))"improvement" in results and ::
                        "first_inference_improvement_pct" in results[]],,"improvement"] and:
                        results[]],,"improvement"][]],,"first_inference_improvement_pct"] > best_improvement):
                        
                            best_improvement = results[]],,"improvement"][]],,"first_inference_improvement_pct"]
                            best_browser = browser
        
        if best_browser:
            report += f"- **Recommendation:** For text and vision models, {}}}}}}}}}}}}}}}}}}}}}}}best_browser} with shader precompilation shows the best first inference performance\n"
            report += f"- Shader precompilation reduces first inference time by up to {}}}}}}}}}}}}}}}}}}}}}}}best_improvement:.1f}%\n"
        else:
            report += "- No clear browser preference for text/vision models was detected in this test run\n"
        
            report += "\n"
        
            return report

    def _generate_html_report()))))))))))))))self) -> str:
        """Generate an HTML report from test results.
        
        Returns:
            HTML report string.
            """
        # Basic HTML report with styling
            html = """<!DOCTYPE html>
            <html>
            <head>
            <title>WebGPU Cross-Browser Comparison Report</title>
            <style>
            body {}}}}}}}}}}}}}}}}}}}}}}} font-family: Arial, sans-serif; margin: 20px; max-width: 1200px; margin: 0 auto; line-height: 1.6; }
            h1 {}}}}}}}}}}}}}}}}}}}}}}} color: #333; border-bottom: 2px solid #eee; padding-bottom: 10px; }
            h2 {}}}}}}}}}}}}}}}}}}}}}}} color: #444; margin-top: 25px; border-bottom: 1px solid #eee; padding-bottom: 5px; }
            h3 {}}}}}}}}}}}}}}}}}}}}}}} color: #555; margin-top: 20px; }
            h4 {}}}}}}}}}}}}}}}}}}}}}}} color: #666; }
            table {}}}}}}}}}}}}}}}}}}}}}}} border-collapse: collapse; width: 100%; margin-bottom: 20px; }
            th, td {}}}}}}}}}}}}}}}}}}}}}}} border: 1px solid #ddd; padding: 8px; text-align: left; }
            th {}}}}}}}}}}}}}}}}}}}}}}} background-color: #f2f2f2; }
            tr:nth-child()))))))))))))))even) {}}}}}}}}}}}}}}}}}}}}}}} background-color: #f9f9f9; }
            .success {}}}}}}}}}}}}}}}}}}}}}}} color: green; }
            .failure {}}}}}}}}}}}}}}}}}}}}}}} color: red; }
            .warning {}}}}}}}}}}}}}}}}}}}}}}} color: orange; }
            .recommendation {}}}}}}}}}}}}}}}}}}}}}}} background-color: #f8f8f8; padding: 15px; border-left: 4px solid #2196F3; margin: 20px 0; }
            .improvement-high {}}}}}}}}}}}}}}}}}}}}}}} color: green; font-weight: bold; }
            .improvement-medium {}}}}}}}}}}}}}}}}}}}}}}} color: green; }
            .improvement-low {}}}}}}}}}}}}}}}}}}}}}}} color: #888; }
            </style>
            </head>
            <body>
            <h1>WebGPU Cross-Browser Comparison Report</h1>
            """
        
        # System information
            html += "<h2>System Information</h2>"
            html += "<ul>"
            html += f"<li><strong>Platform:</strong> {}}}}}}}}}}}}}}}}}}}}}}}self.results[]],,'system'][]],,'platform']}</li>"
            html += f"<li><strong>Platform Version:</strong> {}}}}}}}}}}}}}}}}}}}}}}}self.results[]],,'system'][]],,'platform_version']}</li>"
            html += f"<li><strong>Processor:</strong> {}}}}}}}}}}}}}}}}}}}}}}}self.results[]],,'system'][]],,'processor']}</li>"
            html += f"<li><strong>Timestamp:</strong> {}}}}}}}}}}}}}}}}}}}}}}}time.ctime()))))))))))))))self.results[]],,'timestamp'])}</li>"
            html += "</ul>"
        
        # Browser capabilities
            html += "<h2>Browser WebGPU Capabilities</h2>"
            html += "<table>"
            html += "<tr><th>Browser</th><th>WebGPU Available</th><th>Hardware Acceleration</th><th>GPU Device</th></tr>"
        
        for browser, capabilities in self.results[]],,"browsers"].items()))))))))))))))):
            webgpu = '<span class="success">✓</span>' if capabilities.get()))))))))))))))"webgpu_available", False) else '<span class="failure">✗</span>'
            hw_accel = '<span class="success">✓</span>' if capabilities.get()))))))))))))))"hardware_acceleration", False) else '<span class="failure">✗</span>'
            device = capabilities.get()))))))))))))))"gpu_device", "N/A")
            
            html += f"<tr><td>{}}}}}}}}}}}}}}}}}}}}}}}browser}</td><td>{}}}}}}}}}}}}}}}}}}}}}}}webgpu}</td><td>{}}}}}}}}}}}}}}}}}}}}}}}hw_accel}</td><td>{}}}}}}}}}}}}}}}}}}}}}}}device}</td></tr>"
        
            html += "</table>"
        
        # Browser comparison summary
            html += "<h2>Cross-Browser Performance Summary</h2>"
        :
        for model, comparisons in self.results[]],,"browser_comparison"].items()))))))))))))))):
            html += f"<h3>Model: {}}}}}}}}}}}}}}}}}}}}}}}model}</h3>"
            
            # No optimizations vs All optimizations
            if "no_optimizations" in comparisons and "all_optimizations" in comparisons:
                no_opt = comparisons[]],,"no_optimizations"]
                all_opt = comparisons[]],,"all_optimizations"]
                
                html += "<h4>No Optimizations vs All Optimizations</h4>"
                html += "<table>"
                html += "<tr><th>Browser</th><th>No Optimizations Time ()))))))))))))))ms)</th><th>All Optimizations Time ()))))))))))))))ms)</th><th>Improvement</th><th>Simulated</th></tr>"
                
                for browser in self.browsers:
                    if browser not in no_opt[]],,"browsers"] or browser not in all_opt[]],,"browsers"]:
                    continue
                    
                    if all_opt[]],,"browsers"][]],,browser].get()))))))))))))))"error") or no_opt[]],,"browsers"][]],,browser].get()))))))))))))))"error"):
                        html += f"<tr><td>{}}}}}}}}}}}}}}}}}}}}}}}browser}</td><td colspan='4'>Error: {}}}}}}}}}}}}}}}}}}}}}}}all_opt[]],,'browsers'][]],,browser].get()))))))))))))))'error', 'Unknown error')}</td></tr>"
                    continue
                    
                    no_opt_time = no_opt[]],,"browsers"][]],,browser].get()))))))))))))))"inference_time_ms", "N/A")
                    all_opt_time = all_opt[]],,"browsers"][]],,browser].get()))))))))))))))"inference_time_ms", "N/A")
                    
                    improvement_html = "N/A"
                    if isinstance()))))))))))))))no_opt_time, ()))))))))))))))int, float)) and isinstance()))))))))))))))all_opt_time, ()))))))))))))))int, float)) and no_opt_time > 0:
                        improvement_val = ()))))))))))))))no_opt_time - all_opt_time) / no_opt_time * 100
                        
                        # Classify improvement for styling
                        if improvement_val > 20:
                            improvement_class = "improvement-high"
                        elif improvement_val > 5:
                            improvement_class = "improvement-medium"
                        else:
                            improvement_class = "improvement-low"
                            
                            improvement_html = f'<span class="{}}}}}}}}}}}}}}}}}}}}}}}improvement_class}">{}}}}}}}}}}}}}}}}}}}}}}}improvement_val:.1f}%</span>'
                    
                            simulated = '<span class="warning">Yes</span>' if all_opt[]],,"browsers"][]],,browser].get()))))))))))))))"simulated", False) else '<span class="success">No</span>'
                    
                            html += f"<tr><td>{}}}}}}}}}}}}}}}}}}}}}}}browser}</td><td>{}}}}}}}}}}}}}}}}}}}}}}}no_opt_time}</td><td>{}}}}}}}}}}}}}}}}}}}}}}}all_opt_time}</td><td>{}}}}}}}}}}}}}}}}}}}}}}}improvement_html}</td><td>{}}}}}}}}}}}}}}}}}}}}}}}simulated}</td></tr>"
                
                            html += "</table>"
                
                # Best browser:
                if all_opt.get()))))))))))))))"best_browser"):
                    html += f'<p><strong>Best browser with all optimizations:</strong> <span class="success">{}}}}}}}}}}}}}}}}}}}}}}}all_opt[]],,"best_browser"]}</span></p>'
            
            # Firefox vs Chrome for audio models
            if model in AUDIO_MODELS and "all_optimizations" in comparisons:
                all_opt = comparisons[]],,"all_optimizations"]
                
                firefox_time = None
                chrome_time = None
                
                if "firefox" in all_opt[]],,"browsers"] and "chrome" in all_opt[]],,"browsers"]:
                    firefox_time = all_opt[]],,"browsers"][]],,"firefox"].get()))))))))))))))"inference_time_ms")
                    chrome_time = all_opt[]],,"browsers"][]],,"chrome"].get()))))))))))))))"inference_time_ms")
                    
                    if firefox_time and chrome_time and not all_opt[]],,"browsers"][]],,"firefox"].get()))))))))))))))"error") and not all_opt[]],,"browsers"][]],,"chrome"].get()))))))))))))))"error"):
                        html += "<h4>Firefox vs Chrome Audio Model Performance</h4>"
                        
                        if firefox_time < chrome_time:
                            improvement = ()))))))))))))))chrome_time - firefox_time) / chrome_time * 100
                            html += f'<p>Firefox is <span class="improvement-high">{}}}}}}}}}}}}}}}}}}}}}}}improvement:.1f}% faster</span> than Chrome for audio model {}}}}}}}}}}}}}}}}}}}}}}}model}</p>'
                        else:
                            improvement = ()))))))))))))))firefox_time - chrome_time) / firefox_time * 100
                            html += f'<p>Chrome is <span class="improvement-high">{}}}}}}}}}}}}}}}}}}}}}}}improvement:.1f}% faster</span> than Firefox for audio model {}}}}}}}}}}}}}}}}}}}}}}}model}</p>'
        
        # Audio optimization results
        if self.results[]],,"audio_optimization"]:
            html += "<h2>Audio Model Compute Shader Optimization</h2>"
            
            for model, browser_results in self.results[]],,"audio_optimization"].items()))))))))))))))):
                html += f"<h3>Model: {}}}}}}}}}}}}}}}}}}}}}}}model}</h3>"
                html += "<table>"
                html += "<tr><th>Browser</th><th>Batch Size</th><th>Without Optimization ()))))))))))))))ms)</th><th>With Optimization ()))))))))))))))ms)</th><th>Improvement</th><th>Simulated</th></tr>"
                
                for key, results in browser_results.items()))))))))))))))):
                    browser, batch = key.split()))))))))))))))"_batch")
                    
                    if results[]],,"with_optimization"].get()))))))))))))))"error") or results[]],,"without_optimization"].get()))))))))))))))"error"):
                        html += f"<tr><td>{}}}}}}}}}}}}}}}}}}}}}}}browser}</td><td>{}}}}}}}}}}}}}}}}}}}}}}}batch}</td><td colspan='4'>Error: {}}}}}}}}}}}}}}}}}}}}}}}results[]],,'with_optimization'].get()))))))))))))))'error', 'Unknown error')}</td></tr>"
                    continue
                    
                    without_opt = results[]],,"without_optimization"].get()))))))))))))))"inference_time_ms", "N/A")
                    with_opt = results[]],,"with_optimization"].get()))))))))))))))"inference_time_ms", "N/A")
                    
                    improvement_html = "N/A"
                    if "improvement" in results and "inference_time_improvement_pct" in results[]],,"improvement"]:
                        improvement_val = results[]],,"improvement"][]],,"inference_time_improvement_pct"]
                        
                        # Classify improvement for styling
                        if improvement_val > 20:
                            improvement_class = "improvement-high"
                        elif improvement_val > 5:
                            improvement_class = "improvement-medium"
                        else:
                            improvement_class = "improvement-low"
                            
                            improvement_html = f'<span class="{}}}}}}}}}}}}}}}}}}}}}}}improvement_class}">{}}}}}}}}}}}}}}}}}}}}}}}improvement_val:.1f}%</span>'
                    
                            simulated = '<span class="warning">Yes</span>' if results[]],,"with_optimization"].get()))))))))))))))"simulated", False) else '<span class="success">No</span>'
                    
                            html += f"<tr><td>{}}}}}}}}}}}}}}}}}}}}}}}browser}</td><td>{}}}}}}}}}}}}}}}}}}}}}}}batch}</td><td>{}}}}}}}}}}}}}}}}}}}}}}}without_opt}</td><td>{}}}}}}}}}}}}}}}}}}}}}}}with_opt}</td><td>{}}}}}}}}}}}}}}}}}}}}}}}improvement_html}</td><td>{}}}}}}}}}}}}}}}}}}}}}}}simulated}</td></tr>"
                
                            html += "</table>"
        
        # Multimodal loading optimization results:
        if self.results[]],,"multimodal_loading"]:
            html += "<h2>Multimodal Model Parallel Loading Optimization</h2>"
            
            for model, browser_results in self.results[]],,"multimodal_loading"].items()))))))))))))))):
                html += f"<h3>Model: {}}}}}}}}}}}}}}}}}}}}}}}model}</h3>"
                html += "<table>"
                html += "<tr><th>Browser</th><th>Without Optimization ()))))))))))))))ms)</th><th>With Optimization ()))))))))))))))ms)</th><th>Improvement</th><th>Simulated</th></tr>"
                
                for browser, results in browser_results.items()))))))))))))))):
                    if results[]],,"with_optimization"].get()))))))))))))))"error") or results[]],,"without_optimization"].get()))))))))))))))"error"):
                        html += f"<tr><td>{}}}}}}}}}}}}}}}}}}}}}}}browser}</td><td colspan='4'>Error: {}}}}}}}}}}}}}}}}}}}}}}}results[]],,'with_optimization'].get()))))))))))))))'error', 'Unknown error')}</td></tr>"
                    continue
                    
                    without_opt = results[]],,"without_optimization"].get()))))))))))))))"loading_time_ms", "N/A")
                    with_opt = results[]],,"with_optimization"].get()))))))))))))))"loading_time_ms", "N/A")
                    
                    improvement_html = "N/A"
                    if "improvement" in results and "loading_time_improvement_pct" in results[]],,"improvement"]:
                        improvement_val = results[]],,"improvement"][]],,"loading_time_improvement_pct"]
                        
                        # Classify improvement for styling
                        if improvement_val > 30:
                            improvement_class = "improvement-high"
                        elif improvement_val > 10:
                            improvement_class = "improvement-medium"
                        else:
                            improvement_class = "improvement-low"
                            
                            improvement_html = f'<span class="{}}}}}}}}}}}}}}}}}}}}}}}improvement_class}">{}}}}}}}}}}}}}}}}}}}}}}}improvement_val:.1f}%</span>'
                    
                            simulated = '<span class="warning">Yes</span>' if results[]],,"with_optimization"].get()))))))))))))))"simulated", False) else '<span class="success">No</span>'
                    
                            html += f"<tr><td>{}}}}}}}}}}}}}}}}}}}}}}}browser}</td><td>{}}}}}}}}}}}}}}}}}}}}}}}without_opt}</td><td>{}}}}}}}}}}}}}}}}}}}}}}}with_opt}</td><td>{}}}}}}}}}}}}}}}}}}}}}}}improvement_html}</td><td>{}}}}}}}}}}}}}}}}}}}}}}}simulated}</td></tr>"
                
                            html += "</table>"
        
        # Shader precompilation optimization results:
        if self.results[]],,"shader_precompilation"]:
            html += "<h2>Shader Precompilation Optimization</h2>"
            
            for model, browser_results in self.results[]],,"shader_precompilation"].items()))))))))))))))):
                html += f"<h3>Model: {}}}}}}}}}}}}}}}}}}}}}}}model}</h3>"
                html += "<table>"
                html += "<tr><th>Browser</th><th>First Inference Without ()))))))))))))))ms)</th><th>First Inference With ()))))))))))))))ms)</th><th>Improvement</th><th>Simulated</th></tr>"
                
                for browser, results in browser_results.items()))))))))))))))):
                    if results[]],,"with_optimization"].get()))))))))))))))"error") or results[]],,"without_optimization"].get()))))))))))))))"error"):
                        html += f"<tr><td>{}}}}}}}}}}}}}}}}}}}}}}}browser}</td><td colspan='4'>Error: {}}}}}}}}}}}}}}}}}}}}}}}results[]],,'with_optimization'].get()))))))))))))))'error', 'Unknown error')}</td></tr>"
                    continue
                    
                    without_opt = results[]],,"without_optimization"].get()))))))))))))))"first_inference_time_ms", "N/A")
                    with_opt = results[]],,"with_optimization"].get()))))))))))))))"first_inference_time_ms", "N/A")
                    
                    improvement_html = "N/A"
                    if "improvement" in results and "first_inference_improvement_pct" in results[]],,"improvement"]:
                        improvement_val = results[]],,"improvement"][]],,"first_inference_improvement_pct"]
                        
                        # Classify improvement for styling
                        if improvement_val > 30:
                            improvement_class = "improvement-high"
                        elif improvement_val > 10:
                            improvement_class = "improvement-medium"
                        else:
                            improvement_class = "improvement-low"
                            
                            improvement_html = f'<span class="{}}}}}}}}}}}}}}}}}}}}}}}improvement_class}">{}}}}}}}}}}}}}}}}}}}}}}}improvement_val:.1f}%</span>'
                    
                            simulated = '<span class="warning">Yes</span>' if results[]],,"with_optimization"].get()))))))))))))))"simulated", False) else '<span class="success">No</span>'
                    
                            html += f"<tr><td>{}}}}}}}}}}}}}}}}}}}}}}}browser}</td><td>{}}}}}}}}}}}}}}}}}}}}}}}without_opt}</td><td>{}}}}}}}}}}}}}}}}}}}}}}}with_opt}</td><td>{}}}}}}}}}}}}}}}}}}}}}}}improvement_html}</td><td>{}}}}}}}}}}}}}}}}}}}}}}}simulated}</td></tr>"
                
                            html += "</table>"
        
        # Recommendations
                            html += "<h2>Browser-Specific Recommendations</h2>"
        
        # Audio model recommendations
                            html += "<div class='recommendation'>"
                            html += "<h3>Audio Models ()))))))))))))))Whisper, Wav2Vec2, CLAP)</h3>"
        
                            firefox_better = False
                            recommendation_details = []],,]
        :
        for model in AUDIO_MODELS:
            if ()))))))))))))))model in self.results[]],,"audio_optimization"] and 
                "firefox_batch1" in self.results[]],,"audio_optimization"][]],,model] and ::
                "chrome_batch1" in self.results[]],,"audio_optimization"][]],,model]):
                
                    firefox_results = self.results[]],,"audio_optimization"][]],,model][]],,"firefox_batch1"]
                    chrome_results = self.results[]],,"audio_optimization"][]],,model][]],,"chrome_batch1"]
                
                    if ()))))))))))))))not firefox_results[]],,"with_optimization"].get()))))))))))))))"error") and
                    not chrome_results[]],,"with_optimization"].get()))))))))))))))"error") and
                    "inference_time_ms" in firefox_results[]],,"with_optimization"] and::
                    "inference_time_ms" in chrome_results[]],,"with_optimization"]):
                    
                        firefox_time = firefox_results[]],,"with_optimization"][]],,"inference_time_ms"]
                        chrome_time = chrome_results[]],,"with_optimization"][]],,"inference_time_ms"]
                    
                    if firefox_time < chrome_time:
                        firefox_better = True
                        diff_pct = ()))))))))))))))chrome_time - firefox_time) / chrome_time * 100
                        recommendation_details.append()))))))))))))))f"<li>For {}}}}}}}}}}}}}}}}}}}}}}}model}, Firefox is <strong>{}}}}}}}}}}}}}}}}}}}}}}}diff_pct:.1f}%</strong> faster than Chrome</li>")
        
        if firefox_better:
            html += "<p><strong>Recommendation:</strong> For audio models like Whisper and Wav2Vec2, Firefox with compute shader optimization is the preferred browser</p>"
            html += "<ul>"
            for detail in recommendation_details:
                html += detail
                html += "</ul>"
                html += "<p>Enable compute shader optimization when running audio models</p>"
        else:
            html += "<p>No clear browser preference for audio models was detected in this test run</p>"
        
            html += "</div>"
        
        # Multimodal model recommendations
            html += "<div class='recommendation'>"
            html += "<h3>Multimodal Models ()))))))))))))))CLIP, LLaVA)</h3>"
        
            best_browser = None
            best_improvement = 0
            improvement_details = []],,]
        
        for model in MULTIMODAL_MODELS:
            if model in self.results[]],,"multimodal_loading"]:
                for browser, results in self.results[]],,"multimodal_loading"][]],,model].items()))))))))))))))):
                    if ()))))))))))))))"improvement" in results and ::
                        "loading_time_improvement_pct" in results[]],,"improvement"]):
                        
                            improvement_val = results[]],,"improvement"][]],,"loading_time_improvement_pct"]
                            improvement_details.append()))))))))))))))f"<li>For {}}}}}}}}}}}}}}}}}}}}}}}model} on {}}}}}}}}}}}}}}}}}}}}}}}browser}: <strong>{}}}}}}}}}}}}}}}}}}}}}}}improvement_val:.1f}%</strong> loading time reduction</li>")
                        
                        if improvement_val > best_improvement:
                            best_improvement = improvement_val
                            best_browser = browser
        
        if best_browser:
            html += f"<p><strong>Recommendation:</strong> For multimodal models, {}}}}}}}}}}}}}}}}}}}}}}}best_browser} with parallel loading enabled shows the best performance</p>"
            html += "<ul>"
            for detail in improvement_details:
                html += detail
                html += "</ul>"
                html += f"<p>Parallel loading reduces model loading time by up to <strong>{}}}}}}}}}}}}}}}}}}}}}}}best_improvement:.1f}%</strong></p>"
        else:
            html += "<p>No clear browser preference for multimodal models was detected in this test run</p>"
        
            html += "</div>"
        
        # Text/Vision model recommendations
            html += "<div class='recommendation'>"
            html += "<h3>Text and Vision Models ()))))))))))))))BERT, ViT)</h3>"
        
            best_browser = None
            best_improvement = 0
            improvement_details = []],,]
        
        for model in TEXT_VISION_MODELS:
            if model in self.results[]],,"shader_precompilation"]:
                for browser, results in self.results[]],,"shader_precompilation"][]],,model].items()))))))))))))))):
                    if ()))))))))))))))"improvement" in results and ::
                        "first_inference_improvement_pct" in results[]],,"improvement"]):
                        
                            improvement_val = results[]],,"improvement"][]],,"first_inference_improvement_pct"]
                            improvement_details.append()))))))))))))))f"<li>For {}}}}}}}}}}}}}}}}}}}}}}}model} on {}}}}}}}}}}}}}}}}}}}}}}}browser}: <strong>{}}}}}}}}}}}}}}}}}}}}}}}improvement_val:.1f}%</strong> first inference time reduction</li>")
                        
                        if improvement_val > best_improvement:
                            best_improvement = improvement_val
                            best_browser = browser
        
        if best_browser:
            html += f"<p><strong>Recommendation:</strong> For text and vision models, {}}}}}}}}}}}}}}}}}}}}}}}best_browser} with shader precompilation shows the best first inference performance</p>"
            html += "<ul>"
            for detail in improvement_details:
                html += detail
                html += "</ul>"
                html += f"<p>Shader precompilation reduces first inference time by up to <strong>{}}}}}}}}}}}}}}}}}}}}}}}best_improvement:.1f}%</strong></p>"
        else:
            html += "<p>No clear browser preference for text/vision models was detected in this test run</p>"
        
            html += "</div>"
        
            html += "</body></html>"
        
                return html

def parse_args()))))))))))))))):
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()))))))))))))))description="WebGPU cross-browser comparison test suite.")
    parser.add_argument()))))))))))))))"--browser", type=str, help="Browser to test ()))))))))))))))chrome, firefox, edge, safari)")
    parser.add_argument()))))))))))))))"--model", type=str, help="Model to test")
    parser.add_argument()))))))))))))))"--models", type=str, nargs='+', help="List of models to test")
    parser.add_argument()))))))))))))))"--all-browsers", action="store_true", help="Test all supported browsers")
    parser.add_argument()))))))))))))))"--batch-sizes", type=int, nargs='+', help="List of batch sizes to test")
    parser.add_argument()))))))))))))))"--enable-compute-shaders", action="store_true", help="Enable compute shader optimizations")
    parser.add_argument()))))))))))))))"--enable-parallel-loading", action="store_true", help="Enable parallel loading optimizations")
    parser.add_argument()))))))))))))))"--enable-shader-precompile", action="store_true", help="Enable shader precompilation")
    parser.add_argument()))))))))))))))"--all-optimizations", action="store_true", help="Enable all optimizations")
    parser.add_argument()))))))))))))))"--timeout", type=int, default=DEFAULT_TIMEOUT, help="Timeout in seconds for each test")
    parser.add_argument()))))))))))))))"--output-dir", type=str, default="./webgpu_browser_comparison_results", help="Directory to store test results")
    parser.add_argument()))))))))))))))"--db-path", type=str, help="Path to benchmark database")
    parser.add_argument()))))))))))))))"--report", type=str, choices=[]],,"markdown", "html"], help="Generate report in specified format")
    parser.add_argument()))))))))))))))"--report-output", type=str, help="Path to save the report")
    
                return parser.parse_args())))))))))))))))

def main()))))))))))))))):
    """Main function."""
    args = parse_args())))))))))))))))
    
    # Determine browsers to test
    if args.all_browsers:
        browsers = SUPPORTED_BROWSERS
    elif args.browser:
        browsers = []],,args.browser]
    else:
        browsers = []],,"chrome", "firefox"],  # Default browsers
    
    # Determine models to test
    if args.models and "all" in args.models:
        models = AUDIO_MODELS + MULTIMODAL_MODELS + TEXT_VISION_MODELS
    elif args.models:
        models = args.models
    elif args.model:
        models = []],,args.model]
    else:
        models = []],,"whisper-tiny"],  # Default model
    
    # Create and run the test suite
        suite = WebGPUBrowserComparison()))))))))))))))
        browsers=browsers,
        models=models,
        batch_sizes=args.batch_sizes,
        enable_compute_shaders=args.enable_compute_shaders,
        enable_parallel_loading=args.enable_parallel_loading,
        enable_shader_precompile=args.enable_shader_precompile,
        all_optimizations=args.all_optimizations,
        timeout=args.timeout,
        output_dir=args.output_dir,
        db_path=args.db_path
        )
    
        suite.run_tests())))))))))))))))
    
    # Generate report if requested:
    if args.report:
        report = suite.generate_report()))))))))))))))args.report)
        
        if args.report_output:
            with open()))))))))))))))args.report_output, 'w') as f:
                f.write()))))))))))))))report)
                print()))))))))))))))f"Report saved to {}}}}}}}}}}}}}}}}}}}}}}}args.report_output}")
        else:
            print()))))))))))))))report)

if __name__ == "__main__":
    main())))))))))))))))