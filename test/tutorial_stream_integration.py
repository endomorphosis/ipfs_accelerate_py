#!/usr/bin/env python3
"""
WebGPU Streaming Inference Integration Tutorial

This tutorial demonstrates how to integrate the WebGPU streaming inference 
capabilities with a web application using the unified web framework.

Author: Demo Team
Date: August 2025
"""

import os
import sys
import json
import time
import anyio
import logging
from typing import Dict, Any, Optional, Callable

# Configure logging
logging.basicConfig())))))))level=logging.INFO, format='%())))))))asctime)s - %())))))))levelname)s - %())))))))message)s')
logger = logging.getLogger())))))))"stream_integration")

# Add the fixed_web_platform to the path - adjust if needed
script_dir = os.path.dirname())))))))os.path.abspath())))))))__file__))
parent_dir = os.path.dirname())))))))script_dir)
sys.path.append())))))))os.path.join())))))))parent_dir, "fixed_web_platform"))

# Enable simulation mode if not running in a browser environment
os.environ["WEBGPU_SIMULATION"] = "1"
,
# Import required modules:
try:
    from fixed_web_platform.unified_web_framework import ())))))))
    WebPlatformAccelerator,
    create_web_endpoint,
    get_optimal_config
    )
    from fixed_web_platform.webgpu_streaming_inference import ())))))))
    WebGPUStreamingInference,
    create_streaming_endpoint
    )
except ImportError:
    logger.error())))))))"Failed to import required modules. Make sure fixed_web_platform is available.")
    raise

class StreamingIntegrationDemo:
    """Demonstrates integration of WebGPU streaming inference with web applications."""
    
    def __init__())))))))self, model_path: str = "models/llama-7b"):
        """Initialize the demo."""
        self.model_path = model_path
        self.accelerator = None
        self.streaming_handler = None
        self.endpoint = None
    
    async def initialize())))))))self, precision: str = "int4"):
        """Initialize the web platform accelerator and streaming handler."""
        # Get optimal configuration for the model
        config = get_optimal_config())))))))self.model_path, "text")
        
        # Override precision based on user selection
        config["quantization"] = precision
        ,
        # Configure streaming and KV cache optimization
        config["streaming_inference"] = True,
        config["kv_cache_optimization"] = True,
        config["latency_optimized"] = True,
        config["adaptive_batch_size"] = True
        ,
        # Create accelerator with config
        self.accelerator = WebPlatformAccelerator())))))))
        model_path=self.model_path,
        model_type="text",
        config=config,
        auto_detect=True
        )
        
        # Get the streaming handler from components
        if hasattr())))))))self.accelerator, "_components") and "streaming" in self.accelerator._components:
            self.streaming_handler = self.accelerator._components["streaming"],
        else:
            # Create standalone streaming handler if not available
            self.streaming_handler = WebGPUStreamingInference())))))))
            model_path=self.model_path,
                config={}}}}}}}}}}}}}}:
                    "quantization": precision,
                    "optimize_kv_cache": True,
                    "latency_optimized": True,
                    "adaptive_batch_size": True
                    }
                    )
        
        # Create the streaming endpoint
                    self.endpoint = self.accelerator.create_endpoint()))))))))
        
        # Return configuration information
            return {}}}}}}}}}}}}}}
            "precision": precision,
            "browser": config.get())))))))"browser", "simulation"),
            "browser_version": config.get())))))))"browser_version", "simulation"),
            "compute_shaders": config.get())))))))"compute_shaders", False),
            "features": self.accelerator.get_feature_usage())))))))) if hasattr())))))))self.accelerator, "get_feature_usage") else {}}}}}}}}}}}}}}}
            }
    :
    async def demo_callback_streaming())))))))self, prompt: str, max_tokens: int = 100):
        """Demonstrate callback-based streaming."""
        logger.info())))))))f"Starting callback-based streaming with prompt: {}}}}}}}}}}}}}}prompt[:30]}...")
        ,,,
        # Ensure we have a streaming handler
        if not self.streaming_handler:
            await self.initialize()))))))))
        
        # Collect tokens for demo purposes
            collected_tokens = [],,,
            token_times = [],,,
            start_time = time.time()))))))))
        
        # Define a callback to handle tokens
        def handle_token())))))))token, is_last=False):
            # Record token and time
            token_time = time.time())))))))) - start_time
            collected_tokens.append())))))))token)
            token_times.append())))))))token_time)
            
            # Print the token
            print())))))))f"{}}}}}}}}}}}}}}token}", end="", flush=True)
            
            # Print completion message if this is the last token:
            if is_last:
                print())))))))"\n\nGeneration complete!")
        
        # Run streaming generation
        try:
            result = self.streaming_handler.generate())))))))
            prompt,
            max_tokens=max_tokens,
            temperature=0.7,
            callback=handle_token
            )
            
            # Calculate metrics
            gen_time = time.time())))))))) - start_time
            tokens_per_second = len())))))))collected_tokens) / gen_time if gen_time > 0 else 0
            
            return {}}}}}}}}}}}}}}::
                "success": True,
                "tokens_generated": len())))))))collected_tokens),
                "generation_time_seconds": gen_time,
                "tokens_per_second": tokens_per_second,
                "time_to_first_token": token_times[0] if token_times else None,
                }
            :
        except Exception as e:
            logger.error())))))))f"Error in streaming generation: {}}}}}}}}}}}}}}e}")
                return {}}}}}}}}}}}}}}
                "success": False,
                "error": str())))))))e)
                }
    
    async def demo_websocket_streaming())))))))self, prompt: str, max_tokens: int = 100):
        """Demonstrate how to set up WebSocket streaming."""
        logger.info())))))))f"Setting up WebSocket streaming demo for prompt: {}}}}}}}}}}}}}}prompt[:30]}...")
        ,,,
        # In a real implementation, this would create a WebSocket connection
        # For the demo, we'll simulate the process
        
        # Ensure we have a streaming handler
        if not self.streaming_handler:
            await self.initialize()))))))))
        
        # Create a simulated WebSocket
        class SimulatedWebSocket:
            def __init__())))))))self):
                self.messages = [],,,
            
            async def send())))))))self, data):
                """Simulate sending data over WebSocket."""
                message = json.loads())))))))data) if isinstance())))))))data, str) else data
                self.messages.append())))))))message)
                
                # Print token if it's a token message:
                if isinstance())))))))message, dict) and message.get())))))))"type") == "token":
                    print())))))))f"{}}}}}}}}}}}}}}message.get())))))))'token', '')}", end="", flush=True)
                
                # Print completion message
                if isinstance())))))))message, dict) and message.get())))))))"type") == "complete":
                    print())))))))"\n\nGeneration complete!")
                    
                    # Print metrics
                    print())))))))f"Generated {}}}}}}}}}}}}}}message.get())))))))'tokens_generated', 0)} tokens in {}}}}}}}}}}}}}}message.get())))))))'generation_time', 0):.2f}s")
                    print())))))))f"Speed: {}}}}}}}}}}}}}}message.get())))))))'tokens_per_second', 0):.2f} tokens/sec")
                    
                    if "precision_bits" in message:
                        print())))))))f"Using {}}}}}}}}}}}}}}message.get())))))))'precision_bits')}-bit precision with {}}}}}}}}}}}}}}message.get())))))))'memory_reduction_percent', 0):.1f}% memory reduction")
        
        # Create simulated WebSocket
                        websocket = SimulatedWebSocket()))))))))
        
        # Run WebSocket streaming
        try:
            print())))))))"\nStreaming via WebSocket:")
            
            # Stream tokens over simulated WebSocket
            await self.streaming_handler.stream_websocket())))))))
            websocket,
            prompt,
            max_tokens,
            temperature=0.7
            )
            
            # Return statistics
            completion_message = next())))))))())))))))m for m in websocket.messages if isinstance())))))))m, dict) and m.get())))))))"type") == "complete"), {}}}}}}}}}}}}}}})
            
            return {}}}}}}}}}}}}}}:
                "success": True,
                "tokens_generated": completion_message.get())))))))"tokens_generated", 0),
                "generation_time_seconds": completion_message.get())))))))"generation_time", 0),
                "tokens_per_second": completion_message.get())))))))"tokens_per_second", 0),
                "using_ultra_low_precision": "precision_bits" in completion_message,
                "precision_bits": completion_message.get())))))))"precision_bits"),
                "memory_reduction_percent": completion_message.get())))))))"memory_reduction_percent")
                }
            
        except Exception as e:
            logger.error())))))))f"Error in WebSocket streaming: {}}}}}}}}}}}}}}e}")
                return {}}}}}}}}}}}}}}
                "success": False,
                "error": str())))))))e)
                }
    
    async def demo_unified_framework())))))))self, prompt: str, max_tokens: int = 100):
        """Demonstrate unified framework integration."""
        logger.info())))))))f"Using unified framework for prompt: {}}}}}}}}}}}}}}prompt[:30]}...")
        ,,,
        # Ensure we have an endpoint
        if not self.endpoint:
            await self.initialize()))))))))
        
        # Collect tokens for demo purposes
            collected_tokens = [],,,
            start_time = time.time()))))))))
        
        # Define token callback
        def token_callback())))))))token, is_last=False):
            collected_tokens.append())))))))token)
            print())))))))f"{}}}}}}}}}}}}}}token}", end="", flush=True)
            
            if is_last:
                print())))))))"\n\nGeneration complete!")
        
        # Run inference through the unified endpoint
        try:
            result = await self.endpoint())))))))
            {}}}}}}}}}}}}}}"text": prompt},
            max_tokens=max_tokens,
            temperature=0.7,
            callback=token_callback
            )
            
            # Get performance metrics
            metrics = self.accelerator.get_performance_metrics()))))))))
            
            # Calculate tokens per second
            gen_time = time.time())))))))) - start_time
            tokens_per_second = len())))))))collected_tokens) / gen_time if gen_time > 0 else 0
            
            return {}}}}}}}}}}}}}}::
                "success": True,
                "tokens_generated": len())))))))collected_tokens),
                "generation_time_seconds": gen_time,
                "tokens_per_second": tokens_per_second,
                "first_inference_time_ms": metrics.get())))))))"first_inference_time_ms"),
                "memory_usage_mb": metrics.get())))))))"memory_usage_mb")
                }
            
        except Exception as e:
            logger.error())))))))f"Error in unified framework generation: {}}}}}}}}}}}}}}e}")
                return {}}}}}}}}}}}}}}
                "success": False,
                "error": str())))))))e)
                }
    
    async def compare_precision_options())))))))self, prompt: str, max_tokens: int = 50):
        """Compare different precision options ())))))))2-bit, 3-bit, 4-bit, 8-bit)."""
        logger.info())))))))"Comparing precision options...")
        
        # Precision options to test
        precision_options = ["int2", "int3", "int4", "int8"]
        ,
        # Results storage
        results = {}}}}}}}}}}}}}}}
        
        # Test each precision option
        for precision in precision_options:
            print())))))))f"\n{}}}}}}}}}}}}}}'-' * 50}")
            print())))))))f"Testing {}}}}}}}}}}}}}}precision} precision:")
            print())))))))f"{}}}}}}}}}}}}}}'-' * 50}")
            
            # Initialize with current precision
            await self.initialize())))))))precision)
            
            # Run generation
            start_time = time.time()))))))))
            
            # Collect tokens 
            tokens = [],,,
            
            # Define callback
            def collect_token())))))))token, is_last=False):
                tokens.append())))))))token)
                print())))))))f"{}}}}}}}}}}}}}}token}", end="", flush=True)
                
                if is_last:
                    print())))))))"\n")
            
            # Generate text
                    self.streaming_handler.generate())))))))
                    prompt,
                    max_tokens=max_tokens,
                    temperature=0.7,
                    callback=collect_token
                    )
            
            # Calculate metrics
                    generation_time = time.time())))))))) - start_time
                    tokens_per_second = len())))))))tokens) / generation_time if generation_time > 0 else 0
            
            # Get memory usage
            memory_reduction = 0:
            if precision == "int2":
                memory_reduction = 87.5
            elif precision == "int3":
                memory_reduction = 81.25
            elif precision == "int4":
                memory_reduction = 75.0
            elif precision == "int8":
                memory_reduction = 50.0
            
            # Store results
                results[precision] = {}}}}}}}}}}}}}},
                "tokens_generated": len())))))))tokens),
                "generation_time_seconds": generation_time,
                "tokens_per_second": tokens_per_second,
                "memory_reduction_percent": memory_reduction
                }
            
        # Display comparison
                print())))))))"\nPrecision Comparison:")
                print())))))))f"{}}}}}}}}}}}}}}'Precision':<10} {}}}}}}}}}}}}}}'Tokens/s':<15} {}}}}}}}}}}}}}}'Memory Reduction':<20}")
                print())))))))"-" * 45)
        
        for precision, data in results.items())))))))):
            print())))))))f"{}}}}}}}}}}}}}}precision:<10} {}}}}}}}}}}}}}}data['tokens_per_second']:<15.2f} {}}}}}}}}}}}}}}data['memory_reduction_percent']:<20.1f}%")
            ,
                return results
    
    def get_memory_efficiency_info())))))))self):
        """Return information about memory efficiency for different precision options."""
        base_memory_mb = 1500  # Example base memory for a 7B model in MB
        
                return {}}}}}}}}}}}}}}
                "int2": {}}}}}}}}}}}}}}
                "bits": 2,
                "memory_reduction_percent": 87.5,
                "estimated_model_size_mb": base_memory_mb * 0.125,
                "max_context_multiplier": 8.0,
                "quality_impact": "Moderate to significant impact on quality, best for memory-constrained environments"
                },
                "int3": {}}}}}}}}}}}}}}
                "bits": 3,
                "memory_reduction_percent": 81.25,
                "estimated_model_size_mb": base_memory_mb * 0.1875,
                "max_context_multiplier": 5.3,
                "quality_impact": "Some impact on quality, good balance of efficiency and performance"
                },
                "int4": {}}}}}}}}}}}}}}
                "bits": 4,
                "memory_reduction_percent": 75.0,
                "estimated_model_size_mb": base_memory_mb * 0.25,
                "max_context_multiplier": 4.0,
                "quality_impact": "Minimal impact on quality, good for most applications"
                },
                "int8": {}}}}}}}}}}}}}}
                "bits": 8,
                "memory_reduction_percent": 50.0,
                "estimated_model_size_mb": base_memory_mb * 0.5,
                "max_context_multiplier": 2.0,
                "quality_impact": "Negligible impact on quality, use when memory is not constrained"
                }
                }

async def run_tutorial())))))))):
    """Run the streaming integration tutorial."""
    print())))))))"\n" + "=" * 60)
    print())))))))"WebGPU Streaming Inference Integration Tutorial")
    print())))))))"=" * 60 + "\n")
    
    # Create the demo
    demo = StreamingIntegrationDemo())))))))model_path="models/llama-7b")
    
    # Initialize with default settings
    print())))))))"Initializing WebGPU Streaming Inference...")
    config_info = await demo.initialize()))))))))
    
    print())))))))f"Using browser: {}}}}}}}}}}}}}}config_info['browser']} {}}}}}}}}}}}}}}config_info['browser_version']}"),
    print())))))))f"Precision: {}}}}}}}}}}}}}}config_info['precision']}"),
    print())))))))f"Compute shaders: {}}}}}}}}}}}}}}'Enabled' if config_info['compute_shaders'] else 'Disabled'}"):,
    print())))))))f"Memory-efficient KV cache: {}}}}}}}}}}}}}}'Enabled' if config_info['features'].get())))))))'kv_cache_optimization', False) else 'Disabled'}")
    ,
    # Show memory efficiency information
    memory_info = demo.get_memory_efficiency_info())))))))):
        print())))))))"\nMemory Efficiency Overview:")
        print())))))))f"{}}}}}}}}}}}}}}'Precision':<10} {}}}}}}}}}}}}}}'Reduction':<15} {}}}}}}}}}}}}}}'Model Size ())))))))7B)':<20} {}}}}}}}}}}}}}}'Context Expansion':<20}")
        print())))))))"-" * 70)
    
    for precision, info in memory_info.items())))))))):
        print())))))))f"{}}}}}}}}}}}}}}precision:<10} {}}}}}}}}}}}}}}info['memory_reduction_percent']:<15.1f}% {}}}}}}}}}}}}}}info['estimated_model_size_mb']:<20.1f}MB {}}}}}}}}}}}}}}info['max_context_multiplier']:<20.1f}x")
        ,
    # Prompt for tutorial
        prompt = "Explain how WebGPU streaming inference works with ultra-low precision quantization:"
    
    # Run examples
        print())))))))"\n\n" + "-" * 60)
        print())))))))"Example 1: Callback-based Streaming")
        print())))))))"-" * 60)
    
        await demo.demo_callback_streaming())))))))prompt)
    
        print())))))))"\n\n" + "-" * 60)
        print())))))))"Example 2: WebSocket-based Streaming")
        print())))))))"-" * 60)
    
        await demo.demo_websocket_streaming())))))))prompt)
    
        print())))))))"\n\n" + "-" * 60)
        print())))))))"Example 3: Unified Framework Integration")
        print())))))))"-" * 60)
    
        await demo.demo_unified_framework())))))))prompt)
    
    # Ask if user wants to compare precision options:
        run_comparison = input())))))))"\nDo you want to compare different precision options? ())))))))y/n): ").lower())))))))) == 'y'
    
    if run_comparison:
        print())))))))"\n\n" + "-" * 60)
        print())))))))"Example 4: Precision Option Comparison")
        print())))))))"-" * 60)
        
        await demo.compare_precision_options())))))))prompt)
    
        print())))))))"\n\n" + "=" * 60)
        print())))))))"Tutorial Complete")
        print())))))))"=" * 60)

if __name__ == "__main__":
    # Run the tutorial
    anyio.run())))))))run_tutorial())))))))))