#!/usr/bin/env python3
"""
WebNN/WebGPU with IPFS Acceleration Demo

This demo showcases the WebNN/WebGPU integration with IPFS acceleration
for efficient model inference in web browsers with IPFS-based content delivery.

Usage:
    python demo_webnn_webgpu.py --model bert-base-uncased --platform webgpu --browser firefox
"""

import os
import sys
import json
import argparse
import asyncio
from typing import Dict, Any, Optional
from pathlib import Path

# Add parent directory to path for module imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Try to import the webnn_webgpu_integration module
try:
    from ipfs_accelerate_py.webnn_webgpu_integration import accelerate_with_browser
    from ipfs_accelerate_py.webnn_webgpu_integration import get_accelerator
except ImportError:
    print("Error: Could not import WebNN/WebGPU integration module.")
    print("Make sure ipfs_accelerate_py is properly installed.")
    sys.exit(1)

def create_test_inputs(model_name: str, model_type: Optional[str] = None) -> Dict[str, Any]:
    """
    Create test inputs for various model types.
    
    Args:
        model_name: Name of the model
        model_type: Optional model type
        
    Returns:
        Dictionary with appropriate test inputs
    """
    model_name_lower = model_name.lower()
    
    # Determine model type if not provided
    if not model_type:
        if any(x in model_name_lower for x in ["bert", "roberta", "mpnet", "minilm"]):
            model_type = "text_embedding"
        elif any(x in model_name_lower for x in ["t5", "mt5", "bart"]):
            model_type = "text2text"
        elif any(x in model_name_lower for x in ["llama", "gpt", "qwen", "phi", "mistral"]):
            model_type = "text_generation"
        elif any(x in model_name_lower for x in ["whisper", "wav2vec", "clap", "audio"]):
            model_type = "audio"
        elif any(x in model_name_lower for x in ["vit", "clip", "detr", "image"]):
            model_type = "vision"
        elif any(x in model_name_lower for x in ["llava", "xclip", "blip"]):
            model_type = "multimodal"
        else:
            model_type = "text"
    
    # Create appropriate inputs based on model type
    if model_type == "text_embedding":
        return {
            "input_ids": [101, 2023, 2003, 1037, 3231, 102],
            "attention_mask": [1, 1, 1, 1, 1, 1]
        }
    elif model_type == "text_generation":
        return {
            "inputs": "This is a test for language model inference."
        }
    elif model_type == "text2text":
        return {
            "inputs": "Translate this to French: Hello, how are you?"
        }
    elif model_type == "vision":
        # Mock image as a 224x224x3 tensor with all values being 0.5
        return {
            "pixel_values": [[[0.5 for _ in range(3)] for _ in range(224)] for _ in range(224)]
        }
    elif model_type == "audio":
        # Mock audio features (spectrogram-like)
        return {
            "input_features": [[0.1 for _ in range(80)] for _ in range(3000)]
        }
    elif model_type == "multimodal":
        # Mock combined image and text
        return {
            "pixel_values": [[[0.5 for _ in range(3)] for _ in range(224)] for _ in range(224)],
            "input_ids": [101, 2023, 2003, 1037, 3231, 102],
            "attention_mask": [1, 1, 1, 1, 1, 1]
        }
    else:
        return {
            "inputs": "This is a generic test input."
        }

async def run_benchmark(
    model_name: str,
    model_type: Optional[str] = None,
    platform: str = "webgpu",
    browser: Optional[str] = None,
    precision: int = 16,
    mixed_precision: bool = False,
    use_real_browser: bool = False,
    num_runs: int = 3
) -> Dict[str, Any]:
    """
    Run a benchmark of the model with WebNN/WebGPU acceleration.
    
    Args:
        model_name: Name of the model
        model_type: Type of model
        platform: Platform to use (webnn or webgpu)
        browser: Browser to use
        precision: Precision to use
        mixed_precision: Whether to use mixed precision
        use_real_browser: Whether to use a real browser
        num_runs: Number of benchmark runs
        
    Returns:
        Dictionary with benchmark results
    """
    print(f"\nRunning benchmark for {model_name} with {platform}")
    print(f"Browser: {browser or 'auto'}, Precision: {precision}, Mixed precision: {mixed_precision}")
    print(f"Real browser: {use_real_browser}, Runs: {num_runs}")
    
    # Create inputs
    inputs = create_test_inputs(model_name, model_type)
    
    # Get accelerator
    accelerator = get_accelerator(enable_ipfs=True, headless=not use_real_browser)
    
    results = []
    for i in range(num_runs):
        print(f"\nRun {i+1}/{num_runs}...")
        
        # Run inference
        try:
            result = await accelerator.accelerate_with_browser(
                model_name=model_name,
                inputs=inputs,
                model_type=model_type,
                platform=platform,
                browser=browser,
                precision=precision,
                mixed_precision=mixed_precision,
                use_real_browser=use_real_browser
            )
            
            results.append(result)
            
            # Print performance metrics
            print(f"Status: {result.get('status')}")
            print(f"Inference time: {result.get('inference_time', 0):.3f}s")
            print(f"Latency: {result.get('latency_ms', 0):.2f}ms")
            print(f"Throughput: {result.get('throughput_items_per_sec', 0):.2f} items/s")
            
            if result.get('ipfs_accelerated'):
                print(f"IPFS acceleration: {'Hit' if result.get('ipfs_cache_hit') else 'Miss'}")
                print(f"CID: {result.get('cid')}")
            
            # Print a small preview of the output
            output = result.get("output", {})
            if isinstance(output, dict):
                for k, v in output.items():
                    if isinstance(v, list) and len(v) > 5:
                        print(f"  {k}: [...] (list with {len(v)} items)")
                    else:
                        print(f"  {k}: {v}")
            else:
                print(f"  {output}")
                
        except Exception as e:
            print(f"Error during inference: {e}")
            
    # Calculate aggregate statistics
    if results:
        avg_latency = sum(r.get('latency_ms', 0) for r in results) / len(results)
        avg_throughput = sum(r.get('throughput_items_per_sec', 0) for r in results) / len(results)
        
        print("\nBenchmark summary:")
        print(f"Average latency: {avg_latency:.2f}ms")
        print(f"Average throughput: {avg_throughput:.2f} items/s")
        
        return {
            "model_name": model_name,
            "platform": platform,
            "browser": browser,
            "precision": precision,
            "mixed_precision": mixed_precision,
            "use_real_browser": use_real_browser,
            "num_runs": num_runs,
            "avg_latency_ms": avg_latency,
            "avg_throughput_items_per_sec": avg_throughput,
            "results": results
        }
    
    return {"error": "No successful benchmark runs"}

async def run_demo():
    """Run the WebNN/WebGPU with IPFS Acceleration demo."""
    parser = argparse.ArgumentParser(description="WebNN/WebGPU with IPFS Acceleration Demo")
    parser.add_argument("--model", type=str, default="bert-base-uncased", help="Model name")
    parser.add_argument("--model-type", type=str, help="Model type (text, vision, audio, etc.)")
    parser.add_argument("--platform", type=str, choices=["webnn", "webgpu"], default="webgpu", help="Platform to use")
    parser.add_argument("--browser", type=str, choices=["chrome", "firefox", "edge", "safari"], help="Browser to use")
    parser.add_argument("--precision", type=int, choices=[4, 8, 16, 32], default=16, help="Precision to use")
    parser.add_argument("--mixed-precision", action="store_true", help="Use mixed precision")
    parser.add_argument("--real-browser", action="store_true", help="Use a real browser (not headless)")
    parser.add_argument("--runs", type=int, default=1, help="Number of benchmark runs")
    parser.add_argument("--save", type=str, help="Save benchmark results to file")
    args = parser.parse_args()
    
    # Run benchmark
    results = await run_benchmark(
        model_name=args.model,
        model_type=args.model_type,
        platform=args.platform,
        browser=args.browser,
        precision=args.precision,
        mixed_precision=args.mixed_precision,
        use_real_browser=args.real_browser,
        num_runs=args.runs
    )
    
    # Save results if requested
    if args.save:
        # Clean up results for JSON serialization
        if "results" in results:
            for result in results["results"]:
                # Convert numpy arrays to lists if present
                for key, value in result.items():
                    if hasattr(value, "tolist"):
                        result[key] = value.tolist()
        
        with open(args.save, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nBenchmark results saved to {args.save}")
    
    print("\nDemo completed.")

def main():
    """Entry point for the demo."""
    try:
        asyncio.run(run_demo())
    except KeyboardInterrupt:
        print("\nDemo interrupted by user.")
    except Exception as e:
        print(f"\nError in demo: {e}")

if __name__ == "__main__":
    main()