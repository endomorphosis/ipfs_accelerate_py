#!/usr/bin/env python3
"""
Demo script for WebGPU ultra-low precision functionality.

This script demonstrates the use of ultra-low precision (2-bit, 3-bit) quantization
with WebGPU to achieve significant memory savings and context extension.
"""

import os
import sys
import json
import argparse
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_ultra_low_precision(model_name, model_type, precision_bits, browser, extended_context=False):
    """
    Test ultra-low precision quantization for a model.
    
    Args:
        model_name: Name of the model
        model_type: Type of the model ('text', 'vision', 'audio')
        precision_bits: Number of bits for quantization (2, 3, or 4)
        browser: Browser to use ('chrome', 'firefox', 'edge', 'safari')
        extended_context: Whether to enable extended context window
    """
    try:
        from fixed_web_platform.webgpu_ultra_low_precision import setup_ultra_low_precision
        
        # Set up ultra-low precision
        result = setup_ultra_low_precision(
            model_name=model_name,
            model_type=model_type,
            precision_bits=precision_bits,
            mixed_precision=True,
            enable_kv_cache=True,
            extended_context=extended_context,
            browser=browser
        )
        
        # Print results
        if result['success']:
            print(f"\n===== Ultra-Low Precision Setup Results =====")
            print(f"Model: {model_name} ({model_type})")
            print(f"Precision: {precision_bits}-bit with mixed precision")
            print(f"Browser: {browser}")
            print(f"Memory reduction: {result['ultra_low_precision']['memory_reduction_percent']:.1f}%")
            
            # Show memory savings details
            memory_savings = result['ultra_low_precision']['memory_savings']
            print(f"\nMemory usage:")
            print(f"  Original size: {memory_savings['original_size_mb']:.1f} MB")
            print(f"  New size: {memory_savings['new_size_mb']:.1f} MB")
            print(f"  Saved: {memory_savings['saved_mb']:.1f} MB ({memory_savings['reduction_percent']:.1f}%)")
            
            # Show context extension if enabled
            if extended_context:
                context_factor = result['ultra_low_precision']['context_extension_factor']
                print(f"\nContext extension:")
                print(f"  Extension factor: {context_factor:.1f}x")
                print(f"  Example: 4K context -> {int(4096 * context_factor)} tokens")
            
            # Show layer-specific precision configuration
            layer_config = result['ultra_low_precision']['layer_config']
            print(f"\nLayer-specific precision configuration:")
            for layer, bits in layer_config.items():
                print(f"  {layer}: {bits}-bit")
            
            # Show accuracy impact
            accuracy_impact = result['ultra_low_precision']['accuracy_impact_percent']
            print(f"\nAccuracy impact:")
            print(f"  Expected accuracy reduction: {accuracy_impact:.1f}%")
            
            return True
        else:
            print(f"Failed to set up ultra-low precision: {result.get('error', 'Unknown error')}")
            return False
    except ImportError:
        print("Ultra-low precision module not found.")
        return False
    except Exception as e:
        print(f"Error testing ultra-low precision: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_context_extension(model_name, target_length=32768, browser='chrome'):
    """
    Test context extension functionality.
    
    Args:
        model_name: Name of the model
        target_length: Target context length
        browser: Browser to use
    """
    try:
        from fixed_web_platform.webgpu_ultra_low_precision import extend_context_window
        
        # Try to extend the context window
        context_config = extend_context_window(
            model_name=model_name,
            original_length=4096,  # Standard context for most models
            target_length=target_length,
            browser=browser
        )
        
        # Print results
        print(f"\n===== Context Extension Results =====")
        print(f"Model: {model_name}")
        print(f"Browser: {browser}")
        print(f"Original context length: {context_config['original_context_length']} tokens")
        print(f"Target context length: {context_config['target_context_length']} tokens")
        print(f"Achieved context length: {context_config['achieved_context_length']} tokens")
        print(f"Extension factor: {context_config['extension_factor']:.1f}x")
        print(f"Precision bits: {context_config['precision_bits']}-bit")
        print(f"Memory reduction: {context_config['memory_reduction_percent']:.1f}%")
        print(f"Target achieved: {'Yes' if context_config['target_achieved'] else 'No'}")
        
        return context_config['target_achieved']
    except ImportError:
        print("Context extension module not found.")
        return False
    except Exception as e:
        print(f"Error testing context extension: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_resource_pool_with_ulp(model_name, model_type, precision_bits=2, browser=None):
    """
    Test resource pool integration with ultra-low precision.
    
    Args:
        model_name: Name of the model
        model_type: Type of model
        precision_bits: Number of bits for quantization
        browser: Browser to use (or None for automatic selection)
    """
    try:
        from fixed_web_platform.resource_pool_bridge import ResourcePoolBridgeIntegration
        
        # Create resource pool integration
        integration = ResourcePoolBridgeIntegration(
            max_connections=2,
            browser_preferences={
                'audio': 'firefox',
                'vision': 'chrome',
                'text': 'edge'
            },
            adaptive_scaling=True
        )
        
        # Initialize integration
        integration.initialize()
        
        # Create hardware preferences with ultra-low precision
        hardware_preferences = {
            'priority_list': ['webgpu', 'cpu'],
            'precision_bits': precision_bits,
            'mixed_precision': True,
            'enable_kv_cache': True,
            'extended_context': True,
            'target_context_length': 16384
        }
        
        # Get model with ultra-low precision
        model = integration.get_model(model_type, model_name, hardware_preferences)
        
        # Check if model has ultra-low precision configuration
        has_ulp = hasattr(model, 'ulp_config')
        
        # Print results
        print(f"\n===== Resource Pool + Ultra-Low Precision Results =====")
        print(f"Model: {model_name} ({model_type})")
        print(f"Hardware: {model.hardware_type}")
        print(f"Browser: {model.browser}")
        print(f"Ultra-Low Precision enabled: {'Yes' if has_ulp else 'No'}")
        
        if has_ulp:
            ulp_config = model.ulp_config
            print(f"Precision: {ulp_config['ultra_low_precision']['bits']}-bit")
            print(f"Memory reduction: {ulp_config['ultra_low_precision']['memory_reduction_percent']:.1f}%")
            if ulp_config['ultra_low_precision']['extended_context']:
                print(f"Context extension: {ulp_config['ultra_low_precision']['context_extension_factor']:.1f}x")
        
        # Run inference
        inputs = "Sample text for testing ultra-low precision inference."
        result = model(inputs)
        
        # Print inference results
        print(f"\nInference result:")
        print(f"  Success: {result.get('success', False)}")
        print(f"  Compute shader optimized: {result.get('compute_shader_optimized', False)}")
        print(f"  Precompile shaders: {result.get('precompile_shaders', False)}")
        print(f"  Mixed precision: {result.get('mixed_precision', False)}")
        print(f"  Precision: {result.get('precision', 16)}-bit")
        
        return True
    except ImportError as e:
        print(f"Import error: {e}")
        return False
    except Exception as e:
        print(f"Error testing resource pool with ultra-low precision: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description="Test WebGPU ultra-low precision functionality")
    parser.add_argument("--model", type=str, default="llama-7b", help="Model name")
    parser.add_argument("--type", type=str, default="text", choices=["text", "vision", "audio"], help="Model type")
    parser.add_argument("--bits", type=int, default=2, choices=[2, 3, 4], help="Bits for quantization")
    parser.add_argument("--browser", type=str, default="chrome", choices=["chrome", "firefox", "edge", "safari"], help="Browser to use")
    parser.add_argument("--extended-context", action="store_true", help="Enable extended context")
    parser.add_argument("--context-length", type=int, default=32768, help="Target context length")
    parser.add_argument("--test-mode", type=str, default="basic", choices=["basic", "context", "resource-pool", "all"], help="Test mode")
    
    args = parser.parse_args()
    
    # Choose test based on mode
    if args.test_mode == "basic" or args.test_mode == "all":
        test_ultra_low_precision(args.model, args.type, args.bits, args.browser, args.extended_context)
    
    if args.test_mode == "context" or args.test_mode == "all":
        test_context_extension(args.model, args.context_length, args.browser)
    
    if args.test_mode == "resource-pool" or args.test_mode == "all":
        test_resource_pool_with_ulp(args.model, args.type, args.bits, args.browser)

if __name__ == "__main__":
    main()