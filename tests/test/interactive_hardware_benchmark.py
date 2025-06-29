#!/usr/bin/env python3
"""
Interactive script for running hardware benchmarks.
This script guides the user through the process of running benchmarks
for different models on various hardware backends.
"""

import os
import sys
import subprocess
import time
from typing import List, Dict, Any, Optional

def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    """Print the header."""
    clear_screen()
    print("=" * 80)
    print(" " * 20 + "HARDWARE BENCHMARK INTERACTIVE RUNNER" + " " * 20)
    print("=" * 80)
    print()

def detect_hardware():
    """Detect available hardware backends."""
    print("Detecting available hardware backends...")
    
    available_hardware = {
        "cpu": True  # CPU is always available
    }
    
    # Check CUDA
    try:
        import torch
        available_hardware["cuda"] = torch.cuda.is_available()
        if available_hardware["cuda"]:
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
            print(f"✅ CUDA available: {device_count} device(s) - {device_name}")
        else:
            print("❌ CUDA not available")
    except ImportError:
        available_hardware["cuda"] = False
        print("❌ PyTorch not installed, CUDA detection failed")
    
    # Check MPS (Apple Silicon)
    try:
        import torch
        mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        available_hardware["mps"] = mps_available
        if mps_available:
            print("✅ MPS (Apple Silicon) available")
        else:
            print("❌ MPS (Apple Silicon) not available")
    except ImportError:
        available_hardware["mps"] = False
    
    # Check ROCm
    try:
        import torch
        rocm_available = torch.cuda.is_available() and hasattr(torch.version, "hip")
        available_hardware["rocm"] = rocm_available
        if rocm_available:
            print("✅ ROCm (AMD GPU) available")
        else:
            print("❌ ROCm (AMD GPU) not available")
    except ImportError:
        available_hardware["rocm"] = False
    
    # Check OpenVINO
    try:
        import openvino
        available_hardware["openvino"] = True
        print(f"✅ OpenVINO available (version {openvino.__version__})")
    except ImportError:
        available_hardware["openvino"] = False
        print("❌ OpenVINO not available")
    
    # Add CPU info
    import platform
    cpu_info = platform.processor()
    if not cpu_info:
        # Try alternative methods if platform.processor() returns empty string
        if sys.platform == "linux" or sys.platform == "linux2":
            try:
                with open('/proc/cpuinfo', 'r') as f:
                    for line in f:
                        if line.startswith('model name'):
                            cpu_info = line.split(':', 1)[1].strip()
                            break
            except:
                cpu_info = "Unknown CPU"
    
    print(f"✅ CPU available: {cpu_info}")
    
    return {k: v for k, v in available_hardware.items() if v}

def select_from_options(options: List[str], title: str, multi_select: bool = False) -> List[str]:
    """
    Display a list of options and let the user select one or more.
    
    Args:
        options: List of options to display
        title: Title of the selection
        multi_select: Whether to allow multiple selections
    
    Returns:
        List of selected options
    """
    print(f"\n{title}")
    print("-" * len(title))
    
    for i, option in enumerate(options):
        print(f"{i+1}. {option}")
    
    if multi_select:
        print("\nEnter numbers separated by spaces, or 'all' for all options:")
        selection = input("> ").strip()
        
        if selection.lower() == 'all':
            return options
        
        try:
            indices = [int(idx) - 1 for idx in selection.split()]
            return [options[i] for i in indices if 0 <= i < len(options)]
        except (ValueError, IndexError):
            print("Invalid selection. Please try again.")
            return select_from_options(options, title, multi_select)
    else:
        print("\nEnter a number:")
        selection = input("> ").strip()
        
        try:
            index = int(selection) - 1
            if 0 <= index < len(options):
                return [options[index]]
            else:
                print("Invalid selection. Please try again.")
                return select_from_options(options, title, multi_select)
        except ValueError:
            print("Invalid selection. Please try again.")
            return select_from_options(options, title, multi_select)

def get_model_sets():
    """Get available model sets."""
    return [
        "text_embedding",
        "text_generation",
        "vision",
        "audio",
        "all",
        "quick"
    ]

def get_custom_models():
    """Get a list of custom models from the user."""
    print("\nEnter model names separated by spaces:")
    print("Examples: prajjwal1/bert-tiny google/t5-efficient-tiny")
    return input("> ").strip().split()

def get_batch_sizes():
    """Get batch sizes from the user."""
    print("\nEnter batch sizes separated by spaces (e.g., '1 4 16'):")
    batch_sizes = input("> ").strip()
    if not batch_sizes:
        batch_sizes = "1 4 16"  # Default
    return batch_sizes

def select_output_format():
    """Select output format."""
    formats = ["markdown", "json"]
    selected = select_from_options(formats, "Select output format:")
    return selected[0]

def get_output_directory():
    """Get output directory from the user."""
    print("\nEnter output directory (default: ./benchmark_results):")
    output_dir = input("> ").strip()
    if not output_dir:
        output_dir = "./benchmark_results"
    return output_dir

def select_openvino_precision():
    """Select OpenVINO precision."""
    precisions = ["FP32", "FP16", "INT8"]
    selected = select_from_options(precisions, "Select OpenVINO precision:")
    return selected[0]

def run_benchmark(models, hardware, batch_sizes, output_format, output_dir, openvino_precision=None, debug=False):
    """Run the benchmark with the selected options."""
    print("\nRunning benchmark with the following configuration:")
    print(f"Models: {models}")
    print(f"Hardware: {hardware}")
    print(f"Batch sizes: {batch_sizes}")
    print(f"Output format: {output_format}")
    print(f"Output directory: {output_dir}")
    if openvino_precision:
        print(f"OpenVINO precision: {openvino_precision}")
    
    command = [
        "python", "run_hardware_comparison.py",
        "--models", *models,
        "--hardware", *hardware,
        "--batch-sizes", *batch_sizes.split(),
        "--format", output_format,
        "--output-dir", output_dir
    ]
    
    if openvino_precision:
        command.extend(["--openvino-precision", openvino_precision])
    
    if debug:
        command.append("--debug")
    
    print("\nExecuting command:")
    print(" ".join(command))
    print("\nStarting benchmark. This may take a while...\n")
    
    # Execute the command
    start_time = time.time()
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
        
        # Print output in real-time
        for line in iter(process.stdout.readline, ''):
            print(line, end='')
        
        process.wait()
        exit_code = process.returncode
        
        if exit_code != 0:
            print(f"\nBenchmark process exited with code {exit_code}")
        else:
            end_time = time.time()
            duration = end_time - start_time
            print(f"\nBenchmark completed in {duration:.2f} seconds.")
    except Exception as e:
        print(f"\nError executing benchmark: {e}")
        return False
    
    return True

def check_requirements():
    """Check if required packages are installed."""
    try:
        import torch
        torch_version = torch.__version__
        print(f"✅ PyTorch installed (version {torch_version})")
    except ImportError:
        print("❌ PyTorch not installed. Required for running benchmarks.")
        print("   Please install PyTorch with: pip install torch")
        return False
    
    try:
        import transformers
        transformers_version = transformers.__version__
        print(f"✅ Transformers installed (version {transformers_version})")
    except ImportError:
        print("❌ Transformers not installed. Required for running benchmarks.")
        print("   Please install Transformers with: pip install transformers")
        return False
    
    # Check if run_hardware_comparison.py exists
    if not os.path.exists("run_hardware_comparison.py"):
        print("❌ run_hardware_comparison.py not found in the current directory.")
        print("   Please make sure you're in the correct directory.")
        return False
    
    return True

def main():
    """Main function."""
    print_header()
    
    # Check requirements
    if not check_requirements():
        print("\nPlease install the required packages and try again.")
        return
    
    # Detect available hardware
    available_hardware = detect_hardware()
    hardware_options = list(available_hardware.keys())
    
    # Let the user select model set or custom models
    model_set_options = get_model_sets()
    model_selection_type = select_from_options(
        ["Use a predefined model set", "Enter custom models"],
        "How would you like to select models?"
    )[0]
    
    if model_selection_type == "Use a predefined model set":
        model_set = select_from_options(model_set_options, "Select a model set:")[0]
        models_arg = ["--model-set", model_set]
        models_display = f"Model set: {model_set}"
    else:
        custom_models = get_custom_models()
        models_arg = custom_models
        models_display = f"Custom models: {' '.join(custom_models)}"
    
    # Let the user select hardware backends
    selected_hardware = select_from_options(
        hardware_options, 
        "Select hardware backends to benchmark:", 
        multi_select=True
    )
    
    # Get batch sizes
    batch_sizes = get_batch_sizes()
    
    # Select output format
    output_format = select_output_format()
    
    # Get output directory
    output_dir = get_output_directory()
    
    # If OpenVINO is selected, get precision
    openvino_precision = None
    if "openvino" in selected_hardware:
        openvino_precision = select_openvino_precision()
    
    # Ask about debug mode
    debug_mode = input("\nEnable debug logging? (y/n): ").strip().lower() == 'y'
    
    # Final confirmation
    print("\n=== Benchmark Configuration ===")
    print(f"Models: {models_display}")
    print(f"Hardware backends: {', '.join(selected_hardware)}")
    print(f"Batch sizes: {batch_sizes}")
    print(f"Output format: {output_format}")
    print(f"Output directory: {output_dir}")
    if openvino_precision:
        print(f"OpenVINO precision: {openvino_precision}")
    print(f"Debug mode: {'Enabled' if debug_mode else 'Disabled'}")
    
    print("\nStart benchmark with this configuration? (y/n):")
    if input("> ").strip().lower() != 'y':
        print("\nBenchmark cancelled.")
        return
    
    # Run the benchmark
    if model_selection_type == "Use a predefined model set":
        # Use the shell script for model sets
        command = ["./run_hardware_benchmark.sh"]
        command.extend(["--model-set", model_set])
        command.extend(["--hardware", " ".join(selected_hardware)])
        command.extend(["--batch-sizes", batch_sizes])
        command.extend(["--format", output_format])
        command.extend(["--output-dir", output_dir])
        if openvino_precision:
            command.extend(["--openvino-precision", openvino_precision])
        if debug_mode:
            command.append("--debug")
        
        print("\nExecuting command:")
        print(" ".join(command))
        print("\nStarting benchmark. This may take a while...\n")
        
        # Execute the command
        start_time = time.time()
        try:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
            
            # Print output in real-time
            for line in iter(process.stdout.readline, ''):
                print(line, end='')
            
            process.wait()
            exit_code = process.returncode
            
            if exit_code != 0:
                print(f"\nBenchmark process exited with code {exit_code}")
            else:
                end_time = time.time()
                duration = end_time - start_time
                print(f"\nBenchmark completed in {duration:.2f} seconds.")
        except Exception as e:
            print(f"\nError executing benchmark: {e}")
    else:
        # Run directly with custom models
        success = run_benchmark(
            models=models_arg,
            hardware=selected_hardware,
            batch_sizes=batch_sizes,
            output_format=output_format,
            output_dir=output_dir,
            openvino_precision=openvino_precision,
            debug=debug_mode
        )
        
        if success:
            print("\nBenchmark completed successfully!")
        else:
            print("\nBenchmark encountered errors. Please check the logs.")
    
    # Ask if the user wants to run another benchmark
    print("\nWould you like to run another benchmark? (y/n):")
    if input("> ").strip().lower() == 'y':
        main()
    else:
        print("\nThank you for using the Hardware Benchmark Interactive Runner. Goodbye!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nBenchmark cancelled by user. Exiting...")
        sys.exit(1)