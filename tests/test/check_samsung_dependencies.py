#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Samsung NPU Support Dependency Checker

This script checks if all required dependencies for Samsung NPU support are installed.
It categorizes dependencies into core, database, API, and visualization groups,
and suggests installation commands for any missing dependencies.
"""

import importlib
import sys
import os
from pathlib import Path

# Define dependencies by category
DEPENDENCIES = {
    "Core": ["numpy"],
    "Database": ["duckdb", "pandas"],
    "API": ["fastapi", "uvicorn", "pydantic"],
    "Visualization": ["matplotlib", "plotly"]
}

# Track installation status
status = {
    "Core": {},
    "Database": {},
    "API": {},
    "Visualization": {}
}

# Function to check if a module is installed
def check_dependency(module_name):
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False

# Check all dependencies
print("Checking Samsung NPU support dependencies...\n")

for category, deps in DEPENDENCIES.items():
    print(f"=== {category} Dependencies ===")
    for dep in deps:
        is_installed = check_dependency(dep)
        status[category][dep] = is_installed
        print(f"  {dep}: {'✓ Found' if is_installed else '✗ Not found'}")
    print()

# Test basic Samsung functionality
print("=== Testing Basic Samsung NPU Support ===")
try:
    # Set simulation mode for testing
    os.environ["TEST_SAMSUNG_CHIPSET"] = "exynos_2400"
    
    # Configure path if needed
    current_dir = Path(__file__).resolve().parent
    if str(current_dir) not in sys.path:
        sys.path.append(str(current_dir))
    
    # Try to import core components
    from samsung_support import SamsungChipset, SamsungChipsetRegistry, SamsungDetector
    minimal_works = True
    print("  Basic functionality: ✓ Works")
    
    # Try to create a simulator
    detector = SamsungDetector()
    chipset = detector.detect_samsung_hardware()
    if chipset:
        print(f"  Detected chipset (simulation): {chipset.name}")
        
        try:
            # Test more advanced functionality that requires database
            from samsung_support import SamsungBenchmarkRunner
            runner = SamsungBenchmarkRunner()
            print("  Benchmark functionality: ✓ Available")
        except (ImportError, AttributeError):
            print("  Benchmark functionality: ✗ Requires database dependencies")
            
except ImportError as e:
    minimal_works = False
    print(f"  Basic functionality: ✗ Failed ({e})")

# Print summary
print("\n=== Summary ===")

# Check Core dependencies status
all_core_installed = all(status["Core"].values())
if all_core_installed:
    print("✓ Core functionality available (minimal usage)")
else:
    print("✗ Core functionality not available - install required dependencies")

# Check Database dependencies status
all_db_installed = all(status["Database"].values())
if all_db_installed:
    print("✓ Database integration available (benchmarking and result storage)")
else:
    print("✗ Database integration not available - some benchmarking features will be limited")

# Check API dependencies status
all_api_installed = all(status["API"].values())
if all_api_installed:
    print("✓ API functionality available (network services)")
else:
    print("✗ API functionality not available - network services will be limited")

# Check Visualization dependencies status
all_viz_installed = all(status["Visualization"].values())
if all_viz_installed:
    print("✓ Visualization functionality available (result visualization)")
else:
    print("✗ Visualization functionality not available - result visualization will be limited")

# Print installation suggestions
print("\n=== Installation Commands ===")

# Core dependencies
missing_core = [dep for dep, installed in status["Core"].items() if not installed]
if missing_core:
    print(f"# Install core dependencies (required for basic functionality)")
    print(f"pip install {' '.join(missing_core)}")

# Database dependencies
missing_db = [dep for dep, installed in status["Database"].items() if not installed]
if missing_db:
    print(f"# Install database dependencies (required for benchmarking)")
    print(f"pip install {' '.join(missing_db)}")

# API dependencies
missing_api = [dep for dep, installed in status["API"].items() if not installed]
if missing_api:
    print(f"# Install API dependencies (required for network services)")
    print(f"pip install {' '.join(missing_api)}")

# Visualization dependencies
missing_viz = [dep for dep, installed in status["Visualization"].items() if not installed]
if missing_viz:
    print(f"# Install visualization dependencies (required for result visualization)")
    print(f"pip install {' '.join(missing_viz)}")

# Add all-in-one command
all_missing = missing_core + missing_db + missing_api + missing_viz
if all_missing:
    print("\n# Install all missing dependencies at once")
    print(f"pip install {' '.join(all_missing)}")
else:
    print("\n# All dependencies are already installed")

# Install from requirements file
print("\n# Or install all Samsung NPU dependencies at once")
print("pip install -r requirements_samsung.txt")

if minimal_works:
    print("\n=== Basic Simulator Test ===")
    # Create a simple test
    print("Running a minimal simulation test...")
    chipset = SamsungChipset(
        name="Exynos 2400",
        npu_cores=8,
        npu_tops=34.4,
        max_precision="FP16",
        supported_precisions=["FP32", "FP16", "BF16", "INT8", "INT4"],
        max_power_draw=8.5,
        typical_power=3.5
    )
    
    print(f"Created test chipset: {chipset.name}")
    print(f"  NPU Cores: {chipset.npu_cores}")
    print(f"  NPU Performance: {chipset.npu_tops} TOPS")
    print(f"  Max Precision: {chipset.max_precision}")
    print(f"  Supported Precisions: {', '.join(chipset.supported_precisions)}")
    print(f"  Typical Power: {chipset.typical_power}W")
    
    print("\nSimulator test completed successfully")

# Exit with appropriate status code
if not all_core_installed:
    sys.exit(1)  # Exit with error if core dependencies are missing
sys.exit(0)      # Exit with success otherwise