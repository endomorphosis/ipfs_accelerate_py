#!/usr/bin/env python
"""
Verify that the test environment is correctly set up.

This script checks that all required dependencies are installed
and that the test framework structure is properly configured.
"""

import os
import sys
import importlib
import platform
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional


def check_python_version() -> bool:
    """Check that Python version is 3.8 or higher."""
    min_version = (3, 8)
    current_version = sys.version_info[:2]
    
    print(f"Python version: {'.'.join(map(str, current_version))}")
    
    if current_version >= min_version:
        print("✅ Python version check passed")
        return True
    else:
        print(f"❌ Python version check failed. Required: {'.'.join(map(str, min_version))} or higher")
        return False


def check_required_packages() -> Tuple[bool, List[str]]:
    """Check that required packages are installed."""
    required_packages = [
        "pytest",
        "pytest-html",
        "pytest-cov",
        "torch",
        "numpy"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package} is installed")
        except ImportError:
            print(f"❌ {package} is not installed")
            missing_packages.append(package)
    
    return len(missing_packages) == 0, missing_packages


def check_optional_packages() -> Dict[str, bool]:
    """Check for optional packages."""
    optional_packages = {
        "pytest-selenium": "browser testing",
        "duckdb": "database tests",
        "pandas": "data analysis",
        "matplotlib": "visualization",
        "torchvision": "vision models",
        "torchaudio": "audio models",
        "transformers": "HuggingFace models",
        "openai": "OpenAI API testing"
    }
    
    status = {}
    
    for package, purpose in optional_packages.items():
        try:
            importlib.import_module(package)
            print(f"✅ {package} is installed (for {purpose})")
            status[package] = True
        except ImportError:
            print(f"⚠️ {package} is not installed (for {purpose})")
            status[package] = False
    
    return status


def check_hardware_support() -> Dict[str, bool]:
    """Check for hardware support."""
    support = {}
    
    # Check for CUDA
    try:
        import torch
        support["cuda"] = torch.cuda.is_available()
        if support["cuda"]:
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
            print(f"✅ CUDA is available ({device_count} devices, {device_name})")
        else:
            print("⚠️ CUDA is not available")
    except Exception as e:
        print(f"⚠️ Error checking CUDA: {e}")
        support["cuda"] = False
    
    # Check for ROCm (AMD)
    try:
        import torch
        support["rocm"] = hasattr(torch, 'hip') and torch.hip.is_available() if hasattr(torch, 'hip') else False
        if support["rocm"]:
            print("✅ ROCm is available")
        else:
            print("⚠️ ROCm is not available")
    except Exception as e:
        print(f"⚠️ Error checking ROCm: {e}")
        support["rocm"] = False
    
    # Check for MPS (Apple Silicon)
    try:
        import torch
        support["mps"] = hasattr(torch, 'mps') and torch.backends.mps.is_available() if hasattr(torch, 'mps') else False
        if support["mps"]:
            print("✅ MPS (Apple Silicon) is available")
        else:
            print("⚠️ MPS (Apple Silicon) is not available")
    except Exception as e:
        print(f"⚠️ Error checking MPS: {e}")
        support["mps"] = False
    
    # Check for WebGPU (based on browser availability, simplified check)
    chrome_path = None
    firefox_path = None
    
    if platform.system() == "Windows":
        chrome_candidates = [
            os.path.expandvars(r"%ProgramFiles%\Google\Chrome\Application\chrome.exe"),
            os.path.expandvars(r"%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe"),
        ]
        firefox_candidates = [
            os.path.expandvars(r"%ProgramFiles%\Mozilla Firefox\firefox.exe"),
            os.path.expandvars(r"%ProgramFiles(x86)%\Mozilla Firefox\firefox.exe"),
        ]
    elif platform.system() == "Darwin":  # macOS
        chrome_candidates = [
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            "/Applications/Chrome.app/Contents/MacOS/Chrome",
        ]
        firefox_candidates = [
            "/Applications/Firefox.app/Contents/MacOS/firefox",
        ]
    else:  # Linux and others
        chrome_candidates = [
            "/usr/bin/google-chrome",
            "/usr/bin/chrome",
            "/usr/bin/chromium",
            "/usr/bin/chromium-browser",
        ]
        firefox_candidates = [
            "/usr/bin/firefox",
        ]
    
    for path in chrome_candidates:
        if os.path.exists(path):
            chrome_path = path
            break
    
    for path in firefox_candidates:
        if os.path.exists(path):
            firefox_path = path
            break
    
    support["chrome"] = chrome_path is not None
    support["firefox"] = firefox_path is not None
    support["webgpu"] = support["chrome"] or support["firefox"]  # Simplified check
    
    if support["chrome"]:
        print("✅ Chrome is available (potentially WebGPU capable)")
    else:
        print("⚠️ Chrome is not available")
    
    if support["firefox"]:
        print("✅ Firefox is available (potentially WebGPU capable)")
    else:
        print("⚠️ Firefox is not available")
    
    return support


def check_directory_structure() -> bool:
    """Check that the test directory structure is correctly set up."""
    base_dir = Path(__file__).parent
    test_dir = base_dir / "test"
    
    required_dirs = [
        test_dir,
        test_dir / "models",
        test_dir / "hardware",
        test_dir / "api",
        test_dir / "integration",
        test_dir / "common",
        test_dir / "docs",
    ]
    
    missing_dirs = []
    
    for directory in required_dirs:
        if not directory.exists():
            missing_dirs.append(directory)
            print(f"❌ Directory not found: {directory}")
        else:
            print(f"✅ Directory exists: {directory}")
    
    if missing_dirs:
        print(f"❌ {len(missing_dirs)} required directories are missing")
        return False
    else:
        print("✅ All required directories exist")
        return True


def check_required_files() -> bool:
    """Check that required files exist."""
    base_dir = Path(__file__).parent
    
    required_files = [
        base_dir / "run.py",
        base_dir / "pytest.ini",
        base_dir / "conftest.py",
    ]
    
    missing_files = []
    
    for file_path in required_files:
        if not file_path.exists():
            missing_files.append(file_path)
            print(f"❌ File not found: {file_path}")
        else:
            print(f"✅ File exists: {file_path}")
    
    if missing_files:
        print(f"❌ {len(missing_files)} required files are missing")
        return False
    else:
        print("✅ All required files exist")
        return True


def print_environment_summary() -> None:
    """Print a summary of the environment."""
    # System information
    print("\n===== System Information =====")
    print(f"Platform: {platform.platform()}")
    print(f"Processor: {platform.processor()}")
    
    # Python information
    print("\n===== Python Information =====")
    print(f"Python version: {platform.python_version()}")
    print(f"Python implementation: {platform.python_implementation()}")
    print(f"Python path: {sys.executable}")
    
    # Directory information
    print("\n===== Directory Information =====")
    print(f"Current directory: {os.getcwd()}")
    print(f"Script directory: {Path(__file__).parent}")


def main() -> int:
    """Main entry point."""
    print("Verifying IPFS Accelerate test environment...\n")
    
    # Print environment summary
    print_environment_summary()
    
    # Check Python version
    print("\n===== Python Version Check =====")
    python_version_ok = check_python_version()
    
    # Check required packages
    print("\n===== Required Packages Check =====")
    packages_ok, missing_packages = check_required_packages()
    
    # Check optional packages
    print("\n===== Optional Packages Check =====")
    optional_packages = check_optional_packages()
    
    # Check hardware support
    print("\n===== Hardware Support Check =====")
    hardware_support = check_hardware_support()
    
    # Check directory structure
    print("\n===== Directory Structure Check =====")
    directory_structure_ok = check_directory_structure()
    
    # Check required files
    print("\n===== Required Files Check =====")
    required_files_ok = check_required_files()
    
    # Overall result
    print("\n===== Overall Result =====")
    passed = python_version_ok and packages_ok and directory_structure_ok and required_files_ok
    
    if passed:
        print("✅ Environment verification passed!")
        if missing_packages:
            print(f"⚠️ Warning: {len(missing_packages)} required packages are missing")
    else:
        print("❌ Environment verification failed!")
        
        if not python_version_ok:
            print("❌ Python version check failed")
        
        if not packages_ok:
            print(f"❌ Required packages check failed. Missing: {', '.join(missing_packages)}")
        
        if not directory_structure_ok:
            print("❌ Directory structure check failed")
        
        if not required_files_ok:
            print("❌ Required files check failed")
    
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())