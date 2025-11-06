#!/usr/bin/env python3
"""
Docker Container Dependency Validator and System Check

This script runs at container startup to validate all dependencies,
check system configuration, and ensure proper setup across different
architectures and operating systems.

Usage:
    python docker_startup_check.py [--verbose] [--fix] [--arch-check]
"""

import sys
import os
import platform
import subprocess
import logging
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("docker_startup_check")


class DockerEnvironmentValidator:
    """Validates Docker container environment and dependencies"""
    
    def __init__(self, verbose: bool = False, fix: bool = False):
        self.verbose = verbose
        self.fix = fix
        self.issues = []
        self.warnings = []
        self.info = []
        
        if verbose:
            logger.setLevel(logging.DEBUG)
    
    def log_issue(self, msg: str):
        """Log a critical issue"""
        self.issues.append(msg)
        logger.error(f"❌ {msg}")
    
    def log_warning(self, msg: str):
        """Log a warning"""
        self.warnings.append(msg)
        logger.warning(f"⚠️  {msg}")
    
    def log_info(self, msg: str):
        """Log informational message"""
        self.info.append(msg)
        logger.info(f"ℹ️  {msg}")
    
    def log_success(self, msg: str):
        """Log success message"""
        logger.info(f"✅ {msg}")
    
    def check_system_info(self) -> Dict:
        """Collect and validate system information"""
        logger.info("=" * 70)
        logger.info("Docker Container System Information")
        logger.info("=" * 70)
        
        system_info = {
            "platform": platform.system(),
            "platform_release": platform.release(),
            "platform_version": platform.version(),
            "architecture": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
        }
        
        # Detect container environment
        in_docker = os.path.exists('/.dockerenv') or os.path.exists('/run/.containerenv')
        system_info["in_container"] = in_docker
        
        # CPU info
        try:
            cpu_count = os.cpu_count()
            system_info["cpu_count"] = cpu_count
        except:
            system_info["cpu_count"] = "unknown"
        
        # Memory info
        try:
            with open('/proc/meminfo', 'r') as f:
                meminfo = f.read()
                for line in meminfo.split('\n'):
                    if line.startswith('MemTotal:'):
                        mem_kb = int(line.split()[1])
                        system_info["memory_total_mb"] = mem_kb // 1024
                        break
        except:
            system_info["memory_total_mb"] = "unknown"
        
        # Display system info
        for key, value in system_info.items():
            logger.info(f"  {key}: {value}")
        
        # Validate architecture
        arch = platform.machine().lower()
        supported_archs = ['x86_64', 'amd64', 'aarch64', 'arm64', 'armv7l']
        
        if arch in supported_archs:
            self.log_success(f"Architecture {arch} is supported")
        else:
            self.log_warning(f"Architecture {arch} may not be fully supported")
        
        if not in_docker:
            self.log_warning("Not running in a container environment")
        else:
            self.log_success("Running in container environment")
        
        return system_info
    
    def check_python_environment(self) -> bool:
        """Validate Python environment and core packages"""
        logger.info("\n" + "=" * 70)
        logger.info("Python Environment Check")
        logger.info("=" * 70)
        
        all_ok = True
        
        # Check Python version
        py_version = sys.version_info
        if py_version.major == 3 and py_version.minor >= 8:
            self.log_success(f"Python version {py_version.major}.{py_version.minor}.{py_version.micro}")
        else:
            self.log_issue(f"Python 3.8+ required, found {py_version.major}.{py_version.minor}")
            all_ok = False
        
        # Core Python packages
        core_packages = [
            'pip',
            'setuptools',
            'wheel',
        ]
        
        for package in core_packages:
            try:
                __import__(package)
                self.log_success(f"Package '{package}' is available")
            except ImportError:
                self.log_issue(f"Core package '{package}' is missing")
                all_ok = False
        
        return all_ok
    
    def check_ipfs_accelerate_package(self) -> bool:
        """Validate ipfs_accelerate_py package installation"""
        logger.info("\n" + "=" * 70)
        logger.info("IPFS Accelerate Package Check")
        logger.info("=" * 70)
        
        all_ok = True
        
        # Try to import the main package
        try:
            import ipfs_accelerate_py
            self.log_success("ipfs_accelerate_py package is importable")
            
            # Check for version
            if hasattr(ipfs_accelerate_py, '__version__'):
                logger.info(f"  Version: {ipfs_accelerate_py.__version__}")
            
            # Check for main modules
            modules_to_check = [
                'ipfs_accelerate_py.cli',
                'ipfs_accelerate_py.mcp',
                'shared',
            ]
            
            for module in modules_to_check:
                try:
                    __import__(module)
                    self.log_success(f"Module '{module}' is available")
                except ImportError as e:
                    self.log_warning(f"Optional module '{module}' not available: {e}")
            
        except ImportError as e:
            self.log_issue(f"ipfs_accelerate_py package not properly installed: {e}")
            all_ok = False
        
        return all_ok
    
    def check_system_dependencies(self) -> bool:
        """Check system-level dependencies"""
        logger.info("\n" + "=" * 70)
        logger.info("System Dependencies Check")
        logger.info("=" * 70)
        
        all_ok = True
        
        # Check for common system tools
        system_tools = {
            'curl': 'curl --version',
            'wget': 'wget --version',
            'git': 'git --version',
        }
        
        for tool, cmd in system_tools.items():
            try:
                result = subprocess.run(
                    cmd.split(),
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    version = result.stdout.split('\n')[0] if result.stdout else "installed"
                    self.log_success(f"{tool}: {version}")
                else:
                    self.log_warning(f"{tool} not available")
            except (subprocess.TimeoutExpired, FileNotFoundError):
                self.log_warning(f"{tool} not available")
        
        return all_ok
    
    def check_hardware_acceleration(self) -> Dict:
        """Check for hardware acceleration capabilities"""
        logger.info("\n" + "=" * 70)
        logger.info("Hardware Acceleration Check")
        logger.info("=" * 70)
        
        hw_info = {
            "cuda": False,
            "rocm": False,
            "openvino": False,
            "opencl": False,
        }
        
        # Check for CUDA
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi'], capture_output=True, timeout=5)
            if result.returncode == 0:
                hw_info["cuda"] = True
                self.log_success("NVIDIA CUDA detected")
                logger.info(f"  {result.stdout.decode().split(chr(10))[0]}")
        except:
            self.log_info("NVIDIA CUDA not available")
        
        # Check for ROCm (AMD)
        try:
            result = subprocess.run(['rocm-smi'], capture_output=True, timeout=5)
            if result.returncode == 0:
                hw_info["rocm"] = True
                self.log_success("AMD ROCm detected")
        except:
            self.log_info("AMD ROCm not available")
        
        # Check for OpenCL
        try:
            import pyopencl
            platforms = pyopencl.get_platforms()
            if platforms:
                hw_info["opencl"] = True
                self.log_success(f"OpenCL available ({len(platforms)} platform(s))")
        except:
            self.log_info("OpenCL not available")
        
        # Architecture-specific checks
        arch = platform.machine().lower()
        if arch in ['aarch64', 'arm64']:
            self.log_info("ARM64 architecture - checking for ARM-specific acceleration")
            # Check for ARM NN or other ARM-specific acceleration
            try:
                import subprocess
                # Check for ARM NN libraries
                result = subprocess.run(['ldconfig', '-p'], capture_output=True, text=True)
                if 'armnn' in result.stdout.lower():
                    self.log_success("ARM NN libraries detected")
                    hw_info["arm_nn"] = True
            except:
                self.log_info("ARM NN not available")
        
        if not any(hw_info.values()):
            self.log_info("No hardware acceleration detected - using CPU only")
        
        return hw_info
    
    def check_network_connectivity(self) -> bool:
        """Check network connectivity"""
        logger.info("\n" + "=" * 70)
        logger.info("Network Connectivity Check")
        logger.info("=" * 70)
        
        all_ok = True
        
        # Test DNS resolution
        try:
            import socket
            socket.gethostbyname('github.com')
            self.log_success("DNS resolution working")
        except:
            self.log_warning("DNS resolution may not be working")
            all_ok = False
        
        # Test HTTPS connectivity
        try:
            import urllib.request
            urllib.request.urlopen('https://www.google.com', timeout=5)
            self.log_success("HTTPS connectivity working")
        except:
            self.log_warning("HTTPS connectivity may be limited")
            all_ok = False
        
        return all_ok
    
    def check_file_permissions(self) -> bool:
        """Check file system permissions"""
        logger.info("\n" + "=" * 70)
        logger.info("File System Permissions Check")
        logger.info("=" * 70)
        
        all_ok = True
        
        # Check write permissions in key directories
        test_dirs = [
            '/app',
            '/tmp',
            os.path.expanduser('~'),
        ]
        
        for directory in test_dirs:
            if os.path.exists(directory):
                if os.access(directory, os.W_OK):
                    self.log_success(f"Write permission OK: {directory}")
                else:
                    self.log_warning(f"No write permission: {directory}")
                    all_ok = False
            else:
                self.log_info(f"Directory does not exist: {directory}")
        
        return all_ok
    
    def check_mcp_server_requirements(self) -> bool:
        """Check MCP server specific requirements"""
        logger.info("\n" + "=" * 70)
        logger.info("MCP Server Requirements Check")
        logger.info("=" * 70)
        
        all_ok = True
        
        # Check for MCP server dependencies
        mcp_packages = [
            'flask',
            'flask_cors',
            'jinja2',
            'werkzeug',
        ]
        
        for package in mcp_packages:
            try:
                __import__(package)
                self.log_success(f"MCP dependency '{package}' available")
            except ImportError:
                self.log_warning(f"MCP dependency '{package}' not available (will use fallback)")
        
        # Check if ports are available
        import socket
        test_ports = [8000, 5000]
        
        for port in test_ports:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(('127.0.0.1', port))
            sock.close()
            
            if result != 0:
                self.log_success(f"Port {port} is available")
            else:
                self.log_warning(f"Port {port} is already in use")
        
        return all_ok
    
    def run_all_checks(self) -> Tuple[bool, Dict]:
        """Run all validation checks"""
        logger.info("\n" + "🚀" * 35)
        logger.info("IPFS Accelerate Docker Container Startup Validation")
        logger.info("🚀" * 35 + "\n")
        
        results = {}
        
        # System info
        results['system_info'] = self.check_system_info()
        
        # Python environment
        results['python_ok'] = self.check_python_environment()
        
        # Package installation
        results['package_ok'] = self.check_ipfs_accelerate_package()
        
        # System dependencies
        results['system_deps_ok'] = self.check_system_dependencies()
        
        # Hardware acceleration
        results['hardware'] = self.check_hardware_acceleration()
        
        # Network connectivity
        results['network_ok'] = self.check_network_connectivity()
        
        # File permissions
        results['permissions_ok'] = self.check_file_permissions()
        
        # MCP server requirements
        results['mcp_ok'] = self.check_mcp_server_requirements()
        
        # Summary
        logger.info("\n" + "=" * 70)
        logger.info("Validation Summary")
        logger.info("=" * 70)
        
        if self.issues:
            logger.error(f"\n❌ Critical Issues Found ({len(self.issues)}):")
            for issue in self.issues:
                logger.error(f"  - {issue}")
        
        if self.warnings:
            logger.warning(f"\n⚠️  Warnings ({len(self.warnings)}):")
            for warning in self.warnings:
                logger.warning(f"  - {warning}")
        
        if self.info:
            logger.info(f"\nℹ️  Information ({len(self.info)}):")
            for info in self.info:
                logger.info(f"  - {info}")
        
        # Overall status
        all_critical_ok = (
            results['python_ok'] and 
            results['package_ok']
        )
        
        logger.info("\n" + "=" * 70)
        if all_critical_ok:
            logger.info("✅ Container is ready for operation")
            logger.info("=" * 70)
            return True, results
        else:
            logger.error("❌ Container has critical issues that must be resolved")
            logger.info("=" * 70)
            return False, results


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Validate Docker container environment for IPFS Accelerate'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    parser.add_argument(
        '--fix',
        action='store_true',
        help='Attempt to fix issues automatically'
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output results as JSON'
    )
    parser.add_argument(
        '--exit-on-error',
        action='store_true',
        default=True,
        help='Exit with error code if validation fails (default: True)'
    )
    
    args = parser.parse_args()
    
    # Run validation
    validator = DockerEnvironmentValidator(verbose=args.verbose, fix=args.fix)
    success, results = validator.run_all_checks()
    
    # Output results
    if args.json:
        output = {
            'success': success,
            'issues': validator.issues,
            'warnings': validator.warnings,
            'info': validator.info,
            'results': {
                k: v for k, v in results.items() 
                if not isinstance(v, dict) or k == 'system_info'
            }
        }
        print(json.dumps(output, indent=2))
    
    # Exit with appropriate code
    if args.exit_on_error and not success:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == '__main__':
    main()
