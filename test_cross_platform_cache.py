#!/usr/bin/env python3
"""
Cross-Platform Cache Test Suite

This script tests the P2P cache functionality across different platforms:
- Linux laptop
- Windows laptop
- macOS (if available)

Tests cache operations, P2P connectivity, and platform-specific behaviors.
"""

import os
import sys
import platform
import socket
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Platform detection
IS_WINDOWS = platform.system() == "Windows"
IS_LINUX = platform.system() == "Linux"
IS_MACOS = platform.system() == "Darwin"


class CrossPlatformCacheTest:
    """Test suite for cross-platform cache functionality."""
    
    def __init__(self):
        self.platform = platform.system()
        self.passed = 0
        self.failed = 0
        self.warnings = 0
        self.test_results = []
        self.platform_issues = []
        self.alternative_backends = []  # Track working alternative backends
        
    def log(self, message: str, level: str = "INFO"):
        """Log with platform-specific formatting."""
        emoji = {
            "INFO": "ℹ️",
            "SUCCESS": "✅",
            "WARNING": "⚠️",
            "ERROR": "❌",
            "DEBUG": "🔍"
        }
        # Windows CMD doesn't always support emojis well
        if IS_WINDOWS:
            prefix = level[0]
        else:
            prefix = emoji.get(level, "•")
        print(f"{prefix}  {message}")
    
    def run_test(self, test_name: str, test_func) -> bool:
        """Run a single test and record result."""
        print(f"\n{'='*70}")
        print(f"Test: {test_name}")
        print('='*70)
        
        try:
            test_func()
            self.log(f"PASSED: {test_name}", "SUCCESS")
            self.passed += 1
            self.test_results.append(("PASS", test_name))
            return True
        except Exception as e:
            self.log(f"FAILED: {test_name}", "ERROR")
            self.log(f"Error: {e}", "ERROR")
            self.failed += 1
            self.test_results.append(("FAIL", test_name, str(e)))
            return False
    
    def test_platform_info(self):
        """Test platform detection and information."""
        self.log("Detecting platform information...", "INFO")
        
        info = {
            "System": platform.system(),
            "Release": platform.release(),
            "Version": platform.version(),
            "Machine": platform.machine(),
            "Processor": platform.processor(),
            "Python": platform.python_version(),
        }
        
        for key, value in info.items():
            self.log(f"{key}: {value}", "INFO")
        
        # Check if we're in a known supported platform
        if not (IS_WINDOWS or IS_LINUX or IS_MACOS):
            self.platform_issues.append(f"Unknown platform: {self.platform}")
            self.log(f"Warning: Unknown platform {self.platform}", "WARNING")
        else:
            self.log(f"Platform {self.platform} is supported", "SUCCESS")
    
    def test_python_environment(self):
        """Test Python environment compatibility."""
        self.log("Testing Python environment...", "INFO")
        
        # Check Python version
        py_version = sys.version_info
        if py_version >= (3, 8):
            self.log(f"Python {py_version.major}.{py_version.minor} OK (>= 3.8)", "SUCCESS")
        else:
            raise AssertionError(f"Python {py_version.major}.{py_version.minor} < 3.8 (unsupported)")
        
        # Check for virtual environment
        in_venv = hasattr(sys, 'real_prefix') or (
            hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
        )
        if in_venv:
            self.log("Running in virtual environment", "INFO")
        else:
            self.log("Not in virtual environment", "WARNING")
        
        # Check sys.path
        self.log(f"Python path entries: {len(sys.path)}", "INFO")
    
    def test_file_system_operations(self):
        """Test file system operations for cache storage."""
        self.log("Testing file system operations...", "INFO")
        
        # Test temp directory creation
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            self.log(f"Temp directory: {tmppath}", "INFO")
            
            # Test write
            test_file = tmppath / "test.txt"
            test_file.write_text("test data")
            self.log("File write successful", "SUCCESS")
            
            # Test read
            data = test_file.read_text()
            assert data == "test data", "File content mismatch"
            self.log("File read successful", "SUCCESS")
            
            # Test directory creation
            test_dir = tmppath / "subdir" / "nested"
            test_dir.mkdir(parents=True, exist_ok=True)
            self.log("Directory creation successful", "SUCCESS")
            
            # Test path operations
            if IS_WINDOWS:
                # Windows uses backslashes
                self.log("Windows path format detected", "INFO")
                assert "\\" in str(tmppath) or "/" in str(tmppath), "Path format issue"
            else:
                # Unix uses forward slashes
                self.log("Unix path format detected", "INFO")
                assert "/" in str(tmppath), "Path format issue"
    
    def test_network_operations(self):
        """Test network operations for P2P connectivity."""
        self.log("Testing network operations...", "INFO")
        
        # Get hostname
        hostname = socket.gethostname()
        self.log(f"Hostname: {hostname}", "INFO")
        
        # Get local IP addresses
        try:
            # Get IP by connecting to external address (doesn't actually connect)
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.settimeout(0)
            try:
                s.connect(('10.255.255.255', 1))
                local_ip = s.getsockname()[0]
            except Exception:
                local_ip = '127.0.0.1'
            finally:
                s.close()
            
            self.log(f"Local IP: {local_ip}", "SUCCESS")
        except Exception as e:
            self.log(f"Could not determine local IP: {e}", "WARNING")
            local_ip = "127.0.0.1"
        
        # Test localhost resolution
        try:
            localhost_ip = socket.gethostbyname('localhost')
            self.log(f"localhost resolves to {localhost_ip}", "SUCCESS")
        except Exception as e:
            self.log(f"Cannot resolve localhost: {e}", "ERROR")
            raise
        
        # Test port binding (simulate cache listener)
        test_port = 19999  # Use high port to avoid permissions
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(('127.0.0.1', test_port))
            sock.listen(1)
            self.log(f"Successfully bound to port {test_port}", "SUCCESS")
            sock.close()
        except Exception as e:
            self.log(f"Cannot bind to port {test_port}: {e}", "ERROR")
            raise
    
    def test_cache_module_import(self):
        """Test importing cache module."""
        self.log("Testing cache module import...", "INFO")
        
        try:
            from ipfs_accelerate_py.github_cli.cache import GitHubAPICache, configure_cache
            self.log("Cache module imported successfully", "SUCCESS")
            
            # Check platform-specific cache availability
            import ipfs_accelerate_py.github_cli.cache as cache_module
            
            if hasattr(cache_module, 'HAVE_LIBP2P'):
                libp2p_available = cache_module.HAVE_LIBP2P
                self.log(f"P2P support available: {libp2p_available}", "INFO")
                
                if not libp2p_available:
                    self.platform_issues.append("libp2p not available on this platform")
            
            if hasattr(cache_module, 'HAVE_CRYPTO'):
                crypto_available = cache_module.HAVE_CRYPTO
                self.log(f"Cryptography available: {crypto_available}", "INFO")
                
                if not crypto_available:
                    self.platform_issues.append("cryptography not available")
                    
        except ImportError as e:
            self.log(f"Failed to import cache module: {e}", "ERROR")
            raise AssertionError(f"Cannot import cache module: {e}")
    
    def test_cache_initialization(self):
        """Test cache initialization on this platform."""
        self.log("Testing cache initialization...", "INFO")
        
        try:
            from ipfs_accelerate_py.github_cli.cache import configure_cache
            
            with tempfile.TemporaryDirectory() as tmpdir:
                # Test without P2P (should work on all platforms)
                cache = configure_cache(
                    cache_dir=tmpdir,
                    enable_p2p=False,
                    enable_persistence=True
                )
                
                self.log("Cache initialized without P2P", "SUCCESS")
                self.log(f"Cache directory: {cache.cache_dir}", "INFO")
                
                # Clean up
                cache.shutdown()
                
                # Test with P2P if available
                try:
                    cache_p2p = configure_cache(
                        cache_dir=tmpdir,
                        enable_p2p=True,
                        p2p_listen_port=19999,
                        enable_persistence=False,
                        enable_peer_discovery=False
                    )
                    
                    if cache_p2p.enable_p2p:
                        self.log("P2P cache initialized successfully", "SUCCESS")
                    else:
                        self.log("P2P initialization failed (missing dependencies)", "WARNING")
                        self.platform_issues.append("P2P not functional")
                    
                    cache_p2p.shutdown()
                except Exception as e:
                    self.log(f"P2P initialization error: {e}", "WARNING")
                    self.platform_issues.append(f"P2P init error: {e}")
                    
                    # If P2P failed, try alternative backends
                    self.log("P2P not available, testing alternative backends...", "INFO")
                    self._test_alternative_backends(tmpdir)
                    
        except Exception as e:
            self.log(f"Cache initialization failed: {e}", "ERROR")
            raise
    
    def _test_alternative_backends(self, cache_dir: str):
        """Test alternative backends when libp2p is not available."""
        self.log("Testing ipfs-kit-py alternative backends...", "INFO")
        
        # Try to import ipfs_kit
        try:
            import ipfs_kit
            self.log("ipfs-kit-py package available", "SUCCESS")
        except ImportError:
            self.log("ipfs-kit-py not installed (pip install ipfs-kit-py)", "WARNING")
            self.platform_issues.append("ipfs-kit-py not available for alternative backends")
            return
        
        # Test Kubo backend
        self._test_kubo_backend(cache_dir)
        
        # Test Storacha backend
        self._test_storacha_backend(cache_dir)
        
        # Test S3 backend
        self._test_s3_backend(cache_dir)
        
        # Report results
        if self.alternative_backends:
            self.log(f"Alternative backends available: {', '.join(self.alternative_backends)}", "SUCCESS")
        else:
            self.log("No alternative backends available", "WARNING")
    
    def _test_kubo_backend(self, cache_dir: str):
        """Test Kubo (IPFS) backend."""
        self.log("Testing Kubo (IPFS) backend...", "INFO")
        
        try:
            from ipfs_kit import IPFSKit
            
            # Check if Kubo daemon is running
            try:
                kit = IPFSKit(backend='kubo')
                
                # Test basic operations
                test_data = {"test": "data", "backend": "kubo"}
                cid = kit.add_json(test_data)
                retrieved = kit.get_json(cid)
                
                if retrieved == test_data:
                    self.log("Kubo backend works!", "SUCCESS")
                    self.alternative_backends.append("kubo")
                else:
                    self.log("Kubo data mismatch", "WARNING")
                    
            except Exception as e:
                self.log(f"Kubo not available: {e}", "WARNING")
                self.log("Hint: Start IPFS daemon with 'ipfs daemon'", "INFO")
                
        except ImportError as e:
            self.log(f"Cannot test Kubo: {e}", "WARNING")
    
    def _test_storacha_backend(self, cache_dir: str):
        """Test Storacha (web3.storage) backend."""
        self.log("Testing Storacha (web3.storage) backend...", "INFO")
        
        try:
            from ipfs_kit import IPFSKit
            
            # Check for API token
            token = os.environ.get('WEB3_STORAGE_TOKEN') or os.environ.get('STORACHA_TOKEN')
            
            if not token:
                self.log("Storacha token not set (WEB3_STORAGE_TOKEN or STORACHA_TOKEN)", "INFO")
                self.log("Skipping Storacha test (requires API token)", "WARNING")
                return
            
            try:
                kit = IPFSKit(backend='storacha', token=token)
                
                # Test basic operations
                test_data = {"test": "data", "backend": "storacha"}
                cid = kit.add_json(test_data)
                retrieved = kit.get_json(cid)
                
                if retrieved == test_data:
                    self.log("Storacha backend works!", "SUCCESS")
                    self.alternative_backends.append("storacha")
                else:
                    self.log("Storacha data mismatch", "WARNING")
                    
            except Exception as e:
                self.log(f"Storacha error: {e}", "WARNING")
                self.log("Hint: Get token from https://web3.storage", "INFO")
                
        except ImportError as e:
            self.log(f"Cannot test Storacha: {e}", "WARNING")
    
    def _test_s3_backend(self, cache_dir: str):
        """Test S3-compatible backend."""
        self.log("Testing S3-compatible backend...", "INFO")
        
        try:
            from ipfs_kit import IPFSKit
            
            # Check for S3 credentials
            access_key = os.environ.get('AWS_ACCESS_KEY_ID')
            secret_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
            bucket = os.environ.get('AWS_S3_BUCKET') or os.environ.get('S3_BUCKET')
            
            if not (access_key and secret_key and bucket):
                self.log("S3 credentials not set (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, S3_BUCKET)", "INFO")
                self.log("Skipping S3 test (requires credentials)", "WARNING")
                return
            
            try:
                kit = IPFSKit(
                    backend='s3',
                    access_key=access_key,
                    secret_key=secret_key,
                    bucket=bucket
                )
                
                # Test basic operations
                test_data = {"test": "data", "backend": "s3"}
                cid = kit.add_json(test_data)
                retrieved = kit.get_json(cid)
                
                if retrieved == test_data:
                    self.log("S3 backend works!", "SUCCESS")
                    self.alternative_backends.append("s3")
                else:
                    self.log("S3 data mismatch", "WARNING")
                    
            except Exception as e:
                self.log(f"S3 error: {e}", "WARNING")
                self.log("Hint: Check credentials and bucket configuration", "INFO")
                
        except ImportError as e:
            self.log(f"Cannot test S3: {e}", "WARNING")
    
    def test_cache_operations(self):
        """Test basic cache operations."""
        self.log("Testing cache operations...", "INFO")
        
        try:
            from ipfs_accelerate_py.github_cli.cache import configure_cache
            
            with tempfile.TemporaryDirectory() as tmpdir:
                cache = configure_cache(
                    cache_dir=tmpdir,
                    enable_p2p=False,
                    enable_persistence=False
                )
                
                # Test PUT
                test_data = {
                    "platform": self.platform,
                    "test": "data",
                    "timestamp": time.time()
                }
                cache.put("test_key", test_data, ttl=300, param="value")
                self.log("Cache PUT successful", "SUCCESS")
                
                # Test GET
                result = cache.get("test_key", param="value")
                assert result == test_data, f"Data mismatch: {result} != {test_data}"
                self.log("Cache GET successful", "SUCCESS")
                
                # Test stats
                stats = cache.get_stats()
                self.log(f"Cache stats: hit_rate={stats['hit_rate']:.1%}", "INFO")
                assert stats['hits'] > 0, "No cache hits recorded"
                
                # Clean up
                cache.shutdown()
                
        except Exception as e:
            self.log(f"Cache operations failed: {e}", "ERROR")
            raise
    
    def test_multiprocessing_support(self):
        """Test if multiprocessing works (needed for P2P)."""
        self.log("Testing multiprocessing support...", "INFO")
        
        try:
            import multiprocessing as mp
            
            # Test creating a process
            def dummy_func():
                return "test"
            
            # Use spawn on all platforms for consistency
            ctx = mp.get_context('spawn')
            self.log(f"Multiprocessing context: spawn", "INFO")
            
            # Test queue (used by P2P cache)
            queue = ctx.Queue()
            queue.put("test")
            result = queue.get()
            assert result == "test", "Queue test failed"
            self.log("Multiprocessing queue works", "SUCCESS")
            
        except Exception as e:
            self.log(f"Multiprocessing test failed: {e}", "WARNING")
            self.platform_issues.append(f"Multiprocessing issues: {e}")
    
    def test_asyncio_support(self):
        """Test asyncio support (needed for libp2p)."""
        self.log("Testing asyncio support...", "INFO")
        
        try:
            import asyncio
            
            # Test event loop
            async def test_coro():
                await anyio.sleep(0.01)
                return "test"
            
            # Windows has special handling for event loops
            if IS_WINDOWS:
                # On Windows, use WindowsSelectorEventLoopPolicy
                asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
                self.log("Using WindowsSelectorEventLoopPolicy", "INFO")
            
            # Run test coroutine
            result = anyio.run(test_coro())
            assert result == "test", "Asyncio test failed"
            self.log("Asyncio works correctly", "SUCCESS")
            
        except Exception as e:
            self.log(f"Asyncio test failed: {e}", "WARNING")
            self.platform_issues.append(f"Asyncio issues: {e}")
    
    def test_cross_platform_paths(self):
        """Test path handling across platforms."""
        self.log("Testing cross-platform path handling...", "INFO")
        
        # Test Path operations
        test_path = Path.home() / ".cache" / "test_cache"
        self.log(f"Home cache path: {test_path}", "INFO")
        
        # Test path exists check
        home_exists = Path.home().exists()
        assert home_exists, "Home directory doesn't exist"
        self.log("Home directory exists", "SUCCESS")
        
        # Test creating directory in temp
        with tempfile.TemporaryDirectory() as tmpdir:
            nested = Path(tmpdir) / "nested" / "dir" / "structure"
            nested.mkdir(parents=True, exist_ok=True)
            assert nested.exists(), "Directory creation failed"
            self.log("Nested directory creation works", "SUCCESS")
            
            # Test file operations with pathlib
            test_file = nested / "test.json"
            test_file.write_text('{"test": "data"}')
            content = test_file.read_text()
            assert "test" in content, "File content wrong"
            self.log("Pathlib file operations work", "SUCCESS")
    
    def test_environment_variables(self):
        """Test environment variable access."""
        self.log("Testing environment variables...", "INFO")
        
        # Set test env var
        test_key = "TEST_CACHE_VAR"
        test_value = "test_value_123"
        os.environ[test_key] = test_value
        
        # Retrieve it
        retrieved = os.environ.get(test_key)
        assert retrieved == test_value, "Env var retrieval failed"
        self.log("Environment variables work", "SUCCESS")
        
        # Clean up
        del os.environ[test_key]
        
        # Test common env vars
        if IS_WINDOWS:
            self.log(f"USERPROFILE: {os.environ.get('USERPROFILE', 'Not set')}", "INFO")
            self.log(f"TEMP: {os.environ.get('TEMP', 'Not set')}", "INFO")
        else:
            self.log(f"HOME: {os.environ.get('HOME', 'Not set')}", "INFO")
            self.log(f"TMPDIR: {os.environ.get('TMPDIR', '/tmp')}", "INFO")
    
    def generate_platform_report(self):
        """Generate platform-specific compatibility report."""
        print("\n" + "="*70)
        print("PLATFORM COMPATIBILITY REPORT")
        print("="*70)
        
        print(f"\nPlatform: {self.platform}")
        print(f"Python: {platform.python_version()}")
        print(f"Architecture: {platform.machine()}")
        
        # Test summary
        total = self.passed + self.failed
        print(f"\nTest Results: {self.passed}/{total} passed")
        
        for result in self.test_results:
            status = result[0]
            name = result[1]
            if status == "PASS":
                print(f"  ✅ {name}")
            else:
                error = result[2] if len(result) > 2 else ""
                print(f"  ❌ {name}: {error}")
        
        # Platform-specific issues
        if self.platform_issues:
            print(f"\n{'='*70}")
            print("PLATFORM-SPECIFIC ISSUES")
            print("="*70)
            for i, issue in enumerate(self.platform_issues, 1):
                print(f"{i}. {issue}")
        
        # Alternative backends
        if self.alternative_backends:
            print(f"\n{'='*70}")
            print("ALTERNATIVE BACKENDS AVAILABLE")
            print("="*70)
            for backend in self.alternative_backends:
                print(f"  ✅ {backend.upper()} backend is working")
            print("\nYou can use these backends even without libp2p-py:")
            print("  • Kubo: IPFS daemon (ipfs daemon)")
            print("  • Storacha: web3.storage cloud service")
            print("  • S3: AWS S3 or compatible storage")
        elif "P2P" in str(self.platform_issues):
            print(f"\n{'='*70}")
            print("ALTERNATIVE BACKENDS RECOMMENDATION")
            print("="*70)
            print("Since libp2p-py is not working, consider using alternative backends:")
            print("\n1. Install ipfs-kit-py:")
            print("   pip install ipfs-kit-py")
            print("\n2. Choose a backend:")
            print("   • Kubo (IPFS): ipfs daemon (local)")
            print("   • Storacha: web3.storage (cloud, requires token)")
            print("   • S3: AWS S3 or compatible (cloud, requires credentials)")
            print("\n3. Set environment variables:")
            print("   export CACHE_BACKEND=kubo|storacha|s3")
            print("   # For Storacha:")
            print("   export WEB3_STORAGE_TOKEN=your_token")
            print("   # For S3:")
            print("   export AWS_ACCESS_KEY_ID=your_key")
            print("   export AWS_SECRET_ACCESS_KEY=your_secret")
            print("   export S3_BUCKET=your_bucket")
        
        # Recommendations
        print(f"\n{'='*70}")
        print("RECOMMENDATIONS FOR", self.platform)
        print("="*70)
        
        if IS_WINDOWS:
            print("\nWindows-Specific:")
            print("  1. Ensure Python installed from python.org (not Microsoft Store)")
            print("  2. Run PowerShell/CMD as Administrator for some operations")
            print("  3. May need Windows Subsystem for Linux (WSL) for full P2P support")
            print("  4. Check Windows Firewall settings for port access")
            print("  5. Consider using Windows Terminal for better emoji support")
        
        elif IS_LINUX:
            print("\nLinux-Specific:")
            print("  1. Ensure all system packages updated: sudo apt update")
            print("  2. May need build tools: sudo apt install build-essential")
            print("  3. Check firewall: sudo ufw status")
            print("  4. Ensure user in docker group: sudo usermod -aG docker $USER")
            print("  5. libp2p should work natively")
        
        elif IS_MACOS:
            print("\nmacOS-Specific:")
            print("  1. Ensure Xcode Command Line Tools installed")
            print("  2. May need Homebrew for dependencies")
            print("  3. Check System Preferences > Security for network access")
            print("  4. libp2p should work with some setup")
        
        # Overall status
        print(f"\n{'='*70}")
        if self.failed == 0 and len(self.platform_issues) == 0:
            print(f"✅ {self.platform} is fully compatible!")
            print("\nNext steps:")
            print("  1. Test on other platform (Windows/Linux)")
            print("  2. Run full diagnostic: python test_docker_runner_cache_connectivity.py")
            print("  3. Proceed with Docker testing")
        elif self.failed == 0 and len(self.platform_issues) > 0:
            print(f"⚠️  {self.platform} is mostly compatible with some warnings")
            print("\nNext steps:")
            print("  1. Review platform-specific issues above")
            print("  2. Install missing dependencies if needed")
            if not self.alternative_backends and "P2P" in str(self.platform_issues):
                print("  3. Install ipfs-kit-py for alternative backends:")
                print("     pip install ipfs-kit-py")
                print("  4. Configure alternative backend (kubo, storacha, or s3)")
            else:
                print("  3. Test without P2P if libp2p unavailable")
        else:
            print(f"❌ {self.platform} has compatibility issues")
            print("\nNext steps:")
            print("  1. Review failed tests above")
            print("  2. Fix identified issues")
            print("  3. Re-run this test")
        print("="*70)
        
        return 0 if self.failed == 0 else 1
    
    def run_all_tests(self):
        """Run all cross-platform tests."""
        print("="*70)
        print(f"Cross-Platform Cache Test - {self.platform}")
        print("="*70)
        print("\nTesting cache functionality for platform compatibility.")
        print("This ensures the cache works on both Linux and Windows laptops")
        print("before testing Docker container scenarios.\n")
        
        # Run all tests
        self.run_test("Platform Information", self.test_platform_info)
        self.run_test("Python Environment", self.test_python_environment)
        self.run_test("File System Operations", self.test_file_system_operations)
        self.run_test("Network Operations", self.test_network_operations)
        self.run_test("Cache Module Import", self.test_cache_module_import)
        self.run_test("Cache Initialization", self.test_cache_initialization)
        self.run_test("Cache Operations", self.test_cache_operations)
        self.run_test("Multiprocessing Support", self.test_multiprocessing_support)
        self.run_test("Asyncio Support", self.test_asyncio_support)
        self.run_test("Cross-Platform Paths", self.test_cross_platform_paths)
        self.run_test("Environment Variables", self.test_environment_variables)
        
        return self.generate_platform_report()


def main():
    """Main entry point."""
    tester = CrossPlatformCacheTest()
    return tester.run_all_tests()


if __name__ == "__main__":
    sys.exit(main())
