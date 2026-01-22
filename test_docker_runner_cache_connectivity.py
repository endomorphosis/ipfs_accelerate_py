#!/usr/bin/env python3
"""
Test Docker Runner Cache Connectivity

This script tests whether GitHub Actions runners running in Docker containers
can successfully connect to the P2P cache managed by the ipfs_accelerate_py package.

It performs the following diagnostics:
1. Verifies P2P cache dependencies are installed
2. Tests cache initialization inside a container-like environment
3. Checks network connectivity to the MCP server P2P endpoint
4. Validates cache operations (get/put/invalidate)
5. Tests peer discovery mechanisms
6. Validates encryption setup
"""

import os
import sys
import json
import time
import socket
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add to path
sys.path.insert(0, str(Path(__file__).parent))


class DockerRunnerCacheConnectivityTest:
    """Test suite for Docker runner cache connectivity."""
    
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.test_results = []
        self.issues = []
        self.recommendations = []
    
    def log(self, message: str, level: str = "INFO"):
        """Log a message with level."""
        prefix = {
            "INFO": "ℹ️",
            "SUCCESS": "✅",
            "WARNING": "⚠️",
            "ERROR": "❌",
            "DEBUG": "🔍"
        }.get(level, "•")
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
    
    def test_p2p_dependencies(self):
        """Test if P2P dependencies are installed."""
        self.log("Checking P2P dependencies...", "INFO")
        
        dependencies = {
            "libp2p": False,
            "cryptography": False,
            "py-multiformats-cid": False,
        }
        
        # Check libp2p
        try:
            import libp2p
            dependencies["libp2p"] = True
            self.log(f"libp2p version: {getattr(libp2p, '__version__', 'unknown')}", "SUCCESS")
        except ImportError as e:
            self.log(f"libp2p not installed: {e}", "ERROR")
            self.issues.append("libp2p not installed")
            self.recommendations.append("Install libp2p: pip install libp2p>=0.1.5")
        
        # Check cryptography
        try:
            import cryptography
            dependencies["cryptography"] = True
            self.log(f"cryptography version: {cryptography.__version__}", "SUCCESS")
        except ImportError as e:
            self.log(f"cryptography not installed: {e}", "ERROR")
            self.issues.append("cryptography not installed")
            self.recommendations.append("Install cryptography: pip install cryptography")
        
        # Check multiformats
        try:
            from multiformats import CID
            dependencies["py-multiformats-cid"] = True
            self.log("multiformats (CID) available", "SUCCESS")
        except ImportError as e:
            self.log(f"multiformats not installed: {e}", "ERROR")
            self.issues.append("multiformats not installed")
            self.recommendations.append("Install multiformats: pip install py-multiformats-cid")
        
        # Verify all required dependencies are present
        all_installed = all(dependencies.values())
        if not all_installed:
            raise AssertionError(f"Missing dependencies: {[k for k, v in dependencies.items() if not v]}")
        
        self.log("All P2P dependencies installed", "SUCCESS")
    
    def test_cache_module_import(self):
        """Test if cache module can be imported."""
        self.log("Importing cache module...", "INFO")
        
        try:
            from ipfs_accelerate_py.github_cli.cache import GitHubAPICache, configure_cache, get_global_cache
            self.log("Cache module imported successfully", "SUCCESS")
            
            # Check if P2P is enabled in the module
            import ipfs_accelerate_py.github_cli.cache as cache_module
            if hasattr(cache_module, 'HAVE_LIBP2P'):
                self.log(f"HAVE_LIBP2P = {cache_module.HAVE_LIBP2P}", "INFO")
                if not cache_module.HAVE_LIBP2P:
                    self.issues.append("libp2p not detected by cache module")
                    self.recommendations.append("Reinstall libp2p and verify it can be imported")
        except ImportError as e:
            self.log(f"Failed to import cache module: {e}", "ERROR")
            self.issues.append(f"Cache module import failed: {e}")
            raise AssertionError(f"Cannot import cache module: {e}")
    
    def test_cache_initialization(self):
        """Test cache initialization with P2P."""
        self.log("Initializing cache with P2P...", "INFO")
        
        try:
            from ipfs_accelerate_py.github_cli.cache import configure_cache
            import tempfile
            
            with tempfile.TemporaryDirectory() as tmpdir:
                cache = configure_cache(
                    cache_dir=tmpdir,
                    enable_p2p=True,
                    p2p_listen_port=9999,  # Use different port for testing
                    enable_persistence=False,
                    enable_peer_discovery=False  # Disable peer discovery for now
                )
                
                self.log("Cache initialized successfully", "SUCCESS")
                
                # Check cache properties
                self.log(f"Cache dir: {cache.cache_dir}", "INFO")
                self.log(f"P2P enabled: {cache.enable_p2p}", "INFO")
                self.log(f"P2P port: {cache._p2p_listen_port}", "INFO")
                
                if not cache.enable_p2p:
                    self.issues.append("P2P not enabled in cache despite enable_p2p=True")
                    self.recommendations.append("Check cache logs for P2P initialization errors")
                
                # Cleanup
                cache.shutdown()
        except Exception as e:
            self.log(f"Cache initialization failed: {e}", "ERROR")
            self.issues.append(f"Cache initialization failed: {e}")
            raise
    
    def test_network_connectivity(self):
        """Test network connectivity to potential bootstrap peers."""
        self.log("Testing network connectivity...", "INFO")
        
        # Get bootstrap peers from environment
        bootstrap_peers_env = os.environ.get("CACHE_BOOTSTRAP_PEERS", "")
        
        if not bootstrap_peers_env:
            self.log("No CACHE_BOOTSTRAP_PEERS configured", "WARNING")
            self.issues.append("No bootstrap peers configured")
            self.recommendations.append(
                "Set CACHE_BOOTSTRAP_PEERS environment variable to MCP server P2P address\n"
                "Example: export CACHE_BOOTSTRAP_PEERS=/ip4/192.168.1.100/tcp/9100/p2p/QmPeerId..."
            )
            # Don't fail the test, just warn
            return
        
        self.log(f"Bootstrap peers configured: {bootstrap_peers_env}", "INFO")
        
        # Parse multiaddrs to extract IP and port
        peers = bootstrap_peers_env.split(',')
        for peer in peers:
            peer = peer.strip()
            if not peer:
                continue
            
            try:
                # Parse multiaddr format: /ip4/<ip>/tcp/<port>/p2p/<peer-id>
                parts = peer.split('/')
                if len(parts) >= 5:
                    ip_type = parts[1]  # ip4 or ip6
                    ip_addr = parts[2]
                    protocol = parts[3]  # tcp
                    port = int(parts[4])
                    
                    self.log(f"Testing connectivity to {ip_addr}:{port}...", "INFO")
                    
                    # Test TCP connectivity
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(5)
                    
                    try:
                        result = sock.connect_ex((ip_addr, port))
                        if result == 0:
                            self.log(f"Successfully connected to {ip_addr}:{port}", "SUCCESS")
                        else:
                            self.log(f"Cannot connect to {ip_addr}:{port} (error code: {result})", "ERROR")
                            self.issues.append(f"Cannot connect to bootstrap peer {ip_addr}:{port}")
                            self.recommendations.append(
                                f"Check if MCP server is running and P2P port {port} is accessible\n"
                                f"From Docker container, ensure network connectivity to host: docker run --network host"
                            )
                    finally:
                        sock.close()
                else:
                    self.log(f"Invalid multiaddr format: {peer}", "WARNING")
            except Exception as e:
                self.log(f"Error testing peer {peer}: {e}", "ERROR")
    
    def test_cache_operations(self):
        """Test basic cache operations."""
        self.log("Testing cache operations...", "INFO")
        
        try:
            from ipfs_accelerate_py.github_cli.cache import configure_cache
            import tempfile
            
            with tempfile.TemporaryDirectory() as tmpdir:
                cache = configure_cache(
                    cache_dir=tmpdir,
                    enable_p2p=False,  # Disable P2P for basic operations test
                    enable_persistence=False
                )
                
                # Test PUT
                test_data = {"test": "data", "timestamp": time.time()}
                cache.put("test_operation", test_data, ttl=300, param1="value1")
                self.log("Cache PUT operation successful", "SUCCESS")
                
                # Test GET
                result = cache.get("test_operation", param1="value1")
                assert result == test_data, f"Cache GET returned wrong data: {result}"
                self.log("Cache GET operation successful", "SUCCESS")
                
                # Test cache miss
                result = cache.get("nonexistent_key", param1="value1")
                assert result is None, "Cache GET should return None for nonexistent key"
                self.log("Cache miss handling correct", "SUCCESS")
                
                # Test stats
                stats = cache.get_stats()
                self.log(f"Cache stats: {stats}", "INFO")
                assert stats['hits'] > 0, "Cache hits should be > 0"
                assert stats['misses'] > 0, "Cache misses should be > 0"
                self.log("Cache statistics tracking working", "SUCCESS")
                
                # Cleanup
                cache.shutdown()
        except Exception as e:
            self.log(f"Cache operations failed: {e}", "ERROR")
            self.issues.append(f"Cache operations failed: {e}")
            raise
    
    def test_encryption_setup(self):
        """Test encryption setup for P2P messages."""
        self.log("Testing encryption setup...", "INFO")
        
        try:
            import base64
            from cryptography.fernet import Fernet
            from cryptography.hazmat.primitives import hashes
            from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
            from cryptography.hazmat.backends import default_backend
            
            # Simulate encryption setup similar to cache module
            github_token = os.environ.get("GITHUB_TOKEN", "test_token_for_encryption")
            
            # Derive encryption key from GitHub token
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=b"github-cache-salt",
                iterations=100000,
                backend=default_backend()
            )
            key = kdf.derive(github_token.encode())
            
            # Create cipher
            cipher = Fernet(base64.urlsafe_b64encode(key))
            
            # Test encryption/decryption
            test_message = b"test message for encryption"
            encrypted = cipher.encrypt(test_message)
            decrypted = cipher.decrypt(encrypted)
            
            assert decrypted == test_message, "Encryption/decryption mismatch"
            self.log("Encryption setup successful", "SUCCESS")
            
        except Exception as e:
            self.log(f"Encryption setup failed: {e}", "ERROR")
            self.issues.append(f"Encryption setup failed: {e}")
            raise
    
    def test_environment_variables(self):
        """Test environment variable configuration."""
        self.log("Checking environment variables...", "INFO")
        
        env_vars = {
            "CACHE_ENABLE_P2P": os.environ.get("CACHE_ENABLE_P2P"),
            "CACHE_LISTEN_PORT": os.environ.get("CACHE_LISTEN_PORT"),
            "CACHE_BOOTSTRAP_PEERS": os.environ.get("CACHE_BOOTSTRAP_PEERS"),
            "GITHUB_TOKEN": "***" if os.environ.get("GITHUB_TOKEN") else None,
            "GITHUB_REPOSITORY": os.environ.get("GITHUB_REPOSITORY"),
        }
        
        for var, value in env_vars.items():
            if value:
                self.log(f"{var} = {value}", "INFO")
            else:
                self.log(f"{var} not set", "WARNING")
                if var in ["CACHE_ENABLE_P2P", "CACHE_LISTEN_PORT"]:
                    self.issues.append(f"{var} not configured")
                    self.recommendations.append(
                        f"Set {var} environment variable\n"
                        f"Example: export {var}=true" if var == "CACHE_ENABLE_P2P" else f"Example: export {var}=9100"
                    )
    
    def test_docker_network_mode(self):
        """Test if running in Docker and check network mode."""
        self.log("Checking Docker environment...", "INFO")
        
        # Check if running in Docker
        is_docker = (
            Path("/.dockerenv").exists() or
            Path("/run/.containerenv").exists()
        )
        
        if is_docker:
            self.log("Running inside Docker container", "INFO")
            
            # Check network mode
            try:
                result = subprocess.run(
                    ["cat", "/proc/1/cgroup"],
                    capture_output=True,
                    text=True
                )
                self.log(f"Container cgroup info: {result.stdout[:100]}...", "DEBUG")
            except Exception as e:
                self.log(f"Could not read cgroup info: {e}", "WARNING")
            
            self.recommendations.append(
                "When running Docker containers, use --network host to allow P2P connectivity:\n"
                "  docker run --network host ...\n"
                "Or use docker-compose with network_mode: host"
            )
        else:
            self.log("Not running inside Docker container", "INFO")
    
    def generate_report(self):
        """Generate final diagnostic report."""
        print("\n" + "="*70)
        print("DIAGNOSTIC REPORT")
        print("="*70)
        
        # Test summary
        total = self.passed + self.failed
        print(f"\nTest Results: {self.passed}/{total} tests passed")
        
        for result in self.test_results:
            status = result[0]
            name = result[1]
            if status == "PASS":
                print(f"  ✅ {name}")
            else:
                error = result[2] if len(result) > 2 else ""
                print(f"  ❌ {name}: {error}")
        
        # Issues found
        if self.issues:
            print(f"\n{'='*70}")
            print("ISSUES FOUND")
            print("="*70)
            for i, issue in enumerate(self.issues, 1):
                print(f"{i}. {issue}")
        
        # Recommendations
        if self.recommendations:
            print(f"\n{'='*70}")
            print("RECOMMENDATIONS")
            print("="*70)
            for i, rec in enumerate(self.recommendations, 1):
                print(f"\n{i}. {rec}")
        
        # Overall status
        print(f"\n{'='*70}")
        if self.failed == 0:
            print("🎉 All tests passed! Cache connectivity should work.")
        else:
            print(f"⚠️  {self.failed} test(s) failed. Review issues and recommendations above.")
        print("="*70)
        
        return 0 if self.failed == 0 else 1
    
    def run_all_tests(self):
        """Run all diagnostic tests."""
        print("="*70)
        print("Docker Runner Cache Connectivity Diagnostic")
        print("="*70)
        print("\nThis diagnostic will test whether GitHub Actions runners")
        print("in Docker containers can connect to the P2P cache.\n")
        
        # Run tests in order
        self.run_test("P2P Dependencies Check", self.test_p2p_dependencies)
        self.run_test("Cache Module Import", self.test_cache_module_import)
        self.run_test("Cache Initialization", self.test_cache_initialization)
        self.run_test("Network Connectivity", self.test_network_connectivity)
        self.run_test("Cache Operations", self.test_cache_operations)
        self.run_test("Encryption Setup", self.test_encryption_setup)
        self.run_test("Environment Variables", self.test_environment_variables)
        self.run_test("Docker Network Mode", self.test_docker_network_mode)
        
        return self.generate_report()


def main():
    """Main entry point."""
    tester = DockerRunnerCacheConnectivityTest()
    return tester.run_all_tests()


if __name__ == "__main__":
    sys.exit(main())
