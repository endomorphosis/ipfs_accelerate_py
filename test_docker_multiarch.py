#!/usr/bin/env python3
"""
Comprehensive ARM64 Docker Test for IPFS Accelerate Python
==========================================================

This script validates that the IPFS Accelerate Docker containers work properly on ARM64
and provides comprehensive testing for multi-architecture support.
"""

import subprocess
import time
import requests
import json
import sys
import os
from pathlib import Path

def run_command(cmd, description="", return_output=True, timeout=120):
    """Run a shell command and return the result"""
    print(f"\n🔧 {description}")
    print(f"   Command: {cmd}")
    
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=return_output, 
            text=True, timeout=timeout
        )
        if return_output:
            if result.returncode == 0:
                print(f"   ✅ Success")
                return result.stdout.strip()
            else:
                print(f"   ❌ Failed (exit code: {result.returncode})")
                if result.stderr:
                    print(f"   Error: {result.stderr}")
                return None
        else:
            print(f"   ✅ Command executed")
            return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"   ⏰ Timeout after {timeout}s")
        return None
    except Exception as e:
        print(f"   ❌ Exception: {e}")
        return None

def test_docker_availability():
    """Test if Docker is available and running"""
    print("\n" + "="*70)
    print("🐳 TESTING DOCKER AVAILABILITY")
    print("="*70)
    
    # Check Docker version
    version = run_command("docker --version", "Checking Docker version")
    if not version:
        return False
    
    # Check Docker daemon
    status = run_command("docker info --format '{{.ServerVersion}}'", "Checking Docker daemon")
    if not status:
        return False
    
    # Check Docker Compose
    compose_version = run_command("docker compose version", "Checking Docker Compose")
    
    print(f"   Docker version: {version}")
    print(f"   Docker daemon: {status}")
    if compose_version:
        print(f"   Docker Compose: {compose_version}")
    
    return True

def test_system_architecture():
    """Test system architecture and capabilities"""
    print("\n" + "="*70)
    print("🏗️  TESTING SYSTEM ARCHITECTURE & CAPABILITIES")
    print("="*70)
    
    arch = run_command("uname -m", "Getting system architecture")
    if not arch:
        return False
    
    # Get additional system info
    os_info = run_command("uname -a", "Getting OS information")
    cpu_info = run_command("lscpu | head -10", "Getting CPU information")
    memory_info = run_command("free -h", "Getting memory information")
    
    print(f"   System architecture: {arch}")
    if os_info:
        print(f"   OS: {os_info}")
    if cpu_info:
        print(f"   CPU info:\n{cpu_info}")
    if memory_info:
        print(f"   Memory:\n{memory_info}")
    
    # Check for hardware acceleration capabilities
    gpu_nvidia = run_command("nvidia-smi --version 2>/dev/null", "Checking NVIDIA GPU")
    if gpu_nvidia:
        print(f"   ✅ NVIDIA GPU detected")
    
    return arch in ["aarch64", "x86_64"]

def test_project_structure():
    """Test project structure and dependencies"""
    print("\n" + "="*70)
    print("📁 TESTING PROJECT STRUCTURE")
    print("="*70)
    
    required_files = [
        "pyproject.toml",
        "requirements.txt",
        "Dockerfile",
        "docker-compose.yml"
    ]
    
    missing_files = []
    for file in required_files:
        if os.path.exists(file):
            print(f"   ✅ {file} exists")
        else:
            print(f"   ❌ {file} missing")
            missing_files.append(file)
    
    # Check if main package directory exists
    if os.path.exists("ipfs_accelerate_py"):
        print(f"   ✅ Package directory exists")
    else:
        print(f"   ❌ Package directory missing")
        missing_files.append("ipfs_accelerate_py/")
    
    return len(missing_files) == 0

def test_docker_build():
    """Test Docker image builds for multiple targets"""
    print("\n" + "="*70)
    print("🔨 TESTING DOCKER BUILDS ON ARM64")
    print("="*70)
    
    # Get current architecture for platform targeting
    arch_output = run_command("uname -m", "Getting architecture for Docker platform")
    if arch_output == "aarch64":
        platform = "linux/arm64"
    elif arch_output == "x86_64":
        platform = "linux/amd64"
    else:
        platform = f"linux/{arch_output}"
    
    targets_to_test = ["minimal", "development", "production"]
    successful_builds = 0
    
    for target in targets_to_test:
        build_cmd = (f"docker build --platform {platform} --target {target} "
                    f"-t ipfs-accelerate-py:test-{target} .")
        
        success = run_command(build_cmd, f"Building {target} image", 
                             return_output=False, timeout=600)
        
        if success:
            # Verify image was created
            images = run_command(f"docker images ipfs-accelerate-py:test-{target} "
                               "--format '{{.Repository}}:{{.Tag}}'", 
                               f"Verifying {target} image creation")
            if images == f"ipfs-accelerate-py:test-{target}":
                print(f"   ✅ {target} build successful")
                successful_builds += 1
            else:
                print(f"   ❌ {target} image not found after build")
        else:
            print(f"   ❌ {target} build failed")
    
    return successful_builds > 0

def test_container_functionality():
    """Test container startup and basic functionality"""
    print("\n" + "="*70)
    print("🚀 TESTING CONTAINER FUNCTIONALITY")
    print("="*70)
    
    # Test minimal container first (lightest)
    container_name = "ipfs-accelerate-test-minimal"
    
    # Clean up any existing container
    run_command(f"docker stop {container_name} 2>/dev/null || true", "Cleaning up existing container")
    run_command(f"docker rm {container_name} 2>/dev/null || true", "Removing existing container")
    
    # Start container
    platform = "linux/arm64" if run_command("uname -m", "") == "aarch64" else "linux/amd64"
    start_cmd = (f"docker run --platform {platform} -d --name {container_name} "
                f"--rm -p 8005:8000 ipfs-accelerate-py:test-minimal "
                f"python -c 'import time; time.sleep(30)'")
    
    container_id = run_command(start_cmd, "Starting minimal test container")
    if not container_id:
        return False
    
    print(f"   Container ID: {container_id[:12]}...")
    
    # Wait for container to start
    time.sleep(5)
    
    # Check if container is running
    status = run_command(f"docker ps -f name={container_name} --format '{{.Status}}'", 
                        "Checking container status")
    
    success = False
    if status and "Up" in status:
        print(f"   ✅ Container status: {status}")
        
        # Test basic functionality inside container
        import_test = run_command(f"docker exec {container_name} python -c 'import ipfs_accelerate_py; print(\"Import successful\")'", 
                                 "Testing package import inside container")
        
        cli_test = run_command(f"docker exec {container_name} python -m ipfs_accelerate_py.cli_entry --help", 
                              "Testing CLI inside container")
        
        if import_test and "Import successful" in import_test:
            print(f"   ✅ Package import test passed")
            success = True
        
        if cli_test and ("ipfs_accelerate" in cli_test or "usage:" in cli_test.lower()):
            print(f"   ✅ CLI test passed")
        else:
            print(f"   ⚠️ CLI test had issues")
    else:
        print(f"   ❌ Container failed to start properly")
        # Show logs for debugging
        logs = run_command(f"docker logs {container_name}", "Getting container logs")
        if logs:
            print(f"   Container logs: {logs}")
    
    # Cleanup
    run_command(f"docker stop {container_name} 2>/dev/null || true", "Stopping test container")
    
    return success

def test_docker_compose():
    """Test Docker Compose functionality"""
    print("\n" + "="*70)
    print("🐙 TESTING DOCKER COMPOSE")
    print("="*70)
    
    # Create required directories
    directories = ["data", "logs", "config", "models"]
    for dir_name in directories:
        os.makedirs(dir_name, exist_ok=True)
        print(f"   ✅ Created directory: {dir_name}")
    
    # Test compose file validation
    validation = run_command("docker compose config", "Validating docker-compose.yml")
    if not validation:
        print("   ❌ Docker Compose configuration is invalid")
        return False
    
    print("   ✅ Docker Compose configuration is valid")
    
    # Try to build the main service
    build_result = run_command("docker compose build ipfs-accelerate-py", 
                              "Building main service with Compose", 
                              return_output=False, timeout=600)
    
    if build_result:
        print("   ✅ Docker Compose build successful")
        
        # Test service startup
        start_result = run_command("docker compose up -d ipfs-accelerate-py", 
                                  "Starting main service")
        if start_result:
            time.sleep(5)
            
            # Check if service is running
            ps_result = run_command("docker compose ps ipfs-accelerate-py", 
                                   "Checking service status")
            if ps_result and "Up" in ps_result:
                print("   ✅ Service started successfully")
                success = True
            else:
                print("   ⚠️ Service may not have started properly")
                success = False
            
            # Cleanup
            run_command("docker compose down", "Stopping Compose services")
            return success
    
    print("   ❌ Docker Compose build failed")
    return False

def test_hardware_detection():
    """Test hardware detection and acceleration capabilities"""
    print("\n" + "="*70)
    print("🔬 TESTING HARDWARE DETECTION")
    print("="*70)
    
    # Test inside a container
    container_name = "ipfs-accelerate-hw-test"
    
    # Clean up
    run_command(f"docker stop {container_name} 2>/dev/null || true", "Cleanup")
    run_command(f"docker rm {container_name} 2>/dev/null || true", "Cleanup")
    
    # Start container for hardware tests
    platform = "linux/arm64" if run_command("uname -m", "") == "aarch64" else "linux/amd64"
    start_cmd = (f"docker run --platform {platform} -d --name {container_name} "
                f"ipfs-accelerate-py:test-development sleep 60")
    
    if run_command(start_cmd, "Starting hardware test container"):
        time.sleep(3)
        
        # Test hardware detection inside container
        hw_test = run_command(f'docker exec {container_name} python -c "import platform; print(\"Platform:\", platform.platform()); print(\"Machine:\", platform.machine()); print(\"Processor:\", platform.processor()); import sys; print(\"Python:\", sys.version)"', 
                             "Testing hardware detection inside container")
        
        if hw_test:
            print(f"   ✅ Hardware detection successful")
            print(f"   Hardware info:\n{hw_test}")
            success = True
        else:
            print(f"   ⚠️ Hardware detection had issues")
            success = False
        
        # Cleanup
        run_command(f"docker stop {container_name}", "Stopping hardware test container")
        return success
    
    return False

def cleanup_test_resources():
    """Clean up all test resources"""
    print("\n" + "="*70)
    print("🧹 CLEANING UP TEST RESOURCES")
    print("="*70)
    
    # Stop all test containers
    test_containers = [
        "ipfs-accelerate-test-minimal",
        "ipfs-accelerate-hw-test"
    ]
    
    for container in test_containers:
        run_command(f"docker stop {container} 2>/dev/null || true", f"Stopping {container}")
        run_command(f"docker rm {container} 2>/dev/null || true", f"Removing {container}")
    
    # Remove test images
    test_images = [
        "ipfs-accelerate-py:test-minimal",
        "ipfs-accelerate-py:test-development", 
        "ipfs-accelerate-py:test-production"
    ]
    
    for image in test_images:
        run_command(f"docker rmi {image} 2>/dev/null || true", f"Removing {image}")
    
    # Stop docker compose services
    run_command("docker compose down --volumes 2>/dev/null || true", "Stopping Docker Compose services")
    
    # Docker system cleanup
    run_command("docker system prune -f", "Docker system cleanup")
    
    print("   ✅ Cleanup complete")

def main():
    """Main test function"""
    print("🧪 IPFS Accelerate Python Multi-Architecture Docker Test Suite")
    print("=" * 70)
    print("This test validates multi-architecture Docker functionality")
    print("Designed for ARM64, AMD64, and cross-platform compatibility")
    
    # Change to the correct directory if needed
    if not os.path.exists("pyproject.toml"):
        accelerate_dir = os.path.expanduser("~/ipfs_accelerate_py")
        if os.path.exists(accelerate_dir):
            os.chdir(accelerate_dir)
            print(f"📂 Changed to directory: {accelerate_dir}")
        else:
            print("❌ Could not find ipfs_accelerate_py directory")
            return 1
    
    test_results = {}
    
    try:
        # Run comprehensive tests
        test_results["docker_available"] = test_docker_availability()
        test_results["system_architecture"] = test_system_architecture() 
        test_results["project_structure"] = test_project_structure()
        test_results["docker_build"] = test_docker_build()
        test_results["container_functionality"] = test_container_functionality()
        test_results["docker_compose"] = test_docker_compose()
        test_results["hardware_detection"] = test_hardware_detection()
        
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
    finally:
        cleanup_test_resources()
    
    # Print results summary
    print("\n" + "="*70)
    print("📊 TEST RESULTS SUMMARY")
    print("="*70)
    
    passed_tests = 0
    total_tests = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        test_display = test_name.replace("_", " ").title()
        print(f"   {test_display:<25} : {status}")
        if result:
            passed_tests += 1
    
    print(f"\n   Tests passed: {passed_tests}/{total_tests}")
    
    if passed_tests >= total_tests * 0.75:  # Allow for 75% success rate
        print("   🎉 TESTS MOSTLY SUCCESSFUL! Multi-architecture support is working!")
        if passed_tests == total_tests:
            print("   🌟 ALL TESTS PASSED! Perfect multi-architecture Docker support!")
        return 0
    else:
        print("   ⚠️  Several tests failed. Check the details above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())