#!/usr/bin/env python3
"""
Test script to validate sudo configuration for GitHub Actions runner
"""

import subprocess
import sys
import os


def test_sudo_access():
    """Test passwordless sudo access"""
    print("🔍 Testing sudo configuration for GitHub Actions runner...")
    
    try:
        # Test basic sudo access
        try:
            result = subprocess.run(['sudo', '-n', 'whoami'], 
                                  capture_output=True, text=True, timeout=10)
        except subprocess.TimeoutExpired:
            print("❌ Passwordless sudo access: TIMED OUT")
            print("   Sudo configuration may be severely misconfigured")
            print("   Check /etc/sudoers.d/ for proper passwordless configuration")
            return False
        
        if result.returncode == 0:
            print("✅ Passwordless sudo access: WORKING")
            print(f"   Command output: {result.stdout.strip()}")
        else:
            print("❌ Passwordless sudo access: FAILED") 
            print(f"   Error: {result.stderr.strip()}")
            return False
            
        # Test apt-get access (common CI/CD requirement)
        apt_update_timeout = int(os.getenv("APT_UPDATE_TIMEOUT", "60"))
        result = subprocess.run(['sudo', '-n', 'apt-get', 'update', '-qq'], 
                              capture_output=True, text=True, timeout=apt_update_timeout)
        
        if result.returncode == 0:
            print("✅ Package manager access: WORKING")
        else:
            print("❌ Package manager access: FAILED")
            print(f"   Error: {result.stderr.strip()}")
            
        # Test system info access
        result = subprocess.run(['sudo', '-n', 'lscpu'], 
                              capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("✅ System information access: WORKING")
        else:
            print("❌ System information access: FAILED")
            print(f"   Error: {result.stderr.strip()}")
            
        return True
        
    except subprocess.TimeoutExpired:
        print("❌ Sudo test timed out - check configuration")
        return False
    except Exception as e:
        print(f"❌ Sudo test failed with exception: {e}")
        return False


def main():
    print("🚀 ARM64 GitHub Actions Runner - Infrastructure Validation")
    print("=" * 60)
    
    print(f"User: {os.getenv('USER', 'unknown')}")
    print(f"Architecture: {os.uname().machine}")
    print(f"Operating System: {os.uname().sysname} {os.uname().release}")
    print()
    
    if test_sudo_access():
        print()
        print("🎉 SUCCESS: Infrastructure configuration is correct!")
        print("   ARM64 CI/CD pipeline should now work without sudo issues")
        return 0
    else:
        print()
        print("❌ FAILED: Infrastructure configuration needs attention")
        print("   Check sudoers configuration in /etc/sudoers.d/")
        return 1


if __name__ == "__main__":
    sys.exit(main())