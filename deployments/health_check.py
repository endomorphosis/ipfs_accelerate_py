#!/usr/bin/env python3
"""Health check script for IPFS Accelerate Python deployment."""

import sys
import time
import requests
from utils.production_validation import run_production_validation


def check_health():
    """Comprehensive health check."""
    checks_passed = 0
    total_checks = 3

    # Check production validation
    try:
        result = run_production_validation("basic")
        if result.overall_score > 80:
            print("✅ Production validation: PASSED")
            checks_passed += 1
        else:
            print(f"❌ Production validation: FAILED (score: {result.overall_score})")
    except Exception as e:
        print(f"❌ Production validation: ERROR ({e})")

    # Check HTTP endpoint (if available)
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("✅ HTTP endpoint: PASSED")
            checks_passed += 1
        else:
            print(f"❌ HTTP endpoint: FAILED (status: {response.status_code})")
    except Exception as e:
        print(f"❌ HTTP endpoint: ERROR ({e})")

    # Check hardware detection
    try:
        from hardware_detection import HardwareDetector

        detector = HardwareDetector()
        hardware = detector.get_available_hardware()
        if hardware:
            print("✅ Hardware detection: PASSED")
            checks_passed += 1
        else:
            print("❌ Hardware detection: FAILED")
    except Exception as e:
        print(f"❌ Hardware detection: ERROR ({e})")

    print(f"\nHealth check: {checks_passed}/{total_checks} checks passed")

    if checks_passed == total_checks:
        print("🎉 System is healthy!")
        return 0
    elif checks_passed >= total_checks * 0.6:
        print("⚠️  System has issues but is operational")
        return 1
    else:
        print("🚨 System is unhealthy!")
        return 2


if __name__ == "__main__":
    sys.exit(check_health())
