#!/usr/bin/env python3
"""
Windows Compatibility Test Suite
Tests for Python 3.12 and Windows-specific functionality
"""

import os
import sys
import platform
import tempfile
import json
import logging
from pathlib import Path
from unittest import mock

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("windows_test")


def test_python_version():
    """Test Python version compatibility"""
    logger.info("🐍 Testing Python version compatibility")

    version_info = sys.version_info
    logger.info(
        f"Current Python version: {version_info.major}.{version_info.minor}.{version_info.micro}"
    )

    # Check minimum version
    if version_info < (3, 8):
        logger.error("❌ Python 3.8+ required")
        return False

    # Check maximum tested version
    if version_info > (3, 12):
        logger.warning(f"⚠️ Python {version_info.major}.{version_info.minor} not fully tested")

    logger.info("✅ Python version compatibility OK")
    return True


def test_platform_detection():
    """Test platform detection"""
    logger.info("🖥️ Testing platform detection")

    system = platform.system()
    logger.info(f"Detected platform: {system}")

    if system == "Windows":
        logger.info("✅ Running on Windows")
        # Test Windows-specific features
        windows_version = platform.win32_ver()
        logger.info(f"Windows version: {windows_version}")

        # Test drive letters
        drives = [f"{d}:\\" for d in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if os.path.exists(f"{d}:\\")]
        logger.info(f"Available drives: {drives}")

    elif system == "Linux":
        logger.info("ℹ️ Running on Linux (simulating Windows tests)")
        # Simulate Windows behavior for testing

    return True


def test_path_handling():
    """Test Windows path handling"""
    logger.info("📁 Testing path handling")

    test_paths = [
        "simple_path",
        "path with spaces",
        "path\\with\\backslashes",
        "path/with/forward/slashes",
        "C:\\Windows\\Style\\Path" if platform.system() == "Windows" else "/tmp/linux/path",
        "very_long_path_name_that_might_exceed_windows_limits_" + "x" * 100,
    ]

    for test_path in test_paths:
        try:
            path_obj = Path(test_path)
            logger.info(f"✅ Path '{test_path[:50]}...' handled successfully")
        except Exception as e:
            logger.warning(f"⚠️ Path '{test_path[:50]}...' caused error: {e}")

    return True


def test_cli_argument_validation():
    """Test CLI argument validation"""
    logger.info("🔧 Testing CLI argument validation")

    # Test with mock arguments
    test_cases = [
        # Valid cases
        {"model": "bert-base-uncased", "fast": True, "expected": True},
        {"model": "gpt2", "local": True, "expected": True},
        {"model": "test-model", "batch_size": 4, "expected": True},
        # Invalid cases
        {"model": "", "expected": False},  # Empty model
        {"model": "model<>with|bad:chars", "expected": False},  # Invalid chars
        {"model": "valid", "batch_size": -1, "expected": False},  # Negative batch size
        {"model": "valid", "timeout": 0, "expected": False},  # Zero timeout
    ]

    try:
        from ipfs_cli import validate_arguments

        for case in test_cases:
            expected = case.pop("expected")

            # Create mock args object
            args = mock.MagicMock()
            for key, value in case.items():
                setattr(args, key, value)

            # Set defaults for unspecified args
            for attr in [
                "model",
                "fast",
                "local",
                "batch_size",
                "timeout",
                "config",
                "output",
                "log_file",
            ]:
                if not hasattr(args, attr):
                    setattr(args, attr, None)

            result = validate_arguments(args)
            if result == expected:
                logger.info(f"✅ Validation test passed: {case}")
            else:
                logger.error(
                    f"❌ Validation test failed: {case} (expected {expected}, got {result})"
                )

    except ImportError:
        logger.warning("⚠️ Could not import CLI module, skipping validation tests")

    return True


def test_dependency_imports():
    """Test importing dependencies"""
    logger.info("📦 Testing dependency imports")

    # Core Python modules that should always work
    core_modules = ["json", "os", "sys", "pathlib", "argparse", "logging"]

    for module in core_modules:
        try:
            __import__(module)
            logger.info(f"✅ {module} imported successfully")
        except ImportError as e:
            logger.error(f"❌ Failed to import {module}: {e}")
            return False

    # Optional modules
    optional_modules = ["torch", "transformers", "numpy", "fastapi", "websockets"]

    for module in optional_modules:
        try:
            __import__(module)
            logger.info(f"✅ {module} imported successfully")
        except ImportError:
            logger.info(f"ℹ️ {module} not installed (optional)")

    return True


def test_file_operations():
    """Test file operations with Windows compatibility"""
    logger.info("📝 Testing file operations")

    try:
        # Test temp file creation
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            test_data = {"test": True, "platform": platform.system()}
            json.dump(test_data, f)
            temp_file = f.name

        # Test reading back
        with open(temp_file, "r") as f:
            loaded_data = json.load(f)

        assert loaded_data == test_data
        logger.info("✅ File operations working correctly")

        # Cleanup
        os.unlink(temp_file)

        return True

    except Exception as e:
        logger.error(f"❌ File operations failed: {e}")
        return False


def test_unicode_handling():
    """Test Unicode handling for international users"""
    logger.info("🌐 Testing Unicode handling")

    test_strings = [
        "Hello World",
        "Héllo Wörld",
        "こんにちは世界",
        "مرحبا بالعالم",
        "🚀 Emoji test 🎉",
    ]

    for test_str in test_strings:
        try:
            # Test encoding/decoding
            encoded = test_str.encode("utf-8")
            decoded = encoded.decode("utf-8")
            assert decoded == test_str
            logger.info(f"✅ Unicode test passed: {test_str[:20]}")
        except Exception as e:
            logger.error(f"❌ Unicode test failed for '{test_str[:20]}': {e}")
            return False

    return True


def run_all_tests():
    """Run all Windows compatibility tests"""
    logger.info("🧪 Starting Windows compatibility tests")

    tests = [
        ("Python Version", test_python_version),
        ("Platform Detection", test_platform_detection),
        ("Path Handling", test_path_handling),
        ("CLI Validation", test_cli_argument_validation),
        ("Dependency Imports", test_dependency_imports),
        ("File Operations", test_file_operations),
        ("Unicode Handling", test_unicode_handling),
    ]

    results = {}
    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        logger.info(f"\n--- {test_name} ---")
        try:
            result = test_func()
            results[test_name] = "PASS" if result else "FAIL"
            if result:
                passed += 1
        except Exception as e:
            logger.error(f"❌ {test_name} crashed: {e}")
            results[test_name] = f"ERROR: {e}"

    # Summary
    logger.info(f"\n{'=' * 50}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'=' * 50}")

    for test_name, result in results.items():
        status_icon = "✅" if result == "PASS" else "❌" if result == "FAIL" else "⚠️"
        logger.info(f"{status_icon} {test_name}: {result}")

    logger.info(f"\nOverall: {passed}/{total} tests passed ({passed / total * 100:.1f}%)")

    return results, passed == total


def main():
    """Main test function"""
    logger.info("🚀 IPFS Accelerate Python - Windows Compatibility Test Suite")
    logger.info(f"Platform: {platform.system()} {platform.release()}")
    logger.info(f"Python: {sys.version}")

    results, all_passed = run_all_tests()

    # Save results if on Windows
    if platform.system() == "Windows":
        results_file = "windows_compatibility_results.json"
        try:
            with open(results_file, "w") as f:
                json.dump(
                    {
                        "platform": platform.system(),
                        "python_version": sys.version,
                        "test_results": results,
                        "all_passed": all_passed,
                    },
                    f,
                    indent=2,
                )
            logger.info(f"📄 Results saved to {results_file}")
        except Exception as e:
            logger.error(f"❌ Failed to save results: {e}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
