#!/usr/bin/env python3
"""
Phase 10b: Fix more remaining relative imports.
Focus on the largest remaining categories.
"""

import os
import re
from pathlib import Path

def fix_file_imports(file_path, replacements):
    """Fix imports in a single file."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        original_content = content
        modified = False
        
        for pattern, replacement in replacements:
            if pattern.search(content):
                content = pattern.sub(replacement, content)
                modified = True
        
        if modified and content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        return False
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def fix_distributed_testing_more():
    """Fix more imports in distributed testing directory."""
    base_path = Path("/home/runner/work/ipfs_accelerate_py/ipfs_accelerate_py/test/tests/distributed/distributed_testing")
    
    # Comprehensive list of modules
    modules = [
        'task_scheduler', 'worker', 'coordinator', 'circuit_breaker', 'plugin_architecture', 
        'plugin_base', 'error_recovery_with_performance_tracking', 'distributed_error_handler',
        'error_recovery_strategies', 'hardware_capability_detector', 'coordinator_redundancy',
        'hardware_aware_scheduler', 'result_aggregator', 'adaptive_circuit_breaker',
        'browser_failure_injector', 'load_balancer_integration', 'load_balancer_resource_pool_bridge',
        'resource_pool_bridge', 'selenium_browser_bridge', 'hardware_aware_visualization',
    ]
    
    replacements = []
    for module in modules:
        replacements.append((
            re.compile(rf'from \.{module} import'),
            f'from test.tests.distributed.distributed_testing.{module} import'
        ))
        replacements.append((
            re.compile(rf'from \.\.{module} import'),
            f'from test.tests.distributed.distributed_testing.{module} import'
        ))
    
    fixed_count = 0
    # Fix in tests subdirectory
    tests_dir = base_path / "tests"
    if tests_dir.exists():
        for file_path in tests_dir.glob("*.py"):
            if file_path.is_file():
                if fix_file_imports(file_path, replacements):
                    fixed_count += 1
                    print(f"Fixed: {file_path.relative_to(base_path.parent)}")
    
    # Fix in plugins subdirectory
    plugins_dir = base_path / "plugins"
    if plugins_dir.exists():
        for file_path in plugins_dir.rglob("*.py"):
            if file_path.is_file():
                if fix_file_imports(file_path, replacements):
                    fixed_count += 1
                    print(f"Fixed: {file_path.relative_to(base_path.parent)}")
    
    # Fix in external_systems subdirectory
    ext_dir = base_path / "external_systems"
    if ext_dir.exists():
        for file_path in ext_dir.glob("*.py"):
            if file_path.is_file():
                if fix_file_imports(file_path, replacements):
                    fixed_count += 1
                    print(f"Fixed: {file_path.relative_to(base_path.parent)}")
    
    # Fix in result_aggregator subdirectory
    result_dir = base_path / "result_aggregator"
    if result_dir.exists():
        for file_path in result_dir.glob("*.py"):
            if file_path.is_file():
                if fix_file_imports(file_path, replacements):
                    fixed_count += 1
                    print(f"Fixed: {file_path.relative_to(base_path.parent)}")
    
    # Fix in examples subdirectory
    examples_dir = base_path / "examples"
    if examples_dir.exists():
        for file_path in examples_dir.glob("*.py"):
            if file_path.is_file():
                if fix_file_imports(file_path, replacements):
                    fixed_count += 1
                    print(f"Fixed: {file_path.relative_to(base_path.parent)}")
    
    return fixed_count

def fix_ipfs_accelerate_py_tests_worker():
    """Fix imports in ipfs_accelerate_py_tests/worker directory."""
    base_path = Path("/home/runner/work/ipfs_accelerate_py/ipfs_accelerate_py/test/tests/other/ipfs_accelerate_py_tests/worker")
    
    replacements = [
        # Worker internal imports
        (re.compile(r'from \.worker_utils import'), 
         'from test.tests.other.ipfs_accelerate_py_tests.worker.worker_utils import'),
        (re.compile(r'from \.worker_config import'), 
         'from test.tests.other.ipfs_accelerate_py_tests.worker.worker_config import'),
    ]
    
    fixed_count = 0
    if base_path.exists():
        for file_path in base_path.glob("*.py"):
            if file_path.is_file():
                if fix_file_imports(file_path, replacements):
                    fixed_count += 1
                    print(f"Fixed: {file_path.relative_to(base_path.parent.parent.parent)}")
    
    return fixed_count

def fix_duckdb_api_load_balancer():
    """Fix imports in duckdb_api load_balancer directory."""
    base_path = Path("/home/runner/work/ipfs_accelerate_py/ipfs_accelerate_py/test/tests/api/duckdb_api/distributed_testing/load_balancer")
    
    replacements = [
        # Load balancer relative imports
        (re.compile(r'from \.resource_pool import'), 
         'from test.tests.api.duckdb_api.distributed_testing.load_balancer.resource_pool import'),
        (re.compile(r'from \.load_balancer_base import'), 
         'from test.tests.api.duckdb_api.distributed_testing.load_balancer.load_balancer_base import'),
        (re.compile(r'from \.strategies import'), 
         'from test.tests.api.duckdb_api.distributed_testing.load_balancer.strategies import'),
    ]
    
    fixed_count = 0
    if base_path.exists():
        for file_path in base_path.glob("*.py"):
            if file_path.is_file():
                if fix_file_imports(file_path, replacements):
                    fixed_count += 1
                    print(f"Fixed: {file_path.relative_to(base_path.parent.parent)}")
    
    return fixed_count

def fix_refactored_benchmark_hardware():
    """Fix imports in refactored_benchmark_suite/hardware directory."""
    base_path = Path("/home/runner/work/ipfs_accelerate_py/ipfs_accelerate_py/test/tools/skills/refactored_benchmark_suite/hardware")
    
    replacements = [
        # Hardware module imports
        (re.compile(r'from \.\.benchmark import'), 
         'from test.tools.skills.refactored_benchmark_suite.benchmark import'),
        (re.compile(r'from \.\.metrics import'), 
         'from test.tools.skills.refactored_benchmark_suite.metrics import'),
        (re.compile(r'from \.\.utils import'), 
         'from test.tools.skills.refactored_benchmark_suite.utils import'),
        (re.compile(r'from \.hardware_detector import'), 
         'from test.tools.skills.refactored_benchmark_suite.hardware.hardware_detector import'),
    ]
    
    fixed_count = 0
    if base_path.exists():
        for file_path in base_path.glob("*.py"):
            if file_path.is_file():
                if fix_file_imports(file_path, replacements):
                    fixed_count += 1
                    print(f"Fixed: {file_path.relative_to(base_path.parent.parent.parent)}")
    
    return fixed_count

def fix_web_unified_framework():
    """Fix imports in web unified_framework directory."""
    base_path = Path("/home/runner/work/ipfs_accelerate_py/ipfs_accelerate_py/test/tests/web/fixed_web_platform/unified_framework")
    
    replacements = [
        # Unified framework relative imports
        (re.compile(r'from \.platform_detector import'), 
         'from test.tests.web.fixed_web_platform.unified_framework.platform_detector import'),
        (re.compile(r'from \.fallback_manager import'), 
         'from test.tests.web.fixed_web_platform.unified_framework.fallback_manager import'),
        (re.compile(r'from \.multimodal_integration import'), 
         'from test.tests.web.fixed_web_platform.unified_framework.multimodal_integration import'),
        (re.compile(r'from \.string_utils import'), 
         'from test.tests.web.fixed_web_platform.unified_framework.string_utils import'),
    ]
    
    fixed_count = 0
    if base_path.exists():
        for file_path in base_path.glob("*.py"):
            if file_path.is_file():
                if fix_file_imports(file_path, replacements):
                    fixed_count += 1
                    print(f"Fixed: {file_path.relative_to(base_path.parent.parent)}")
    
    return fixed_count

def fix_android_test_harness():
    """Fix imports in android_test_harness directory."""
    base_path = Path("/home/runner/work/ipfs_accelerate_py/ipfs_accelerate_py/test/tests/mobile/android_test_harness")
    
    replacements = [
        # Android test harness imports
        (re.compile(r'from \.test_runner import'), 
         'from test.tests.mobile.android_test_harness.test_runner import'),
        (re.compile(r'from \.device_manager import'), 
         'from test.tests.mobile.android_test_harness.device_manager import'),
    ]
    
    fixed_count = 0
    if base_path.exists():
        for file_path in base_path.glob("*.py"):
            if file_path.is_file():
                if fix_file_imports(file_path, replacements):
                    fixed_count += 1
                    print(f"Fixed: {file_path.relative_to(base_path.parent.parent)}")
    
    return fixed_count

def main():
    """Run all Phase 10b fixes."""
    print("="*80)
    print("PHASE 10B: FIXING MORE REMAINING RELATIVE IMPORTS")
    print("="*80)
    print()
    
    total_fixed = 0
    
    print("1. Fixing more distributed_testing imports...")
    total_fixed += fix_distributed_testing_more()
    print()
    
    print("2. Fixing ipfs_accelerate_py_tests/worker...")
    total_fixed += fix_ipfs_accelerate_py_tests_worker()
    print()
    
    print("3. Fixing duckdb_api load_balancer...")
    total_fixed += fix_duckdb_api_load_balancer()
    print()
    
    print("4. Fixing refactored_benchmark_suite/hardware...")
    total_fixed += fix_refactored_benchmark_hardware()
    print()
    
    print("5. Fixing web unified_framework...")
    total_fixed += fix_web_unified_framework()
    print()
    
    print("6. Fixing android_test_harness...")
    total_fixed += fix_android_test_harness()
    print()
    
    print("="*80)
    print(f"PHASE 10B COMPLETE: Fixed {total_fixed} files")
    print("="*80)
    
    return total_fixed

if __name__ == "__main__":
    main()
