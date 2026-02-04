#!/usr/bin/env python3
"""
Phase 10: Fix remaining 277 relative import issues.
This handles the final cleanup of relative imports.
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

def fix_refactored_benchmark_suite():
    """Fix imports in refactored_benchmark_suite package."""
    base_path = Path("/home/runner/work/ipfs_accelerate_py/ipfs_accelerate_py/test/tools/skills/refactored_benchmark_suite")
    
    files_to_fix = [
        "__main__.py",
        "__init__.py",
        "metrics/__init__.py",
        "utils/importers.py",
        "hardware/*.py",
        "models/*.py",
    ]
    
    replacements = [
        # From relative to absolute imports
        (re.compile(r'from \.utils\.logging import'), 
         'from test.tools.skills.refactored_benchmark_suite.utils.logging import'),
        (re.compile(r'from \.visualizers\.dashboard import'), 
         'from test.tools.skills.refactored_benchmark_suite.visualizers.dashboard import'),
        (re.compile(r'from \.config\.benchmark_config import'), 
         'from test.tools.skills.refactored_benchmark_suite.config.benchmark_config import'),
        (re.compile(r'from \.benchmark import'), 
         'from test.tools.skills.refactored_benchmark_suite.benchmark import'),
        (re.compile(r'from \.metrics import'), 
         'from test.tools.skills.refactored_benchmark_suite.metrics import'),
        (re.compile(r'from \.timing import'), 
         'from test.tools.skills.refactored_benchmark_suite.metrics.timing import'),
        (re.compile(r'from \.memory import'), 
         'from test.tools.skills.refactored_benchmark_suite.metrics.memory import'),
        (re.compile(r'from \.flops import'), 
         'from test.tools.skills.refactored_benchmark_suite.metrics.flops import'),
        (re.compile(r'from \.\.benchmark import'), 
         'from test.tools.skills.refactored_benchmark_suite.benchmark import'),
    ]
    
    fixed_count = 0
    for pattern in files_to_fix:
        for file_path in base_path.glob(pattern):
            if file_path.is_file():
                if fix_file_imports(file_path, replacements):
                    fixed_count += 1
                    print(f"Fixed: {file_path.relative_to(base_path.parent)}")
    
    return fixed_count

def fix_distributed_testing_ci():
    """Fix imports in distributed_testing/ci directory."""
    base_path = Path("/home/runner/work/ipfs_accelerate_py/ipfs_accelerate_py/test/tests/distributed/distributed_testing/ci")
    
    replacements = [
        # CI module relative imports to absolute
        (re.compile(r'from \.api_interface import'), 
         'from test.tests.distributed.distributed_testing.ci.api_interface import'),
        (re.compile(r'from \.base_ci_client import'), 
         'from test.tests.distributed.distributed_testing.ci.base_ci_client import'),
        (re.compile(r'from \.github_client import'), 
         'from test.tests.distributed.distributed_testing.ci.github_client import'),
        (re.compile(r'from \.gitlab_client import'), 
         'from test.tests.distributed.distributed_testing.ci.gitlab_client import'),
        (re.compile(r'from \.result_reporter import'), 
         'from test.tests.distributed.distributed_testing.ci.result_reporter import'),
        (re.compile(r'from \.url_validator import'), 
         'from test.tests.distributed.distributed_testing.ci.url_validator import'),
        (re.compile(r'from \.register_providers import'), 
         'from test.tests.distributed.distributed_testing.ci.register_providers import'),
    ]
    
    fixed_count = 0
    for file_path in base_path.glob("*.py"):
        if file_path.is_file():
            if fix_file_imports(file_path, replacements):
                fixed_count += 1
                print(f"Fixed: {file_path.relative_to(base_path.parent.parent)}")
    
    return fixed_count

def fix_distributed_testing_core():
    """Fix imports in distributed_testing main directory."""
    base_path = Path("/home/runner/work/ipfs_accelerate_py/ipfs_accelerate_py/test/tests/distributed/distributed_testing")
    
    replacements = [
        # Core module relative imports
        (re.compile(r'from \.coordinator import'), 
         'from test.tests.distributed.distributed_testing.coordinator import'),
        (re.compile(r'from \.worker import'), 
         'from test.tests.distributed.distributed_testing.worker import'),
        (re.compile(r'from \.circuit_breaker import'), 
         'from test.tests.distributed.distributed_testing.circuit_breaker import'),
        (re.compile(r'from \.task_scheduler import'), 
         'from test.tests.distributed.distributed_testing.task_scheduler import'),
        (re.compile(r'from \.hardware_capability_detector import'), 
         'from test.tests.distributed.distributed_testing.hardware_capability_detector import'),
        (re.compile(r'from \.plugin_architecture import'), 
         'from test.tests.distributed.distributed_testing.plugin_architecture import'),
        (re.compile(r'from \.plugin_base import'), 
         'from test.tests.distributed.distributed_testing.plugin_base import'),
    ]
    
    fixed_count = 0
    for file_path in base_path.glob("*.py"):
        if file_path.is_file() and file_path.name != "__init__.py":
            if fix_file_imports(file_path, replacements):
                fixed_count += 1
                print(f"Fixed: {file_path.relative_to(base_path.parent)}")
    
    return fixed_count

def fix_duckdb_api_tests():
    """Fix imports in duckdb_api test directories."""
    base_paths = [
        Path("/home/runner/work/ipfs_accelerate_py/ipfs_accelerate_py/test/tests/api/duckdb_api/distributed_testing/tests"),
        Path("/home/runner/work/ipfs_accelerate_py/ipfs_accelerate_py/test/tests/api/duckdb_api/distributed_testing/load_balancer"),
    ]
    
    replacements = [
        # Hardware taxonomy imports
        (re.compile(r'from \.\.hardware_taxonomy import'), 
         'from test.tests.api.duckdb_api.distributed_testing.hardware_taxonomy import'),
        (re.compile(r'from \.\.enhanced_hardware_taxonomy import'), 
         'from test.tests.api.duckdb_api.distributed_testing.enhanced_hardware_taxonomy import'),
        (re.compile(r'from \.\.hardware_abstraction_layer import'), 
         'from test.tests.api.duckdb_api.distributed_testing.hardware_abstraction_layer import'),
        (re.compile(r'from \.\.load_balancer import'), 
         'from test.tests.api.duckdb_api.distributed_testing.load_balancer import'),
    ]
    
    fixed_count = 0
    for base_path in base_paths:
        if base_path.exists():
            for file_path in base_path.glob("*.py"):
                if file_path.is_file():
                    if fix_file_imports(file_path, replacements):
                        fixed_count += 1
                        print(f"Fixed: {file_path.relative_to(base_path.parent.parent)}")
    
    return fixed_count

def fix_web_platform_imports():
    """Fix imports in web platform directories."""
    base_path = Path("/home/runner/work/ipfs_accelerate_py/ipfs_accelerate_py/test/tests/web/fixed_web_platform")
    
    replacements = [
        # Web platform relative imports
        (re.compile(r'from \.\.webgpu_quantization import'), 
         'from test.tests.web.fixed_web_platform.webgpu_quantization import'),
        (re.compile(r'from \.\.browser_capability_detector import'), 
         'from test.tests.web.fixed_web_platform.browser_capability_detector import'),
        (re.compile(r'from \.\.webgpu_implementation import'), 
         'from test.tests.web.fixed_web_platform.webgpu_implementation import'),
        (re.compile(r'from \.\.webnn_implementation import'), 
         'from test.tests.web.fixed_web_platform.webnn_implementation import'),
    ]
    
    fixed_count = 0
    for file_path in base_path.rglob("*.py"):
        if file_path.is_file():
            if fix_file_imports(file_path, replacements):
                fixed_count += 1
                print(f"Fixed: {file_path.relative_to(base_path.parent)}")
    
    return fixed_count

def fix_common_test_utils():
    """Fix imports in common test utilities."""
    file_path = Path("/home/runner/work/ipfs_accelerate_py/ipfs_accelerate_py/test/common/test_utils.py")
    
    replacements = [
        (re.compile(r'from \.performance_baseline import'), 
         'from test.common.performance_baseline import'),
    ]
    
    if file_path.exists():
        if fix_file_imports(file_path, replacements):
            print(f"Fixed: {file_path.relative_to(file_path.parent.parent)}")
            return 1
    return 0

def fix_apis_directory():
    """Fix imports in tests/api/apis directory."""
    base_path = Path("/home/runner/work/ipfs_accelerate_py/ipfs_accelerate_py/test/tests/api/apis")
    
    replacements = [
        # API relative imports
        (re.compile(r'from \.base_api import'), 
         'from test.tests.api.apis.base_api import'),
        (re.compile(r'from \.openai_api import'), 
         'from test.tests.api.apis.openai_api import'),
        (re.compile(r'from \.claude_api import'), 
         'from test.tests.api.apis.claude_api import'),
    ]
    
    fixed_count = 0
    if base_path.exists():
        for file_path in base_path.glob("*.py"):
            if file_path.is_file():
                if fix_file_imports(file_path, replacements):
                    fixed_count += 1
                    print(f"Fixed: {file_path.relative_to(base_path.parent.parent)}")
    
    return fixed_count

def fix_plugin_scheduler():
    """Fix the triple-dot import in plugin scheduler."""
    file_path = Path("/home/runner/work/ipfs_accelerate_py/ipfs_accelerate_py/test/tests/distributed/distributed_testing/plugins/scheduler/scheduler_coordinator.py")
    
    replacements = [
        # Triple-dot import
        (re.compile(r'from \.\.\.plugin_architecture import'), 
         'from test.tests.distributed.distributed_testing.plugin_architecture import'),
    ]
    
    if file_path.exists():
        if fix_file_imports(file_path, replacements):
            print(f"Fixed: {file_path.relative_to(file_path.parent.parent.parent.parent)}")
            return 1
    return 0

def main():
    """Run all import fixes."""
    print("="*80)
    print("PHASE 10: FIXING REMAINING RELATIVE IMPORTS")
    print("="*80)
    print()
    
    total_fixed = 0
    
    print("1. Fixing refactored_benchmark_suite...")
    total_fixed += fix_refactored_benchmark_suite()
    print()
    
    print("2. Fixing distributed_testing/ci...")
    total_fixed += fix_distributed_testing_ci()
    print()
    
    print("3. Fixing distributed_testing core...")
    total_fixed += fix_distributed_testing_core()
    print()
    
    print("4. Fixing duckdb_api tests...")
    total_fixed += fix_duckdb_api_tests()
    print()
    
    print("5. Fixing web platform imports...")
    total_fixed += fix_web_platform_imports()
    print()
    
    print("6. Fixing common test utils...")
    total_fixed += fix_common_test_utils()
    print()
    
    print("7. Fixing apis directory...")
    total_fixed += fix_apis_directory()
    print()
    
    print("8. Fixing plugin scheduler (triple-dot)...")
    total_fixed += fix_plugin_scheduler()
    print()
    
    print("="*80)
    print(f"PHASE 10 COMPLETE: Fixed {total_fixed} files")
    print("="*80)
    
    return total_fixed

if __name__ == "__main__":
    main()
