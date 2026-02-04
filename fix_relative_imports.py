#!/usr/bin/env python3
"""
Fix remaining relative import issues after refactoring.
"""
import os
import re
from pathlib import Path

def fix_anyio_queue_imports():
    """Fix anyio_queue imports in skillset files."""
    test_dir = Path('test/tests/other/ipfs_accelerate_py_tests/worker/skillset')
    
    if not test_dir.exists():
        print(f"Directory not found: {test_dir}")
        return 0
    
    count = 0
    for py_file in test_dir.glob('*.py'):
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original = content
            
            # Fix: from ..anyio_queue import AnyioQueue
            # To: from ipfs_accelerate_py.worker.anyio_queue import AnyioQueue
            content = re.sub(
                r'from \.\.anyio_queue import',
                r'from ipfs_accelerate_py.worker.anyio_queue import',
                content
            )
            
            if content != original:
                with open(py_file, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"Fixed: {py_file}")
                count += 1
                
        except Exception as e:
            print(f"Error processing {py_file}: {e}")
    
    return count

def fix_distributed_testing_imports():
    """Fix distributed testing relative imports."""
    base_dir = Path('test/tests/distributed/distributed_testing')
    
    if not base_dir.exists():
        print(f"Directory not found: {base_dir}")
        return 0
    
    count = 0
    
    # Mapping of relative imports to absolute imports
    import_mappings = {
        # CI module imports
        r'from \.ci import': 'from test.tests.distributed.distributed_testing.ci import',
        r'from \.\.ci import': 'from test.tests.distributed.distributed_testing.ci import',
        r'from \.\.\.ci import': 'from test.tests.distributed.distributed_testing.ci import',
        
        # Coordinator imports
        r'from \.coordinator import': 'from test.tests.distributed.distributed_testing.coordinator import',
        r'from \.\.coordinator import': 'from test.tests.distributed.distributed_testing.coordinator import',
        
        # Worker imports
        r'from \.worker import': 'from test.tests.distributed.distributed_testing.worker import',
        r'from \.\.worker import': 'from test.tests.distributed.distributed_testing.worker import',
        
        # Circuit breaker imports
        r'from \.circuit_breaker import': 'from test.tests.distributed.distributed_testing.circuit_breaker import',
        r'from \.\.circuit_breaker import': 'from test.tests.distributed.distributed_testing.circuit_breaker import',
        
        # Task scheduler imports
        r'from \.task_scheduler import': 'from test.tests.distributed.distributed_testing.task_scheduler import',
        r'from \.\.task_scheduler import': 'from test.tests.distributed.distributed_testing.task_scheduler import',
        
        # Plugin architecture imports
        r'from \.plugin_architecture import': 'from test.tests.distributed.distributed_testing.plugin_architecture import',
        r'from \.\.plugin_architecture import': 'from test.tests.distributed.distributed_testing.plugin_architecture import',
        
        # External systems imports
        r'from \.external_systems import': 'from test.tests.distributed.distributed_testing.external_systems import',
        r'from \.\.external_systems import': 'from test.tests.distributed.distributed_testing.external_systems import',
        
        # Hardware workload management imports
        r'from \.hardware_workload_management import': 'from test.tests.distributed.distributed_testing.hardware_workload_management import',
        r'from \.\.hardware_workload_management import': 'from test.tests.distributed.distributed_testing.hardware_workload_management import',
        
        # Browser recovery strategies imports
        r'from \.browser_recovery_strategies import': 'from test.tests.distributed.distributed_testing.browser_recovery_strategies import',
        r'from \.\.browser_recovery_strategies import': 'from test.tests.distributed.distributed_testing.browser_recovery_strategies import',
        
        # Integration mode imports
        r'from \.integration_mode import': 'from test.tests.distributed.distributed_testing.integration_mode import',
        r'from \.\.integration_mode import': 'from test.tests.distributed.distributed_testing.integration_mode import',
        
        # Dynamic resource manager imports
        r'from \.dynamic_resource_manager import': 'from test.tests.distributed.distributed_testing.dynamic_resource_manager import',
        r'from \.\.dynamic_resource_manager import': 'from test.tests.distributed.distributed_testing.dynamic_resource_manager import',
        
        # Performance trend analyzer imports
        r'from \.performance_trend_analyzer import': 'from test.tests.distributed.distributed_testing.performance_trend_analyzer import',
        r'from \.\.performance_trend_analyzer import': 'from test.tests.distributed.distributed_testing.performance_trend_analyzer import',
        
        # Hardware aware scheduler imports
        r'from \.hardware_aware_scheduler import': 'from test.tests.distributed.distributed_testing.hardware_aware_scheduler import',
        r'from \.\.hardware_aware_scheduler import': 'from test.tests.distributed.distributed_testing.hardware_aware_scheduler import',
        
        # Create task imports
        r'from \.create_task import': 'from test.tests.distributed.distributed_testing.create_task import',
        r'from \.\.create_task import': 'from test.tests.distributed.distributed_testing.create_task import',
        
        # Plugins imports
        r'from \.plugins import': 'from test.tests.distributed.distributed_testing.plugins import',
        r'from \.\.plugins import': 'from test.tests.distributed.distributed_testing.plugins import',
    }
    
    for py_file in base_dir.rglob('*.py'):
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original = content
            
            for pattern, replacement in import_mappings.items():
                content = re.sub(pattern, replacement, content)
            
            if content != original:
                with open(py_file, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"Fixed: {py_file}")
                count += 1
                
        except Exception as e:
            print(f"Error processing {py_file}: {e}")
    
    return count

def fix_other_relative_imports():
    """Fix other relative import issues."""
    count = 0
    
    # Fix ipfs_accelerate_py_tests imports
    py_file = Path('test/tests/other/ipfs_accelerate_py_tests/__init__.py')
    if py_file.exists():
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original = content
            
            # Fix relative imports to use absolute imports
            content = re.sub(
                r'from \.container_backends import',
                r'from ipfs_accelerate_py.container_backends import',
                content
            )
            content = re.sub(
                r'from \.install_depends import',
                r'from ipfs_accelerate_py.install_depends import',
                content
            )
            content = re.sub(
                r'from \.config import',
                r'from ipfs_accelerate_py.config import',
                content
            )
            
            if content != original:
                with open(py_file, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"Fixed: {py_file}")
                count += 1
                
        except Exception as e:
            print(f"Error processing {py_file}: {e}")
    
    # Fix webgpu_quantization imports
    web_platform_dir = Path('test/tests/web/fixed_web_platform')
    if web_platform_dir.exists():
        for py_file in web_platform_dir.glob('*.py'):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                original = content
                
                # Fix relative imports for webgpu_quantization
                content = re.sub(
                    r'from \.webgpu_quantization import',
                    r'from test.tests.web.fixed_web_platform.webgpu_quantization import',
                    content
                )
                
                if content != original:
                    with open(py_file, 'w', encoding='utf-8') as f:
                        f.write(content)
                    print(f"Fixed: {py_file}")
                    count += 1
                    
            except Exception as e:
                print(f"Error processing {py_file}: {e}")
    
    return count

def main():
    """Main function to fix all relative imports."""
    print("=" * 80)
    print("Fixing relative import issues")
    print("=" * 80)
    
    print("\n1. Fixing anyio_queue imports...")
    count1 = fix_anyio_queue_imports()
    print(f"   Fixed {count1} files")
    
    print("\n2. Fixing distributed testing imports...")
    count2 = fix_distributed_testing_imports()
    print(f"   Fixed {count2} files")
    
    print("\n3. Fixing other relative imports...")
    count3 = fix_other_relative_imports()
    print(f"   Fixed {count3} files")
    
    total = count1 + count2 + count3
    print("\n" + "=" * 80)
    print(f"Total files fixed: {total}")
    print("=" * 80)
    
    return total

if __name__ == '__main__':
    import sys
    sys.exit(0 if main() >= 0 else 1)
