#!/usr/bin/env python3
"""
Fix remaining relative import issues - Phase 3
Focus on single-level relative imports
"""
import os
import re
from pathlib import Path

def fix_single_level_ci_imports():
    """Fix single-level ci imports like 'from .ci.XXX import'."""
    base_dir = Path('test/tests/distributed/distributed_testing')
    
    if not base_dir.exists():
        print(f"Directory not found: {base_dir}")
        return 0
    
    count = 0
    
    # CI submodules
    ci_submodules = [
        'api_interface', 'github_client', 'gitlab_client', 'register_providers',
        'result_reporter', 'url_validator', 'artifact_handler', 'artifact_discovery',
        'artifact_metadata', 'artifact_retriever', 'azure_client', 'bitbucket_client',
        'circleci_client', 'jenkins_client', 'teamcity_client', 'travis_client'
    ]
    
    # Fix in examples/ and tests/ subdirectories
    for subdir in ['examples', 'tests']:
        search_dir = base_dir / subdir
        if not search_dir.exists():
            continue
            
        for py_file in search_dir.glob('*.py'):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                original = content
                
                # Fix: from .ci.XXX import (single level)
                for submodule in ci_submodules:
                    content = re.sub(
                        rf'from \.ci\.{submodule} import',
                        rf'from test.tests.distributed.distributed_testing.ci.{submodule} import',
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

def fix_all_relative_patterns():
    """Fix all remaining relative import patterns in distributed testing."""
    base_dir = Path('test/tests/distributed/distributed_testing')
    
    if not base_dir.exists():
        return 0
    
    count = 0
    
    # Map of all known modules in distributed_testing
    known_modules = {
        # Direct children
        'ci', 'coordinator', 'worker', 'circuit_breaker', 'task_scheduler',
        'plugin_architecture', 'external_systems', 'hardware_workload_management',
        'browser_recovery_strategies', 'integration_mode', 'dynamic_resource_manager',
        'performance_trend_analyzer', 'hardware_aware_scheduler', 'create_task',
        'plugins', 'plugin_base', 'examples', 'tests', 'integration_tests',
        
        # Submodules
        'hardware_capability_detector', 'load_balancer_integration',
        'load_balancer_resource_pool_bridge', 'enhanced_hardware_capability',
        'hardware_aware_visualization', 'model_sharding',
    }
    
    for py_file in base_dir.rglob('*.py'):
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original = content
            
            # Fix single-level relative imports (from .module import)
            for module in known_modules:
                content = re.sub(
                    rf'from \.{module} import',
                    rf'from test.tests.distributed.distributed_testing.{module} import',
                    content
                )
            
            # Fix nested single-level relative imports (from .subdir.module import)
            # This handles patterns like from .examples.XXX import
            content = re.sub(
                r'from \.(\w+)\.(\w+) import',
                lambda m: f'from test.tests.distributed.distributed_testing.{m.group(1)}.{m.group(2)} import' 
                    if m.group(1) in known_modules else m.group(0),
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
    """Main function to fix remaining relative imports."""
    print("=" * 80)
    print("Fixing remaining relative import issues - Phase 3")
    print("=" * 80)
    
    print("\n1. Fixing single-level ci imports...")
    count1 = fix_single_level_ci_imports()
    print(f"   Fixed {count1} files")
    
    print("\n2. Fixing all remaining relative patterns...")
    count2 = fix_all_relative_patterns()
    print(f"   Fixed {count2} files")
    
    total = count1 + count2
    print("\n" + "=" * 80)
    print(f"Total files fixed: {total}")
    print("=" * 80)
    
    return total

if __name__ == '__main__':
    import sys
    sys.exit(0 if main() >= 0 else 1)
