#!/usr/bin/env python3
"""
Fix remaining relative import issues - Phase 2
Focus on distributed testing submodules
"""
import os
import re
from pathlib import Path

def fix_ci_submodule_imports():
    """Fix imports for ci submodules in distributed testing."""
    base_dir = Path('test/tests/distributed/distributed_testing')
    
    if not base_dir.exists():
        print(f"Directory not found: {base_dir}")
        return 0
    
    count = 0
    
    # CI submodule mappings
    ci_submodules = [
        'api_interface', 'github_client', 'gitlab_client', 'register_providers',
        'result_reporter', 'url_validator', 'artifact_handler', 'artifact_discovery',
        'artifact_metadata', 'artifact_retriever', 'azure_client', 'bitbucket_client',
        'circleci_client', 'jenkins_client', 'teamcity_client', 'travis_client'
    ]
    
    for py_file in base_dir.rglob('*.py'):
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original = content
            
            # Fix: from ..ci.XXX import or from ...ci.XXX import
            for submodule in ci_submodules:
                # Two levels up
                content = re.sub(
                    rf'from \.\.ci\.{submodule} import',
                    rf'from test.tests.distributed.distributed_testing.ci.{submodule} import',
                    content
                )
                # Three levels up
                content = re.sub(
                    rf'from \.\.\.ci\.{submodule} import',
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

def fix_examples_subdir_imports():
    """Fix imports in examples subdirectory."""
    base_dir = Path('test/tests/distributed/distributed_testing/examples')
    
    if not base_dir.exists():
        return 0
    
    count = 0
    
    for py_file in base_dir.glob('*.py'):
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original = content
            
            # Fix: from .examples.XXX import (examples/examples pattern)
            content = re.sub(
                r'from \.examples\.(\w+) import',
                r'from test.tests.distributed.distributed_testing.examples.\1 import',
                content
            )
            
            # Fix other examples submodule imports
            modules = [
                'enhanced_hardware_capability', 'hardware_aware_visualization',
                'hardware_capability_detector', 'load_balancer_integration',
                'load_balancer_resource_pool_bridge'
            ]
            
            for module in modules:
                content = re.sub(
                    rf'from \.{module} import',
                    rf'from test.tests.distributed.distributed_testing.examples.{module} import',
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

def fix_external_systems_imports():
    """Fix external_systems submodule imports."""
    base_dir = Path('test/tests/distributed/distributed_testing')
    
    if not base_dir.exists():
        return 0
    
    count = 0
    
    for py_file in base_dir.rglob('*.py'):
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original = content
            
            # Fix: from .external_systems.XXX import
            # Fix: from ..external_systems.XXX import
            content = re.sub(
                r'from \.external_systems\.(\w+) import',
                r'from test.tests.distributed.distributed_testing.external_systems.\1 import',
                content
            )
            content = re.sub(
                r'from \.\.external_systems\.(\w+) import',
                r'from test.tests.distributed.distributed_testing.external_systems.\1 import',
                content
            )
            
            # Fix nested external_systems/external_systems pattern
            content = re.sub(
                r'from \.external_systems\.external_systems\.(\w+) import',
                r'from test.tests.distributed.distributed_testing.external_systems.\1 import',
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

def fix_plugins_imports():
    """Fix plugins submodule imports."""
    base_dir = Path('test/tests/distributed/distributed_testing')
    
    if not base_dir.exists():
        return 0
    
    count = 0
    
    for py_file in base_dir.rglob('*.py'):
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original = content
            
            # Fix: from .plugin_base import
            content = re.sub(
                r'from \.plugin_base import',
                r'from test.tests.distributed.distributed_testing.plugin_base import',
                content
            )
            content = re.sub(
                r'from \.\.plugin_base import',
                r'from test.tests.distributed.distributed_testing.plugin_base import',
                content
            )
            
            # Fix: from .plugins.XXX.XXX import (nested plugins pattern)
            content = re.sub(
                r'from \.plugins\.(\w+)\.(\w+) import',
                r'from test.tests.distributed.distributed_testing.plugins.\1.\2 import',
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

def fix_integration_tests_imports():
    """Fix integration_tests submodule imports."""
    base_dir = Path('test/tests/distributed/distributed_testing/integration_tests')
    
    if not base_dir.exists():
        return 0
    
    count = 0
    
    for py_file in base_dir.glob('*.py'):
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original = content
            
            # Fix: from .model_sharding import
            content = re.sub(
                r'from \.model_sharding import',
                r'from test.tests.distributed.distributed_testing.integration_tests.model_sharding import',
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
    print("Fixing remaining relative import issues - Phase 2")
    print("=" * 80)
    
    print("\n1. Fixing ci submodule imports...")
    count1 = fix_ci_submodule_imports()
    print(f"   Fixed {count1} files")
    
    print("\n2. Fixing examples subdirectory imports...")
    count2 = fix_examples_subdir_imports()
    print(f"   Fixed {count2} files")
    
    print("\n3. Fixing external_systems imports...")
    count3 = fix_external_systems_imports()
    print(f"   Fixed {count3} files")
    
    print("\n4. Fixing plugins imports...")
    count4 = fix_plugins_imports()
    print(f"   Fixed {count4} files")
    
    print("\n5. Fixing integration_tests imports...")
    count5 = fix_integration_tests_imports()
    print(f"   Fixed {count5} files")
    
    total = count1 + count2 + count3 + count4 + count5
    print("\n" + "=" * 80)
    print(f"Total files fixed: {total}")
    print("=" * 80)
    
    return total

if __name__ == '__main__':
    import sys
    sys.exit(0 if main() >= 0 else 1)
