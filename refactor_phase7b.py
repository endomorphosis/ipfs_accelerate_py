#!/usr/bin/env python3
"""Phase 7b: Organize remaining test/ subdirectories with content."""

import os
import subprocess
import shutil
from pathlib import Path
from collections import defaultdict

def safe_git_mv(source, target):
    """Move using git mv, with fallback."""
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            ['git', 'mv', str(source), str(target)],
            capture_output=True, text=True, check=True
        )
        return True, None
    except subprocess.CalledProcessError as e:
        return False, e.stderr

def merge_directory_contents(source, target_base, category):
    """Merge directory contents into target."""
    moves = []
    source_path = Path('test') / source
    
    if not source_path.exists():
        return moves
    
    # Find all Python files
    py_files = list(source_path.rglob('*.py'))
    
    for py_file in py_files:
        # Calculate relative path within source
        rel_path = py_file.relative_to(source_path)
        
        # Determine target
        target_path = Path(target_base) / source / rel_path
        
        moves.append((py_file, target_path))
    
    return moves

def main():
    test_dir = Path('test')
    
    # Define comprehensive organization plan
    organization_plan = {
        # API-related directories → test/tests/api/
        'api': 'test/tests/api/api',
        'api_client': 'test/tests/api/api_client',
        'api_server': 'test/tests/api/api_server',
        'apis': 'test/tests/api/apis',
        'duckdb_api': 'test/tests/api/duckdb_api',
        
        # Distributed testing → test/tests/distributed/
        'distributed_testing': 'test/tests/distributed/distributed_testing',
        
        # Web platform tests → test/tests/web/
        'fixed_web_platform': 'test/tests/web/fixed_web_platform',
        'fixed_web_tests': 'test/tests/web/fixed_web_tests',
        'web_platform': 'test/tests/web/web_platform',
        'web_platform_integration': 'test/tests/web/web_platform_integration',
        'web_platform_tests': 'test/tests/web/web_platform_tests',
        'web_audio_tests': 'test/tests/web/web_audio_tests',
        'web_interface': 'test/tests/web/web_interface',
        'web_testing_env': 'test/tests/web/web_testing_env',
        
        # Hardware-related → test/tests/hardware/
        'hardware': 'test/tests/hardware/hardware',
        'hardware_detection': 'test/tests/hardware/hardware_detection',
        'centralized_hardware_detection': 'test/tests/hardware/centralized_hardware_detection',
        'key_models_hardware_fixes': 'test/tests/hardware/key_models_hardware_fixes',
        
        # Integration tests → test/tests/integration/
        'integration': 'test/tests/integration/integration',
        'ha_cluster_example': 'test/tests/integration/ha_cluster_example',
        
        # Mobile testing → test/tests/mobile/
        'android_test_harness': 'test/tests/mobile/android_test_harness',
        'ios_test_harness': 'test/tests/mobile/ios_test_harness',
        
        # Unit tests → test/tests/unit/
        'unit': 'test/tests/unit/unit',
        
        # Common/shared code → test/common/
        'common': 'test/common/common',
        
        # Skills/capabilities → test/tools/
        'skills': 'test/tools/skills',
        'skillset': 'test/tools/skillset',
        
        # Templates → test/templates/
        'enhanced_templates': 'test/templates/enhanced_templates',
        'template_verification': 'test/templates/template_verification',
        
        # Examples → test/examples/
        'test_examples': 'test/examples/test_examples',
        'sample_tests': 'test/examples/sample_tests',
        
        # Test data/results → test/data/
        'sample_data': 'test/data/sample_data',
        'firefox_webgpu_results': 'test/data/results/firefox_webgpu',
        'webnn_webgpu_fixed_results': 'test/data/results/webnn_webgpu',
        'quant_test_results_targeted': 'test/data/results/quant_targeted',
        'validation_results': 'test/data/results/validation',
        
        # Reports → test/data/reports/
        'reports': 'test/data/reports/reports',
        'report_assets': 'test/data/reports/assets',
        'test_reports': 'test/data/reports/test_reports',
        'test_reports_comparative': 'test/data/reports/comparative',
        'test_reports_fixed': 'test/data/reports/fixed',
        
        # Visualizations → test/data/visualizations/
        'visualizations': 'test/data/visualizations/visualizations',
        
        # Mock/test environments → test/tools/
        'mock_test_env': 'test/tools/mock_test_env',
        
        # Predictive performance → test/tests/other/
        'predictive_performance': 'test/tests/other/predictive_performance',
        'simulation_validation': 'test/tests/other/simulation_validation',
        
        # High priority tests → test/tests/other/
        'high_priority_tests': 'test/tests/other/high_priority_tests',
        'remaining_model_tests': 'test/tests/other/remaining_model_tests',
        
        # Implementation files → test/implementations/
        'implementation_files': 'test/implementations/implementation_files',
        'integrated_improvements': 'test/implementations/integrated_improvements',
        
        # Test pages → test/data/
        'test_pages': 'test/data/test_pages',
        
        # Browser flags → test/data/
        'browser_flags': 'test/data/browser_flags',
        
        # Optimization → test/tools/
        'optimization_recommendation': 'test/tools/optimization_recommendation',
        
        # Phase 16 models → test/tests/models/
        'phase16_key_models': 'test/tests/models/phase16_key_models',
        
        # Transformers analysis → test/tools/
        'transformers_analysis': 'test/tools/transformers_analysis',
        
        # GitHub workflows → .github/
        '.github': '.github/test_workflows',
        
        # Visualization cache → test/data/
        '.visualization_cache': 'test/data/visualization_cache',
        
        # Src (if it's source code) → check if should go to main package
        'src': 'test/tools/src',  # or could go to main package
    }
    
    # Special cases that need to go to root level
    root_moves = {
        'ipfs_accelerate_js': 'ipfs_accelerate_js_extra',  # Merge with existing
        'ipfs_accelerate_py': None,  # Skip - already exists at root
    }
    
    print("=" * 80)
    print("PHASE 7B: ORGANIZING REMAINING TEST SUBDIRECTORIES")
    print("=" * 80)
    
    stats = defaultdict(int)
    moved_dirs = []
    skipped_dirs = []
    
    # Process organization plan
    print("\nMoving directories to proper locations...")
    print("-" * 80)
    
    for source_name, target_path in sorted(organization_plan.items()):
        source = test_dir / source_name
        
        if not source.exists():
            print(f"  [SKIP] {source} - doesn't exist")
            skipped_dirs.append(source_name)
            continue
        
        target = Path(target_path)
        
        success, error = safe_git_mv(source, target)
        if success:
            print(f"  [MOVE] {source} -> {target}")
            moved_dirs.append(source_name)
            stats['moved'] += 1
        else:
            print(f"  [ERR] {source}: {error}")
            stats['errors'] += 1
    
    # Handle special root-level moves
    print("\nHandling special cases...")
    print("-" * 80)
    
    # ipfs_accelerate_js in test/ - this appears to be test content, not the SDK
    if (test_dir / 'ipfs_accelerate_js').exists():
        source = test_dir / 'ipfs_accelerate_js'
        target = Path('test/tests/web/ipfs_accelerate_js_tests')
        success, error = safe_git_mv(source, target)
        if success:
            print(f"  [MOVE] {source} -> {target}")
            stats['moved'] += 1
        else:
            print(f"  [ERR] {source}: {error}")
    
    # ipfs_accelerate_py in test/ - check what it is
    if (test_dir / 'ipfs_accelerate_py').exists():
        source = test_dir / 'ipfs_accelerate_py'
        # Check if it's actually test content
        py_count = len(list(source.rglob('*.py')))
        print(f"  [INFO] test/ipfs_accelerate_py has {py_count} Python files")
        target = Path('test/tests/other/ipfs_accelerate_py_tests')
        success, error = safe_git_mv(source, target)
        if success:
            print(f"  [MOVE] {source} -> {target}")
            stats['moved'] += 1
    
    print("\n" + "=" * 80)
    print("SUMMARY:")
    print(f"  Successfully moved:  {stats['moved']} directories")
    print(f"  Errors:              {stats['errors']} directories")
    print(f"  Skipped (not found): {len(skipped_dirs)} directories")
    print("=" * 80)
    
    print(f"\nMoved {len(moved_dirs)} directories:")
    for d in sorted(moved_dirs)[:20]:
        print(f"  - {d}")
    if len(moved_dirs) > 20:
        print(f"  ... and {len(moved_dirs) - 20} more")
    
    print("\nPhase 7b complete!")

if __name__ == '__main__':
    main()
