#!/usr/bin/env python3
"""
Test script for GitHub Actions Runner Autoscaler architecture filtering.

This script validates the architecture detection and filtering logic
without requiring GitHub CLI authentication.
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_architecture_detection():
    """Test system architecture detection."""
    print("=" * 80)
    print("Testing Architecture Detection")
    print("=" * 80)
    
    from ipfs_accelerate_py.github_cli import RunnerManager
    
    # Create runner manager (will fail auth check, but that's ok for this test)
    try:
        rm = RunnerManager()
    except RuntimeError as e:
        # Expected if not authenticated, create without gh CLI
        import platform
        import subprocess
        
        class MockGH:
            pass
        
        rm = RunnerManager.__new__(RunnerManager)
        rm.gh = MockGH()
        rm._system_arch = rm._detect_system_architecture()
        rm._runner_labels = rm._generate_runner_labels()
    
    print(f"\n✓ System Architecture: {rm.get_system_architecture()}")
    print(f"✓ Runner Labels: {rm.get_runner_labels()}")
    print(f"✓ System Cores: {rm.get_system_cores()}")
    
    # Verify architecture is one of the expected values
    assert rm.get_system_architecture() in ['x64', 'arm64', 'aarch64'], \
        f"Unexpected architecture: {rm.get_system_architecture()}"
    
    # Verify labels include the architecture
    assert rm.get_system_architecture() in rm.get_runner_labels(), \
        "Architecture not in runner labels"
    
    # Verify docker label is present
    assert 'docker' in rm.get_runner_labels(), \
        "Docker label missing from runner labels"
    
    print("\n✓ All architecture detection tests passed!")
    return True


def test_workflow_filtering():
    """Test workflow filtering by architecture."""
    print("\n" + "=" * 80)
    print("Testing Workflow Architecture Filtering")
    print("=" * 80)
    
    from ipfs_accelerate_py.github_cli import WorkflowQueue
    
    # Create workflow queue (will fail auth check, but that's ok for this test)
    try:
        wq = WorkflowQueue()
    except RuntimeError:
        # Expected if not authenticated
        class MockGH:
            pass
        
        wq = WorkflowQueue.__new__(WorkflowQueue)
        wq.gh = MockGH()
    
    # Test cases for x64 architecture
    print("\nTesting x64 architecture filtering:")
    
    test_cases_x64 = [
        # (workflow_name, expected_result_on_x64)
        ('amd64-ci.yml', True),
        ('arm64-ci.yml', False),
        ('test-amd64-containerized', True),
        ('test-arm64-containerized', False),
        ('test-x64-build', True),
        ('test-aarch64-build', False),
        ('generic-test.yml', True),  # No arch specified, assume compatible
        ('python-tests.yml', True),  # No arch specified, assume compatible
    ]
    
    for workflow_name, expected in test_cases_x64:
        workflow = {'workflowName': workflow_name}
        result = wq._check_workflow_runner_compatibility(workflow, 'test/repo', 'x64')
        status = "✓" if result == expected else "✗"
        print(f"  {status} {workflow_name}: {result} (expected {expected})")
        assert result == expected, f"Failed for {workflow_name} on x64"
    
    # Test cases for arm64 architecture
    print("\nTesting arm64 architecture filtering:")
    
    test_cases_arm64 = [
        # (workflow_name, expected_result_on_arm64)
        ('amd64-ci.yml', False),
        ('arm64-ci.yml', True),
        ('test-amd64-containerized', False),
        ('test-arm64-containerized', True),
        ('test-x64-build', False),
        ('test-aarch64-build', True),
        ('generic-test.yml', True),  # No arch specified, assume compatible
        ('python-tests.yml', True),  # No arch specified, assume compatible
    ]
    
    for workflow_name, expected in test_cases_arm64:
        workflow = {'workflowName': workflow_name}
        result = wq._check_workflow_runner_compatibility(workflow, 'test/repo', 'arm64')
        status = "✓" if result == expected else "✗"
        print(f"  {status} {workflow_name}: {result} (expected {expected})")
        assert result == expected, f"Failed for {workflow_name} on arm64"
    
    print("\n✓ All workflow filtering tests passed!")
    return True


def test_integration():
    """Test the overall integration."""
    print("\n" + "=" * 80)
    print("Testing Integration")
    print("=" * 80)
    
    import platform
    
    arch = platform.machine().lower()
    print(f"\nCurrent system: {arch}")
    
    # Map to expected architecture
    arch_map = {
        'x86_64': 'x64',
        'amd64': 'x64',
        'aarch64': 'arm64',
        'arm64': 'arm64',
    }
    
    expected_arch = arch_map.get(arch, arch)
    print(f"Expected architecture: {expected_arch}")
    
    # Test that incompatible workflows are filtered
    from ipfs_accelerate_py.github_cli import WorkflowQueue
    
    try:
        wq = WorkflowQueue()
    except RuntimeError:
        class MockGH:
            pass
        wq = WorkflowQueue.__new__(WorkflowQueue)
        wq.gh = MockGH()
    
    if expected_arch == 'x64':
        # On x64, arm64 workflows should be filtered out
        workflow = {'workflowName': 'arm64-ci.yml'}
        assert not wq._check_workflow_runner_compatibility(workflow, 'test/repo', 'x64'), \
            "x64 system should filter out arm64 workflows"
        print("✓ x64 system correctly filters ARM64 workflows")
        
    elif expected_arch == 'arm64':
        # On arm64, x64 workflows should be filtered out
        workflow = {'workflowName': 'amd64-ci.yml'}
        assert not wq._check_workflow_runner_compatibility(workflow, 'test/repo', 'arm64'), \
            "arm64 system should filter out x64 workflows"
        print("✓ ARM64 system correctly filters x64 workflows")
    
    print("\n✓ All integration tests passed!")
    return True


def main():
    """Run all tests."""
    print("\nGitHub Actions Runner Autoscaler - Architecture Filtering Tests")
    print("=" * 80)
    
    all_passed = True
    
    try:
        test_architecture_detection()
    except Exception as e:
        print(f"\n✗ Architecture detection tests failed: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    try:
        test_workflow_filtering()
    except Exception as e:
        print(f"\n✗ Workflow filtering tests failed: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    try:
        test_integration()
    except Exception as e:
        print(f"\n✗ Integration tests failed: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    print("\n" + "=" * 80)
    if all_passed:
        print("✓ ALL TESTS PASSED!")
        print("=" * 80)
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        print("=" * 80)
        return 1


if __name__ == "__main__":
    sys.exit(main())
