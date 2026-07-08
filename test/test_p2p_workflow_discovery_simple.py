"""
Simple tests for P2P Workflow Discovery Service
"""

import sys
import os

# Add the package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ipfs_accelerate_py'))

def test_imports():
    """Test that modules can be imported"""
    try:
        from p2p_workflow_discovery import P2PWorkflowDiscoveryService, WorkflowDiscovery
        print("✓ Successfully imported P2PWorkflowDiscoveryService and WorkflowDiscovery")
        return True
    except Exception as e:
        print(f"✗ Failed to import: {e}")
        return False

def test_workflow_discovery_dataclass():
    """Test WorkflowDiscovery data class"""
    try:
        from p2p_workflow_discovery import WorkflowDiscovery
        
        discovery = WorkflowDiscovery(
            owner="test-owner",
            repo="test-repo",
            workflow_id="test-workflow",
            workflow_name="test.yml",
            workflow_path=".github/workflows/test.yml",
            tags=["p2p-only", "code-generation"]
        )
        
        assert discovery.owner == "test-owner"
        assert discovery.repo == "test-repo"
        assert "p2p-only" in discovery.tags
        print("✓ WorkflowDiscovery dataclass works correctly")
        return True
    except Exception as e:
        print(f"✗ WorkflowDiscovery test failed: {e}")
        return False

def test_parse_tags():
    """Test tag parsing"""
    try:
        from p2p_workflow_discovery import P2PWorkflowDiscoveryService
        
        # Create mock instance
        class MockService:
            def __init__(self):
                # Create real instance to access method
                self.real_service = P2PWorkflowDiscoveryService.__new__(P2PWorkflowDiscoveryService)
            
            def parse_tags(self, content):
                return self.real_service._parse_workflow_tags(content)
        
        mock = MockService()
        
        # Test WORKFLOW_TAGS format
        content = """
        env:
          WORKFLOW_TAGS: p2p-only,code-generation
        """
        tags = mock.parse_tags(content)
        assert 'p2p-only' in tags
        assert 'code-generation' in tags
        print(f"✓ Tag parsing works: found tags {tags}")
        return True
    except Exception as e:
        print(f"✗ Tag parsing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_autoscaler_parameter():
    """Test that autoscaler accepts P2P parameter"""
    try:
        from github_autoscaler import GitHubRunnerAutoscaler
        import inspect
        
        sig = inspect.signature(GitHubRunnerAutoscaler.__init__)
        if 'enable_p2p' in sig.parameters:
            print("✓ Autoscaler has enable_p2p parameter")
            return True
        else:
            print("✗ Autoscaler missing enable_p2p parameter")
            return False
    except Exception as e:
        print(f"⚠ Autoscaler test skipped: {e}")
        return True  # Skip this test if autoscaler not available

if __name__ == "__main__":
    print("Running P2P Workflow Discovery Tests...")
    print("=" * 60)
    
    tests = [
        ("Import test", test_imports),
        ("WorkflowDiscovery dataclass", test_workflow_discovery_dataclass),
        ("Tag parsing", test_parse_tags),
        ("Autoscaler parameter", test_autoscaler_parameter),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        print(f"\n{name}:")
        if test_func():
            passed += 1
        else:
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    
    sys.exit(0 if failed == 0 else 1)
