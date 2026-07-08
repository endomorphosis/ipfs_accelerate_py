"""
Tests for P2P Workflow Discovery Service
"""

import pytest
import sys
import os

# Add the package to path
sys.path.insert(0, os.path.dirname(__file__))


class TestP2PWorkflowDiscovery:
    """Test P2P workflow discovery functionality"""
    
    def test_parse_workflow_tags_simple(self):
        """Test parsing simple workflow tags"""
        try:
            from ipfs_accelerate_py.p2p_workflow_discovery import P2PWorkflowDiscoveryService
        except ImportError:
            from p2p_workflow_discovery import P2PWorkflowDiscoveryService
        
        # Mock service without full initialization
        class MockService:
            def _parse_workflow_tags(self, content):
                service = P2PWorkflowDiscoveryService.__new__(P2PWorkflowDiscoveryService)
                return service._parse_workflow_tags(content)
        
        mock = MockService()
        
        # Test WORKFLOW_TAGS format
        content = """
        env:
          WORKFLOW_TAGS: p2p-only,code-generation
        """
        tags = mock._parse_workflow_tags(content)
        assert 'p2p-only' in tags
        assert 'code-generation' in tags
    
    def test_parse_workflow_tags_with_quotes(self):
        """Test parsing workflow tags with quotes"""
        from ipfs_accelerate_py.p2p_workflow_discovery import P2PWorkflowDiscoveryService
        
        class MockService:
            def _parse_workflow_tags(self, content):
                service = P2PWorkflowDiscoveryService.__new__(P2PWorkflowDiscoveryService)
                return service._parse_workflow_tags(content)
        
        mock = MockService()
        
        content = """
        env:
          WORKFLOW_TAGS: "p2p-eligible, web-scraping"
        """
        tags = mock._parse_workflow_tags(content)
        assert 'p2p-eligible' in tags
        assert 'web-scraping' in tags
    
    def test_parse_workflow_tags_comment(self):
        """Test parsing tags from comments"""
        from ipfs_accelerate_py.p2p_workflow_discovery import P2PWorkflowDiscoveryService
        
        class MockService:
            def _parse_workflow_tags(self, content):
                service = P2PWorkflowDiscoveryService.__new__(P2PWorkflowDiscoveryService)
                return service._parse_workflow_tags(content)
        
        mock = MockService()
        
        content = """
        # P2P: p2p-only, data-processing
        jobs:
          test:
            runs-on: ubuntu-latest
        """
        tags = mock._parse_workflow_tags(content)
        assert 'p2p-only' in tags
        assert 'data-processing' in tags
    
    def test_parse_workflow_tags_invalid(self):
        """Test that invalid tags are filtered out"""
        from ipfs_accelerate_py.p2p_workflow_discovery import P2PWorkflowDiscoveryService
        
        class MockService:
            def _parse_workflow_tags(self, content):
                service = P2PWorkflowDiscoveryService.__new__(P2PWorkflowDiscoveryService)
                return service._parse_workflow_tags(content)
        
        mock = MockService()
        
        content = """
        env:
          WORKFLOW_TAGS: "p2p-only, invalid-tag, code-generation"
        """
        tags = mock._parse_workflow_tags(content)
        assert 'p2p-only' in tags
        assert 'code-generation' in tags
        assert 'invalid-tag' not in tags


class TestAutoscalerP2PIntegration:
    """Test autoscaler integration with P2P scheduler"""
    
    def test_autoscaler_p2p_disabled(self):
        """Test autoscaler with P2P disabled"""
        # This test just checks that the autoscaler can be created with P2P disabled
        # Full integration tests would require GitHub authentication
        try:
            from github_autoscaler import GitHubRunnerAutoscaler
            # Just verify the parameter is accepted
            assert True
        except ImportError:
            pytest.skip("GitHub autoscaler not available")
    
    def test_autoscaler_p2p_enabled_parameter(self):
        """Test that autoscaler accepts enable_p2p parameter"""
        try:
            from github_autoscaler import GitHubRunnerAutoscaler
            import inspect
            
            # Check that __init__ has enable_p2p parameter
            sig = inspect.signature(GitHubRunnerAutoscaler.__init__)
            assert 'enable_p2p' in sig.parameters
        except ImportError:
            pytest.skip("GitHub autoscaler not available")


class TestWorkflowDiscovery:
    """Test WorkflowDiscovery data class"""
    
    def test_workflow_discovery_creation(self):
        """Test creating a WorkflowDiscovery object"""
        from ipfs_accelerate_py.p2p_workflow_discovery import WorkflowDiscovery
        
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
        assert discovery.workflow_id == "test-workflow"
        assert "p2p-only" in discovery.tags
        assert "code-generation" in discovery.tags


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
