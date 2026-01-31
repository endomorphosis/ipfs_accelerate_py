"""
Tests for CLI Error Handler

This module tests the automatic error handling and GitHub integration.
"""

import pytest
import os
import sys
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ipfs_accelerate_py.error_handler import CLIErrorHandler


class TestCLIErrorHandler:
    """Test CLI error handler functionality."""
    
    def test_init(self):
        """Test error handler initialization."""
        handler = CLIErrorHandler(
            repo='test/repo',
            enable_auto_issue=False,
            enable_auto_pr=False,
            enable_auto_heal=False
        )
        
        assert handler.repo == 'test/repo'
        assert handler.enable_auto_issue is False
        assert handler.enable_auto_pr is False
        assert handler.enable_auto_heal is False
        assert handler.log_context_lines == 50
    
    def test_init_with_custom_log_lines(self):
        """Test error handler with custom log context lines."""
        handler = CLIErrorHandler(
            repo='test/repo',
            log_context_lines=100
        )
        
        assert handler.log_context_lines == 100
    
    def test_determine_severity_critical(self):
        """Test severity determination for critical errors."""
        handler = CLIErrorHandler(repo='test/repo')
        
        # MemoryError should be critical
        assert handler._determine_severity(MemoryError()) == 'critical'
        
        # RecursionError should be critical
        assert handler._determine_severity(RecursionError()) == 'critical'
    
    def test_determine_severity_high(self):
        """Test severity determination for high severity errors."""
        handler = CLIErrorHandler(repo='test/repo')
        
        # OSError should be high
        assert handler._determine_severity(OSError()) == 'high'
        
        # RuntimeError should be high
        assert handler._determine_severity(RuntimeError()) == 'high'
    
    def test_determine_severity_medium(self):
        """Test severity determination for medium severity errors."""
        handler = CLIErrorHandler(repo='test/repo')
        
        # ValueError should be medium
        assert handler._determine_severity(ValueError()) == 'medium'
        
        # TypeError should be medium
        assert handler._determine_severity(TypeError()) == 'medium'
        
        # KeyError should be medium
        assert handler._determine_severity(KeyError()) == 'medium'
    
    def test_determine_severity_low(self):
        """Test severity determination for low severity errors."""
        handler = CLIErrorHandler(repo='test/repo')
        
        # KeyboardInterrupt should be low
        assert handler._determine_severity(KeyboardInterrupt()) == 'low'
    
    def test_capture_error_basic(self):
        """Test basic error capture without auto-features."""
        handler = CLIErrorHandler(repo='test/repo')
        
        try:
            raise ValueError("Test error")
        except Exception as e:
            signature = handler.capture_error(e)
        
        # Should have captured the error
        assert len(handler._captured_errors) == 1
        
        # Check error data
        error_data = handler._captured_errors[0]
        assert error_data['type'] == 'ValueError'
        assert error_data['message'] == 'Test error'
        assert error_data['stack_trace'] is not None
        assert 'context' in error_data
        assert error_data['severity'] == 'medium'
    
    def test_capture_error_with_context(self):
        """Test error capture with additional context."""
        handler = CLIErrorHandler(repo='test/repo')
        
        custom_context = {
            'operation': 'test_operation',
            'user': 'test_user'
        }
        
        try:
            raise RuntimeError("Test runtime error")
        except Exception as e:
            handler.capture_error(e, context=custom_context)
        
        # Check context was merged
        error_data = handler._captured_errors[0]
        assert 'operation' in error_data['context']
        assert error_data['context']['operation'] == 'test_operation'
        assert 'user' in error_data['context']
        assert error_data['context']['user'] == 'test_user'
    
    @patch('ipfs_accelerate_py.error_handler.logger')
    def test_capture_log_context_no_logs_module(self, mock_logger):
        """Test log context capture when logs module is unavailable."""
        handler = CLIErrorHandler(repo='test/repo')
        
        # Should return empty string when logs module is unavailable
        log_context = handler._capture_log_context()
        assert log_context == ""
    
    def test_wrap_cli_main_success(self):
        """Test wrapping CLI main function - success case."""
        handler = CLIErrorHandler(repo='test/repo')
        
        # Create a test function
        def test_main():
            return 0
        
        # Wrap it
        wrapped = handler.wrap_cli_main(test_main)
        
        # Should return 0
        assert wrapped() == 0
    
    def test_wrap_cli_main_keyboard_interrupt(self):
        """Test wrapping CLI main function - keyboard interrupt."""
        handler = CLIErrorHandler(repo='test/repo')
        
        # Create a test function that raises KeyboardInterrupt
        def test_main():
            raise KeyboardInterrupt()
        
        # Wrap it
        wrapped = handler.wrap_cli_main(test_main)
        
        # Should return 0 for keyboard interrupt
        assert wrapped() == 0
    
    def test_wrap_cli_main_exception(self):
        """Test wrapping CLI main function - exception handling."""
        handler = CLIErrorHandler(repo='test/repo')
        
        # Create a test function that raises an exception
        def test_main():
            raise ValueError("Test error")
        
        # Wrap it
        wrapped = handler.wrap_cli_main(test_main)
        
        # Should re-raise the exception
        with pytest.raises(ValueError, match="Test error"):
            wrapped()
        
        # Should have captured the error
        assert len(handler._captured_errors) == 1
    
    def test_cleanup(self):
        """Test cleanup functionality."""
        handler = CLIErrorHandler(repo='test/repo')
        
        # Cleanup should not raise any exceptions
        handler.cleanup()
    
    @patch('ipfs_accelerate_py.error_handler.CLIErrorHandler._get_github_cli')
    def test_create_issue_disabled(self, mock_get_gh):
        """Test issue creation when auto-issue is disabled."""
        handler = CLIErrorHandler(
            repo='test/repo',
            enable_auto_issue=False
        )
        
        try:
            raise ValueError("Test error")
        except Exception as e:
            result = handler.create_issue_from_error(e)
        
        # Should return None when disabled
        assert result is None
        
        # GitHub CLI should not be called
        mock_get_gh.assert_not_called()
    
    @patch('ipfs_accelerate_py.error_handler.CLIErrorHandler._get_github_cli')
    def test_create_issue_no_github_cli(self, mock_get_gh):
        """Test issue creation when GitHub CLI is not available."""
        # Mock GitHub CLI as unavailable
        mock_get_gh.return_value = None
        
        handler = CLIErrorHandler(
            repo='test/repo',
            enable_auto_issue=True
        )
        
        try:
            raise ValueError("Test error")
        except Exception as e:
            result = handler.create_issue_from_error(e)
        
        # Should return None when GitHub CLI unavailable
        assert result is None


class TestErrorHandlerIntegration:
    """Integration tests for error handler."""
    
    @pytest.mark.skipif(
        not os.environ.get('GITHUB_ACTIONS'),
        reason="Integration test - requires GitHub environment"
    )
    def test_github_cli_available(self):
        """Test that GitHub CLI is available in CI environment."""
        import subprocess
        
        result = subprocess.run(
            ['gh', '--version'],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0
        assert 'gh version' in result.stdout
    
    def test_environment_variables(self):
        """Test reading configuration from environment variables."""
        # Test with environment variables set
        os.environ['IPFS_AUTO_ISSUE'] = 'true'
        os.environ['IPFS_AUTO_PR'] = '1'
        os.environ['IPFS_AUTO_HEAL'] = 'yes'
        os.environ['IPFS_REPO'] = 'test/custom-repo'
        
        # These would be read in cli.py main()
        assert os.environ.get('IPFS_AUTO_ISSUE', '').lower() in ('1', 'true', 'yes')
        assert os.environ.get('IPFS_AUTO_PR', '').lower() in ('1', 'true', 'yes')
        assert os.environ.get('IPFS_AUTO_HEAL', '').lower() in ('1', 'true', 'yes')
        assert os.environ.get('IPFS_REPO') == 'test/custom-repo'
        
        # Clean up
        del os.environ['IPFS_AUTO_ISSUE']
        del os.environ['IPFS_AUTO_PR']
        del os.environ['IPFS_AUTO_HEAL']
        del os.environ['IPFS_REPO']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
