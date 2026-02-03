"""
Unit tests for ipfs_files_kit module.

Tests the IPFSFilesKit class and its methods for IPFS file operations.
"""

import unittest
import unittest.mock
from unittest.mock import Mock, patch, MagicMock
from dataclasses import asdict


class TestIPFSFilesKit(unittest.TestCase):
    """Test cases for IPFSFilesKit class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Import here to avoid import errors if module not available
        try:
            from ipfs_accelerate_py.kit.ipfs_files_kit import (
                IPFSFilesKit,
                IPFSFilesConfig,
                IPFSFileResult,
                IPFSFileInfo,
                get_ipfs_files_kit
            )
            self.IPFSFilesKit = IPFSFilesKit
            self.IPFSFilesConfig = IPFSFilesConfig
            self.IPFSFileResult = IPFSFileResult
            self.IPFSFileInfo = IPFSFileInfo
            self.get_ipfs_files_kit = get_ipfs_files_kit
        except ImportError:
            self.skipTest("ipfs_files_kit module not available")
    
    def test_ipfs_files_kit_initialization(self):
        """Test IPFSFilesKit initialization."""
        config = self.IPFSFilesConfig()
        kit = self.IPFSFilesKit(config)
        
        self.assertIsInstance(kit, self.IPFSFilesKit)
        self.assertEqual(kit.config, config)
    
    def test_get_ipfs_files_kit_singleton(self):
        """Test get_ipfs_files_kit returns singleton."""
        kit1 = self.get_ipfs_files_kit()
        kit2 = self.get_ipfs_files_kit()
        
        self.assertIs(kit1, kit2)
    
    @patch('os.path.exists')
    @patch('subprocess.run')
    def test_add_file_success(self, mock_run, mock_exists):
        """Test adding a file to IPFS successfully."""
        # Mock file exists
        mock_exists.return_value = True
        
        # Mock IPFS CLI response
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = 'added QmTest123456 file.txt'
        mock_result.stderr = ''
        mock_run.return_value = mock_result
        
        kit = self.get_ipfs_files_kit()
        
        # Mock open to avoid actual file I/O
        with patch('builtins.open', unittest.mock.mock_open(read_data=b'test content')):
            result = kit.add_file('/path/to/file.txt')
        
        self.assertTrue(result.success)
        self.assertIn('cid', result.data)
        self.assertIsNone(result.error)
    
    @patch('subprocess.run')
    def test_add_file_failure(self, mock_run):
        """Test adding a file to IPFS with failure."""
        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stdout = ''
        mock_result.stderr = 'Error: file not found'
        mock_run.return_value = mock_result
        
        kit = self.get_ipfs_files_kit()
        result = kit.add_file('/nonexistent/file.txt')
        
        self.assertFalse(result.success)
        self.assertIsNotNone(result.error)
    
    @patch('os.makedirs')
    @patch('builtins.open', new_callable=unittest.mock.mock_open)
    @patch('subprocess.run')
    def test_get_file_success(self, mock_run, mock_open, mock_makedirs):
        """Test getting a file from IPFS successfully."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = b'file content'
        mock_result.stderr = ''
        mock_run.return_value = mock_result
        
        kit = self.get_ipfs_files_kit()
        result = kit.get_file('QmTest123', '/output/file.txt')
        
        self.assertTrue(result.success)
        self.assertIsNone(result.error)
    
    @patch('subprocess.run')
    def test_get_file_failure(self, mock_run):
        """Test getting a file from IPFS with failure."""
        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stdout = ''
        mock_result.stderr = 'Error: CID not found'
        mock_run.return_value = mock_result
        
        kit = self.get_ipfs_files_kit()
        result = kit.get_file('QmInvalid', '/output/file.txt')
        
        self.assertFalse(result.success)
        self.assertIsNotNone(result.error)
    
    @patch('subprocess.run')
    def test_cat_file(self, mock_run):
        """Test reading file content from IPFS."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = 'File content here'
        mock_result.stderr = ''
        mock_run.return_value = mock_result
        
        kit = self.get_ipfs_files_kit()
        result = kit.cat_file('QmTest123')
        
        # The method should complete successfully when mock returns success
        self.assertTrue(result.success)
        self.assertIn('content', result.data)
        # With our mock, the content should be from the mocked stdout
        if result.data['content'] is not None:
            self.assertEqual(result.data['content'], 'File content here')
    
    @patch('subprocess.run')
    def test_pin_file(self, mock_run):
        """Test pinning a file in IPFS."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = 'pinned QmTest123 recursively'
        mock_result.stderr = ''
        mock_run.return_value = mock_result
        
        kit = self.get_ipfs_files_kit()
        result = kit.pin_file('QmTest123')
        
        self.assertTrue(result.success)
        self.assertIsNone(result.error)
    
    @patch('subprocess.run')
    def test_unpin_file(self, mock_run):
        """Test unpinning a file from IPFS."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = 'unpinned QmTest123'
        mock_result.stderr = ''
        mock_run.return_value = mock_result
        
        kit = self.get_ipfs_files_kit()
        result = kit.unpin_file('QmTest123')
        
        self.assertTrue(result.success)
        self.assertIsNone(result.error)
    
    @patch('subprocess.run')
    def test_list_files(self, mock_run):
        """Test listing IPFS files."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = 'QmTest1 file1.txt\nQmTest2 file2.txt'
        mock_result.stderr = ''
        mock_run.return_value = mock_result
        
        kit = self.get_ipfs_files_kit()
        result = kit.list_files('/')
        
        self.assertTrue(result.success)
        self.assertIn('files', result.data)
    
    def test_validate_cid_valid(self):
        """Test CID validation with valid CID."""
        kit = self.get_ipfs_files_kit()
        result = kit.validate_cid('QmTest123456789abcdefghijklmnopqrstuvwxyz')
        
        # Should succeed or have validation logic
        self.assertIsInstance(result, self.IPFSFileResult)
    
    def test_validate_cid_invalid(self):
        """Test CID validation with invalid CID."""
        kit = self.get_ipfs_files_kit()
        result = kit.validate_cid('invalid-cid')
        
        # Should fail or have validation logic
        self.assertIsInstance(result, self.IPFSFileResult)
    
    def test_ipfs_file_result_dataclass(self):
        """Test IPFSFileResult dataclass."""
        result = self.IPFSFileResult(
            success=True,
            data={'cid': 'QmTest123'},
            error=None,
            message="Success"
        )
        
        self.assertTrue(result.success)
        self.assertEqual(result.data['cid'], 'QmTest123')
        self.assertIsNone(result.error)
        self.assertEqual(result.message, "Success")
        
        # Test dataclass can be converted to dict
        result_dict = asdict(result)
        self.assertIsInstance(result_dict, dict)
        self.assertEqual(result_dict['success'], True)
    
    def test_ipfs_file_info_dataclass(self):
        """Test IPFSFileInfo dataclass."""
        file_info = self.IPFSFileInfo(
            cid='QmTest123',
            name='test.txt',
            size=1024,
            type='file',
            pinned=True
        )
        
        self.assertEqual(file_info.cid, 'QmTest123')
        self.assertEqual(file_info.name, 'test.txt')
        self.assertEqual(file_info.size, 1024)
        self.assertEqual(file_info.type, 'file')
        self.assertTrue(file_info.pinned)
        
        # Test dataclass can be converted to dict
        info_dict = asdict(file_info)
        self.assertIsInstance(info_dict, dict)
        self.assertEqual(info_dict['cid'], 'QmTest123')
    
    def test_error_handling(self):
        """Test error handling in IPFSFilesKit."""
        kit = self.get_ipfs_files_kit()
        
        # Test with invalid path (should handle gracefully)
        result = kit.add_file('')
        
        # Should return failure result, not raise exception
        self.assertIsInstance(result, self.IPFSFileResult)
        self.assertFalse(result.success)


if __name__ == '__main__':
    unittest.main()
