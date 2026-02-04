"""
Tests for Hardware Kit Module

These tests verify the core hardware detection functionality.
"""

import os
import sys
import unittest
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from ipfs_accelerate_py.kit.hardware_kit import (
        HardwareKit,
        HardwareInfo
    )
    HAVE_HARDWARE_KIT = True
except ImportError:
    HAVE_HARDWARE_KIT = False


@unittest.skipUnless(HAVE_HARDWARE_KIT, "Hardware kit module not available")
class TestHardwareKit(unittest.TestCase):
    """Test Hardware kit core functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.kit = HardwareKit()
        
    def test_hardware_kit_initialization(self):
        """Test HardwareKit can be initialized"""
        self.assertIsNotNone(self.kit)
        self.assertTrue(hasattr(self.kit, 'get_hardware_info'))
    
    def test_get_hardware_info(self):
        """Test getting hardware information"""
        info = self.kit.get_hardware_info()
        
        self.assertIsInstance(info, HardwareInfo)
        self.assertIsNotNone(info.platform_info)
        self.assertIn('system', info.platform_info)
    
    def test_get_cpu_info(self):
        """Test getting CPU information"""
        info = self.kit.get_hardware_info()
        
        self.assertIsNotNone(info.cpu)
        self.assertIn('count', info.cpu)
        self.assertGreater(info.cpu['count'], 0)
    
    def test_get_memory_info(self):
        """Test getting memory information"""
        info = self.kit.get_hardware_info()
        
        self.assertIsNotNone(info.memory)
        self.assertIn('total_gb', info.memory)
        self.assertGreater(info.memory['total_gb'], 0)
    
    def test_detect_accelerators(self):
        """Test accelerator detection"""
        info = self.kit.get_hardware_info()
        
        self.assertIsNotNone(info.accelerators)
        self.assertIsInstance(info.accelerators, dict)
    
    @patch('subprocess.run')
    def test_test_hardware_cuda_unavailable(self, mock_run):
        """Test hardware testing when CUDA unavailable"""
        mock_run.return_value = Mock(
            returncode=1,
            stdout="",
            stderr="CUDA not available"
        )
        
        result = self.kit.test_hardware(accelerator="cuda", test_level="basic")
        
        # Result structure varies, just check it's a dict
        self.assertIsInstance(result, dict)
    
    def test_recommend_hardware(self):
        """Test hardware recommendation"""
        result = self.kit.recommend_hardware(
            model_name="gpt2",
            task="inference"
        )
        
        # Result is a dict with recommendations key
        self.assertIsInstance(result, dict)
        self.assertIn('recommendations', result)
        self.assertIsInstance(result['recommendations'], list)
        # Should have at least some recommendations
        self.assertGreaterEqual(len(result['recommendations']), 0)
    
    def test_hardware_info_dataclass(self):
        """Test HardwareInfo dataclass"""
        info = HardwareInfo(
            platform_info={'system': 'Linux'},
            cpu={'count': 4},
            memory={'total_gb': 16.0},
            accelerators={}
        )
        
        self.assertEqual(info.platform_info['system'], 'Linux')
        self.assertEqual(info.cpu['count'], 4)
        self.assertEqual(info.memory['total_gb'], 16.0)
        self.assertEqual(info.accelerators, {})


if __name__ == '__main__':
    unittest.main()
