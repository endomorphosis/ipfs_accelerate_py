"""
Unit tests for network_kit module.

Tests the NetworkKit class and its methods for network and peer operations.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from dataclasses import asdict


class TestNetworkKit(unittest.TestCase):
    """Test cases for NetworkKit class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Import here to avoid import errors if module not available
        try:
            from ipfs_accelerate_py.kit.network_kit import (
                NetworkKit,
                NetworkConfig,
                NetworkResult,
                PeerInfo,
                BandwidthStats,
                get_network_kit
            )
            self.NetworkKit = NetworkKit
            self.NetworkConfig = NetworkConfig
            self.NetworkResult = NetworkResult
            self.PeerInfo = PeerInfo
            self.BandwidthStats = BandwidthStats
            self.get_network_kit = get_network_kit
        except ImportError:
            self.skipTest("network_kit module not available")
    
    def test_network_kit_initialization(self):
        """Test NetworkKit initialization."""
        config = self.NetworkConfig()
        kit = self.NetworkKit(config)
        
        self.assertIsInstance(kit, self.NetworkKit)
        self.assertEqual(kit.config, config)
    
    def test_get_network_kit_singleton(self):
        """Test get_network_kit returns singleton."""
        kit1 = self.get_network_kit()
        kit2 = self.get_network_kit()
        
        self.assertIs(kit1, kit2)
    
    @patch('subprocess.run')
    def test_list_peers(self, mock_run):
        """Test listing connected peers."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = 'QmPeer1\nQmPeer2\nQmPeer3'
        mock_result.stderr = ''
        mock_run.return_value = mock_result
        
        kit = self.get_network_kit()
        result = kit.list_peers()
        
        self.assertTrue(result.success)
        self.assertIn('peers', result.data)
        self.assertIsNone(result.error)
    
    @patch('subprocess.run')
    def test_connect_peer_success(self, mock_run):
        """Test connecting to a peer successfully."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = 'connect QmPeer123 success'
        mock_result.stderr = ''
        mock_run.return_value = mock_result
        
        kit = self.get_network_kit()
        result = kit.connect_peer('/ip4/1.2.3.4/tcp/4001/p2p/QmPeer123')
        
        self.assertTrue(result.success)
        self.assertIsNone(result.error)
    
    @patch('subprocess.run')
    def test_connect_peer_failure(self, mock_run):
        """Test connecting to a peer with failure."""
        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stdout = ''
        mock_result.stderr = 'Error: failed to dial'
        mock_run.return_value = mock_result
        
        kit = self.get_network_kit()
        result = kit.connect_peer('/ip4/1.2.3.4/tcp/4001/p2p/QmPeer123')
        
        self.assertFalse(result.success)
        self.assertIsNotNone(result.error)
    
    @patch('subprocess.run')
    def test_disconnect_peer(self, mock_run):
        """Test disconnecting from a peer."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = 'disconnect QmPeer123 success'
        mock_result.stderr = ''
        mock_run.return_value = mock_result
        
        kit = self.get_network_kit()
        result = kit.disconnect_peer('QmPeer123')
        
        self.assertTrue(result.success)
        self.assertIsNone(result.error)
    
    @patch('subprocess.run')
    def test_dht_put(self, mock_run):
        """Test storing a value in DHT."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = 'success'
        mock_result.stderr = ''
        mock_run.return_value = mock_result
        
        kit = self.get_network_kit()
        result = kit.dht_put('test_key', 'test_value')
        
        self.assertTrue(result.success)
        self.assertIsNone(result.error)
    
    @patch('subprocess.run')
    def test_dht_get(self, mock_run):
        """Test retrieving a value from DHT."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = 'test_value'
        mock_result.stderr = ''
        mock_run.return_value = mock_result
        
        kit = self.get_network_kit()
        result = kit.dht_get('test_key')
        
        self.assertTrue(result.success)
        self.assertIn('value', result.data)
        self.assertEqual(result.data['value'], 'test_value')
    
    @patch('subprocess.run')
    def test_get_swarm_info(self, mock_run):
        """Test getting swarm information."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = 'Peers: 5\nAddresses: 3'
        mock_result.stderr = ''
        mock_run.return_value = mock_result
        
        kit = self.get_network_kit()
        result = kit.get_swarm_info()
        
        self.assertTrue(result.success)
        self.assertIn('swarm', result.data)
        self.assertIsNone(result.error)
    
    @patch('subprocess.run')
    def test_get_bandwidth_stats(self, mock_run):
        """Test getting bandwidth statistics."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = 'TotalIn: 1024\nTotalOut: 2048\nRateIn: 100\nRateOut: 200'
        mock_result.stderr = ''
        mock_run.return_value = mock_result
        
        kit = self.get_network_kit()
        result = kit.get_bandwidth_stats()
        
        self.assertTrue(result.success)
        self.assertIn('bandwidth', result.data)
        self.assertIsNone(result.error)
    
    @patch('subprocess.run')
    def test_ping_peer(self, mock_run):
        """Test pinging a peer."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = 'Pong received from QmPeer123\nLatency: 50ms'
        mock_result.stderr = ''
        mock_run.return_value = mock_result
        
        kit = self.get_network_kit()
        result = kit.ping_peer('QmPeer123', count=3)
        
        self.assertTrue(result.success)
        self.assertIn('ping', result.data)
        self.assertIsNone(result.error)
    
    def test_network_result_dataclass(self):
        """Test NetworkResult dataclass."""
        result = self.NetworkResult(
            success=True,
            data={'peers': ['QmPeer1', 'QmPeer2']},
            error=None
        )
        
        self.assertTrue(result.success)
        self.assertEqual(len(result.data['peers']), 2)
        self.assertIsNone(result.error)
        
        # Test dataclass can be converted to dict
        result_dict = asdict(result)
        self.assertIsInstance(result_dict, dict)
        self.assertEqual(result_dict['success'], True)
    
    def test_peer_info_dataclass(self):
        """Test PeerInfo dataclass."""
        peer_info = self.PeerInfo(
            peer_id='QmPeer123',
            addresses=['/ip4/1.2.3.4/tcp/4001'],
            latency=50.5,
            connected=True
        )
        
        self.assertEqual(peer_info.peer_id, 'QmPeer123')
        self.assertEqual(len(peer_info.addresses), 1)
        self.assertEqual(peer_info.latency, 50.5)
        self.assertTrue(peer_info.connected)
        
        # Test dataclass can be converted to dict
        info_dict = asdict(peer_info)
        self.assertIsInstance(info_dict, dict)
        self.assertEqual(info_dict['peer_id'], 'QmPeer123')
    
    def test_bandwidth_stats_dataclass(self):
        """Test BandwidthStats dataclass."""
        stats = self.BandwidthStats(
            total_in=1024,
            total_out=2048,
            rate_in=100.5,
            rate_out=200.5
        )
        
        self.assertEqual(stats.total_in, 1024)
        self.assertEqual(stats.total_out, 2048)
        self.assertEqual(stats.rate_in, 100.5)
        self.assertEqual(stats.rate_out, 200.5)
        
        # Test dataclass can be converted to dict
        stats_dict = asdict(stats)
        self.assertIsInstance(stats_dict, dict)
        self.assertEqual(stats_dict['total_in'], 1024)
    
    def test_error_handling(self):
        """Test error handling in NetworkKit."""
        kit = self.get_network_kit()
        
        # Test with invalid peer ID (should handle gracefully)
        result = kit.disconnect_peer('')
        
        # Should return failure result, not raise exception
        self.assertIsInstance(result, self.NetworkResult)
        self.assertFalse(result.success)


if __name__ == '__main__':
    unittest.main()
