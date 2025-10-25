"""
Integration tests for the Caselaw GraphRAG system.
"""

import unittest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from ipfs_accelerate_py.caselaw_dataset_loader import CaselawDatasetLoader
    from ipfs_accelerate_py.caselaw_graphrag_processor import CaselawGraphRAGProcessor
    from ipfs_accelerate_py.temporal_deontic_logic import TemporalDeonticLogicProcessor
    from ipfs_accelerate_py.caselaw_dashboard import CaselawDashboard
    from ipfs_accelerate_py.mcp_dashboard import MCPDashboard
    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"Import error: {e}")
    IMPORTS_AVAILABLE = False


class TestCaselawIntegration(unittest.TestCase):
    """Test caselaw system integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not IMPORTS_AVAILABLE:
            self.skipTest("Required modules not available")
    
    def test_dataset_loader(self):
        """Test dataset loader functionality."""
        loader = CaselawDatasetLoader()
        dataset_info = loader.load_sample_dataset(max_samples=5)
        
        self.assertIsInstance(dataset_info, dict)
        self.assertIn('total_cases', dataset_info)
        self.assertIn('all_cases', dataset_info)
        self.assertEqual(dataset_info['total_cases'], 5)
        
        # Test search functionality
        results = loader.search_cases('civil rights')
        self.assertIsInstance(results, list)
        
        # Test doctrines
        doctrines = loader.get_legal_doctrines()
        self.assertIsInstance(doctrines, list)
        self.assertGreater(len(doctrines), 0)
    
    def test_graphrag_processor(self):
        """Test GraphRAG processor functionality."""
        loader = CaselawDatasetLoader()
        dataset_info = loader.load_sample_dataset(max_samples=3)
        
        processor = CaselawGraphRAGProcessor()
        stats = processor.process_cases(dataset_info)
        
        self.assertIsInstance(stats, dict)
        self.assertIn('total_nodes', stats)
        self.assertIn('total_edges', stats)
        self.assertGreater(stats['total_nodes'], 0)
        
        # Test querying
        results = processor.query_graph('civil rights', limit=5)
        self.assertIsInstance(results, list)
    
    def test_temporal_logic_processor(self):
        """Test temporal deontic logic processor."""
        processor = TemporalDeonticLogicProcessor()
        
        # Create sample lineage
        lineage = [
            {
                'case_id': 'test-1',
                'doctrine': 'test doctrine',
                'year': 1967,
                'holding': 'Initial holding'
            },
            {
                'case_id': 'test-2',
                'doctrine': 'test doctrine',
                'year': 1980,
                'holding': 'Evolved holding'
            }
        ]
        
        analysis = processor.analyze_lineage(lineage)
        
        self.assertIsInstance(analysis, dict)
        self.assertIn('evolution_steps', analysis)
        self.assertIn('theorems', analysis)
        self.assertIn('consistency_score', analysis)
        
        # Test IPLD export
        ipld_data = processor.export_to_ipld(analysis)
        self.assertIsInstance(ipld_data, dict)
        self.assertIn('@context', ipld_data)
    
    def test_dashboard_initialization(self):
        """Test dashboard initialization."""
        # Test Caselaw Dashboard
        caselaw_dashboard = CaselawDashboard(port=5001)  # Use different port
        self.assertIsNotNone(caselaw_dashboard.app)
        
        # Test MCP Dashboard
        mcp_dashboard = MCPDashboard(port=8900)  # Use different port
        self.assertIsNotNone(mcp_dashboard.app)
    
    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow."""
        # Step 1: Load data
        loader = CaselawDatasetLoader()
        dataset_info = loader.load_sample_dataset(max_samples=5)
        
        # Step 2: Process with GraphRAG
        processor = CaselawGraphRAGProcessor()
        graph_stats = processor.process_cases(dataset_info)
        
        # Step 3: Temporal analysis
        temporal_processor = TemporalDeonticLogicProcessor()
        sample_lineage = [
            {
                'case_id': 'workflow-test-1',
                'doctrine': 'test doctrine',
                'year': 1970,
                'holding': 'First holding'
            },
            {
                'case_id': 'workflow-test-2', 
                'doctrine': 'test doctrine',
                'year': 1985,
                'holding': 'Second holding'
            }
        ]
        temporal_analysis = temporal_processor.analyze_lineage(sample_lineage)
        
        # Verify all steps completed successfully
        self.assertIsInstance(dataset_info, dict)
        self.assertIsInstance(graph_stats, dict)
        self.assertIsInstance(temporal_analysis, dict)
        
        self.assertGreater(graph_stats['total_nodes'], 0)
        self.assertGreater(len(temporal_analysis['evolution_steps']), 0)


if __name__ == '__main__':
    unittest.main()