import os
import sys
import json
import argparse
import unittest
from unittest.mock import MagicMock, patch

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'ipfs_accelerate_py'))

# Import the test classes
from test_hf_tei import test_hf_tei
from test_hf_tei_container import test_hf_tei_container

class TestHuggingFaceTEI(unittest.TestCase):
    """Unified test suite for HuggingFace Text Embedding Inference API"""
    
    def setUp(self):
        """Set up test environment"""
        self.metadata = {
            "hf_api_key": os.environ.get("HF_API_KEY", ""),
            "hf_container_url": os.environ.get("HF_CONTAINER_URL", "http://localhost:8080"),
            "docker_registry": os.environ.get("DOCKER_REGISTRY", "ghcr.io/huggingface/text-embeddings-inference"),
            "container_tag": os.environ.get("CONTAINER_TAG", "latest"),
            "gpu_device": os.environ.get("GPU_DEVICE", "0"),
            "model_id": os.environ.get("HF_MODEL_ID", "BAAI/bge-small-en-v1.5")
        }
        self.resources = {}
    
    def test_standard_api(self):
        """Test the standard HuggingFace TEI API"""
        tester = test_hf_tei(self.resources, self.metadata)
        results = tester.test()
        
        # Verify critical tests passed
        self.assertEqual(results.get("endpoint_handler"), "Success")
        self.assertEqual(results.get("test_endpoint"), "Success")
        self.assertEqual(results.get("post_request"), "Success")
        
        # Save test results
        self._save_test_results(results, "hf_tei_test_results.json")
    
    def test_container_api(self):
        """Test the container-based HuggingFace TEI API"""
        # Skip if explicitly disabled
        if os.environ.get("SKIP_CONTAINER_TESTS", "").lower() in ("true", "1", "yes"):
            self.skipTest("Container tests disabled by environment variable")
        
        with patch('subprocess.run') as mock_run:
            # Mock successful container operations
            mock_run.return_value = MagicMock(returncode=0, stdout="container_id_123")
            
            with patch.object(requests, 'get') as mock_get, \
                 patch.object(requests, 'post') as mock_post:
                
                # Mock successful API responses
                mock_get_response = MagicMock()
                mock_get_response.status_code = 200
                mock_get_response.json.return_value = {
                    "model_id": self.metadata.get("model_id"),
                    "dim": 384,  # Common embedding dimension
                    "status": "ok"
                }
                mock_get.return_value = mock_get_response
                
                mock_post_response = MagicMock()
                mock_post_response.status_code = 200
                mock_post_response.json.return_value = [0.1, 0.2, 0.3] * 100  # 300-dimensional vector
                mock_post.return_value = mock_post_response
                
                # Run container tests
                tester = test_hf_tei_container(self.resources, self.metadata)
                results = tester.test()
                
                # Verify critical tests passed
                self.assertEqual(results.get("container_deployment"), "Success")
                self.assertEqual(results.get("embedding_generation"), "Success")
                self.assertEqual(results.get("batch_embedding"), "Success")
                self.assertEqual(results.get("container_shutdown"), "Success")
                
                # Save test results
                self._save_test_results(results, "hf_tei_container_test_results.json")
    
    def _save_test_results(self, results, filename):
        """Save test results to file"""
        # Create directories if they don't exist
        base_dir = os.path.dirname(os.path.abspath(__file__))
        collected_dir = os.path.join(base_dir, 'collected_results')
        os.makedirs(collected_dir, exist_ok=True)
        
        # Save results
        results_file = os.path.join(collected_dir, filename)
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

def run_tests():
    """Run all tests or selected tests based on command line arguments"""
    parser = argparse.ArgumentParser(description='Test HuggingFace Text Embedding Inference API')
    parser.add_argument('--standard', action='store_true', help='Run standard API tests only')
    parser.add_argument('--container', action='store_true', help='Run container API tests only')
    parser.add_argument('--model', type=str, default='BAAI/bge-small-en-v1.5', help='Model ID to use for testing')
    parser.add_argument('--container-url', type=str, help='URL for TEI container')
    parser.add_argument('--api-key', type=str, help='HuggingFace API key')
    
    args = parser.parse_args()
    
    # Set environment variables from arguments
    if args.model:
        os.environ["HF_MODEL_ID"] = args.model
    if args.container_url:
        os.environ["HF_CONTAINER_URL"] = args.container_url
    if args.api_key:
        os.environ["HF_API_KEY"] = args.api_key
    
    # Create test suite
    suite = unittest.TestSuite()
    
    if args.standard or (not args.standard and not args.container):
        suite.addTest(TestHuggingFaceTEI('test_standard_api'))
    
    if args.container or (not args.standard and not args.container):
        suite.addTest(TestHuggingFaceTEI('test_container_api'))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return 0 if result.wasSuccessful() else 1

if __name__ == "__main__":
    # Make sure imports are available
    import requests
    
    # Run tests and exit with appropriate status code
    sys.exit(run_tests())