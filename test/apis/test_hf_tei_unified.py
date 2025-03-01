import os
import sys
import json
import argparse
import unittest
import time
import datetime
from unittest.mock import MagicMock, patch

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'ipfs_accelerate_py'))

# Import the test classes
from test_hf_tei import test_hf_tei
from test_hf_tei_container import test_hf_tei_container

# Import for performance testing
import numpy as np

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
        
        # Standard test sentences for performance and consistency testing
        self.test_sentences = [
            "The quick brown fox jumps over the lazy dog.",
            "I enjoy walking in the park on sunny days.",
            "Machine learning models can be used for natural language processing tasks.",
            "The capital city of France is Paris, which is known for the Eiffel Tower.",
            "Deep learning has revolutionized the field of computer vision in recent years."
        ]
    
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
    
    def test_performance(self):
        """Test the performance of the TEI API"""
        # Skip if performance tests disabled
        if os.environ.get("SKIP_PERFORMANCE_TESTS", "").lower() in ("true", "1", "yes"):
            self.skipTest("Performance tests disabled by environment variable")
        
        performance_results = {}
        
        with patch.object(requests, 'post') as mock_post:
            # Create mock response with realistic embedding dimensions
            embedding_dim = 384  # Standard dimension for many embedding models
            mock_response = MagicMock()
            mock_response.status_code = 200
            
            # For single embedding
            mock_response.json.return_value = np.random.rand(embedding_dim).tolist()
            mock_post.return_value = mock_response
            
            # Initialize test objects
            standard_tester = test_hf_tei(self.resources, self.metadata)
            
            # Test API endpoint
            endpoint_url = f"https://api-inference.huggingface.co/models/{self.metadata.get('model_id')}"
            api_key = self.metadata.get("hf_api_key", "")
            
            # Test single embedding performance
            start_time = time.time()
            for _ in range(10):  # Run 10 iterations for more stable measurement
                standard_tester.hf_tei.make_post_request_hf_tei(
                    endpoint_url, {"inputs": self.test_sentences[0]}, api_key
                )
            single_time = (time.time() - start_time) / 10  # Average time
            performance_results["single_embedding_time"] = f"{single_time:.4f}s"
            
            # Test batch embedding performance
            # Update mock for batch response
            mock_response.json.return_value = [np.random.rand(embedding_dim).tolist() for _ in range(len(self.test_sentences))]
            mock_post.return_value = mock_response
            
            start_time = time.time()
            for _ in range(10):  # Run 10 iterations
                standard_tester.hf_tei.make_post_request_hf_tei(
                    endpoint_url, {"inputs": self.test_sentences}, api_key
                )
            batch_time = (time.time() - start_time) / 10  # Average time
            performance_results["batch_embedding_time"] = f"{batch_time:.4f}s"
            
            # Calculate throughput
            performance_results["sentences_per_second"] = f"{len(self.test_sentences) / batch_time:.2f}"
            
            # Speedup factor
            speedup = (single_time * len(self.test_sentences)) / batch_time
            performance_results["batch_speedup_factor"] = f"{speedup:.2f}x"
            
        # Save performance results
        performance_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 
            'collected_results', 
            f'hf_tei_performance_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        )
        os.makedirs(os.path.dirname(performance_file), exist_ok=True)
        with open(performance_file, 'w') as f:
            json.dump(performance_results, f, indent=2)
            
        return performance_results
    
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
    parser.add_argument('--performance', action='store_true', help='Run performance tests')
    parser.add_argument('--all', action='store_true', help='Run all tests (standard, container, performance)')
    parser.add_argument('--model', type=str, default='BAAI/bge-small-en-v1.5', help='Model ID to use for testing')
    parser.add_argument('--container-url', type=str, help='URL for TEI container')
    parser.add_argument('--api-key', type=str, help='HuggingFace API key')
    parser.add_argument('--batch-size', type=int, default=5, help='Batch size for performance tests')
    parser.add_argument('--compare', action='store_true', help='Compare results with expected results')
    parser.add_argument('--timeout', type=int, default=30, help='Timeout in seconds for API requests')
    
    args = parser.parse_args()
    
    # Set environment variables from arguments
    if args.model:
        os.environ["HF_MODEL_ID"] = args.model
    if args.container_url:
        os.environ["HF_CONTAINER_URL"] = args.container_url
    if args.api_key:
        os.environ["HF_API_KEY"] = args.api_key
    if args.timeout:
        os.environ["HF_API_TIMEOUT"] = str(args.timeout)
    
    # Skip container tests if requested
    if args.standard and not args.container and not args.all:
        os.environ["SKIP_CONTAINER_TESTS"] = "true"
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add standard API tests
    if args.standard or args.all or (not args.standard and not args.container and not args.performance):
        suite.addTest(TestHuggingFaceTEI('test_standard_api'))
    
    # Add container API tests
    if args.container or args.all or (not args.standard and not args.container and not args.performance):
        suite.addTest(TestHuggingFaceTEI('test_container_api'))
    
    # Add performance tests
    if args.performance or args.all:
        suite.addTest(TestHuggingFaceTEI('test_performance'))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Save summary report
    summary = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": os.environ.get("HF_MODEL_ID", "BAAI/bge-small-en-v1.5"),
        "tests_run": len(result.failures) + len(result.errors) + result.testsRun - len(result.skipped),
        "success": result.wasSuccessful(),
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped)
    }
    
    # Create summary file with timestamp
    summary_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'collected_results',
        f'hf_tei_summary_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    )
    os.makedirs(os.path.dirname(summary_file), exist_ok=True)
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    return 0 if result.wasSuccessful() else 1

if __name__ == "__main__":
    # Make sure imports are available
    import requests
    
    # Run tests and exit with appropriate status code
    sys.exit(run_tests())