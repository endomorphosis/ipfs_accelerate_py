import os
import io
import sys
import json
import tempfile
from unittest.mock import MagicMock, patch

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'ipfs_accelerate_py'))
from api_backends import apis, s3_kit

class test_s3_kit:
    def __init__(self, resources=None, metadata=None):
        self.resources = resources if resources else {}
        self.metadata = metadata if metadata else {
            "s3cfg": {
                "accessKey": os.environ.get("S3_ACCESS_KEY", "test_access_key"),
                "secretKey": os.environ.get("S3_SECRET_KEY", "test_secret_key"),
                "endpoint": os.environ.get("S3_ENDPOINT", "http://localhost:9000")
            }
        }
        self.s3_kit = s3_kit(resources=self.resources, metadata=self.metadata)
        return None
    
    def test(self):
        """Run all tests for the S3 API backend"""
        results = {}
        
        # Test endpoint handler creation
        try:
            endpoint_url = self.metadata["s3cfg"]["endpoint"]
            endpoint_handler = self.s3_kit.create_s3_endpoint_handler(endpoint_url)
            results["endpoint_handler"] = "Success" if callable(endpoint_handler) else "Failed to create endpoint handler"
        except Exception as e:
            results["endpoint_handler"] = f"Error: {str(e)}"
            
        # Test S3 file operations with mocked boto3
        try:
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(b"Test content")
                temp_file_path = temp_file.name
            
            # Test upload
            upload_result = self._test_s3_file_operation(
                lambda: self.s3_kit.upload_file(
                    temp_file_path,
                    "test-bucket",
                    "test-key.txt"
                )
            )
            results["upload_file"] = "Success" if upload_result else "Failed upload operation"
            
            # Test download
            download_result = self._test_s3_file_operation(
                lambda: self.s3_kit.download_file(
                    "test-bucket",
                    "test-key.txt",
                    temp_file_path
                )
            )
            results["download_file"] = "Success" if download_result else "Failed download operation"
            
            # Clean up temp file
            os.unlink(temp_file_path)
            
        except Exception as e:
            results["file_operations"] = f"Error: {str(e)}"
            
        # Test error handling
        try:
            with patch('boto3.client') as mock_client:
                # Test connection error
                mock_client.side_effect = Exception("Connection failed")
                
                try:
                    self.s3_kit.create_s3_endpoint_handler("http://invalid:9000")
                    results["error_handling_connection"] = "Failed to catch connection error"
                except Exception:
                    results["error_handling_connection"] = "Success"
                    
                # Test invalid credentials
                mock_client.side_effect = None
                mock_s3 = MagicMock()
                mock_s3.upload_file.side_effect = Exception("Invalid credentials")
                mock_client.return_value = mock_s3
                
                try:
                    self.s3_kit.upload_file("nonexistent.txt", "test-bucket", "test.txt")
                    results["error_handling_auth"] = "Failed to catch auth error"
                except Exception:
                    results["error_handling_auth"] = "Success"
        except Exception as e:
            results["error_handling"] = f"Error: {str(e)}"
        
        return results
    
    def _test_s3_file_operation(self, operation_func):
        """Helper method to test S3 file operations with mocked boto3"""
        with patch('boto3.client') as mock_client:
            mock_s3 = MagicMock()
            mock_client.return_value = mock_s3
            
            try:
                operation_func()
                return True
            except Exception:
                return False

    def __test__(self):
        """Run tests and compare/save results"""
        test_results = {}
        try:
            test_results = self.test()
        except Exception as e:
            test_results = {"test_error": str(e)}
        
        # Create directories if they don't exist
        expected_dir = os.path.join(os.path.dirname(__file__), 'expected_results')
        collected_dir = os.path.join(os.path.dirname(__file__), 'collected_results')
        os.makedirs(expected_dir, exist_ok=True)
        os.makedirs(collected_dir, exist_ok=True)
        
        # Save collected results
        with open(os.path.join(collected_dir, 's3_kit_test_results.json'), 'w') as f:
            json.dump(test_results, f, indent=2)
            
        # Compare with expected results if they exist
        expected_file = os.path.join(expected_dir, 's3_kit_test_results.json')
        if os.path.exists(expected_file):
            with open(expected_file, 'r') as f:
                expected_results = json.load(f)
                if expected_results != test_results:
                    print("Test results differ from expected results!")
                    print(f"Expected: {expected_results}")
                    print(f"Got: {test_results}")
        else:
            # Create expected results file if it doesn't exist
            with open(expected_file, 'w') as f:
                json.dump(test_results, f, indent=2)
                print(f"Created new expected results file: {expected_file}")

        return test_results
        
if __name__ == "__main__":
    metadata = {
        "s3cfg": {
            "accessKey": os.environ.get("S3_ACCESS_KEY", "test_access_key"),
            "secretKey": os.environ.get("S3_SECRET_KEY", "test_secret_key"),
            "endpoint": os.environ.get("S3_ENDPOINT", "http://localhost:9000")
        }
    }
    resources = {}
    try:
        this_s3_kit = test_s3_kit(resources, metadata)
        results = this_s3_kit.__test__()
        print(f"S3 Kit Test Results: {json.dumps(results, indent=2)}")
    except KeyboardInterrupt:
        print("Tests stopped by user.")
        sys.exit(1)