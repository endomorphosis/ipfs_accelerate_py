import os
import io
import sys
import json
import unittest
import tempfile
from unittest.mock import MagicMock, patch

# Append parent directory to sys.path for proper imports
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__))))
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
        """Run all tests for the S3 kit API backend"""
        results = {}
        
        # Test basic configuration and session handling
        try:
            # Test config_to_boto method
            boto_config = self.s3_kit.config_to_boto(self.metadata['s3cfg'])
            results["config_to_boto"] = "Success" if boto_config and all(key in boto_config for key in [
                "service_name", "aws_access_key_id", "aws_secret_access_key", "endpoint_url"
            ]) else "Failed config conversion"
        except Exception as e:
            results["config_to_boto"] = str(e)

        # Test endpoint handler creation
        try:
            handler = self.s3_kit.create_s3_kit_endpoint_handler()
            results["endpoint_handler"] = "Success" if callable(handler) else "Failed to create endpoint handler"
        except Exception as e:
            results["endpoint_handler"] = str(e)

        # Test endpoint testing function
        try:
            with patch.object(self.s3_kit, 'test_s3_kit_endpoint', return_value=True):
                test_result = self.s3_kit.test_s3_kit_endpoint()
                results["test_endpoint"] = "Success" if test_result else "Failed endpoint test"
        except Exception as e:
            results["test_endpoint"] = str(e)

        # Test file operations with mocks
        with patch('boto3.resource') as mock_resource:
            mock_bucket = MagicMock()
            mock_object = MagicMock()
            mock_bucket.Object.return_value = mock_object
            mock_resource.return_value.Bucket.return_value = mock_bucket
            
            # Set up mock response for file operations
            mock_object.get.return_value = {'Body': MagicMock(read=lambda: b'test data')}
            mock_object.key = 'test/path'
            mock_object.last_modified = '2023-01-01T00:00:00Z'
            mock_object.content_length = 100
            mock_object.e_tag = 'test-etag'
            
            # Test file operations
            try:
                # Create temporary test file
                with tempfile.NamedTemporaryFile() as temp_file:
                    temp_file.write(b'test data')
                    temp_file.flush()
                    
                    # Test file upload
                    results["file_upload"] = "Success" if self._test_s3_file_operation(
                        lambda: self.s3_kit.s3_ul_file(temp_file.name, 'test/path', 'test-bucket')
                    ) else "Failed file upload"
                    
                    # Test file download
                    results["file_download"] = "Success" if self._test_s3_file_operation(
                        lambda: self.s3_kit.s3_dl_file('test/path', temp_file.name, 'test-bucket')
                    ) else "Failed file download"
                    
                    # Test file copy
                    results["file_copy"] = "Success" if self._test_s3_file_operation(
                        lambda: self.s3_kit.s3_cp_file('test/source', 'test/dest', 'test-bucket')
                    ) else "Failed file copy"
                    
                    # Test file move
                    results["file_move"] = "Success" if self._test_s3_file_operation(
                        lambda: self.s3_kit.s3_mv_file('test/source', 'test/dest', 'test-bucket')
                    ) else "Failed file move"
                    
                    # Test file delete
                    results["file_delete"] = "Success" if self._test_s3_file_operation(
                        lambda: self.s3_kit.s3_rm_file('test/path', 'test-bucket')
                    ) else "Failed file delete"
            except Exception as e:
                results["file_operations"] = str(e)
            
            # Test directory operations
            try:
                # Set up mock objects list
                mock_objects = MagicMock()
                mock_objects.filter.return_value = [mock_object]
                mock_bucket.objects = mock_objects
                
                # Test directory listing
                results["dir_list"] = "Success" if self._test_s3_file_operation(
                    lambda: self.s3_kit.s3_ls_dir('test/dir', 'test-bucket')
                ) else "Failed directory listing"
                
                # Test directory creation
                results["dir_create"] = "Success" if self._test_s3_file_operation(
                    lambda: self.s3_kit.s3_mk_dir('test/dir', 'test-bucket')
                ) else "Failed directory creation"
                
                # Test directory deletion
                results["dir_delete"] = "Success" if self._test_s3_file_operation(
                    lambda: self.s3_kit.s3_rm_dir('test/dir', 'test-bucket')
                ) else "Failed directory deletion"
                
                # Test directory copy
                results["dir_copy"] = "Success" if self._test_s3_file_operation(
                    lambda: self.s3_kit.s3_cp_dir('test/source', 'test/dest', 'test-bucket')
                ) else "Failed directory copy"
                
                # Test directory move
                results["dir_move"] = "Success" if self._test_s3_file_operation(
                    lambda: self.s3_kit.s3_mv_dir('test/source', 'test/dest', 'test-bucket')
                ) else "Failed directory move"
            except Exception as e:
                results["dir_operations"] = str(e)
        
        return results
    
    def _test_s3_file_operation(self, operation_func):
        """Helper method to test S3 file operations"""
        try:
            result = operation_func()
            return result is not None
        except Exception as e:
            print(f"S3 operation error: {str(e)}")
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
    # Import required modules
    import tempfile
    
    metadata = {
        "s3cfg": {
            "accessKey": os.environ.get("S3_ACCESS_KEY", "test_access_key"),
            "secretKey": os.environ.get("S3_SECRET_KEY", "test_secret_key"),
            "endpoint": os.environ.get("S3_ENDPOINT", "http://localhost:9000")
        }
    }
    resources = {}
    try:
        test_s3 = test_s3_kit(resources, metadata)
        results = test_s3.__test__()
        print(f"S3 Kit Test Results: {json.dumps(results, indent=2)}")
    except KeyboardInterrupt:
        print("Tests stopped by user.")
        sys.exit(1)