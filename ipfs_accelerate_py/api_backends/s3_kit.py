import os
import time
import threading
import hashlib
import uuid
import boto3
from concurrent.futures import Future
from queue import Queue
from dotenv import load_dotenv

class s3_kit:
    def __init__(self, resources=None, metadata=None):
        self.resources = resources if resources else {}
        self.metadata = metadata if metadata else {}
        
        # Get S3 configuration from metadata or environment
        self.s3cfg = self._get_s3_config()
        
        # Initialize queue and backoff systems
        self.max_concurrent_requests = 5
        self.queue_size = 100
        self.request_queue = Queue(maxsize=self.queue_size)
        self.active_requests = 0
        self.queue_lock = threading.RLock()
        
        # Start queue processor
        self.queue_processor = threading.Thread(target=self._process_queue)
        self.queue_processor.daemon = True
        self.queue_processor.start()
        
        # Initialize backoff configuration
        self.max_retries = 5
        self.initial_retry_delay = 1
        self.backoff_factor = 2
        self.max_retry_delay = 16
        
        # Request tracking
        self.request_tracking = True
        self.recent_requests = {}
        
        
        # Retry and backoff settings
        self.max_retries = 5
        self.initial_retry_delay = 1
        self.backoff_factor = 2
        self.max_retry_delay = 60  # Maximum delay in seconds
        
        # Request queue settings
        self.queue_enabled = True
        self.queue_size = 100
        self.queue_processing = False
        self.current_requests = 0
        self.max_concurrent_requests = 5
        self.request_queue = []
        self.queue_lock = threading.RLock()
        return None

    def _get_s3_config(self):
        """Get S3 configuration from metadata or environment"""
        # Try to get from metadata
        if "s3cfg" in self.metadata:
            return self.metadata["s3cfg"]
        
        # Try to get from environment
        s3cfg = {
            "accessKey": os.environ.get("S3_ACCESS_KEY"),
            "secretKey": os.environ.get("S3_SECRET_KEY"),
            "endpoint": os.environ.get("S3_ENDPOINT", "https://s3.amazonaws.com")
        }
        
        # Check if we have essential keys
        if s3cfg["accessKey"] and s3cfg["secretKey"]:
            return s3cfg
        
        # Try to load from dotenv
        try:
            load_dotenv()
            s3cfg = {
                "accessKey": os.environ.get("S3_ACCESS_KEY"),
                "secretKey": os.environ.get("S3_SECRET_KEY"),
                "endpoint": os.environ.get("S3_ENDPOINT", "https://s3.amazonaws.com")
            }
            
            if s3cfg["accessKey"] and s3cfg["secretKey"]:
                return s3cfg
        except ImportError:
            pass
        
        # Return default with placeholders if no config found
        return {
            "accessKey": "default_access_key",
            "secretKey": "default_secret_key",
            "endpoint": "https://s3.amazonaws.com"
        }
        
    def _process_queue(self):
        """Process queued requests with proper concurrency management"""
        while True:
            try:
                future, operation, args, kwargs, request_id = self.request_queue.get()
                
                with self.queue_lock:
                    self.active_requests += 1
                
                # Process with retry logic
                retry_count = 0
                while retry_count <= self.max_retries:
                    try:
                        # Create S3 client
                        s3_client = self._get_s3_client()
                        
                        # Dispatch to appropriate operation
                        if operation == "upload_file":
                            result = s3_client.upload_file(**kwargs)
                        elif operation == "download_file":
                            result = s3_client.download_file(**kwargs)
                        elif operation == "list_objects":
                            result = s3_client.list_objects_v2(**kwargs)
                        elif operation == "delete_object":
                            result = s3_client.delete_object(**kwargs)
                        elif operation == "head_object":
                            result = s3_client.head_object(**kwargs)
                        else:
                            # For other operations, call the method dynamically
                            result = getattr(s3_client, operation)(**kwargs)
                        
                        # Update tracking with success
                        if self.request_tracking:
                            self.recent_requests[request_id] = {
                                "timestamp": time.time(),
                                "operation": operation,
                                "status": "success"
                            }
                        
                        future.set_result(result)
                        break
                        
                    except Exception as e:
                        retry_count += 1
                        
                        if retry_count > self.max_retries:
                            # Update tracking with error
                            if self.request_tracking:
                                self.recent_requests[request_id] = {
                                    "timestamp": time.time(),
                                    "operation": operation,
                                    "status": "error",
                                    "error": str(e)
                                }
                            
                            future.set_exception(e)
                            break
                        
                        # Calculate backoff delay
                        delay = min(
                            self.initial_retry_delay * (self.backoff_factor ** (retry_count - 1)),
                            self.max_retry_delay
                        )
                        
                        # Sleep with backoff
                        time.sleep(delay)
                
                with self.queue_lock:
                    self.active_requests -= 1
                
                self.request_queue.task_done()
                
            except Exception as e:
                print(f"Error in queue processor: {e}")
    
    def _get_s3_client(self):
        """Create an S3 client"""
        return boto3.client(
            's3',
            aws_access_key_id=self.s3cfg['accessKey'],
            aws_secret_access_key=self.s3cfg['secretKey'],
            endpoint_url=self.s3cfg['endpoint']
        )
    
    def _queue_operation(self, operation, **kwargs):
        """Queue an S3 operation with backoff retry"""
        # Generate unique request ID
        request_id = str(uuid.uuid4())
        
        # Queue system with proper concurrency management
        future = Future()
        
        # Add to queue
        self.request_queue.put((future, operation, [], kwargs, request_id))
        
        # Get result (blocks until request is processed)
        return future.result()
            
    def upload_file(self, file_path, bucket, key):
        """Upload a file to S3"""
        return self._queue_operation(
            "upload_file",
            Filename=file_path,
            Bucket=bucket,
            Key=key
        )
    
    def download_file(self, bucket, key, file_path):
        """Download a file from S3"""
        return self._queue_operation(
            "download_file",
            Bucket=bucket,
            Key=key,
            Filename=file_path
        )
    
    def list_objects(self, bucket, prefix=None):
        """List objects in a bucket"""
        kwargs = {"Bucket": bucket}
        if prefix:
            kwargs["Prefix"] = prefix
            
        return self._queue_operation("list_objects_v2", **kwargs)
    
    def delete_object(self, bucket, key):
        """Delete an object from S3"""
        return self._queue_operation(
            "delete_object",
            Bucket=bucket,
            Key=key
        )
    
    def head_object(self, bucket, key):
        """Get object metadata"""
        return self._queue_operation(
            "head_object",
            Bucket=bucket,
            Key=key
        )
    
    def create_presigned_url(self, bucket, key, expiration=3600):
        """Create a presigned URL for an object"""
        s3_client = self._get_s3_client()
        return s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': bucket, 'Key': key},
            ExpiresIn=expiration
        )
        
    def create_s3_endpoint_handler(self, endpoint_url=None):
        """Create an endpoint handler for S3 operations"""
        # Use provided endpoint or default from config
        if endpoint_url:
            self.s3cfg["endpoint"] = endpoint_url
            
        async def endpoint_handler(operation, **kwargs):
            """Handle requests to S3 endpoint"""
            try:
                if operation == "upload_file":
                    return self.upload_file(
                        kwargs.get("file_path"),
                        kwargs.get("bucket"),
                        kwargs.get("key")
                    )
                elif operation == "download_file":
                    return self.download_file(
                        kwargs.get("bucket"),
                        kwargs.get("key"),
                        kwargs.get("file_path")
                    )
                elif operation == "list_objects":
                    return self.list_objects(
                        kwargs.get("bucket"),
                        kwargs.get("prefix")
                    )
                elif operation == "delete_object":
                    return self.delete_object(
                        kwargs.get("bucket"),
                        kwargs.get("key")
                    )
                elif operation == "head_object":
                    return self.head_object(
                        kwargs.get("bucket"),
                        kwargs.get("key")
                    )
                elif operation == "create_presigned_url":
                    return self.create_presigned_url(
                        kwargs.get("bucket"),
                        kwargs.get("key"),
                        kwargs.get("expiration", 3600)
                    )
                else:
                    return {"error": f"Unsupported operation: {operation}"}
            except Exception as e:
                print(f"Error handling S3 operation: {e}")
                return {"error": str(e)}
        
        return endpoint_handler
    
    def test_s3_endpoint(self, endpoint_url=None):
        """Test the S3 endpoint"""
        if endpoint_url:
            self.s3cfg["endpoint"] = endpoint_url
            
        try:
            # Create a simple test - just list buckets
            s3_client = self._get_s3_client()
            response = s3_client.list_buckets()
            return True
        except Exception as e:
            print(f"Error testing S3 endpoint: {e}")
            return False