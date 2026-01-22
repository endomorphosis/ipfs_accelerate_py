#!/usr/bin/env python3
"""
API Distributed Testing Worker Node

This script runs a worker node for the API Distributed Testing framework.
The worker node connects to a coordinator server and executes API test tasks.
"""

import os
import sys
import time
import json
import logging
import argparse
import datetime
import platform
from typing import Dict, Any, Optional

# Add parent directory to path for local development
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import needed modules
try:
    from distributed_testing.worker import WorkerNode
    from distributed_testing.task import Task, TaskResult, TaskStatus
    from api_unified_testing_interface import (
        APIBackendFactory, 
        APITester, 
        APIProvider, 
        APITestType
    )
    from api_anomaly_detection import AnomalyDetector
    from api_predictive_analytics import TimeSeriesPredictor
except ImportError:
    # Add test directory to path
    sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "test"))
    # Try import again
    from distributed_testing.worker import WorkerNode
    from distributed_testing.task import Task, TaskResult, TaskStatus
    from api_unified_testing_interface import (
        APIBackendFactory, 
        APITester, 
        APIProvider, 
        APITestType
    )
    from api_anomaly_detection import AnomalyDetector
    from api_predictive_analytics import TimeSeriesPredictor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class APIWorkerNode(WorkerNode):
    """
    Worker node for executing API test tasks in the distributed testing framework.
    """
    
    def __init__(self, 
                 coordinator_url: str,
                 worker_id: Optional[str] = None,
                 enable_anomaly_detection: bool = True,
                 enable_predictive_analytics: bool = True,
                 results_dir: str = "./api_test_results",
                 tags: Optional[list] = None):
        """
        Initialize the API worker node.
        
        Args:
            coordinator_url: URL of the coordinator server
            worker_id: Custom worker ID (if None, a unique ID will be generated)
            enable_anomaly_detection: Whether to enable anomaly detection
            enable_predictive_analytics: Whether to enable predictive analytics
            results_dir: Directory for storing test results
            tags: Tags for worker selection
        """
        # Initialize base worker node
        super().__init__(
            coordinator_url=coordinator_url,
            worker_id=worker_id,
            task_types=["api_test"],
            tags=tags or []
        )
        
        # Initialize API-specific components
        self.enable_anomaly_detection = enable_anomaly_detection
        self.enable_predictive_analytics = enable_predictive_analytics
        self.results_dir = results_dir
        
        # Ensure results directory exists
        os.makedirs(results_dir, exist_ok=True)
        
        # Initialize capabilities
        self.capabilities = self._detect_capabilities()
        
        # Add capability tags
        self.add_tags(["api_worker"] + [f"api_{provider.lower()}" for provider in self.capabilities["providers"]])
        
        logger.info(f"API Worker Node initialized with ID: {self.worker_id}")
        logger.info(f"Supported API providers: {', '.join(self.capabilities['providers'])}")
    
    def _detect_capabilities(self) -> Dict[str, Any]:
        """
        Detect available API capabilities on this worker.
        
        Returns:
            Dictionary with capability information
        """
        capabilities = {
            "providers": [],
            "hardware": {},
            "system": {
                "platform": platform.system(),
                "platform_version": platform.version(),
                "processor": platform.processor(),
                "python_version": platform.python_version()
            }
        }
        
        # Check for API keys to determine available providers
        if os.environ.get("OPENAI_API_KEY"):
            capabilities["providers"].append("openai")
        
        if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("CLAUDE_API_KEY"):
            capabilities["providers"].append("claude")
        
        if os.environ.get("GROQ_API_KEY"):
            capabilities["providers"].append("groq")
        
        if os.environ.get("MISTRAL_API_KEY"):
            capabilities["providers"].append("mistral")
        
        if os.environ.get("COHERE_API_KEY"):
            capabilities["providers"].append("cohere")
        
        # Detect hardware capabilities
        try:
            import torch
            capabilities["hardware"]["cuda"] = torch.cuda.is_available()
            if capabilities["hardware"]["cuda"]:
                capabilities["hardware"]["cuda_devices"] = torch.cuda.device_count()
                capabilities["hardware"]["cuda_version"] = torch.version.cuda
        except ImportError:
            capabilities["hardware"]["cuda"] = False
        
        return capabilities
    
    def process_task(self, task: Task) -> TaskResult:
        """
        Process an API test task.
        
        Args:
            task: Task to process
            
        Returns:
            Task result
        """
        if task.task_type != "api_test":
            return TaskResult(
                task_id=task.task_id,
                status=TaskStatus.FAILED,
                result={"error": f"Unsupported task type: {task.task_type}"},
                metadata={}
            )
        
        # Extract test parameters
        api_type = task.parameters.get("api_type")
        test_type = task.parameters.get("test_type")
        test_parameters = task.parameters.get("test_parameters", {})
        
        if not api_type or not test_type:
            return TaskResult(
                task_id=task.task_id,
                status=TaskStatus.FAILED,
                result={"error": "Missing required parameters: api_type and test_type"},
                metadata={}
            )
        
        # Check if API type is supported
        if api_type.lower() not in [p.lower() for p in self.capabilities["providers"]]:
            return TaskResult(
                task_id=task.task_id,
                status=TaskStatus.FAILED,
                result={"error": f"Unsupported API type: {api_type}"},
                metadata={}
            )
        
        # Run the test
        try:
            # Create API backend
            backend = APIBackendFactory.create_backend(api_type)
            
            # Create tester
            tester = APITester(
                backend=backend,
                enable_anomaly_detection=self.enable_anomaly_detection,
                enable_predictive_analytics=self.enable_predictive_analytics,
                results_dir=self.results_dir
            )
            
            # Run test based on test type
            if test_type == "latency":
                result = tester.run_latency_test(**test_parameters)
            elif test_type == "throughput":
                result = tester.run_throughput_test(**test_parameters)
            elif test_type == "reliability":
                result = tester.run_reliability_test(**test_parameters)
            elif test_type == "cost_efficiency":
                result = tester.run_cost_efficiency_test(**test_parameters)
            elif test_type == "all":
                result = tester.run_all_tests(**test_parameters)
            else:
                return TaskResult(
                    task_id=task.task_id,
                    status=TaskStatus.FAILED,
                    result={"error": f"Unsupported test type: {test_type}"},
                    metadata={}
                )
            
            # Create task result
            return TaskResult(
                task_id=task.task_id,
                status=TaskStatus.COMPLETED,
                result=result,
                metadata={
                    "provider": api_type,
                    "model": backend.model,
                    "test_type": test_type,
                    "worker_id": self.worker_id,
                    "timestamp": datetime.datetime.now().isoformat()
                }
            )
        except Exception as e:
            logger.error(f"Error processing task {task.task_id}: {e}")
            import traceback
            traceback.print_exc()
            
            return TaskResult(
                task_id=task.task_id,
                status=TaskStatus.FAILED,
                result={"error": str(e)},
                metadata={
                    "provider": api_type,
                    "test_type": test_type,
                    "worker_id": self.worker_id,
                    "timestamp": datetime.datetime.now().isoformat()
                }
            )
    
    def run(self, poll_interval: int = 5) -> None:
        """
        Run the worker node.
        
        Args:
            poll_interval: Time in seconds between polling for new tasks
        """
        logger.info(f"Starting API Worker Node {self.worker_id}")
        
        # Register worker with coordinator
        self.register()
        
        try:
            while True:
                # Poll for tasks
                tasks = self.get_tasks()
                
                if tasks:
                    for task in tasks:
                        logger.info(f"Processing task {task.task_id}")
                        
                        # Process task
                        result = self.process_task(task)
                        
                        # Submit result
                        self.submit_result(result)
                        
                        logger.info(f"Completed task {task.task_id} with status {result.status}")
                
                # Sleep before next poll
                time.sleep(poll_interval)
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received. Shutting down worker node.")
        except Exception as e:
            logger.error(f"Error in worker node main loop: {e}")
        finally:
            # Unregister worker from coordinator
            self.unregister()
            logger.info(f"Worker node {self.worker_id} shut down")


def main():
    """Command-line interface for the API Worker Node."""
    parser = argparse.ArgumentParser(description='API Distributed Testing Worker Node')
    
    # Coordinator connection
    parser.add_argument('--coordinator', type=str, required=True, 
                       help='URL of the coordinator server')
    
    # Worker configuration
    parser.add_argument('--worker-id', type=str, 
                       help='Custom worker ID (if not provided, a unique ID will be generated)')
    parser.add_argument('--tags', type=str, 
                       help='Comma-separated list of tags for worker selection')
    
    # Features
    parser.add_argument('--disable-anomaly-detection', action='store_true', 
                       help='Disable anomaly detection')
    parser.add_argument('--disable-predictive-analytics', action='store_true', 
                       help='Disable predictive analytics')
    
    # Output
    parser.add_argument('--results-dir', type=str, default='./api_test_results', 
                       help='Directory for storing test results')
    
    # Running options
    parser.add_argument('--poll-interval', type=int, default=5, 
                       help='Time in seconds between polling for new tasks')
    
    args = parser.parse_args()
    
    # Parse tags
    tags = []
    if args.tags:
        tags = [tag.strip() for tag in args.tags.split(',')]
    
    # Create worker node
    worker = APIWorkerNode(
        coordinator_url=args.coordinator,
        worker_id=args.worker_id,
        enable_anomaly_detection=not args.disable_anomaly_detection,
        enable_predictive_analytics=not args.disable_predictive_analytics,
        results_dir=args.results_dir,
        tags=tags
    )
    
    # Run worker node
    worker.run(poll_interval=args.poll_interval)


if __name__ == "__main__":
    main()