"""
Benchmark Base Module

This module provides the base class for all benchmark implementations.
"""

import logging
import time
from typing import Dict, Any, Optional, List, Tuple

class BenchmarkBase:
    """
    Base class for all benchmark implementations.
    
    This class defines the common interface and behavior for all benchmark implementations.
    Subclasses should implement the setup(), execute(), and process_results() methods.
    
    Attributes:
        hardware: The hardware backend to use for the benchmark
        config: Configuration parameters for the benchmark
        logger: Logger instance for this benchmark
        results: Results from the last execution
    """
    
    def __init__(self, hardware: Any, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the benchmark.
        
        Args:
            hardware: Hardware backend to use
            config: Optional configuration parameters
        """
        self.hardware = hardware
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        self.results = None
        
    def setup(self) -> None:
        """
        Set up the benchmark environment.
        
        This method should prepare any resources needed for the benchmark,
        such as loading models, preparing data, etc.
        
        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("Subclasses must implement setup()")
        
    def run(self) -> Dict[str, Any]:
        """
        Run the complete benchmark.
        
        This method handles the overall benchmark flow, including setup,
        execution, result processing, and cleanup.
        
        Returns:
            Dictionary containing benchmark results
        """
        self.setup()
        
        # Record benchmark start time
        start_time = time.time()
        
        try:
            # Execute the benchmark
            raw_results = self.execute()
            
            # Process the results
            self.results = self.process_results(raw_results)
            
            # Add execution time
            self.results['execution_time'] = time.time() - start_time
            
            return self.results
            
        except Exception as e:
            self.logger.error(f"Benchmark execution failed: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'error_type': type(e).__name__,
                'execution_time': time.time() - start_time
            }
            
        finally:
            self.cleanup()
            
    def execute(self) -> Any:
        """
        Execute the actual benchmark code.
        
        This is the core benchmark implementation that performs the operations
        being measured.
        
        Returns:
            Raw benchmark results (format depends on implementation)
        
        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("Subclasses must implement execute()")
        
    def process_results(self, raw_results: Any) -> Dict[str, Any]:
        """
        Process and validate the raw benchmark results.
        
        This method should transform the raw results from execute() into
        a standardized format.
        
        Args:
            raw_results: Raw results from execute()
            
        Returns:
            Dictionary containing processed benchmark results
            
        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("Subclasses must implement process_results()")
        
    def cleanup(self) -> None:
        """
        Clean up resources after benchmark execution.
        
        This method should release any resources acquired during setup()
        or execute().
        """
        # Base implementation does nothing
        pass
        
    def validate_config(self) -> Tuple[bool, List[str]]:
        """
        Validate the benchmark configuration.
        
        Returns:
            Tuple containing (is_valid, list_of_errors)
        """
        # Base implementation assumes config is valid
        return True, []
        
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get benchmark metadata.
        
        Returns:
            Dictionary containing benchmark metadata
        """
        return {
            'class': self.__class__.__name__,
            'hardware': str(self.hardware),
            'config': self.config
        }