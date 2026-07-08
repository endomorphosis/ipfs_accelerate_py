"""
Base template for generating test files.

This module provides the foundation for all test templates in the IPFS Accelerate test framework.
"""

import os
import time
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union


class BaseTemplate:
    """
    Base class for all test templates.
    
    This class provides the core functionality for generating test files
    based on templates.
    """
    
    def __init__(self, name: str, **kwargs):
        """
        Initialize the template.
        
        Args:
            name: Name of the test (used in filename)
            **kwargs: Additional template parameters
        """
        self.name = name
        self.output_dir = kwargs.get('output_dir', None)
        self.timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.author = kwargs.get('author', os.environ.get('USER', 'unknown'))
        self.overwrite = kwargs.get('overwrite', False)
        
    def generate_header(self) -> str:
        """
        Generate the file header.
        
        Returns:
            Header content as a string
        """
        return f"""#!/usr/bin/env python3
\"\"\"
Test file for {self.name}.

This file is auto-generated using the template-based test generator.
Generated: {self.timestamp}
\"\"\"

import os
import sys
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
"""
    
    def generate_imports(self) -> str:
        """
        Generate the import statements.
        
        Returns:
            Import statements as a string
        """
        return """
import pytest
"""
    
    def generate_test_class(self) -> str:
        """
        Generate the test class.
        
        Returns:
            Test class content as a string
        """
        class_name = ''.join(word.capitalize() for word in self.name.replace('-', '_').split('_'))
        return f"""
class Test{class_name}:
    \"\"\"Test class for {self.name}.\"\"\"
    
    def setup_method(self):
        \"\"\"Set up test environment.\"\"\"
        logger.info(f"Setting up test for {self.name}")
    
    def test_base(self):
        \"\"\"Basic test for {self.name}.\"\"\"
        logger.info(f"Running test for {self.name}")
        assert True
    
    def teardown_method(self):
        \"\"\"Clean up after test.\"\"\"
        logger.info(f"Cleaning up after test for {self.name}")
"""
    
    def generate_main_section(self) -> str:
        """
        Generate the main section of the file.
        
        Returns:
            Main section content as a string
        """
        return """

if __name__ == "__main__":
    # Run tests directly
    pytest.main(["-xvs", __file__])
"""
    
    def generate_content(self) -> str:
        """
        Generate the complete file content.
        
        Returns:
            Complete file content as a string
        """
        sections = [
            self.generate_header(),
            self.generate_imports(),
            self.generate_test_class(),
            self.generate_main_section()
        ]
        
        content = '\n'.join(sections)
        
        # Allow for customization of content
        content = self.customize_content(content)
        
        return content
    
    def customize_content(self, content: str) -> str:
        """
        Customize the generated content.
        
        This method can be overridden by subclasses to make specific modifications
        to the generated content.
        
        Args:
            content: The generated content
            
        Returns:
            The customized content
        """
        return content
    
    def get_output_path(self) -> str:
        """
        Get the output path for the generated file.
        
        Returns:
            Output file path
        """
        if self.output_dir:
            # Ensure output directory exists
            os.makedirs(self.output_dir, exist_ok=True)
            
            # Determine filename
            filename = f"test_{self.name.replace('-', '_')}.py"
            
            return os.path.join(self.output_dir, filename)
        else:
            # Default to current directory
            filename = f"test_{self.name.replace('-', '_')}.py"
            return filename
    
    def write_to_file(self, content: str) -> str:
        """
        Write the generated content to a file.
        
        Args:
            content: The content to write
            
        Returns:
            The path to the generated file
        """
        output_path = self.get_output_path()
        
        # Check if file exists and overwrite is not enabled
        if os.path.exists(output_path) and not self.overwrite:
            raise FileExistsError(f"File {output_path} already exists. Use overwrite=True to overwrite.")
        
        # Write content to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
            
        return output_path
    
    def generate(self) -> str:
        """
        Generate the test file.
        
        This method generates the content and writes it to a file.
        
        Returns:
            The path to the generated file
        """
        self.before_generate()
        
        content = self.generate_content()
        output_path = self.write_to_file(content)
        
        self.after_generate()
        
        return output_path
    
    def before_generate(self) -> None:
        """
        Hook called before generating the file.
        
        This method can be overridden by subclasses to perform
        setup tasks before generation.
        """
        pass
    
    def after_generate(self) -> None:
        """
        Hook called after generating the file.
        
        This method can be overridden by subclasses to perform
        cleanup or post-processing tasks after generation.
        """
        pass