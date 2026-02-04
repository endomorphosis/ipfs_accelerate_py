#!/usr/bin/env python3
"""
Test Refactoring Utilities

This module provides utilities to assist with the test refactoring process.
It helps with analyzing test patterns, identifying duplicates, and migrating
tests to the new structure.
"""

import os
import ast
import re
import json
import shutil
from typing import Dict, List, Set, Tuple, Any, Optional
from collections import defaultdict


class TestRefactoringHelper:
    """Helper class for test refactoring tasks."""
    
    def __init__(self, ast_report_path: Optional[str] = None, root_dir: str = '.'):
        """
        Initialize the refactoring helper.
        
        Args:
            ast_report_path: Path to the AST report JSON file (optional)
            root_dir: Root directory of the project
        """
        self.root_dir = os.path.abspath(root_dir)
        
        if ast_report_path:
            self.ast_report = self._load_ast_report(ast_report_path)
            self.duplicate_tests = self._find_duplicate_tests()
            self.test_clusters = self._cluster_related_tests()
        else:
            # For operations that don't require AST report
            self.ast_report = None
            self.duplicate_tests = {}
            self.test_clusters = {}
        
    def _load_ast_report(self, ast_report_path: str) -> Dict[str, Any]:
        """Load the AST report from JSON."""
        with open(ast_report_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _find_duplicate_tests(self) -> Dict[str, List[str]]:
        """
        Find duplicate test methods across different files.
        
        Returns:
            Dict mapping test method names to lists of files containing them
        """
        duplicates = defaultdict(list)
        
        for file_info in self.ast_report['files']:
            file_path = file_info['path']
            
            # Extract test methods from classes
            for class_info in file_info['classes']:
                for method in class_info['methods']:
                    if method.get('is_test', False):
                        method_name = method['name']
                        duplicates[method_name].append(file_path)
            
            # Extract standalone test functions
            for func in file_info['functions']:
                if func.get('is_test', False):
                    func_name = func['name']
                    duplicates[func_name].append(file_path)
        
        # Filter to only include methods that appear in multiple files
        return {name: files for name, files in duplicates.items() if len(files) > 1}
    
    def _cluster_related_tests(self) -> Dict[str, List[str]]:
        """
        Cluster related test files based on naming patterns.
        
        Returns:
            Dict mapping cluster types to lists of file paths
        """
        clusters = defaultdict(list)
        
        # Define common test categories
        categories = {
            'model': ['bert', 'vit', 'gpt', 'llama', 'model'],
            'hardware': ['hardware', 'webgpu', 'webnn', 'device', 'platform'],
            'browser': ['browser', 'firefox', 'chrome', 'safari', 'edge'],
            'api': ['api', 'endpoint', 'service', 'server', 'client'],
            'resource_pool': ['resource', 'pool', 'allocation', 'bridge'],
            'integration': ['integration', 'connect', 'e2e'],
        }
        
        for file_info in self.ast_report['files']:
            file_path = file_info['path']
            file_name = os.path.basename(file_path).lower()
            
            # Categorize file by keywords in the name
            categorized = False
            for category, keywords in categories.items():
                if any(keyword in file_name for keyword in keywords):
                    clusters[category].append(file_path)
                    categorized = True
                    break
            
            # If not categorized, put in "other"
            if not categorized:
                clusters['other'].append(file_path)
        
        return clusters
    
    def generate_migration_plan(self, output_path: str) -> None:
        """
        Generate a migration plan for tests.
        
        Args:
            output_path: Path to save the migration plan
        """
        migration_plan = {
            'duplicates': self.duplicate_tests,
            'clusters': self.test_clusters,
            'migration_targets': {}
        }
        
        # Map files to their new locations
        for category, files in self.test_clusters.items():
            for file_path in files:
                rel_path = os.path.relpath(file_path, self.root_dir)
                filename = os.path.basename(file_path)
                
                if category == 'model':
                    # Further categorize models by type
                    if any(x in filename.lower() for x in ['bert', 'gpt', 'llama', 't5', 'roberta']):
                        new_path = f"tests/models/text/{filename}"
                    elif any(x in filename.lower() for x in ['vit', 'clip', 'image', 'vision']):
                        new_path = f"tests/models/vision/{filename}"
                    elif any(x in filename.lower() for x in ['audio', 'wav', 'whisper', 'speech']):
                        new_path = f"tests/models/audio/{filename}"
                    else:
                        new_path = f"tests/models/other/{filename}"
                elif category == 'hardware':
                    if 'webgpu' in filename.lower():
                        new_path = f"tests/hardware/webgpu/{filename}"
                    elif 'webnn' in filename.lower():
                        new_path = f"tests/hardware/webnn/{filename}"
                    else:
                        new_path = f"tests/hardware/platform/{filename}"
                elif category == 'browser':
                    new_path = f"tests/browser/{filename}"
                elif category == 'api':
                    new_path = f"tests/api/{filename}"
                elif category == 'resource_pool':
                    new_path = f"tests/resource_pool/{filename}"
                elif category == 'integration':
                    new_path = f"tests/integration/{filename}"
                else:
                    new_path = f"tests/unit/{filename}"
                
                migration_plan['migration_targets'][rel_path] = new_path
        
        # Save migration plan
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(migration_plan, f, indent=2)
    
    def create_directory_structure(self, base_dir: str = 'refactored_test_suite') -> None:
        """
        Create the new directory structure for tests.
        
        Args:
            base_dir: Base directory for the new test structure
        """
        directories = [
            os.path.join(base_dir, 'unit'),
            os.path.join(base_dir, 'integration'),
            os.path.join(base_dir, 'models/text'),
            os.path.join(base_dir, 'models/vision'),
            os.path.join(base_dir, 'models/audio'),
            os.path.join(base_dir, 'models/other'),
            os.path.join(base_dir, 'hardware/webgpu'),
            os.path.join(base_dir, 'hardware/webnn'),
            os.path.join(base_dir, 'hardware/platform'),
            os.path.join(base_dir, 'browser'),
            os.path.join(base_dir, 'api'),
            os.path.join(base_dir, 'resource_pool'),
            os.path.join(base_dir, 'e2e'),
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            # Create an __init__.py file to make the directory a package
            with open(os.path.join(directory, '__init__.py'), 'w') as f:
                f.write("# Test package")
    
    def generate_base_classes(self, base_dir: str = 'refactored_test_suite') -> None:
        """
        Generate base test classes.
        
        Args:
            base_dir: Base directory for the new test structure
        """
        # Create conftest.py for pytest fixtures
        conftest_content = """
import pytest
import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

@pytest.fixture(scope="session")
def test_dir():
    \"\"\"Return the directory containing test data.\"\"\"
    return os.path.join(os.path.dirname(__file__), "test_data")

@pytest.fixture(scope="session")
def project_root():
    \"\"\"Return the project root directory.\"\"\"
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
"""
        with open(os.path.join(base_dir, 'conftest.py'), 'w') as f:
            f.write(conftest_content)
        
        # Create base_test.py
        base_test_content = """
import unittest
import pytest
import os
import logging
import json

class BaseTest(unittest.TestCase):
    \"\"\"Base class for all tests.\"\"\"
    
    @classmethod
    def setUpClass(cls):
        \"\"\"Set up resources for the entire test class.\"\"\"
        cls.logger = logging.getLogger(cls.__name__)
        cls.logger.setLevel(logging.INFO)
        
        # Ensure we have a handler to prevent "no handlers" warnings
        if not cls.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            cls.logger.addHandler(handler)
    
    def setUp(self):
        \"\"\"Set up resources for each test method.\"\"\"
        self.logger.info(f"Setting up test: {self._testMethodName}")
    
    def tearDown(self):
        \"\"\"Clean up resources after each test method.\"\"\"
        self.logger.info(f"Tearing down test: {self._testMethodName}")
    
    @classmethod
    def tearDownClass(cls):
        \"\"\"Clean up resources after the entire test class.\"\"\"
        pass
    
    def load_json_data(self, filepath):
        \"\"\"Load JSON data from a file.\"\"\"
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def assertDictContainsSubset(self, subset, dictionary, msg=None):
        \"\"\"Assert that dictionary contains subset.\"\"\"
        for key, value in subset.items():
            if key not in dictionary:
                self.fail(f"{msg or 'Missing key'}: {key}")
            if value != dictionary[key]:
                self.fail(f"{msg or 'Value mismatch for key'}: {key}")
"""
        with open(os.path.join(base_dir, 'base_test.py'), 'w') as f:
            f.write(base_test_content)
        
        # Create model_test.py
        model_test_content = """
from test.tests.other.base_test import BaseTest
import os
import tempfile

class ModelTest(BaseTest):
    \"\"\"Base class for model tests.\"\"\"
    
    def setUp(self):
        super().setUp()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.model_dir = self.temp_dir.name
        
    def tearDown(self):
        self.temp_dir.cleanup()
        super().tearDown()
    
    def load_model(self, model_name):
        \"\"\"Load a model for testing.\"\"\"
        raise NotImplementedError("Subclasses must implement load_model")
    
    def verify_model_output(self, model, input_data, expected_output):
        \"\"\"Verify that model produces expected output.\"\"\"
        output = model(input_data)
        self.assertEqual(expected_output, output)
"""
        with open(os.path.join(base_dir, 'model_test.py'), 'w') as f:
            f.write(model_test_content)
        
        # Create hardware_test.py
        hardware_test_content = """
from test.tests.other.base_test import BaseTest

class HardwareTest(BaseTest):
    \"\"\"Base class for hardware tests.\"\"\"
    
    def setUp(self):
        super().setUp()
        self.detect_hardware()
        
    def detect_hardware(self):
        \"\"\"Detect available hardware.\"\"\"
        self.has_webgpu = self._check_webgpu()
        self.has_webnn = self._check_webnn()
        
    def _check_webgpu(self):
        \"\"\"Check if WebGPU is available.\"\"\"
        # Placeholder for hardware detection
        return False
        
    def _check_webnn(self):
        \"\"\"Check if WebNN is available.\"\"\"
        # Placeholder for hardware detection
        return False
    
    def skip_if_no_webgpu(self):
        \"\"\"Skip test if WebGPU is not available.\"\"\"
        if not self.has_webgpu:
            self.skipTest("WebGPU not available")
    
    def skip_if_no_webnn(self):
        \"\"\"Skip test if WebNN is not available.\"\"\"
        if not self.has_webnn:
            self.skipTest("WebNN not available")
"""
        with open(os.path.join(base_dir, 'hardware_test.py'), 'w') as f:
            f.write(hardware_test_content)
        
        # Create browser_test.py
        browser_test_content = """
from test.tests.other.base_test import BaseTest
import os

class BrowserTest(BaseTest):
    \"\"\"Base class for browser tests.\"\"\"
    
    def setUp(self):
        super().setUp()
        self.browser_type = os.environ.get("BROWSER_TYPE", "chrome")
        
    def get_browser_driver(self):
        \"\"\"Get browser driver for testing.\"\"\"
        raise NotImplementedError("Subclasses must implement get_browser_driver")
"""
        with open(os.path.join(base_dir, 'browser_test.py'), 'w') as f:
            f.write(browser_test_content)
        
        # Create api_test.py
        api_test_content = """
from test.tests.other.base_test import BaseTest
import requests
import os

class APITest(BaseTest):
    \"\"\"Base class for API tests.\"\"\"
    
    def setUp(self):
        super().setUp()
        self.base_url = os.environ.get("API_BASE_URL", "http://localhost:8000")
        self.session = requests.Session()
        
    def tearDown(self):
        self.session.close()
        super().tearDown()
    
    def get_endpoint_url(self, endpoint):
        \"\"\"Get full URL for an endpoint.\"\"\"
        return f"{self.base_url}/{endpoint.lstrip('/')}"
    
    def assertStatusCode(self, response, expected_code):
        \"\"\"Assert that response has expected status code.\"\"\"
        self.assertEqual(expected_code, response.status_code, 
                        f"Expected status code {expected_code}, got {response.status_code}")
"""
        with open(os.path.join(base_dir, 'api_test.py'), 'w') as f:
            f.write(api_test_content)
        
        # Create test_utils.py
        test_utils_content = """
import os
import json
import tempfile
import random
import string

def get_test_data_path(filename):
    \"\"\"Get path to a test data file.\"\"\"
    test_data_dir = os.path.join(os.path.dirname(__file__), "test_data")
    return os.path.join(test_data_dir, filename)

def create_temp_file(content, suffix=".txt"):
    \"\"\"Create a temporary file with the given content.\"\"\"
    fd, path = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(fd, 'w') as f:
            f.write(content)
        return path
    except:
        os.unlink(path)
        raise

def random_string(length=10):
    \"\"\"Generate a random string of the given length.\"\"\"
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

def load_json_data(filepath):
    \"\"\"Load JSON data from a file.\"\"\"
    with open(filepath, 'r') as f:
        return json.load(f)

def save_json_data(data, filepath):
    \"\"\"Save JSON data to a file.\"\"\"
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
"""
        with open(os.path.join(base_dir, 'test_utils.py'), 'w') as f:
            f.write(test_utils_content)


def generate_migration_report(ast_report_path: str, output_path: str) -> None:
    """
    Generate a migration report from AST analysis.
    
    Args:
        ast_report_path: Path to the AST report JSON file
        output_path: Path to save the migration report
    """
    # Load the AST report
    with open(ast_report_path, 'r') as f:
        ast_report = json.load(f)
    
    # Analyze test methods
    total_files = len(ast_report['files'])
    total_classes = sum(f['metrics']['num_classes'] for f in ast_report['files'])
    total_methods = sum(f['metrics']['num_methods'] for f in ast_report['files'])
    total_test_methods = sum(f['metrics']['num_test_methods'] for f in ast_report['files'])
    
    # Find duplicate test methods
    test_methods = defaultdict(list)
    for file_info in ast_report['files']:
        for class_info in file_info['classes']:
            for method in class_info['methods']:
                if method.get('is_test', False):
                    test_methods[method['name']].append(file_info['path'])
    
    duplicate_methods = {name: files for name, files in test_methods.items() if len(files) > 1}
    
    # Calculate metrics
    duplicated_method_count = len(duplicate_methods)
    duplicated_file_count = len(set(file for files in duplicate_methods.values() for file in files))
    
    # Generate report
    report = {
        'total_files': total_files,
        'total_classes': total_classes,
        'total_methods': total_methods,
        'total_test_methods': total_test_methods,
        'duplicated_method_count': duplicated_method_count,
        'duplicated_file_count': duplicated_file_count,
        'duplication_percentage': round(duplicated_method_count / total_test_methods * 100, 2),
        'top_duplicated_methods': sorted(
            [(name, len(files)) for name, files in duplicate_methods.items()],
            key=lambda x: x[1], reverse=True
        )[:20]
    }
    
    # Save report
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Refactoring Utilities')
    subparsers = parser.add_subparsers(dest='command')
    
    # Migration plan command
    plan_parser = subparsers.add_parser('plan', help='Generate migration plan')
    plan_parser.add_argument('--ast-report', required=True, help='Path to AST report JSON')
    plan_parser.add_argument('--output', required=True, help='Output path for migration plan')
    plan_parser.add_argument('--root-dir', default='.', help='Root directory of project')
    
    # Create structure command
    structure_parser = subparsers.add_parser('structure', help='Create directory structure')
    structure_parser.add_argument('--base-dir', default='refactored_test_suite', help='Base directory for test structure')
    
    # Generate base classes command
    base_parser = subparsers.add_parser('base-classes', help='Generate base test classes')
    base_parser.add_argument('--base-dir', default='refactored_test_suite', help='Base directory for test structure')
    
    # Migration report command
    report_parser = subparsers.add_parser('report', help='Generate migration report')
    report_parser.add_argument('--ast-report', required=True, help='Path to AST report JSON')
    report_parser.add_argument('--output', required=True, help='Output path for migration report')
    
    args = parser.parse_args()
    
    if args.command == 'plan':
        helper = TestRefactoringHelper(args.ast_report, args.root_dir)
        helper.generate_migration_plan(args.output)
        print(f"Migration plan saved to {args.output}")
    
    elif args.command == 'structure':
        helper = TestRefactoringHelper(None)
        helper.create_directory_structure(args.base_dir)
        print(f"Directory structure created in {args.base_dir}")
    
    elif args.command == 'base-classes':
        helper = TestRefactoringHelper(None)
        helper.generate_base_classes(args.base_dir)
        print(f"Base classes generated in {args.base_dir}")
    
    elif args.command == 'report':
        generate_migration_report(args.ast_report, args.output)
        print(f"Migration report saved to {args.output}")
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()