#!/usr/bin/env python3
"""
Unit tests for the API Backend Converter

This module provides comprehensive test coverage for the convert_api_backends.py script
which converts Python API backends to TypeScript.
"""

import os
import sys
import unittest
import tempfile
import shutil
import ast
from unittest.mock import patch, mock_open, MagicMock
from typing import Dict, List, Any

# Import the converter module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from convert_api_backends import APIBackendConverter, TYPE_MAPPINGS, METHOD_MAPPINGS


class TestAPIBackendConverter(unittest.TestCase):
    """Test cases for the APIBackendConverter class"""

    def setUp(self):
        """Set up test environment"""
        # Create temporary directories for testing
        self.temp_dir = tempfile.mkdtemp()
        self.python_dir = os.path.join(self.temp_dir, "python")
        self.ts_dir = os.path.join(self.temp_dir, "typescript")
        os.makedirs(self.python_dir, exist_ok=True)
        os.makedirs(self.ts_dir, exist_ok=True)
        
        # Create a sample Python file for testing
        self.sample_python_file = os.path.join(self.python_dir, "sample_backend.py")
        sample_code = '''
import os
import requests
import logging
from typing import Dict, List, Any, Optional

class sample_backend:
    """Sample API backend implementation for testing"""
    
    def __init__(self, resources: Dict[str, Any] = None, metadata: Dict[str, Any] = None):
        """Initialize the API backend"""
        self.resources = resources or {}
        self.metadata = metadata or {}
        self.api_endpoint = "https://api.sample.com/v1/chat"
        self.default_model = "sample-model"
        
    def get_api_key(self, metadata: Dict[str, Any]) -> str:
        """Get API key from metadata or environment"""
        return metadata.get("sample_api_key") or os.environ.get("SAMPLE_API_KEY", "")
    
    def get_default_model(self) -> str:
        """Get the default model name"""
        return self.default_model
    
    def is_compatible_model(self, model: str) -> bool:
        """Check if the model is compatible with this backend"""
        return model.startswith("sample")
'''
        with open(self.sample_python_file, "w") as f:
            f.write(sample_code)
        
        # Initialize converter
        self.converter = APIBackendConverter(
            python_file=self.sample_python_file,
            output_dir=self.ts_dir,
            dry_run=True
        )

    def tearDown(self):
        """Clean up temporary files and directories"""
        shutil.rmtree(self.temp_dir)

    def test_initialization(self):
        """Test converter initialization"""
        self.assertEqual(self.converter.python_file, self.sample_python_file)
        self.assertEqual(self.converter.output_dir, self.ts_dir)
        self.assertEqual(self.converter.backend_name, "sample_backend")
        self.assertEqual(self.converter.backend_class_name, "SampleBackend")
        self.assertEqual(self.converter.ts_dir, os.path.join(self.ts_dir, "sample_backend"))
        self.assertEqual(self.converter.main_file, os.path.join(self.ts_dir, "sample_backend", "sample_backend.ts"))
        self.assertEqual(self.converter.index_file, os.path.join(self.ts_dir, "sample_backend", "index.ts"))
        self.assertEqual(self.converter.types_file, os.path.join(self.ts_dir, "sample_backend", "types.ts"))

    def test_to_camel_case(self):
        """Test snake_case to CamelCase conversion"""
        self.assertEqual(self.converter._to_camel_case("sample_backend"), "SampleBackend")
        self.assertEqual(self.converter._to_camel_case("test_api"), "TestApi")
        
        # Test known backends special cases
        converter = APIBackendConverter("ollama.py", self.ts_dir, dry_run=True)
        self.assertEqual(converter.backend_class_name, "Ollama")
        
        converter = APIBackendConverter("hf_tei.py", self.ts_dir, dry_run=True)
        self.assertEqual(converter.backend_class_name, "HfTei")

    def test_to_camel_case_lower(self):
        """Test snake_case to camelCase (first letter lowercase) conversion"""
        self.assertEqual(self.converter._to_camel_case_lower("get_api_key"), "getApiKey")
        self.assertEqual(self.converter._to_camel_case_lower("is_compatible_model"), "isCompatibleModel")

    def test_parse_python_file(self):
        """Test parsing a Python file"""
        success = self.converter.parse_python_file()
        self.assertTrue(success)
        self.assertIsNotNone(self.converter.ast)
        self.assertIsNotNone(self.converter.class_node)
        self.assertEqual(self.converter.class_node.name, "sample_backend")

    def test_extract_method_info(self):
        """Test extraction of method information"""
        self.converter.parse_python_file()
        
        # Check if methods were extracted correctly
        method_names = [m["original_name"] for m in self.converter.methods]
        self.assertIn("__init__", method_names)
        self.assertIn("get_api_key", method_names)
        self.assertIn("get_default_model", method_names)
        self.assertIn("is_compatible_model", method_names)
        
        # Check if constructor params were extracted
        self.assertEqual(len(self.converter.constructor_params), 2)
        self.assertEqual(self.converter.constructor_params[0]["name"], "resources")
        self.assertEqual(self.converter.constructor_params[1]["name"], "metadata")

    def test_convert_complex_type(self):
        """Test conversion of complex Python types to TypeScript"""
        # Create a simple List[str] type annotation
        list_str_node = ast.parse("x: List[str]").body[0].annotation
        ts_type = self.converter._convert_complex_type(list_str_node)
        self.assertEqual(ts_type, "Array<string>")
        
        # Create a Dict[str, Any] type annotation
        dict_node = ast.parse("x: Dict[str, Any]").body[0].annotation
        ts_type = self.converter._convert_complex_type(dict_node)
        self.assertEqual(ts_type, "Record<string, any>")
        
        # Create an Optional[str] type annotation
        optional_node = ast.parse("x: Optional[str]").body[0].annotation
        ts_type = self.converter._convert_complex_type(optional_node)
        self.assertEqual(ts_type, "string | null | undefined")

    def test_get_default_value(self):
        """Test extraction of default values"""
        # Test None default
        none_node = ast.Constant(value=None)
        self.assertEqual(self.converter._get_default_value(none_node), "null")
        
        # Test string default
        str_node = ast.Constant(value="test")
        self.assertEqual(self.converter._get_default_value(str_node), '"test"')
        
        # Test number default
        num_node = ast.Constant(value=42)
        self.assertEqual(self.converter._get_default_value(num_node), "42")
        
        # Test boolean default
        bool_node = ast.Constant(value=True)
        self.assertEqual(self.converter._get_default_value(bool_node), "true")
        
        # Test empty dict default
        dict_node = ast.Dict(keys=[], values=[])
        self.assertEqual(self.converter._get_default_value(dict_node), "{}")
        
        # Test empty list default
        list_node = ast.List(elts=[])
        self.assertEqual(self.converter._get_default_value(list_node), "[]")

    def test_clean_source(self):
        """Test cleaning of Python source code"""
        source = """
import os
import requests)))
    from typing import Dict, List
class TestClass:
    def method(self):
        pass
"""
        cleaned = self.converter._clean_source(source)
        self.assertNotIn("requests)))", cleaned)
        self.assertIn("requests)", cleaned)
        self.assertIn("from typing import Dict, List", cleaned)

    def test_backend_specific_customizations(self):
        """Test backend-specific customizations"""
        # Test Claude backend
        claude_converter = APIBackendConverter("claude.py", self.ts_dir, dry_run=True)
        claude_converter._analyze_api_specific_types()
        
        # Check if Claude-specific types were generated
        claude_types = [t["name"] for t in claude_converter.api_specific_types]
        self.assertIn("ClaudeResponse", claude_types)
        self.assertIn("ClaudeRequest", claude_types)
        
        # Find Claude request type
        claude_request = next((t for t in claude_converter.api_specific_types 
                             if t["name"] == "ClaudeRequest"), None)
        
        # Check if it has Claude-specific fields
        if claude_request:
            field_names = [f["name"] for f in claude_request["fields"]]
            self.assertIn("system", field_names)
            self.assertIn("stop_sequences", field_names)
            self.assertIn("anthropic_version", field_names)

    def test_generate_typescript_files(self):
        """Test generation of TypeScript files"""
        self.converter.parse_python_file()
        result = self.converter.generate_typescript_files()
        self.assertTrue(result)

    @patch("os.makedirs")
    @patch("builtins.open", new_callable=mock_open)
    def test_file_writing(self, mock_file, mock_makedirs):
        """Test file writing operations"""
        self.converter.dry_run = False
        self.converter.parse_python_file()
        self.converter.generate_typescript_files()
        
        # Check if directories were created
        mock_makedirs.assert_called_with(self.converter.ts_dir, exist_ok=True)
        
        # Check if files were opened for writing
        open_calls = [call.args[0] for call in mock_file.call_args_list]
        self.assertIn(self.converter.main_file, open_calls)
        self.assertIn(self.converter.index_file, open_calls)
        self.assertIn(self.converter.types_file, open_calls)


class TestTypeMapping(unittest.TestCase):
    """Test cases for type mapping functionality"""

    def test_basic_type_mappings(self):
        """Test basic Python to TypeScript type mappings"""
        self.assertEqual(TYPE_MAPPINGS["str"], "string")
        self.assertEqual(TYPE_MAPPINGS["int"], "number")
        self.assertEqual(TYPE_MAPPINGS["float"], "number")
        self.assertEqual(TYPE_MAPPINGS["bool"], "boolean")
        self.assertEqual(TYPE_MAPPINGS["dict"], "Record<string, any>")
        self.assertEqual(TYPE_MAPPINGS["list"], "Array<any>")
        self.assertEqual(TYPE_MAPPINGS["None"], "null")

    def test_method_mappings(self):
        """Test Python method to TypeScript method name mappings"""
        self.assertEqual(METHOD_MAPPINGS["create_endpoint_handler"], "createEndpointHandler")
        self.assertEqual(METHOD_MAPPINGS["get_api_key"], "getApiKey")
        self.assertEqual(METHOD_MAPPINGS["get_default_model"], "getDefaultModel")
        self.assertEqual(METHOD_MAPPINGS["is_compatible_model"], "isCompatibleModel")


class TestMainFunctionality(unittest.TestCase):
    """Test cases for the main functionality"""

    @patch("sys.argv", ["convert_api_backends.py", "--backend", "sample_backend", "--dry-run"])
    @patch("convert_api_backends.APIBackendConverter")
    def test_main_single_backend(self, mock_converter):
        """Test running the main function with a single backend"""
        from convert_api_backends import main
        
        # Mock the converter and its methods
        mock_instance = MagicMock()
        mock_instance.parse_python_file.return_value = True
        mock_instance.generate_typescript_files.return_value = True
        mock_converter.return_value = mock_instance
        
        # Run main function
        with patch("os.path.exists", return_value=True):
            main()
        
        # Check if converter was created with correct arguments
        mock_converter.assert_called_once()
        mock_instance.parse_python_file.assert_called_once()
        mock_instance.generate_typescript_files.assert_called_once()

    @patch("sys.argv", ["convert_api_backends.py", "--all", "--dry-run"])
    @patch("os.listdir")
    @patch("convert_api_backends.APIBackendConverter")
    def test_main_all_backends(self, mock_converter, mock_listdir):
        """Test running the main function for all backends"""
        from convert_api_backends import main
        
        # Mock listing directory
        mock_listdir.return_value = ["sample_backend.py", "another_backend.py", "__init__.py"]
        
        # Mock the converter and its methods
        mock_instance = MagicMock()
        mock_instance.parse_python_file.return_value = True
        mock_instance.generate_typescript_files.return_value = True
        mock_converter.return_value = mock_instance
        
        # Run main function
        with patch("os.path.exists", return_value=True):
            main()
        
        # Should be called twice (once for each backend, skipping __init__.py)
        self.assertEqual(mock_converter.call_count, 2)
        self.assertEqual(mock_instance.parse_python_file.call_count, 2)
        self.assertEqual(mock_instance.generate_typescript_files.call_count, 2)

    @patch("sys.argv", ["convert_api_backends.py", "--backend", "sample_backend", "--fix-source", "--dry-run"])
    @patch("convert_api_backends.APIBackendConverter")
    def test_main_with_fix_source(self, mock_converter):
        """Test running the main function with fix-source option"""
        from convert_api_backends import main
        
        # Mock the converter and its methods
        mock_instance = MagicMock()
        mock_instance.parse_python_file.return_value = True
        mock_instance.generate_typescript_files.return_value = True
        mock_converter.return_value = mock_instance
        
        # Mock file operations
        mock_open_obj = mock_open(read_data="import os\nimport requests)))")
        
        # Run main function
        with patch("os.path.exists", return_value=True), \
             patch("builtins.open", mock_open_obj):
            main()
        
        # Check if converter was created with correct arguments
        mock_converter.assert_called_once()


if __name__ == "__main__":
    unittest.main()