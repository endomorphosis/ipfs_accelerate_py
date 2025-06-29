"""
Tests for the model converter base classes.
"""

import os
import tempfile
import unittest
from unittest import mock

from refactored_model_conversion_generator.core.converter import ModelConverter, ConversionResult
from refactored_model_conversion_generator.core.registry import ModelConverterRegistry, register_converter

# Sample model converter for testing
class SampleModelConverter(ModelConverter):
    """Sample converter for testing."""
    
    def _get_source_format(self) -> str:
        return 'sample'
        
    def _get_target_format(self) -> str:
        return 'target'
        
    def _get_supported_model_types(self) -> list:
        return ['test', 'bert']
        
    def _execute_conversion(self, model_path, output_path, model_type=None, **kwargs):
        # Simple mock implementation that writes a test file
        with open(output_path, 'w') as f:
            f.write(f"Converted {model_path} to {output_path} (model_type={model_type})")
            
        return ConversionResult(
            success=True,
            output_path=output_path,
            format=self.target_format,
            metadata={'model_type': model_type}
        )


class TestModelConverter(unittest.TestCase):
    """Tests for ModelConverter base class."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.converter = SampleModelConverter()
        
        # Create a sample model file
        self.model_path = os.path.join(self.temp_dir.name, 'sample_model.sample')
        with open(self.model_path, 'w') as f:
            f.write("Sample model data")
            
    def tearDown(self):
        """Clean up test environment."""
        self.temp_dir.cleanup()
        
    def test_convert_basic(self):
        """Test basic conversion."""
        output_path = os.path.join(self.temp_dir.name, 'converted_model.target')
        
        result = self.converter.convert(
            model_path=self.model_path,
            output_path=output_path,
            model_type='test'
        )
        
        self.assertTrue(result.success)
        self.assertEqual(result.output_path, output_path)
        self.assertEqual(result.format, 'target')
        self.assertEqual(result.metadata, {'model_type': 'test'})
        self.assertGreater(result.conversion_time, 0)
        
        # Check that the file was created
        self.assertTrue(os.path.exists(output_path))
        
    def test_convert_default_output_path(self):
        """Test conversion with default output path."""
        result = self.converter.convert(
            model_path=self.model_path,
            model_type='test'
        )
        
        self.assertTrue(result.success)
        
        # Default output path should be in the same directory with target format extension
        expected_path = os.path.join(
            self.temp_dir.name, 
            'sample_model_target.target'
        )
        self.assertEqual(result.output_path, expected_path)
        
    def test_convert_missing_model(self):
        """Test conversion with missing model."""
        non_existent_path = os.path.join(self.temp_dir.name, 'non_existent.sample')
        
        result = self.converter.convert(
            model_path=non_existent_path,
            model_type='test'
        )
        
        self.assertFalse(result.success)
        self.assertIn("not found", result.error)
        

class TestModelConverterRegistry(unittest.TestCase):
    """Tests for ModelConverterRegistry."""
    
    def setUp(self):
        """Set up test environment."""
        # Clear registry before each test
        ModelConverterRegistry._registry = {}
        
    def test_register_decorator(self):
        """Test register decorator."""
        @register_converter(source_format='test', target_format='output')
        class TestConverter(ModelConverter):
            def _get_source_format(self) -> str:
                return 'test'
                
            def _get_target_format(self) -> str:
                return 'output'
                
            def _get_supported_model_types(self) -> list:
                return ['test']
                
            def _execute_conversion(self, *args, **kwargs):
                return ConversionResult(success=True)
                
        # Check that converter was registered
        converter_class = ModelConverterRegistry.get_converter('test', 'output')
        self.assertEqual(converter_class, TestConverter)
        
    def test_register_with_model_type(self):
        """Test register with model type."""
        @register_converter(source_format='test', target_format='output', model_type='bert')
        class BertConverter(ModelConverter):
            def _get_source_format(self) -> str:
                return 'test'
                
            def _get_target_format(self) -> str:
                return 'output'
                
            def _get_supported_model_types(self) -> list:
                return ['bert']
                
            def _execute_conversion(self, *args, **kwargs):
                return ConversionResult(success=True)
                
        @register_converter(source_format='test', target_format='output', model_type='vit')
        class VitConverter(ModelConverter):
            def _get_source_format(self) -> str:
                return 'test'
                
            def _get_target_format(self) -> str:
                return 'output'
                
            def _get_supported_model_types(self) -> list:
                return ['vit']
                
            def _execute_conversion(self, *args, **kwargs):
                return ConversionResult(success=True)
                
        # Check that model-specific converters were registered
        bert_converter = ModelConverterRegistry.get_converter('test', 'output', 'bert')
        vit_converter = ModelConverterRegistry.get_converter('test', 'output', 'vit')
        
        self.assertEqual(bert_converter, BertConverter)
        self.assertEqual(vit_converter, VitConverter)
        
    def test_get_converter_fallback(self):
        """Test getting converter with fallback to generic."""
        @register_converter(source_format='test', target_format='output')
        class GenericConverter(ModelConverter):
            def _get_source_format(self) -> str:
                return 'test'
                
            def _get_target_format(self) -> str:
                return 'output'
                
            def _get_supported_model_types(self) -> list:
                return ['test']
                
            def _execute_conversion(self, *args, **kwargs):
                return ConversionResult(success=True)
                
        # Check that generic converter is used as fallback
        converter_class = ModelConverterRegistry.get_converter('test', 'output', 'unknown')
        self.assertEqual(converter_class, GenericConverter)
        
    def test_list_converters(self):
        """Test listing registered converters."""
        @register_converter(source_format='test1', target_format='output1')
        class Converter1(ModelConverter):
            def _get_source_format(self) -> str:
                return 'test1'
                
            def _get_target_format(self) -> str:
                return 'output1'
                
            def _get_supported_model_types(self) -> list:
                return ['test']
                
            def _execute_conversion(self, *args, **kwargs):
                return ConversionResult(success=True)
                
        @register_converter(source_format='test2', target_format='output2', model_type='bert')
        class Converter2(ModelConverter):
            def _get_source_format(self) -> str:
                return 'test2'
                
            def _get_target_format(self) -> str:
                return 'output2'
                
            def _get_supported_model_types(self) -> list:
                return ['bert']
                
            def _execute_conversion(self, *args, **kwargs):
                return ConversionResult(success=True)
                
        # List converters
        converters = ModelConverterRegistry.list_converters()
        
        # Should have 2 converters
        self.assertEqual(len(converters), 2)
        
        # Check that converter info is correct
        converter1_info = next(c for c in converters if c['source_format'] == 'test1')
        converter2_info = next(c for c in converters if c['source_format'] == 'test2')
        
        self.assertEqual(converter1_info['target_format'], 'output1')
        self.assertEqual(converter1_info['model_type'], '*')
        self.assertEqual(converter1_info['converter_class'], 'Converter1')
        
        self.assertEqual(converter2_info['target_format'], 'output2')
        self.assertEqual(converter2_info['model_type'], 'bert')
        self.assertEqual(converter2_info['converter_class'], 'Converter2')


if __name__ == '__main__':
    unittest.main()