#!/usr/bin/env python3
"""
Test migration script for IPFS Accelerate.

This script analyzes existing test files and migrates them to the new test structure.
"""

import os
import sys
import re
import ast
import argparse
import logging
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import templates
try:
    from test.template_system.templates.model_test_template import ModelTestTemplate
    from test.template_system.templates.hardware_test_template import HardwareTestTemplate
    from test.template_system.templates.api_test_template import APITestTemplate
except ImportError:
    logger.warning("Could not import templates. Running in analysis-only mode.")


class TestAnalyzer:
    """
    Class to analyze test files and determine their type.
    """
    
    def __init__(self, file_path: str):
        """
        Initialize the analyzer.
        
        Args:
            file_path: Path to the test file
        """
        self.file_path = file_path
        self.content = self._read_file(file_path)
        self.ast = self._parse_ast()
        
    def _read_file(self, file_path: str) -> str:
        """
        Read the file content.
        
        Args:
            file_path: Path to the file
            
        Returns:
            File content as string
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return ""
    
    def _parse_ast(self) -> Optional[ast.Module]:
        """
        Parse the file into an AST.
        
        Returns:
            AST module object or None if parsing fails
        """
        try:
            return ast.parse(self.content)
        except SyntaxError as e:
            logger.error(f"Syntax error parsing {self.file_path}: {e}")
            return None
    
    def analyze(self) -> Dict[str, Any]:
        """
        Analyze the test file to determine its type and parameters.
        
        Returns:
            Dictionary with analysis results
        """
        result = {
            'file_path': self.file_path,
            'test_type': 'unknown',
            'imports': self._analyze_imports(),
            'classes': self._analyze_classes(),
            'functions': self._analyze_functions(),
            'markers': self._analyze_markers(),
            'parameters': {}
        }
        
        # Determine test type
        if self._is_model_test():
            result['test_type'] = 'model'
            result['parameters'] = self._extract_model_parameters()
        elif self._is_hardware_test():
            result['test_type'] = 'hardware'
            result['parameters'] = self._extract_hardware_parameters()
        elif self._is_api_test():
            result['test_type'] = 'api'
            result['parameters'] = self._extract_api_parameters()
        
        return result
    
    def _analyze_imports(self) -> List[str]:
        """
        Analyze imports in the file.
        
        Returns:
            List of imported modules
        """
        if not self.ast:
            return []
        
        imports = []
        
        for node in ast.walk(self.ast):
            if isinstance(node, ast.Import):
                for name in node.names:
                    imports.append(name.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(f"{node.module}.{name.name}" for name in node.names)
                else:
                    imports.extend(name.name for name in node.names)
        
        return imports
    
    def _analyze_classes(self) -> List[Dict[str, Any]]:
        """
        Analyze classes in the file.
        
        Returns:
            List of class information dictionaries
        """
        if not self.ast:
            return []
        
        classes = []
        
        for node in ast.walk(self.ast):
            if isinstance(node, ast.ClassDef):
                methods = []
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        methods.append({
                            'name': item.name,
                            'decorator_list': [self._get_decorator_name(d) for d in item.decorator_list]
                        })
                
                classes.append({
                    'name': node.name,
                    'methods': methods
                })
        
        return classes
    
    def _get_decorator_name(self, decorator: ast.expr) -> str:
        """
        Get the name of a decorator.
        
        Args:
            decorator: AST decorator node
            
        Returns:
            Decorator name as string
        """
        if isinstance(decorator, ast.Name):
            return decorator.id
        elif isinstance(decorator, ast.Attribute):
            return f"{self._get_decorator_name(decorator.value)}.{decorator.attr}"
        elif isinstance(decorator, ast.Call):
            return self._get_decorator_name(decorator.func)
        return str(decorator)
    
    def _analyze_functions(self) -> List[Dict[str, Any]]:
        """
        Analyze functions in the file.
        
        Returns:
            List of function information dictionaries
        """
        if not self.ast:
            return []
        
        functions = []
        
        for node in ast.walk(self.ast):
            if isinstance(node, ast.FunctionDef) and node.parent_field != 'body':
                functions.append({
                    'name': node.name,
                    'decorator_list': [self._get_decorator_name(d) for d in node.decorator_list]
                })
        
        return functions
    
    def _analyze_markers(self) -> List[str]:
        """
        Analyze pytest markers in the file.
        
        Returns:
            List of marker names
        """
        markers = []
        
        # Use regex to find pytest markers
        marker_pattern = r'@pytest\.mark\.(\w+)'
        markers = re.findall(marker_pattern, self.content)
        
        return list(set(markers))
    
    def _is_model_test(self) -> bool:
        """
        Check if the file is a model test.
        
        Returns:
            True if the file is a model test, False otherwise
        """
        model_keywords = ['model', 'tokenizer', 'transformer', 'bert', 'gpt', 't5', 'vit', 'whisper']
        model_imports = ['transformers', 'torch', 'tensorflow']
        
        # Check imports
        for imp in self._analyze_imports():
            if any(keyword in imp.lower() for keyword in model_imports):
                return True
        
        # Check content
        if any(keyword in self.content.lower() for keyword in model_keywords):
            return True
        
        # Check markers
        markers = self._analyze_markers()
        if 'model' in markers or 'text' in markers or 'vision' in markers or 'audio' in markers:
            return True
        
        return False
    
    def _is_hardware_test(self) -> bool:
        """
        Check if the file is a hardware test.
        
        Returns:
            True if the file is a hardware test, False otherwise
        """
        hardware_keywords = ['webgpu', 'webnn', 'cuda', 'rocm', 'mps', 'hardware']
        
        # Check content
        if any(keyword in self.content.lower() for keyword in hardware_keywords):
            return True
        
        # Check markers
        markers = self._analyze_markers()
        if any(hw in markers for hw in ['webgpu', 'webnn', 'cuda', 'rocm', 'hardware']):
            return True
        
        return False
    
    def _is_api_test(self) -> bool:
        """
        Check if the file is an API test.
        
        Returns:
            True if the file is an API test, False otherwise
        """
        api_keywords = ['api', 'endpoint', 'client', 'server', 'http', 'rest', 'openai', 'hf_tei', 'hf_tgi', 'ollama', 'vllm', 'claude']
        api_imports = ['requests', 'openai', 'anthropic']
        
        # Check imports
        for imp in self._analyze_imports():
            if any(keyword in imp.lower() for keyword in api_imports):
                return True
        
        # Check content
        if any(keyword in self.content.lower() for keyword in api_keywords):
            return True
        
        # Check markers
        markers = self._analyze_markers()
        if 'api' in markers:
            return True
        
        return False
    
    def _extract_model_parameters(self) -> Dict[str, Any]:
        """
        Extract model test parameters.
        
        Returns:
            Dictionary with model parameters
        """
        parameters = {
            'model_name': 'unknown',
            'model_type': 'unknown',
            'framework': 'transformers'
        }
        
        # Try to extract model name
        model_name_pattern = r'(?:model_name|name)\s*=\s*["\']([^"\']+)["\']'
        model_name_match = re.search(model_name_pattern, self.content)
        if model_name_match:
            parameters['model_name'] = model_name_match.group(1)
        else:
            # Try to extract from class name or file name
            file_name = os.path.basename(self.file_path)
            if 'test_' in file_name:
                model_name = file_name.replace('test_', '').replace('.py', '')
                parameters['model_name'] = model_name
        
        # Determine model type
        if any(keyword in self.content.lower() for keyword in ['bert', 't5', 'gpt', 'llama']):
            parameters['model_type'] = 'text'
        elif any(keyword in self.content.lower() for keyword in ['vit', 'resnet', 'image']):
            parameters['model_type'] = 'vision'
        elif any(keyword in self.content.lower() for keyword in ['whisper', 'wav2vec', 'audio']):
            parameters['model_type'] = 'audio'
        elif any(keyword in self.content.lower() for keyword in ['clip', 'multimodal']):
            parameters['model_type'] = 'multimodal'
        
        return parameters
    
    def _extract_hardware_parameters(self) -> Dict[str, Any]:
        """
        Extract hardware test parameters.
        
        Returns:
            Dictionary with hardware parameters
        """
        parameters = {
            'hardware_platform': 'unknown',
            'test_name': 'unknown',
            'test_operation': 'matmul',
            'test_category': 'compute'
        }
        
        # Determine hardware platform
        for platform in ['webgpu', 'webnn', 'cuda', 'rocm', 'cpu']:
            if platform in self.content.lower():
                parameters['hardware_platform'] = platform
                break
        
        # Extract test name from file name
        file_name = os.path.basename(self.file_path).replace('test_', '').replace('.py', '')
        parameters['test_name'] = file_name
        
        # Determine test operation
        for op in ['matmul', 'conv', 'inference']:
            if op in self.content.lower():
                parameters['test_operation'] = op
                break
        
        # Determine test category
        for cat in ['compute', 'memory', 'throughput', 'latency']:
            if cat in self.content.lower():
                parameters['test_category'] = cat
                break
        
        return parameters
    
    def _extract_api_parameters(self) -> Dict[str, Any]:
        """
        Extract API test parameters.
        
        Returns:
            Dictionary with API parameters
        """
        parameters = {
            'api_name': 'unknown',
            'test_name': 'unknown',
            'api_type': 'internal'
        }
        
        # Extract API name
        for api in ['openai', 'hf_tei', 'hf_tgi', 'ollama', 'vllm', 'claude']:
            if api in self.content.lower():
                parameters['api_name'] = api
                parameters['api_type'] = api
                break
        
        # Extract test name from file name
        file_name = os.path.basename(self.file_path).replace('test_', '').replace('.py', '')
        parameters['test_name'] = file_name
        
        return parameters


class TestMigrator:
    """
    Class to migrate test files to the new structure.
    """
    
    def __init__(self, 
                 source_path: str, 
                 output_dir: str, 
                 analysis_result: Dict[str, Any],
                 dry_run: bool = False):
        """
        Initialize the migrator.
        
        Args:
            source_path: Path to the source test file
            output_dir: Output directory for the migrated test
            analysis_result: Analysis result from TestAnalyzer
            dry_run: Whether to perform a dry run (no file creation)
        """
        self.source_path = source_path
        self.output_dir = output_dir
        self.analysis = analysis_result
        self.dry_run = dry_run
    
    def migrate(self) -> Optional[str]:
        """
        Migrate the test file to the new structure.
        
        Returns:
            Path to the migrated file or None if migration fails
        """
        test_type = self.analysis['test_type']
        
        try:
            if test_type == 'model':
                return self._migrate_model_test()
            elif test_type == 'hardware':
                return self._migrate_hardware_test()
            elif test_type == 'api':
                return self._migrate_api_test()
            else:
                logger.warning(f"Unknown test type for {self.source_path}")
                return None
        except Exception as e:
            logger.error(f"Error migrating {self.source_path}: {e}")
            return None
    
    def _migrate_model_test(self) -> Optional[str]:
        """
        Migrate a model test.
        
        Returns:
            Path to the migrated file or None if migration fails
        """
        params = self.analysis['parameters']
        
        # Check for required parameters
        if 'model_name' not in params or params['model_name'] == 'unknown':
            logger.warning(f"Missing model_name for {self.source_path}")
            return None
        
        if 'model_type' not in params or params['model_type'] == 'unknown':
            logger.warning(f"Missing model_type for {self.source_path}")
            return None
        
        if self.dry_run:
            logger.info(f"Would migrate model test {self.source_path} to {self.output_dir}")
            return None
        
        try:
            # Create the template
            template = ModelTestTemplate(
                model_name=params['model_name'],
                model_type=params['model_type'],
                framework=params.get('framework', 'transformers'),
                output_dir=self.output_dir,
                overwrite=True
            )
            
            # Generate the test file
            output_path = template.generate()
            
            logger.info(f"Migrated model test from {self.source_path} to {output_path}")
            
            return output_path
        except Exception as e:
            logger.error(f"Error migrating model test {self.source_path}: {e}")
            return None
    
    def _migrate_hardware_test(self) -> Optional[str]:
        """
        Migrate a hardware test.
        
        Returns:
            Path to the migrated file or None if migration fails
        """
        params = self.analysis['parameters']
        
        # Check for required parameters
        if params['hardware_platform'] == 'unknown':
            logger.warning(f"Unknown hardware platform for {self.source_path}")
            return None
        
        if self.dry_run:
            logger.info(f"Would migrate hardware test {self.source_path} to {self.output_dir}")
            return None
        
        try:
            # Create the template
            template = HardwareTestTemplate(
                parameters={
                    'hardware_platform': params['hardware_platform'],
                    'test_name': params['test_name'],
                    'test_operation': params.get('test_operation', 'matmul'),
                    'test_category': params.get('test_category', 'compute')
                },
                output_dir=self.output_dir
            )
            
            # Generate the test file
            output_path = template.write()
            
            logger.info(f"Migrated hardware test from {self.source_path} to {output_path}")
            
            return output_path
        except Exception as e:
            logger.error(f"Error migrating hardware test {self.source_path}: {e}")
            return None
    
    def _migrate_api_test(self) -> Optional[str]:
        """
        Migrate an API test.
        
        Returns:
            Path to the migrated file or None if migration fails
        """
        params = self.analysis['parameters']
        
        if self.dry_run:
            logger.info(f"Would migrate API test {self.source_path} to {self.output_dir}")
            return None
        
        try:
            # Create the template
            template = APITestTemplate(
                parameters={
                    'api_name': params['api_name'],
                    'test_name': params['test_name'],
                    'api_type': params.get('api_type', 'internal')
                },
                output_dir=self.output_dir
            )
            
            # Generate the test file
            output_path = template.write()
            
            logger.info(f"Migrated API test from {self.source_path} to {output_path}")
            
            return output_path
        except Exception as e:
            logger.error(f"Error migrating API test {self.source_path}: {e}")
            return None


def find_test_files(directory: str, regex_pattern: Optional[str] = None) -> List[str]:
    """
    Find test files in a directory.
    
    Args:
        directory: Directory to search
        regex_pattern: Optional regex pattern to filter files
        
    Returns:
        List of file paths
    """
    if not os.path.isdir(directory):
        logger.error(f"{directory} is not a directory")
        return []
    
    test_files = []
    
    for root, _, files in os.walk(directory):
        for file in files:
            if file.startswith('test_') and file.endswith('.py'):
                file_path = os.path.join(root, file)
                
                # Apply regex filter if provided
                if regex_pattern and not re.search(regex_pattern, file_path):
                    continue
                
                test_files.append(file_path)
    
    return test_files


def save_analysis_report(analysis_results: List[Dict[str, Any]], output_path: str) -> None:
    """
    Save analysis report to a file.
    
    Args:
        analysis_results: List of analysis results
        output_path: Path to save the report
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, indent=2)
        
        logger.info(f"Analysis report saved to {output_path}")
    except Exception as e:
        logger.error(f"Error saving analysis report: {e}")


def main() -> None:
    """
    Main function to migrate test files.
    """
    parser = argparse.ArgumentParser(description='Migrate test files to the new structure')
    
    parser.add_argument('--source-dir', required=True, help='Source directory containing test files')
    parser.add_argument('--output-dir', default='test', help='Output directory for migrated tests')
    parser.add_argument('--pattern', help='Regex pattern to filter test files')
    parser.add_argument('--analyze-only', action='store_true', help='Only analyze, do not migrate')
    parser.add_argument('--report-file', default='migration_analysis.json', help='Path to save the analysis report')
    parser.add_argument('--dry-run', action='store_true', help='Perform a dry run (no file creation)')
    
    args = parser.parse_args()
    
    # Find test files
    logger.info(f"Searching for test files in {args.source_dir}")
    test_files = find_test_files(args.source_dir, args.pattern)
    logger.info(f"Found {len(test_files)} test files")
    
    # Analyze test files
    analysis_results = []
    for file_path in test_files:
        logger.info(f"Analyzing {file_path}")
        analyzer = TestAnalyzer(file_path)
        result = analyzer.analyze()
        analysis_results.append(result)
        logger.info(f"  Type: {result['test_type']}")
    
    # Save analysis report
    save_analysis_report(analysis_results, args.report_file)
    
    if args.analyze_only:
        logger.info("Analysis completed, skipping migration")
        return
    
    # Migrate test files
    logger.info(f"Migrating test files to {args.output_dir}")
    
    for result in analysis_results:
        if result['test_type'] == 'unknown':
            logger.warning(f"Skipping {result['file_path']} with unknown type")
            continue
        
        migrator = TestMigrator(
            source_path=result['file_path'],
            output_dir=args.output_dir,
            analysis_result=result,
            dry_run=args.dry_run
        )
        
        migrated_path = migrator.migrate()
        if migrated_path:
            logger.info(f"Migration successful: {migrated_path}")
    
    logger.info("Migration completed")


if __name__ == "__main__":
    main()