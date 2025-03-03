#!/usr/bin/env python3
"""
Unit tests for model benchmarking tools
"""

import os
import sys
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the code to test
from test.run_model_benchmarks import ModelBenchmarkRunner, KEY_MODEL_SET, SMALL_MODEL_SET

class TestModelBenchmarks(unittest.TestCase):
    """Test model benchmarking functionality"""
    
    def setUp(self):
        """Set up for tests"""
        # Create a temporary directory for test output
        self.test_dir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.test_dir.name)
    
    def tearDown(self):
        """Clean up after tests"""
        self.test_dir.cleanup()
    
    @patch('test.run_model_benchmarks.ModelBenchmarkRunner._detect_hardware')
    def test_initialization(self, mock_detect_hardware):
        """Test initialization of ModelBenchmarkRunner"""
        # Set up mock
        mock_detect_hardware.return_value = {'cpu': True, 'cuda': False}
        
        # Initialize with default settings
        runner = ModelBenchmarkRunner(output_dir=self.output_dir)
        
        # Check basic initialization
        self.assertEqual(runner.output_dir, self.output_dir)
        self.assertEqual(runner.models_set, 'key')
        self.assertEqual(runner.models, KEY_MODEL_SET)
        self.assertTrue(runner.verify_functionality)
        self.assertTrue(runner.measure_performance)
        
        # Check hardware detection was called
        mock_detect_hardware.assert_called_once()
        
        # Check run directory was created
        run_dirs = list(self.output_dir.glob('*'))
        self.assertEqual(len(run_dirs), 1)
        self.assertTrue(run_dirs[0].is_dir())
    
    @patch('test.run_model_benchmarks.ModelBenchmarkRunner._detect_hardware')
    def test_custom_models(self, mock_detect_hardware):
        """Test initialization with custom models"""
        # Set up mock
        mock_detect_hardware.return_value = {'cpu': True}
        
        # Create custom models
        custom_models = {
            'test_model': {
                'name': 'test/model',
                'family': 'test',
                'size': 'small',
                'modality': 'text'
            }
        }
        
        # Initialize with custom models
        runner = ModelBenchmarkRunner(
            output_dir=self.output_dir,
            models_set='custom',
            custom_models=custom_models
        )
        
        # Check custom models were set
        self.assertEqual(runner.models_set, 'custom')
        self.assertEqual(runner.models, custom_models)
    
    @patch('test.run_model_benchmarks.ModelBenchmarkRunner._detect_hardware')
    def test_small_models_set(self, mock_detect_hardware):
        """Test initialization with small models set"""
        # Set up mock
        mock_detect_hardware.return_value = {'cpu': True}
        
        # Initialize with small models set
        runner = ModelBenchmarkRunner(
            output_dir=self.output_dir,
            models_set='small'
        )
        
        # Check small models set was used
        self.assertEqual(runner.models_set, 'small')
        self.assertEqual(runner.models, SMALL_MODEL_SET)
    
    @patch('test.run_model_benchmarks.ModelBenchmarkRunner._detect_hardware')
    def test_save_run_configuration(self, mock_detect_hardware):
        """Test saving run configuration"""
        # Set up mock
        mock_detect_hardware.return_value = {'cpu': True}
        
        # Initialize runner
        runner = ModelBenchmarkRunner(output_dir=self.output_dir)
        
        # Save configuration
        runner._save_run_configuration()
        
        # Check configuration file was created
        config_file = list(self.output_dir.glob('*/benchmark_config.json'))[0]
        self.assertTrue(config_file.exists())
        
        # Check configuration contents
        with open(config_file, 'r') as f:
            config = json.load(f)
            
        self.assertEqual(config['models_set'], 'key')
        self.assertEqual(config['verify_functionality'], True)
        self.assertEqual(config['measure_performance'], True)
        self.assertEqual(config['available_hardware'], {'cpu': True})
    
    @patch('test.run_model_benchmarks.ModelBenchmarkRunner._detect_hardware')
    @patch('test.run_model_benchmarks.ModelBenchmarkRunner._verify_model_functionality')
    @patch('test.run_model_benchmarks.ModelBenchmarkRunner._run_performance_benchmarks')
    @patch('test.run_model_benchmarks.ModelBenchmarkRunner._generate_plots')
    @patch('test.run_model_benchmarks.ModelBenchmarkRunner._update_compatibility_matrix')
    @patch('test.run_model_benchmarks.ModelBenchmarkRunner._save_results')
    @patch('test.run_model_benchmarks.ModelBenchmarkRunner._generate_report')
    def test_run_benchmarks(
        self, mock_generate_report, mock_save_results,
        mock_update_matrix, mock_generate_plots,
        mock_run_benchmarks, mock_verify, mock_detect_hardware
    ):
        """Test running benchmarks"""
        # Set up mocks
        mock_detect_hardware.return_value = {'cpu': True}
        mock_generate_report.return_value = Path('test_report.md')
        
        # Initialize runner
        runner = ModelBenchmarkRunner(
            output_dir=self.output_dir,
            models_set='small',
            verify_functionality=True,
            measure_performance=True,
            generate_plots=True,
            update_compatibility_matrix=True
        )
        
        # Run benchmarks
        results = runner.run_benchmarks()
        
        # Check methods were called
        mock_verify.assert_called_once()
        mock_run_benchmarks.assert_called_once()
        mock_generate_plots.assert_called_once()
        mock_update_matrix.assert_called_once()
        mock_save_results.assert_called_once()
        mock_generate_report.assert_called_once()
        
        # Check results
        self.assertIn('timestamp', results)
        self.assertIn('models', results)
        self.assertIn('hardware_detected', results)
    
    @patch('test.run_model_benchmarks.ModelBenchmarkRunner._detect_hardware')
    @patch('test.run_model_benchmarks.os.path.exists')
    @patch('subprocess.run')
    def test_verify_model_functionality(self, mock_subprocess_run, mock_path_exists, mock_detect_hardware):
        """Test model functionality verification"""
        # Set up mocks
        mock_detect_hardware.return_value = {'cpu': True}
        
        # Force the path check for verify_model_functionality.py to return False
        # This will use subprocess to run the verification
        mock_path_exists.return_value = False
        
        # Mock subprocess result
        mock_process = MagicMock()
        mock_process.returncode = 1  # Force fallback to manual verification
        mock_subprocess_run.return_value = mock_process
        
        # Initialize runner with small model set
        runner = ModelBenchmarkRunner(
            output_dir=self.output_dir,
            models_set='small',
            hardware_types=['cpu']
        )
        
        # Mock the _verify_models_manually method
        mock_verify_result = {
            'status': 'completed',
            'hardware': 'cpu',
            'models': {
                'bert': {'success': True},
                't5': {'success': True}
            },
            'summary': {
                'total': 2,
                'successful': 2,
                'failed': 0,
                'success_rate': 100.0
            }
        }
        
        with patch.object(runner, '_verify_models_manually', return_value=mock_verify_result) as mock_verify_manually:
            # Run verification
            runner._verify_model_functionality()
            
            # Check results
            self.assertIn('functionality_verification', runner.results)
            self.assertIn('cpu', runner.results['functionality_verification'])
            
            # Manual verification should be called since subprocess fails
            mock_verify_manually.assert_called_once()
    
    @patch('test.run_model_benchmarks.subprocess.run')
    @patch('test.run_model_benchmarks.ModelBenchmarkRunner._detect_hardware')
    def test_run_performance_benchmarks(self, mock_detect_hardware, mock_subprocess_run):
        """Test running performance benchmarks"""
        # Set up mocks
        mock_detect_hardware.return_value = {'cpu': True}
        
        # Mock subprocess result
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_subprocess_run.return_value = mock_process
        
        # Create a mock benchmark results file
        def mock_file_exists(path):
            # Simulate benchmark_results.json existing after subprocess.run is called
            if str(path).endswith('benchmark_results.json'):
                if mock_subprocess_run.call_count > 0:
                    return True
            return os.path.exists(path)
        
        # Initialize runner with small model set
        runner = ModelBenchmarkRunner(
            output_dir=self.output_dir,
            models_set='small',
            hardware_types=['cpu'],
            verify_functionality=False,
            measure_performance=True
        )
        
        # Create fake benchmark results for mocking
        os.makedirs(self.output_dir / runner.timestamp / 'performance' / 'embedding', exist_ok=True)
        with open(self.output_dir / runner.timestamp / 'performance' / 'embedding' / 'benchmark_results.json', 'w') as f:
            json.dump({
                'benchmarks': {
                    'embedding': {
                        'bert-base-uncased': {
                            'cpu': {
                                'status': 'completed',
                                'performance_summary': {
                                    'latency': {'mean': 0.05},
                                    'throughput': {'mean': 20}
                                }
                            }
                        }
                    }
                }
            }, f)
        
        # Run benchmarks with patched Path.exists
        with patch('pathlib.Path.exists', side_effect=mock_file_exists):
            runner._run_performance_benchmarks()
        
        # Check results
        self.assertIn('performance_benchmarks', runner.results)
        
        # Check subprocess was called
        mock_subprocess_run.assert_called()
    
    @patch('test.run_model_benchmarks.plt')
    @patch('test.run_model_benchmarks.pd.DataFrame')
    @patch('test.run_model_benchmarks.ModelBenchmarkRunner._detect_hardware')
    def test_generate_plots(self, mock_detect_hardware, mock_dataframe, mock_plt):
        """Test generating plots"""
        # Set up mocks
        mock_detect_hardware.return_value = {'cpu': True}
        
        # Create dataframe mock
        df_mock = MagicMock()
        mock_dataframe.return_value = df_mock
        
        # Create plot mock
        plot_mock = MagicMock()
        df_mock.plot.return_value = plot_mock
        df_mock.pivot_table.return_value = df_mock
        
        # Initialize runner
        runner = ModelBenchmarkRunner(
            output_dir=self.output_dir,
            models_set='small',
            hardware_types=['cpu']
        )
        
        # Add fake results data
        runner.results['functionality_verification'] = {
            'cpu': {
                'summary': {
                    'total': 2,
                    'successful': 2,
                    'failed': 0,
                    'success_rate': 100.0
                },
                'models': {
                    'bert': {'success': True},
                    't5': {'success': True}
                }
            }
        }
        
        runner.results['performance_benchmarks'] = {
            'embedding': {
                'benchmarks': {
                    'bert-base-uncased': {
                        'cpu': {
                            'status': 'completed',
                            'performance_summary': {
                                'latency': {'mean': 0.05},
                                'throughput': {'mean': 20}
                            }
                        }
                    }
                }
            }
        }
        
        # Override HAS_VISUALIZATION to ensure plots are attempted
        with patch('test.run_model_benchmarks.HAS_VISUALIZATION', True):
            runner._generate_plots()
        
        # Check plots directory was created
        plots_dir = runner.run_dir / 'plots'
        self.assertTrue(plots_dir.exists())
        
        # Check plot methods were called
        mock_plt.figure.assert_called()
        mock_plt.savefig.assert_called()
    
    @patch('test.run_model_benchmarks.ModelBenchmarkRunner._detect_hardware')
    def test_save_results(self, mock_detect_hardware):
        """Test saving benchmark results"""
        # Set up mock
        mock_detect_hardware.return_value = {'cpu': True}
        
        # Initialize runner
        runner = ModelBenchmarkRunner(output_dir=self.output_dir)
        
        # Save results
        runner._save_results()
        
        # Check results file was created
        results_file = runner.run_dir / 'benchmark_results.json'
        self.assertTrue(results_file.exists())
        
        # Check results contents
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        self.assertIn('timestamp', results)
        self.assertIn('models', results)
        self.assertIn('hardware_detected', results)
    
    @patch('test.run_model_benchmarks.ModelBenchmarkRunner._detect_hardware')
    def test_generate_report(self, mock_detect_hardware):
        """Test generating benchmark report"""
        # Set up mock
        mock_detect_hardware.return_value = {'cpu': True}
        
        # Initialize runner
        runner = ModelBenchmarkRunner(output_dir=self.output_dir)
        
        # Add fake results data
        runner.results['functionality_verification'] = {
            'cpu': {
                'summary': {
                    'total': 2,
                    'successful': 2,
                    'failed': 0,
                    'success_rate': 100.0
                },
                'models': {
                    'bert': {'success': True},
                    't5': {'success': True}
                }
            }
        }
        
        # Generate report
        report_path = runner._generate_report()
        
        # Check report file was created
        self.assertTrue(report_path.exists())
        
        # Check report contents
        with open(report_path, 'r') as f:
            content = f.read()
        
        self.assertIn('Model Benchmark Report', content)
        self.assertIn('Hardware Platforms', content)
        self.assertIn('Models Tested', content)
        self.assertIn('Functionality Verification Results', content)

if __name__ == '__main__':
    unittest.main()