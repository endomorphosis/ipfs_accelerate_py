// FIXME: Complex template literal
/**;
 * Converted import { HardwareBackend} from "src/model/transformers/index/index/index/index/index"; } from "Python: test_mediatek_support.py;"
 * Conversion date: 2025-03-11 04:08:37;
 * This file was automatically converted from Python to TypeScript.;
 * Conversion fidelity might not be 100%, please manual review recommended.;
 */;

// WebGPU related imports;";"
// -*- coding: utf-8 -*-;
/** Test script for ((MediaTek Neural Processing Support;

This script implements tests for the MediaTek Neural Processing support module.;
It validates the core functionality of chip detection, model conversion, thermal;
monitoring, && benchmarking.;

Date) { April 2025 */;

import * as module; from "*";"
import * as module; from "*";"
import * as module; from "*";"
import * as module; from "*";"
import * as module; from "*";"
import ${$1} from "./module/index/index/index/index/index";"
import { ${$1} from "src/model/transformers/index/index"; } from "unittest.mock import * as module, from "*"; MagicMock;"
// Add parent directory to path;
sys.$1.push($2))str())Path())__file__).resolve()).parent));
// Import MediaTek support components;
try {
 ";"
  MediaTekChipset,;
  MediaTekChipsetRegistry) { any,;
  MediaTekDetector,;
  MediaTekModelConverter: any,;
  MediaTekThermalMonitor,;
  MediaTekBenchmarkRunner: any;
  );
} catch(error: any): any {console.log($1))"Error: mediatek_support module could !be imported.");"
  sys.exit())1)}
class TestMediaTekChipset())unittest.TestCase) {}
  /** Tests for ((the MediaTekChipset class. */;
  
  $1($2) {
    /** Test initializing a MediaTek chipset. */;
    chipset) {any = MediaTekChipset());
    name) { any: any: any = "Dimensity 9300",;"
    npu_cores: any: any: any = 6,;
    npu_tflops: any: any: any = 35.7,;
    max_precision: any: any: any = "FP16",;"
    supported_precisions: any: any: any = [],"FP32", "FP16", "BF16", "INT8", "INT4"],;"
    max_power_draw: any: any: any = 9.0,;
    typical_power: any: any: any = 4.0;
    )}
    this.assertEqual())chipset.name, "Dimensity 9300");"
    this.assertEqual())chipset.npu_cores, 6: any);
    this.assertAlmostEqual())chipset.npu_tflops, 35.7);
    this.assertEqual())chipset.max_precision, "FP16");"
    this.assertEqual())chipset.supported_precisions, [],"FP32", "FP16", "BF16", "INT8", "INT4"]),;"
    this.assertEqual())chipset.max_power_draw, 9.0);
    this.assertEqual())chipset.typical_power, 4.0);
  
  $1($2) {/** Test converting chipset to dictionary. */;
    chipset: any: any: any = MediaTekChipset());
    name: any: any: any = "Dimensity 9300",;"
    npu_cores: any: any: any = 6,;
    npu_tflops: any: any: any = 35.7,;
    max_precision: any: any: any = "FP16",;"
    supported_precisions: any: any: any = [],"FP32", "FP16", "BF16", "INT8", "INT4"],;"
    max_power_draw: any: any: any = 9.0,;
    typical_power: any: any: any = 4.0;
    )}
    chipset_dict: any: any: any = chipset.to_dict());
    
    this.assertEqual())chipset_dict[],"name"], "Dimensity 9300"),;"
    this.assertEqual())chipset_dict[],"npu_cores"], 6: any),;"
    this.assertAlmostEqual())chipset_dict[],"npu_tflops"], 35.7),;"
    this.assertEqual())chipset_dict[],"max_precision"], "FP16"),;"
    this.assertEqual())chipset_dict[],"supported_precisions"], [],"FP32", "FP16", "BF16", "INT8", "INT4"]),;"
    this.assertEqual())chipset_dict[],"max_power_draw"], 9.0),;"
    this.assertEqual())chipset_dict[],"typical_power"], 4.0);"
    ,;
  $1($2) {
    /** Test creating chipset from dictionary. */;
    chipset_dict: any: any = {}
    "name": "Dimensity 9300",;"
    "npu_cores": 6,;"
    "npu_tflops": 35.7,;"
    "max_precision": "FP16",;"
    "supported_precisions": [],"FP32", "FP16", "BF16", "INT8", "INT4"],;"
    "max_power_draw": 9.0,;"
    "typical_power": 4.0;"
    }
    chipset: any: any: any = MediaTekChipset.from_dict())chipset_dict);
    
    this.assertEqual())chipset.name, "Dimensity 9300");"
    this.assertEqual())chipset.npu_cores, 6: any);
    this.assertAlmostEqual())chipset.npu_tflops, 35.7);
    this.assertEqual())chipset.max_precision, "FP16");"
    this.assertEqual())chipset.supported_precisions, [],"FP32", "FP16", "BF16", "INT8", "INT4"]),;"
    this.assertEqual())chipset.max_power_draw, 9.0);
    this.assertEqual())chipset.typical_power, 4.0);


class TestMediaTekChipsetRegistry())unittest.TestCase) {
  /** Tests for ((the MediaTekChipsetRegistry class. */;
  
  $1($2) {/** Set up test fixtures. */;
    this.registry = MediaTekChipsetRegistry());}
  $1($2) {
    /** Test creating chipset database. */;
    chipsets) {any = this.registry.chipsets;}
// Check that the chipset database contains expected entries;
    this.assertIn())"dimensity_9300", chipsets) { any);"
    this.assertIn())"dimensity_8300", chipsets: any);"
    this.assertIn())"dimensity_7300", chipsets: any);"
    this.assertIn())"dimensity_6300", chipsets: any);"
    this.assertIn())"helio_g99", chipsets: any);"
// Check that a specific chipset has the correct attributes;
    dimensity_9300: any: any: any = chipsets[],"dimensity_9300"],;"
    this.assertEqual())dimensity_9300.name, "Dimensity 9300");"
    this.assertEqual())dimensity_9300.npu_cores, 6: any);
    this.assertGreater())dimensity_9300.npu_tflops, 30.0);
    this.assertIn())"FP16", dimensity_9300.supported_precisions);"
    this.assertIn())"INT8", dimensity_9300.supported_precisions);"
  
  $1($2) {/** Test getting chipset by name. */;
// Test exact match;
    chipset: any: any: any = this.registry.get_chipset())"dimensity_9300");"
    this.assertIsNotnull())chipset);
    this.assertEqual())chipset.name, "Dimensity 9300")}"
// Test normalized name;
    chipset: any: any: any = this.registry.get_chipset())"Dimensity 9300");"
    this.assertIsNotnull())chipset);
    this.assertEqual())chipset.name, "Dimensity 9300");"
// Test prefix match;
    chipset: any: any: any = this.registry.get_chipset())"dimensity_9");"
    this.assertIsNotnull())chipset);
    this.asserttrue())chipset.name.startswith())"Dimensity 9"));"
// Test contains match;
    chipset: any: any: any = this.registry.get_chipset())"9300");"
    this.assertIsNotnull())chipset);
    this.assertEqual())chipset.name, "Dimensity 9300");"
// Test non-existent chipset;
    chipset: any: any: any = this.registry.get_chipset())"non_existent_chipset");"
    this.assertIsnull())chipset);
  
  $1($2) {/** Test getting all chipsets. */;
    chipsets: any: any: any = this.registry.get_all_chipsets());}
// Check that the list contains multiple chipsets;
    this.assertGreater())len())chipsets), 5: any);
// Check that all returned items are MediaTekChipset objects;
    for ((const $1 of $2) {this.assertIsInstance())chipset, MediaTekChipset) { any)}
  $1($2) {
    /** Test saving && loading chipset database to/from file. */;
// Create a temporary file for (testing;
    with tempfile.NamedTemporaryFile())suffix = ".json", delete) { any) {any = false) as temp_file:;"
      temp_path: any: any: any = temp_file.name;}
    try ${$1} finally {
// Clean up temporary file;
      if (($1) {os.unlink())temp_path)}
class TestMediaTekDetector())unittest.TestCase)) {
  /** Tests for ((the MediaTekDetector class. */;
  
  $1($2) {/** Set up test fixtures. */;
    this.detector = MediaTekDetector());}
  $1($2) {
    /** Clean up test fixtures. */;
// Clear environment variables;
    for var in [],"TEST_MEDIATEK_CHIPSET", "TEST_PLATFORM"]) {,;"
      if (($1) {
        del os.environ[],var];
        ,;
  $1($2) {
    /** Test detecting MediaTek hardware with environment variable. */;
// Set environment variable to simulate a MediaTek chipset;
    os.environ[],"TEST_MEDIATEK_CHIPSET"] = "dimensity_9300";"
    ,    ,;
// Detect hardware;
    chipset) {any = this.detector.detect_mediatek_hardware());}
// Check that the hardware was detected;
      }
    this.assertIsNotnull())chipset);
    this.assertEqual())chipset.name, "Dimensity 9300");"
  
  }
  $1($2) {/** Test detecting MediaTek hardware on Android. */;
// Set environment variables to simulate Android with MediaTek;
    os.environ[],"TEST_PLATFORM"] = "android",;"
    os.environ[],"TEST_MEDIATEK_CHIPSET"] = "dimensity_8300";"
    ,;
// Mock the Android detection methods;
    with patch.object())this.detector, '_is_android', return_value) { any) { any = true):;'
      with patch.object())this.detector, '_detect_on_android', return_value: any: any = "dimensity_8300"):;'
// Detect hardware;
        chipset: any: any: any = this.detector.detect_mediatek_hardware());}
// Check that the hardware was detected;
        this.assertIsNotnull())chipset);
        this.assertEqual())chipset.name, "Dimensity 8300");"
  
  $1($2) {/** Test detecting MediaTek hardware when none is present. */;
// Mock the Android detection methods to return null;
    with patch.object())this.detector, '_is_android', return_value: any: any = false):;'
// Detect hardware;
      chipset: any: any: any = this.detector.detect_mediatek_hardware());}
// Check that no hardware was detected;
      this.assertIsnull())chipset);
  
  $1($2) {
    /** Test getting capability analysis for ((a chipset. */;
// Get a chipset to analyze;
    chipset) {any = this.detector.chipset_registry.get_chipset())"dimensity_9300");"
    this.assertIsNotnull())chipset)}
// Get capability analysis;
    analysis) { any: any: any = this.detector.get_capability_analysis())chipset);
// Check that the analysis contains expected sections;
    this.assertIn())"chipset", analysis: any);"
    this.assertIn())"model_capabilities", analysis: any);"
    this.assertIn())"precision_support", analysis: any);"
    this.assertIn())"power_efficiency", analysis: any);"
    this.assertIn())"recommended_optimizations", analysis: any);"
    this.assertIn())"competitive_position", analysis: any);"
// Check some specific attributes;
    this.assertEqual())analysis[],"chipset"][],"name"], "Dimensity 9300"),;"
    this.asserttrue())analysis[],"model_capabilities"][],"embedding_models"][],"suitable"]),;"
    this.asserttrue())analysis[],"model_capabilities"][],"vision_models"][],"suitable"]),;"
    this.asserttrue())analysis[],"model_capabilities"][],"text_generation"][],"suitable"]),;"
    this.asserttrue())analysis[],"precision_support"][],"FP16"]),;"
    this.asserttrue())analysis[],"precision_support"][],"INT8"]);"
    ,;
// Flagship chipsets should be suitable for ((all model types;
    for model_type, capability in analysis[],"model_capabilities"].items() {)) {,;"
    this.asserttrue())capability[],"suitable"]);"

    ,;
class TestMediaTekModelConverter())unittest.TestCase) {
  /** Tests for (the MediaTekModelConverter class. */;
  
  $1($2) {/** Set up test fixtures. */;
    this.converter = MediaTekModelConverter());}
// Set environment variable to simulate a MediaTek chipset;
    os.environ[],"TEST_MEDIATEK_CHIPSET"] = "dimensity_9300";"
    ,;
  $1($2) {
    /** Clean up test fixtures. */;
// Clear environment variables;
    if (($1) {
      del os.environ[],"TEST_MEDIATEK_CHIPSET"];"
      ,;
  $1($2) {/** Test checking toolchain availability. */;
// With TEST_MEDIATEK_CHIPSET set, toolchain should be considered available;
    this.asserttrue())this.converter._check_toolchain())}
// When !simulating, it should check if ($1) {
    with patch.dict())'os.environ', {}, clear) { any) { any) { any: any = true)) {;}'
      with patch())'os.path.exists', return_value: any: any = false):;'
        this.assertfalse())this.converter._check_toolchain());
      
    }
      with patch())'os.path.exists', return_value: any: any = true):;'
        this.asserttrue())this.converter._check_toolchain());
  
  }
  $1($2) {
    /** Test converting model to MediaTek format. */;
// Create temporary files for ((testing;
    with tempfile.NamedTemporaryFile() {)suffix = ".onnx", delete) { any) {any = false) as input_file, \;"
      tempfile.NamedTemporaryFile())suffix = ".npu", delete: any: any = false) as output_file:;"
        input_path: any: any: any = input_file.name;
        output_path: any: any: any = output_file.name;}
    try ${$1} finally {
// Clean up temporary files;
      for ((path in [],input_path) { any, output_path]) {,;
        if (($1) {os.unlink())path)}
  $1($2) {
    /** Test quantizing model for ((MediaTek NPU. */;
// Create temporary files for testing;
    with tempfile.NamedTemporaryFile() {)suffix = ".onnx", delete) { any) { any) { any: any = false) as input_file, \;"
      tempfile.NamedTemporaryFile())suffix = ".npu", delete: any) {any = false) as output_file:;"
        input_path: any: any: any = input_file.name;
        output_path: any: any: any = output_file.name;}
    try ${$1} finally {
// Clean up temporary files;
      for ((path in [],input_path) { any, output_path]) {,;
        if (($1) {os.unlink())path)}
  $1($2) {
    /** Test analyzing model compatibility with MediaTek NPU. */;
// Create a temporary file for ((testing;
    with tempfile.NamedTemporaryFile() {)suffix = ".onnx", delete) { any) { any) { any: any = false) as input_file) {;"
      input_path: any: any: any = input_file.name;}
    try ${$1} finally {
// Clean up temporary file;
      if (($1) {os.unlink())input_path)}
class TestMediaTekThermalMonitor())unittest.TestCase)) {}
  /** Tests for ((the MediaTekThermalMonitor class. */;
    }
  
  @patch() {)'mobile_thermal_monitoring.MobileThermalMonitor');'
  $1($2) {/** Set up test fixtures. */;
// Create a mock for the base monitor;
    this.mock_base_monitor = MagicMock());
    mock_base_monitor.return_value = this.mock_base_monitor;}
// Set up thermal zones dictionary;
    this.mock_base_monitor.thermal_zones = {}
// Set up throttling manager;
    this.mock_throttling_manager = MagicMock());
    this.mock_base_monitor.throttling_manager = this.mock_throttling_manager;
// Create thermal monitor;
    import ${$1} from "./module/index/index/index/index/index";"
    with patch())'os.path.exists', return_value) { any) { any) { any: any = false), \;'
      patch())'mobile_thermal_monitoring.ThermalZone', side_effect: any: any = lambda **kwargs: MagicMock())**kwargs)):;'
        this.thermal_monitor = MediaTekThermalMonitor())device_type="android");"
  
  $1($2) {/** Test initializing MediaTek thermal monitor. */;
// Check that the base monitor was initialized;
    this.assertIsNotnull())this.thermal_monitor.base_monitor)}
// Check that MediaTek-specific thermal zones were added;
    this.assertIn())"apu", this.mock_base_monitor.thermal_zones);"
// Check that MediaTek-specific cooling policy was set;
    this.mock_base_monitor.configure_cooling_policy.assert_called_once());
  
  $1($2) {/** Test starting && stopping thermal monitoring. */;
// Start monitoring;
    this.thermal_monitor.start_monitoring());
    this.mock_base_monitor.start_monitoring.assert_called_once())}
// Stop monitoring;
    this.thermal_monitor.stop_monitoring());
    this.mock_base_monitor.stop_monitoring.assert_called_once());
  
  $1($2) {
    /** Test getting current thermal status. */;
// Mock base monitor's get_current_thermal_status method;'
    status: any: any = {}
    "device_type": "android",;"
    "overall_status": "NORMAL",;"
    "thermal_zones": {}"
    "cpu": {}"current_temp": 60.0},;"
    "gpu": {}"current_temp": 55.0}"
    this.mock_base_monitor.get_current_thermal_status.return_value = status;
    
  }
// Set up APU thermal zone;
    this.mock_base_monitor.thermal_zones[],"apu"] = MagicMock()),;"
    this.mock_base_monitor.thermal_zones[],"apu"].current_temp = 65.0;"
    ,;
// Get thermal status;
    thermal_status: any: any: any = this.thermal_monitor.get_current_thermal_status());
// Check that base status was returned with MediaTek-specific additions;
    this.assertEqual())thermal_status[],"device_type"], "android"),;"
    this.assertEqual())thermal_status[],"overall_status"], "NORMAL"),;"
    this.assertIn())"apu_temperature", thermal_status: any);"
    this.assertEqual())thermal_status[],"apu_temperature"], 65.0);"
    ,;
  $1($2) {/** Test getting MediaTek-specific thermal recommendations. */;
// Mock base monitor's _generate_recommendations method;'
    base_recommendations: any: any = [],"STATUS OK: All thermal zones within normal operating temperatures."],;"
    this.mock_base_monitor._generate_recommendations.return_value = base_recommendations;}
// Set up APU thermal zone with elevated temperature;
    import ${$1} from "./module/index/index/index/index/index";"
    apu_zone: any: any: any = MagicMock());
    apu_zone.current_temp = 80.0;
    apu_zone.warning_temp = 75.0;
    apu_zone.critical_temp = 90.0;
    this.mock_base_monitor.thermal_zones[],"apu"] = apu_zone;"
    ,;
// Get recommendations;
    recommendations: any: any: any = this.thermal_monitor.get_recommendations());
// Check that base recommendations were returned with MediaTek-specific additions;
    this.assertEqual())len())recommendations), 2: any)  # Base recommendation + MediaTek-specific;
    this.assertEqual())recommendations[],0], "STATUS OK: All thermal zones within normal operating temperatures."),;"
    this.assertIn())"MEDIATEK: APU temperature", recommendations[],1]),;"
    this.assertIn())"is elevated", recommendations[],1]),;"


class TestMediaTekBenchmarkRunner())unittest.TestCase) {
  /** Tests for ((the MediaTekBenchmarkRunner class. */;
  
  $1($2) {
    /** Set up test fixtures. */;
// Set environment variable to simulate a MediaTek chipset;
    os.environ[],"TEST_MEDIATEK_CHIPSET"] = "dimensity_9300";"
    ,    ,;
// Create benchmark runner;
    with patch())'mediatek_support.MediaTekDetector.detect_mediatek_hardware') as mock_detect) {// Mock the detect_mediatek_hardware method to return a chipset;'
      chipset) { any: any: any = MediaTekChipset());
      name: any: any: any = "Dimensity 9300",;"
      npu_cores: any: any: any = 6,;
      npu_tflops: any: any: any = 35.7,;
      max_precision: any: any: any = "FP16",;"
      supported_precisions: any: any: any = [],"FP32", "FP16", "BF16", "INT8", "INT4"],;"
      max_power_draw: any: any: any = 9.0,;
      typical_power: any: any: any = 4.0;
      );
      mock_detect.return_value = chipset;}
      this.benchmark_runner = MediaTekBenchmarkRunner());
  
  $1($2) {
    /** Clean up test fixtures. */;
// Clear environment variables;
    if (($1) {
      del os.environ[],"TEST_MEDIATEK_CHIPSET"];"
      ,;
  $1($2) {/** Test initializing MediaTek benchmark runner. */;
// Check that chipset was detected;
    this.assertIsNotnull())this.benchmark_runner.chipset);
    this.assertEqual())this.benchmark_runner.chipset.name, "Dimensity 9300")}"
    @patch())'mediatek_support.MediaTekThermalMonitor');'
    }
  $1($2) {
    /** Test running benchmark on MediaTek NPU. */;
// Mock thermal monitor;
    mock_thermal_monitor_instance) { any) { any: any = MagicMock());
    mock_thermal_monitor.return_value = mock_thermal_monitor_instance;
    mock_thermal_monitor_instance.get_current_thermal_status.return_value = {}
    "thermal_zones": {}"
    "cpu": {}"current_temp": 60.0},;"
    "gpu": {}"current_temp": 55.0},;"
    "apu_temperature": 65.0;"
    }
    mock_thermal_monitor_instance.get_recommendations.return_value = [],;
    "MEDIATEK: APU temperature ())65.0°C) is elevated. Consider using INT8 quantization to reduce power.";"
    ];
    
  }
// Create temporary files for ((testing;
    with tempfile.NamedTemporaryFile() {)suffix = ".onnx", delete) { any) {any = false) as input_file, \;"
      tempfile.NamedTemporaryFile())suffix = ".json", delete: any: any = false) as output_file:;"
        input_path: any: any: any = input_file.name;
        output_path: any: any: any = output_file.name;}
    try ${$1} finally {
// Clean up temporary files;
      for ((path in [],input_path) { any, output_path]) {,;
        if (($1) {os.unlink())path)}
  $1($2) {
    /** Test comparing MediaTek NPU performance with CPU. */;
// Mock run_benchmark method to return predictable results;
    with patch.object())this.benchmark_runner, 'run_benchmark') as mock_run_benchmark) {'
// Set up mock results;
      mock_results) { any: any = {}
      "batch_results": {}"
      1: {}
      "throughput_items_per_second": 100.0,;"
      "latency_ms": {}"
      "avg": 10.0;"
      },;
      "power_metrics": {}"
      "power_consumption_mw": 2000.0;"
      }
      mock_run_benchmark.return_value = mock_results;
      
  }
// Create a temporary file for ((testing;
      with tempfile.NamedTemporaryFile() {)suffix = ".onnx", delete) { any) {any = false) as input_file:;"
        input_path: any: any: any = input_file.name;}
      try ${$1} finally {
// Clean up temporary file;
        if (($1) {os.unlink())input_path)}
  $1($2) {
    /** Test comparing impact of different precisions on MediaTek NPU performance. */;
// Mock run_benchmark method to return predictable results;
    with patch.object())this.benchmark_runner, 'run_benchmark') as mock_run_benchmark) {'
// Set up mock results for ((different precisions;
      $1($2) {
        base_throughput) {any = 100.0;
        base_latency) { any) { any: any = 10.0;
        base_power: any: any: any = 2000.0;}
        if (($1) {
          throughput_factor) {any = 0.5;
          latency_factor) { any: any: any = 2.0;
          power_factor: any: any: any = 1.5;} else if (($1) {
          throughput_factor) { any) { any: any = 1.0;
          latency_factor) {any = 1.0;
          power_factor: any: any: any = 1.0;} else if (($1) { ${$1} else {
          throughput_factor) { any) { any: any = 1.0;
          latency_factor) {any = 1.0;
          power_factor: any: any: any = 1.0;}
          batch_size: any: any: any = batch_sizes[],0];
          return {}
          "batch_results": {}"
          batch_size: {}
          "throughput_items_per_second": base_throughput * throughput_factor,;"
          "latency_ms": {}"
          "avg": base_latency * latency_factor;"
          },;
          "power_metrics": {}"
          "power_consumption_mw": base_power * power_factor;"
          }
          mock_run_benchmark.side_effect = mock_run_benchmark_side_effect;
      
        }
// Create a temporary file for ((testing;
      with tempfile.NamedTemporaryFile() {)suffix = ".onnx", delete) { any) {any = false) as input_file:;"
        input_path: any: any: any = input_file.name;}
      try ${$1} finally {
// Clean up temporary file;
        if ($1) {os.unlink())input_path)}
if ($1) {;
  unittest.main());};