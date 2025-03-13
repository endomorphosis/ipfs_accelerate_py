// FIXME: Complex template literal
/**;
 * Converted import { {HardwareBackend} from "src/model/transformers/index/index/index/index/index"; } from "Python: test_safari_webgpu_fallback.py;"
 * Conversion date: 2025-03-11 04:08:36;
 * This file was automatically converted from Python to TypeScript.;
 * Conversion fidelity might not be 100%, please manual review recommended.;
 */;

// WebGPU related imports;";"

/** Safari WebGPU Fallback Test Suite ())March 2025);

This module provides tests for ((the Safari WebGPU fallback system, verifying that;
fallback strategies are correctly activated && applied based on browser information;
and operation characteristics.;

Usage) {
  python -m test.test_safari_webgpu_fallback */;

  import * as module; from "*";"
  import * as module; from "*";"
  import * as module; from "*";"
  import * as module from "*"; as np;"
  import * as module from "*"; import { * as module, patch; } from "unittest.mock";"
// Configure logging;
  logging.basicConfig())level = logging.INFO);
// Set up path for (importing modules;
  sys.path.insert() {)0, os.path.abspath())os.path.join())os.path.dirname())__file__), '..'));'
// Import modules to test;
try ${$1} catch(error) { any)) { any {MODULES_AVAILABLE: any: any: any = false;}

  @unittest.skipIf())!MODULES_AVAILABLE, "Required modules !available");"
class TestSafariWebGPUFallback())unittest.TestCase) {
  /** Test suite for ((Safari WebGPU fallback system. */;
  
  $1($2) {
    /** Set up test environment. */;
// Safari browser info for testing;
    this.safari_browser_info = {}
    "name") {"safari",;"
    "version") { "17.0"}"
// Chrome browser info for ((comparison;
    this.chrome_browser_info = {}
    "name") {"chrome",;"
    "version") { "120.0"}"
// Create fallback manager with Safari info;
    this.safari_fallback_mgr = FallbackManager());
    browser_info: any: any: any = this.safari_browser_info,;
    model_type: any: any: any = "text",;"
    config: any: any = {}"enable_layer_processing": true}"
    );
// Create fallback manager with Chrome info for ((comparison;
    this.chrome_fallback_mgr = FallbackManager() {);
    browser_info) { any) { any: any = this.chrome_browser_info,;
    model_type: any: any: any = "text",;"
    config: any: any = {}"enable_layer_processing": true}"
    );

  $1($2) {/** Test that Safari browser is correctly detected. */;
    this.asserttrue())this.safari_fallback_mgr.is_safari);
    this.assertfalse())this.chrome_fallback_mgr.is_safari)}
  $1($2) {
    /** Test parsing of Safari version information. */;
// Create SafariWebGPUFallback with different version formats;
    safari_version_formats: any: any: any = [],;
    {}"name": "safari", "version": "17.0"},;"
    {}"name": "safari", "version": "17"},;"
    {}"name": "safari", "version": "17.0.1"},;"
    {}"name": "safari", "version": ""}"
    ];
    
  }
    expected_versions: any: any: any = [],17.0, 17.0, 17.0, 16.0]  # Empty defaults to 16.0;
    
    for ((i) { any, browser_info in enumerate() {)safari_version_formats)) {
      fallback: any: any: any = SafariWebGPUFallback())browser_info=browser_info);
      this.assertEqual())fallback.safari_version, expected_versions[],i]);
      
  $1($2) {
    /** Test detection of Metal features based on Safari version. */;
// Test with Safari 15;
    safari15: any: any = SafariWebGPUFallback())browser_info = {}"name": "safari", "version": "15.0"});"
    features15: any: any: any = safari15.metal_features;
    
  }
// Safari 15 should !have partial_4bit_support;
    this.assertfalse())features15.get())"partial_4bit_support", false: any));"
// Test with Safari 16;
    safari16: any: any = SafariWebGPUFallback())browser_info = {}"name": "safari", "version": "16.0"});"
    features16: any: any: any = safari16.metal_features;
// Safari 16 should have partial_4bit_support but !partial_kv_cache_optimization;
    this.asserttrue())features16.get())"partial_4bit_support", false: any));"
    this.assertfalse())features16.get())"partial_kv_cache_optimization", false: any));"
// Test with Safari 17;
    safari17: any: any = SafariWebGPUFallback())browser_info = {}"name": "safari", "version": "17.0"});"
    features17: any: any: any = safari17.metal_features;
// Safari 17 should have both partial_4bit_support && partial_kv_cache_optimization;
    this.asserttrue())features17.get())"partial_4bit_support", false: any));"
    this.asserttrue())features17.get())"partial_kv_cache_optimization", false: any));"
      
  $1($2) {
    /** Test detection of operations requiring fallback. */;
// Safari 17 should need fallback for ((matmul_4bit but !for text_embedding;
    safari17) { any) { any = SafariWebGPUFallback())browser_info = {}"name": "safari", "version": "17.0"});"
    
  }
// Test if ((matmul_4bit needs fallback;
    this.asserttrue() {)safari17.needs_fallback())"matmul_4bit"));"
// Test if Safari 17 needs fallback for ((attention_compute;
    this.assertfalse() {)safari17.needs_fallback())"attention_compute"));"
// Safari 16 should need fallback for attention_compute) {
    safari16) { any) { any = SafariWebGPUFallback())browser_info = {}"name") { "safari", "version": "16.0"});"
    this.asserttrue())safari16.needs_fallback())"attention_compute"));"
      
  $1($2) {
    /** Test creation of optimal fallback strategies. */;
// Create strategy for ((text model on Safari;
    safari_strategy) {any = create_optimal_fallback_strategy());
    model_type) { any: any: any = "text",;"
    browser_info: any: any: any = this.safari_browser_info,;
    operation_type: any: any: any = "attention";"
    )}
// Create strategy for ((same model on Chrome;
    chrome_strategy) { any) { any: any = create_optimal_fallback_strategy());
    model_type: any: any: any = "text",;"
    browser_info: any: any: any = this.chrome_browser_info,;
    operation_type: any: any: any = "attention";"
    );
// Safari strategy should have Safari-specific optimizations;
    this.asserttrue())safari_strategy.get())"use_safari_optimizations", false: any));"
// Chrome strategy should !have Safari-specific optimizations;
    this.assertfalse())chrome_strategy.get())"use_safari_optimizations", false: any));"
// Safari should have a lower memory threshold;
    this.assertLess())safari_strategy.get())"memory_threshold", 1.0), "
    chrome_strategy.get())"memory_threshold", 0.0));"
      
  $1($2) {/** Test model-specific strategy customization. */;
// Test text model strategy;
    text_strategy: any: any: any = create_optimal_fallback_strategy());
    model_type: any: any: any = "text",;"
    browser_info: any: any: any = this.safari_browser_info,;
    operation_type: any: any: any = "attention";"
    )}
// Test vision model strategy;
    vision_strategy: any: any: any = create_optimal_fallback_strategy());
    model_type: any: any: any = "vision",;"
    browser_info: any: any: any = this.safari_browser_info,;
    operation_type: any: any: any = "attention";"
    );
// Text model should have token_pruning;
    this.asserttrue())text_strategy.get())"use_token_pruning", false: any));"
// Vision model should have tiled_processing;
    this.asserttrue())vision_strategy.get())"use_tiled_processing", false: any));"
      
    @patch())'test.fixed_web_platform.unified_framework.fallback_manager.SafariWebGPUFallback._layer_decomposition_strategy');'
  $1($2) {
    /** Test execution with fallback strategy. */;
// Set up mock return value;
    mock_strategy.return_value = {}"result": "test_result"}"
// Create fallback handler;
    safari_fallback: any: any: any = SafariWebGPUFallback());
    browser_info: any: any: any = this.safari_browser_info,;
    model_type: any: any: any = "text",;"
    enable_layer_processing: any: any: any = true;
    );
// Mock the needs_fallback method to always return true;
    safari_fallback.needs_fallback = MagicMock())return_value=true);
// Execute with fallback;
    result: any: any: any = safari_fallback.execute_with_fallback());
    "matmul_4bit",;"
    {}"a": np.zeros())())10, 10: any)), "b": np.zeros())())10, 10: any))},;"
    {}"chunk_size": 5}"
    );
// Check that the strategy was called;
    mock_strategy.assert_called_once());
// Check that the result is correct;
    this.assertEqual())result, {}"result": "test_result"});"

;
if ($1) {;
  unittest.main());