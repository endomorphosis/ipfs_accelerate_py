/**
 * WebGPU Optimizer Comprehensive Benchmark Runner
 * 
 * This script runs all WebGPU optimizer benchmarks sequentially and generates
 * a comprehensive HTML dashboard with the results.
 */

const fs = require('fs');
const path = require('path');
const child_process = require('child_process');

// Configuration
const DEFAULT_CONFIG = {
  // Directory to save benchmark results
  outputDir: path.join(__dirname, 'benchmark_results'),
  
  // Dashboard output directory
  dashboardDir: path.join(__dirname, 'dashboard_output'),
  
  // Benchmark scripts
  benchmarkScripts: [
    {
      name: 'General Benchmarks',
      script: path.join(__dirname, 'test_webgpu_optimizer_benchmark.ts'),
      type: 'general'
    },
    {
      name: 'Memory Layout Optimization',
      script: path.join(__dirname, 'test_memory_layout_optimization.ts'),
      type: 'memory-layout'
    },
    {
      name: 'Browser-Specific Optimizations',
      script: path.join(__dirname, 'test_browser_specific_optimizations.ts'),
      type: 'browser-specific'
    },
    {
      name: 'Operation Fusion',
      script: path.join(__dirname, 'test_operation_fusion.ts'),
      type: 'operation-fusion'
    },
    {
      name: 'Neural Network Pattern Recognition',
      script: path.join(__dirname, 'test_neural_network_pattern_recognition.ts'),
      type: 'neural-pattern'
    }
  ],
  
  // Number of iterations for each benchmark
  iterations: 5,
  
  // Number of warmup iterations
  warmupIterations: 2,
  
  // Whether to run correctness tests
  runCorrectnessTests: true,
  
  // Whether to generate the dashboard
  generateDashboard: true,
  
  // Whether to open the dashboard
  openDashboard: true,
  
  // Verbose logging
  verbose: false,
  
  // Whether to simulate browsers (for CI environments)
  simulateBrowsers: true,
  
  // Browser types to include (for real browser testing)
  browsers: ['chrome'],
  
  // Maximum execution time per benchmark in seconds
  maxExecutionTime: 600
};

// Command line arguments
function parseArgs() {
  const args = process.argv.slice(2);
  const config = { ...DEFAULT_CONFIG };
  
  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    
    if (arg === '--help' || arg === '-h') {
      showHelp();
      process.exit(0);
    }
    
    if (arg === '--verbose' || arg === '-v') {
      config.verbose = true;
      continue;
    }
    
    if (arg === '--no-dashboard') {
      config.generateDashboard = false;
      continue;
    }
    
    if (arg === '--no-open') {
      config.openDashboard = false;
      continue;
    }
    
    if (arg === '--no-correctness') {
      config.runCorrectnessTests = false;
      continue;
    }
    
    if (arg === '--no-simulate') {
      config.simulateBrowsers = false;
      continue;
    }
    
    if (arg.startsWith('--output-dir=')) {
      config.outputDir = arg.substring('--output-dir='.length);
      continue;
    }
    
    if (arg.startsWith('--dashboard-dir=')) {
      config.dashboardDir = arg.substring('--dashboard-dir='.length);
      continue;
    }
    
    if (arg.startsWith('--iterations=')) {
      config.iterations = parseInt(arg.substring('--iterations='.length), 10);
      continue;
    }
    
    if (arg.startsWith('--warmup=')) {
      config.warmupIterations = parseInt(arg.substring('--warmup='.length), 10);
      continue;
    }
    
    if (arg.startsWith('--timeout=')) {
      config.maxExecutionTime = parseInt(arg.substring('--timeout='.length), 10);
      continue;
    }
    
    if (arg.startsWith('--browsers=')) {
      config.browsers = arg.substring('--browsers='.length).split(',');
      continue;
    }
    
    if (arg.startsWith('--only=')) {
      const types = arg.substring('--only='.length).split(',');
      config.benchmarkScripts = config.benchmarkScripts.filter(script => 
        types.some(type => script.type.includes(type))
      );
      continue;
    }
  }
  
  return config;
}

// Show help message
function showHelp() {
  console.log(`
WebGPU Optimizer Comprehensive Benchmark Runner

Usage:
  node run_comprehensive_benchmarks.js [options]

Options:
  --help, -h                Show this help message
  --verbose, -v             Enable verbose output
  --no-dashboard            Don't generate dashboard
  --no-open                 Don't open dashboard after generation
  --no-correctness          Skip correctness tests
  --no-simulate             Don't simulate browsers (use real browsers)
  --output-dir=<dir>        Directory to save benchmark results
  --dashboard-dir=<dir>     Dashboard output directory
  --iterations=<number>     Number of iterations for each benchmark
  --warmup=<number>         Number of warmup iterations
  --timeout=<seconds>       Maximum execution time per benchmark
  --browsers=<list>         Browsers to use (comma-separated: chrome,firefox,edge)
  --only=<types>            Only run specific benchmark types (comma-separated: 
                            general,memory-layout,browser-specific,operation-fusion,neural-pattern)

Examples:
  # Run all benchmarks with default settings
  node run_comprehensive_benchmarks.js

  # Run only memory layout and operation fusion benchmarks
  node run_comprehensive_benchmarks.js --only=memory-layout,operation-fusion

  # Run with more iterations for more stable results
  node run_comprehensive_benchmarks.js --iterations=10 --warmup=5

  # Run with real Chrome browser (no simulation)
  node run_comprehensive_benchmarks.js --no-simulate --browsers=chrome
`);
}

// Create output directory
function createOutputDir(dir) {
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }
}

// Run a command with timeout
function runCommand(command, options = {}) {
  return new Promise((resolve, reject) => {
    const timeout = options.timeout * 1000 || 10 * 60 * 1000; // default 10 min
    const process = child_process.exec(command, options, (error, stdout, stderr) => {
      if (error) {
        reject({ error, stdout, stderr });
      } else {
        resolve({ stdout, stderr });
      }
    });
    
    // Set timeout
    const timer = setTimeout(() => {
      process.kill();
      reject({ error: new Error(`Command timed out after ${timeout / 1000} seconds`), stdout: '', stderr: '' });
    }, timeout);
    
    // Clear timeout when process exits
    process.on('exit', () => {
      clearTimeout(timer);
    });
  });
}

// Run correctness tests
async function runCorrectnessTests(config) {
  console.log('\n=== Running Correctness Tests ===\n');
  
  try {
    const command = `npx jest ${path.join(__dirname, 'test_optimizer_correctness.ts')} --config ${path.join(__dirname, 'jest.config.js')} --testTimeout=${config.maxExecutionTime * 1000}`;
    
    console.log(`Executing: ${command}`);
    
    const { stdout, stderr } = await runCommand(command, {
      timeout: config.maxExecutionTime,
      env: {
        ...process.env,
        VERBOSE: config.verbose ? 'true' : 'false'
      }
    });
    
    if (config.verbose) {
      console.log('Stdout:', stdout);
      if (stderr) console.error('Stderr:', stderr);
    }
    
    console.log('Correctness tests completed successfully.');
    return true;
  } catch (error) {
    console.error('Correctness tests failed:');
    console.error(error.error ? error.error.message : 'Unknown error');
    if (config.verbose) {
      console.log('Stdout:', error.stdout);
      console.error('Stderr:', error.stderr);
    }
    return false;
  }
}

// Run a benchmark script
async function runBenchmark(script, config) {
  console.log(`\n=== Running ${script.name} ===\n`);
  
  try {
    // Environment variables for the benchmark
    const env = {
      ...process.env,
      BENCHMARK_TYPE: script.type,
      BENCHMARK_OUTPUT_DIR: config.outputDir,
      BENCHMARK_ITERATIONS: config.iterations.toString(),
      BENCHMARK_WARMUP_ITERATIONS: config.warmupIterations.toString(),
      BENCHMARK_SIMULATE_BROWSERS: config.simulateBrowsers ? 'true' : 'false',
      BENCHMARK_BROWSERS: config.browsers.join(','),
      VERBOSE: config.verbose ? 'true' : 'false'
    };
    
    // Command to run the benchmark
    const command = `npx ts-node ${script.script}`;
    
    console.log(`Executing: ${command}`);
    
    const { stdout, stderr } = await runCommand(command, {
      timeout: config.maxExecutionTime,
      env
    });
    
    if (config.verbose) {
      console.log('Stdout:', stdout);
      if (stderr) console.error('Stderr:', stderr);
    }
    
    // Save benchmark results
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const outputFile = path.join(
      config.outputDir, 
      `${script.type}_${timestamp}.json`
    );
    
    // Try to extract JSON results from stdout
    let results = {};
    try {
      // Find JSON data in stdout
      const jsonMatch = stdout.match(/BENCHMARK_RESULTS_BEGIN(.*?)BENCHMARK_RESULTS_END/s);
      if (jsonMatch && jsonMatch[1]) {
        results = JSON.parse(jsonMatch[1]);
      }
    } catch (error) {
      console.error(`Failed to parse benchmark results: ${error.message}`);
    }
    
    // If we couldn't extract results, create basic results
    if (Object.keys(results).length === 0) {
      results = {
        benchmarkType: script.type,
        name: script.name,
        timestamp,
        results: []
      };
    }
    
    // Save results to file
    fs.writeFileSync(outputFile, JSON.stringify(results, null, 2), 'utf8');
    
    console.log(`Benchmark ${script.name} completed successfully.`);
    console.log(`Results saved to: ${outputFile}`);
    
    return true;
  } catch (error) {
    console.error(`Benchmark ${script.name} failed:`);
    console.error(error.error ? error.error.message : 'Unknown error');
    if (config.verbose) {
      console.log('Stdout:', error.stdout);
      console.error('Stderr:', error.stderr);
    }
    return false;
  }
}

// Generate dashboard
async function generateDashboard(config) {
  console.log('\n=== Generating Dashboard ===\n');
  
  try {
    const dashboardScript = path.join(__dirname, 'dashboard', 'generate_dashboard.js');
    
    // Command to generate dashboard
    const command = `node ${dashboardScript} --results-dir=${config.outputDir} --output-dir=${config.dashboardDir} ${config.openDashboard ? '' : '--no-open'} ${config.verbose ? '--verbose' : ''}`;
    
    console.log(`Executing: ${command}`);
    
    const { stdout, stderr } = await runCommand(command, {
      timeout: 60 // 1 minute timeout for dashboard generation
    });
    
    if (config.verbose) {
      console.log('Stdout:', stdout);
      if (stderr) console.error('Stderr:', stderr);
    }
    
    console.log('Dashboard generated successfully.');
    return true;
  } catch (error) {
    console.error('Dashboard generation failed:');
    console.error(error.error ? error.error.message : 'Unknown error');
    if (config.verbose) {
      console.log('Stdout:', error.stdout);
      console.error('Stderr:', error.stderr);
    }
    return false;
  }
}

// Main function
async function main() {
  const config = parseArgs();
  
  if (config.verbose) {
    console.log('Configuration:');
    console.log(JSON.stringify(config, null, 2));
  }
  
  // Create output directories
  createOutputDir(config.outputDir);
  createOutputDir(config.dashboardDir);
  
  console.log('\n=== WebGPU Optimizer Comprehensive Benchmark Runner ===\n');
  console.log(`Output directory: ${config.outputDir}`);
  console.log(`Dashboard directory: ${config.dashboardDir}`);
  console.log(`Iterations: ${config.iterations}`);
  console.log(`Warmup iterations: ${config.warmupIterations}`);
  console.log(`Browser simulation: ${config.simulateBrowsers ? 'Enabled' : 'Disabled'}`);
  if (!config.simulateBrowsers) {
    console.log(`Browsers: ${config.browsers.join(', ')}`);
  }
  console.log(`Benchmarks to run: ${config.benchmarkScripts.map(s => s.name).join(', ')}`);
  
  // Run correctness tests if enabled
  let correctnessSuccess = true;
  if (config.runCorrectnessTests) {
    correctnessSuccess = await runCorrectnessTests(config);
  }
  
  // Run benchmarks
  const benchmarkResults = [];
  for (const script of config.benchmarkScripts) {
    const success = await runBenchmark(script, config);
    benchmarkResults.push({ script, success });
  }
  
  // Print summary
  console.log('\n=== Benchmark Summary ===\n');
  console.log(`Correctness tests: ${config.runCorrectnessTests ? (correctnessSuccess ? 'Passed' : 'Failed') : 'Skipped'}`);
  console.log('Benchmarks:');
  benchmarkResults.forEach(({ script, success }) => {
    console.log(`  ${script.name}: ${success ? 'Success' : 'Failed'}`);
  });
  
  // Generate dashboard if enabled
  if (config.generateDashboard) {
    await generateDashboard(config);
  }
  
  console.log('\nAll tasks completed.');
}

// Run main function
main().catch(error => {
  console.error('Unhandled error:', error);
  process.exit(1);
});