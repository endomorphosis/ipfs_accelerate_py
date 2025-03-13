/**
 * WebGPU Optimizer Dashboard Generator
 * 
 * This script generates a comprehensive HTML dashboard from benchmark results.
 * It combines results from different benchmark types and creates interactive
 * visualizations to help analyze WebGPU optimization performance.
 */

const fs = require('fs');
const path = require('path');
const child_process = require('child_process');

// Configuration
const DEFAULT_CONFIG = {
  // Directory containing benchmark results
  resultsDir: path.join(__dirname, '..', 'benchmark_results'),
  
  // Output directory for the dashboard
  outputDir: path.join(__dirname, '..', 'dashboard_output'),
  
  // Template file path
  templatePath: path.join(__dirname, 'template.html'),
  
  // Dashboard output file
  outputFile: 'index.html',
  
  // Max number of results to keep in history
  maxHistoryResults: 50,
  
  // History file path
  historyFilePath: path.join(__dirname, '..', 'benchmark_history.json'),
  
  // Open dashboard after generation
  openDashboard: true,
  
  // Verbose logging
  verbose: false
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
    
    if (arg === '--no-open') {
      config.openDashboard = false;
      continue;
    }
    
    if (arg.startsWith('--results-dir=')) {
      config.resultsDir = arg.substring('--results-dir='.length);
      continue;
    }
    
    if (arg.startsWith('--output-dir=')) {
      config.outputDir = arg.substring('--output-dir='.length);
      continue;
    }
    
    if (arg.startsWith('--output-file=')) {
      config.outputFile = arg.substring('--output-file='.length);
      continue;
    }
    
    if (arg.startsWith('--template=')) {
      config.templatePath = arg.substring('--template='.length);
      continue;
    }
    
    if (arg.startsWith('--history-file=')) {
      config.historyFilePath = arg.substring('--history-file='.length);
      continue;
    }
    
    if (arg.startsWith('--max-history=')) {
      config.maxHistoryResults = parseInt(arg.substring('--max-history='.length), 10);
      continue;
    }
  }
  
  return config;
}

// Show help message
function showHelp() {
  console.log(`
WebGPU Optimizer Dashboard Generator

Usage:
  node generate_dashboard.js [options]

Options:
  --help, -h                Show this help message
  --verbose, -v             Enable verbose output
  --no-open                 Don't open dashboard after generation
  --results-dir=<dir>       Directory containing benchmark results
  --output-dir=<dir>        Output directory for the dashboard
  --output-file=<file>      Dashboard output file name
  --template=<file>         Template file path
  --history-file=<file>     History file path
  --max-history=<number>    Maximum number of historical results to keep

Examples:
  # Generate dashboard with default settings
  node generate_dashboard.js

  # Generate dashboard with custom results directory
  node generate_dashboard.js --results-dir=./my_benchmark_results

  # Generate dashboard with verbose output
  node generate_dashboard.js --verbose
`);
}

// Load JSON files from directory
function loadJsonFiles(directory) {
  const results = [];
  
  if (!fs.existsSync(directory)) {
    console.error(`Error: Results directory not found: ${directory}`);
    return results;
  }
  
  const files = fs.readdirSync(directory).filter(file => file.endsWith('.json'));
  
  files.forEach(file => {
    try {
      const filePath = path.join(directory, file);
      const content = fs.readFileSync(filePath, 'utf8');
      const data = JSON.parse(content);
      
      // Add file metadata
      data._file = {
        name: file,
        path: filePath,
        timestamp: fs.statSync(filePath).mtime.toISOString()
      };
      
      results.push(data);
    } catch (error) {
      console.error(`Error loading ${file}: ${error.message}`);
    }
  });
  
  return results;
}

// Load benchmark history
function loadHistory(filePath) {
  if (!fs.existsSync(filePath)) {
    return [];
  }
  
  try {
    const content = fs.readFileSync(filePath, 'utf8');
    return JSON.parse(content);
  } catch (error) {
    console.error(`Error loading history file: ${error.message}`);
    return [];
  }
}

// Save benchmark history
function saveHistory(history, filePath, maxHistory) {
  // Sort by timestamp (newest first) and limit to maxHistory
  const limitedHistory = history
    .sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp))
    .slice(0, maxHistory);
  
  try {
    const directory = path.dirname(filePath);
    if (!fs.existsSync(directory)) {
      fs.mkdirSync(directory, { recursive: true });
    }
    
    fs.writeFileSync(filePath, JSON.stringify(limitedHistory, null, 2), 'utf8');
  } catch (error) {
    console.error(`Error saving history file: ${error.message}`);
  }
}

// Update history with new results
function updateHistory(currentResults, history) {
  const timestamp = new Date().toISOString();
  
  // Create entry for current results
  const entry = {
    timestamp,
    results: currentResults
  };
  
  // Add version info if available
  try {
    const packageJsonPath = path.join(__dirname, '..', '..', '..', '..', 'package.json');
    const packageJson = JSON.parse(fs.readFileSync(packageJsonPath, 'utf8'));
    entry.version = packageJson.version;
  } catch (error) {
    entry.version = 'unknown';
  }
  
  // Add git commit info if available
  try {
    entry.commit = child_process.execSync('git rev-parse HEAD', { encoding: 'utf8' }).trim();
    entry.commitShort = entry.commit.substring(0, 7);
    
    const commitMessage = child_process.execSync(`git log -1 --pretty=%B ${entry.commit}`, { encoding: 'utf8' }).trim();
    entry.commitMessage = commitMessage;
  } catch (error) {
    entry.commit = 'unknown';
    entry.commitShort = 'unknown';
    entry.commitMessage = '';
  }
  
  // Add to history
  history.push(entry);
  
  return history;
}

// Process benchmark results
function processBenchmarks(benchmarkResults) {
  // Process and combine benchmark results
  const processedData = {
    lastUpdated: new Date().toISOString(),
    summary: {},
    operationFusion: {
      results: []
    },
    memoryLayout: {
      results: []
    },
    browserSpecific: {
      results: []
    },
    neuralPattern: {
      results: []
    },
    browserComparison: {},
    topImprovements: []
  };
  
  // Process each benchmark result file
  benchmarkResults.forEach(result => {
    // Process based on benchmark type
    if (result.benchmarkType === 'operation-fusion' || result._file?.name.includes('operation_fusion')) {
      processOperationFusionResults(result, processedData);
    } else if (result.benchmarkType === 'memory-layout' || result._file?.name.includes('memory_layout')) {
      processMemoryLayoutResults(result, processedData);
    } else if (result.benchmarkType === 'browser-specific' || result._file?.name.includes('browser_specific')) {
      processBrowserSpecificResults(result, processedData);
    } else if (result.benchmarkType === 'neural-pattern' || result._file?.name.includes('neural_network_pattern')) {
      processNeuralPatternResults(result, processedData);
    } else {
      // General benchmark results
      processGeneralBenchmarkResults(result, processedData);
    }
  });
  
  // Calculate summary statistics
  calculateSummaryStatistics(processedData);
  
  // Find top improvements
  findTopImprovements(processedData);
  
  return processedData;
}

// Process operation fusion benchmark results
function processOperationFusionResults(result, processedData) {
  if (!result.results) return;
  
  result.results.forEach(fusionResult => {
    processedData.operationFusion.results.push({
      operation: fusionResult.operation || fusionResult.operationName || 'Unknown',
      fusionPattern: fusionResult.fusionPattern || 'Unknown',
      shapeInfo: fusionResult.shapeInfo || fusionResult.shape?.join('x') || 'Unknown',
      unfusedTime: fusionResult.unfusedTime || fusionResult.standardTime || 0,
      fusedTime: fusionResult.fusedTime || fusionResult.optimizedTime || 0,
      speedup: fusionResult.speedup || (fusionResult.unfusedTime / fusionResult.fusedTime) || 1,
      memorySavings: fusionResult.memorySavings || 0,
      browser: fusionResult.browser || result.browser || 'Unknown'
    });
  });
}

// Process memory layout benchmark results
function processMemoryLayoutResults(result, processedData) {
  if (!result.results) return;
  
  result.results.forEach(layoutResult => {
    processedData.memoryLayout.results.push({
      operation: layoutResult.operation || layoutResult.operationName || 'Unknown',
      shape: layoutResult.shape?.join('x') || 'Unknown',
      rowMajorTime: layoutResult.rowMajorTime || 0,
      columnMajorTime: layoutResult.columnMajorTime || 0,
      optimalTime: layoutResult.optimalTime || 0,
      optimalLayout: layoutResult.optimalLayout || 'Unknown',
      speedupVsWorst: layoutResult.speedupVsWorst || 1,
      browser: layoutResult.browser || result.browser || 'Unknown'
    });
  });
}

// Process browser-specific benchmark results
function processBrowserSpecificResults(result, processedData) {
  if (!result.results) return;
  
  result.results.forEach(browserResult => {
    processedData.browserSpecific.results.push({
      operation: browserResult.operation || browserResult.operationName || 'Unknown',
      shape: browserResult.shape?.join('x') || 'Unknown',
      genericTime: browserResult.genericTime || 0,
      browserOptimizedTime: browserResult.browserOptimizedTime || 0,
      speedup: browserResult.speedup || (browserResult.genericTime / browserResult.browserOptimizedTime) || 1,
      detectedOptimizations: browserResult.detectedOptimizations || [],
      browser: browserResult.browser || result.browser || 'Unknown'
    });
  });
}

// Process neural network pattern recognition results
function processNeuralPatternResults(result, processedData) {
  if (!result.results) return;
  
  result.results.forEach(neuralResult => {
    processedData.neuralPattern.results.push({
      networkName: neuralResult.networkName || 'Unknown',
      patternName: neuralResult.patternName || 'Unknown',
      networkConfig: neuralResult.networkConfig || 'Unknown',
      standardTime: neuralResult.standardTime || 0,
      optimizedTime: neuralResult.optimizedTime || 0,
      speedup: neuralResult.speedup || (neuralResult.standardTime / neuralResult.optimizedTime) || 1,
      memorySavings: neuralResult.memorySavings || 0,
      patternsDetected: neuralResult.patternsDetected || [],
      browser: neuralResult.browser || result.browser || 'Unknown'
    });
  });
}

// Process general benchmark results
function processGeneralBenchmarkResults(result, processedData) {
  if (!result.results) return;
  
  result.results.forEach(benchmarkResult => {
    // Determine category based on result properties
    if (benchmarkResult.optimizedTime && benchmarkResult.standardTime) {
      processedData.topImprovements.push({
        operation: benchmarkResult.name || 'Unknown',
        optimization: determineOptimizationType(benchmarkResult),
        shapeConfig: benchmarkResult.shape?.join('x') || 'Unknown',
        browser: benchmarkResult.browser || result.browser || 'Unknown',
        standardTime: benchmarkResult.standardTime || 0,
        optimizedTime: benchmarkResult.optimizedTime || 0,
        speedup: benchmarkResult.speedup || (benchmarkResult.standardTime / benchmarkResult.optimizedTime) || 1,
        memorySavings: benchmarkResult.memorySavings || 0
      });
    }
  });
}

// Helper to determine optimization type from result
function determineOptimizationType(result) {
  if (result.fusionPattern) return 'Operation Fusion';
  if (result.optimalLayout) return 'Memory Layout';
  if (result.detectedOptimizations?.length > 0) return 'Browser-Specific';
  if (result.patternsDetected?.length > 0) return 'Neural Pattern';
  return 'General';
}

// Calculate summary statistics
function calculateSummaryStatistics(processedData) {
  const summary = {
    optimizationTypes: {
      fusion: { count: 0, totalSpeedup: 0, totalMemorySavings: 0 },
      memoryLayout: { count: 0, totalSpeedup: 0, totalMemorySavings: 0 },
      browserSpecific: { count: 0, totalSpeedup: 0, totalMemorySavings: 0 },
      neuralPattern: { count: 0, totalSpeedup: 0, totalMemorySavings: 0 }
    },
    browsers: {
      chrome: { count: 0, totalSpeedup: 0, totalMemorySavings: 0 },
      firefox: { count: 0, totalSpeedup: 0, totalMemorySavings: 0 },
      edge: { count: 0, totalSpeedup: 0, totalMemorySavings: 0 },
      other: { count: 0, totalSpeedup: 0, totalMemorySavings: 0 }
    },
    operationTypes: {},
    totalTests: 0,
    avgSpeedup: 0,
    avgMemorySavings: 0
  };
  
  // Process operation fusion results
  processedData.operationFusion.results.forEach(result => {
    summary.optimizationTypes.fusion.count++;
    summary.optimizationTypes.fusion.totalSpeedup += result.speedup || 1;
    summary.optimizationTypes.fusion.totalMemorySavings += result.memorySavings || 0;
    
    // Process by browser
    const browser = (result.browser || '').toLowerCase();
    if (browser === 'chrome' || browser === 'firefox' || browser === 'edge') {
      summary.browsers[browser].count++;
      summary.browsers[browser].totalSpeedup += result.speedup || 1;
      summary.browsers[browser].totalMemorySavings += result.memorySavings || 0;
    } else {
      summary.browsers.other.count++;
      summary.browsers.other.totalSpeedup += result.speedup || 1;
      summary.browsers.other.totalMemorySavings += result.memorySavings || 0;
    }
    
    // Process by operation type
    const operation = result.operation || 'Unknown';
    if (!summary.operationTypes[operation]) {
      summary.operationTypes[operation] = { count: 0, totalSpeedup: 0, totalMemorySavings: 0 };
    }
    summary.operationTypes[operation].count++;
    summary.operationTypes[operation].totalSpeedup += result.speedup || 1;
    summary.operationTypes[operation].totalMemorySavings += result.memorySavings || 0;
    
    summary.totalTests++;
  });
  
  // Process memory layout results
  processedData.memoryLayout.results.forEach(result => {
    summary.optimizationTypes.memoryLayout.count++;
    summary.optimizationTypes.memoryLayout.totalSpeedup += result.speedupVsWorst || 1;
    
    // Process by browser
    const browser = (result.browser || '').toLowerCase();
    if (browser === 'chrome' || browser === 'firefox' || browser === 'edge') {
      summary.browsers[browser].count++;
      summary.browsers[browser].totalSpeedup += result.speedupVsWorst || 1;
    } else {
      summary.browsers.other.count++;
      summary.browsers.other.totalSpeedup += result.speedupVsWorst || 1;
    }
    
    // Process by operation type
    const operation = result.operation || 'Unknown';
    if (!summary.operationTypes[operation]) {
      summary.operationTypes[operation] = { count: 0, totalSpeedup: 0, totalMemorySavings: 0 };
    }
    summary.operationTypes[operation].count++;
    summary.operationTypes[operation].totalSpeedup += result.speedupVsWorst || 1;
    
    summary.totalTests++;
  });
  
  // Process browser-specific results
  processedData.browserSpecific.results.forEach(result => {
    summary.optimizationTypes.browserSpecific.count++;
    summary.optimizationTypes.browserSpecific.totalSpeedup += result.speedup || 1;
    
    // Process by browser
    const browser = (result.browser || '').toLowerCase();
    if (browser === 'chrome' || browser === 'firefox' || browser === 'edge') {
      summary.browsers[browser].count++;
      summary.browsers[browser].totalSpeedup += result.speedup || 1;
    } else {
      summary.browsers.other.count++;
      summary.browsers.other.totalSpeedup += result.speedup || 1;
    }
    
    // Process by operation type
    const operation = result.operation || 'Unknown';
    if (!summary.operationTypes[operation]) {
      summary.operationTypes[operation] = { count: 0, totalSpeedup: 0, totalMemorySavings: 0 };
    }
    summary.operationTypes[operation].count++;
    summary.operationTypes[operation].totalSpeedup += result.speedup || 1;
    
    summary.totalTests++;
  });
  
  // Process neural pattern results
  processedData.neuralPattern.results.forEach(result => {
    summary.optimizationTypes.neuralPattern.count++;
    summary.optimizationTypes.neuralPattern.totalSpeedup += result.speedup || 1;
    summary.optimizationTypes.neuralPattern.totalMemorySavings += result.memorySavings || 0;
    
    // Process by browser
    const browser = (result.browser || '').toLowerCase();
    if (browser === 'chrome' || browser === 'firefox' || browser === 'edge') {
      summary.browsers[browser].count++;
      summary.browsers[browser].totalSpeedup += result.speedup || 1;
      summary.browsers[browser].totalMemorySavings += result.memorySavings || 0;
    } else {
      summary.browsers.other.count++;
      summary.browsers.other.totalSpeedup += result.speedup || 1;
      summary.browsers.other.totalMemorySavings += result.memorySavings || 0;
    }
    
    // Process by operation type
    const operation = result.networkName || 'Unknown';
    if (!summary.operationTypes[operation]) {
      summary.operationTypes[operation] = { count: 0, totalSpeedup: 0, totalMemorySavings: 0 };
    }
    summary.operationTypes[operation].count++;
    summary.operationTypes[operation].totalSpeedup += result.speedup || 1;
    summary.operationTypes[operation].totalMemorySavings += result.memorySavings || 0;
    
    summary.totalTests++;
  });
  
  // Calculate averages
  if (summary.totalTests > 0) {
    const totalSpeedup = 
      summary.optimizationTypes.fusion.totalSpeedup + 
      summary.optimizationTypes.memoryLayout.totalSpeedup + 
      summary.optimizationTypes.browserSpecific.totalSpeedup + 
      summary.optimizationTypes.neuralPattern.totalSpeedup;
    
    const totalMemorySavings = 
      summary.optimizationTypes.fusion.totalMemorySavings + 
      summary.optimizationTypes.neuralPattern.totalMemorySavings;
    
    summary.avgSpeedup = totalSpeedup / summary.totalTests;
    
    const memorySavingsTests = 
      summary.optimizationTypes.fusion.count + 
      summary.optimizationTypes.neuralPattern.count;
    
    if (memorySavingsTests > 0) {
      summary.avgMemorySavings = totalMemorySavings / memorySavingsTests;
    }
  }
  
  // Calculate average speedup by optimization type
  if (summary.optimizationTypes.fusion.count > 0) {
    summary.optimizationTypes.fusion.avgSpeedup = 
      summary.optimizationTypes.fusion.totalSpeedup / summary.optimizationTypes.fusion.count;
  }
  
  if (summary.optimizationTypes.memoryLayout.count > 0) {
    summary.optimizationTypes.memoryLayout.avgSpeedup = 
      summary.optimizationTypes.memoryLayout.totalSpeedup / summary.optimizationTypes.memoryLayout.count;
  }
  
  if (summary.optimizationTypes.browserSpecific.count > 0) {
    summary.optimizationTypes.browserSpecific.avgSpeedup = 
      summary.optimizationTypes.browserSpecific.totalSpeedup / summary.optimizationTypes.browserSpecific.count;
  }
  
  if (summary.optimizationTypes.neuralPattern.count > 0) {
    summary.optimizationTypes.neuralPattern.avgSpeedup = 
      summary.optimizationTypes.neuralPattern.totalSpeedup / summary.optimizationTypes.neuralPattern.count;
  }
  
  // Calculate average speedup by browser
  if (summary.browsers.chrome.count > 0) {
    summary.browsers.chrome.avgSpeedup = 
      summary.browsers.chrome.totalSpeedup / summary.browsers.chrome.count;
  }
  
  if (summary.browsers.firefox.count > 0) {
    summary.browsers.firefox.avgSpeedup = 
      summary.browsers.firefox.totalSpeedup / summary.browsers.firefox.count;
  }
  
  if (summary.browsers.edge.count > 0) {
    summary.browsers.edge.avgSpeedup = 
      summary.browsers.edge.totalSpeedup / summary.browsers.edge.count;
  }
  
  if (summary.browsers.other.count > 0) {
    summary.browsers.other.avgSpeedup = 
      summary.browsers.other.totalSpeedup / summary.browsers.other.count;
  }
  
  // Calculate average speedup by operation type
  Object.keys(summary.operationTypes).forEach(operation => {
    if (summary.operationTypes[operation].count > 0) {
      summary.operationTypes[operation].avgSpeedup = 
        summary.operationTypes[operation].totalSpeedup / summary.operationTypes[operation].count;
    }
  });
  
  processedData.summary = summary;
}

// Find top improvements
function findTopImprovements(processedData) {
  const allImprovements = [
    ...processedData.operationFusion.results.map(r => ({
      operation: r.operation,
      optimization: 'Operation Fusion',
      shapeConfig: r.shapeInfo,
      browser: r.browser,
      standardTime: r.unfusedTime,
      optimizedTime: r.fusedTime,
      speedup: r.speedup,
      memorySavings: r.memorySavings
    })),
    ...processedData.memoryLayout.results.map(r => ({
      operation: r.operation,
      optimization: 'Memory Layout',
      shapeConfig: r.shape,
      browser: r.browser,
      standardTime: Math.max(r.rowMajorTime, r.columnMajorTime),
      optimizedTime: r.optimalTime,
      speedup: r.speedupVsWorst,
      memorySavings: 0
    })),
    ...processedData.browserSpecific.results.map(r => ({
      operation: r.operation,
      optimization: 'Browser-Specific',
      shapeConfig: r.shape,
      browser: r.browser,
      standardTime: r.genericTime,
      optimizedTime: r.browserOptimizedTime,
      speedup: r.speedup,
      memorySavings: 0
    })),
    ...processedData.neuralPattern.results.map(r => ({
      operation: r.networkName,
      optimization: 'Neural Pattern',
      shapeConfig: r.networkConfig,
      browser: r.browser,
      standardTime: r.standardTime,
      optimizedTime: r.optimizedTime,
      speedup: r.speedup,
      memorySavings: r.memorySavings
    })),
    ...processedData.topImprovements
  ];
  
  // Sort by speedup (highest first)
  const topImprovements = allImprovements.sort((a, b) => b.speedup - a.speedup).slice(0, 10);
  
  processedData.topImprovements = topImprovements;
}

// Create dashboard HTML
function createDashboard(templatePath, data) {
  if (!fs.existsSync(templatePath)) {
    console.error(`Error: Template file not found: ${templatePath}`);
    return '';
  }
  
  try {
    const template = fs.readFileSync(templatePath, 'utf8');
    
    // Replace template placeholders
    let html = template
      .replace('{{LAST_UPDATED}}', new Date().toLocaleString())
      .replace('{{DASHBOARD_DATA}}', JSON.stringify(data, null, 2));
    
    // Add dashboard initialization code
    html = addDashboardInitScript(html, data);
    
    return html;
  } catch (error) {
    console.error(`Error creating dashboard: ${error.message}`);
    return '';
  }
}

// Add dashboard initialization script
function addDashboardInitScript(html, data) {
  // Create JavaScript for initializing charts and tables
  const initScript = `
function initDashboard(data) {
  // Initialize summary tab
  initSummaryTab(data);
  
  // Initialize operation fusion tab
  initOperationFusionTab(data);
  
  // Initialize memory layout tab
  initMemoryLayoutTab(data);
  
  // Initialize browser-specific tab
  initBrowserSpecificTab(data);
  
  // Initialize neural pattern tab
  initNeuralPatternTab(data);
  
  // Initialize browser comparison tab
  initBrowserComparisonTab(data);
  
  // Initialize history tab
  initHistoryTab(data);
  
  // Set up event handlers
  setupEventHandlers(data);
}

// Initialize summary tab
function initSummaryTab(data) {
  // Setup optimization summary chart
  const optimizationSummaryCtx = document.getElementById('optimizationSummaryChart').getContext('2d');
  new Chart(optimizationSummaryCtx, {
    type: 'bar',
    data: {
      labels: ['Operation Fusion', 'Memory Layout', 'Browser-Specific', 'Neural Network'],
      datasets: [{
        label: 'Average Speedup',
        data: [
          data.summary.optimizationTypes.fusion.avgSpeedup || 0,
          data.summary.optimizationTypes.memoryLayout.avgSpeedup || 0,
          data.summary.optimizationTypes.browserSpecific.avgSpeedup || 0,
          data.summary.optimizationTypes.neuralPattern.avgSpeedup || 0
        ],
        backgroundColor: [
          'rgba(76, 175, 80, 0.6)',
          'rgba(33, 150, 243, 0.6)',
          'rgba(255, 193, 7, 0.6)',
          'rgba(156, 39, 176, 0.6)'
        ],
        borderColor: [
          'rgb(76, 175, 80)',
          'rgb(33, 150, 243)',
          'rgb(255, 193, 7)',
          'rgb(156, 39, 176)'
        ],
        borderWidth: 1
      }]
    },
    options: {
      scales: {
        y: {
          beginAtZero: true,
          title: {
            display: true,
            text: 'Average Speedup Factor (x)'
          }
        }
      },
      plugins: {
        title: {
          display: true,
          text: 'Average Speedup by Optimization Type'
        },
        tooltip: {
          callbacks: {
            label: function(context) {
              return \`Speedup: \${context.parsed.y.toFixed(2)}x\`;
            },
            footer: function(tooltipItems) {
              const index = tooltipItems[0].dataIndex;
              const label = ['fusion', 'memoryLayout', 'browserSpecific', 'neuralPattern'][index];
              const count = data.summary.optimizationTypes[label].count;
              return \`Based on \${count} test(s)\`;
            }
          }
        }
      }
    }
  });
  
  // Setup memory savings chart
  const memorySavingsCtx = document.getElementById('memorySavingsChart').getContext('2d');
  new Chart(memorySavingsCtx, {
    type: 'bar',
    data: {
      labels: ['Operation Fusion', 'Neural Network'],
      datasets: [{
        label: 'Average Memory Savings',
        data: [
          (data.summary.optimizationTypes.fusion.avgMemorySavings || 0) * 100,
          (data.summary.optimizationTypes.neuralPattern.avgMemorySavings || 0) * 100
        ],
        backgroundColor: [
          'rgba(76, 175, 80, 0.6)',
          'rgba(156, 39, 176, 0.6)'
        ],
        borderColor: [
          'rgb(76, 175, 80)',
          'rgb(156, 39, 176)'
        ],
        borderWidth: 1
      }]
    },
    options: {
      scales: {
        y: {
          beginAtZero: true,
          title: {
            display: true,
            text: 'Average Memory Savings (%)'
          }
        }
      },
      plugins: {
        title: {
          display: true,
          text: 'Memory Savings by Optimization Type'
        },
        tooltip: {
          callbacks: {
            label: function(context) {
              return \`Memory Savings: \${context.parsed.y.toFixed(1)}%\`;
            },
            footer: function(tooltipItems) {
              const index = tooltipItems[0].dataIndex;
              const label = ['fusion', 'neuralPattern'][index];
              const count = data.summary.optimizationTypes[label].count;
              return \`Based on \${count} test(s)\`;
            }
          }
        }
      }
    }
  });
  
  // Setup browser comparison chart
  const browserComparisonCtx = document.getElementById('browserComparisonChart').getContext('2d');
  new Chart(browserComparisonCtx, {
    type: 'bar',
    data: {
      labels: ['Chrome', 'Firefox', 'Edge', 'Other'],
      datasets: [{
        label: 'Average Speedup',
        data: [
          data.summary.browsers.chrome.avgSpeedup || 0,
          data.summary.browsers.firefox.avgSpeedup || 0,
          data.summary.browsers.edge.avgSpeedup || 0,
          data.summary.browsers.other.avgSpeedup || 0
        ],
        backgroundColor: [
          'rgba(255, 87, 34, 0.6)',
          'rgba(33, 150, 243, 0.6)',
          'rgba(0, 150, 136, 0.6)',
          'rgba(158, 158, 158, 0.6)'
        ],
        borderColor: [
          'rgb(255, 87, 34)',
          'rgb(33, 150, 243)',
          'rgb(0, 150, 136)',
          'rgb(158, 158, 158)'
        ],
        borderWidth: 1
      }]
    },
    options: {
      scales: {
        y: {
          beginAtZero: true,
          title: {
            display: true,
            text: 'Average Speedup Factor (x)'
          }
        }
      },
      plugins: {
        title: {
          display: true,
          text: 'Average Speedup by Browser'
        },
        tooltip: {
          callbacks: {
            label: function(context) {
              return \`Speedup: \${context.parsed.y.toFixed(2)}x\`;
            },
            footer: function(tooltipItems) {
              const index = tooltipItems[0].dataIndex;
              const browser = ['chrome', 'firefox', 'edge', 'other'][index];
              const count = data.summary.browsers[browser].count;
              return \`Based on \${count} test(s)\`;
            }
          }
        }
      }
    }
  });
  
  // Setup operation type chart
  const operationTypeCtx = document.getElementById('operationTypeChart').getContext('2d');
  
  // Get top 5 operation types by speedup
  const operationTypes = Object.keys(data.summary.operationTypes)
    .map(op => ({
      name: op, 
      avgSpeedup: data.summary.operationTypes[op].avgSpeedup || 0,
      count: data.summary.operationTypes[op].count || 0
    }))
    .sort((a, b) => b.avgSpeedup - a.avgSpeedup)
    .slice(0, 5);
  
  new Chart(operationTypeCtx, {
    type: 'bar',
    data: {
      labels: operationTypes.map(op => op.name),
      datasets: [{
        label: 'Average Speedup',
        data: operationTypes.map(op => op.avgSpeedup),
        backgroundColor: 'rgba(63, 81, 181, 0.6)',
        borderColor: 'rgb(63, 81, 181)',
        borderWidth: 1
      }]
    },
    options: {
      scales: {
        y: {
          beginAtZero: true,
          title: {
            display: true,
            text: 'Average Speedup Factor (x)'
          }
        }
      },
      plugins: {
        title: {
          display: true,
          text: 'Top Operations by Performance Improvement'
        },
        tooltip: {
          callbacks: {
            label: function(context) {
              return \`Speedup: \${context.parsed.y.toFixed(2)}x\`;
            },
            footer: function(tooltipItems) {
              const index = tooltipItems[0].dataIndex;
              const count = operationTypes[index].count;
              return \`Based on \${count} test(s)\`;
            }
          }
        }
      }
    }
  });
  
  // Setup top improvements table
  const topImprovementsTable = document.getElementById('topImprovementsTable');
  topImprovementsTable.innerHTML = '';
  
  data.topImprovements.forEach(improvement => {
    const row = document.createElement('tr');
    
    // Determine speedup class
    const speedupClass = improvement.speedup > 2 ? 'speedup-high' : 
                        (improvement.speedup > 1.5 ? 'speedup-medium' : 'speedup-low');
    
    // Format memory savings
    const memorySavings = improvement.memorySavings
      ? \`\${(improvement.memorySavings * 100).toFixed(1)}%\`
      : 'N/A';
    
    row.innerHTML = \`
      <td>\${improvement.operation}</td>
      <td>\${improvement.optimization}</td>
      <td>\${improvement.shapeConfig}</td>
      <td>\${improvement.browser}</td>
      <td>\${improvement.standardTime.toFixed(2)}</td>
      <td>\${improvement.optimizedTime.toFixed(2)}</td>
      <td class="\${speedupClass}">\${improvement.speedup.toFixed(2)}x</td>
      <td>\${memorySavings}</td>
    \`;
    
    topImprovementsTable.appendChild(row);
  });
}

// Additional initialization functions will be implemented for other tabs
// ...

// Set up event handlers
function setupEventHandlers(data) {
  // Refresh button
  const refreshButton = document.getElementById('refreshButton');
  if (refreshButton) {
    refreshButton.addEventListener('click', function() {
      location.reload();
    });
  }
  
  // Show/hide relative values
  const showRelativeValues = document.getElementById('showRelativeValues');
  if (showRelativeValues) {
    showRelativeValues.addEventListener('change', function() {
      // Implementation will update table display
    });
  }
  
  // Filter handlers
  // ...
}`;

  // Add initialization script to HTML
  return html.replace('// Dashboard initialization will be added here', initScript);
}

// Save HTML to file
function saveHtml(html, outputDir, outputFile) {
  try {
    if (!fs.existsSync(outputDir)) {
      fs.mkdirSync(outputDir, { recursive: true });
    }
    
    const outputPath = path.join(outputDir, outputFile);
    fs.writeFileSync(outputPath, html, 'utf8');
    
    return outputPath;
  } catch (error) {
    console.error(`Error saving HTML file: ${error.message}`);
    return null;
  }
}

// Copy assets to output directory
function copyAssets(outputDir) {
  const assetsDir = path.join(__dirname, 'assets');
  
  if (!fs.existsSync(assetsDir)) {
    // No assets directory, nothing to copy
    return;
  }
  
  const targetDir = path.join(outputDir, 'assets');
  
  if (!fs.existsSync(targetDir)) {
    fs.mkdirSync(targetDir, { recursive: true });
  }
  
  try {
    const assets = fs.readdirSync(assetsDir);
    
    assets.forEach(asset => {
      const sourcePath = path.join(assetsDir, asset);
      const targetPath = path.join(targetDir, asset);
      
      fs.copyFileSync(sourcePath, targetPath);
    });
  } catch (error) {
    console.error(`Error copying assets: ${error.message}`);
  }
}

// Open dashboard in browser
function openDashboard(filePath) {
  try {
    const url = `file://${filePath}`;
    
    // Different open commands based on platform
    switch (process.platform) {
      case 'darwin':
        child_process.exec(`open "${url}"`);
        break;
      case 'win32':
        child_process.exec(`start "" "${url}"`);
        break;
      default:
        child_process.exec(`xdg-open "${url}"`);
        break;
    }
  } catch (error) {
    console.error(`Error opening dashboard: ${error.message}`);
  }
}

// Main function
async function main() {
  const config = parseArgs();
  
  if (config.verbose) {
    console.log('Configuration:');
    console.log(JSON.stringify(config, null, 2));
  }
  
  // Load benchmark results
  console.log(`Loading benchmark results from ${config.resultsDir}...`);
  const benchmarkResults = loadJsonFiles(config.resultsDir);
  
  if (benchmarkResults.length === 0) {
    console.error('No benchmark results found. Run benchmarks first.');
    process.exit(1);
  }
  
  console.log(`Found ${benchmarkResults.length} benchmark result files.`);
  
  // Process benchmark results
  console.log('Processing benchmark results...');
  const processedData = processBenchmarks(benchmarkResults);
  
  // Load and update history
  console.log('Updating benchmark history...');
  let history = loadHistory(config.historyFilePath);
  history = updateHistory(processedData, history);
  saveHistory(history, config.historyFilePath, config.maxHistoryResults);
  
  // Add history to processed data
  processedData.history = history;
  
  // Create dashboard HTML
  console.log('Creating dashboard HTML...');
  const html = createDashboard(config.templatePath, processedData);
  
  if (!html) {
    console.error('Failed to create dashboard HTML.');
    process.exit(1);
  }
  
  // Save HTML to file
  console.log(`Saving dashboard to ${config.outputDir}/${config.outputFile}...`);
  const outputPath = saveHtml(html, config.outputDir, config.outputFile);
  
  if (!outputPath) {
    console.error('Failed to save dashboard HTML.');
    process.exit(1);
  }
  
  // Copy assets
  console.log('Copying assets...');
  copyAssets(config.outputDir);
  
  console.log(`Dashboard created at: ${outputPath}`);
  
  // Open dashboard in browser
  if (config.openDashboard) {
    console.log('Opening dashboard in browser...');
    openDashboard(outputPath);
  }
  
  console.log('Done!');
}

// Run main function
main().catch(error => {
  console.error('Unhandled error:', error);
  process.exit(1);
});