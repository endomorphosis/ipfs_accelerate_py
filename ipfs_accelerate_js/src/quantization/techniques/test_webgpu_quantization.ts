/**
 * Converted from Python: test_webgpu_quantization.py
 * Conversion date: 2025-03-11 04:08:37
 * This file was automatically converted from Python to TypeScript.
 * Conversion fidelity might not be 100%, please manual review recommended.
 */

// WebGPU related imports
import { HardwareBackend } from "../hardware_abstraction";


export interface Props {
  headless: options;
  headless: options;
  headless: options;
  html_path: logger;
  headless: try;
  driver: try;
}

#!/usr/bin/env python3
"""
WebGPU Quantization Test - Real browser implementation for WebGPU with quantization

This script tests WebGPU hardware acceleration with different quantization levels
using real Chrome/Edge/Firefox browsers. It tests multiple precision levels ())16-bit,
8-bit, 4-bit, 2-bit) && mixed precision options to provide comprehensive performance
and memory usage data.

Usage:
  python test_webgpu_quantization.py --model prajjwal1/bert-tiny --browser chrome
  python test_webgpu_quantization.py --model prajjwal1/bert-tiny --browser firefox --bits 4
  python test_webgpu_quantization.py --model prajjwal1/bert-tiny --browser chrome --bits 4 --mixed-precision
  python test_webgpu_quantization.py --run-all --output-dir webgpu_results
  """

  import * as $1
  import * as $1
  import * as $1
  import * as $1
  import * as $1
  import * as $1
  import * as $1
  import * as $1
  import ${$1} from "$1"
  import ${$1} from "$1"

# Setup logging
  logging.basicConfig())level=logging.INFO, format='%())asctime)s - %())levelname)s - %())message)s')
  logger = logging.getLogger())__name__)

try {
  import ${$1} from "$1"
  from selenium.webdriver.chrome.service import * as $1 as ChromeService
  from selenium.webdriver.firefox.service import * as $1 as FirefoxService
  from selenium.webdriver.edge.service import * as $1 as EdgeService
  from selenium.webdriver.chrome.options import * as $1 as ChromeOptions
  from selenium.webdriver.firefox.options import * as $1 as FirefoxOptions
  from selenium.webdriver.edge.options import * as $1 as EdgeOptions
  from selenium.webdriver.common.by import * as $1
  from selenium.webdriver.support.ui import * as $1
  from selenium.webdriver.support import * as $1 as EC
  HAS_SELENIUM = true
} catch($2: $1) {
  logger.error())"selenium package is required. Install with: pip install selenium")
  HAS_SELENIUM = false

}
# HTML template for WebGPU testing
}
  WEBGPU_TEST_HTML = """
  <!DOCTYPE html>
  <html lang="en">
  <head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>WebGPU Quantization Test</title>
  <style>
  body {}}}}}}}}}}}}}}}}}}}}}}}}}}} font-family: Arial, sans-serif; padding: 20px; }
    #log {}}}}}}}}}}}}}}}}}}}}}}}}}}} height: 300px; overflow-y: scroll; border: 1px solid #ccc; padding: 10px; margin-bottom: 20px; }
  .success {}}}}}}}}}}}}}}}}}}}}}}}}}}} color: green; font-weight: bold; }
  .error {}}}}}}}}}}}}}}}}}}}}}}}}}}} color: red; font-weight: bold; }
  .warning {}}}}}}}}}}}}}}}}}}}}}}}}}}} color: orange; font-weight: bold; }
  pre {}}}}}}}}}}}}}}}}}}}}}}}}}}} background-color: #f5f5f5; padding: 10px; border-radius: 5px; }
  </style>
  </head>
  <body>
  <h1>WebGPU Quantization Test</h1>
  <div id="status">
  <p>WebGPU Support: <span id="webgpu-support">Checking...</span></p>
  <p>Adapter Info: <span id="adapter-info">Unknown</span></p>
  <p>Precision: <span id="precision-level">Default</span></p>
  </div>
  <div id="log"></div>
  <div>
  <h2>Results:</h2>
  <pre id="results">Running test...</pre>
  </div>
  
  <script type="module">
  // Log function
  function log())message, className) {}}}}}}}}}}}}}}}}}}}}}}}}}}}
  const logElem = document.getElementById())'log');
  const entry = document.createElement())'div');
  if ())className) entry.className = className;
  entry.textContent = `[],${}}}}}}}}}}}}}}}}}}}}}}}}}}}new Date())).toLocaleTimeString()))}] ${}}}}}}}}}}}}}}}}}}}}}}}}}}}message}`;,
  logElem.appendChild())entry);
  logElem.scrollTop = logElem.scrollHeight;
  console.log())message);
  }
    
  // Check WebGPU support
  async function checkWebGPU())) {}}}}}}}}}}}}}}}}}}}}}}}}}}}
  const supportElem = document.getElementById())'webgpu-support');
  const adapterElem = document.getElementById())'adapter-info');
      
  if ())!())'gpu' in navigator)) {}}}}}}}}}}}}}}}}}}}}}}}}}}}
  supportElem.textContent = 'Not Supported';
  supportElem.className = 'error';
  log())'WebGPU is !supported in this browser', 'error');
  return false;
  }
      
  try {}}}}}}}}}}}}}}}}}}}}}}}}}}}
  // Request adapter
  const adapter = await navigator.gpu.requestAdapter()));
  if ())!adapter) {}}}}}}}}}}}}}}}}}}}}}}}}}}}
  supportElem.textContent = 'Adapter !available';
  supportElem.className = 'error';
  log())'WebGPU adapter !available', 'error');
  return false;
  }
        
  // Get adapter info
  const adapterInfo = await adapter.requestAdapterInfo()));
  adapterElem.textContent = `${}}}}}}}}}}}}}}}}}}}}}}}}}}}adapterInfo.vendor} - ${}}}}}}}}}}}}}}}}}}}}}}}}}}}adapterInfo.architecture || 'Unknown'}`;
        
  // Request device
  const device = await adapter.requestDevice()));
  if ())!device) {}}}}}}}}}}}}}}}}}}}}}}}}}}}
  supportElem.textContent = 'Device !available';
  supportElem.className = 'error';
  log())'WebGPU device !available', 'error');
return false;
}
        
supportElem.textContent = 'Supported';
        supportElem.className = 'success';:
          log())`WebGPU is available with adapter: ${}}}}}}}}}}}}}}}}}}}}}}}}}}}adapterInfo.vendor} - ${}}}}}}}}}}}}}}}}}}}}}}}}}}}adapterInfo.architecture || 'Unknown'}`, 'success');
        
          // Store for later use
          window.webgpuAdapter = adapter;
          window.webgpuDevice = device;
          window.adapterInfo = adapterInfo;
return true;
} catch ())e) {}}}}}}}}}}}}}}}}}}}}}}}}}}}
supportElem.textContent = `Error: ${}}}}}}}}}}}}}}}}}}}}}}}}}}}e.message}`;
supportElem.className = 'error';
log())`WebGPU error: ${}}}}}}}}}}}}}}}}}}}}}}}}}}}e.message}`, 'error');
return false;
}
}
    
// Run test with the given model
async function runTest())modelId, bitPrecision, mixedPrecision) {}}}}}}}}}}}}}}}}}}}}}}}}}}}
const resultsElem = document.getElementById())'results');
const precisionElem = document.getElementById())'precision-level');
      
// Update precision display
let precisionText = `${}}}}}}}}}}}}}}}}}}}}}}}}}}}bitPrecision}-bit`;
if ())mixedPrecision) {}}}}}}}}}}}}}}}}}}}}}}}}}}}
precisionText += " mixed precision";
}
precisionElem.textContent = precisionText;
      :
        resultsElem.textContent = `Loading model: ${}}}}}}}}}}}}}}}}}}}}}}}}}}}modelId} with ${}}}}}}}}}}}}}}}}}}}}}}}}}}}precisionText}...`;
      
        try {}}}}}}}}}}}}}}}}}}}}}}}}}}}
        // Check WebGPU support first
        const webgpuSupported = await checkWebGPU()));
        if ())!webgpuSupported) {}}}}}}}}}}}}}}}}}}}}}}}}}}}
        resultsElem.textContent = 'WebGPU is !supported in this browser';
return;
}
        
// Import transformers.js
        log())'Loading transformers.js library...');:
          const {}}}}}}}}}}}}}}}}}}}}}}}}}}} pipeline, env } = await import())'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.0');
          log())'transformers.js library loaded successfully', 'success');
        
          // Configure precision if specified
        if ($1) {
          log())`Setting quantization to ${}}}}}}}}}}}}}}}}}}}}}}}}}}}bitPrecision}-bit${}}}}}}}}}}}}}}}}}}}}}}}}}}}mixedPrecision ? ' mixed precision' : ''}`, 'warning');
          
        }
          // Set quantization configuration
          env.USE_INT8 = bitPrecision <= 8;
          env.USE_INT4 = bitPrecision <= 4;
          env.USE_INT2 = bitPrecision <= 2;
          
          // Mixed precision settings
          if ())mixedPrecision) {}}}}}}}}}}}}}}}}}}}}}}}}}}}
          env.MIXED_PRECISION = true;
          log())'Mixed precision enabled for attention layers', 'warning');
          }
          
          // Log current settings:
            log())`Quantization config: INT8=${}}}}}}}}}}}}}}}}}}}}}}}}}}}env.USE_INT8}, INT4=${}}}}}}}}}}}}}}}}}}}}}}}}}}}env.USE_INT4}, INT2=${}}}}}}}}}}}}}}}}}}}}}}}}}}}env.USE_INT2}, MIXED=${}}}}}}}}}}}}}}}}}}}}}}}}}}}env.MIXED_PRECISION}`, 'warning');
          
            // Estimate memory savings
            const memoryReduction = {}}}}}}}}}}}}}}}}}}}}}}}}}}}
            8: '50%', // 16-bit → 8-bit
            4: '75%', // 16-bit → 4-bit
            2: '87.5%' // 16-bit → 2-bit
            };
            log())`Estimated memory reduction: ${}}}}}}}}}}}}}}}}}}}}}}}}}}}memoryReduction[],bitPrecision]}`, 'success');,
            }
        
            // Initialize the model
            log())`Loading model: ${}}}}}}}}}}}}}}}}}}}}}}}}}}}modelId}...`);
            const startTime = performance.now()));
        
            // For text models, use feature-extraction
            const pipe = await pipeline())'feature-extraction', modelId, {}}}}}}}}}}}}}}}}}}}}}}}}}}}
            backend: 'webgpu',
            quantized: bitPrecision < 16,
            revision: 'default'
            });
        
            const loadTime = performance.now())) - startTime;
            log())`Model loaded in ${}}}}}}}}}}}}}}}}}}}}}}}}}}}loadTime.toFixed())2)}ms`, 'success');
        
            // Run inference
            log())'Running inference...');
            const inferenceStart = performance.now()));
        
            // Run multiple inferences to get better timing data
            const numRuns = 5;
            let totalTime = 0;
            let result;
        
            for ())let i = 0; i < numRuns; i++) {}}}}}}}}}}}}}}}}}}}}}}}}}}}
            const runStart = performance.now()));
            result = await pipe())'This is a test input for WebGPU inference with quantization testing.');
            const runTime = performance.now())) - runStart;
            totalTime += runTime;
            log())`Run ${}}}}}}}}}}}}}}}}}}}}}}}}}}}i+1}: ${}}}}}}}}}}}}}}}}}}}}}}}}}}}runTime.toFixed())2)}ms`);
            }
        
            const averageInferenceTime = totalTime / numRuns;
            const inferenceTime = performance.now())) - inferenceStart;
        
            log())`All inference runs completed in ${}}}}}}}}}}}}}}}}}}}}}}}}}}}inferenceTime.toFixed())2)}ms ())avg: ${}}}}}}}}}}}}}}}}}}}}}}}}}}}averageInferenceTime.toFixed())2)}ms)`, 'success');
        
            // Get memory usage if possible
            let memoryUsage = null;
            try {}}}}}}}}}}}}}}}}}}}}}}}}}}}
            if ())performance.memory) {}}}}}}}}}}}}}}}}}}}}}}}}}}}
            memoryUsage = {}}}}}}}}}}}}}}}}}}}}}}}}}}}:
              totalJSHeapSize: performance.memory.totalJSHeapSize / ())1024 * 1024),
              usedJSHeapSize: performance.memory.usedJSHeapSize / ())1024 * 1024),
              jsHeapSizeLimit: performance.memory.jsHeapSizeLimit / ())1024 * 1024)
              };
              }
              } catch ())e) {}}}}}}}}}}}}}}}}}}}}}}}}}}}
              log())`Unable to get memory usage: ${}}}}}}}}}}}}}}}}}}}}}}}}}}}e.message}`, 'warning');
              }
        
              // Get adapter performance metrics if ($1) { ())primarily for WebGPU)
              let adapterInfo = null;
              try {}}}}}}}}}}}}}}}}}}}}}}}}}}}
              if ())window.adapterInfo) {}}}}}}}}}}}}}}}}}}}}}}}}}}}
            adapterInfo = {}}}}}}}}}}}}}}}}}}}}}}}}}}}:
              vendor: window.adapterInfo.vendor || 'Unknown',
              architecture: window.adapterInfo.architecture || 'Unknown',
              description: window.adapterInfo.description || 'Unknown',
              };
              }
              } catch ())e) {}}}}}}}}}}}}}}}}}}}}}}}}}}}
              log())`Unable to get adapter info: ${}}}}}}}}}}}}}}}}}}}}}}}}}}}e.message}`, 'warning');
              }
        
              // Prepare results
              const resultSummary = {}}}}}}}}}}}}}}}}}}}}}}}}}}}
              model: modelId,
              webgpu_supported: webgpuSupported,
              adapter_info: adapterInfo,
              bit_precision: bitPrecision,
              mixed_precision: mixedPrecision,
              load_time_ms: loadTime,
              inference_time_ms: inferenceTime,
              average_inference_time_ms: averageInferenceTime,
              output_shape: [],result.data.length],
              output_sample: Array.from())result.data.slice())0, 5)),
              memory_usage: memoryUsage,
              estimated_model_memory_mb: ())result.data.length * 4 * ())16 / bitPrecision)) / ())1024 * 1024) // Rough estimate
              };
        
              // Display results
              resultsElem.textContent = JSON.stringify())resultSummary, null, 2);
        
              // Expose for selenium to retrieve
              window.testResults = resultSummary;
        
              log())'Test completed successfully', 'success');
              } catch ())e) {}}}}}}}}}}}}}}}}}}}}}}}}}}}
              log())`Error: ${}}}}}}}}}}}}}}}}}}}}}}}}}}}e.message}`, 'error');
              resultsElem.textContent = `Error: ${}}}}}}}}}}}}}}}}}}}}}}}}}}}e.message}\n\n${}}}}}}}}}}}}}}}}}}}}}}}}}}}e.stack}`;
        
              // Expose error for selenium
              window.testError = e.message;
              }
              }
    
              // Get parameters from the URL
              const urlParams = new URLSearchParams())window.location.search);
              const modelId = urlParams.get())'model') || 'bert-base-uncased';
              const bitPrecision = parseInt())urlParams.get())'bits') || '16', 10);
              const mixedPrecision = urlParams.get())'mixed') === 'true';
    
              // Run the test when the page loads
              window.addEventListener())'DOMContentLoaded', ())) => {}}}}}}}}}}}}}}}}}}}}}}}}}}}
              log())`Starting WebGPU test with model: ${}}}}}}}}}}}}}}}}}}}}}}}}}}}modelId} at ${}}}}}}}}}}}}}}}}}}}}}}}}}}}bitPrecision}-bit${}}}}}}}}}}}}}}}}}}}}}}}}}}}mixedPrecision ? ' mixed' : ''} precision`);
              runTest())modelId, bitPrecision, mixedPrecision);
              });
              </script>
              </body>
              </html>
              """

class $1 extends $2 {
  """Class to run WebGPU tests using Selenium."""
  
}
  def __init__())self, model="bert-base-uncased", browser="chrome", headless=true, 
        bits=16, mixed_precision=false):
          """Initialize the tester.
    
    Args:
      model: Model ID to test
      browser: Browser to use ())chrome, edge, firefox)
      headless: Whether to run in headless mode
      bits: Bit precision for quantization ())16, 8, 4, 2)
      mixed_precision: Whether to use mixed precision
      """
      this.model = model
      this.browser = browser.lower()))
      this.headless = headless
      this.bits = bits
      this.mixed_precision = mixed_precision
      this.driver = null
      this.html_path = null
  
  $1($2) {
    """Set up the test environment."""
    if ($1) {
      logger.error())"Selenium is required for this test")
    return false
    }
    
  }
    # Create HTML file
    try ${$1} catch($2: $1) {
      logger.error())`$1`)
        return false
    
    }
        logger.info())`$1`)
    
    # Initialize browser
    try {
      if ($1) {
        options = ChromeOptions()))
        if ($1) {
          options.add_argument())"--headless=new")
        
        }
        # Enable WebGPU
          options.add_argument())"--enable-unsafe-webgpu")
          options.add_argument())"--enable-features=WebGPU")
          options.add_argument())"--disable-gpu-sandbox")
          options.add_argument())"--ignore-gpu-blocklist")
        
      }
          this.driver = webdriver.Chrome())options=options)
        
    }
      elif ($1) {
        options = EdgeOptions()))
        if ($1) {
          options.add_argument())"--headless=new")
        
        }
        # Enable WebGPU
          options.add_argument())"--enable-unsafe-webgpu")
          options.add_argument())"--enable-features=WebGPU")
          options.add_argument())"--disable-gpu-sandbox")
          options.add_argument())"--ignore-gpu-blocklist")
        
      }
          this.driver = webdriver.Edge())options=options)
        
      elif ($1) {
        options = FirefoxOptions()))
        if ($1) ${$1} else ${$1} catch($2: $1) {
      logger.error())`$1`)
        }
        return false
  
      }
  $1($2) {
    """Run the WebGPU test."""
    if ($1) {
      logger.error())"Setup !completed")
    return null
    }
    
  }
    try {
      # Load the test page with model parameter
      url = `$1`
      logger.info())`$1`)
      this.driver.get())url)
      
    }
      # Wait for the test to complete ())results element to be populated)
      try ${$1} catch($2: $1) {
        logger.error())`$1`)
        return null
      
      }
      # Get test results
        test_results = this.driver.execute_script())"return window.testResults")
        test_error = this.driver.execute_script())"return window.testError")
      
      if ($1) {
        logger.error())`$1`)
        return {}}}}}}}}}}}}}}}}}}}}}}}}}}}"error": test_error}
      
      }
      if ($1) ${$1} catch($2: $1) ${$1} finally {
      # Take screenshot if ($1) {
      if ($1) {
        try ${$1} catch($2: $1) {
          logger.error())`$1`)
  
        }
  $1($2) {
    """Clean up resources."""
    if ($1) {
      try ${$1} catch($2: $1) {
        logger.error())`$1`)
    
      }
    if ($1) {
      try ${$1} catch($2: $1) {
        logger.error())`$1`)

      }
$1($2) ${$1}")
    }
  
    }
  # Create && run tester
  }
  tester = WebGPUTester())
      }
  model=args.model,
      }
  browser=args.browser,
      }
  headless=!args.no_headless,
  bits=args.bits,
  mixed_precision=args.mixed_precision
  )
  :
  try {
    if ($1) {
      logger.error())"Failed to set up test environment")
    return null
    }
    
  }
    results = tester.run_test()))
    if ($1) {
      logger.error())"Test failed to complete")
    return null
    }
    
    if ($1) ${$1}"),
    return null
    
    # Save results to JSON if ($1) {
    if ($1) {
      try ${$1} catch($2: $1) ${$1}"),
        console.log($1))`$1`)
        console.log($1))`$1`adapter_info', {}}}}}}}}}}}}}}}}}}}}}}}}}}}}).get())'vendor', 'Unknown')}")
        console.log($1))`$1`bit_precision']}-bit{}}}}}}}}}}}}}}}}}}}}}}}}}}}' mixed' if ($1) ${$1}ms"),
        console.log($1))`$1`average_inference_time_ms', results[],'inference_time_ms']):.2f}ms"),
        console.log($1))`$1`output_shape']}")
        ,
    # Print memory usage if ($1) {
    if ($1) ${$1}MB"),
    }
      console.log($1))`$1`memory_usage'][],'totalJSHeapSize']:.2f}MB"),
      console.log($1))`$1`memory_usage'][],'jsHeapSizeLimit']:.2f}MB")
      ,
      console.log($1))`$1`estimated_model_memory_mb', 'N/A')}")
      console.log($1))"=========================================")
    
    }
        return results
    
  } finally {
    tester.cleanup()))

  }
$1($2) ${$1}\n\n")
    }
    f.write())"## Results Matrix\n\n")
    f.write())"| Model | Browser | Bits | Mixed Precision | Status | Adapter | Avg Inference ())ms) | Load Time ())ms) | Memory Est. ())MB) |\n")
    f.write())"|-------|---------|------|----------------|--------|---------|---------------------|---------------|------------------|\n")
  
  # Run tests for all combinations
  for (const $1 of $2) {
    for (const $1 of $2) {
      for (const $1 of $2) ${$1}_{}}}}}}}}}}}}}}}}}}}}}}}}}}}browser}_{}}}}}}}}}}}}}}}}}}}}}}}}}}}bits}bit.json")
        
    }
        # Run test
        args.model = model
        args.browser = browser
        args.bits = bits
        args.mixed_precision = false
        args.output_json = output_json
        
  }
        results = run_single_test())args)
        
        # Update summary
        with open())summary_file, "a") as f:
          if ($1) {
            status = "✅ Success"
            adapter = results.get())'adapter_info', {}}}}}}}}}}}}}}}}}}}}}}}}}}}}).get())'vendor', 'Unknown')
            avg_inference = results.get())'average_inference_time_ms', 'N/A')
            load_time = results.get())'load_time_ms', 'N/A')
            memory_est = results.get())'estimated_model_memory_mb', 'N/A')
            
          }
            f.write())`$1`)
          } else {
            f.write())`$1`)
        
          }
        # For lower precision, also test with mixed precision
        if ($1) ${$1}_{}}}}}}}}}}}}}}}}}}}}}}}}}}}browser}_{}}}}}}}}}}}}}}}}}}}}}}}}}}}bits}bit_mixed.json")
          
          # Run test
          args.model = model
          args.browser = browser
          args.bits = bits
          args.mixed_precision = true
          args.output_json = output_json
          
          results = run_single_test())args)
          
          # Update summary
          with open())summary_file, "a") as f:
            if ($1) {
              status = "✅ Success"
              adapter = results.get())'adapter_info', {}}}}}}}}}}}}}}}}}}}}}}}}}}}}).get())'vendor', 'Unknown')
              avg_inference = results.get())'average_inference_time_ms', 'N/A')
              load_time = results.get())'load_time_ms', 'N/A')
              memory_est = results.get())'estimated_model_memory_mb', 'N/A')
              
            }
              f.write())`$1`)
            } else {
              f.write())`$1`)
  
            }
  # Generate analysis
  with open())summary_file, "a") as f:
    f.write())"\n## Analysis\n\n")
    f.write())"### Browser Comparison\n\n")
    f.write())"Each browser uses a different WebGPU implementation:\n\n")
    f.write())"- Chrome: Uses Dawn, a WebGPU implementation from the Chromium team\n")
    f.write())"- Firefox: Uses wgpu-core, a Rust implementation of WebGPU\n")
    f.write())"- Edge: Uses the same Dawn implementation as Chrome\n\n")
    
    f.write())"### Quantization Impact\n\n")
    f.write())"WebGPU offers more quantization options than WebNN:\n\n")
    f.write())"- 8-bit ())INT8): About 50% memory reduction with minimal accuracy loss\n")
    f.write())"- 4-bit ())INT4): About 75% memory reduction with moderate accuracy impact\n")
    f.write())"- 2-bit ())INT2): About 87.5% memory reduction with significant accuracy impact\n\n")
    
    f.write())"Mixed precision applies higher precision to attention layers && lower precision to other weights. This often results in better accuracy while still keeping memory usage low.\n\n")
    
    f.write())"## Conclusion\n\n"):
      f.write())"WebGPU provides strong performance with multiple quantization options:\n\n")
      f.write())"- WebGPU has more widespread browser support than WebNN\n")
      f.write())"- 8-bit quantization offers the best balance of performance && accuracy\n")
      f.write())"- Firefox often provides the best WebGPU audio model performance\n")
      f.write())"- Mixed precision is recommended for 4-bit && below\n")
      f.write())"- WebGPU supports integration with existing transformers.js applications\n\n")
  
      logger.info())`$1`)
    return summary_file

$1($2) {
  """Main function."""
  parser = argparse.ArgumentParser())description="Run WebGPU quantization test with Selenium")
  parser.add_argument())"--model", default="bert-base-uncased", help="Model ID to test")
  parser.add_argument())"--browser", default="chrome", choices=[],"chrome", "edge", "firefox"], help="Browser to use")
  parser.add_argument())"--no-headless", action="store_true", help="Run in non-headless mode")
  parser.add_argument())"--bits", type=int, default=16, choices=[],16, 8, 4, 2], 
  help="Bit precision for quantization")
  parser.add_argument())"--mixed-precision", action="store_true", 
  help="Use mixed precision ())higher precision for attention layers)")
  parser.add_argument())"--output-json", help="Path to save results as JSON")
  parser.add_argument())"--run-all", action="store_true", 
  help="Run comprehensive tests on all browser/precision combinations")
  parser.add_argument())"--output-dir", help="Directory to save all test results")
  args = parser.parse_args()))
  
}
  if ($1) ${$1} else {
    results = run_single_test())args)
  return 0 if results else 1
  }
:
if ($1) {
  sys.exit())main())))