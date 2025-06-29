# Configuration Validation and Auto-Correction Guide

**Version:** 1.0.0 (March 2025)

## Overview

The WebGPU Streaming Inference framework includes a comprehensive configuration validation and auto-correction system that ensures optimal settings across different browsers and hardware environments. This guide explains how to use this system to improve compatibility, prevent errors, and optimize performance.

## Key Features

- **Browser-specific validation:** Automatically validates and corrects configuration based on browser capabilities
- **Model-specific optimization:** Applies optimal settings based on model type and architecture
- **Auto-correction:** Fixes invalid configurations with appropriate defaults
- **Comprehensive profiles:** Pre-defined profiles for major browsers and hardware combinations
- **Graceful degradation:** Falls back to compatible options when features aren't supported
- **Performance telemetry:** Tracks configuration changes and their impact on performance

## Using Configuration Validation

### Basic Usage

The configuration validation system works automatically when you use the `WebPlatformAccelerator` with auto-detection enabled:

```javascript
import { WebPlatformAccelerator } from '@ipfs-accelerate/web-platform';

// Create accelerator with auto-detection
const accelerator = new WebPlatformAccelerator({
  modelPath: 'models/bert-base',
  modelType: 'text',
  autoDetect: true  // Enables automatic validation and correction
});

// Get validated configuration
const config = accelerator.getConfig();
console.log("Validated configuration:", config);
```

### Manual Validation

You can also manually validate configurations using the `ConfigurationManager`:

```javascript
import { ConfigurationManager } from '@ipfs-accelerate/web-platform';

// Create configuration manager for specific model type and browser
const configManager = new ConfigurationManager({
  modelType: 'text',
  browser: 'firefox',
  autoCorrect: true  // Enable auto-correction
});

// Define configuration to validate
const myConfig = {
  quantization: "2bit",
  workgroupSize: [32, 32, 1],
  useComputeShaders: true,
  shaderPrecompilation: true
};

// Validate configuration
const validationResult = configManager.validateConfiguration(myConfig);

if (validationResult.valid) {
  console.log("Configuration is valid!");
} else {
  console.log("Configuration issues:", validationResult.errors);
  
  if (validationResult.autoCorrected) {
    console.log("Auto-corrected configuration:", validationResult.config);
  }
}

// Get browser-optimized configuration
const optimizedConfig = configManager.getOptimizedConfiguration(myConfig);
console.log("Optimized for Firefox:", optimizedConfig);
```

## Configuration Rules and Auto-Correction

The validation system applies different rules based on model type and browser. Here are some of the key validation rules:

### Precision Validation

```javascript
// Valid precision values
const validPrecisionValues = ["2bit", "3bit", "4bit", "8bit", "16bit"];

// Invalid precision will be auto-corrected to 4-bit
const invalidConfig = { quantization: "invalid" };
// After validation: { quantization: "4bit" }

// Safari-specific correction
const safariConfig = { 
  browser: "safari",
  quantization: "2bit"  // Safari doesn't support 2-bit
};
// After validation: { browser: "safari", quantization: "4bit" }
```

### Workgroup Size Validation

```javascript
// Workgroup size must be a list of 3 positive integers
const invalidWorkgroup = { workgroupSize: "not_a_list" };
// After validation: { workgroupSize: [8, 8, 1] }

// Firefox-specific optimization for audio models
const firefoxAudioConfig = {
  browser: "firefox",
  modelType: "audio",
  useComputeShaders: true
};
// After validation: { browser: "firefox", modelType: "audio", useComputeShaders: true, workgroupSize: [256, 1, 1] }
```

### Model-Specific Validation

```javascript
// KV-cache not applicable for vision models
const visionModelConfig = {
  modelType: "vision",
  kvCacheOptimization: true  // Not applicable for vision models
};
// After validation: { modelType: "vision", kvCacheOptimization: false }

// Audio models benefit from compute shaders in Firefox
const audioModelConfig = {
  modelType: "audio",
  browser: "firefox"
};
// After optimization: { modelType: "audio", browser: "firefox", useComputeShaders: true }
```

## Browser-Specific Profiles

The framework includes profiles for major browsers with their capabilities and optimal settings:

### Chrome/Edge Profile

```javascript
// Chrome/Edge capabilities
const chromeCapabilities = {
  "2bit": true,
  "3bit": true,
  "4bit": true,
  "8bit": true,
  "16bit": true,
  "shaderPrecompilation": true,
  "computeShaders": true,
  "parallelLoading": true,
  "modelSharding": true,
  "kvCache": true
};

// Chrome optimization example for text models
const chromeTextConfig = {
  useShaderPrecompilation: true,
  useComputeShaders: true,
  workgroupSize: [8, 8, 1],
  enableParallelLoading: true
};
```

### Firefox Profile

```javascript
// Firefox capabilities
const firefoxCapabilities = {
  "2bit": true,
  "3bit": true,
  "4bit": true,
  "8bit": true,
  "16bit": true,
  "shaderPrecompilation": false,  // Limited support
  "computeShaders": true,
  "parallelLoading": true,
  "modelSharding": true,
  "kvCache": true
};

// Firefox optimization example for audio models
const firefoxAudioConfig = {
  useComputeShaders: true,
  workgroupSize: [256, 1, 1],  // Firefox-specific for audio
  firefoxAudioOptimization: true
};
```

### Safari Profile

```javascript
// Safari capabilities
const safariCapabilities = {
  "2bit": false,
  "3bit": false,
  "4bit": true,
  "8bit": true,
  "16bit": true,
  "shaderPrecompilation": true,
  "computeShaders": true,  // Limited but supported
  "parallelLoading": true,
  "modelSharding": false,
  "kvCache": false
};

// Safari optimization example
const safariConfig = {
  useShaderPrecompilation: true,
  useComputeShaders: false,  // Disabled for better compatibility
  workgroupSize: [4, 4, 1],  // Better for Safari/Metal
  useMetalOptimizations: true,
  enableParallelLoading: true,
  useKvCache: false  // Not well supported in Safari
};
```

## Streaming Configuration Validation

The StreamingAdapter includes specialized validation for streaming configurations:

```javascript
// Create framework with adapter
const accelerator = new WebPlatformAccelerator({
  modelPath: 'models/llama-7b',
  modelType: 'text',
  config: { browser: "firefox" }
});

const adapter = new StreamingAdapter(accelerator);

// Validate streaming configuration
const streamingConfig = {
  quantization: "int256",       // Invalid quantization
  maxTokensPerStep: 100,        // Too high
  maxBatchSize: 64              // Will be browser-adjusted
};

// Validation corrects issues
const validatedConfig = adapter._validateStreamingConfig(streamingConfig);
/* Result:
{
  quantization: "int4",          // Corrected to valid value
  maxTokensPerStep: 32,          // Limited to reasonable maximum
  maxBatchSize: 64,
  useComputeShaders: true,       // Added for Firefox
  workgroupSize: [256, 1, 1],    // Firefox-optimized for streaming
  validationTimestamp: 1732646400000
}
*/
```

## Incorporating Validation in Your Application

### React Application Example

```jsx
import React, { useState, useEffect } from 'react';
import { WebPlatformAccelerator, detectBrowser } from '@ipfs-accelerate/web-platform';

function StreamingApp() {
  const [accelerator, setAccelerator] = useState(null);
  const [endpoint, setEndpoint] = useState(null);
  const [config, setConfig] = useState({});
  const [output, setOutput] = useState('');
  const [validationStatus, setValidationStatus] = useState('');
  
  // Initialize accelerator with validation
  useEffect(() => {
    async function initializeAccelerator() {
      try {
        // Detect browser
        const browserInfo = detectBrowser();
        console.log(`Using ${browserInfo.name} ${browserInfo.version}`);
        
        // Create accelerator with validation
        const accel = new WebPlatformAccelerator({
          modelPath: 'models/llama-7b',
          modelType: 'text',
          config: {
            streamingInference: true,
            quantization: 4,
            kvCacheOptimization: true
          },
          autoDetect: true
        });
        
        // Get validated configuration
        const validatedConfig = accel.getConfig();
        
        // Store accelerator and config
        setAccelerator(accel);
        setConfig(validatedConfig);
        setEndpoint(accel.createEndpoint());
        
        // Report validation success
        setValidationStatus('Configuration validated successfully');
      } catch (error) {
        console.error("Initialization error:", error);
        setValidationStatus(`Validation error: ${error.message}`);
      }
    }
    
    initializeAccelerator();
  }, []);
  
  // Handle message submission
  const handleSubmit = async (prompt) => {
    if (!endpoint) return;
    
    setOutput('');
    
    try {
      await endpoint({
        text: prompt,
        maxTokens: 100,
        temperature: 0.7,
        callback: (token) => {
          setOutput(prev => prev + token);
        }
      });
    } catch (error) {
      console.error("Generation error:", error);
      setOutput(`Error: ${error.message}`);
    }
  };
  
  return (
    <div className="app">
      <h1>Streaming Chat App</h1>
      
      <div className="validation-status">
        Status: {validationStatus}
      </div>
      
      <div className="config-info">
        <h3>Active Configuration:</h3>
        <pre>{JSON.stringify(config, null, 2)}</pre>
      </div>
      
      <div className="chat-interface">
        <input
          type="text"
          placeholder="Enter your message..."
          onKeyDown={(e) => e.key === 'Enter' && handleSubmit(e.target.value)}
        />
        <button onClick={(e) => handleSubmit(e.target.previousSibling.value)}>
          Send
        </button>
      </div>
      
      <div className="output">
        {output || "Output will appear here..."}
      </div>
    </div>
  );
}
```

## Best Practices

### 1. Use Auto-Detection

Always enable auto-detection for the best experience across different browsers:

```javascript
const accelerator = new WebPlatformAccelerator({
  modelPath: 'models/bert-base',
  modelType: 'text',
  autoDetect: true  // Enable auto-detection
});
```

### 2. Provide Model Type

Always specify the model type to get optimal configuration for your specific model:

```javascript
// For text models (BERT, T5, LLaMA)
const textAccelerator = new WebPlatformAccelerator({
  modelPath: 'models/llama-7b',
  modelType: 'text',  // Specify model type
  autoDetect: true
});

// For vision models (ViT, ResNet)
const visionAccelerator = new WebPlatformAccelerator({
  modelPath: 'models/vit-base',
  modelType: 'vision',  // Specify model type
  autoDetect: true
});

// For audio models (Whisper, Wav2Vec2)
const audioAccelerator = new WebPlatformAccelerator({
  modelPath: 'models/whisper-small',
  modelType: 'audio',  // Specify model type
  autoDetect: true
});

// For multimodal models (CLIP, LLaVA)
const multimodalAccelerator = new WebPlatformAccelerator({
  modelPath: 'models/clip-vit',
  modelType: 'multimodal',  // Specify model type
  autoDetect: true
});
```

### 3. Handle Validation Errors

Always handle validation errors gracefully:

```javascript
try {
  const configManager = new ConfigurationManager({
    modelType: 'text',
    browser: 'safari',
    autoCorrect: true
  });
  
  const result = configManager.validateConfiguration(userConfig);
  
  if (!result.valid) {
    console.warn("Configuration validation issues:", result.errors);
    
    if (result.autoCorrected) {
      console.log("Auto-corrected configuration:", result.config);
      // Use the corrected configuration
      useConfig(result.config);
    } else {
      // Handle uncorrectable issues
      for (const error of result.errors) {
        if (error.severity === "error") {
          console.error(`Critical error: ${error.message}`);
          showUserError(`Configuration error: ${error.message}`);
        }
      }
    }
  }
} catch (error) {
  console.error("Validation system error:", error);
  // Fall back to default configuration
  useConfig(defaultConfig);
}
```

### 4. Provide Feedback to Users

Let users know when configuration has been auto-corrected:

```javascript
function initializeWithFeedback(userConfig) {
  try {
    const accelerator = new WebPlatformAccelerator({
      modelPath: 'models/llama-7b',
      modelType: 'text',
      config: userConfig,
      autoDetect: true
    });
    
    const finalConfig = accelerator.getConfig();
    
    // Check for differences between user config and final config
    const differences = [];
    for (const [key, value] of Object.entries(userConfig)) {
      if (JSON.stringify(finalConfig[key]) !== JSON.stringify(value)) {
        differences.push({
          key,
          original: value,
          corrected: finalConfig[key]
        });
      }
    }
    
    // Display differences to user
    if (differences.length > 0) {
      console.log("Some configuration values were auto-corrected:");
      for (const diff of differences) {
        console.log(`- ${diff.key}: ${diff.original} → ${diff.corrected}`);
      }
      
      showUserNotification(
        "Some settings were adjusted for better compatibility with your browser.",
        differences
      );
    }
    
    return accelerator;
  } catch (error) {
    console.error("Initialization error:", error);
    showUserError("Failed to initialize accelerator: " + error.message);
    return null;
  }
}
```

## Troubleshooting

### Configuration Not Applied

If your configuration settings don't seem to be applied:

1. Check if auto-detection is overriding your settings
2. Verify that your browser supports the requested features
3. Use `console.log(accelerator.getConfig())` to see the actual configuration being used

### Browser-Specific Issues

#### Safari Issues

```javascript
// Common Safari issues and solutions
const safariIssues = {
  "2-bit/3-bit precision": "Safari doesn't support 2-bit/3-bit precision yet. Use 4-bit minimum.",
  "KV cache optimization": "Safari has limited KV-cache support. Disable this feature.",
  "Model sharding": "Safari doesn't support model sharding. Use smaller models instead."
};

// Safari-specific configuration
const safariConfig = {
  quantization: 4,               // Minimum 4-bit for Safari
  useKvCache: false,             // Disable KV cache on Safari
  useModelSharding: false,       // Disable model sharding on Safari
  useMetalOptimizations: true,   // Enable Metal-specific optimizations
  conservativeMemory: true       // Use conservative memory settings
};
```

#### Firefox Audio Optimization

```javascript
// Firefox audio optimization
if (browser === "firefox" && modelType === "audio") {
  // Apply Firefox-specific audio optimizations
  config.useComputeShaders = true;
  config.workgroupSize = [256, 1, 1];      // Optimal for Firefox audio processing
  config.firefoxAudioOptimization = true;   // Enable Firefox audio optimizations
  
  console.log("Applied Firefox audio optimizations");
}
```

## Extending the Validation System

You can extend the validation system with custom rules:

```javascript
import { ConfigurationManager, ConfigValidationRule } from '@ipfs-accelerate/web-platform';

// Create custom validation rule
const customRule = new ConfigValidationRule({
  name: "custom_batch_size",
  condition: (config) => config.batchSize <= 16,
  errorMessage: "Batch size cannot exceed 16",
  severity: "error",
  canAutoCorrect: true,
  correctionFunction: (config) => ({...config, batchSize: Math.min(config.batchSize, 16)})
});

// Create configuration manager with custom rule
class CustomConfigManager extends ConfigurationManager {
  constructor(options) {
    super(options);
    
    // Add custom validation rule
    this.validationRules.push(customRule);
  }
  
  // Override/extend optimization method
  getOptimizedConfiguration(config) {
    // Get base optimized config
    const baseConfig = super.getOptimizedConfiguration(config);
    
    // Add custom optimizations
    return {
      ...baseConfig,
      customOptimization: true,
      optimizedBy: "CustomConfigManager"
    };
  }
}

// Use custom manager
const customManager = new CustomConfigManager({
  modelType: 'text',
  browser: 'chrome',
  autoCorrect: true
});

const result = customManager.validateConfiguration({
  batchSize: 32  // Will be auto-corrected to 16
});

console.log("Custom validation result:", result);
```

## Related Documentation

- [WebGPU Streaming Documentation](WEBGPU_STREAMING_DOCUMENTATION.md)
- [Unified Framework Guide](UNIFIED_FRAMEWORK_WITH_STREAMING_GUIDE.md)
- [Browser Compatibility Guide](WEB_PLATFORM_INTEGRATION_GUIDE.md)
- [Error Handling Guide](docs/ERROR_HANDLING_GUIDE.md)
- [Browser-Specific Optimizations](docs/browser_specific_optimizations.md)