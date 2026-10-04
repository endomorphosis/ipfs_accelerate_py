/**
 * BERT model example with hardware acceleration
 * Demonstrates using the BERT model with WebGPU/WebNN hardware acceleration
 */

import {
  createBertModel,
  BertConfig,
  BertInput,
  BertOutput
} from '../../../src/model/transformers/bert';
import { WebGPUBackend } from '../../../src/hardware/webgpu/backend';
import { WebNNBackend } from '../../../src/hardware/webnn/backend';
import { HardwareBackend } from '../../../src/hardware/interfaces/hardware_backend';
import { detectHardwareCapabilities } from '../../../src/hardware/detection/hardware_detector';

/**
 * Example demonstrating BERT model inference with hardware acceleration
 */
class BertExample {
  private hardware: HardwareBackend | null = null;
  private statusElement: HTMLElement;
  private inputElement: HTMLTextAreaElement;
  private resultElement: HTMLElement;
  private timeElement: HTMLElement;
  private modelSelect: HTMLSelectElement;
  private backendSelect: HTMLSelectElement;
  private optimizationCheckbox: HTMLInputElement;
  private runButton: HTMLButtonElement;
  
  constructor() {
    // Get UI elements
    this.statusElement = document.getElementById('status') as HTMLElement;
    this.inputElement = document.getElementById('input-text') as HTMLTextAreaElement;
    this.resultElement = document.getElementById('result') as HTMLElement;
    this.timeElement = document.getElementById('time') as HTMLElement;
    this.modelSelect = document.getElementById('model-id') as HTMLSelectElement;
    this.backendSelect = document.getElementById('backend') as HTMLSelectElement;
    this.optimizationCheckbox = document.getElementById('use-optimizations') as HTMLInputElement;
    this.runButton = document.getElementById('run-button') as HTMLButtonElement;
    
    // Initialize UI
    this.initializeUI();
  }
  
  /**
   * Initialize UI elements and event handlers
   */
  private initializeUI(): void {
    // Set default input text
    this.inputElement.value = "The quick brown fox jumps over the lazy dog. BERT models are used for natural language understanding tasks.";
    
    // Set up run button
    this.runButton.addEventListener('click', () => this.runInference());
    
    // Check hardware capabilities and populate backend options
    this.checkHardwareCapabilities();
  }
  
  /**
   * Check available hardware capabilities
   */
  private async checkHardwareCapabilities(): Promise<void> {
    try {
      this.updateStatus('Detecting hardware capabilities...');
      
      const capabilities = await detectHardwareCapabilities();
      
      // Clear options
      this.backendSelect.innerHTML = '';
      
      // Add WebGPU if available
      if (capabilities.webgpu.available) {
        const option = document.createElement('option');
        option.value = 'webgpu';
        option.textContent = `WebGPU (${capabilities.webgpu.deviceName || 'Unknown GPU'})`;
        this.backendSelect.appendChild(option);
      }
      
      // Add WebNN if available
      if (capabilities.webnn.available) {
        const option = document.createElement('option');
        option.value = 'webnn';
        option.textContent = `WebNN (${capabilities.webnn.deviceType || 'Unknown Device'})`;
        this.backendSelect.appendChild(option);
      }
      
      // Always add CPU as fallback
      const cpuOption = document.createElement('option');
      cpuOption.value = 'cpu';
      cpuOption.textContent = 'CPU';
      this.backendSelect.appendChild(cpuOption);
      
      // Select best available backend
      if (capabilities.webgpu.available) {
        this.backendSelect.value = 'webgpu';
      } else if (capabilities.webnn.available) {
        this.backendSelect.value = 'webnn';
      } else {
        this.backendSelect.value = 'cpu';
      }
      
      this.updateStatus('Hardware detection complete. Ready to run inference.');
      this.runButton.disabled = false;
    } catch (error) {
      this.updateStatus(`Error detecting hardware capabilities: ${error.message}`);
      console.error('Hardware detection error:', error);
    }
  }
  
  /**
   * Initialize selected hardware backend
   */
  private async initializeHardware(): Promise<HardwareBackend> {
    const backendType = this.backendSelect.value;
    
    // Clean up existing hardware if any
    if (this.hardware) {
      await this.hardware.dispose();
      this.hardware = null;
    }
    
    // Create new hardware backend
    switch (backendType) {
      case 'webgpu':
        this.hardware = new WebGPUBackend();
        break;
      case 'webnn':
        this.hardware = new WebNNBackend();
        break;
      default:
        throw new Error(`Backend type not implemented: ${backendType}`);
    }
    
    // Initialize hardware
    await this.hardware.initialize();
    return this.hardware;
  }
  
  /**
   * Run BERT inference on the input text
   */
  public async runInference(): Promise<void> {
    try {
      // Disable run button during inference
      this.runButton.disabled = true;
      this.updateStatus('Initializing hardware backend...');
      
      // Get input text
      const inputText = this.inputElement.value;
      if (!inputText.trim()) {
        this.updateStatus('Please enter some text to process.');
        this.runButton.disabled = false;
        return;
      }
      
      // Initialize hardware
      const hardware = await this.initializeHardware();
      this.updateStatus('Creating BERT model...');
      
      // Get selected model ID
      const modelId = this.modelSelect.value;
      
      // Get optimization setting
      const useOptimizedOps = this.optimizationCheckbox.checked;
      
      // Create BERT model
      const bertConfig: Partial<BertConfig> = {
        modelId,
        useOptimizedOps
      };
      
      const bert = createBertModel(hardware, bertConfig);
      
      // Initialize model
      this.updateStatus('Initializing BERT model and loading weights...');
      await bert.initialize();
      
      // Tokenize input
      this.updateStatus('Tokenizing input text...');
      const tokenizedInput = await bert.tokenize(inputText);
      
      // Run inference
      this.updateStatus('Running inference...');
      console.time('BERT inference');
      const startTime = performance.now();
      
      const output = await bert.process(tokenizedInput);
      
      const endTime = performance.now();
      const inferenceTime = endTime - startTime;
      console.timeEnd('BERT inference');
      
      // Display results
      this.displayResults(output, inferenceTime);
      
      // Clean up
      this.updateStatus('Cleaning up resources...');
      await bert.dispose();
      
      this.updateStatus('Inference complete.');
      this.runButton.disabled = false;
    } catch (error) {
      this.updateStatus(`Error running inference: ${error.message}`);
      console.error('Inference error:', error);
      this.runButton.disabled = false;
    }
  }
  
  /**
   * Display inference results
   */
  private displayResults(output: BertOutput, inferenceTime: number): void {
    // Display inference time
    this.timeElement.textContent = `${inferenceTime.toFixed(2)} ms`;
    
    // Create result HTML
    let resultHtml = `
      <h3>BERT Model Output</h3>
      <p><strong>Model:</strong> ${output.model}</p>
      <p><strong>Backend:</strong> ${output.backend}</p>
      <p><strong>Pooled Output (CLS token embedding):</strong></p>
      <div class="output-container">
        <pre>[${output.pooledOutput?.slice(0, 10).map(v => v.toFixed(4)).join(', ')}... (${output.pooledOutput?.length} values)]</pre>
      </div>
      <p><strong>Last Hidden State Shape:</strong> [${output.lastHiddenState.length}, ${output.lastHiddenState[0]?.length || 0}]</p>
    `;
    
    // Set result HTML
    this.resultElement.innerHTML = resultHtml;
  }
  
  /**
   * Update status message
   */
  private updateStatus(message: string): void {
    this.statusElement.textContent = message;
    console.log(message);
  }
}

// Initialize example when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
  new BertExample();
});