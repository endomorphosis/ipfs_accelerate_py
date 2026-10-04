/**
 * Whisper speech recognition example with hardware acceleration
 * Demonstrates using the Whisper model with WebGPU/WebNN hardware acceleration
 */

import {
  createWhisperModel,
  WhisperConfig,
  WhisperInput,
  WhisperOutput,
  WhisperSegment
} from '../../../src/model/audio/whisper';
import { WebGPUBackend } from '../../../src/hardware/webgpu/backend';
import { WebNNBackend } from '../../../src/hardware/webnn/backend';
import { HardwareBackend } from '../../../src/hardware/interfaces/hardware_backend';
import { detectHardwareCapabilities } from '../../../src/hardware/detection/hardware_detector';

/**
 * Example demonstrating Whisper model inference with hardware acceleration
 */
class WhisperExample {
  private hardware: HardwareBackend | null = null;
  private statusElement: HTMLElement;
  private resultElement: HTMLElement;
  private timeElement: HTMLElement;
  private durationElement: HTMLElement;
  private modelSelect: HTMLSelectElement;
  private backendSelect: HTMLSelectElement;
  private optimizationCheckbox: HTMLInputElement;
  private timestampsCheckbox: HTMLInputElement;
  private multilingualCheckbox: HTMLInputElement;
  private runButton: HTMLButtonElement;
  private recordButton: HTMLButtonElement;
  private stopButton: HTMLButtonElement;
  private uploadButton: HTMLButtonElement;
  private audioFileInput: HTMLInputElement;
  private audioPlayer: HTMLAudioElement;
  private audioVisualizer: HTMLCanvasElement;
  
  private mediaRecorder: MediaRecorder | null = null;
  private audioChunks: Blob[] = [];
  private audioData: Float32Array | null = null;
  private audioContext: AudioContext | null = null;
  private analyser: AnalyserNode | null = null;
  private animationFrameId: number | null = null;
  
  constructor() {
    // Get UI elements
    this.statusElement = document.getElementById('status') as HTMLElement;
    this.resultElement = document.getElementById('result') as HTMLElement;
    this.timeElement = document.getElementById('time') as HTMLElement;
    this.durationElement = document.getElementById('duration') as HTMLElement;
    this.modelSelect = document.getElementById('model-id') as HTMLSelectElement;
    this.backendSelect = document.getElementById('backend') as HTMLSelectElement;
    this.optimizationCheckbox = document.getElementById('use-optimizations') as HTMLInputElement;
    this.timestampsCheckbox = document.getElementById('return-timestamps') as HTMLInputElement;
    this.multilingualCheckbox = document.getElementById('multilingual') as HTMLInputElement;
    this.runButton = document.getElementById('run-button') as HTMLButtonElement;
    this.recordButton = document.getElementById('record-button') as HTMLButtonElement;
    this.stopButton = document.getElementById('stop-button') as HTMLButtonElement;
    this.uploadButton = document.getElementById('upload-button') as HTMLButtonElement;
    this.audioFileInput = document.getElementById('audio-file') as HTMLInputElement;
    this.audioPlayer = document.getElementById('audio-player') as HTMLAudioElement;
    this.audioVisualizer = document.getElementById('audio-wave') as HTMLCanvasElement;
    
    // Initialize UI
    this.initializeUI();
  }
  
  /**
   * Initialize UI elements and event handlers
   */
  private initializeUI(): void {
    // Set up run button
    this.runButton.addEventListener('click', () => this.runInference());
    
    // Set up record button
    this.recordButton.addEventListener('click', () => this.startRecording());
    
    // Set up stop button
    this.stopButton.addEventListener('click', () => this.stopRecording());
    
    // Set up upload button
    this.uploadButton.addEventListener('click', () => {
      this.audioFileInput.click();
    });
    
    // Set up file input
    this.audioFileInput.addEventListener('change', (event) => {
      const fileInput = event.target as HTMLInputElement;
      if (fileInput.files && fileInput.files.length > 0) {
        this.handleAudioFile(fileInput.files[0]);
      }
    });
    
    // Check hardware capabilities
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
      
      // Browser-specific recommendations for Whisper
      if (navigator.userAgent.includes('Firefox')) {
        this.updateStatus('Firefox detected! Firefox has 20-25% better performance for audio models. Using Firefox-optimized settings.');
        this.optimizationCheckbox.checked = true;
      }
      
      this.updateStatus('Hardware detection complete. Ready to record or upload audio.');
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
   * Start recording audio from the microphone
   */
  private async startRecording(): Promise<void> {
    try {
      this.updateStatus('Requesting microphone access...');
      
      // Get audio stream
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      
      // Create media recorder
      this.mediaRecorder = new MediaRecorder(stream);
      this.audioChunks = [];
      
      // Set up event handlers
      this.mediaRecorder.addEventListener('dataavailable', (event) => {
        if (event.data.size > 0) {
          this.audioChunks.push(event.data);
        }
      });
      
      this.mediaRecorder.addEventListener('stop', async () => {
        try {
          // Create blob from chunks
          const audioBlob = new Blob(this.audioChunks, { type: 'audio/wav' });
          
          // Create audio URL
          const audioUrl = URL.createObjectURL(audioBlob);
          
          // Set audio player source
          this.audioPlayer.src = audioUrl;
          this.audioPlayer.style.display = 'block';
          
          // Convert blob to audio data
          await this.convertBlobToAudioData(audioBlob);
          
          // Enable run button
          this.runButton.disabled = false;
          
          this.updateStatus('Recording complete. Ready for transcription.');
        } catch (error) {
          this.updateStatus(`Error processing recorded audio: ${error.message}`);
          console.error('Audio processing error:', error);
        }
      });
      
      // Set up audio visualization
      this.setupAudioVisualization(stream);
      
      // Start recording
      this.mediaRecorder.start();
      
      // Update UI
      this.recordButton.disabled = true;
      this.stopButton.disabled = false;
      this.updateStatus('Recording audio... (Click "Stop Recording" when finished)');
    } catch (error) {
      this.updateStatus(`Error starting recording: ${error.message}`);
      console.error('Recording error:', error);
    }
  }
  
  /**
   * Stop recording audio
   */
  private stopRecording(): void {
    if (this.mediaRecorder && this.mediaRecorder.state !== 'inactive') {
      this.mediaRecorder.stop();
      
      // Stop all tracks in the stream
      if (this.mediaRecorder.stream) {
        this.mediaRecorder.stream.getTracks().forEach(track => track.stop());
      }
      
      // Update UI
      this.recordButton.disabled = false;
      this.stopButton.disabled = true;
      this.updateStatus('Processing recorded audio...');
      
      // Stop visualization
      this.stopVisualization();
    }
  }
  
  /**
   * Handle uploaded audio file
   */
  private async handleAudioFile(file: File): Promise<void> {
    try {
      this.updateStatus(`Loading audio file: ${file.name}`);
      
      // Create audio URL
      const audioUrl = URL.createObjectURL(file);
      
      // Set audio player source
      this.audioPlayer.src = audioUrl;
      this.audioPlayer.style.display = 'block';
      
      // Convert file to audio data
      await this.convertBlobToAudioData(file);
      
      // Enable run button
      this.runButton.disabled = false;
      
      this.updateStatus('Audio file loaded. Ready for transcription.');
    } catch (error) {
      this.updateStatus(`Error loading audio file: ${error.message}`);
      console.error('File loading error:', error);
    }
  }
  
  /**
   * Convert audio blob to Float32Array for processing
   */
  private async convertBlobToAudioData(blob: Blob): Promise<void> {
    // Create audio context
    this.audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
    
    // Read file as array buffer
    const arrayBuffer = await blob.arrayBuffer();
    
    // Decode audio data
    const audioBuffer = await this.audioContext.decodeAudioData(arrayBuffer);
    
    // Get audio data (use first channel if stereo)
    const audioData = audioBuffer.getChannelData(0);
    
    // Store audio data
    this.audioData = audioData;
    
    // Update duration
    const duration = audioBuffer.duration;
    this.durationElement.textContent = `${duration.toFixed(2)} s`;
  }
  
  /**
   * Set up audio visualization
   */
  private setupAudioVisualization(stream: MediaStream): void {
    // Create audio context and analyser
    this.audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
    const source = this.audioContext.createMediaStreamSource(stream);
    this.analyser = this.audioContext.createAnalyser();
    source.connect(this.analyser);
    
    // Configure analyser
    this.analyser.fftSize = 2048;
    const bufferLength = this.analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    
    // Get canvas context
    const canvas = this.audioVisualizer;
    const canvasCtx = canvas.getContext('2d')!;
    
    // Draw function
    const draw = () => {
      // Request next frame
      this.animationFrameId = requestAnimationFrame(draw);
      
      // Get current data
      this.analyser!.getByteTimeDomainData(dataArray);
      
      // Clear canvas
      canvasCtx.fillStyle = '#f0f8ff';
      canvasCtx.fillRect(0, 0, canvas.width, canvas.height);
      
      // Draw wave
      canvasCtx.lineWidth = 2;
      canvasCtx.strokeStyle = '#4a90e2';
      canvasCtx.beginPath();
      
      const sliceWidth = canvas.width / bufferLength;
      let x = 0;
      
      for (let i = 0; i < bufferLength; i++) {
        const v = dataArray[i] / 128.0;
        const y = v * canvas.height / 2;
        
        if (i === 0) {
          canvasCtx.moveTo(x, y);
        } else {
          canvasCtx.lineTo(x, y);
        }
        
        x += sliceWidth;
      }
      
      canvasCtx.lineTo(canvas.width, canvas.height / 2);
      canvasCtx.stroke();
    };
    
    // Start drawing
    draw();
  }
  
  /**
   * Stop audio visualization
   */
  private stopVisualization(): void {
    if (this.animationFrameId !== null) {
      cancelAnimationFrame(this.animationFrameId);
      this.animationFrameId = null;
    }
    
    // Clear canvas
    const canvas = this.audioVisualizer;
    const canvasCtx = canvas.getContext('2d')!;
    canvasCtx.fillStyle = '#f0f8ff';
    canvasCtx.fillRect(0, 0, canvas.width, canvas.height);
  }
  
  /**
   * Run speech recognition inference on the audio data
   */
  public async runInference(): Promise<void> {
    try {
      // Check if we have audio data
      if (!this.audioData) {
        this.updateStatus('No audio data available. Please record or upload audio first.');
        return;
      }
      
      // Disable run button during inference
      this.runButton.disabled = true;
      this.updateStatus('Initializing hardware backend...');
      
      // Initialize hardware
      const hardware = await this.initializeHardware();
      this.updateStatus('Creating Whisper model...');
      
      // Get selected model ID
      const modelId = this.modelSelect.value;
      
      // Get option settings
      const useBrowserOptimizations = this.optimizationCheckbox.checked;
      const returnTimestamps = this.timestampsCheckbox.checked;
      const multilingual = this.multilingualCheckbox.checked;
      
      // Create Whisper model
      const whisperConfig: Partial<WhisperConfig> = {
        modelId,
        useBrowserOptimizations
      };
      
      const whisper = createWhisperModel(hardware, whisperConfig);
      
      // Initialize model
      this.updateStatus('Initializing Whisper model and loading weights...');
      await whisper.initialize();
      
      // Prepare input
      const input: WhisperInput = {
        audioData: this.audioData,
        sampleRate: this.audioContext?.sampleRate || 16000,
        multilingual,
        returnTimestamps,
        beamSize: 5
      };
      
      // Run inference
      this.updateStatus('Running speech recognition...');
      console.time('Whisper inference');
      const startTime = performance.now();
      
      const output = await whisper.process(input);
      
      const endTime = performance.now();
      const inferenceTime = endTime - startTime;
      console.timeEnd('Whisper inference');
      
      // Display results
      this.displayResults(output, inferenceTime);
      
      // Clean up
      this.updateStatus('Cleaning up resources...');
      await whisper.dispose();
      
      this.updateStatus('Transcription complete.');
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
  private displayResults(output: WhisperOutput, inferenceTime: number): void {
    // Display inference time
    this.timeElement.textContent = `${inferenceTime.toFixed(2)} ms`;
    
    // Create result HTML
    let resultHtml = `
      <div class="transcription">
        ${output.text}
      </div>
    `;
    
    // Add metadata
    resultHtml += `
      <div>
        <p><strong>Model:</strong> ${output.model}</p>
        <p><strong>Backend:</strong> ${output.backend}</p>
        ${output.detectedLanguage ? `<p><strong>Detected Language:</strong> ${output.detectedLanguage}</p>` : ''}
      </div>
    `;
    
    // Add segments if available
    if (output.segments && output.segments.length > 0) {
      resultHtml += `
        <div class="segments">
          <h3>Segments with Timestamps</h3>
      `;
      
      output.segments.forEach((segment, index) => {
        resultHtml += `
          <div class="segment">
            <div class="segment-time">${formatTime(segment.start)} → ${formatTime(segment.end)}</div>
            <div>${segment.text}</div>
          </div>
        `;
      });
      
      resultHtml += `</div>`;
    }
    
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

/**
 * Format time in seconds to MM:SS.MS format
 */
function formatTime(seconds: number): string {
  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = Math.floor(seconds % 60);
  const milliseconds = Math.floor((seconds % 1) * 100);
  
  return `${String(minutes).padStart(2, '0')}:${String(remainingSeconds).padStart(2, '0')}.${String(milliseconds).padStart(2, '0')}`;
}

// Initialize example when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
  new WhisperExample();
});