
/**
 * WebGPU Utilities for Audio Model Testing
 */

// Initialize WebGPU
async function initWebGPU(preferredFormat = 'bgra8unorm') {
    try {
        // Check if WebGPU is available
        if (!navigator.gpu) {
            throw new Error('WebGPU is not supported in this browser');
        }
        
        // Request adapter
        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) {
            throw new Error('No WebGPU adapter found');
        }
        
        // Request device
        const device = await adapter.requestDevice();
        
        // Get preferred format
        const format = preferredFormat || navigator.gpu.getPreferredCanvasFormat();
        
        console.log(`WebGPU initialized with format: ${format}`);
        return { adapter, device, format };
    } catch (error) {
        console.error('Failed to initialize WebGPU:', error);
        throw error;
    }
}

// Create WebGPU buffer
function createBuffer(device, data, usage) {
    const buffer = device.createBuffer({
        size: data.byteLength,
        usage: usage,
        mappedAtCreation: true
    });
    
    const mapped = new Float32Array(buffer.getMappedRange());
    mapped.set(data);
    buffer.unmap();
    
    return buffer;
}

// Helper function to convert audio data to WebGPU buffer
function prepareAudioInputForWebGPU(device, audioData, sampleRate, modelType) {
    // Implementation depends on the model type
    switch (modelType) {
        case 'whisper':
            // Whisper expects mel spectrograms
            const melSpectrogram = computeMelSpectrogram(audioData, sampleRate);
            return createBuffer(
                device, 
                new Float32Array(melSpectrogram.data), 
                GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
            );
            
        case 'wav2vec2':
        case 'hubert':
            // Wav2Vec2 and HuBERT expect raw audio waveform
            // Normalize audio to float32 in [-1, 1] range
            const normalizedAudio = normalizeAudio(audioData);
            return createBuffer(
                device, 
                new Float32Array(normalizedAudio), 
                GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
            );
            
        case 'audio_spectrogram_transformer':
            // AST expects a spectrogram
            const spectrogram = computeSpectrogram(audioData, sampleRate);
            return createBuffer(
                device, 
                new Float32Array(spectrogram.data), 
                GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
            );
            
        default:
            throw new Error(`Unsupported model type: ${modelType}`);
    }
}

// Create a compute pipeline for audio processing
async function createComputePipeline(device, shaderCode) {
    const shaderModule = device.createShaderModule({
        code: shaderCode
    });
    
    const pipeline = await device.createComputePipelineAsync({
        layout: 'auto',
        compute: {
            module: shaderModule,
            entryPoint: 'main'
        }
    });
    
    return pipeline;
}

// Execute a compute shader for audio processing
async function executeComputeShader(device, pipeline, bindings, dispatchX, dispatchY = 1, dispatchZ = 1) {
    const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: bindings
    });
    
    const commandEncoder = device.createCommandEncoder();
    const computePass = commandEncoder.beginComputePass();
    
    computePass.setPipeline(pipeline);
    computePass.setBindGroup(0, bindGroup);
    computePass.dispatchWorkgroups(dispatchX, dispatchY, dispatchZ);
    computePass.end();
    
    const commands = commandEncoder.finish();
    device.queue.submit([commands]);
    
    // Wait for GPU to complete execution
    await device.queue.onSubmittedWorkDone();
}

// Get result from GPU buffer
async function getBufferData(device, buffer, size) {
    const readBuffer = device.createBuffer({
        size: size,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });
    
    const commandEncoder = device.createCommandEncoder();
    commandEncoder.copyBufferToBuffer(buffer, 0, readBuffer, 0, size);
    
    const commands = commandEncoder.finish();
    device.queue.submit([commands]);
    
    await readBuffer.mapAsync(GPUMapMode.READ);
    const result = new Float32Array(readBuffer.getMappedRange());
    const resultCopy = new Float32Array(result);
    readBuffer.unmap();
    
    return resultCopy;
}

// Export utilities
export {
    initWebGPU,
    createBuffer,
    prepareAudioInputForWebGPU,
    createComputePipeline,
    executeComputeShader,
    getBufferData
};
