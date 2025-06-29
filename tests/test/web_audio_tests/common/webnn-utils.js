
/**
 * WebNN Utilities for Audio Model Testing
 */

// Initialize WebNN Context
async function initWebNN(backendPreference = 'gpu') {
    try {
        // Check if WebNN is available
        if (!('ml' in navigator)) {
            throw new Error('WebNN is not supported in this browser');
        }
        
        // Try to get context with the preferred backend
        const contextOptions = {};
        if (backendPreference !== 'default') {
            contextOptions.devicePreference = backendPreference;
        }
        
        const context = await navigator.ml.createContext(contextOptions);
        const deviceType = await context.queryDevice();
        
        console.log(`WebNN initialized with backend: ${deviceType}`);
        return { context, deviceType };
    } catch (error) {
        console.error('Failed to initialize WebNN:', error);
        throw error;
    }
}

// Load ONNX model with WebNN
async function loadOnnxModelWithWebNN(modelUrl, context) {
    try {
        const response = await fetch(modelUrl);
        if (!response.ok) {
            throw new Error(`Failed to fetch model: ${response.statusText}`);
        }
        
        const modelBuffer = await response.arrayBuffer();
        const graphBuilder = new MLGraphBuilder(context);
        const model = await graphBuilder.buildFromOnnxBuffer(modelBuffer);
        
        return { graphBuilder, model };
    } catch (error) {
        console.error('Failed to load ONNX model with WebNN:', error);
        throw error;
    }
}

// Run inference with WebNN
async function runWebNNInference(model, inputs) {
    try {
        const start = performance.now();
        const results = await model.compute(inputs);
        const end = performance.now();
        
        return {
            results,
            inferenceTime: end - start
        };
    } catch (error) {
        console.error('Failed to run WebNN inference:', error);
        throw error;
    }
}

// Create WebNN tensor from array
function createWebNNTensor(graphBuilder, data, dimensions) {
    const tensorData = new Float32Array(data);
    return graphBuilder.constant({
        dataType: 'float32',
        dimensions,
        value: tensorData
    });
}

// Helper function to convert audio data to WebNN input
function prepareAudioInputForWebNN(graphBuilder, audioData, sampleRate, modelType) {
    // Implementation depends on the model type
    switch (modelType) {
        case 'whisper':
            // Whisper expects mel spectrograms
            const melSpectrogram = computeMelSpectrogram(audioData, sampleRate);
            return createWebNNTensor(graphBuilder, melSpectrogram.data, melSpectrogram.dimensions);
            
        case 'wav2vec2':
        case 'hubert':
            // Wav2Vec2 and HuBERT expect raw audio waveform
            // Normalize audio to float32 in [-1, 1] range
            const normalizedAudio = normalizeAudio(audioData);
            return createWebNNTensor(graphBuilder, normalizedAudio, [1, normalizedAudio.length]);
            
        case 'audio_spectrogram_transformer':
            // AST expects a spectrogram
            const spectrogram = computeSpectrogram(audioData, sampleRate);
            return createWebNNTensor(graphBuilder, spectrogram.data, spectrogram.dimensions);
            
        default:
            throw new Error(`Unsupported model type: ${modelType}`);
    }
}

// Export utilities
export {
    initWebNN,
    loadOnnxModelWithWebNN,
    runWebNNInference,
    createWebNNTensor,
    prepareAudioInputForWebNN
};
