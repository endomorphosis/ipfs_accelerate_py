
/**
 * Audio Processing Utilities for Web Audio Model Testing
 */

// Load audio file and convert to audio buffer
async function loadAudioFile(url) {
    try {
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`Failed to fetch audio: ${response.statusText}`);
        }
        
        const arrayBuffer = await response.arrayBuffer();
        const audioContext = new (window.AudioContext || window.webkitAudioContext)();
        const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
        
        return { audioBuffer, audioContext };
    } catch (error) {
        console.error('Failed to load audio file:', error);
        throw error;
    }
}

// Convert audio buffer to mono channel Float32Array
function audioBufferToMono(audioBuffer) {
    const numChannels = audioBuffer.numberOfChannels;
    const length = audioBuffer.length;
    const sampleRate = audioBuffer.sampleRate;
    
    let monoData = new Float32Array(length);
    
    // For mono audio, just copy the first channel
    if (numChannels === 1) {
        monoData.set(audioBuffer.getChannelData(0));
    } else {
        // For multi-channel audio, average all channels
        for (let i = 0; i < numChannels; i++) {
            const channelData = audioBuffer.getChannelData(i);
            for (let j = 0; j < length; j++) {
                monoData[j] += channelData[j] / numChannels;
            }
        }
    }
    
    return { data: monoData, sampleRate };
}

// Resample audio to target sample rate
function resampleAudio(audioData, originalSampleRate, targetSampleRate) {
    if (originalSampleRate === targetSampleRate) {
        return audioData;
    }
    
    const ratio = originalSampleRate / targetSampleRate;
    const newLength = Math.round(audioData.length / ratio);
    const result = new Float32Array(newLength);
    
    for (let i = 0; i < newLength; i++) {
        // Simple linear interpolation resampling
        const position = i * ratio;
        const index = Math.floor(position);
        const fraction = position - index;
        
        if (index >= audioData.length - 1) {
            result[i] = audioData[audioData.length - 1];
        } else {
            result[i] = audioData[index] * (1 - fraction) + audioData[index + 1] * fraction;
        }
    }
    
    return result;
}

// Normalize audio to range [-1, 1]
function normalizeAudio(audioData) {
    let max = 0;
    for (let i = 0; i < audioData.length; i++) {
        const abs = Math.abs(audioData[i]);
        if (abs > max) {
            max = abs;
        }
    }
    
    if (max === 0) return audioData;
    
    const normalized = new Float32Array(audioData.length);
    for (let i = 0; i < audioData.length; i++) {
        normalized[i] = audioData[i] / max;
    }
    
    return normalized;
}

// Compute mel spectrogram from audio data
function computeMelSpectrogram(audioData, sampleRate, options = {}) {
    // Default parameters for mel spectrogram
    const n_fft = options.n_fft || 1024;
    const hop_length = options.hop_length || 512;
    const n_mels = options.n_mels || 80;
    const win_length = options.win_length || n_fft;
    const fmin = options.fmin || 0;
    const fmax = options.fmax || sampleRate / 2;
    
    // Create window function (Hann window)
    const window = new Float32Array(win_length);
    for (let i = 0; i < win_length; i++) {
        window[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / (win_length - 1));
    }
    
    // Calculate number of frames
    const numFrames = Math.floor((audioData.length - n_fft) / hop_length) + 1;
    
    // Initialize output spectrogram (n_mels x numFrames)
    const spectrogram = new Float32Array(n_mels * numFrames);
    const dimensions = [1, n_mels, numFrames];
    
    // Compute spectrogram using Web Audio API if available
    if (window.AnalyserNode) {
        // This is a simplified implementation - in a real scenario,
        // you would use the Web Audio API's AnalyserNode for FFT
        // and then convert to mel scale
        
        // For now, we'll generate a dummy spectrogram
        for (let i = 0; i < spectrogram.length; i++) {
            spectrogram[i] = Math.random() * 2 - 1;
        }
    } else {
        // Fallback to dummy spectrogram
        for (let i = 0; i < spectrogram.length; i++) {
            spectrogram[i] = Math.random() * 2 - 1;
        }
    }
    
    return { data: spectrogram, dimensions };
}

// Compute regular spectrogram from audio data
function computeSpectrogram(audioData, sampleRate, options = {}) {
    // Default parameters for spectrogram
    const n_fft = options.n_fft || 1024;
    const hop_length = options.hop_length || 512;
    const win_length = options.win_length || n_fft;
    
    // Create window function (Hann window)
    const window = new Float32Array(win_length);
    for (let i = 0; i < win_length; i++) {
        window[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / (win_length - 1));
    }
    
    // Calculate number of frames
    const numFrames = Math.floor((audioData.length - n_fft) / hop_length) + 1;
    const numFreqBins = n_fft / 2 + 1;
    
    // Initialize output spectrogram (numFreqBins x numFrames)
    const spectrogram = new Float32Array(numFreqBins * numFrames);
    const dimensions = [1, numFreqBins, numFrames];
    
    // Compute spectrogram using Web Audio API if available
    if (window.AnalyserNode) {
        // This is a simplified implementation - in a real scenario,
        // you would use the Web Audio API's AnalyserNode for FFT
        
        // For now, we'll generate a dummy spectrogram
        for (let i = 0; i < spectrogram.length; i++) {
            spectrogram[i] = Math.random() * 2 - 1;
        }
    } else {
        // Fallback to dummy spectrogram
        for (let i = 0; i < spectrogram.length; i++) {
            spectrogram[i] = Math.random() * 2 - 1;
        }
    }
    
    return { data: spectrogram, dimensions };
}

// Generate synthetic audio for testing
function generateSyntheticAudio(duration, sampleRate) {
    const numSamples = Math.round(duration * sampleRate);
    const audioData = new Float32Array(numSamples);
    
    // Generate a simple sine wave
    const frequency = 440; // A4 note
    for (let i = 0; i < numSamples; i++) {
        audioData[i] = Math.sin(2 * Math.PI * frequency * i / sampleRate);
    }
    
    return { data: audioData, sampleRate };
}

// Record audio from microphone
async function recordAudio(duration) {
    try {
        // Request microphone access
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        const audioContext = new (window.AudioContext || window.webkitAudioContext)();
        const source = audioContext.createMediaStreamSource(stream);
        
        // Create processor to capture audio data
        const processor = audioContext.createScriptProcessor(1024, 1, 1);
        const chunks = [];
        
        return new Promise((resolve, reject) => {
            let recordingStopped = false;
            
            processor.onaudioprocess = (e) => {
                if (recordingStopped) return;
                
                // Get audio data from input channel
                const audioData = e.inputBuffer.getChannelData(0);
                chunks.push(new Float32Array(audioData));
            };
            
            // Connect the processor to the source and destination
            source.connect(processor);
            processor.connect(audioContext.destination);
            
            // Stop recording after duration
            setTimeout(() => {
                recordingStopped = true;
                
                // Disconnect everything
                processor.disconnect();
                source.disconnect();
                stream.getTracks().forEach(track => track.stop());
                
                // Combine all chunks into a single Float32Array
                const totalLength = chunks.reduce((acc, chunk) => acc + chunk.length, 0);
                const result = new Float32Array(totalLength);
                
                let offset = 0;
                for (const chunk of chunks) {
                    result.set(chunk, offset);
                    offset += chunk.length;
                }
                
                resolve({ data: result, sampleRate: audioContext.sampleRate });
            }, duration * 1000);
        });
    } catch (error) {
        console.error('Failed to record audio:', error);
        throw error;
    }
}

// Export utilities
export {
    loadAudioFile,
    audioBufferToMono,
    resampleAudio,
    normalizeAudio,
    computeMelSpectrogram,
    computeSpectrogram,
    generateSyntheticAudio,
    recordAudio
};
