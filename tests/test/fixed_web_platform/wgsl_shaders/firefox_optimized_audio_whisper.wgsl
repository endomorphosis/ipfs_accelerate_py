// Optimized WebGPU compute shader for Whisper model audio processing
// Auto-generated for FIREFOX browser with workgroup size 256x1x1

struct ComputeUniforms {
    audio_length: u32,          // Length of audio in samples
    feature_size: u32,          // Size of mel spectrogram features
    n_mels: u32,                // Number of mel bands
    sample_rate: u32,           // Audio sample rate
    hop_length: u32,            // Hop length for STFT
    chunk_size: u32,            // Processing chunk size
};

@group(0) @binding(0) var<uniform> uniforms: ComputeUniforms;
@group(0) @binding(1) var<storage, read> audio_samples: array<f32>;
@group(0) @binding(2) var<storage, read_write> mel_spectrogram: array<f32>;
@group(0) @binding(3) var<storage, read> window_function: array<f32>;

// Firefox-optimized shared memory configuration with 256x1x1 workgroup
var<workgroup> shared_samples: array<f32, 1024>;
var<workgroup> shared_fft_real: array<f32, 512>;
var<workgroup> shared_fft_imag: array<f32, 512>;

@compute @workgroup_size(256, 1, 1)
fn process_audio_frame(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    // Frame index (time position in audio)
    let frame_idx = global_id.x;
    let mel_idx = global_id.y;
    
    // Skip if beyond dimensions
    if (frame_idx >= uniforms.feature_size || mel_idx >= uniforms.n_mels) {
        return;
    }
    
    // Calculate frame start in samples
    let frame_start = frame_idx * uniforms.hop_length;
    
    // Short-time Fourier Transform implementation
    // In a real implementation, this would be much more complex
    
    // 1. Apply window function to audio frame
    var windowed_frame: array<f32, 1024>;  // 1024 is a common FFT size for audio
    for (var i = 0u; i < 1024u && i < uniforms.chunk_size; i += 1u) {
        let sample_idx = frame_start + i;
        if (sample_idx < uniforms.audio_length) {
            windowed_frame[i] = audio_samples[sample_idx] * window_function[i];
        }
    }
    
    // 2. Compute FFT (simplified)
    // In a real implementation, this would be a full FFT algorithm
    var fft_real: array<f32, 513>;  // n_fft/2 + 1 bins for real FFT
    var fft_imag: array<f32, 513>;
    
    // Simulate FFT computation
    // This is a placeholder for actual FFT computation
    for (var k = 0u; k < 513u; k += 1u) {
        var real_sum = 0.0;
        var imag_sum = 0.0;
        
        for (var n = 0u; n < 1024u && n < uniforms.chunk_size; n += 1u) {
            let angle = 2.0 * 3.14159265359 * f32(k) * f32(n) / 1024.0;
            real_sum += windowed_frame[n] * cos(angle);
            imag_sum -= windowed_frame[n] * sin(angle);
        }
        
        fft_real[k] = real_sum;
        fft_imag[k] = imag_sum;
    }
    
    // 3. Compute power spectrum
    var power_spectrum: array<f32, 513>;
    for (var k = 0u; k < 513u; k += 1u) {
        power_spectrum[k] = fft_real[k] * fft_real[k] + fft_imag[k] * fft_imag[k];
    }
    
    // 4. Apply mel filterbank
    // In a real implementation, mel_filterbank would be passed as a buffer
    // Here we approximate with a triangular filter
    var mel_energy = 0.0;
    let bin_start = mel_idx * 10u;  // Simplified mel filter spacing
    let bin_center = bin_start + 5u;
    let bin_end = bin_start + 10u;
    
    for (var k = bin_start; k < bin_end && k < 513u; k += 1u) {
        var weight = 0.0;
        if (k < bin_center) {
            weight = f32(k - bin_start) / f32(bin_center - bin_start);
        } else {
            weight = f32(bin_end - k) / f32(bin_end - bin_center);
        }
        mel_energy += power_spectrum[k] * weight;
    }
    
    // 5. Apply log transform and store
    let output_idx = frame_idx * uniforms.n_mels + mel_idx;
    mel_spectrogram[output_idx] = log(mel_energy + 1.0e-10);
}

// Additional compute shader for batch processing
@compute @workgroup_size(256, 1, 1)
fn normalize_spectrogram(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let idx = global_id.x;
    let total_elements = uniforms.feature_size * uniforms.n_mels;
    
    if (idx >= total_elements) {
        return;
    }
    
    // Apply normalization
    // This would typically normalize across the spectrogram
    // Here we just clamp values to a reasonable range
    mel_spectrogram[idx] = clamp(mel_spectrogram[idx], -100.0, 20.0);
}