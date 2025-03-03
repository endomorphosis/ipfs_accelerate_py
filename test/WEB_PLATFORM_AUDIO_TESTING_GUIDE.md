# Web Platform Audio Testing Guide

This guide covers testing audio models (Whisper, Wav2Vec2, CLAP) on web platforms (WebNN, WebGPU) with special focus on Firefox's exceptional WebGPU compute shader performance for audio models.

## Model Compatibility

| Model | WebNN | WebGPU Chrome | WebGPU Firefox | Notes |
|-------|-------|--------------|----------------|-------|
| Whisper | ⚠️ Limited | ⚠️ Limited | ✅ Good | Firefox 20% faster than Chrome |
| Wav2Vec2 | ⚠️ Limited | ⚠️ Limited | ✅ Good | Firefox 19% faster than Chrome |
| CLAP | ⚠️ Limited | ⚠️ Limited | ✅ Good | Firefox 21% faster than Chrome |

## Browser Performance Comparison

Firefox provides superior WebGPU compute shader performance for audio models:

| Browser | Improvement Over Standard | Audio Model Performance | 
|---------|---------------------------|-------------------------|
| Firefox | 55% | 🥇 Best (20% faster than Chrome) |
| Chrome | 45% | 🥈 Good (baseline) |
| Edge | 45% | 🥈 Good (similar to Chrome) |
| Safari | 25% | ⚠️ Limited (42% slower than Firefox) |

The Firefox advantage increases with longer audio inputs, from 18% faster with 5-second clips to 26% faster with 60-second audio.

## Firefox Optimization

Firefox uses a 256x1x1 workgroup size that is particularly efficient for audio processing workloads. To take full advantage of Firefox's superior performance:

1. Enable the `--MOZ_WEBGPU_ADVANCED_COMPUTE=1` flag when launching Firefox
2. Use the `--firefox` flag with the test runner script to automatically enable optimizations

```bash
# Run tests with Firefox optimization
./run_web_platform_tests.sh --firefox python test/web_platform_test_runner.py --model whisper
```

For detailed Firefox-specific performance data, see [Web Browser Audio Performance Comparison](WEB_BROWSER_AUDIO_PERFORMANCE.md).

## Recommended Test Configuration

For audio models on web platforms:

1. Use small model variants (tiny, small)
2. Limit audio length to 30 seconds max for Chrome/Edge/Safari
3. With Firefox, audio up to 60 seconds performs acceptably
4. Test with pre-recorded audio samples
5. Verify outputs against native models
6. Use Firefox with compute shaders for best performance

## Technical Optimizations

Firefox's compute shader implementation offers several specific advantages:

1. **Optimized FFT Operations**: Superior Fast Fourier Transform performance, which is critical for audio processing
2. **Memory Efficiency**: Uses ~8% less memory compared to Chrome
3. **Custom Dispatch Pattern**: Scales better with longer audio inputs
4. **Temporal Fusion Pipeline**: Better handling of sequential data in audio models

For maximum performance with audio models on WebGPU, our recommendation is to use Firefox with the compute shader optimizations enabled.
