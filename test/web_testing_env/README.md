# Web Testing Environment

Created: 2025-03-06T19:38:43.464447

## Available Browsers
chrome, firefox

## Environment Status
- WebNN: ✅ Available
- WebGPU: ✅ Available

## Usage with run_comprehensive_benchmarks.py
```bash
# For WebNN benchmarks
python run_comprehensive_benchmarks.py --models bert,t5 --hardware webnn --web-testing-dir ./web_testing_env

# For WebGPU benchmarks
python run_comprehensive_benchmarks.py --models bert,vit --hardware webgpu --web-testing-dir ./web_testing_env

# For WebGPU with compute shader optimization (audio models)
python run_comprehensive_benchmarks.py --models whisper,wav2vec2 --hardware webgpu --web-compute-shaders --web-testing-dir ./web_testing_env

# For WebGPU with parallel loading (multimodal models)
python run_comprehensive_benchmarks.py --models clip,llava --hardware webgpu --web-parallel-loading --web-testing-dir ./web_testing_env

# For WebGPU with shader precompilation (all models)
python run_comprehensive_benchmarks.py --models bert,vit --hardware webgpu --web-shader-precompile --web-testing-dir ./web_testing_env

# For all optimizations
python run_comprehensive_benchmarks.py --models all --hardware webgpu --web-all-optimizations --web-testing-dir ./web_testing_env
```

## Browser-Specific Features
- Edge: Best for WebNN
- Chrome: Best for general WebGPU
- Firefox: Best for WebGPU audio models (with compute shaders)
