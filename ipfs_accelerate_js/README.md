# IPFS Accelerate JavaScript SDK

A hardware-accelerated AI model framework for web browsers using WebGPU, WebNN, and IPFS optimization.

## Features

- 🚀 **Hardware Acceleration**: Run AI models with WebGPU and WebNN for maximum performance
- 🌐 **Cross-Browser Support**: Works in all modern browsers with automatic fallbacks
- 📦 **Model Optimization**: Automatic model quantization and optimization
- 🧠 **Multiple Model Support**: Run BERT, ViT, Whisper, and more
- ⚛️ **React Integration**: Easy integration with React applications using custom hooks
- 📱 **Mobile Support**: Optimized for mobile devices with power-efficient execution
- 🔄 **Resource Pooling**: Efficient resource management for running multiple models
- ⚡ **Advanced WebGPU Optimization**: Operation fusion, memory layout optimization, and neural network pattern recognition

## Installation

```bash
npm install ipfs-accelerate
```

## Quick Start

```typescript
import { createModel, HardwareAbstraction } from 'ipfs-accelerate';

// Initialize hardware acceleration
const hardware = new HardwareAbstraction();
await hardware.initialize();

// Load a BERT model
const model = await createModel({
  modelId: 'bert-base-uncased',
  modelType: 'text',
  hardware
});

// Run inference
const result = await model.execute({
  input: "Hello, world\!"
});

console.log(result);
```

## React Integration

```tsx
import React from 'react';
import { useModel } from 'ipfs-accelerate/react';

function BertComponent() {
  const { model, status, error, loadModel } = useModel({
    modelId: 'bert-base-uncased',
    modelType: 'text',
    autoLoad: true
  });

  const [input, setInput] = React.useState('');
  const [result, setResult] = React.useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (model) {
      const output = await model.execute({ input });
      setResult(output);
    }
  };

  if (status === 'loading') return <div>Loading model...</div>;
  if (error) return <div>Error: {error.message}</div>;

  return (
    <div>
      <form onSubmit={handleSubmit}>
        <input
          type="text"
          value={input}
          onChange={e => setInput(e.target.value)}
          placeholder="Enter text"
        />
        <button type="submit">Process</button>
      </form>
      {result && (
        <pre>{JSON.stringify(result, null, 2)}</pre>
      )}
    </div>
  );
}
```

## Hardware Acceleration

The SDK automatically selects the best hardware acceleration method available:

1. **WebGPU**: Fastest option for browsers with WebGPU support
2. **WebNN**: Great performance for neural network operations
3. **WASM**: Fallback for browsers without WebGPU/WebNN support
4. **CPU**: Last resort fallback for all browsers

The selection process is transparent to developers and ensures optimal performance on each device and browser.

## Documentation

For detailed documentation, see:

- [API Documentation](docs/API_DOCUMENTATION.md)
- [TypeScript Implementation](docs/TYPESCRIPT_IMPLEMENTATION_SUMMARY.md)
- [Developer Guide](docs/DEVELOPER_GUIDE.md)
- [Model Support](docs/MODEL_SUPPORT.md)
- [WebGPU Optimization Guide](test/performance/webgpu_optimizer/OPTIMIZER_TESTING_GUIDE.md)

## Browser Support

- Chrome 113+ (WebGPU support)
- Edge 113+ (WebGPU support)
- Firefox 114+ (WebGPU support through flags)
- Safari 17+ (WebGPU support)
- All modern browsers (Fallback to WASM/CPU)

## Testing and Benchmarking

The project includes comprehensive testing and benchmarking tools:

```bash
# Run all tests
npm test

# Run WebGPU optimizer correctness tests
npm run test:webgpu:correctness

# Run WebGPU optimizer benchmarks (simulated environment)
npm run benchmark:webgpu

# Run specific benchmarks
npm run benchmark:webgpu:matmul
npm run benchmark:webgpu:memory-layout
npm run benchmark:webgpu:operation-fusion
npm run benchmark:webgpu:neural-network

# Run comprehensive benchmarks (all types)
npm run benchmark:webgpu:comprehensive

# Run benchmarks in real browsers (with Selenium)
npm run benchmark:webgpu:browser

# Generate HTML dashboard with visualizations
npm run benchmark:webgpu:dashboard

# Use the shell script for more options
./test/run_webgpu_benchmarks.sh --help
```

The benchmark framework includes an interactive dashboard that visualizes results, allowing you to:

- Compare performance of different optimization techniques
- Analyze memory savings from optimizations
- View browser-specific performance differences
- Track performance trends over time
- Identify top-performing optimization patterns

For more details, see the [WebGPU Optimization Testing Guide](test/performance/webgpu_optimizer/OPTIMIZER_TESTING_GUIDE.md).

## License

MIT
