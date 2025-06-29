"""
Mojo Template for High-Performance Model Inference

This template provides optimized Mojo code for AI/ML model inference
with SIMD vectorization and memory optimization.
"""

from memory import memset_zero
from algorithm import vectorize, parallelize
from tensor import Tensor, TensorSpec, TensorShape
from utils.index import Index
from python import Python
import math

struct ModelConfig:
    var input_shape: StaticIntTuple[2]
    var hidden_size: Int
    var num_layers: Int
    var vocab_size: Int
    var max_seq_length: Int

struct OptimizedInference:
    var config: ModelConfig
    var weights: DynamicVector[Tensor[DType.float32]]
    var biases: DynamicVector[Tensor[DType.float32]]
    var initialized: Bool
    
    fn __init__(inout self, config: ModelConfig):
        self.config = config
        self.weights = DynamicVector[Tensor[DType.float32]]()
        self.biases = DynamicVector[Tensor[DType.float32]]()
        self.initialized = False
        self._initialize_weights()
    
    fn _initialize_weights(inout self):
        """Initialize model weights and biases."""
        # Initialize weights for each layer
        for i in range(self.config.num_layers):
            let weight_shape = TensorShape(self.config.hidden_size, self.config.hidden_size)
            let bias_shape = TensorShape(self.config.hidden_size)
            
            var weight = Tensor[DType.float32](weight_shape)
            var bias = Tensor[DType.float32](bias_shape)
            
            # Initialize with small random values (simplified)
            self._init_tensor_xavier(weight)
            self._init_tensor_zeros(bias)
            
            self.weights.push_back(weight)
            self.biases.push_back(bias)
        
        self.initialized = True
        print("Model weights initialized successfully")
    
    fn _init_tensor_xavier(inout self, inout tensor: Tensor[DType.float32]):
        """Initialize tensor with Xavier initialization."""
        let fan_in = tensor.shape()[0]
        let fan_out = tensor.shape()[1] if tensor.rank() > 1 else tensor.shape()[0]
        let limit = math.sqrt(6.0 / (fan_in + fan_out))
        
        # Fill with uniform random values (simplified implementation)
        for i in range(tensor.num_elements()):
            tensor[i] = (Float32(i % 100) / 100.0 - 0.5) * 2.0 * limit
    
    fn _init_tensor_zeros(inout self, inout tensor: Tensor[DType.float32]):
        """Initialize tensor with zeros."""
        memset_zero(tensor.data(), tensor.num_elements())
    
    fn forward(self, input: Tensor[DType.float32]) -> Tensor[DType.float32]:
        """Optimized forward pass with vectorization."""
        if not self.initialized:
            print("Error: Model not initialized")
            return input
        
        var current = input
        
        # Process through each layer
        for layer_idx in range(self.config.num_layers):
            current = self._layer_forward(current, layer_idx)
            current = self._apply_activation(current)
        
        return current
    
    fn _layer_forward(self, input: Tensor[DType.float32], layer_idx: Int) -> Tensor[DType.float32]:
        """Forward pass for a single layer with optimized matrix multiplication."""
        let weights = self.weights[layer_idx]
        let biases = self.biases[layer_idx]
        
        # Output shape calculation
        let batch_size = input.shape()[0]
        let output_dim = weights.shape()[1]
        let output_shape = TensorShape(batch_size, output_dim)
        var output = Tensor[DType.float32](output_shape)
        
        # Optimized matrix multiplication
        self._matmul_optimized(input, weights, output)
        
        # Add bias
        self._add_bias_vectorized(output, biases)
        
        return output
    
    @always_inline
    fn _matmul_optimized(self, a: Tensor[DType.float32], 
                        b: Tensor[DType.float32], 
                        inout result: Tensor[DType.float32]):
        """Highly optimized matrix multiplication with SIMD and parallelization."""
        let M = a.shape()[0]  # batch size
        let K = a.shape()[1]  # input dimension
        let N = b.shape()[1]  # output dimension
        
        @parameter
        fn compute_row(m: Int):
            @parameter
            fn vectorized_inner[simd_width: Int](k_start: Int):
                let k_end = min(k_start + simd_width, K)
                
                for n in range(N):
                    var sum = SIMD[DType.float32, simd_width](0.0)
                    
                    for k in range(k_start, k_end):
                        let a_val = a[m * K + k]
                        let b_val = b[k * N + n]
                        sum[k - k_start] = a_val * b_val
                    
                    # Accumulate SIMD results
                    var total: Float32 = 0.0
                    for i in range(simd_width):
                        if k_start + i < K:
                            total += sum[i]
                    
                    result[m * N + n] += total
        
            # Vectorize over K dimension
            vectorize[vectorized_inner, simd_width_of[DType.float32]()](K)
        
        # Parallelize over batch dimension
        parallelize[compute_row](M)
    
    @always_inline
    fn _add_bias_vectorized(self, inout tensor: Tensor[DType.float32], 
                           bias: Tensor[DType.float32]):
        """Add bias with vectorization."""
        let batch_size = tensor.shape()[0]
        let hidden_size = tensor.shape()[1]
        
        @parameter
        fn add_bias_batch(batch_idx: Int):
            @parameter  
            fn vectorized_add[simd_width: Int](idx: Int):
                let offset = batch_idx * hidden_size + idx
                let bias_vec = bias.load[simd_width](idx)
                let tensor_vec = tensor.load[simd_width](offset)
                tensor.store[simd_width](offset, tensor_vec + bias_vec)
            
            vectorize[vectorized_add, simd_width_of[DType.float32]()](hidden_size)
        
        parallelize[add_bias_batch](batch_size)
    
    @always_inline
    fn _apply_activation(self, inout tensor: Tensor[DType.float32]) -> Tensor[DType.float32]:
        """Apply ReLU activation with vectorization."""
        var result = Tensor[DType.float32](tensor.shape())
        
        @parameter
        fn vectorized_relu[simd_width: Int](idx: Int):
            let input_vec = tensor.load[simd_width](idx)
            let zero_vec = SIMD[DType.float32, simd_width](0.0)
            let result_vec = max(input_vec, zero_vec)
            result.store[simd_width](idx, result_vec)
        
        vectorize[vectorized_relu, simd_width_of[DType.float32]()](tensor.num_elements())
        return result
    
    fn benchmark_inference(self, num_iterations: Int = 100) -> Float64:
        """Benchmark inference performance."""
        let input_shape = TensorShape(1, self.config.hidden_size)
        var input = Tensor[DType.float32](input_shape)
        
        # Initialize with dummy data
        for i in range(input.num_elements()):
            input[i] = Float32(i % 10) / 10.0
        
        # Warmup
        for _ in range(10):
            _ = self.forward(input)
        
        # Benchmark
        let start_time = now()
        for _ in range(num_iterations):
            _ = self.forward(input)
        let end_time = now()
        
        let total_time = (end_time - start_time).to_float64()
        let avg_time_ms = (total_time / num_iterations) * 1000.0
        
        print("Benchmark Results:")
        print("  Iterations:", num_iterations)
        print("  Average inference time:", avg_time_ms, "ms")
        print("  Throughput:", 1000.0 / avg_time_ms, "inferences/sec")
        
        return avg_time_ms

struct TokenProcessor:
    """Optimized token processing for language models."""
    var vocab_size: Int
    var embedding_dim: Int
    var embeddings: Tensor[DType.float32]
    
    fn __init__(inout self, vocab_size: Int, embedding_dim: Int):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        # Initialize embedding table
        let embedding_shape = TensorShape(vocab_size, embedding_dim)
        self.embeddings = Tensor[DType.float32](embedding_shape)
        self._init_embeddings()
    
    fn _init_embeddings(inout self):
        """Initialize embedding table with random values."""
        for i in range(self.embeddings.num_elements()):
            # Simple random initialization
            self.embeddings[i] = (Float32(i % 1000) / 1000.0 - 0.5) * 0.1
    
    fn embed_tokens(self, token_ids: Tensor[DType.int32]) -> Tensor[DType.float32]:
        """Convert token IDs to embeddings with optimized lookup."""
        let batch_size = token_ids.shape()[0]
        let seq_length = token_ids.shape()[1]
        let output_shape = TensorShape(batch_size, seq_length, self.embedding_dim)
        var output = Tensor[DType.float32](output_shape)
        
        @parameter
        fn process_batch(batch_idx: Int):
            for seq_idx in range(seq_length):
                let token_id = int(token_ids[batch_idx * seq_length + seq_idx])
                if token_id >= 0 and token_id < self.vocab_size:
                    # Copy embedding vector
                    let embed_start = token_id * self.embedding_dim
                    let output_start = (batch_idx * seq_length + seq_idx) * self.embedding_dim
                    
                    @parameter
                    fn copy_vectorized[simd_width: Int](dim_idx: Int):
                        let embed_vec = self.embeddings.load[simd_width](embed_start + dim_idx)
                        output.store[simd_width](output_start + dim_idx, embed_vec)
                    
                    vectorize[copy_vectorized, simd_width_of[DType.float32]()](self.embedding_dim)
        
        parallelize[process_batch](batch_size)
        return output

fn main():
    """Main function demonstrating optimized model inference."""
    print("Initializing Mojo Optimized Model...")
    
    # Model configuration
    let config = ModelConfig(
        StaticIntTuple[2](1, 512),  # input_shape  
        768,  # hidden_size
        12,   # num_layers
        50000,  # vocab_size
        2048    # max_seq_length
    )
    
    # Create and initialize model
    var model = OptimizedInference(config)
    
    # Create sample input
    let input_shape = TensorShape(1, 768)
    var input = Tensor[DType.float32](input_shape)
    
    # Fill with sample data
    for i in range(input.num_elements()):
        input[i] = Float32(i % 100) / 100.0
    
    print("Running inference...")
    let output = model.forward(input)
    
    print("Inference completed successfully!")
    print("Input shape:", input.shape()[0], "x", input.shape()[1])
    print("Output shape:", output.shape()[0], "x", output.shape()[1])
    
    # Run benchmark
    print("\nRunning performance benchmark...")
    let avg_time = model.benchmark_inference(100)
    print("Average inference time:", avg_time, "ms")
    
    # Token processing demonstration
    print("\nTesting token processing...")
    var token_processor = TokenProcessor(50000, 768)
    
    let token_shape = TensorShape(1, 10)
    var token_ids = Tensor[DType.int32](token_shape)
    for i in range(10):
        token_ids[i] = i * 1000
    
    let embeddings = token_processor.embed_tokens(token_ids)
    print("Token embedding completed!")
    print("Token embeddings shape:", embeddings.shape()[0], "x", 
          embeddings.shape()[1], "x", embeddings.shape()[2])
    
    print("\nMojo model optimization completed successfully!")
