"""
MAX Graph Template for Optimized Model Deployment

This template provides MAX Engine graph construction for
high-performance model inference and deployment.
"""

import max
from max.graph import Graph, TensorType, ops
from max.graph.quantization import Float32Encoding, QuantizationEncoding
from max.engine import InferenceSession
from typing import Dict, List, Any, Optional
import numpy as np

class OptimizedMAXGraph:
    """Create optimized MAX graphs for model deployment."""
    
    def __init__(self, model_config: Dict[str, Any]):
        self.config = model_config
        self.graph = None
        self.session = None
        
    def create_language_model_graph(self) -> Graph:
        """Create optimized graph for language model inference."""
        graph = Graph()
        
        # Input specifications
        batch_size = self.config.get('batch_size', 1)
        seq_length = self.config.get('seq_length', 512)
        vocab_size = self.config.get('vocab_size', 50000)
        hidden_size = self.config.get('hidden_size', 768)
        num_layers = self.config.get('num_layers', 12)
        num_heads = self.config.get('num_heads', 12)
        
        # Token input
        token_input_spec = TensorType(max.DType.int32, (batch_size, seq_length))
        token_ids = graph.input(token_input_spec)
        
        # Embedding layer
        embedding_weights = graph.constant(
            self._create_embedding_weights(vocab_size, hidden_size)
        )
        embeddings = ops.gather(embedding_weights, token_ids, axis=0)
        
        # Position embeddings
        position_weights = graph.constant(
            self._create_position_weights(seq_length, hidden_size)
        )
        position_ids = ops.range(graph, 0, seq_length, 1, dtype=max.DType.int32)
        position_embeddings = ops.gather(position_weights, position_ids, axis=0)
        
        # Add embeddings
        hidden_states = ops.add(embeddings, position_embeddings)
        
        # Layer normalization
        hidden_states = self._layer_norm(graph, hidden_states, hidden_size)
        
        # Transformer layers
        for layer_idx in range(num_layers):
            hidden_states = self._transformer_layer(
                graph, hidden_states, layer_idx, 
                hidden_size, num_heads, batch_size, seq_length
            )
        
        # Final layer normalization
        hidden_states = self._layer_norm(graph, hidden_states, hidden_size)
        
        # Language modeling head
        lm_head_weights = graph.constant(
            self._create_lm_head_weights(hidden_size, vocab_size)
        )
        logits = ops.matmul(hidden_states, lm_head_weights)
        
        # Output
        output = graph.output(logits)
        
        # Apply optimizations
        optimized_graph = self._apply_optimizations(graph)
        
        self.graph = optimized_graph
        return optimized_graph
    
    def _transformer_layer(self, graph: Graph, hidden_states, layer_idx: int,
                          hidden_size: int, num_heads: int, 
                          batch_size: int, seq_length: int):
        """Create a transformer layer with multi-head attention."""
        
        # Multi-head attention
        attention_output = self._multi_head_attention(
            graph, hidden_states, layer_idx, 
            hidden_size, num_heads, batch_size, seq_length
        )
        
        # Residual connection + layer norm
        hidden_states = ops.add(hidden_states, attention_output)
        hidden_states = self._layer_norm(graph, hidden_states, hidden_size)
        
        # Feed forward network
        ff_output = self._feed_forward(graph, hidden_states, layer_idx, hidden_size)
        
        # Residual connection + layer norm
        hidden_states = ops.add(hidden_states, ff_output)
        hidden_states = self._layer_norm(graph, hidden_states, hidden_size)
        
        return hidden_states
    
    def _multi_head_attention(self, graph: Graph, hidden_states, layer_idx: int,
                             hidden_size: int, num_heads: int,
                             batch_size: int, seq_length: int):
        """Multi-head attention implementation."""
        head_dim = hidden_size // num_heads
        
        # Query, Key, Value projections
        q_weights = graph.constant(self._create_attention_weights(hidden_size, hidden_size, f"q_{layer_idx}"))
        k_weights = graph.constant(self._create_attention_weights(hidden_size, hidden_size, f"k_{layer_idx}"))
        v_weights = graph.constant(self._create_attention_weights(hidden_size, hidden_size, f"v_{layer_idx}"))
        
        queries = ops.matmul(hidden_states, q_weights)
        keys = ops.matmul(hidden_states, k_weights)
        values = ops.matmul(hidden_states, v_weights)
        
        # Reshape for multi-head attention
        queries = ops.reshape(queries, (batch_size, seq_length, num_heads, head_dim))
        keys = ops.reshape(keys, (batch_size, seq_length, num_heads, head_dim))
        values = ops.reshape(values, (batch_size, seq_length, num_heads, head_dim))
        
        # Transpose for attention computation
        queries = ops.transpose(queries, (0, 2, 1, 3))  # (batch, heads, seq, head_dim)
        keys = ops.transpose(keys, (0, 2, 1, 3))
        values = ops.transpose(values, (0, 2, 1, 3))
        
        # Attention scores
        scale = graph.constant(np.array(1.0 / np.sqrt(head_dim), dtype=np.float32))
        scores = ops.matmul(queries, ops.transpose(keys, (0, 1, 3, 2)))
        scores = ops.multiply(scores, scale)
        
        # Apply causal mask
        mask = self._create_causal_mask(graph, seq_length)
        scores = ops.add(scores, mask)
        
        # Softmax attention weights
        attention_weights = ops.softmax(scores, axis=-1)
        
        # Apply attention to values
        attention_output = ops.matmul(attention_weights, values)
        
        # Transpose back and reshape
        attention_output = ops.transpose(attention_output, (0, 2, 1, 3))
        attention_output = ops.reshape(attention_output, (batch_size, seq_length, hidden_size))
        
        # Output projection
        out_weights = graph.constant(self._create_attention_weights(hidden_size, hidden_size, f"out_{layer_idx}"))
        attention_output = ops.matmul(attention_output, out_weights)
        
        return attention_output
    
    def _feed_forward(self, graph: Graph, hidden_states, layer_idx: int, hidden_size: int):
        """Feed forward network implementation."""
        intermediate_size = hidden_size * 4  # Standard transformer ratio
        
        # First linear layer
        w1 = graph.constant(self._create_ff_weights(hidden_size, intermediate_size, f"w1_{layer_idx}"))
        intermediate = ops.matmul(hidden_states, w1)
        
        # Activation function (GELU)
        intermediate = self._gelu_activation(graph, intermediate)
        
        # Second linear layer
        w2 = graph.constant(self._create_ff_weights(intermediate_size, hidden_size, f"w2_{layer_idx}"))
        output = ops.matmul(intermediate, w2)
        
        return output
    
    def _layer_norm(self, graph: Graph, input_tensor, hidden_size: int):
        """Layer normalization implementation."""
        # Compute mean and variance
        mean = ops.reduce_mean(input_tensor, axes=[-1], keepdims=True)
        variance = ops.reduce_mean(
            ops.square(ops.subtract(input_tensor, mean)), 
            axes=[-1], keepdims=True
        )
        
        # Normalize
        epsilon = graph.constant(np.array(1e-5, dtype=np.float32))
        normalized = ops.divide(
            ops.subtract(input_tensor, mean),
            ops.sqrt(ops.add(variance, epsilon))
        )
        
        # Scale and shift (simplified - would use learnable parameters)
        gamma = graph.constant(np.ones((hidden_size,), dtype=np.float32))
        beta = graph.constant(np.zeros((hidden_size,), dtype=np.float32))
        
        output = ops.add(ops.multiply(normalized, gamma), beta)
        return output
    
    def _gelu_activation(self, graph: Graph, input_tensor):
        """GELU activation function implementation."""
        # GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
        const_sqrt_2_pi = graph.constant(np.array(np.sqrt(2.0 / np.pi), dtype=np.float32))
        const_coeff = graph.constant(np.array(0.044715, dtype=np.float32))
        const_half = graph.constant(np.array(0.5, dtype=np.float32))
        const_one = graph.constant(np.array(1.0, dtype=np.float32))
        
        x_cubed = ops.multiply(ops.multiply(input_tensor, input_tensor), input_tensor)
        inner = ops.add(input_tensor, ops.multiply(const_coeff, x_cubed))
        tanh_arg = ops.multiply(const_sqrt_2_pi, inner)
        tanh_result = ops.tanh(tanh_arg)
        
        gelu_result = ops.multiply(
            ops.multiply(const_half, input_tensor),
            ops.add(const_one, tanh_result)
        )
        
        return gelu_result
    
    def _create_causal_mask(self, graph: Graph, seq_length: int):
        """Create causal attention mask."""
        # Create lower triangular mask
        mask_np = np.triu(np.ones((seq_length, seq_length)) * -1e9, k=1)
        mask = graph.constant(mask_np.astype(np.float32))
        return mask
    
    def _apply_optimizations(self, graph: Graph) -> Graph:
        """Apply MAX Engine optimizations."""
        optimization_level = self.config.get('optimization_level', 2)
        target_device = self.config.get('target_device', 'auto')
        
        # Configure optimization passes
        optimization_config = max.OptimizationConfig()
        
        if optimization_level >= 1:
            optimization_config.enable_operator_fusion = True
            optimization_config.enable_constant_folding = True
        
        if optimization_level >= 2:
            optimization_config.enable_memory_optimization = True
            optimization_config.enable_layout_optimization = True
        
        if optimization_level >= 3:
            optimization_config.enable_aggressive_optimization = True
            optimization_config.enable_kernel_specialization = True
        
        # Apply quantization if specified
        if self.config.get('quantization', False):
            quantization_encoding = self._get_quantization_encoding()
            optimization_config.quantization = quantization_encoding
        
        # Compile with optimizations
        optimized_graph = graph.compile(
            optimization_config=optimization_config,
            target_device=target_device
        )
        
        return optimized_graph
    
    def _get_quantization_encoding(self) -> QuantizationEncoding:
        """Get quantization encoding configuration."""
        quant_type = self.config.get('quantization_type', 'int8')
        
        if quant_type == 'int8':
            return max.graph.quantization.Int8Encoding()
        elif quant_type == 'int4':
            return max.graph.quantization.Int4Encoding() 
        elif quant_type == 'fp16':
            return max.graph.quantization.Float16Encoding()
        else:
            return Float32Encoding()
    
    def create_inference_session(self) -> InferenceSession:
        """Create optimized inference session."""
        if self.graph is None:
            raise ValueError("Graph not created. Call create_language_model_graph() first.")
        
        session_config = max.SessionConfig()
        session_config.num_threads = self.config.get('num_threads', 4)
        session_config.memory_pool_size = self.config.get('memory_pool_size', 1024 * 1024 * 1024)  # 1GB
        
        self.session = InferenceSession(self.graph, session_config)
        return self.session
    
    def benchmark_inference(self, num_iterations: int = 100) -> Dict[str, float]:
        """Benchmark inference performance."""
        if self.session is None:
            self.create_inference_session()
        
        # Create sample input
        batch_size = self.config.get('batch_size', 1)
        seq_length = self.config.get('seq_length', 512)
        sample_input = np.random.randint(0, 1000, (batch_size, seq_length), dtype=np.int32)
        
        # Warmup
        for _ in range(10):
            _ = self.session.run([sample_input])
        
        # Benchmark
        import time
        start_time = time.time()
        
        for _ in range(num_iterations):
            outputs = self.session.run([sample_input])
        
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time_ms = (total_time / num_iterations) * 1000.0
        throughput = num_iterations / total_time
        
        return {
            'avg_inference_time_ms': avg_time_ms,
            'throughput_inferences_per_sec': throughput,
            'total_time_sec': total_time,
            'iterations': num_iterations
        }
    
    # Weight creation helper methods (simplified implementations)
    def _create_embedding_weights(self, vocab_size: int, hidden_size: int) -> np.ndarray:
        """Create embedding weight matrix."""
        return np.random.normal(0, 0.02, (vocab_size, hidden_size)).astype(np.float32)
    
    def _create_position_weights(self, seq_length: int, hidden_size: int) -> np.ndarray:
        """Create positional embedding weights."""
        return np.random.normal(0, 0.02, (seq_length, hidden_size)).astype(np.float32)
    
    def _create_attention_weights(self, input_size: int, output_size: int, name: str) -> np.ndarray:
        """Create attention weight matrix."""
        return np.random.normal(0, 0.02, (input_size, output_size)).astype(np.float32)
    
    def _create_ff_weights(self, input_size: int, output_size: int, name: str) -> np.ndarray:
        """Create feed-forward weight matrix."""
        return np.random.normal(0, 0.02, (input_size, output_size)).astype(np.float32)
    
    def _create_lm_head_weights(self, hidden_size: int, vocab_size: int) -> np.ndarray:
        """Create language modeling head weights."""
        return np.random.normal(0, 0.02, (hidden_size, vocab_size)).astype(np.float32)

def create_optimized_graph(model_config: Dict[str, Any]) -> Graph:
    """Factory function to create optimized MAX graph."""
    graph_builder = OptimizedMAXGraph(model_config)
    
    if model_config.get('model_type') == 'language_model':
        return graph_builder.create_language_model_graph()
    else:
        raise ValueError(f"Unsupported model type: {model_config.get('model_type')}")

def deploy_max_model(model_config: Dict[str, Any]) -> Dict[str, Any]:
    """Deploy model using MAX Engine."""
    try:
        # Create optimized graph
        graph_builder = OptimizedMAXGraph(model_config)
        graph = graph_builder.create_language_model_graph()
        
        # Create inference session
        session = graph_builder.create_inference_session()
        
        # Run benchmark
        benchmark_results = graph_builder.benchmark_inference()
        
        return {
            'success': True,
            'model_id': model_config.get('model_id', 'unknown'),
            'graph_compiled': True,
            'session_created': True,
            'benchmark_results': benchmark_results,
            'optimization_level': model_config.get('optimization_level', 2),
            'target_device': model_config.get('target_device', 'auto'),
            'quantization': model_config.get('quantization', False)
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'model_id': model_config.get('model_id', 'unknown')
        }

# Example usage and testing
if __name__ == "__main__":
    # Example model configuration
    config = {
        'model_id': 'test_llama_7b',
        'model_type': 'language_model',
        'batch_size': 1,
        'seq_length': 512,
        'vocab_size': 32000,
        'hidden_size': 4096,
        'num_layers': 32,
        'num_heads': 32,
        'optimization_level': 2,
        'target_device': 'auto',
        'quantization': False,
        'num_threads': 4
    }
    
    print("Creating optimized MAX graph...")
    result = deploy_max_model(config)
    
    if result['success']:
        print("✅ MAX model deployment successful!")
        print(f"Average inference time: {result['benchmark_results']['avg_inference_time_ms']:.2f} ms")
        print(f"Throughput: {result['benchmark_results']['throughput_inferences_per_sec']:.2f} inferences/sec")
    else:
        print(f"❌ MAX model deployment failed: {result['error']}")
