"""
Distributed Inference with IPFS Accelerate

This module provides prompts for IPFS Accelerate's distributed inference capabilities.
"""

import logging
from typing import Dict, List, Any

logger = logging.getLogger("ipfs_accelerate_mcp.prompts.distributed_inference")

def register_prompts(mcp) -> None:
    """
    Register distributed inference prompts with MCP
    
    Args:
        mcp: FastMCPServer instance
    """
    logger.info("Registering distributed inference prompts")
    
    @mcp.prompt("distributed_inference_guide")
    def distributed_inference_guide() -> str:
        """
        Guide for distributed inference
        
        This prompt provides guidance on using IPFS Accelerate's distributed inference capabilities.
        """
        return """
        # Distributed Inference with IPFS Accelerate

        IPFS Accelerate provides powerful distributed inference capabilities, allowing you to run large language models across multiple devices.

        ## Basic Concepts

        **Distributed Inference** allows you to:
        - Run larger models than would fit on a single device
        - Parallelize inference for faster processing
        - Combine heterogeneous hardware for optimal performance

        ## Setting Up Distributed Inference

        ```python
        from ipfs_accelerate_py import get_instance
        
        # Initialize IPFS Accelerate with distributed mode
        ipfs = get_instance(distributed_mode=True)
        
        # Connect to distributed nodes
        ipfs.connect_nodes(["node1.example.com", "node2.example.com"])
        
        # Check connected nodes
        nodes = ipfs.list_nodes()
        print(f"Connected to {len(nodes)} nodes")
        ```

        ## Running Distributed Inference

        ```python
        # Load a large model in distributed mode
        model_id = "meta-llama/Llama-2-70b"
        result = ipfs.run_inference(
            model_id=model_id,
            inputs=["Explain how IPFS works"],
            distributed=True
        )
        
        # The model will automatically be sharded across available nodes
        print(result)
        ```

        ## Advanced Configuration

        You can customize how models are distributed:

        ```python
        # Specify node allocation preferences
        config = {
            "sharding_strategy": "auto",  # or "tensor", "pipeline", "expert"
            "node_preferences": {
                "node1.example.com": ["attention_layers"],
                "node2.example.com": ["ffn_layers"]
            }
        }
        
        # Run with custom configuration
        result = ipfs.run_inference(
            model_id=model_id,
            inputs=["Summarize the benefits of distributed computing"],
            distributed_config=config
        )
        ```

        ## Monitoring Distributed Inference

        To monitor the performance of distributed inference:

        ```python
        # Get distribution statistics
        stats = ipfs.get_distributed_stats(model_id)
        
        # Print node utilization
        for node, metrics in stats["nodes"].items():
            print(f"Node: {node}")
            print(f"  Utilization: {metrics['utilization']:.1f}%")
            print(f"  Memory used: {metrics['memory_used_gb']:.1f}GB")
            print(f"  Throughput: {metrics['tokens_per_second']:.1f} tokens/sec")
        ```

        ## Best Practices

        1. **Balanced Hardware**: For best results, use nodes with similar hardware capabilities
        2. **Network Bandwidth**: Ensure high-bandwidth connections between nodes
        3. **Model Selection**: Some models are more amenable to distribution than others
        4. **Failure Handling**: Configure fallback options for node failures
        5. **Monitoring**: Regularly monitor node performance and resource utilization

        For more detailed information, see the full IPFS Accelerate Distributed Inference documentation.
        """
    
    @mcp.prompt("ipfs_model_parallelism")
    def ipfs_model_parallelism() -> str:
        """
        Guide for IPFS model parallelism
        
        This prompt explains how IPFS Accelerate implements model parallelism.
        """
        return """
        # IPFS Accelerate Model Parallelism Guide
        
        IPFS Accelerate implements several forms of model parallelism to efficiently distribute large language models.
        
        ## Parallelism Types
        
        IPFS Accelerate supports multiple parallelism strategies:
        
        1. **Tensor Parallelism**: Splits individual tensors across devices
           - Best for: Models with large dense layers
           - Example: Splitting attention heads across GPUs
        
        2. **Pipeline Parallelism**: Splits the model by layers
           - Best for: Deep models with many sequential layers
           - Example: Early layers on one device, later layers on another
        
        3. **Expert Parallelism**: Used with Mixture of Experts models
           - Best for: MoE models like Mixtral
           - Example: Different expert networks on different devices
        
        4. **Data Parallelism**: Processes different batches in parallel
           - Best for: High-throughput batch processing
           - Example: Processing multiple prompts simultaneously
        
        ## Choosing the Right Strategy
        
        IPFS Accelerate can automatically select the optimal strategy based on your model and available hardware:
        
        ```python
        # Let IPFS Accelerate choose the best strategy
        result = ipfs.run_inference(
            model_id="mistralai/Mixtral-8x7B-v0.1",
            inputs=["Explain model parallelism"],
            distributed=True,
            auto_parallelize=True
        )
        ```
        
        Or you can manually specify the strategy:
        
        ```python
        # Manually select tensor parallelism
        result = ipfs.run_inference(
            model_id="mistralai/Mixtral-8x7B-v0.1",
            inputs=["Explain model parallelism"],
            distributed=True,
            parallelism_strategy="tensor"
        )
        ```
        
        ## Advanced Configuration
        
        For fine-grained control over model distribution:
        
        ```python
        # Detailed parallelism configuration
        config = {
            "tensor_parallel_size": 2,
            "pipeline_parallel_size": 2,
            "expert_parallel_size": 4,
            "data_parallel_size": 1,
            "communication_pattern": "nccl",
            "memory_optimization_level": "moderate"
        }
        
        # Apply configuration
        ipfs.configure_parallelism(config)
        ```
        
        ## Hardware Considerations
        
        Different parallelism strategies have different hardware requirements:
        
        1. **Tensor Parallelism**: Requires high-bandwidth interconnect (NVLink, InfiniBand)
        2. **Pipeline Parallelism**: More tolerant of lower bandwidth but sensitive to latency
        3. **Expert Parallelism**: Works well with heterogeneous hardware
        4. **Data Parallelism**: Requires similar devices for balanced performance
        
        Use `ipfs.evaluate_hardware_topology()` to analyze your hardware configuration and get recommendations for optimal parallelism strategies.
        """
    
    logger.info("Distributed inference prompts registered successfully")
