# Using IPFS Accelerate MCP with Language Models

This guide demonstrates how to use the IPFS Accelerate MCP server with language models like Anthropic Claude or OpenAI's models.

## Introduction

The Model Context Protocol (MCP) allows language models to access external tools and resources in a standardized way. By exposing IPFS Accelerate's capabilities through MCP, language models can:

1. Access hardware information
2. Run inference on ML models
3. Manage inference endpoints
4. Optimize inference performance

## Example Interactions with Anthropic Claude

Here are some example conversations with Claude showing how it might use IPFS Accelerate capabilities through MCP:

### Example 1: Hardware Detection

```
Human: What hardware accelerators are available on my system?

Claude: Let me check the available hardware accelerators on your system.

[Claude is using the test_hardware tool from IPFS Accelerate]

Based on the hardware detection, your system has the following accelerators available:

- CPU: Intel(R) Xeon(R) CPU @ 2.20GHz (8 cores)
- CUDA: Available (1 device)
  - NVIDIA Tesla T4 (16GB VRAM)

OpenVINO, ROCm, WebGPU, and WebNN are not available on this system.
```

### Example 2: Running Inference

```
Human: Can you generate an embedding for the phrase "machine learning with IPFS" using a model available on my system?

Claude: I'll generate an embedding for that phrase using one of your available models. Let me check what models are available first.

[Claude is accessing the models://available resource]

I found several embedding models available on your system. I'll use the BAAI/bge-small-en-v1.5 model to generate the embedding.

[Claude is using the run_inference tool from IPFS Accelerate]

I've successfully generated an embedding for "machine learning with IPFS". The embedding is a 384-dimensional vector, which I won't display in full, but here are the first few values:

[0.0235, -0.0124, 0.0358, 0.0147, ...]

This embedding can be used for semantic search, clustering, or other NLP tasks. The inference was performed on your CUDA device, which completed the operation in 13ms.
```

### Example 3: Hardware Recommendations

```
Human: I want to run a 7B parameter language model on my system. What hardware would you recommend?

Claude: Let me analyze your system and provide hardware recommendations for running a 7B parameter language model.

[Claude is using the get_hardware_recommendation tool from IPFS Accelerate]

Based on your system's hardware and the 7B parameter model size, here are my recommendations:

Best recommendation: CUDA GPU
- Performance: High
- Suitability: Yes, your GPU can handle this model
- Memory required: Approximately 28GB
- Note: Your NVIDIA T4 has 16GB VRAM, so you'll need to use CPU offloading or model quantization

Alternative options:
1. CPU
   - Performance: Low
   - Suitability: Yes, but will be slow
   - Memory required: Approximately 35GB
   - Note: You have 64GB system RAM, so this is feasible but not optimal

I recommend using your CUDA GPU with 4-bit quantization to reduce the memory footprint to around 14GB, which will fit on your GPU while maintaining reasonable performance.
```

### Example 4: Configuring and Testing a New Endpoint

```
Human: Can you help me set up a new endpoint for the Mistral-7B model on my CPU?

Claude: I'll help you set up a new endpoint for Mistral-7B on your CPU. First, let me check if there are any existing endpoints for this model.

[Claude is using the list_endpoints tool from IPFS Accelerate]

I don't see any existing endpoints for Mistral-7B. Let's create a new one.

[Claude is using the add_endpoint tool from IPFS Accelerate]

I've added a new endpoint for Mistral-7B on your CPU with the following configuration:
- Model: mistralai/Mistral-7B-v0.1
- Device: cpu
- Max batch size: 1
- Max sequence length: 2048

Now, let's test if this endpoint is working properly.

[Claude is using the test_endpoint tool from IPFS Accelerate]

Great! The endpoint test was successful. The model loaded correctly and can process inputs. However, I should note that inference with this 7B parameter model on CPU will be relatively slow. For a short prompt, it took approximately 3.5 seconds to generate a response.

Would you like me to try setting up a more optimized version of this model? I could help you set up a quantized version which would run faster on your CPU.
```

## Setting Up MCP with Your Language Model

To use IPFS Accelerate MCP with your preferred language model:

1. Start the IPFS Accelerate MCP server:
   ```bash
   python -m ipfs_accelerate_py.mcp.standalone
   ```

2. For Anthropic Claude:
   - Use Claude Desktop with MCP support
   - Point it to your MCP server URL (e.g., `http://localhost:8000`)

3. For OpenAI models:
   - Use an MCP-compatible client library
   - Configure it to use your MCP server URL

## Benefits of Using IPFS Accelerate via MCP

1. **Hardware-Aware Inference**: The language model can automatically select the best hardware for different models and tasks.

2. **Distributed Processing**: Access distributed inference across IPFS networks.

3. **Dynamic Configuration**: The language model can optimize configurations based on task requirements.

4. **Interactive Exploration**: Users can interactively explore hardware capabilities and model performance through natural language.
