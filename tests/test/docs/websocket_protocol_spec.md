# WebSocket Protocol Specification for Streaming Inference

**Version:** 1.0.0  
**Last Updated:** March 6, 2025

## Overview

This document defines the WebSocket protocol for streaming inference in the framework. The protocol enables real-time token streaming from server to client, and supports bidirectional communication for controlling the generation process.

## Protocol Design Principles

The protocol is designed with the following principles:

1. **Simplicity**: Use a straightforward JSON-based message format
2. **Efficiency**: Minimize overhead for streaming token generation
3. **Extensibility**: Allow for future extensions without breaking compatibility
4. **Control**: Enable clients to control the generation process in real-time
5. **Error Handling**: Provide robust error handling and recovery mechanisms

## Connection Establishment

The WebSocket connection is established with a standard WebSocket handshake. The server endpoint should be configured according to your application needs. For example:

```
ws://example.com/generate
wss://example.com/generate  # Secure WebSocket
```

## Message Format

All messages are JSON objects with a required `type` field that indicates the message type. Additional fields depend on the message type.

## Client → Server Messages

### 1. Configuration Message

Sent by the client to configure the generation process. Must be the first message sent after connection establishment.

```json
{
  "type": "config",
  "model": "llama-7b",           // Optional - Use the server's default model if omitted
  "prompt": "Once upon a time",  // Required - Initial prompt text
  "parameters": {                // Optional - Generation parameters
    "max_tokens": 100,           // Maximum tokens to generate
    "temperature": 0.7,          // Temperature for sampling
    "top_p": 0.9,                // Top-p (nucleus) sampling parameter
    "top_k": 40,                 // Top-k sampling parameter
    "repetition_penalty": 1.1,   // Penalty for repeated tokens
    "stop": ["\n\n", "THE END"]  // Stop generation when these strings are generated
  },
  "options": {                   // Optional - Additional options
    "stream": true,              // Should be true for streaming responses
    "echo_prompt": false,        // Whether to echo the prompt in the output
    "include_token_ids": false,  // Whether to include token IDs in the response
    "include_logprobs": false,   // Whether to include token logprobs
    "include_timing": false      // Whether to include timing information
  }
}
```

### 2. Control Message

Sent by the client to control the ongoing generation process.

```json
{
  "type": "control",
  "action": "stop"               // Stop the generation process
}
```

**Supported actions:**
- `"stop"`: Stop the generation process
- `"pause"`: Pause generation (if supported by the model)
- `"resume"`: Resume generation after pause
- `"regenerate"`: Restart generation with current configuration
- `"modify"`: Modify the generation process (see below)

### 3. Modification Message

Sent by the client to modify generation parameters during generation.

```json
{
  "type": "control",
  "action": "modify",
  "parameters": {                // Parameters to modify
    "temperature": 0.9,          // Only parameters to change need to be included
    "top_p": 0.95
  }
}
```

### 4. Feedback Message

Sent by the client to provide feedback on the generated content.

```json
{
  "type": "feedback",
  "rating": 1,                   // Rating from -1 (negative) to 1 (positive)
  "text": "Great response!",     // Optional text feedback
  "position": 120                // Optional position in the generation sequence
}
```

## Server → Client Messages

### 1. Initialization Message

Sent by the server after receiving the configuration message, before starting the generation.

```json
{
  "type": "init",
  "request_id": "gen_12345",     // Unique ID for this generation
  "model": "llama-7b",           // Model being used
  "model_info": {                // Optional model information
    "version": "v2",
    "max_tokens": 4096,
    "quantization": "int4"
  }
}
```

### 2. Token Message

Sent by the server for each generated token.

```json
{
  "type": "token",
  "token": " the",                          // Generated token text
  "token_id": 262,                          // Token ID (if requested)
  "generated_text": "Once upon a time the", // Cumulative generated text so far
  "logprobs": [                             // Included if requested
    {"token": " the", "logprob": -0.3201},
    {"token": " a", "logprob": -1.2341}
  ],
  "timing": {                               // Included if requested
    "token_ms": 42.5,                       // Time to generate this token
    "total_ms": 156.3                       // Total time elapsed
  },
  "finish_reason": null,                    // null if not finished
  "finished": false                         // false if generation continues
}
```

### 3. Completion Message

Sent by the server when generation completes.

```json
{
  "type": "completion",
  "generated_text": "Once upon a time there was a magical forest...",
  "finish_reason": "length",     // "length", "stop", "content_filter", etc.
  "usage": {
    "prompt_tokens": 4,
    "completion_tokens": 100,
    "total_tokens": 104
  },
  "timing": {                    // Included if requested
    "first_token_ms": 45.2,      // Time to first token
    "total_ms": 1240.5           // Total generation time
  }
}
```

### 4. Error Message

Sent by the server when an error occurs.

```json
{
  "type": "error",
  "error": "memory_pressure",     // Error type
  "message": "WebGPU memory limit exceeded",  // Human-readable message
  "recoverable": true,            // Whether the error is recoverable
  "recovery_action": "retry",     // Suggested recovery action
  "request_id": "gen_12345"       // Request ID from init
}
```

**Common error types:**
- `"invalid_request"`: Invalid request format or parameters
- `"context_length_exceeded"`: Prompt too long for model context
- `"content_filter"`: Content filtered by safety system
- `"memory_pressure"`: System under memory pressure
- `"timeout"`: Operation timed out
- `"internal_error"`: Unspecified internal error
- `"model_loading_error"`: Failed to load model
- `"rate_limited"`: Too many requests

### 5. Status Message

Sent by the server to provide status updates during long operations.

```json
{
  "type": "status",
  "operation": "model_loading",    // Current operation
  "progress": 67,                  // Progress percentage (0-100)
  "message": "Loading language model components",  // Status message
  "eta_seconds": 3.5               // Estimated time remaining
}
```

## Example Communication Flow

### Basic Text Generation

1. Client establishes WebSocket connection
2. Client sends Configuration Message:
   ```json
   {
     "type": "config",
     "prompt": "Write a short story about a robot learning to paint.",
     "parameters": {
       "max_tokens": 200,
       "temperature": 0.7
     },
     "options": {
       "stream": true
     }
   }
   ```

3. Server sends Initialization Message:
   ```json
   {
     "type": "init",
     "request_id": "gen_12345",
     "model": "llama-7b"
   }
   ```

4. Server streams Token Messages:
   ```json
   {"type": "token", "token": " In", "finished": false}
   {"type": "token", "token": " the", "finished": false}
   {"type": "token", "token": " year", "finished": false}
   ...
   ```

5. Server sends Completion Message when done:
   ```json
   {
     "type": "completion",
     "generated_text": "In the year 2045, a robot named...",
     "finish_reason": "length",
     "usage": {
       "prompt_tokens": 9,
       "completion_tokens": 200,
       "total_tokens": 209
     }
   }
   ```

### Error and Recovery

1. Client establishes WebSocket connection
2. Client sends Configuration Message
3. Server sends Initialization Message
4. Server begins streaming Token Messages
5. Server encounters error and sends Error Message:
   ```json
   {
     "type": "error",
     "error": "memory_pressure",
     "message": "WebGPU memory limit exceeded",
     "recoverable": true,
     "recovery_action": "retry",
     "request_id": "gen_12345"
   }
   ```

6. Client waits briefly and sends new Configuration Message with reduced parameters:
   ```json
   {
     "type": "config",
     "prompt": "Write a short story about a robot learning to paint.",
     "parameters": {
       "max_tokens": 100,  // Reduced from 200
       "temperature": 0.7
     },
     "options": {
       "stream": true
     }
   }
   ```

7. Server resumes generation with new parameters

### Midstream Control

1. Client establishes WebSocket connection
2. Client sends Configuration Message
3. Server sends Initialization Message
4. Server begins streaming Token Messages
5. Client decides to adjust generation and sends Control Message:
   ```json
   {
     "type": "control",
     "action": "modify",
     "parameters": {
       "temperature": 0.9  // Increase randomness
     }
   }
   ```

6. Server acknowledges and continues generation with updated parameters
7. Client decides to stop generation and sends Control Message:
   ```json
   {
     "type": "control",
     "action": "stop"
   }
   ```

8. Server stops generation and sends final Completion Message

## Implementation Considerations

### WebSocket Libraries

The protocol can be implemented using standard WebSocket libraries:

#### Server-side (Python)

```python
# Using FastAPI with WebSockets
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import json
from fixed_web_platform.webgpu_streaming_inference import WebGPUStreamingInference

app = FastAPI()
streaming_model = WebGPUStreamingInference(model_path="models/llama-7b")

@app.websocket("/generate")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    try:
        # Wait for configuration message
        data = await websocket.receive_text()
        config = json.loads(data)
        
        if config["type"] != "config":
            await websocket.send_json({
                "type": "error",
                "error": "invalid_request",
                "message": "First message must be a config message",
                "recoverable": False
            })
            return
        
        # Send initialization message
        await websocket.send_json({
            "type": "init",
            "request_id": f"gen_{id(websocket)}",
            "model": "llama-7b"
        })
        
        # Start generation with streaming
        async def send_token(token):
            await websocket.send_json({
                "type": "token",
                "token": token,
                "finished": False
            })
        
        result = await streaming_model.generate_async(
            config["prompt"],
            max_tokens=config["parameters"].get("max_tokens", 100),
            callback=send_token
        )
        
        # Send completion message
        await websocket.send_json({
            "type": "completion",
            "generated_text": result,
            "finish_reason": "stop",
            "usage": {
                "prompt_tokens": len(config["prompt"].split()),
                "completion_tokens": len(result.split()),
                "total_tokens": len(config["prompt"].split()) + len(result.split())
            }
        })
        
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        try:
            await websocket.send_json({
                "type": "error",
                "error": "internal_error",
                "message": str(e),
                "recoverable": False
            })
        except:
            pass
```

#### Client-side (JavaScript)

```javascript
// Using browser WebSocket API
const socket = new WebSocket('wss://example.com/generate');
let generatedText = '';

socket.onopen = () => {
  // Send configuration message
  socket.send(JSON.stringify({
    type: 'config',
    prompt: 'Write a short story about a robot learning to paint.',
    parameters: {
      max_tokens: 200,
      temperature: 0.7
    },
    options: {
      stream: true
    }
  }));
};

socket.onmessage = (event) => {
  const message = JSON.parse(event.data);
  
  switch (message.type) {
    case 'init':
      console.log(`Generation started with ID: ${message.request_id}`);
      break;
      
    case 'token':
      // Display token in UI
      generatedText += message.token;
      document.getElementById('output').textContent = generatedText;
      break;
      
    case 'completion':
      console.log('Generation complete:', message.generated_text);
      // Update UI with final text and stats
      document.getElementById('stats').textContent = 
        `Tokens: ${message.usage.total_tokens}, Reason: ${message.finish_reason}`;
      break;
      
    case 'error':
      console.error('Error:', message.message);
      // Show error in UI
      document.getElementById('error').textContent = message.message;
      
      // Implement recovery if possible
      if (message.recoverable && message.recovery_action === 'retry') {
        setTimeout(() => {
          // Retry with reduced parameters
          socket.send(JSON.stringify({
            type: 'config',
            prompt: 'Write a short story about a robot learning to paint.',
            parameters: {
              max_tokens: 100,  // Reduced
              temperature: 0.7
            },
            options: {
              stream: true
            }
          }));
        }, 1000);
      }
      break;
      
    case 'status':
      // Update UI with status
      document.getElementById('status').textContent = 
        `${message.operation}: ${message.progress}% - ${message.message}`;
      break;
  }
};

// Function to stop generation
function stopGeneration() {
  socket.send(JSON.stringify({
    type: 'control',
    action: 'stop'
  }));
}

// Function to adjust parameters
function increaseRandomness() {
  socket.send(JSON.stringify({
    type: 'control',
    action: 'modify',
    parameters: {
      temperature: 0.9
    }
  }));
}
```

### Error Handling and Recovery

Implementations should consider these error handling strategies:

1. **Client reconnection**: Clients should implement automatic reconnection with exponential backoff
2. **State tracking**: Servers should maintain generation state to support resumption
3. **Partial results**: Always return partial results with errors so clients can resume
4. **Parameter validation**: Validate and normalize parameters before starting generation
5. **Resource monitoring**: Monitor system resources and degrade gracefully under pressure

### Security Considerations

When implementing this protocol:

1. **Authentication**: Use secure authentication methods for production systems
2. **Rate limiting**: Implement rate limiting to prevent abuse
3. **Input validation**: Validate all client inputs before processing
4. **Content filtering**: Apply appropriate content filtering rules
5. **Resource limits**: Set appropriate timeouts and resource limits

## WebSocket vs. Server-Sent Events

This protocol uses WebSockets for bidirectional communication. For unidirectional streaming where client control is not needed, Server-Sent Events (SSE) may be a simpler alternative. The message format defined here can be adapted for SSE.

## Compatibility with REST APIs

For compatibility with existing REST APIs, you can implement a similar JSON structure for non-streaming responses. For example:

```json
// POST /v1/generate
{
  "prompt": "Write a short story about a robot learning to paint.",
  "max_tokens": 200,
  "temperature": 0.7,
  "stream": false
}

// Response
{
  "id": "gen_12345",
  "model": "llama-7b",
  "generated_text": "In the year 2045, a robot named...",
  "usage": {
    "prompt_tokens": 9,
    "completion_tokens": 200,
    "total_tokens": 209
  }
}
```

## Versioning

This specification is versioned. Future versions will maintain backward compatibility when possible. Version negotiation can be added in a future update if needed.

## Related Documentation

- [WebGPUStreamingInference API Reference](api_reference/webgpu_streaming_inference.md)
- [Unified Framework API Reference](unified_framework_api.md)
- [Error Handling Guide](ERROR_HANDLING_GUIDE.md)
- [WebGPU Streaming Documentation](WEBGPU_STREAMING_DOCUMENTATION.md)