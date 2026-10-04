#!/usr/bin/env python3
"""
Tutorial: WebGPU Streaming Inference Pipeline

This tutorial demonstrates how to use the WebGPU streaming inference pipeline for token-by-token
generation with various precision options. It covers:

    1. Setting up the WebGPUStreamingInference class
    2. Using callbacks for token-by-token generation
    3. Setting up WebSocket-based streaming
    4. Working with different precision options ()))))))2-bit, 3-bit, 4-bit)
    5. Creating a simple HTTP server for demonstration
    6. Integrating with the unified web framework

    Author: Demo Team
    Date: August 2025
    """

    import os
    import sys
    import json
    import time
    import anyio
    import threading
    import logging
    import http.server
    import socketserver
    import websockets
    from typing import Dict, List, Any, Optional, Union, Callable, Tuple

# Configure logging
    logging.basicConfig()))))))level=logging.INFO, format='%()))))))asctime)s - %()))))))levelname)s - %()))))))message)s')
    logger = logging.getLogger()))))))"streaming_tutorial")

# Add the fixed_web_platform to the path - adjust if needed
    sys.path.append()))))))os.path.join()))))))os.path.dirname()))))))os.path.dirname()))))))__file__)), "fixed_web_platform"))

# Enable simulation mode if not running in a browser environment
    os.environ["WEBGPU_SIMULATION"] = "1"
    ,
# Import the streaming inference module:
try:
    from test.web_platform.webgpu_streaming_inference import ()))))))
    WebGPUStreamingInference,
    create_streaming_endpoint,
    optimize_for_streaming
    )
    from test.web_platform.webgpu_kv_cache_optimization import create_optimized_kv_cache
    from test.web_platform.unified_web_framework import WebPlatformAccelerator
except ImportError:
    logger.error()))))))"Failed to import WebGPU modules. Make sure you have the fixed_web_platform directory available.")
    raise

# HTML template for the demo page
    HTML_TEMPLATE = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>WebGPU Streaming Inference Demo</title>
    <style>
    body {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
    font-family: Arial, sans-serif;
    max-width: 800px;
    margin: 0 auto;
    padding: 20px;
    line-height: 1.6;
    }
    h1 {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
    color: #333;
    border-bottom: 1px solid #ddd;
    padding-bottom: 10px;
    }
    .container {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
    margin-top: 30px;
    }
    textarea {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
    width: 100%;
    height: 100px;
    margin-bottom: 10px;
    padding: 8px;
    border: 1px solid #ddd;
    border-radius: 4px;
    }
    .controls {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
    display: flex;
    margin-bottom: 20px;
    }
    button {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
    padding: 8px 16px;
    background-color: #4CAF50;
    color: white;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    }
    button:hover {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
    background-color: #45a049;
    }
    select {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
    margin-right: 10px;
    padding: 8px;
    border-radius: 4px;
    border: 1px solid #ddd;
    }
    .output {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
    border: 1px solid #ddd;
    border-radius: 4px;
    padding: 16px;
    background-color: #f9f9f9;
    min-height: 200px;
    white-space: pre-wrap;
    }
    .stats {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
    margin-top: 20px;
    font-size: 0.9em;
    color: #666;
    }
    .highlight {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
    background-color: #e6f7ff;
    padding: 2px;
    }
    </style>
    </head>
    <body>
    <h1>WebGPU Streaming Inference Demo</h1>
    
    <div class="container">
    <h2>Input</h2>
    <textarea id="prompt" placeholder="Enter your prompt here...">Explain how WebGPU streaming inference works for large language models:</textarea>
        
    <div class="controls">
    <select id="precision">
    <option value="int4">4-bit precision</option>
    <option value="int3">3-bit precision</option>
    <option value="int2">2-bit precision</option>
    </select>
            
    <select id="maxTokens">
    <option value="50">50 tokens</option>
    <option value="100" selected>100 tokens</option>
    <option value="200">200 tokens</option>
    <option value="500">500 tokens</option>
    </select>
            
    <button id="generate">Generate</button>
    <button id="clear">Clear Output</button>
    </div>
        
    <h2>Output</h2>
    <div id="output" class="output"></div>
        
    <div class="stats" id="stats"></div>
    </div>
    
    <script>
    // WebSocket connection
    let socket;
    let generationStartTime;
    let tokenCount = 0;
        
    // Connect to WebSocket server
    function connectWebSocket()))))))) {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
    socket = new WebSocket()))))))'ws://localhost:8765');
            
    socket.onopen = function()))))))e) {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
    console.log()))))))'WebSocket connection established');
    };
            
    socket.onmessage = function()))))))event) {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
    const data = JSON.parse()))))))event.data);
                
    if ()))))))data.type === 'token') {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
    // Append token to output
    document.getElementById()))))))'output').innerText += data.token;
    tokenCount++;
                    
    // Update statistics
    const elapsedTime = ()))))))Date.now()))))))) - generationStartTime) / 1000;
    const tokensPerSecond = tokenCount / elapsedTime;
                    document.getElementById()))))))'stats').innerHTML = :
                        `Tokens: ${}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}tokenCount} | Time: ${}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}elapsedTime.toFixed()))))))2)}s | Speed: ${}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}tokensPerSecond.toFixed()))))))2)} tokens/sec`;
                        }
                        else if ()))))))data.type === 'start') {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                        // Generation started
                        document.getElementById()))))))'output').innerText = '';
                        document.getElementById()))))))'stats').innerHTML = 'Starting generation...';
                        generationStartTime = Date.now())))))));
                        tokenCount = 0;
                    
                        // Display precision information
                        if ()))))))data.using_ultra_low_precision) {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                        document.getElementById()))))))'stats').innerHTML += 
                        `<br>Using ${}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}data.precision_bits}-bit precision ()))))))${}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}data.memory_reduction_percent.toFixed()))))))1)}% memory reduction)`;
                        }
                        }
                        else if ()))))))data.type === 'complete') {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                        // Generation complete
                    document.getElementById()))))))'stats').innerHTML = :
                        `Generation complete | Tokens: ${}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}data.tokens_generated} | Time: ${}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}data.generation_time.toFixed()))))))2)}s | Speed: ${}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}data.tokens_per_second.toFixed()))))))2)} tokens/sec`;
                    
                        if ()))))))data.precision_bits) {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                        document.getElementById()))))))'stats').innerHTML += 
                        `<br>Used ${}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}data.precision_bits}-bit precision with ${}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}data.memory_reduction_percent.toFixed()))))))1)}% memory reduction`;
                        }
                        }
                else if ()))))))data.type === 'error') {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}:
                    document.getElementById()))))))'output').innerText += `\n\nERROR: ${}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}data.message}`;
                    }
                    };
            
                    socket.onclose = function()))))))event) {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                    console.log()))))))'WebSocket connection closed');
                    };
            
                    socket.onerror = function()))))))error) {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                    console.log()))))))'WebSocket error:', error);
                    };
                    }
        
                    // Initialize
                    document.addEventListener()))))))'DOMContentLoaded', function()))))))) {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                    connectWebSocket())))))));
            
                    // Generate button
                    document.getElementById()))))))'generate').addEventListener()))))))'click', function()))))))) {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                    if ()))))))socket && socket.readyState === WebSocket.OPEN) {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                    const prompt = document.getElementById()))))))'prompt').value;
                    const precision = document.getElementById()))))))'precision').value;
                    const maxTokens = parseInt()))))))document.getElementById()))))))'maxTokens').value);
                    
                    // Send generation request
                    socket.send()))))))JSON.stringify())))))){}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}:
                        prompt: prompt,
                        max_tokens: maxTokens,
                        temperature: 0.7,
                        precision: precision
                        }));
                        } else {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                        document.getElementById()))))))'output').innerText = 'WebSocket not connected. Please refresh the page.';
                        }
                        });
            
                        // Clear button
                        document.getElementById()))))))'clear').addEventListener()))))))'click', function()))))))) {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                        document.getElementById()))))))'output').innerText = '';
                        document.getElementById()))))))'stats').innerHTML = '';
                        });
                        });
                        </script>
                        </body>
                        </html>
                        """

class WebServerThread()))))))threading.Thread):
    """A simple HTTP server for the WebSocket client demo."""
    
    def __init__()))))))self, directory: str = ".", port: int = 8000):
        """Initialize the web server thread."""
        super()))))))).__init__()))))))daemon=True)
        self.directory = directory
        self.port = port
        self.server = None
        self.is_running = False
        
        # Create the HTML file
        with open()))))))os.path.join()))))))directory, "streaming_demo.html"), "w") as f:
            f.write()))))))HTML_TEMPLATE)
        
    def run()))))))self):
        """Run the web server."""
        handler = http.server.SimpleHTTPRequestHandler
        
        # Change to the directory to serve
        original_dir = os.getcwd())))))))
        os.chdir()))))))self.directory)
        
        try:
            with socketserver.TCPServer()))))))()))))))"", self.port), handler) as server:
                self.server = server
                self.is_running = True
                logger.info()))))))f"Web server started at http://localhost:{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}self.port}/streaming_demo.html")
                server.serve_forever())))))))
        finally:
            os.chdir()))))))original_dir)
            self.is_running = False
            
    def stop()))))))self):
        """Stop the web server."""
        if self.server:
            logger.info()))))))"Stopping web server")
            self.server.shutdown())))))))

class WebSocketServerManager:
    """Manage the WebSocket server for streaming inference."""
    
    def __init__()))))))self, model_path: str, host: str = "localhost", port: int = 8765):
        """Initialize the WebSocket server manager."""
        self.model_path = model_path
        self.host = host
        self.port = port
        self.loop = None
        self.server = None
        self.server_task = None
        self.thread = None
        self.streaming_handlers = {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}  # Precision: handler mappings
        
    def _create_streaming_handler()))))))self, precision: str):
        """Create streaming inference handler with specific precision."""
        # Create configuration based on precision
        if precision == "int2":
            config = {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "quantization": "int2",
            "optimize_kv_cache": True,
            "latency_optimized": True,
            "adaptive_batch_size": True,
            "prefill_optimized": True
            }
        elif precision == "int3":
            config = {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "quantization": "int3",
            "optimize_kv_cache": True,
            "latency_optimized": True,
            "adaptive_batch_size": True,
            "prefill_optimized": True
            }
        else:  # int4 ()))))))default)
            config = {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "quantization": "int4",
            "optimize_kv_cache": True,
            "latency_optimized": True,
            "adaptive_batch_size": True,
            "prefill_optimized": True
            }
            
        # Create streaming handler
        return WebGPUStreamingInference()))))))self.model_path, config)
    
    async def handle_websocket()))))))self, websocket, path):
        """Handle WebSocket connections."""
        try:
            # Receive request
            request = await websocket.recv())))))))
            request_data = json.loads()))))))request)
            
            # Extract parameters
            prompt = request_data.get()))))))"prompt", "")
            max_tokens = request_data.get()))))))"max_tokens", 100)
            temperature = request_data.get()))))))"temperature", 0.7)
            precision = request_data.get()))))))"precision", "int4")
            
            # Get the appropriate streaming handler
            if precision not in self.streaming_handlers:
                self.streaming_handlers[precision] = self._create_streaming_handler()))))))precision)
                ,
                streaming_handler = self.streaming_handlers[precision]
                ,
            # Stream response
                await streaming_handler.stream_websocket()))))))
                websocket, prompt, max_tokens, temperature
                )
            
        except Exception as e:
            logger.error()))))))f"Error handling WebSocket connection: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}e}")
            try:
                await websocket.send()))))))json.dumps())))))){}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                "type": "error",
                "message": str()))))))e)
                }))
            except:
                pass
    
    async def start_server()))))))self):
        """Start the WebSocket server."""
        self.server = await websockets.serve()))))))self.handle_websocket, self.host, self.port)
        logger.info()))))))f"WebSocket server started at ws://{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}self.host}:{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}self.port}")
        await self.server.wait_closed())))))))
        
    def start()))))))self):
        """Start the WebSocket server in a separate thread."""
        async def run_server()))))))):
            await self.start_server())))))))
            
        def run_in_thread()))))))):
            self.loop = # TODO: Remove event loop management - anyio.run
            # TODO: Remove event loop management - anyio.run
            self.server_task = self.loop.create_task()))))))run_server()))))))))
            self.loop.run_forever())))))))
            
            self.thread = threading.Thread()))))))target=run_in_thread, daemon=True)
            self.thread.start())))))))
        
    def stop()))))))self):
        """Stop the WebSocket server."""
        if self.loop:
            self.loop.call_soon_threadsafe()))))))self.loop.stop)
        if self.thread:
            self.thread.join()))))))timeout=1.0)
            logger.info()))))))"WebSocket server stopped")


def demonstrate_token_callback()))))))):
    """Demonstrate token-by-token generation with a callback."""
    print()))))))"\n\033[1m1. Token-by-Token Generation with Callback\033[0m"),
    print()))))))"-" * 60)
    
    # Create the streaming inference handler with 4-bit precision
    config = {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
    "quantization": "int4",  # Use 4-bit precision
    "optimize_kv_cache": True,
    "latency_optimized": True,
    "adaptive_batch_size": True
    }
    
    # Create handler with the model path ()))))))will be simulated)
    model_path = "models/llama-7b"
    streaming_handler = WebGPUStreamingInference()))))))model_path, config)
    
    # Define callback function to print tokens as they're generated
    def token_callback()))))))token, is_last=False):
        print()))))))token, end="", flush=True)
        if is_last:
            print()))))))"\n\nGeneration complete!\n")
    
    # Run streaming generation with callback
            prompt = "Explain the concept of streaming inference in large language models"
    
            print()))))))f"Generating response for: '{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}prompt}'\n")
            print()))))))"Response:")
    
            start_time = time.time())))))))
            result = streaming_handler.generate()))))))
            prompt,
            max_tokens=50,
            temperature=0.7,
            callback=token_callback
            )
            generation_time = time.time()))))))) - start_time
    
    # Get performance statistics
            stats = streaming_handler.get_performance_stats())))))))
    
            print()))))))f"Generated {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}stats['tokens_generated']} tokens in {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}generation_time:.2f} seconds"),
            print()))))))f"Speed: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}stats['tokens_generated'] / generation_time:.2f} tokens/second"),
    if hasattr()))))))streaming_handler, "_batch_size_history"):
        print()))))))f"Batch size history: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}streaming_handler._batch_size_history}")
    
            return streaming_handler


def demonstrate_precision_options()))))))):
    """Demonstrate different precision options ()))))))2-bit, 3-bit, 4-bit)."""
    print()))))))"\n\033[1m2. Different Precision Options Comparison\033[0m"),
    print()))))))"-" * 60)
    
    # Model path ()))))))will be simulated)
    model_path = "models/llama-7b"
    
    # Test sample
    prompt = "Demonstrate the difference between 2-bit, 3-bit, and 4-bit precision in LLMs:"
    max_tokens = 30
    
    # Store results for comparison
    results = {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
    
    # Test each precision option
    for bits, precision_name in [()))))))2, "2-bit"), ()))))))3, "3-bit"), ()))))))4, "4-bit")]:,
    print()))))))f"\nTesting {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}precision_name} precision:")
        
        # Create configuration for this precision
    config = {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
    "quantization": f"int{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}bits}",
    "optimize_kv_cache": True,
    "latency_optimized": True,
    "adaptive_batch_size": True,
    "prefill_optimized": True
    }
        
        # Create streaming handler
    streaming_handler = WebGPUStreamingInference()))))))model_path, config)
        
        # Define token accumulator callback
    tokens_collected = []
    ,,
        def collect_tokens()))))))token, is_last=False):
            tokens_collected.append()))))))token)
            # Print with a prefix to identify the precision
            print()))))))f"[{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}precision_name}] {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}token}", end="", flush=True),
            if is_last:
                print()))))))"\n")
        
        # Run streaming generation
                print()))))))f"Generating with {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}precision_name}...")
                start_time = time.time())))))))
                streaming_handler.generate()))))))
                prompt,
                max_tokens=max_tokens,
                temperature=0.7,
                callback=collect_tokens
                )
                generation_time = time.time()))))))) - start_time
        
        # Get performance statistics
                stats = streaming_handler.get_performance_stats())))))))
                tokens_per_second = stats['tokens_generated'] / generation_time if generation_time > 0 else 0
                ,
        # Store results for comparison
                results[precision_name] = {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}:,
                "tokens_generated": stats['tokens_generated'],
                "generation_time": generation_time,
                "tokens_per_second": tokens_per_second,
                "batch_size_history": stats.get()))))))'batch_size_history', []),
                }
        
                print()))))))f"\nGenerated {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}stats['tokens_generated']} tokens in {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}generation_time:.2f}s"),
                print()))))))f"Speed: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}tokens_per_second:.2f} tokens/second")
        
        # If using ultra-low precision, display memory reduction
        if bits <= 3:
            memory_reduction = 87.5 if bits == 2 else 81.25  # Approximate values:
                print()))))))f"Memory reduction: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}memory_reduction:.1f}% vs. FP16")
            
                print()))))))"-" * 40)
    
    # Display comparison summary
                print()))))))"\nPrecision Comparison Summary:")
                print()))))))"-" * 40)
                print()))))))f"{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}'Precision':<10} {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}'Speed ()))))))tokens/s)':<20} {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}'Memory Reduction':<20}")
                print()))))))"-" * 40)
    
    for precision, data in results.items()))))))):
        if precision == "2-bit":
            memory_reduction = "87.5%"
        elif precision == "3-bit":
            memory_reduction = "81.25%"
        else:  # 4-bit
            memory_reduction = "75%"
            
            print()))))))f"{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}precision:<10} {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}data['tokens_per_second']:<20.2f} {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}memory_reduction:<20}")

            ,
def demonstrate_websocket_server()))))))):
    """Demonstrate the WebSocket server for streaming inference."""
    print()))))))"\n\033[1m3. WebSocket Server for Streaming Inference\033[0m"),
    print()))))))"-" * 60)
    
    # Model path ()))))))will be simulated)
    model_path = "models/llama-7b"
    
    # Create HTTP server for the client demo
    web_server = WebServerThread()))))))port=8000)
    
    # Create WebSocket server for streaming
    websocket_server = WebSocketServerManager()))))))model_path, port=8765)
    
    try:
        # Start servers
        web_server.start())))))))
        websocket_server.start())))))))
        
        print()))))))"Servers started:")
        print()))))))f"- HTML demo page: http://localhost:8000/streaming_demo.html")
        print()))))))f"- WebSocket server: ws://localhost:8765")
        print()))))))"\nOpen the demo page in your browser to try streaming inference")
        print()))))))"The demo supports 2-bit, 3-bit, and 4-bit precision options")
        print()))))))"\nPress Ctrl+C to stop the servers...\n")
        
        # Keep running until interrupted
        while True:
            time.sleep()))))))1)
            
    except KeyboardInterrupt:
        print()))))))"\nStopping servers...")
    finally:
        # Clean up
        web_server.stop())))))))
        websocket_server.stop())))))))
        print()))))))"Servers stopped")


def demonstrate_unified_framework_integration()))))))):
    """Demonstrate integration with the unified web framework."""
    print()))))))"\n\033[1m4. Integration with Unified Web Framework\033[0m"),
    print()))))))"-" * 60)
    
    try:
        # Create a WebPlatformAccelerator with the appropriate model type
        accelerator = WebPlatformAccelerator()))))))
        model_path="models/llama-7b",
        model_type="text",
        config={}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
        "streaming_inference": True,
        "quantization": 4,
        "kv_cache_optimization": True,
        "latency_optimized": True
        },
        auto_detect=True
        )
        
        # Create an inference endpoint
        endpoint = accelerator.create_endpoint())))))))
        
        # Define a callback function
        tokens_collected = []
        ,,
        def print_token()))))))token, is_last=False):
            tokens_collected.append()))))))token)
            print()))))))token, end="", flush=True)
            if is_last:
                print()))))))"\n\nGeneration complete!")
        
        # Run streaming inference
                prompt = "Demonstrate how the unified web framework integrates with streaming inference:"
        
                print()))))))f"Generating response for: '{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}prompt}'\n")
                print()))))))"Response:")
        
                result = endpoint()))))))
                {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}"text": prompt},
                max_tokens=50,
                temperature=0.7,
                callback=print_token
                )
        
        # Display the framework's feature usage
                features = accelerator.get_feature_usage())))))))
                print()))))))"\nUnified Framework Feature Usage:")
        for feature, used in features.items()))))))):
            print()))))))f"- {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}feature}: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}'Enabled' if used else 'Disabled'}")
        
        # Display performance metrics
        metrics = accelerator.get_performance_metrics()))))))):
            print()))))))"\nPerformance Metrics:")
            print()))))))f"- Initialization time: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}metrics.get()))))))'initialization_time_ms', 0):.2f} ms")
            print()))))))f"- First inference time: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}metrics.get()))))))'first_inference_time_ms', 0):.2f} ms")
            print()))))))f"- Average inference time: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}metrics.get()))))))'average_inference_time_ms', 0):.2f} ms")
            print()))))))f"- Memory usage: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}metrics.get()))))))'memory_usage_mb', 0):.2f} MB")
        
    except Exception as e:
        print()))))))f"Error demonstrating unified framework integration: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}e}")


def demonstrate_optimized_kv_cache()))))))):
    """Demonstrate the optimized KV cache for streaming inference."""
    print()))))))"\n\033[1m5. Ultra-Low Precision KV Cache Optimization\033[0m"),
    print()))))))"-" * 60)
    
    try:
        # Create optimized KV cache for different precision formats
        precisions = [2, 3, 4]
        ,
        for bits in precisions:
            # Create optimized KV cache
            kv_cache = create_optimized_kv_cache()))))))
            batch_size=1,
            num_heads=32,
            head_dim=128,
            max_seq_len=8192,  # Support very long context
            bits=bits,
            group_size=64
            )
            
            # Calculate memory statistics
            memory_reduction = kv_cache.get()))))))"memory_reduction_percent", 0)
            original_size_mb = kv_cache.get()))))))"original_size_bytes", 0) / ()))))))1024 * 1024)
            quantized_size_mb = kv_cache.get()))))))"quantized_size_bytes", 0) / ()))))))1024 * 1024)
            
            # Display KV cache information
            print()))))))f"\n{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}bits}-bit Precision KV Cache:")
            print()))))))f"- Memory reduction: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}memory_reduction:.1f}%")
            print()))))))f"- Original size: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}original_size_mb:.2f} MB")
            print()))))))f"- Quantized size: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}quantized_size_mb:.2f} MB")
            print()))))))f"- Maximum sequence length: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}kv_cache.get()))))))'max_seq_len', 0)}")
            
    except Exception as e:
        print()))))))f"Error demonstrating KV cache optimization: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}e}")


def demonstrate_all()))))))):
    """Run all demonstrations."""
    print()))))))"\n\033[1;32m=== WebGPU Streaming Inference Tutorial ===\033[0m"),
    print()))))))"This tutorial demonstrates how to use the WebGPU streaming inference")
    print()))))))"pipeline for token-by-token generation with various precision options.")
    print()))))))"=" * 60)
    
    # Run each demonstration
    demonstrate_token_callback())))))))
    demonstrate_precision_options())))))))
    
    # Check if we should run the interactive demos:
    run_servers = input()))))))"\nDo you want to start the interactive WebSocket demo? ()))))))y/n): ").lower()))))))) == 'y'
    if run_servers:
        demonstrate_websocket_server())))))))
    else:
        print()))))))"Skipping interactive WebSocket demo")
    
        demonstrate_unified_framework_integration())))))))
        demonstrate_optimized_kv_cache())))))))
    
        print()))))))"\n\033[1;32m=== Tutorial Complete ===\033[0m")

        ,
if __name__ == "__main__":
    demonstrate_all())))))))