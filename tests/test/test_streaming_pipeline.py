#!/usr/bin/env python3
"""
Test demonstration for WebGPU streaming pipeline.

This script shows how to use the WebGPU streaming pipeline to:
    1. Set up and configure a streaming pipeline with a sample model
    2. Create a simple client that connects to the server and sends a streaming request
    3. Receive and display the streaming response
    4. Handle different message types ()))))))tokens, metrics, completion)
    5. Cancel requests and get status updates
    6. Clean up resources properly when done

Usage:
    python test_streaming_pipeline.py
    python test_streaming_pipeline.py --model models/llama-7b 
    python test_streaming_pipeline.py --quantization int4
    """

    import os
    import sys
    import json
    import time
    import asyncio
    import argparse
    import logging
    from pathlib import Path
    from typing import Dict, List, Any, Optional, Union, Callable, Tuple
    from unittest.mock import MagicMock

# Configure logging
    logging.basicConfig()))))))level=logging.INFO, format='%()))))))asctime)s - %()))))))levelname)s - %()))))))message)s')
    logger = logging.getLogger()))))))"streaming_test")

# Add the parent directory to the path for importing
    script_dir = Path()))))))os.path.dirname()))))))os.path.abspath()))))))__file__)))
    parent_dir = script_dir.parent
    sys.path.insert()))))))0, str()))))))parent_dir))

# Enable WebGPU simulation mode if not in a browser environment
    os.environ["WEBGPU_SIMULATION"] = "1"
    ,
# Import the streaming pipeline module:
try:
    from fixed_web_platform.webgpu_streaming_pipeline import ()))))))
    WebGPUStreamingPipeline,
    create_streaming_pipeline,
    start_streaming_server,
    StreamingRequest
    )
    from fixed_web_platform.webgpu_streaming_inference import ()))))))
    WebGPUStreamingInference,
    optimize_for_streaming
    )
except ImportError:
    logger.error()))))))"Failed to import WebGPU streaming pipeline. Make sure the fixed_web_platform directory is available.")
    sys.exit()))))))1)

class SimulatedWebSocketClient:
    """A simple simulated WebSocket client for testing."""
    
    def __init__()))))))self):
        """Initialize the simulated client."""
        self.messages = [],
        self.closed = False
    
    async def send()))))))self, data):
        """Simulate sending data over the WebSocket."""
        message = json.loads()))))))data) if isinstance()))))))data, str) else data:
            logger.debug()))))))f"Client sent: {}}}}}}}message}")
        return True
    
    async def recv()))))))self):
        """Simulate receiving data from the server."""
        # In a real implementation, this would wait for server messages
        # For simulation, we'll just return a ping message
        await asyncio.sleep()))))))0.1)
        return json.dumps())))))){}}}}}}}"type": "ping"})
    
    def close()))))))self):
        """Close the simulated connection."""
        self.closed = True

class StreamingPipelineDemo:
    """Demonstration of the WebGPU streaming pipeline."""
    
    def __init__()))))))self, model_path: str = "models/llama-7b", quantization: str = "int4"):
        """Initialize the streaming demo."""
        self.model_path = model_path
        self.quantization = quantization
        self.pipeline = None
        self.server_thread = None
        
    def start_server()))))))self):
        """Start the streaming server."""
        logger.info()))))))f"Setting up streaming server with model: {}}}}}}}self.model_path}")
        logger.info()))))))f"Using quantization: {}}}}}}}self.quantization}")
        
        # Create configuration
        config = {}}}}}}}
        "quantization": self.quantization,
        "memory_limit_mb": 4096,
        "max_clients": 5,
        "auto_tune": True,
        "debug_mode": False
        }
        
        # Create and start the pipeline
        self.pipeline = create_streaming_pipeline()))))))self.model_path, config)
        self.server_thread = self.pipeline.start_server()))))))host="localhost", port=8765)
        
        logger.info()))))))"Streaming server started at ws://localhost:8765")
        time.sleep()))))))1)  # Give the server time to start
    
    def stop_server()))))))self):
        """Stop the streaming server."""
        if self.pipeline:
            self.pipeline.stop_server())))))))
            logger.info()))))))"Streaming server stopped")
    
    async def demonstrate_callback_streaming()))))))self):
        """Demonstrate token-by-token generation with callbacks."""
        print()))))))"\n=== Example 1: Token-by-Token Generation with Callback ===")
        
        # Configure for streaming
        config = optimize_for_streaming())))))){}}}}}}}
        "quantization": self.quantization,
        "latency_optimized": True,
        "adaptive_batch_size": True
        })
        
        # Create streaming handler
        streaming_handler = WebGPUStreamingInference()))))))self.model_path, config)
        
        # Define callback to handle tokens
        tokens_received = [],
        
        def token_callback()))))))token, is_last=False):
            tokens_received.append()))))))token)
            print()))))))token, end="", flush=True)
            if is_last:
                print()))))))"\n\nGeneration complete!")
        
        # Generate with streaming
                prompt = "Explain how WebGPU streaming inference works for large language models"
        
                print()))))))f"\nPrompt: {}}}}}}}prompt}")
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
        
                print()))))))f"\nGenerated {}}}}}}}stats.get()))))))'tokens_generated', 0)} tokens in {}}}}}}}generation_time:.2f} seconds")
                print()))))))f"Speed: {}}}}}}}stats.get()))))))'tokens_per_second', 0):.2f} tokens/second")
        
        if hasattr()))))))streaming_handler, "_batch_size_history"):
            print()))))))f"Batch size history: {}}}}}}}streaming_handler._batch_size_history}")
        
                return result
    
    async def demonstrate_websocket_streaming()))))))self):
        """Demonstrate WebSocket-based streaming."""
        print()))))))"\n=== Example 2: WebSocket-Based Streaming ===")
        
        # Create a simulated WebSocket
        mock_websocket = MagicMock())))))))
        sent_messages = [],
        
        async def mock_send()))))))message):
            data = json.loads()))))))message) if isinstance()))))))message, str) else message
            sent_messages.append()))))))data)
            
            # Print different message types:
            if data.get()))))))"type") == "token":
                print()))))))f"{}}}}}}}data.get()))))))'token', '')}", end="", flush=True)
            elif data.get()))))))"type") == "start":
                print()))))))"\nStarting generation...")
                if data.get()))))))"using_ultra_low_precision"):
                    print()))))))f"Using {}}}}}}}data.get()))))))'precision_bits')}-bit precision "
                    f"())))))){}}}}}}}data.get()))))))'memory_reduction_percent', 0):.1f}% memory reduction)")
            elif data.get()))))))"type") == "prefill_complete":
                print()))))))f"\n[Prefill complete: {}}}}}}}data.get()))))))'tokens_processed')} tokens in ",
                f"{}}}}}}}data.get()))))))'time_ms', 0)/1000:.2f}s]")
            elif data.get()))))))"type") == "complete":
                print()))))))"\n\nGeneration complete!")
                print()))))))f"Generated {}}}}}}}data.get()))))))'tokens_generated', 0)} tokens in "
                f"{}}}}}}}data.get()))))))'generation_time', 0):.2f}s")
                print()))))))f"Speed: {}}}}}}}data.get()))))))'tokens_per_second', 0):.2f} tokens/sec")
                
                if "precision_bits" in data:
                    print()))))))f"Using {}}}}}}}data.get()))))))'precision_bits')}-bit precision with "
                    f"{}}}}}}}data.get()))))))'memory_reduction_percent', 0):.1f}% memory reduction")
            elif data.get()))))))"type") == "error":
                print()))))))f"\nError: {}}}}}}}data.get()))))))'message', 'Unknown error')}")
        
                mock_websocket.send = mock_send
                mock_websocket.closed = False
        
        # Configure for streaming
                config = optimize_for_streaming())))))){}}}}}}}
                "quantization": self.quantization,
                "latency_optimized": True,
                "adaptive_batch_size": True,
                "stream_buffer_size": 2  # Small buffer for responsive streaming
                })
        
        # Create streaming handler
                streaming_handler = WebGPUStreamingInference()))))))self.model_path, config)
        
        # Stream tokens over WebSocket
                prompt = "Demonstrate how WebSocket-based streaming works with the WebGPU pipeline"
        
                print()))))))f"\nPrompt: {}}}}}}}prompt}")
                print()))))))"Response:")
        
                start_time = time.time())))))))
                await streaming_handler.stream_websocket()))))))
                mock_websocket,
                prompt,
                max_tokens=50,
                temperature=0.7,
                stream_options={}}}}}}}
                "send_stats_frequency": 10,  # Send stats every 10 tokens
                "memory_metrics": True,
                "latency_metrics": True,
                "batch_metrics": True
                }
                )
                generation_time = time.time()))))))) - start_time
        
        # Analyze the messages
                token_messages = [msg for msg in sent_messages if msg.get()))))))"type") == "token"],
                kv_cache_messages = [msg for msg in sent_messages if msg.get()))))))"type") == "kv_cache_status"],
                complete_messages = [msg for msg in sent_messages if msg.get()))))))"type") == "complete"],
        :
            print()))))))f"\nWebSocket streaming details:")
            print()))))))f"- Total messages: {}}}}}}}len()))))))sent_messages)}")
            print()))))))f"- Token messages: {}}}}}}}len()))))))token_messages)}")
            print()))))))f"- KV cache updates: {}}}}}}}len()))))))kv_cache_messages)}")
            print()))))))f"- Completion messages: {}}}}}}}len()))))))complete_messages)}")
        
        # Get performance statistics
            stats = streaming_handler.get_performance_stats())))))))
        
                return {}}}}}}}
                "tokens_generated": stats.get()))))))"tokens_generated", 0),
                "generation_time": generation_time,
                "messages_sent": len()))))))sent_messages)
                }
    
    async def demonstrate_server_client()))))))self):
        """Demonstrate server-client communication with the streaming pipeline."""
        print()))))))"\n=== Example 3: Full Server-Client Communication ===")
        
        # Start the server
        self.start_server())))))))
        
        try:
            # In a real implementation, we would connect to the server with a WebSocket client
            # For this demonstration, we'll simulate the interaction using our existing components
            
            # Create a simulated request
            request = StreamingRequest()))))))
            id="test-request-123",
            prompt="This is a test of the full server-client streaming pipeline",
            max_tokens=30,
            temperature=0.7
            )
            
            # Create a simulated WebSocket
            mock_client = MagicMock())))))))
            sent_messages = [],
            
            async def mock_send()))))))message):
                data = json.loads()))))))message) if isinstance()))))))message, str) else message
                sent_messages.append()))))))data)
                
                # Print token messages for demonstration:
                if data.get()))))))"type") == "token":
                    print()))))))f"{}}}}}}}data.get()))))))'token', '')}", end="", flush=True)
                elif data.get()))))))"type") == "complete":
                    print()))))))"\n\nGeneration complete!")
            
                    mock_client.send = mock_send
                    mock_client.closed = False
            
            # Attach the mock client to the request
                    request.client = mock_client
            
            # Enqueue the request with the pipeline
                    print()))))))f"\nSending request to pipeline server: {}}}}}}}request.prompt}")
                    print()))))))"Response:")
            
                    success = self.pipeline.enqueue_request()))))))request)
            
            if success:
                # Wait for processing to complete ()))))))simulated)
                await asyncio.sleep()))))))10)  # Wait for generation to complete
                
                print()))))))f"\nRequest completed with {}}}}}}}len()))))))sent_messages)} messages sent")
            else:
                print()))))))"\nFailed to enqueue request - queue may be full")
            
        finally:
            # Stop the server
            self.stop_server())))))))
    
    async def demonstrate_request_cancellation()))))))self):
        """Demonstrate how to cancel a streaming request."""
        print()))))))"\n=== Example 4: Request Cancellation ===")
        
        # Create a mock WebSocket
        mock_websocket = MagicMock())))))))
        sent_messages = [],
        
        async def mock_send()))))))message):
            data = json.loads()))))))message) if isinstance()))))))message, str) else message
            sent_messages.append()))))))data)
            
            # Print token messages:
            if data.get()))))))"type") == "token":
                print()))))))f"{}}}}}}}data.get()))))))'token', '')}", end="", flush=True)
            elif data.get()))))))"type") == "cancelled":
                print()))))))"\n\nRequest cancelled!")
        
                mock_websocket.send = mock_send
                mock_websocket.closed = False
        
        # Set up to simulate receiving a cancel request after a few tokens
                token_count = 0
                cancel_sent = False
        
        async def mock_recv()))))))):
            nonlocal token_count, cancel_sent
            token_messages = [m for m in sent_messages if m.get()))))))"type") == "token"]
            ,,
            # After receiving a few tokens, send a cancellation:
            if len()))))))token_messages) > 5 and not cancel_sent:
                cancel_sent = True
                print()))))))"\n[Sending cancellation request]"),
            return json.dumps())))))){}}}}}}}"type": "cancel", "request_id": "test-cancel-123"})
            
            # Otherwise, just wait
            await asyncio.sleep()))))))0.1)
                return json.dumps())))))){}}}}}}}"type": "ping"})
            
                mock_websocket.recv = mock_recv
        
        # Configure for streaming
                config = optimize_for_streaming())))))){}}}}}}}
                "quantization": self.quantization,
                "latency_optimized": True,
                "adaptive_batch_size": True
                })
        
        # Create streaming handler
                streaming_handler = WebGPUStreamingInference()))))))self.model_path, config)
        
        # Use a longer prompt to ensure we have time to cancel
                prompt = """This is a test of the cancellation functionality. The request will be cancelled
                after a few tokens are generated. In a real-world scenario, this allows users to stop
                generation that is going in an unwanted direction or taking too long."""
        
                print()))))))f"\nPrompt: {}}}}}}}prompt}")
                print()))))))"Response ()))))))will be cancelled after a few tokens):")
        
        try:
            await streaming_handler.stream_websocket()))))))
            mock_websocket,
            prompt,
            max_tokens=100,  # Large token count to ensure we have time to cancel
            temperature=0.7
            )
        except Exception as e:
            logger.debug()))))))f"Exception during cancellation test: {}}}}}}}e}")
        
        # Note about real implementation
            print()))))))"\nNote: In this simulation, cancellation may not be fully demonstrated")
            print()))))))"In a real implementation, the cancellation would be properly handled by the server")
    
    async def demonstrate_status_updates()))))))self):
        """Demonstrate how to get status updates for a request."""
        print()))))))"\n=== Example 5: Status Updates ===")
        
        # Create a mock WebSocket
        mock_websocket = MagicMock())))))))
        sent_messages = [],
        
        async def mock_send()))))))message):
            data = json.loads()))))))message) if isinstance()))))))message, str) else message
            sent_messages.append()))))))data)
            
            # Print status updates:
            if data.get()))))))"type") == "status":
                print()))))))f"\nStatus update:")
                print()))))))f"  Queue position: {}}}}}}}data.get()))))))'queue_position', 'unknown')}")
                print()))))))f"  Estimated wait time: {}}}}}}}data.get()))))))'estimated_wait_time', 'unknown')}s")
                print()))))))f"  Queue length: {}}}}}}}data.get()))))))'queue_length', 'unknown')}")
                print()))))))f"  Active clients: {}}}}}}}data.get()))))))'active_clients', 'unknown')}")
            elif data.get()))))))"type") == "token":
                print()))))))f"{}}}}}}}data.get()))))))'token', '')}", end="", flush=True)
        
                mock_websocket.send = mock_send
                mock_websocket.closed = False
        
        # Set up to simulate sending a status request
                status_requested = False
        
        async def mock_recv()))))))):
            nonlocal status_requested
            token_messages = [m for m in sent_messages if m.get()))))))"type") == "token"]
            ,,
            # After a few tokens, send a status request:
            if len()))))))token_messages) > 3 and not status_requested:
                status_requested = True
                print()))))))"\n[Sending status request]"),
            return json.dumps())))))){}}}}}}}"type": "status", "request_id": "test-status-123"})
            
            # Otherwise, just wait
            await asyncio.sleep()))))))0.1)
                return json.dumps())))))){}}}}}}}"type": "ping"})
            
                mock_websocket.recv = mock_recv
        
        # Configure for streaming
                config = optimize_for_streaming())))))){}}}}}}}
                "quantization": self.quantization,
                "latency_optimized": True,
                "adaptive_batch_size": True
                })
        
        # Create streaming handler
                streaming_handler = WebGPUStreamingInference()))))))self.model_path, config)
        
        # Use a simple prompt
                prompt = "This is a test of the status update functionality."
        
                print()))))))f"\nPrompt: {}}}}}}}prompt}")
                print()))))))"Response ()))))))will request status after a few tokens):")
        
        try:
            await streaming_handler.stream_websocket()))))))
            mock_websocket,
            prompt,
            max_tokens=20,
            temperature=0.7
            )
        except Exception as e:
            logger.debug()))))))f"Exception during status update test: {}}}}}}}e}")
        
        # Note about real implementation
            print()))))))"\nNote: In this simulation, status updates may not be fully demonstrated")
            print()))))))"In a real implementation, the server would provide queue and client information")
    
    async def run_demo()))))))self):
        """Run the complete demonstration."""
        print()))))))"=== WebGPU Streaming Pipeline Demonstration ===")
        print()))))))f"Model: {}}}}}}}self.model_path}")
        print()))))))f"Quantization: {}}}}}}}self.quantization}")
        
        try:
            # Example 1: Token-by-token generation with callbacks
            await self.demonstrate_callback_streaming())))))))
            
            # Example 2: WebSocket-based streaming
            await self.demonstrate_websocket_streaming())))))))
            
            # Example 3: Full server-client communication
            await self.demonstrate_server_client())))))))
            
            # Example 4: Request cancellation
            await self.demonstrate_request_cancellation())))))))
            
            # Example 5: Status updates
            await self.demonstrate_status_updates())))))))
            
            print()))))))"\n=== Demonstration Complete ===")
            print()))))))"The WebGPU streaming pipeline provides an end-to-end solution for:")
            print()))))))"- Token-by-token generation with ultra-low precision")
            print()))))))"- WebSocket-based streaming for real-time updates")
            print()))))))"- Comprehensive metrics and status tracking")
            print()))))))"- Request management with cancellation capabilities")
            print()))))))"- Memory-efficient inference with adaptive optimizations")
            
        except Exception as e:
            logger.error()))))))f"Error in demonstration: {}}}}}}}e}")
            import traceback
            traceback.print_exc())))))))

def main()))))))):
    """Parse arguments and run the demonstration."""
    parser = argparse.ArgumentParser()))))))description="Test WebGPU Streaming Pipeline")
    parser.add_argument()))))))"--model", default="models/llama-7b", help="Path to the model")
    parser.add_argument()))))))"--quantization", default="int4", 
    choices=["int2", "int3", "int4", "int8", "fp16"],
    help="Quantization format to use")
    args = parser.parse_args())))))))
    
    # Create and run the demonstration
    demo = StreamingPipelineDemo()))))))args.model, args.quantization)
    asyncio.run()))))))demo.run_demo()))))))))

if __name__ == "__main__":
    main())))))))