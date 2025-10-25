#!/usr/bin/env python3
"""
Test script for Queue Monitor functionality
"""

import json
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_queue_monitoring_tools():
    """Test the queue monitoring MCP tools"""
    print("🧪 Testing Queue Monitor MCP Tools")
    print("=" * 50)
    
    try:
        # Import the enhanced inference tools
        from ipfs_accelerate_py.mcp.tools.enhanced_inference import QUEUE_MONITOR
        
        # Create a mock MCP object for testing
        class MockMCP:
            def __init__(self):
                self.tools = {}
                
            def tool(self):
                def decorator(func):
                    self.tools[func.__name__] = func
                    return func
                return decorator
        
        # Register the tools
        from ipfs_accelerate_py.mcp.tools.enhanced_inference import register_tools
        mock_mcp = MockMCP()
        register_tools(mock_mcp)
        
        print(f"✅ Registered {len(mock_mcp.tools)} MCP tools")
        print("📋 Available queue monitoring tools:")
        for tool_name in mock_mcp.tools.keys():
            if 'queue' in tool_name:
                print(f"  - {tool_name}")
        
        # Test get_queue_status
        if 'get_queue_status' in mock_mcp.tools:
            print("\n🔍 Testing get_queue_status tool...")
            queue_status = mock_mcp.tools['get_queue_status']()
            
            print("📊 Queue Status Summary:")
            if queue_status.get('status') == 'success':
                summary = queue_status.get('summary', {})
                print(f"  Total Endpoints: {summary.get('total_endpoints', 0)}")
                print(f"  Active Endpoints: {summary.get('active_endpoints', 0)}")
                print(f"  Total Queue Size: {summary.get('total_queue_size', 0)}")
                print(f"  Processing Tasks: {summary.get('total_processing', 0)}")
                
                print("\n🖥️  Endpoint Types:")
                endpoint_queues = queue_status.get('endpoint_queues', {})
                endpoint_types = {}
                for endpoint_id, endpoint in endpoint_queues.items():
                    ep_type = endpoint.get('endpoint_type', 'unknown')
                    if ep_type not in endpoint_types:
                        endpoint_types[ep_type] = []
                    endpoint_types[ep_type].append(endpoint_id)
                
                for ep_type, endpoints in endpoint_types.items():
                    print(f"  {ep_type}: {len(endpoints)} endpoints")
                    for endpoint_id in endpoints[:2]:  # Show first 2 as examples
                        endpoint = endpoint_queues[endpoint_id]
                        status = endpoint.get('status', 'unknown')
                        queue_size = endpoint.get('queue_size', 0)
                        processing = endpoint.get('processing', 0)
                        print(f"    - {endpoint_id}: {status}, queue={queue_size}, processing={processing}")
                
            else:
                print(f"❌ Queue status failed: {queue_status.get('error', 'Unknown error')}")
        
        # Test get_queue_history
        if 'get_queue_history' in mock_mcp.tools:
            print("\n📈 Testing get_queue_history tool...")
            queue_history = mock_mcp.tools['get_queue_history']()
            
            if queue_history.get('status') == 'success':
                model_stats = queue_history.get('model_type_stats', {})
                print("📊 Model Type Statistics:")
                for model_type, stats in model_stats.items():
                    print(f"  {model_type}: {stats['total_requests']} requests, "
                          f"{stats['success_rate']:.1f}% success rate, "
                          f"{stats['avg_time']:.1f}s avg time")
            else:
                print(f"❌ Queue history failed: {queue_history.get('error', 'Unknown error')}")
        
        print("\n✅ Queue monitoring tools test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_queue_monitoring_tools()
    sys.exit(0 if success else 1)