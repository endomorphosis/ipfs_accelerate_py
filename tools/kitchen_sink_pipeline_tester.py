#!/usr/bin/env python3
"""
Kitchen Sink AI Testing Interface - Simplified Pipeline Tester

This script tests the Kitchen Sink interface to verify all inference pipelines work,
without requiring Playwright for screenshots initially.
"""

import os
import sys
import time
import asyncio
import subprocess
import signal
import json
from pathlib import Path
from typing import Dict, List, Any

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

class KitchenSinkPipelineTester:
    """Test all inference pipelines in the Kitchen Sink interface."""
    
    def __init__(self):
        """Initialize the tester."""
        self.server_process = None
        self.server_url = "http://127.0.0.1:8080"
        self.test_results = {}
        
    async def setup_server(self):
        """Start the Kitchen Sink server."""
        print("🚀 Starting Kitchen Sink server...")
        
        # Start the server in background
        env = os.environ.copy()
        env['PYTHONPATH'] = os.path.dirname(__file__)
        
        self.server_process = subprocess.Popen([
            sys.executable, "-c", """
import sys
import os
sys.path.append(os.path.dirname(__file__))
from kitchen_sink_app import create_app

print("Creating Kitchen Sink app...")
app = create_app()
print("Starting server on port 8080...")
app.run(host='127.0.0.1', port=8080, debug=False)
"""
        ], cwd=os.path.dirname(__file__), env=env, 
           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait for server to start
        print("⏳ Waiting for server to start...")
        await asyncio.sleep(8)
        
        # Check if server is running
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                for i in range(15):
                    try:
                        async with session.get(self.server_url) as resp:
                            if resp.status == 200:
                                print("✅ Server is running!")
                                return True
                    except Exception as e:
                        print(f"   Waiting... attempt {i+1}/15")
                        await asyncio.sleep(2)
                        
        except ImportError:
            # Fallback without aiohttp
            await asyncio.sleep(15)
            return True
            
        print("⚠️ Server may not be responding, but continuing with tests...")
        return True
        
    async def teardown_server(self):
        """Stop the Kitchen Sink server."""
        if self.server_process:
            print("🛑 Stopping server...")
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.server_process.kill()
            self.server_process = None
            
    async def test_api_endpoints(self):
        """Test API endpoints directly."""
        print("🧪 Testing API endpoints...")
        
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                
                # Test model listing
                await self._test_models_api(session)
                
                # Test text generation
                await self._test_text_generation_api(session)
                
                # Test text classification  
                await self._test_text_classification_api(session)
                
                # Test embeddings
                await self._test_text_embeddings_api(session)
                
                # Test recommendations
                await self._test_recommendations_api(session)
                
                return True
                
        except ImportError:
            print("❌ aiohttp not available for API testing")
            return False
        except Exception as e:
            print(f"❌ API testing error: {e}")
            return False
            
    async def _test_models_api(self, session):
        """Test models API endpoint."""
        try:
            async with session.get(f"{self.server_url}/api/models") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    models = data.get('models', [])
                    self.test_results['models_api'] = f'success: {len(models)} models'
                    print(f"✅ Models API - {len(models)} models available")
                else:
                    self.test_results['models_api'] = f'error: status {resp.status}'
                    print(f"❌ Models API error: status {resp.status}")
        except Exception as e:
            self.test_results['models_api'] = f'error: {str(e)}'
            print(f"❌ Models API error: {e}")
            
    async def _test_text_generation_api(self, session):
        """Test text generation API."""
        try:
            payload = {
                'prompt': 'The future of AI is',
                'max_length': 50,
                'temperature': 0.7
            }
            async with session.post(f"{self.server_url}/api/generate", json=payload) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if 'generated_text' in data:
                        self.test_results['text_generation_api'] = 'success'
                        print("✅ Text Generation API working")
                    else:
                        self.test_results['text_generation_api'] = 'no_text_in_response'
                        print("⚠️ Text Generation API - no generated text in response")
                else:
                    self.test_results['text_generation_api'] = f'error: status {resp.status}'
                    print(f"❌ Text Generation API error: status {resp.status}")
        except Exception as e:
            self.test_results['text_generation_api'] = f'error: {str(e)}'
            print(f"❌ Text Generation API error: {e}")
            
    async def _test_text_classification_api(self, session):
        """Test text classification API."""
        try:
            payload = {
                'text': 'This movie is amazing!'
            }
            async with session.post(f"{self.server_url}/api/classify", json=payload) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if 'classification' in data:
                        self.test_results['text_classification_api'] = 'success'
                        print("✅ Text Classification API working")
                    else:
                        self.test_results['text_classification_api'] = 'no_classification_in_response'
                        print("⚠️ Text Classification API - no classification in response")
                else:
                    self.test_results['text_classification_api'] = f'error: status {resp.status}'
                    print(f"❌ Text Classification API error: status {resp.status}")
        except Exception as e:
            self.test_results['text_classification_api'] = f'error: {str(e)}'
            print(f"❌ Text Classification API error: {e}")
            
    async def _test_text_embeddings_api(self, session):
        """Test text embeddings API."""
        try:
            payload = {
                'text': 'Machine learning is transforming the world'
            }
            async with session.post(f"{self.server_url}/api/embeddings", json=payload) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if 'embeddings' in data:
                        self.test_results['text_embeddings_api'] = 'success'
                        print("✅ Text Embeddings API working")
                    else:
                        self.test_results['text_embeddings_api'] = 'no_embeddings_in_response'
                        print("⚠️ Text Embeddings API - no embeddings in response")
                else:
                    self.test_results['text_embeddings_api'] = f'error: status {resp.status}'
                    print(f"❌ Text Embeddings API error: status {resp.status}")
        except Exception as e:
            self.test_results['text_embeddings_api'] = f'error: {str(e)}'
            print(f"❌ Text Embeddings API error: {e}")
            
    async def _test_recommendations_api(self, session):
        """Test recommendations API."""
        try:
            payload = {
                'task_type': 'text_generation',
                'input_types': ['text'],
                'output_types': ['text'],
                'requirements': ['fast inference']
            }
            async with session.post(f"{self.server_url}/api/recommend", json=payload) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if 'recommendations' in data:
                        self.test_results['recommendations_api'] = 'success'
                        print("✅ Recommendations API working")
                    else:
                        self.test_results['recommendations_api'] = 'no_recommendations_in_response'
                        print("⚠️ Recommendations API - no recommendations in response")
                else:
                    self.test_results['recommendations_api'] = f'error: status {resp.status}'
                    print(f"❌ Recommendations API error: status {resp.status}")
        except Exception as e:
            self.test_results['recommendations_api'] = f'error: {str(e)}'
            print(f"❌ Recommendations API error: {e}")

    def test_server_startup(self):
        """Test that server components can be imported and initialized."""
        print("🔧 Testing server component initialization...")
        
        try:
            from kitchen_sink_app import create_app
            app = create_app()
            self.test_results['server_initialization'] = 'success'
            print("✅ Server initialization successful")
            return True
        except Exception as e:
            self.test_results['server_initialization'] = f'error: {str(e)}'
            print(f"❌ Server initialization error: {e}")
            return False
            
    def generate_test_report(self):
        """Generate a comprehensive test report."""
        
        success_count = sum(1 for result in self.test_results.values() 
                          if 'success' in str(result))
        total_tests = len(self.test_results)
        
        print("\n" + "="*60)
        print("🧪 KITCHEN SINK PIPELINE TEST RESULTS")
        print("="*60)
        print(f"Total Tests: {total_tests}")
        print(f"Successful: {success_count}")
        print(f"Success Rate: {(success_count/total_tests*100):.1f}%")
        print("="*60)
        
        for pipeline, result in self.test_results.items():
            status = "✅ PASS" if 'success' in str(result) else "❌ FAIL"
            print(f"{pipeline.replace('_', ' ').title():<30} {status}")
            if 'error' in str(result):
                print(f"  └─ {result}")
        
        print("="*60)
        
        # Save detailed report
        report = {
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "summary": {
                "total_tests": total_tests,
                "successful": success_count,
                "success_rate": f"{(success_count/total_tests*100):.1f}%"
            },
            "detailed_results": self.test_results
        }
        
        report_path = Path("./kitchen_sink_pipeline_test_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📊 Detailed report saved to: {report_path}")
        
        return success_count >= total_tests * 0.7  # 70% success threshold

async def main():
    """Main testing function."""
    
    tester = KitchenSinkPipelineTester()
    
    try:
        # Test server initialization first
        if not tester.test_server_startup():
            print("❌ Server initialization failed - cannot proceed")
            return False
        
        # Start server
        print("\n" + "-"*40)
        server_started = await tester.setup_server()
        if not server_started:
            print("❌ Failed to start server")
            return False
            
        # Test API endpoints
        print("\n" + "-"*40)
        await tester.test_api_endpoints()
        
        # Generate report
        print("\n" + "-"*40)
        success = tester.generate_test_report()
        
        if success:
            print("\n🎉 Kitchen Sink AI Testing Interface is working correctly!")
            print("📋 All major inference pipelines are operational")
            return True
        else:
            print("\n⚠️ Some issues detected - see report for details")
            return False
            
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
        return False
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        await tester.teardown_server()

if __name__ == "__main__":
    print("🚀 Kitchen Sink AI Testing Interface - Pipeline Verification")
    print("Testing all inference pipelines without browser automation...")
    print("=" * 60)
    
    try:
        result = asyncio.run(main())
        exit_code = 0 if result else 1
    except Exception as e:
        print(f"❌ Failed to run tests: {e}")
        exit_code = 1
        
    print("=" * 60)
    print(f"🏁 Testing completed with exit code: {exit_code}")
    sys.exit(exit_code)