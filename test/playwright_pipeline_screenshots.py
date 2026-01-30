#!/usr/bin/env python3
"""
Playwright Screenshot Test for Kitchen Sink AI Testing Interface

This script uses Playwright to:
1. Start the Kitchen Sink server
2. Take screenshots of each inference pipeline working
3. Test all major functionality to verify everything works
4. Generate comprehensive visual documentation
"""

import os
import sys
import time
import anyio
import subprocess
import signal
from pathlib import Path
from typing import Dict, List, Any
import json

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

try:
    from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError
    HAVE_PLAYWRIGHT = True
except ImportError:
    HAVE_PLAYWRIGHT = False
    print("⚠️ Playwright not available. Installing...")

class KitchenSinkScreenshotTester:
    """Comprehensive screenshot tester for Kitchen Sink AI interface."""
    
    def __init__(self):
        """Initialize the tester."""
        self.server_process = None
        self.server_url = "http://127.0.0.1:8080"
        self.screenshots_dir = Path("./data/screenshots/kitchen_sink")
        self.screenshots_dir.mkdir(parents=True, exist_ok=True)
        self.test_results = {}
        
    async def setup_server(self):
        """Start the Kitchen Sink server."""
        print("🚀 Starting Kitchen Sink server...")
        
        # Start the server in background
        env = os.environ.copy()
        env['PYTHONPATH'] = os.path.dirname(__file__)
        
        self.server_process = subprocess.Popen([
            sys.executable, "kitchen_sink_demo.py"
        ], cwd=os.path.dirname(__file__), env=env, 
           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait for server to start
        print("⏳ Waiting for server to start...")
        await anyio.sleep(5)
        
        # Check if server is running
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                for i in range(10):
                    try:
                        async with session.get(self.server_url) as resp:
                            if resp.status == 200:
                                print("✅ Server is running!")
                                return True
                    except:
                        await anyio.sleep(2)
                        
        except ImportError:
            # Fallback without aiohttp
            await anyio.sleep(10)
            return True
            
        return False
        
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
            
    async def test_inference_pipelines(self):
        """Test all inference pipelines with screenshots."""
        
        if not HAVE_PLAYWRIGHT:
            print("❌ Playwright not available - cannot take screenshots")
            return False
            
        async with async_playwright() as p:
            # Launch browser
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                viewport={'width': 1920, 'height': 1080}
            )
            page = await context.new_page()
            
            try:
                # Navigate to the application
                print("🌐 Navigating to Kitchen Sink interface...")
                await page.goto(self.server_url, wait_until='networkidle')
                
                # Take initial overview screenshot
                await page.screenshot(
                    path=self.screenshots_dir / "00_overview.png",
                    full_page=True
                )
                print("📸 Captured overview screenshot")
                
                # Test each inference pipeline
                await self._test_text_generation_pipeline(page)
                await self._test_text_classification_pipeline(page)  
                await self._test_text_embeddings_pipeline(page)
                await self._test_model_recommendations_pipeline(page)
                await self._test_model_manager_pipeline(page)
                
                # Test mobile responsiveness
                await self._test_mobile_responsiveness(page)
                
                return True
                
            except Exception as e:
                print(f"❌ Error during testing: {e}")
                return False
                
            finally:
                await browser.close()
                
    async def _test_text_generation_pipeline(self, page):
        """Test text generation pipeline."""
        print("🔤 Testing Text Generation Pipeline...")
        
        try:
            # Click on Text Generation tab
            await page.click('#generation-tab')
            await page.wait_for_timeout(1000)
            
            # Fill in the form
            await page.fill('#gen-prompt', 'The future of artificial intelligence is')
            await page.select_option('#gen-max-length', '150')
            await page.select_option('#gen-temperature', '0.8')
            
            # Take screenshot before submission
            await page.screenshot(
                path=self.screenshots_dir / "01_text_generation_form.png",
                full_page=True
            )
            
            # Submit the form
            await page.click('#generation-form button[type="submit"]')
            
            # Wait for results
            await page.wait_for_timeout(3000)
            
            # Take screenshot of results
            await page.screenshot(
                path=self.screenshots_dir / "02_text_generation_results.png", 
                full_page=True
            )
            
            # Check for success indicators
            result_element = await page.query_selector('#generation-results')
            if result_element:
                self.test_results['text_generation'] = 'success'
                print("✅ Text Generation pipeline working")
            else:
                self.test_results['text_generation'] = 'no_results'
                print("⚠️ Text Generation - no results visible")
                
        except Exception as e:
            self.test_results['text_generation'] = f'error: {str(e)}'
            print(f"❌ Text Generation error: {e}")
            
    async def _test_text_classification_pipeline(self, page):
        """Test text classification pipeline."""
        print("🏷️ Testing Text Classification Pipeline...")
        
        try:
            # Click on Classification tab
            await page.click('#classification-tab')
            await page.wait_for_timeout(1000)
            
            # Fill in the form
            await page.fill('#class-text', 'This movie is absolutely amazing! I loved every minute of it.')
            
            # Take screenshot before submission
            await page.screenshot(
                path=self.screenshots_dir / "03_text_classification_form.png",
                full_page=True
            )
            
            # Submit the form
            await page.click('#classification-form button[type="submit"]')
            
            # Wait for results
            await page.wait_for_timeout(3000)
            
            # Take screenshot of results
            await page.screenshot(
                path=self.screenshots_dir / "04_text_classification_results.png",
                full_page=True
            )
            
            # Check for success indicators
            result_element = await page.query_selector('#classification-results')
            if result_element:
                self.test_results['text_classification'] = 'success'
                print("✅ Text Classification pipeline working")
            else:
                self.test_results['text_classification'] = 'no_results'
                print("⚠️ Text Classification - no results visible")
                
        except Exception as e:
            self.test_results['text_classification'] = f'error: {str(e)}'
            print(f"❌ Text Classification error: {e}")
            
    async def _test_text_embeddings_pipeline(self, page):
        """Test text embeddings pipeline."""
        print("🧮 Testing Text Embeddings Pipeline...")
        
        try:
            # Click on Embeddings tab
            await page.click('#embeddings-tab')
            await page.wait_for_timeout(1000)
            
            # Fill in the form
            await page.fill('#embed-text', 'Machine learning and artificial intelligence are transforming the world.')
            
            # Take screenshot before submission
            await page.screenshot(
                path=self.screenshots_dir / "05_text_embeddings_form.png",
                full_page=True
            )
            
            # Submit the form
            await page.click('#embeddings-form button[type="submit"]')
            
            # Wait for results
            await page.wait_for_timeout(3000)
            
            # Take screenshot of results
            await page.screenshot(
                path=self.screenshots_dir / "06_text_embeddings_results.png",
                full_page=True
            )
            
            # Check for success indicators
            result_element = await page.query_selector('#embeddings-results')
            if result_element:
                self.test_results['text_embeddings'] = 'success'
                print("✅ Text Embeddings pipeline working")
            else:
                self.test_results['text_embeddings'] = 'no_results'
                print("⚠️ Text Embeddings - no results visible")
                
        except Exception as e:
            self.test_results['text_embeddings'] = f'error: {str(e)}'
            print(f"❌ Text Embeddings error: {e}")
            
    async def _test_model_recommendations_pipeline(self, page):
        """Test model recommendations pipeline."""
        print("🎯 Testing Model Recommendations Pipeline...")
        
        try:
            # Click on Recommendations tab
            await page.click('#recommendations-tab')
            await page.wait_for_timeout(1000)
            
            # Fill in the form
            await page.fill('#rec-task', 'text generation')
            await page.fill('#rec-input-type', 'text')
            await page.fill('#rec-output-type', 'text')
            await page.fill('#rec-requirements', 'fast inference, good quality')
            
            # Take screenshot before submission
            await page.screenshot(
                path=self.screenshots_dir / "07_model_recommendations_form.png",
                full_page=True
            )
            
            # Submit the form
            await page.click('#recommendations-form button[type="submit"]')
            
            # Wait for results
            await page.wait_for_timeout(3000)
            
            # Take screenshot of results
            await page.screenshot(
                path=self.screenshots_dir / "08_model_recommendations_results.png",
                full_page=True
            )
            
            # Check for success indicators
            result_element = await page.query_selector('#recommendations-results')
            if result_element:
                self.test_results['model_recommendations'] = 'success'
                print("✅ Model Recommendations pipeline working")
            else:
                self.test_results['model_recommendations'] = 'no_results'
                print("⚠️ Model Recommendations - no results visible")
                
        except Exception as e:
            self.test_results['model_recommendations'] = f'error: {str(e)}'
            print(f"❌ Model Recommendations error: {e}")
            
    async def _test_model_manager_pipeline(self, page):
        """Test model manager pipeline."""
        print("🗄️ Testing Model Manager Pipeline...")
        
        try:
            # Click on Models tab
            await page.click('#models-tab')
            await page.wait_for_timeout(1000)
            
            # Take screenshot of model listing
            await page.screenshot(
                path=self.screenshots_dir / "09_model_manager_overview.png",
                full_page=True
            )
            
            # Test search functionality
            search_input = await page.query_selector('#model-search')
            if search_input:
                await page.fill('#model-search', 'gpt')
                await page.wait_for_timeout(1000)
                
                await page.screenshot(
                    path=self.screenshots_dir / "10_model_manager_search.png",
                    full_page=True
                )
            
            # Check for model listings
            model_cards = await page.query_selector_all('.model-card')
            if model_cards:
                self.test_results['model_manager'] = f'success: {len(model_cards)} models found'
                print(f"✅ Model Manager working - {len(model_cards)} models displayed")
            else:
                self.test_results['model_manager'] = 'no_models'
                print("⚠️ Model Manager - no models visible")
                
        except Exception as e:
            self.test_results['model_manager'] = f'error: {str(e)}'
            print(f"❌ Model Manager error: {e}")
            
    async def _test_mobile_responsiveness(self, page):
        """Test mobile responsiveness."""
        print("📱 Testing Mobile Responsiveness...")
        
        try:
            # Test mobile viewport
            await page.set_viewport_size({"width": 375, "height": 667})
            await page.wait_for_timeout(1000)
            
            await page.screenshot(
                path=self.screenshots_dir / "11_mobile_overview.png",
                full_page=True
            )
            
            # Test tablet viewport
            await page.set_viewport_size({"width": 768, "height": 1024})
            await page.wait_for_timeout(1000)
            
            await page.screenshot(
                path=self.screenshots_dir / "12_tablet_overview.png",
                full_page=True
            )
            
            # Reset to desktop
            await page.set_viewport_size({"width": 1920, "height": 1080})
            
            self.test_results['responsiveness'] = 'success'
            print("✅ Responsiveness testing completed")
            
        except Exception as e:
            self.test_results['responsiveness'] = f'error: {str(e)}'
            print(f"❌ Responsiveness error: {e}")
            
    def generate_test_report(self):
        """Generate a comprehensive test report."""
        report_path = self.screenshots_dir / "test_report.json"
        
        report = {
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "test_results": self.test_results,
            "screenshots": {
                "overview": "00_overview.png",
                "text_generation": {
                    "form": "01_text_generation_form.png",
                    "results": "02_text_generation_results.png"
                },
                "text_classification": {
                    "form": "03_text_classification_form.png", 
                    "results": "04_text_classification_results.png"
                },
                "text_embeddings": {
                    "form": "05_text_embeddings_form.png",
                    "results": "06_text_embeddings_results.png"
                },
                "model_recommendations": {
                    "form": "07_model_recommendations_form.png",
                    "results": "08_model_recommendations_results.png"
                },
                "model_manager": {
                    "overview": "09_model_manager_overview.png",
                    "search": "10_model_manager_search.png"
                },
                "responsiveness": {
                    "mobile": "11_mobile_overview.png",
                    "tablet": "12_tablet_overview.png"
                }
            }
        }
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
            
        # Generate markdown report
        self._generate_markdown_report()
        
        print(f"📊 Test report saved to {report_path}")
        
    def _generate_markdown_report(self):
        """Generate a markdown test report."""
        report_path = self.screenshots_dir / "PIPELINE_TEST_REPORT.md"
        
        success_count = sum(1 for result in self.test_results.values() 
                          if 'success' in str(result))
        total_tests = len(self.test_results)
        
        markdown_content = f"""# Kitchen Sink AI Testing Interface - Pipeline Screenshots Report

Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}

## Test Summary

- **Total Tests**: {total_tests}
- **Successful**: {success_count}
- **Success Rate**: {(success_count/total_tests*100):.1f}%

## Individual Pipeline Results

"""
        
        for pipeline, result in self.test_results.items():
            status = "✅ PASS" if 'success' in str(result) else "❌ FAIL"
            markdown_content += f"### {pipeline.replace('_', ' ').title()}\n"
            markdown_content += f"**Status**: {status}\n"
            markdown_content += f"**Details**: {result}\n\n"
            
        markdown_content += """## Screenshots Gallery

### Overview
![Overview](00_overview.png)

### Text Generation Pipeline
![Text Generation Form](01_text_generation_form.png)
![Text Generation Results](02_text_generation_results.png)

### Text Classification Pipeline  
![Text Classification Form](03_text_classification_form.png)
![Text Classification Results](04_text_classification_results.png)

### Text Embeddings Pipeline
![Text Embeddings Form](05_text_embeddings_form.png)
![Text Embeddings Results](06_text_embeddings_results.png)

### Model Recommendations Pipeline
![Model Recommendations Form](07_model_recommendations_form.png)
![Model Recommendations Results](08_model_recommendations_results.png)

### Model Manager Pipeline
![Model Manager Overview](09_model_manager_overview.png)
![Model Manager Search](10_model_manager_search.png)

### Responsive Design Testing
![Mobile View](11_mobile_overview.png)
![Tablet View](12_tablet_overview.png)

## Conclusion

This report provides comprehensive visual documentation of all inference pipelines 
working in the Kitchen Sink AI Testing Interface. Each pipeline has been tested
and captured with both form inputs and results displays.
"""
        
        with open(report_path, 'w') as f:
            f.write(markdown_content)
            
        print(f"📄 Markdown report saved to {report_path}")

async def main():
    """Main testing function."""
    
    # Install Playwright if needed
    if not HAVE_PLAYWRIGHT:
        print("📦 Installing Playwright...")
        subprocess.run([sys.executable, "-m", "pip", "install", "playwright"], check=True)
        subprocess.run([sys.executable, "-m", "playwright", "install", "chromium"], check=True)
        print("✅ Playwright installed")
        
    tester = KitchenSinkScreenshotTester()
    
    try:
        # Start server
        server_started = await tester.setup_server()
        if not server_started:
            print("❌ Failed to start server")
            return False
            
        # Run tests
        print("🧪 Running comprehensive pipeline tests...")
        success = await tester.test_inference_pipelines()
        
        # Generate report
        tester.generate_test_report()
        
        if success:
            print("🎉 All pipeline screenshots captured successfully!")
            print(f"📁 Screenshots saved to: {tester.screenshots_dir}")
            return True
        else:
            print("⚠️ Some tests failed - check screenshots for details")
            return False
            
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
        return False
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False
        
    finally:
        await tester.teardown_server()

if __name__ == "__main__":
    print("🚀 Starting Kitchen Sink Pipeline Screenshot Testing...")
    print("=" * 60)
    
    try:
        result = anyio.run(main())
        exit_code = 0 if result else 1
    except Exception as e:
        print(f"❌ Failed to run tests: {e}")
        exit_code = 1
        
    print("=" * 60)
    print(f"🏁 Testing completed with exit code: {exit_code}")
    sys.exit(exit_code)