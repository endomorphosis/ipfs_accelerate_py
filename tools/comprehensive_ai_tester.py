#!/usr/bin/env python3
"""
Comprehensive Testing Script for AI Inference CLI and MCP Server

This script tests all inference types available in the CLI and MCP server,
and attempts to use Playwright for browser automation testing.
"""

import asyncio
import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Any

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("comprehensive_ai_test")

class ComprehensiveAITester:
    """Test all AI inference capabilities."""
    
    def __init__(self):
        """Initialize the tester."""
        self.cli_path = Path(__file__).parent / "ai_inference_cli.py"
        self.test_results = {}
        self.browser_available = False
        
    def check_browser_availability(self) -> bool:
        """Check if Playwright browser is available."""
        try:
            from playwright.sync_api import sync_playwright
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                browser.close()
            self.browser_available = True
            logger.info("✅ Playwright browser is available")
            return True
        except Exception as e:
            logger.warning(f"⚠️ Playwright browser not available: {e}")
            self.browser_available = False
            return False
    
    def run_cli_command(self, command_args: List[str]) -> Dict[str, Any]:
        """Run a CLI command and return the result."""
        try:
            cmd = [sys.executable, str(self.cli_path)] + command_args
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=30
            )
            
            if result.returncode == 0:
                try:
                    # Try to parse JSON output
                    output = json.loads(result.stdout)
                    return {"success": True, "data": output, "raw_output": result.stdout}
                except json.JSONDecodeError:
                    # If not JSON, return raw text
                    return {"success": True, "data": result.stdout.strip(), "raw_output": result.stdout}
            else:
                return {
                    "success": False, 
                    "error": result.stderr, 
                    "stdout": result.stdout,
                    "returncode": result.returncode
                }
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Command timed out"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def test_text_processing(self) -> Dict[str, Any]:
        """Test all text processing commands."""
        logger.info("🔤 Testing text processing...")
        tests = {}
        
        # Text generation
        tests["generate"] = self.run_cli_command([
            "text", "generate", 
            "--prompt", "The future of AI is",
            "--max-length", "30"
        ])
        
        # Text classification  
        tests["classify"] = self.run_cli_command([
            "text", "classify",
            "--text", "I love this amazing product!"
        ])
        
        # Text embeddings
        tests["embeddings"] = self.run_cli_command([
            "text", "embeddings",
            "--text", "This is a test sentence for embedding generation"
        ])
        
        # Fill mask
        tests["fill_mask"] = self.run_cli_command([
            "text", "fill-mask",
            "--text", "The [MASK] is shining brightly today"
        ])
        
        # Translation
        tests["translate"] = self.run_cli_command([
            "text", "translate",
            "--text", "Hello world",
            "--source-lang", "en",
            "--target-lang", "es"
        ])
        
        # Summarization
        tests["summarize"] = self.run_cli_command([
            "text", "summarize",
            "--text", "Artificial intelligence is a rapidly growing field that has many applications in various industries. It involves the development of computer systems that can perform tasks that typically require human intelligence."
        ])
        
        # Question answering
        tests["question"] = self.run_cli_command([
            "text", "question",
            "--question", "What is AI?",
            "--context", "Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans."
        ])
        
        return tests
    
    def test_audio_processing(self) -> Dict[str, Any]:
        """Test audio processing commands."""
        logger.info("🎵 Testing audio processing...")
        tests = {}
        
        # Create a dummy audio file for testing
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            # Write minimal WAV header for testing
            f.write(b'RIFF\x24\x08\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x22\x56\x00\x00\x44\xAC\x00\x00\x02\x00\x10\x00data\x00\x08\x00\x00')
            f.write(b'\x00\x00' * 1000)  # Silent audio data
            audio_file = f.name
        
        try:
            # Audio transcription
            tests["transcribe"] = self.run_cli_command([
                "audio", "transcribe",
                "--audio-file", audio_file
            ])
            
            # Audio classification
            tests["classify"] = self.run_cli_command([
                "audio", "classify", 
                "--audio-file", audio_file
            ])
            
            # Speech synthesis
            tests["synthesize"] = self.run_cli_command([
                "audio", "synthesize",
                "--text", "Hello, this is a test of speech synthesis"
            ])
            
            # Audio generation
            tests["generate"] = self.run_cli_command([
                "audio", "generate",
                "--prompt", "Generate peaceful nature sounds"
            ])
            
        finally:
            # Clean up temporary file
            try:
                os.unlink(audio_file)
            except:
                pass
        
        return tests
    
    def test_vision_processing(self) -> Dict[str, Any]:
        """Test vision processing commands."""
        logger.info("👁️ Testing vision processing...")
        tests = {}
        
        # Create a dummy image file for testing
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            # Minimal JPEG header for testing
            f.write(b'\xFF\xD8\xFF\xE0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xFF\xDB\x00C\x00')
            f.write(b'\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f')
            f.write(b'\xFF\xD9')  # End of JPEG
            image_file = f.name
        
        try:
            # Image classification
            tests["classify"] = self.run_cli_command([
                "vision", "classify",
                "--image-file", image_file
            ])
            
            # Object detection
            tests["detect"] = self.run_cli_command([
                "vision", "detect",
                "--image-file", image_file
            ])
            
            # Image segmentation
            tests["segment"] = self.run_cli_command([
                "vision", "segment",
                "--image-file", image_file
            ])
            
            # Image generation
            tests["generate"] = self.run_cli_command([
                "vision", "generate",
                "--prompt", "A beautiful sunset over mountains"
            ])
            
        finally:
            # Clean up temporary file
            try:
                os.unlink(image_file)
            except:
                pass
        
        return tests
    
    def test_multimodal_processing(self) -> Dict[str, Any]:
        """Test multimodal processing commands."""
        logger.info("🔄 Testing multimodal processing...")
        tests = {}
        
        # Create dummy files for testing
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(b'\xFF\xD8\xFF\xE0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xFF\xD9')
            image_file = f.name
        
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(b'%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>\nendobj\nxref\n0 4\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \ntrailer\n<< /Size 4 /Root 1 0 R >>\nstartxref\n174\n%%EOF')
            document_file = f.name
        
        try:
            # Image captioning
            tests["caption"] = self.run_cli_command([
                "multimodal", "caption",
                "--image-file", image_file
            ])
            
            # Visual question answering
            tests["vqa"] = self.run_cli_command([
                "multimodal", "vqa",
                "--image-file", image_file,
                "--question", "What do you see in this image?"
            ])
            
            # Document processing
            tests["document"] = self.run_cli_command([
                "multimodal", "document",
                "--document-file", document_file,
                "--query", "What is the main topic?"
            ])
            
        finally:
            # Clean up temporary files
            for f in [image_file, document_file]:
                try:
                    os.unlink(f)
                except:
                    pass
        
        return tests
    
    def test_specialized_processing(self) -> Dict[str, Any]:
        """Test specialized processing commands."""
        logger.info("⚙️ Testing specialized processing...")
        tests = {}
        
        # Create dummy data files for testing
        timeseries_data = [1.0, 1.1, 1.2, 1.3, 1.1, 1.0, 0.9, 1.0, 1.1, 1.2]
        with tempfile.NamedTemporaryFile(mode='w', suffix=".json", delete=False) as f:
            json.dump(timeseries_data, f)
            timeseries_file = f.name
        
        tabular_data = {
            "feature1": [1, 2, 3, 4, 5],
            "feature2": [0.1, 0.2, 0.3, 0.4, 0.5],
            "target": [0, 1, 0, 1, 0]
        }
        with tempfile.NamedTemporaryFile(mode='w', suffix=".json", delete=False) as f:
            json.dump(tabular_data, f)
            tabular_file = f.name
        
        try:
            # Code generation
            tests["code"] = self.run_cli_command([
                "specialized", "code",
                "--prompt", "Create a function to calculate fibonacci numbers",
                "--language", "python"
            ])
            
            # Time series prediction
            tests["timeseries"] = self.run_cli_command([
                "specialized", "timeseries",
                "--data-file", timeseries_file,
                "--forecast-horizon", "5"
            ])
            
            # Tabular data processing
            tests["tabular"] = self.run_cli_command([
                "specialized", "tabular",
                "--data-file", tabular_file,
                "--task", "classification"
            ])
            
        finally:
            # Clean up temporary files
            for f in [timeseries_file, tabular_file]:
                try:
                    os.unlink(f)
                except:
                    pass
        
        return tests
    
    def test_system_commands(self) -> Dict[str, Any]:
        """Test system management commands."""
        logger.info("🔧 Testing system commands...")
        tests = {}
        
        # List models
        tests["list_models"] = self.run_cli_command([
            "system", "list-models",
            "--limit", "5"
        ])
        
        # Get statistics
        tests["stats"] = self.run_cli_command([
            "system", "stats"
        ])
        
        # Available types
        tests["available_types"] = self.run_cli_command([
            "system", "available-types"
        ])
        
        # Model recommendation
        tests["recommend"] = self.run_cli_command([
            "system", "recommend",
            "--task-type", "text_generation"
        ])
        
        return tests
    
    async def test_browser_automation(self) -> Dict[str, Any]:
        """Test browser automation capabilities."""
        logger.info("🌐 Testing browser automation...")
        
        if not self.browser_available:
            return {"success": False, "error": "Browser not available"}
        
        try:
            from playwright.async_api import async_playwright
            
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()
                
                # Create a simple HTML page for testing
                html_content = """
                <!DOCTYPE html>
                <html>
                <head><title>AI Inference Test</title></head>
                <body>
                    <h1>AI Inference Testing Dashboard</h1>
                    <div id="test-content">
                        <p>This is a test page for browser automation</p>
                        <button id="test-button">Test Button</button>
                        <input id="test-input" placeholder="Test input">
                    </div>
                </body>
                </html>
                """
                
                await page.set_content(html_content)
                
                # Take a screenshot
                screenshot_path = "test_screenshot.png"
                await page.screenshot(path=screenshot_path)
                
                # Test interactions
                await page.click("#test-button")
                await page.fill("#test-input", "Test automation working!")
                
                # Get page title
                title = await page.title()
                
                await browser.close()
                
                return {
                    "success": True,
                    "title": title,
                    "screenshot_saved": screenshot_path,
                    "interactions_tested": ["click", "fill"]
                }
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all tests and return comprehensive results."""
        logger.info("🚀 Starting comprehensive AI inference testing...")
        
        # Check browser availability
        self.check_browser_availability()
        
        # Run all tests
        results = {
            "text_processing": self.test_text_processing(),
            "audio_processing": self.test_audio_processing(),
            "vision_processing": self.test_vision_processing(),
            "multimodal_processing": self.test_multimodal_processing(),
            "specialized_processing": self.test_specialized_processing(),
            "system_commands": self.test_system_commands(),
        }
        
        # Run browser automation test
        if self.browser_available:
            try:
                browser_result = asyncio.run(self.test_browser_automation())
                results["browser_automation"] = browser_result
            except Exception as e:
                results["browser_automation"] = {"success": False, "error": str(e)}
        else:
            results["browser_automation"] = {"success": False, "error": "Browser not available"}
        
        # Calculate overall statistics
        total_tests = 0
        successful_tests = 0
        
        for category, tests in results.items():
            if isinstance(tests, dict):
                for test_name, test_result in tests.items():
                    total_tests += 1
                    if isinstance(test_result, dict) and test_result.get("success"):
                        successful_tests += 1
        
        results["summary"] = {
            "total_tests": total_tests,
            "successful_tests": successful_tests,
            "success_rate": f"{(successful_tests/total_tests*100):.1f}%" if total_tests > 0 else "0%",
            "browser_available": self.browser_available
        }
        
        return results
    
    def print_results(self, results: Dict[str, Any]):
        """Print formatted test results."""
        print("\n" + "="*80)
        print("🧪 COMPREHENSIVE AI INFERENCE TEST RESULTS")
        print("="*80)
        
        summary = results.get("summary", {})
        print(f"📊 Overall Results:")
        print(f"   Total Tests: {summary.get('total_tests', 0)}")
        print(f"   Successful: {summary.get('successful_tests', 0)}")
        print(f"   Success Rate: {summary.get('success_rate', '0%')}")
        print(f"   Browser Available: {'✅' if summary.get('browser_available') else '❌'}")
        print()
        
        for category, tests in results.items():
            if category == "summary":
                continue
                
            print(f"📂 {category.replace('_', ' ').title()}:")
            
            if isinstance(tests, dict):
                for test_name, test_result in tests.items():
                    if isinstance(test_result, dict):
                        status = "✅" if test_result.get("success") else "❌"
                        print(f"   {status} {test_name}")
                        if not test_result.get("success") and "error" in test_result:
                            print(f"      Error: {test_result['error']}")
                    else:
                        print(f"   ℹ️  {test_name}: {test_result}")
            else:
                status = "✅" if tests.get("success") else "❌"
                print(f"   {status} {category}")
            print()

def main():
    """Main function to run comprehensive tests."""
    tester = ComprehensiveAITester()
    results = tester.run_all_tests()
    
    # Print results to console
    tester.print_results(results)
    
    # Save detailed results to file
    with open("comprehensive_test_results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("📄 Detailed results saved to: comprehensive_test_results.json")
    
    # Return appropriate exit code
    summary = results.get("summary", {})
    success_rate = float(summary.get("success_rate", "0%").rstrip("%"))
    
    if success_rate >= 80:
        print("🎉 Tests completed successfully!")
        return 0
    else:
        print("⚠️ Some tests failed. Check the results above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())