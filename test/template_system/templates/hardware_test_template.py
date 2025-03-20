"""
Hardware test template for IPFS Accelerate tests.

This module provides a template for generating hardware-specific tests,
such as tests for WebGPU, WebNN, CUDA, ROCm, etc.
"""

import os
import logging
from typing import Dict, List, Any, Optional

from .base_template import BaseTemplate


class HardwareTestTemplate(BaseTemplate):
    """
    Template for hardware tests.
    
    This template generates test files for specific hardware platforms,
    including tests for device detection, computation, and hardware-specific
    capabilities.
    """
    
    def validate_parameters(self) -> bool:
        """
        Validate hardware test parameters.
        
        Returns:
            True if parameters are valid, False otherwise
        """
        required_params = ['hardware_platform', 'test_name']
        
        for param in required_params:
            if param not in self.parameters:
                self.logger.error(f"Missing required parameter: {param}")
                return False
        
        valid_platforms = ['webgpu', 'webnn', 'cuda', 'rocm', 'cpu']
        if self.parameters['hardware_platform'] not in valid_platforms:
            self.logger.error(f"Invalid hardware_platform: {self.parameters['hardware_platform']}")
            return False
        
        return True
    
    def generate_imports(self) -> str:
        """
        Generate import statements for hardware tests.
        
        Returns:
            String with import statements
        """
        platform = self.parameters['hardware_platform']
        
        imports = [
            "import os",
            "import pytest",
            "import logging",
            "import time",
            "from typing import Dict, List, Any, Optional",
            "",
            "# Import common utilities",
            "from common.hardware_detection import detect_hardware, setup_platform",
            ""
        ]
        
        # Add platform-specific imports
        if platform == 'webgpu':
            imports.extend([
                "# WebGPU-specific imports",
                "try:",
                "    from selenium import webdriver",
                "    from selenium.webdriver.chrome.options import Options",
                "    from selenium.webdriver.common.by import By",
                "except ImportError:",
                "    pass",
                ""
            ])
        elif platform == 'webnn':
            imports.extend([
                "# WebNN-specific imports",
                "try:",
                "    from selenium import webdriver",
                "    from selenium.webdriver.chrome.options import Options",
                "    from selenium.webdriver.common.by import By",
                "except ImportError:",
                "    pass",
                ""
            ])
        elif platform == 'cuda':
            imports.extend([
                "# CUDA-specific imports",
                "try:",
                "    import torch",
                "    import numpy as np",
                "except ImportError:",
                "    pass",
                ""
            ])
        elif platform == 'rocm':
            imports.extend([
                "# ROCm-specific imports",
                "try:",
                "    import torch",
                "    import numpy as np",
                "except ImportError:",
                "    pass",
                ""
            ])
        
        imports.append("")
        
        # Add fixture imports
        if platform == 'webgpu':
            imports.append("from common.fixtures import webgpu_browser")
        elif platform == 'webnn':
            imports.append("from common.fixtures import webnn_browser")
        elif platform == 'cuda':
            imports.append("from common.fixtures import cuda_device")
        elif platform == 'rocm':
            imports.append("from common.fixtures import rocm_device")
        
        imports.append("")
        
        return "\n".join(imports)
    
    def generate_fixtures(self) -> str:
        """
        Generate fixtures for hardware tests.
        
        Returns:
            String with fixture definitions
        """
        platform = self.parameters['hardware_platform']
        test_op = self.parameters.get('test_operation', 'matmul')
        
        fixtures = [
            "# Hardware-specific fixtures",
        ]
        
        if platform in ('cuda', 'rocm'):
            fixtures.extend([
                "@pytest.fixture",
                "def test_tensors(request):",
                "    \"\"\"Create test tensors for computation tests.\"\"\"",
                "    shape = getattr(request, 'param', (1024, 1024))",
                "    try:",
                "        import torch",
                "        a = torch.rand(*shape)",
                "        b = torch.rand(*shape)",
                "        return a, b",
                "    except ImportError:",
                "        pytest.skip(\"PyTorch not available\")",
                ""
            ])
        elif platform in ('webgpu', 'webnn'):
            fixtures.extend([
                "@pytest.fixture",
                f"def {platform}_test_page(temp_dir):",
                f"    \"\"\"Create a test HTML page for {platform} tests.\"\"\"",
                "    html_content = f\"\"\"",
                "    <!DOCTYPE html>",
                "    <html>",
                "    <head>",
                f"        <title>{platform.upper()} Test</title>",
                "        <script>",
                "            async function runTest() {",
                "                const resultElement = document.getElementById('result');",
                "                try {",
                f"                    // Check for {platform} support",
                f"                    if ('{platform}' === 'webgpu') {{",
                "                        if (!navigator.gpu) {",
                "                            resultElement.textContent = 'WebGPU not supported';",
                "                            return;",
                "                        }",
                "                        const adapter = await navigator.gpu.requestAdapter();",
                "                        if (!adapter) {",
                "                            resultElement.textContent = 'Couldn\\'t request WebGPU adapter';",
                "                            return;",
                "                        }",
                "                        const device = await adapter.requestDevice();",
                "                        resultElement.textContent = 'WebGPU device created successfully';",
                f"                    }} else if ('{platform}' === 'webnn') {{",
                "                        if (!('ml' in navigator)) {",
                "                            resultElement.textContent = 'WebNN not supported';",
                "                            return;",
                "                        }",
                "                        const context = await navigator.ml.createContext();",
                "                        if (!context) {",
                "                            resultElement.textContent = 'Couldn\\'t create WebNN context';",
                "                            return;",
                "                        }",
                "                        resultElement.textContent = 'WebNN context created successfully';",
                "                    }}",
                "                } catch (error) {",
                "                    resultElement.textContent = `Error: ${error.message}`;",
                "                }",
                "            }",
                "            ",
                "            window.onload = runTest;",
                "        </script>",
                "    </head>",
                "    <body>",
                f"        <h1>{platform.upper()} Test</h1>",
                "        <div id=\"result\">Testing...</div>",
                "    </body>",
                "    </html>",
                "    \"\"\"",
                "    ",
                "    file_path = os.path.join(temp_dir, 'test_page.html')",
                "    with open(file_path, 'w') as f:",
                "        f.write(html_content)",
                "    ",
                "    return file_path",
                ""
            ])
        
        return "\n".join(fixtures)
    
    def generate_test_class(self) -> str:
        """
        Generate the test class for hardware tests.
        
        Returns:
            String with test class definition
        """
        platform = self.parameters['hardware_platform']
        test_name = self.parameters.get('test_name', f"{platform}_compute")
        class_name = ''.join(word.capitalize() for word in test_name.split('_'))
        
        # Platform-specific test methods
        if platform == 'webgpu':
            test_methods = [
                "@pytest.mark.webgpu",
                "def test_webgpu_available(self):",
                "    \"\"\"Test WebGPU availability.\"\"\"",
                "    hardware_info = detect_hardware()",
                "    assert hardware_info['platforms']['webgpu']['available']",
                "",
                "@pytest.mark.webgpu",
                "def test_webgpu_browser_launch(self, webgpu_browser):",
                "    \"\"\"Test WebGPU browser launch.\"\"\"",
                "    assert webgpu_browser is not None",
                "",
                "@pytest.mark.webgpu",
                "def test_webgpu_device_creation(self, webgpu_browser, webgpu_test_page):",
                "    \"\"\"Test WebGPU device creation.\"\"\"",
                "    webgpu_browser.get(f\"file://{webgpu_test_page}\")",
                "    time.sleep(2)  # Allow time for JavaScript to execute",
                "    result_element = webgpu_browser.find_element(By.ID, 'result')",
                "    assert result_element.text == 'WebGPU device created successfully'",
                "",
                "@pytest.mark.webgpu",
                "def test_webgpu_compute(self, webgpu_browser):",
                "    \"\"\"Test WebGPU compute operation.\"\"\"",
                "    # This would be expanded in a real implementation",
                "    # Currently just a placeholder test",
                "    assert webgpu_browser is not None",
                ""
            ]
        elif platform == 'webnn':
            test_methods = [
                "@pytest.mark.webnn",
                "def test_webnn_available(self):",
                "    \"\"\"Test WebNN availability.\"\"\"",
                "    hardware_info = detect_hardware()",
                "    assert hardware_info['platforms']['webnn']['available']",
                "",
                "@pytest.mark.webnn",
                "def test_webnn_browser_launch(self, webnn_browser):",
                "    \"\"\"Test WebNN browser launch.\"\"\"",
                "    assert webnn_browser is not None",
                "",
                "@pytest.mark.webnn",
                "def test_webnn_context_creation(self, webnn_browser, webnn_test_page):",
                "    \"\"\"Test WebNN context creation.\"\"\"",
                "    webnn_browser.get(f\"file://{webnn_test_page}\")",
                "    time.sleep(2)  # Allow time for JavaScript to execute",
                "    result_element = webnn_browser.find_element(By.ID, 'result')",
                "    assert result_element.text == 'WebNN context created successfully'",
                "",
                "@pytest.mark.webnn",
                "def test_webnn_compute(self, webnn_browser):",
                "    \"\"\"Test WebNN compute operation.\"\"\"",
                "    # This would be expanded in a real implementation",
                "    # Currently just a placeholder test",
                "    assert webnn_browser is not None",
                ""
            ]
        elif platform == 'cuda':
            test_methods = [
                "@pytest.mark.cuda",
                "def test_cuda_available(self):",
                "    \"\"\"Test CUDA availability.\"\"\"",
                "    hardware_info = detect_hardware()",
                "    assert hardware_info['platforms']['cuda']['available']",
                "",
                "@pytest.mark.cuda",
                "def test_cuda_device(self, cuda_device):",
                "    \"\"\"Test CUDA device.\"\"\"",
                "    assert cuda_device.type == 'cuda'",
                "",
                "@pytest.mark.cuda",
                "@pytest.mark.parametrize('test_tensors', [(1024, 1024), (2048, 2048)], indirect=True)",
                "def test_cuda_matmul(self, cuda_device, test_tensors):",
                "    \"\"\"Test matrix multiplication on CUDA.\"\"\"",
                "    a, b = test_tensors",
                "    a_cuda = a.to(cuda_device)",
                "    b_cuda = b.to(cuda_device)",
                "    ",
                "    # Warmup",
                "    for _ in range(5):",
                "        _ = torch.matmul(a_cuda, b_cuda)",
                "    ",
                "    # Benchmark",
                "    start_time = time.time()",
                "    for _ in range(10):",
                "        c_cuda = torch.matmul(a_cuda, b_cuda)",
                "    torch.cuda.synchronize()",
                "    end_time = time.time()",
                "    ",
                "    duration = (end_time - start_time) / 10",
                "    logging.info(f\"CUDA matmul duration: {duration:.6f} seconds\")",
                "    ",
                "    # Verify result is on CUDA",
                "    assert c_cuda.device.type == 'cuda'",
                ""
            ]
        elif platform == 'rocm':
            test_methods = [
                "@pytest.mark.rocm",
                "def test_rocm_available(self):",
                "    \"\"\"Test ROCm availability.\"\"\"",
                "    hardware_info = detect_hardware()",
                "    assert hardware_info['platforms']['rocm']['available']",
                "",
                "@pytest.mark.rocm",
                "def test_rocm_device(self, rocm_device):",
                "    \"\"\"Test ROCm device.\"\"\"",
                "    assert rocm_device.type == 'cuda'  # ROCm uses CUDA device type in PyTorch",
                "",
                "@pytest.mark.rocm",
                "@pytest.mark.parametrize('test_tensors', [(1024, 1024), (2048, 2048)], indirect=True)",
                "def test_rocm_matmul(self, rocm_device, test_tensors):",
                "    \"\"\"Test matrix multiplication on ROCm.\"\"\"",
                "    a, b = test_tensors",
                "    a_rocm = a.to(rocm_device)",
                "    b_rocm = b.to(rocm_device)",
                "    ",
                "    # Warmup",
                "    for _ in range(5):",
                "        _ = torch.matmul(a_rocm, b_rocm)",
                "    ",
                "    # Benchmark",
                "    start_time = time.time()",
                "    for _ in range(10):",
                "        c_rocm = torch.matmul(a_rocm, b_rocm)",
                "    torch.cuda.synchronize()",
                "    end_time = time.time()",
                "    ",
                "    duration = (end_time - start_time) / 10",
                "    logging.info(f\"ROCm matmul duration: {duration:.6f} seconds\")",
                "    ",
                "    # Verify result is on ROCm",
                "    assert c_rocm.device.type == 'cuda'",
                ""
            ]
        else:  # platform == 'cpu'
            test_methods = [
                "def test_cpu_available(self):",
                "    \"\"\"Test CPU availability.\"\"\"",
                "    hardware_info = detect_hardware()",
                "    assert hardware_info['platforms']['cpu']['available']",
                "",
                "def test_cpu_device(self, cpu_device):",
                "    \"\"\"Test CPU device.\"\"\"",
                "    assert cpu_device == 'cpu' or hasattr(cpu_device, 'type') and cpu_device.type == 'cpu'",
                "",
                "def test_cpu_compute(self):",
                "    \"\"\"Test computation on CPU.\"\"\"",
                "    try:",
                "        import torch",
                "        import numpy as np",
                "    except ImportError:",
                "        pytest.skip(\"PyTorch or NumPy not available\")",
                "    ",
                "    # Create test tensors",
                "    a = torch.rand(1024, 1024)",
                "    b = torch.rand(1024, 1024)",
                "    ",
                "    # Benchmark",
                "    start_time = time.time()",
                "    for _ in range(3):",
                "        c = torch.matmul(a, b)",
                "    end_time = time.time()",
                "    ",
                "    duration = (end_time - start_time) / 3",
                "    logging.info(f\"CPU matmul duration: {duration:.6f} seconds\")",
                "    ",
                "    assert c.shape == (1024, 1024)",
                ""
            ]
        
        test_class = [
            f"class Test{class_name}:",
            "    \"\"\"",
            f"    Tests for {platform} platform.",
            "    \"\"\"",
            ""
        ] + test_methods
        
        return "".join(f"    {line}\n" for line in test_class)
    
    def generate_content(self) -> str:
        """
        Generate the full content of the hardware test file.
        
        Returns:
            String with test file content
        """
        if not self.validate_parameters():
            raise ValueError("Invalid template parameters")
        
        platform = self.parameters['hardware_platform']
        
        content = [
            '"""',
            f"Test file for {platform} platform.",
            "",
            f"This file contains tests for the {platform} platform,",
            f"including device detection, computation, and {platform}-specific capabilities.",
            "Generated from HardwareTestTemplate.",
            '"""',
            "",
            self.generate_imports(),
            "",
            self.generate_fixtures(),
            "",
            self.generate_test_class()
        ]
        
        return "\n".join(content)
    
    def write(self, file_path: Optional[str] = None) -> str:
        """
        Write the rendered template to a file.
        
        Args:
            file_path: Path to write the file
            
        Returns:
            Path to the written file
        """
        if file_path is None:
            platform = self.parameters['hardware_platform']
            test_name = self.parameters.get('test_name', f"{platform}_compute")
            
            # Determine test category based on operation
            test_category = self.parameters.get('test_category', 'compute')
            
            dir_path = os.path.join(self.output_dir, "hardware", platform, test_category)
            os.makedirs(dir_path, exist_ok=True)
            
            file_path = os.path.join(dir_path, f"test_{test_name}.py")
        
        return super().write(file_path)