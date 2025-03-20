"""
API test template for IPFS Accelerate tests.

This module provides a template for generating API tests,
including tests for API endpoints, clients, and integrations.
"""

import os
import logging
from typing import Dict, List, Any, Optional

from .base_template import BaseTemplate


class APITestTemplate(BaseTemplate):
    """
    Template for API tests.
    
    This template generates test files for specific APIs,
    including tests for API endpoints, clients, and integrations.
    """
    
    def validate_parameters(self) -> bool:
        """
        Validate API test parameters.
        
        Returns:
            True if parameters are valid, False otherwise
        """
        required_params = ['api_name', 'test_name']
        
        for param in required_params:
            if param not in self.parameters:
                self.logger.error(f"Missing required parameter: {param}")
                return False
        
        valid_api_types = ['openai', 'hf_tei', 'hf_tgi', 'ollama', 'vllm', 'claude', 'internal']
        if 'api_type' in self.parameters and self.parameters['api_type'] not in valid_api_types:
            self.logger.error(f"Invalid api_type: {self.parameters['api_type']}")
            return False
        
        return True
    
    def generate_imports(self) -> str:
        """
        Generate import statements for API tests.
        
        Returns:
            String with import statements
        """
        api_name = self.parameters['api_name']
        api_type = self.parameters.get('api_type', 'internal')
        
        imports = [
            "import os",
            "import pytest",
            "import logging",
            "import json",
            "import time",
            "from typing import Dict, List, Any, Optional",
            "",
            "# Import common utilities",
            "from common.hardware_detection import detect_hardware",
            ""
        ]
        
        # Add API-specific imports
        if api_type == 'openai':
            imports.extend([
                "# OpenAI API imports",
                "try:",
                "    import openai",
                "    from openai import OpenAI",
                "except ImportError:",
                "    pass",
                ""
            ])
        elif api_type == 'hf_tei' or api_type == 'hf_tgi':
            imports.extend([
                f"# HuggingFace {api_type.upper()} imports",
                "try:",
                "    import requests",
                "    import transformers",
                "except ImportError:",
                "    pass",
                ""
            ])
        elif api_type == 'ollama':
            imports.extend([
                "# Ollama API imports",
                "try:",
                "    import requests",
                "except ImportError:",
                "    pass",
                ""
            ])
        elif api_type == 'vllm':
            imports.extend([
                "# vLLM API imports",
                "try:",
                "    import requests",
                "except ImportError:",
                "    pass",
                ""
            ])
        elif api_type == 'claude':
            imports.extend([
                "# Claude API imports",
                "try:",
                "    import anthropic",
                "    from anthropic import Anthropic",
                "except ImportError:",
                "    pass",
                ""
            ])
        else:  # internal
            imports.extend([
                "# Internal API imports",
                "try:",
                "    import requests",
                "except ImportError:",
                "    pass",
                ""
            ])
        
        return "\n".join(imports)
    
    def generate_fixtures(self) -> str:
        """
        Generate fixtures for API tests.
        
        Returns:
            String with fixture definitions
        """
        api_name = self.parameters['api_name']
        api_type = self.parameters.get('api_type', 'internal')
        api_var = api_name.replace('-', '_').lower()
        
        fixtures = [
            "# API-specific fixtures",
            "@pytest.fixture",
            "def api_base_url():",
            f"    \"\"\"Get the base URL for {api_name} API tests.\"\"\"",
            "    return os.environ.get(\"API_BASE_URL\", \"http://localhost:8000\")",
            "",
            "@pytest.fixture",
            "def api_key():",
            f"    \"\"\"Get the API key for {api_name} API tests.\"\"\"",
            "    return os.environ.get(\"API_KEY\", \"test_key\")",
            "",
        ]
        
        # Add API client fixture
        if api_type == 'openai':
            fixtures.extend([
                "@pytest.fixture",
                f"def {api_var}_client(api_base_url, api_key):",
                f"    \"\"\"Create an OpenAI API client for {api_name} tests.\"\"\"",
                "    try:",
                "        client = OpenAI(",
                "            base_url=api_base_url,",
                "            api_key=api_key",
                "        )",
                "        return client",
                "    except (ImportError, Exception) as e:",
                "        pytest.skip(f\"Could not create OpenAI client: {e}\")",
                ""
            ])
        elif api_type == 'hf_tei' or api_type == 'hf_tgi':
            fixtures.extend([
                "@pytest.fixture",
                f"def {api_var}_client(api_base_url, api_key):",
                f"    \"\"\"Create a HuggingFace {api_type.upper()} API client for {api_name} tests.\"\"\"",
                "    try:",
                "        import requests",
                "",
                "        class HFClient:",
                "            def __init__(self, base_url, api_key):",
                "                self.base_url = base_url",
                "                self.api_key = api_key",
                "                self.session = requests.Session()",
                "                self.session.headers.update({",
                "                    \"Authorization\": f\"Bearer {api_key}\",",
                "                    \"Content-Type\": \"application/json\"",
                "                })",
                "",
                "            def generate(self, inputs, **kwargs):",
                "                response = self.session.post(",
                "                    f\"{self.base_url}/generate\",",
                "                    json={\"inputs\": inputs, \"parameters\": kwargs}",
                "                )",
                "                response.raise_for_status()",
                "                return response.json()",
                "",
                "        return HFClient(api_base_url, api_key)",
                "    except (ImportError, Exception) as e:",
                "        pytest.skip(f\"Could not create HuggingFace client: {e}\")",
                ""
            ])
        elif api_type == 'ollama':
            fixtures.extend([
                "@pytest.fixture",
                f"def {api_var}_client(api_base_url, api_key):",
                f"    \"\"\"Create an Ollama API client for {api_name} tests.\"\"\"",
                "    try:",
                "        import requests",
                "",
                "        class OllamaClient:",
                "            def __init__(self, base_url):",
                "                self.base_url = base_url",
                "                self.session = requests.Session()",
                "                self.session.headers.update({",
                "                    \"Content-Type\": \"application/json\"",
                "                })",
                "",
                "            def generate(self, model, prompt, **kwargs):",
                "                response = self.session.post(",
                "                    f\"{self.base_url}/api/generate\",",
                "                    json={\"model\": model, \"prompt\": prompt, **kwargs}",
                "                )",
                "                response.raise_for_status()",
                "                return response.json()",
                "",
                "        return OllamaClient(api_base_url)",
                "    except (ImportError, Exception) as e:",
                "        pytest.skip(f\"Could not create Ollama client: {e}\")",
                ""
            ])
        elif api_type == 'claude':
            fixtures.extend([
                "@pytest.fixture",
                f"def {api_var}_client(api_key):",
                f"    \"\"\"Create a Claude API client for {api_name} tests.\"\"\"",
                "    try:",
                "        client = Anthropic(",
                "            api_key=api_key",
                "        )",
                "        return client",
                "    except (ImportError, Exception) as e:",
                "        pytest.skip(f\"Could not create Anthropic client: {e}\")",
                ""
            ])
        else:  # internal or vllm
            fixtures.extend([
                "@pytest.fixture",
                f"def {api_var}_client(api_base_url, api_key):",
                f"    \"\"\"Create an API client for {api_name} tests.\"\"\"",
                "    try:",
                "        import requests",
                "",
                "        class APIClient:",
                "            def __init__(self, base_url, api_key):",
                "                self.base_url = base_url",
                "                self.api_key = api_key",
                "                self.session = requests.Session()",
                "                self.session.headers.update({",
                "                    \"Authorization\": f\"Bearer {api_key}\",",
                "                    \"Content-Type\": \"application/json\"",
                "                })",
                "",
                "            def get(self, endpoint, params=None):",
                "                return self.session.get(",
                "                    f\"{self.base_url}{endpoint}\",",
                "                    params=params",
                "                )",
                "",
                "            def post(self, endpoint, data=None):",
                "                return self.session.post(",
                "                    f\"{self.base_url}{endpoint}\",",
                "                    json=data",
                "                )",
                "",
                "        return APIClient(api_base_url, api_key)",
                "    except (ImportError, Exception) as e:",
                "        pytest.skip(f\"Could not create API client: {e}\")",
                ""
            ])
        
        # Add mock fixture
        fixtures.extend([
            "@pytest.fixture",
            f"def mock_{api_var}_client():",
            f"    \"\"\"Create a mock client for {api_name} API tests.\"\"\"",
            "    try:",
            "        from unittest.mock import MagicMock",
            "",
            "        mock_client = MagicMock()",
            ""
        ])
        
        # Add mock responses based on API type
        if api_type == 'openai':
            fixtures.extend([
                "        # Mock completion response",
                "        mock_completion = MagicMock()",
                "        mock_completion.choices = [",
                "            MagicMock(message=MagicMock(content=\"Mock response\"))",
                "        ]",
                "        mock_client.chat.completions.create.return_value = mock_completion",
                "",
                "        # Mock embedding response",
                "        mock_embedding = MagicMock()",
                "        mock_embedding.data = [",
                "            MagicMock(embedding=[0.1, 0.2, 0.3])",
                "        ]",
                "        mock_client.embeddings.create.return_value = mock_embedding",
                ""
            ])
        elif api_type == 'hf_tei' or api_type == 'hf_tgi':
            fixtures.extend([
                "        # Mock generation response",
                "        mock_client.generate.return_value = {",
                "            \"generated_text\": \"Mock generated text\"",
                "        }",
                ""
            ])
        elif api_type == 'ollama':
            fixtures.extend([
                "        # Mock generation response",
                "        mock_client.generate.return_value = {",
                "            \"model\": \"llama2\",",
                "            \"response\": \"Mock generated text\",",
                "            \"context\": [1, 2, 3]",
                "        }",
                ""
            ])
        elif api_type == 'claude':
            fixtures.extend([
                "        # Mock message response",
                "        mock_message = MagicMock()",
                "        mock_message.content = [",
                "            {\"type\": \"text\", \"text\": \"Mock response\"}",
                "        ]",
                "        mock_client.messages.create.return_value = mock_message",
                ""
            ])
        else:  # internal or vllm
            fixtures.extend([
                "        # Mock API responses",
                "        mock_response = MagicMock()",
                "        mock_response.status_code = 200",
                "        mock_response.json.return_value = {\"result\": \"success\"}",
                "        mock_client.get.return_value = mock_response",
                "        mock_client.post.return_value = mock_response",
                ""
            ])
        
        fixtures.extend([
            "        return mock_client",
            "    except (ImportError, Exception) as e:",
            "        pytest.skip(f\"Could not create mock client: {e}\")",
            ""
        ])
        
        return "\n".join(fixtures)
    
    def generate_test_class(self) -> str:
        """
        Generate the test class for API tests.
        
        Returns:
            String with test class definition
        """
        api_name = self.parameters['api_name']
        api_type = self.parameters.get('api_type', 'internal')
        api_var = api_name.replace('-', '_').lower()
        test_name = self.parameters.get('test_name', f"{api_var}_api")
        class_name = ''.join(word.capitalize() for word in test_name.split('_'))
        
        test_class = [
            f"@pytest.mark.api",
            f"class Test{class_name}:",
            "    \"\"\"",
            f"    Tests for {api_name} API.",
            "    \"\"\"",
            ""
        ]
        
        # Add connection test
        test_class.extend([
            f"    def test_api_connection(self, {api_var}_client):",
            f"        \"\"\"Test connection to {api_name} API.\"\"\"",
            "        assert {api_var}_client is not None",
            ""
        ])
        
        # Add API-specific tests
        if api_type == 'openai':
            test_class.extend([
                "    def test_chat_completion(self, {api_var}_client):",
                f"        \"\"\"Test chat completion with {api_name} API.\"\"\"",
                "        try:",
                "            response = {api_var}_client.chat.completions.create(",
                "                model=\"gpt-3.5-turbo\",",
                "                messages=[",
                "                    {\"role\": \"system\", \"content\": \"You are a helpful assistant.\"},",
                "                    {\"role\": \"user\", \"content\": \"Hello!\"}"
                "                ]",
                "            )",
                "            ",
                "            assert response is not None",
                "            assert len(response.choices) > 0",
                "            assert response.choices[0].message.content",
                "        except Exception as e:",
                "            pytest.skip(f\"API test failed: {e}\")",
                "",
                "    def test_embeddings(self, {api_var}_client):",
                f"        \"\"\"Test embeddings with {api_name} API.\"\"\"",
                "        try:",
                "            response = {api_var}_client.embeddings.create(",
                "                model=\"text-embedding-ada-002\",",
                "                input=\"The quick brown fox jumps over the lazy dog\"",
                "            )",
                "            ",
                "            assert response is not None",
                "            assert len(response.data) > 0",
                "            assert len(response.data[0].embedding) > 0",
                "        except Exception as e:",
                "            pytest.skip(f\"API test failed: {e}\")",
                ""
            ])
        elif api_type == 'hf_tei' or api_type == 'hf_tgi':
            test_class.extend([
                "    def test_text_generation(self, {api_var}_client):",
                f"        \"\"\"Test text generation with {api_name} API.\"\"\"",
                "        try:",
                "            response = {api_var}_client.generate(",
                "                \"The quick brown fox\",",
                "                max_new_tokens=20,",
                "                temperature=0.7",
                "            )",
                "            ",
                "            assert response is not None",
                "            assert \"generated_text\" in response",
                "            assert response[\"generated_text\"]",
                "        except Exception as e:",
                "            pytest.skip(f\"API test failed: {e}\")",
                ""
            ])
        elif api_type == 'ollama':
            test_class.extend([
                "    def test_ollama_generation(self, {api_var}_client):",
                f"        \"\"\"Test text generation with {api_name} API.\"\"\"",
                "        try:",
                "            response = {api_var}_client.generate(",
                "                \"llama2\",",
                "                \"The capital of France is\",",
                "                temperature=0.7,",
                "                max_tokens=20",
                "            )",
                "            ",
                "            assert response is not None",
                "            assert \"response\" in response",
                "            assert response[\"response\"]",
                "        except Exception as e:",
                "            pytest.skip(f\"API test failed: {e}\")",
                ""
            ])
        elif api_type == 'claude':
            test_class.extend([
                "    def test_claude_messages(self, {api_var}_client):",
                f"        \"\"\"Test message generation with {api_name} API.\"\"\"",
                "        try:",
                "            response = {api_var}_client.messages.create(",
                "                model=\"claude-3-sonnet-20240229\",",
                "                max_tokens=500,",
                "                messages=[",
                "                    {\"role\": \"user\", \"content\": \"Hello, Claude!\"}"
                "                ]",
                "            )",
                "            ",
                "            assert response is not None",
                "            assert response.content",
                "            assert response.content[0].type == \"text\"",
                "            assert response.content[0].text",
                "        except Exception as e:",
                "            pytest.skip(f\"API test failed: {e}\")",
                ""
            ])
        else:  # internal or vllm
            test_class.extend([
                "    def test_models_endpoint(self, {api_var}_client):",
                f"        \"\"\"Test models endpoint of {api_name} API.\"\"\"",
                "        try:",
                "            response = {api_var}_client.get(\"/models\")",
                "            ",
                "            assert response.status_code == 200",
                "            data = response.json()",
                "            assert \"models\" in data",
                "        except Exception as e:",
                "            pytest.skip(f\"API test failed: {e}\")",
                "",
                "    def test_inference_endpoint(self, {api_var}_client):",
                f"        \"\"\"Test inference endpoint of {api_name} API.\"\"\"",
                "        try:",
                "            response = {api_var}_client.post(\"/inference\", {",
                "                \"model\": \"test-model\",",
                "                \"prompt\": \"Test prompt\"",
                "            })",
                "            ",
                "            assert response.status_code == 200",
                "            assert response.json() is not None",
                "        except Exception as e:",
                "            pytest.skip(f\"API test failed: {e}\")",
                ""
            ])
        
        # Add mock test
        test_class.extend([
            f"    def test_with_mock_client(self, mock_{api_var}_client):",
            f"        \"\"\"Test with mock {api_name} API client.\"\"\"",
            f"        assert mock_{api_var}_client is not None",
            "",
        ])
        
        # Add API-specific mock tests
        if api_type == 'openai':
            test_class.extend([
                f"        # Test mock completion",
                f"        response = mock_{api_var}_client.chat.completions.create(",
                f"            model=\"gpt-3.5-turbo\",",
                f"            messages=[{{'role': 'user', 'content': 'Hello'}}]",
                f"        )",
                f"        assert response.choices[0].message.content == \"Mock response\"",
                f"",
                f"        # Test mock embedding",
                f"        embed_response = mock_{api_var}_client.embeddings.create(",
                f"            model=\"text-embedding-ada-002\",",
                f"            input=\"Test input\"",
                f"        )",
                f"        assert embed_response.data[0].embedding == [0.1, 0.2, 0.3]",
                f""
            ])
        elif api_type == 'hf_tei' or api_type == 'hf_tgi':
            test_class.extend([
                f"        # Test mock generation",
                f"        response = mock_{api_var}_client.generate(\"Test input\")",
                f"        assert response[\"generated_text\"] == \"Mock generated text\"",
                f""
            ])
        elif api_type == 'ollama':
            test_class.extend([
                f"        # Test mock generation",
                f"        response = mock_{api_var}_client.generate(\"llama2\", \"Test input\")",
                f"        assert response[\"response\"] == \"Mock generated text\"",
                f""
            ])
        elif api_type == 'claude':
            test_class.extend([
                f"        # Test mock message",
                f"        response = mock_{api_var}_client.messages.create(",
                f"            model=\"claude-3-sonnet-20240229\",",
                f"            messages=[{{'role': 'user', 'content': 'Hello'}}]",
                f"        )",
                f"        assert response.content[0].text == \"Mock response\"",
                f""
            ])
        else:  # internal or vllm
            test_class.extend([
                f"        # Test mock API calls",
                f"        response = mock_{api_var}_client.get(\"/test\")",
                f"        assert response.status_code == 200",
                f"        assert response.json()[\"result\"] == \"success\"",
                f"",
                f"        post_response = mock_{api_var}_client.post(\"/test\", {{\"data\": \"test\"}})",
                f"        assert post_response.status_code == 200",
                f""
            ])
        
        return "".join(f"    {line}\n" for line in test_class)
    
    def generate_content(self) -> str:
        """
        Generate the full content of the API test file.
        
        Returns:
            String with test file content
        """
        if not self.validate_parameters():
            raise ValueError("Invalid template parameters")
        
        api_name = self.parameters['api_name']
        api_type = self.parameters.get('api_type', 'internal')
        
        content = [
            '"""',
            f"Test file for {api_name} API.",
            "",
            f"This file contains tests for the {api_name} API,",
            f"including connection tests and API functionality tests.",
            "Generated from APITestTemplate.",
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
            api_name = self.parameters['api_name']
            api_type = self.parameters.get('api_type', 'internal')
            api_var = api_name.replace('-', '_').lower()
            test_name = self.parameters.get('test_name', f"{api_var}_api")
            
            # Determine directory based on API type
            if api_type in ['openai', 'claude']:
                dir_path = os.path.join(self.output_dir, "api", "llm_providers")
            elif api_type in ['hf_tei', 'hf_tgi']:
                dir_path = os.path.join(self.output_dir, "api", "huggingface")
            elif api_type in ['ollama', 'vllm']:
                dir_path = os.path.join(self.output_dir, "api", "local_servers")
            else:
                dir_path = os.path.join(self.output_dir, "api", "internal")
            
            os.makedirs(dir_path, exist_ok=True)
            
            file_path = os.path.join(dir_path, f"test_{test_name}.py")
        
        return super().write(file_path)