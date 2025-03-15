import os
import requests
import json
import logging
from typing import Dict, List, Any, Optional, Union

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sample_backend")

class sample_backend:
    """Sample API backend implementation for testing"""
    
    def __init__(self, resources: Dict[str, Any] = None, metadata: Dict[str, Any] = None):
        """Initialize the API backend"""
        self.resources = resources or {}
        self.metadata = metadata or {}
        self.api_endpoint = "https://api.sample.com/v1/chat"
        self.default_model = "sample-model"
        self.queue = []
        self.circuit_open = False
        self.error_count = 0
        
    def get_api_key(self, metadata: Dict[str, Any]) -> str:
        """Get API key from metadata or environment"""
        return metadata.get("sample_api_key") or os.environ.get("SAMPLE_API_KEY", "")
    
    def get_default_model(self) -> str:
        """Get the default model name"""
        return self.default_model
    
    def is_compatible_model(self, model: str) -> bool:
        """Check if the model is compatible with this backend"""
        return model.startswith("sample")
    
    def create_endpoint_handler(self) -> Callable:
        """Create an endpoint handler function"""
        def handler(data: Dict[str, Any]) -> Dict[str, Any]:
            return self.make_post_request(data)
        return handler
    
    def test_endpoint(self) -> bool:
        """Test the API endpoint"""
        try:
            api_key = self.get_api_key(self.metadata)
            if not api_key:
                logger.error("API key is required")
                return False
                
            test_request = {
                "model": self.get_default_model(),
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 5
            }
            
            response = self.make_post_request(test_request, api_key)
            return True
        except Exception as e:
            logger.error(f"Endpoint test failed: {e}")
            return False
    
    def make_post_request(self, data: Dict[str, Any], api_key: Optional[str] = None, 
                          timeout: int = 30) -> Dict[str, Any]:
        """Make a POST request to the API"""
        api_key = api_key or self.get_api_key(self.metadata)
        if not api_key:
            raise ValueError("API key is required")
            
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        response = requests.post(
            self.api_endpoint,
            headers=headers,
            json=data,
            timeout=timeout
        )
        
        if response.status_code != 200:
            error_data = response.json() if response.content else {}
            raise Exception(f"API request failed: {error_data}")
            
        return response.json()
    
    def make_stream_request(self, data: Dict[str, Any], api_key: Optional[str] = None,
                           timeout: int = 30) -> Iterator[Dict[str, Any]]:
        """Make a streaming request to the API"""
        api_key = api_key or self.get_api_key(self.metadata)
        if not api_key:
            raise ValueError("API key is required")
            
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "Accept": "text/event-stream"
        }
        
        # Ensure stream parameter is set
        data["stream"] = True
        
        response = requests.post(
            self.api_endpoint,
            headers=headers,
            json=data,
            timeout=timeout,
            stream=True
        )
        
        if response.status_code != 200:
            error_data = response.json() if response.content else {}
            raise Exception(f"API stream request failed: {error_data}")
            
        for line in response.iter_lines():
            if line:
                line_text = line.decode('utf-8')
                if line_text.startswith('data: '):
                    data = line_text[6:]
                    if data == '[DONE]':
                        break
                    try:
                        chunk = json.loads(data)
                        yield chunk
                    except json.JSONDecodeError:
                        logger.error(f"Failed to parse stream data: {data}")
    
    def chat(self, messages: List[Dict[str, str]], model: Optional[str] = None,
             max_tokens: Optional[int] = None, temperature: Optional[float] = None,
             top_p: Optional[float] = None) -> Dict[str, Any]:
        """Generate a response using the API"""
        model = model or self.get_default_model()
        
        request_data = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p
        }
        
        # Remove None values from request data
        request_data = {k: v for k, v in request_data.items() if v is not None}
        
        response = self.make_post_request(request_data)
        
        return {
            "id": response.get("id", ""),
            "model": response.get("model", model),
            "object": response.get("object", "chat.completion"),
            "created": response.get("created", 0),
            "choices": response.get("choices", []),
            "usage": response.get("usage", {})
        }
    
    def stream_chat(self, messages: List[Dict[str, str]], model: Optional[str] = None,
                    max_tokens: Optional[int] = None, temperature: Optional[float] = None,
                    top_p: Optional[float] = None) -> Iterator[Dict[str, Any]]:
        """Generate a streaming response using the API"""
        model = model or self.get_default_model()
        
        request_data = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": True
        }
        
        # Remove None values from request data
        request_data = {k: v for k, v in request_data.items() if v is not None}
        
        for chunk in self.make_stream_request(request_data):
            yield chunk