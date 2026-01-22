
from .base_test import BaseTest
import requests
import os

class APITest(BaseTest):
    """Base class for API tests."""
    
    def setUp(self):
        super().setUp()
        self.base_url = os.environ.get("API_BASE_URL", "http://localhost:8000")
        self.session = requests.Session()
        
    def tearDown(self):
        self.session.close()
        super().tearDown()
    
    def get_endpoint_url(self, endpoint):
        """Get full URL for an endpoint."""
        return f"{self.base_url}/{endpoint.lstrip('/')}"
    
    def assertStatusCode(self, response, expected_code):
        """Assert that response has expected status code."""
        self.assertEqual(expected_code, response.status_code, 
                        f"Expected status code {expected_code}, got {response.status_code}")
