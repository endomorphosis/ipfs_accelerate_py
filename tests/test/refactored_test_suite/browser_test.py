
from .base_test import BaseTest
import os

class BrowserTest(BaseTest):
    """Base class for browser tests."""
    
    def setUp(self):
        super().setUp()
        self.browser_type = os.environ.get("BROWSER_TYPE", "chrome")
        
    def get_browser_driver(self):
        """Get browser driver for testing."""
        raise NotImplementedError("Subclasses must implement get_browser_driver")
