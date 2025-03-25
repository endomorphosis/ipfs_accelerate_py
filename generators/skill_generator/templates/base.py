#!/usr/bin/env python3
"""
Template Base Class

This module provides the base class for all templates in the generator system.
"""

import logging
import datetime
import os
from typing import Dict, Any, Optional

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    import jinja2
    HAS_JINJA2 = True
except ImportError:
    HAS_JINJA2 = False
    logger.warning("Jinja2 not installed. Using simple string replacement for templates.")

class TemplateBase:
    """
    Base class for all templates in the generator system.
    
    This class provides the foundation for template rendering with common functionality.
    Templates can be implemented using Jinja2 (if available) or simple string replacement.
    """
    
    def __init__(self, config=None):
        """
        Initialize the template.
        
        Args:
            config: Configuration for the template
        """
        self.config = config or {}
        self.use_jinja = HAS_JINJA2 and self.config.get("use_jinja", True)
        self.env = None
        
        # Set up Jinja2 environment if available
        if self.use_jinja:
            self.env = jinja2.Environment(
                loader=jinja2.BaseLoader(),
                trim_blocks=True,
                lstrip_blocks=True,
                keep_trailing_newline=True
            )
            
            # Add custom filters
            self.env.filters["capitalize"] = lambda x: x.capitalize()
            self.env.filters["upper"] = lambda x: x.upper()
            self.env.filters["lower"] = lambda x: x.lower()
            self.env.filters["camel"] = lambda x: "".join(word.capitalize() for word in x.split("_"))
    
    def render(self, context: Dict[str, Any]) -> str:
        """
        Render the template with the provided context.
        
        Args:
            context: The context for template rendering
            
        Returns:
            The rendered template as a string
        """
        template_str = self.get_template_str()
        
        # Add standard context variables
        context = self._enhance_context(context)
        
        try:
            # Render using Jinja2 if available
            if self.use_jinja:
                template = self.env.from_string(template_str)
                rendered = template.render(**context)
            else:
                # Simple string replacement
                rendered = self._simple_render(template_str, context)
                
            logger.info("Template rendered successfully")
            return rendered
        except Exception as e:
            logger.error(f"Error rendering template: {str(e)}")
            raise
    
    def get_template_str(self) -> str:
        """
        Get the template string.
        
        Returns:
            The template as a string
        """
        raise NotImplementedError("Subclasses must implement get_template_str()")
    
    def get_imports(self) -> list:
        """
        Get the imports required by this template.
        
        Returns:
            List of import statements
        """
        return [
            "import os",
            "import sys",
            "import json",
            "import time",
            "import datetime",
            "import logging",
            "import argparse",
            "from unittest.mock import patch, MagicMock, Mock",
            "from typing import Dict, List, Any, Optional, Union",
            "from pathlib import Path"
        ]
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about this template.
        
        Returns:
            Dictionary of metadata
        """
        return {
            "name": self.__class__.__name__,
            "version": "1.0.0",
            "description": "Base template class",
            "supported_architectures": [],
            "author": "Generator System",
            "created_at": datetime.datetime.now().isoformat()
        }
    
    def _enhance_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance the context with standard variables.
        
        Args:
            context: The original context
            
        Returns:
            Enhanced context with additional variables
        """
        enhanced = context.copy()
        
        # Add timestamp if not present
        if "timestamp" not in enhanced:
            enhanced["timestamp"] = datetime.datetime.now().isoformat()
            
        # Add imports if not present
        if "imports" not in enhanced:
            enhanced["imports"] = self.get_imports()
            
        # Add generator info
        enhanced["generator"] = {
            "name": "Refactored Generator Suite",
            "version": "0.1.0",
            "template": self.__class__.__name__
        }
        
        # Add environment info
        enhanced["environment"] = {
            "python_version": sys.version,
            "platform": sys.platform,
            "user": os.environ.get("USER", "unknown")
        }
        
        return enhanced
    
    def _simple_render(self, template_str: str, context: Dict[str, Any]) -> str:
        """
        Simple string replacement for templates when Jinja2 is not available.
        
        Args:
            template_str: The template string
            context: The context for rendering
            
        Returns:
            The rendered string
        """
        rendered = template_str
        
        # Simple variable replacement
        for key, value in context.items():
            if isinstance(value, (str, int, float, bool)):
                placeholder = f"{{{{ {key} }}}}"
                rendered = rendered.replace(placeholder, str(value))
                
                # Handle filters
                rendered = rendered.replace(f"{{{{ {key}|capitalize }}}}", str(value).capitalize())
                rendered = rendered.replace(f"{{{{ {key}|upper }}}}", str(value).upper())
                rendered = rendered.replace(f"{{{{ {key}|lower }}}}", str(value).lower())
        
        # Simple conditional blocks (very limited support)
        for key, value in context.items():
            if isinstance(value, bool):
                if value:
                    # Remove conditional markers for true condition
                    start_marker = f"{{% if {key} %}}"
                    end_marker = f"{{% endif %}}"
                    rendered = rendered.replace(start_marker, "")
                    rendered = rendered.replace(end_marker, "")
                else:
                    # Remove blocks for false condition
                    start_marker = f"{{% if {key} %}}"
                    end_marker = f"{{% endif %}}"
                    
                    while start_marker in rendered and end_marker in rendered:
                        start_idx = rendered.find(start_marker)
                        end_idx = rendered.find(end_marker) + len(end_marker)
                        
                        if start_idx < end_idx:
                            rendered = rendered[:start_idx] + rendered[end_idx:]
                        else:
                            break
        
        return rendered