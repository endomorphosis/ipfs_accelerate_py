"""
Model Template Registry - Centralized registry for model templates with inheritance support.
This module provides a registry for all available templates and resolves template inheritance.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Set, Tuple
from pathlib import Path
import importlib.util
import inspect
import re

logger = logging.getLogger(__name__)

class ModelTemplateRegistry:
    """
    Registry for model templates with inheritance and hardware-aware template selection.
    
    This class manages all available templates, resolves template inheritance,
    and provides hardware-aware template selection based on model and hardware requirements.
    
    Attributes:
        template_dir: Directory containing template files
        templates: Dictionary of registered templates
        inheritance_graph: Dependency graph for template inheritance
        section_registry: Registry of template sections
    """
    
    def __init__(self, template_dir: Optional[str] = None):
        """
        Initialize the model template registry.
        
        Args:
            template_dir: Directory containing template files
        """
        self.template_dir = template_dir or os.path.join(os.path.dirname(__file__))
        self.templates = {}
        self.inheritance_graph = {}
        self.section_registry = {}
        
        # Load templates
        self._load_templates()
    
    def _load_templates(self):
        """Load templates from the template directory"""
        logger.info(f"Loading templates from {self.template_dir}")
        
        # Find all template files
        template_files = list(Path(self.template_dir).glob("hf_*.py"))
        
        for template_file in template_files:
            try:
                # Load template
                template_name = template_file.name
                template_info = self._load_template(template_file)
                
                if template_info:
                    self.templates[template_name] = template_info
                    logger.debug(f"Loaded template: {template_name}")
            except Exception as e:
                logger.error(f"Error loading template {template_file}: {str(e)}")
        
        # Build inheritance graph
        self._build_inheritance_graph()
        
        # Register template sections
        self._register_template_sections()
        
        logger.info(f"Loaded {len(self.templates)} templates")
    
    def _load_template(self, template_path: Path) -> Dict[str, Any]:
        """
        Load a template file and extract its metadata.
        
        Args:
            template_path: Path to the template file
            
        Returns:
            Dictionary with template information
        """
        try:
            # Load module
            module_name = template_path.stem
            spec = importlib.util.spec_from_file_location(module_name, template_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Extract metadata
            template_info = {
                "name": template_path.name,
                "path": str(template_path),
                "module": module,
                "inherits_from": getattr(module, "INHERITS_FROM", None),
                "supports_hardware": getattr(module, "SUPPORTS_HARDWARE", ["cpu"]),
                "compatible_families": getattr(module, "COMPATIBLE_FAMILIES", []),
                "sections": {},
                "model_requirements": getattr(module, "MODEL_REQUIREMENTS", {}),
                "version": getattr(module, "TEMPLATE_VERSION", "1.0.0"),
                "description": getattr(module, "TEMPLATE_DESCRIPTION", "")
            }
            
            # Extract sections
            for name, obj in inspect.getmembers(module):
                if name.startswith("SECTION_") and isinstance(obj, str):
                    section_name = name.replace("SECTION_", "").lower()
                    template_info["sections"][section_name] = obj
            
            return template_info
        except Exception as e:
            logger.error(f"Error loading template {template_path}: {str(e)}")
            return None
    
    def _build_inheritance_graph(self):
        """Build template inheritance graph"""
        # Initialize graph
        self.inheritance_graph = {template_name: [] for template_name in self.templates}
        
        # Add inheritance relationships
        for template_name, template_info in self.templates.items():
            parent = template_info.get("inherits_from")
            if parent and parent in self.templates:
                self.inheritance_graph[parent].append(template_name)
        
        logger.debug(f"Template inheritance graph: {self.inheritance_graph}")
    
    def _register_template_sections(self):
        """Register all template sections"""
        self.section_registry = {}
        
        for template_name, template_info in self.templates.items():
            for section_name, section_content in template_info["sections"].items():
                if section_name not in self.section_registry:
                    self.section_registry[section_name] = {}
                self.section_registry[section_name][template_name] = section_content
    
    def resolve_template_chain(self, template_name: str) -> List[str]:
        """
        Resolve the template inheritance chain for a template.
        
        Args:
            template_name: The template name
            
        Returns:
            List of template names in inheritance order (base first)
        """
        if template_name not in self.templates:
            return []
        
        chain = []
        current = template_name
        visited = set()
        
        # Walk up the inheritance chain
        while current and current not in visited:
            visited.add(current)
            
            # Add current to the beginning of the chain
            chain.insert(0, current)
            
            # Get parent
            current = self.templates.get(current, {}).get("inherits_from")
            
            # Stop if parent not found
            if current and current not in self.templates:
                logger.warning(f"Template {current} not found in registry")
                break
        
        return chain
    
    def get_merged_template(self, template_name: str) -> Dict[str, Any]:
        """
        Get a merged template with all inherited sections.
        
        Args:
            template_name: The template name
            
        Returns:
            Dictionary with merged template information
        """
        if template_name not in self.templates:
            logger.warning(f"Template {template_name} not found")
            return {}
        
        # Get inheritance chain
        chain = self.resolve_template_chain(template_name)
        if not chain:
            return {}
        
        # Start with base template
        merged = {
            "name": template_name,
            "inheritance_chain": chain,
            "sections": {},
            "supports_hardware": [],
            "compatible_families": [],
            "model_requirements": {}
        }
        
        # Merge sections from all templates in the chain
        for template in chain:
            template_info = self.templates.get(template, {})
            
            # Merge sections (child templates override parent sections)
            for section_name, section_content in template_info.get("sections", {}).items():
                merged["sections"][section_name] = section_content
            
            # Merge hardware support (union of all supported hardware)
            merged["supports_hardware"] = list(set(
                merged["supports_hardware"] + template_info.get("supports_hardware", [])
            ))
            
            # Merge compatible families (union of all compatible families)
            merged["compatible_families"] = list(set(
                merged["compatible_families"] + template_info.get("compatible_families", [])
            ))
            
            # Merge model requirements (child requirements override parent requirements)
            for req_name, req_value in template_info.get("model_requirements", {}).items():
                merged["model_requirements"][req_name] = req_value
        
        return merged
    
    def select_template_for_model(self, model_family: str, hardware_info: Optional[Dict[str, Any]] = None, 
                                 model_requirements: Optional[Dict[str, Any]] = None) -> str:
        """
        Select the most appropriate template for a model family and hardware.
        
        Args:
            model_family: The model family (e.g., 'embedding', 'text_generation')
            hardware_info: Optional hardware information
            model_requirements: Optional model requirements
            
        Returns:
            Name of the selected template
        """
        # Default to base template
        default_template = "hf_template.py"
        
        # First filter templates by model family
        compatible_templates = []
        
        for template_name, template_info in self.templates.items():
            families = template_info.get("compatible_families", [])
            
            # If no families specified, template is generic (compatible with all)
            if not families or model_family in families:
                compatible_templates.append(template_name)
        
        if not compatible_templates:
            logger.info(f"No templates found for model family {model_family}, using default")
            return default_template
        
        # If only one template is compatible, return it
        if len(compatible_templates) == 1:
            return compatible_templates[0]
        
        # Filter by hardware compatibility
        hardware_compatible = []
        if hardware_info:
            available_hardware = []
            
            # Extract available hardware types
            if hardware_info.get("cuda", False):
                available_hardware.append("cuda")
            if hardware_info.get("rocm", False):
                available_hardware.append("rocm")
            if hardware_info.get("mps", False):
                available_hardware.append("mps")
            if hardware_info.get("openvino", False):
                available_hardware.append("openvino")
            
            # CPU is always available
            available_hardware.append("cpu")
            
            # Filter templates by hardware compatibility
            for template_name in compatible_templates:
                template_hw = self.templates[template_name].get("supports_hardware", ["cpu"])
                
                # Check if template supports any available hardware
                if any(hw in available_hardware for hw in template_hw):
                    hardware_compatible.append(template_name)
        else:
            # No hardware info, all compatible templates are candidates
            hardware_compatible = compatible_templates
        
        if not hardware_compatible:
            logger.info(f"No hardware-compatible templates found, using default")
            return default_template
        
        # Score templates by:
        # 1. Specificity (more specific = higher score)
        # 2. Direct family match (vs generic compatibility)
        # 3. Hardware specialization
        template_scores = {}
        
        for template_name in hardware_compatible:
            template_info = self.templates[template_name]
            
            # Base score
            score = 1.0
            
            # Score for family specificity
            families = template_info.get("compatible_families", [])
            if families:
                if model_family in families:
                    # Direct match
                    score += 2.0
                    # More specific template (fewer families) gets higher score
                    score += 1.0 / max(1, len(families))
            
            # Score for hardware specificity
            hw_support = template_info.get("supports_hardware", ["cpu"])
            if hardware_info:
                available_hardware = []
                if hardware_info.get("cuda", False):
                    available_hardware.append("cuda")
                if hardware_info.get("rocm", False):
                    available_hardware.append("rocm")
                if hardware_info.get("mps", False):
                    available_hardware.append("mps")
                
                # Score for special hardware support
                special_hw = set(hw_support) - {"cpu"}
                matching_hw = special_hw.intersection(available_hardware)
                if matching_hw:
                    score += len(matching_hw) * 0.5
            
            # Score for model requirements match
            if model_requirements and template_info.get("model_requirements"):
                template_reqs = template_info.get("model_requirements", {})
                
                # Check each requirement
                for req_name, req_value in template_reqs.items():
                    if req_name in model_requirements:
                        model_value = model_requirements[req_name]
                        
                        # Check if requirement matches
                        if req_value == model_value:
                            score += 0.5
                        elif isinstance(req_value, (list, tuple)) and model_value in req_value:
                            score += 0.3
            
            template_scores[template_name] = score
        
        # Select highest scoring template
        if template_scores:
            best_template = max(template_scores.items(), key=lambda x: x[1])
            logger.info(f"Selected template {best_template[0]} with score {best_template[1]}")
            return best_template[0]
        
        # No template scored, use default
        return default_template
    
    def render_template(self, template_name: str, context: Dict[str, Any]) -> str:
        """
        Render a template with the given context.
        
        Args:
            template_name: The template name
            context: The template context (variables)
            
        Returns:
            Rendered template string
        """
        # Get merged template
        merged_template = self.get_merged_template(template_name)
        if not merged_template:
            logger.error(f"Template {template_name} not found or could not be merged")
            return ""
        
        # Render sections
        rendered = {}
        for section_name, section_content in merged_template.get("sections", {}).items():
            try:
                # Add section to rendered output
                rendered[section_name] = self._render_section(section_content, context)
            except Exception as e:
                logger.error(f"Error rendering section {section_name}: {str(e)}")
                rendered[section_name] = f"# Error rendering section {section_name}: {str(e)}"
        
        # Combine sections
        combined = []
        
        # Define section order
        section_order = [
            "imports", "class_definition", "init", "properties", 
            "methods", "utility_methods", "hardware_support", 
            "error_handling", "main"
        ]
        
        # Add sections in order
        for section in section_order:
            if section in rendered:
                combined.append(rendered[section])
        
        # Add any remaining sections
        for section, content in rendered.items():
            if section not in section_order:
                combined.append(content)
        
        return "\n\n".join(combined)
    
    def _render_section(self, section_content: str, context: Dict[str, Any]) -> str:
        """
        Render a section with the given context using simple variable substitution.
        
        Args:
            section_content: The section content template
            context: The template context (variables)
            
        Returns:
            Rendered section string
        """
        # Replace variables in the section
        result = section_content
        
        # Find all variables in the section using regex
        variables = re.findall(r'\{\{\s*([a-zA-Z0-9_\.]+)\s*\}\}', section_content)
        
        # Replace each variable
        for var in variables:
            # Split variable by dots to handle nested properties
            parts = var.split('.')
            
            # Get value from context
            value = context
            try:
                for part in parts:
                    value = value[part]
            except (KeyError, TypeError):
                # Variable not found in context
                logger.warning(f"Variable {var} not found in context")
                value = f"{{{{ {var} }}}}"  # Keep the variable in the template
            
            # Replace the variable in the section
            if value is not None:
                # Convert value to string if needed
                value_str = str(value)
                result = re.sub(r'\{\{\s*' + var + r'\s*\}\}', value_str, result)
        
        return result
    
    def get_all_template_names(self) -> List[str]:
        """Get all registered template names"""
        return list(self.templates.keys())
    
    def get_template_info(self, template_name: str) -> Dict[str, Any]:
        """
        Get information about a template.
        
        Args:
            template_name: The template name
            
        Returns:
            Dictionary with template information
        """
        if template_name not in self.templates:
            return {}
        
        template_info = self.templates[template_name].copy()
        
        # Remove module from info
        if "module" in template_info:
            del template_info["module"]
        
        return template_info

# Create a global instance for convenience
registry = ModelTemplateRegistry()

def get_template_registry() -> ModelTemplateRegistry:
    """Get the global template registry instance"""
    return registry

# Function to select a template for a model
def select_template(model_family: str, hardware_info: Optional[Dict[str, Any]] = None, 
                    model_requirements: Optional[Dict[str, Any]] = None) -> str:
    """
    Select the most appropriate template for a model family and hardware.
    
    Args:
        model_family: The model family (e.g., 'embedding', 'text_generation')
        hardware_info: Optional hardware information
        model_requirements: Optional model requirements
        
    Returns:
        Name of the selected template
    """
    return registry.select_template_for_model(model_family, hardware_info, model_requirements)

# Function to render a template
def render_template(template_name: str, context: Dict[str, Any]) -> str:
    """
    Render a template with the given context.
    
    Args:
        template_name: The template name
        context: The template context (variables)
        
    Returns:
        Rendered template string
    """
    return registry.render_template(template_name, context)