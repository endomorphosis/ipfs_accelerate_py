#!/usr/bin/env python
# Advanced template inheritance system for IPFS Accelerate
# Implements multi-template inheritance with validation and compatibility testing

import os
import sys
import json
import logging
import importlib
import inspect
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple, Union
import ast
from collections import defaultdict

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Local imports
try:
    from model_family_classifier import classify_model
except ImportError:
    logger.warning("model_family_classifier not available, some functionality will be limited")
    def classify_model(*args, **kwargs):
        return {"family": None, "confidence": 0}

try:
    from hardware_detection import detect_available_hardware
except ImportError:
    logger.warning("hardware_detection not available, hardware-aware template selection will be limited")
    def detect_available_hardware():
        return {"hardware": {"cpu": True}, "torch_device": "cpu"}

# Template registry to store loaded templates
_TEMPLATE_REGISTRY = {}
_TEMPLATE_INHERITANCE_MAP = {}
_TEMPLATE_VALIDATORS = {}
_TEMPLATE_CACHE = {}

# Extended template processing constants
MARKER_START = "# BEGIN TEMPLATE_SECTION: "
MARKER_END = "# END TEMPLATE_SECTION: "
OVERRIDE_MARKER = "# OVERRIDE"
EXTEND_MARKER = "# EXTEND"
BEFORE_MARKER = "# BEFORE"
AFTER_MARKER = "# AFTER"

class TemplateError(Exception):
    """Base class for template-related errors"""
    pass

class TemplateNotFoundError(TemplateError):
    """Raised when a template is not found"""
    pass

class TemplateValidationError(TemplateError):
    """Raised when template validation fails"""
    pass

class TemplateInheritanceError(TemplateError):
    """Raised when template inheritance issues occur"""
    pass

class TemplateCompatibilityError(TemplateError):
    """Raised when template compatibility issues are detected"""
    pass

class Template:
    """Represents a template with advanced inheritance capabilities"""
    
    def __init__(self, template_path: str, template_name: Optional[str] = None):
        """
        Initialize a template
        
        Args:
            template_path: Path to the template file
            template_name: Optional name for the template (defaults to filename)
        """
        self.path = template_path
        self.name = template_name or os.path.basename(template_path)
        self.content = None
        self.parent_templates = []
        self.sections = {}
        self.metadata = {}
        self.requires = set()
        self.provides = set()
        self.compatibility = {}
        self.modification_directives = {}
        self.specialized_sections = defaultdict(list)
        
        # Load the template
        self._load_template()
        
        # Register the template
        _TEMPLATE_REGISTRY[self.name] = self
    
    def _load_template(self):
        """Load template content and parse structure"""
        try:
            with open(self.path, 'r') as f:
                self.content = f.read()
            
            # Parse template structure and metadata
            self._parse_structure()
            
            logger.debug(f"Loaded template: {self.name}")
        except (IOError, UnicodeDecodeError) as e:
            logger.error(f"Error loading template {self.path}: {str(e)}")
            raise TemplateError(f"Failed to load template {self.path}: {str(e)}")
    
    def _parse_structure(self):
        """Parse template structure, including inheritance directives and sections"""
        if not self.content:
            return
        
        # Look for inheritance directives - we support both Jinja-style and custom directives
        for line in self.content.split('\n'):
            # Jinja-style inheritance
            if line.strip().startswith('{%') and 'extends' in line and '%}' in line:
                match = re.search(r'extends\s+[\'"](.*?)[\'"]', line)
                if match:
                    parent_template = match.group(1)
                    self.parent_templates.append(parent_template)
            
            # Custom inheritance directive using comments
            if line.strip().startswith('#') and 'inherits:' in line:
                match = re.search(r'inherits:\s+(.*?)$', line)
                if match:
                    parent_templates = [t.strip() for t in match.group(1).split(',')]
                    self.parent_templates.extend(parent_templates)
            
            # Multiple inheritance support
            if line.strip().startswith('#') and 'uses:' in line:
                match = re.search(r'uses:\s+(.*?)(?:\s+for\s+(.*?))?$', line)
                if match:
                    used_template = match.group(1).strip()
                    sections = [s.strip() for s in match.group(2).split(',')] if match.group(2) else []
                    
                    # Add as parent but track which sections to include
                    if used_template not in self.parent_templates:
                        self.parent_templates.append(used_template)
                        
                    # Record section usage
                    if sections:
                        self.modification_directives[used_template] = {
                            'type': 'selective',
                            'sections': sections
                        }
        
        # Parse metadata from comments
        self._parse_metadata()
        
        # Parse enhanced sections with markers and directives
        self._parse_enhanced_sections()
        
        # Update inheritance map
        for parent in self.parent_templates:
            if parent not in _TEMPLATE_INHERITANCE_MAP:
                _TEMPLATE_INHERITANCE_MAP[parent] = []
            if self.name not in _TEMPLATE_INHERITANCE_MAP[parent]:
                _TEMPLATE_INHERITANCE_MAP[parent].append(self.name)
    
    def _parse_metadata(self):
        """Parse template metadata from comments"""
        if not self.content:
            return
        
        # Extract metadata from special comment blocks
        metadata_section = re.search(r'"""TEMPLATE_METADATA(.*?)"""', self.content, re.DOTALL)
        if metadata_section:
            metadata_text = metadata_section.group(1).strip()
            # Process metadata entries
            for line in metadata_text.split('\n'):
                line = line.strip()
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip().lower()
                    value = value.strip()
                    
                    if key == 'requires':
                        self.requires.update(r.strip() for r in value.split(','))
                    elif key == 'provides':
                        self.provides.update(p.strip() for p in value.split(','))
                    elif key == 'compatible_with':
                        for item in value.split(','):
                            item = item.strip()
                            if item:
                                self.compatibility[item] = True
                    elif key == 'incompatible_with':
                        for item in value.split(','):
                            item = item.strip()
                            if item:
                                self.compatibility[item] = False
                    elif key == 'priority' and value.isdigit():
                        # Priority for template selection (higher is better)
                        self.metadata[key] = int(value)
                    elif key == 'hardware_requirements':
                        # Parse hardware requirements
                        hw_reqs = {}
                        for hw_spec in value.split(','):
                            hw_spec = hw_spec.strip()
                            if ':' in hw_spec:
                                hw_type, hw_value = hw_spec.split(':')
                                hw_reqs[hw_type.strip()] = hw_value.strip()
                        self.metadata['hardware_requirements'] = hw_reqs
                    else:
                        self.metadata[key] = value
    
    def _parse_enhanced_sections(self):
        """
        Parse template into sections with enhanced inheritance markers
        
        This handles BEGIN/END section markers and special directives
        like OVERRIDE, EXTEND, BEFORE, and AFTER
        """
        if not self.content:
            return
        
        # First pass: extract marked sections
        # Pattern for BEGIN TEMPLATE_SECTION: name to END TEMPLATE_SECTION: name
        section_pattern = re.compile(
            f"{MARKER_START}(\\w+)(.*?){MARKER_END}\\1",
            re.DOTALL
        )
        
        for match in section_pattern.finditer(self.content):
            section_name = match.group(1)
            section_content = match.group(2).strip()
            
            # Check for modification directives
            directive = None
            if section_content.startswith(OVERRIDE_MARKER):
                directive = "override"
                section_content = section_content[len(OVERRIDE_MARKER):].strip()
            elif section_content.startswith(EXTEND_MARKER):
                directive = "extend"
                section_content = section_content[len(EXTEND_MARKER):].strip()
            elif section_content.startswith(BEFORE_MARKER):
                directive = "before"
                section_content = section_content[len(BEFORE_MARKER):].strip()
            elif section_content.startswith(AFTER_MARKER):
                directive = "after"
                section_content = section_content[len(AFTER_MARKER):].strip()
            
            if directive:
                self.specialized_sections[section_name].append({
                    "content": section_content,
                    "directive": directive
                })
            else:
                self.sections[section_name] = section_content
        
        # Second pass: fallback to older SECTION: style markers
        # for compatibility with existing templates
        remaining_content = self.content
        for section_name in self.sections.keys():
            # Remove already processed sections from the content
            section_block = f"{MARKER_START}{section_name}.*?{MARKER_END}{section_name}"
            remaining_content = re.sub(section_block, "", remaining_content, flags=re.DOTALL)
        
        # Process remaining content with old style markers
        current_section = 'main'
        current_content = []
        
        for line in remaining_content.split('\n'):
            section_match = re.match(r'#\s*SECTION:\s*([\w_]+)', line)
            if section_match:
                # Save previous section
                if current_content and current_section not in self.sections:
                    self.sections[current_section] = '\n'.join(current_content)
                current_content = []
                
                # Start new section
                current_section = section_match.group(1)
            else:
                current_content.append(line)
        
        # Save the last section
        if current_content and current_section not in self.sections:
            self.sections[current_section] = '\n'.join(current_content)
        
        # If no sections were found at all, treat entire content as 'main' section
        if not self.sections and self.content:
            # Remove metadata block before treating as main
            content = re.sub(r'"""TEMPLATE_METADATA.*?"""', '', self.content, flags=re.DOTALL)
            self.sections['main'] = content.strip()
    
    def validate(self) -> Tuple[bool, List[str]]:
        """
        Validate the template
        
        Returns:
            Tuple of (valid, list of validation errors)
        """
        errors = []
        
        # Check for required sections
        required_sections = {'main', 'init', 'methods'}
        found_sections = set(self.sections.keys())
        # Include specialized sections that will be included in final output
        for section_name, specializations in self.specialized_sections.items():
            if any(s['directive'] in ('override', 'extend') for s in specializations):
                found_sections.add(section_name)
                
        missing_sections = required_sections - found_sections
        if missing_sections:
            errors.append(f"Missing required sections: {', '.join(missing_sections)}")
        
        # Check for invalid Python syntax in template
        for section_name, section_content in self.sections.items():
            try:
                # Try parsing as Python code
                if section_content.strip():
                    ast.parse(section_content)
            except SyntaxError as e:
                errors.append(f"Syntax error in section '{section_name}' at line {e.lineno}: {e.msg}")
        
        # Check specialized sections for syntax
        for section_name, specializations in self.specialized_sections.items():
            for i, spec in enumerate(specializations):
                try:
                    if spec['content'].strip():
                        ast.parse(spec['content'])
                except SyntaxError as e:
                    errors.append(f"Syntax error in specialized section '{section_name}' ({spec['directive']}) at line {e.lineno}: {e.msg}")
        
        # Check for parent template existence
        for parent in self.parent_templates:
            if parent not in _TEMPLATE_REGISTRY:
                errors.append(f"Parent template '{parent}' not found")
        
        # Check for circular inheritance
        if self._has_circular_inheritance():
            errors.append("Circular inheritance detected")
        
        # Run custom validators
        for validator_name, validator_func in _TEMPLATE_VALIDATORS.items():
            try:
                validator_result = validator_func(self)
                if not validator_result[0]:
                    errors.extend(validator_result[1])
            except Exception as e:
                errors.append(f"Validator '{validator_name}' failed: {str(e)}")
        
        return (len(errors) == 0, errors)
    
    def _has_circular_inheritance(self, visited=None) -> bool:
        """Check for circular inheritance"""
        if visited is None:
            visited = set()
        
        if self.name in visited:
            return True
        
        visited.add(self.name)
        
        for parent_name in self.parent_templates:
            parent = _TEMPLATE_REGISTRY.get(parent_name)
            if parent and parent._has_circular_inheritance(visited.copy()):
                return True
        
        return False
    
    def render(self, context: Dict[str, Any] = None) -> str:
        """
        Render the template with context
        
        Args:
            context: Dictionary of context variables
            
        Returns:
            Rendered template content
        """
        if context is None:
            context = {}
            
        # Check for cached result for this template and context
        cache_key = f"{self.name}:{hash(frozenset(context.items()))}"
        if cache_key in _TEMPLATE_CACHE:
            logger.debug(f"Using cached template: {cache_key}")
            return _TEMPLATE_CACHE[cache_key]
        
        # Build the final template by combining with parent templates
        rendered_sections = self._build_enhanced_inheritance_tree()
        
        # Combine sections into final content
        combined_content = '\n\n'.join(rendered_sections.values())
        
        # Simple template variable substitution
        if context:
            for key, value in context.items():
                combined_content = combined_content.replace(f"{{{{ {key} }}}}", str(value))
        
        # Add to cache
        _TEMPLATE_CACHE[cache_key] = combined_content
        
        return combined_content
        
    def _build_enhanced_inheritance_tree(self) -> Dict[str, str]:
        """
        Build the inheritance tree with support for specialized section directives
        
        Returns:
            Dictionary of section name -> combined content
        """
        # Get section content from parent templates first (depth-first traversal)
        inherited_sections = {}
        
        # Process all ancestors for proper multi-inheritance
        for parent_name in reversed(self.parent_templates):
            parent = _TEMPLATE_REGISTRY.get(parent_name)
            if not parent:
                logger.warning(f"Parent template '{parent_name}' not found")
                continue
            
            # Check if we only want specific sections from this parent
            selective_loading = parent_name in self.modification_directives and \
                               self.modification_directives[parent_name]['type'] == 'selective'
            
            # Get parent sections
            parent_sections = parent._build_enhanced_inheritance_tree()
            
            # Add sections from parent, respecting selective loading
            for section_name, section_content in parent_sections.items():
                if selective_loading:
                    selected_sections = self.modification_directives[parent_name]['sections']
                    if section_name not in selected_sections:
                        continue
                        
                inherited_sections[section_name] = section_content
        
        # Start with base sections inherited from parents
        result = dict(inherited_sections)
        
        # Apply specialized section modifications
        for section_name, specializations in self.specialized_sections.items():
            for spec in specializations:
                directive = spec['directive']
                content = spec['content']
                
                if directive == 'override':
                    # Replace the entire section
                    result[section_name] = content
                elif directive == 'extend':
                    # Add content to the end of the section
                    if section_name in result:
                        result[section_name] = f"{result[section_name]}\n\n{content}"
                    else:
                        result[section_name] = content
                elif directive == 'before':
                    # Add content to the beginning of the section
                    if section_name in result:
                        result[section_name] = f"{content}\n\n{result[section_name]}"
                    else:
                        result[section_name] = content
                elif directive == 'after':
                    # Same as extend but with explicit name
                    if section_name in result:
                        result[section_name] = f"{result[section_name]}\n\n{content}"
                    else:
                        result[section_name] = content
        
        # Add regular sections from this template (overrides inherited values)
        for section_name, section_content in self.sections.items():
            result[section_name] = section_content
        
        return result

def register_template(template_path: str, template_name: Optional[str] = None) -> Template:
    """
    Register a template with the system
    
    Args:
        template_path: Path to the template file
        template_name: Optional name for the template (defaults to filename)
        
    Returns:
        Template instance
    """
    template = Template(template_path, template_name)
    return template

def register_template_directory(directory_path: str) -> List[Template]:
    """
    Register all templates in a directory
    
    Args:
        directory_path: Path to directory containing templates
        
    Returns:
        List of registered templates
    """
    templates = []
    
    for file_path in Path(directory_path).glob('*.py'):
        # Skip files starting with underscore or double underscore
        if file_path.name.startswith('_'):
            continue
        
        # Skip non-Python files
        if not file_path.name.endswith('.py'):
            continue
        
        template = register_template(str(file_path))
        templates.append(template)
    
    return templates

def get_template(template_name: str) -> Template:
    """
    Get a registered template by name
    
    Args:
        template_name: Name of the template
        
    Returns:
        Template instance
    """
    if template_name not in _TEMPLATE_REGISTRY:
        raise TemplateNotFoundError(f"Template '{template_name}' not found")
    
    return _TEMPLATE_REGISTRY[template_name]

def get_template_for_model(model_name: str, 
                          model_requirements: Optional[Dict[str, Any]] = None,
                          hardware_compatibility: Optional[Dict[str, Any]] = None) -> Template:
    """
    Get the most appropriate template for a model
    
    Args:
        model_name: The model name
        model_requirements: Optional model requirements
        hardware_compatibility: Optional hardware compatibility information
        
    Returns:
        Template instance
    """
    # Get model family information
    model_info = classify_model(model_name, model_requirements)
    family = model_info.get('family')
    subfamily = model_info.get('subfamily')
    
    # Define template search order based on family and subfamily
    search_templates = []
    
    # 1. Try subfamily-specific template first
    if family and subfamily:
        search_templates.append(f"hf_{family}_{subfamily}_template.py")
    
    # 2. Try family-specific template
    if family:
        search_templates.append(f"hf_{family}_template.py")
    
    # 3. Try model-specific template
    model_slug = model_name.split('/')[-1].replace('-', '_').lower()
    search_templates.append(f"hf_{model_slug}_template.py")
    
    # 4. Fall back to base template
    search_templates.append("hf_template.py")
    
    # Find the first matching template
    for template_name in search_templates:
        if template_name in _TEMPLATE_REGISTRY:
            template = _TEMPLATE_REGISTRY[template_name]
            
            # If hardware compatibility is provided, check it against the template
            if hardware_compatibility and hasattr(template, 'compatibility'):
                for hw_type, compatible in hardware_compatibility.items():
                    # Skip if template doesn't care about this hardware
                    if hw_type not in template.compatibility:
                        continue
                    
                    # If template says it's incompatible with this hardware, skip this template
                    if not template.compatibility.get(hw_type, True) and compatible:
                        logger.debug(f"Template {template_name} is incompatible with {hw_type}")
                        continue
            
            return template
    
    # If no matching template is found, raise error
    raise TemplateNotFoundError(f"No suitable template found for model {model_name}")

def register_template_validator(validator_name: str, validator_func):
    """
    Register a validator function for templates
    
    Args:
        validator_name: Name of the validator
        validator_func: Function that takes a template and returns (valid, errors)
    """
    _TEMPLATE_VALIDATORS[validator_name] = validator_func

def validate_all_templates() -> Dict[str, Dict[str, Any]]:
    """
    Validate all registered templates
    
    Returns:
        Dictionary mapping template names to validation results
    """
    results = {}
    
    for template_name, template in _TEMPLATE_REGISTRY.items():
        valid, errors = template.validate()
        results[template_name] = {
            'valid': valid,
            'errors': errors
        }
    
    return results

def get_incompatible_templates(hardware_info: Dict[str, Any]) -> List[str]:
    """
    Get list of templates incompatible with the current hardware
    
    Args:
        hardware_info: Hardware information dict from hardware_detection
        
    Returns:
        List of incompatible template names
    """
    incompatible = []
    
    # Check each template for compatibility
    for template_name, template in _TEMPLATE_REGISTRY.items():
        for hw_type, hw_available in hardware_info.get('hardware', {}).items():
            if hw_available and hw_type in template.compatibility:
                if not template.compatibility[hw_type]:
                    incompatible.append(template_name)
                    break
    
    return incompatible

def test_template_compatibility() -> Dict[str, Dict[str, Any]]:
    """
    Test compatibility between templates and current hardware
    
    Returns:
        Dictionary of compatibility test results
    """
    # Get hardware information
    hw_info = detect_available_hardware()
    
    results = {}
    incompatible_templates = get_incompatible_templates(hw_info)
    
    for template_name, template in _TEMPLATE_REGISTRY.items():
        compatible = template_name not in incompatible_templates
        compatibility_details = {}
        
        # Check compatibility with each hardware type
        for hw_type, hw_available in hw_info.get('hardware', {}).items():
            if hw_type in template.compatibility:
                template_compatible = template.compatibility[hw_type]
                compatibility_details[hw_type] = {
                    'template_compatible': template_compatible,
                    'hardware_available': hw_available,
                    'compatible': template_compatible or not hw_available
                }
        
        results[template_name] = {
            'compatible': compatible,
            'details': compatibility_details
        }
    
    return results

def get_template_inheritance_graph() -> Dict[str, List[str]]:
    """
    Get the template inheritance graph
    
    Returns:
        Dictionary mapping template names to their children
    """
    return _TEMPLATE_INHERITANCE_MAP

def create_specialized_template(base_template_name: str, 
                              specialized_name: str, 
                              overrides: Dict[str, str],
                              use_enhanced_sections: bool = True) -> Template:
    """
    Create a specialized template by inheriting from a base template and overriding sections
    
    Args:
        base_template_name: Name of the base template
        specialized_name: Name for the specialized template
        overrides: Dictionary mapping section names to override content
        use_enhanced_sections: Whether to use enhanced section markers
        
    Returns:
        New specialized template
    """
    # Get the base template
    base_template = get_template(base_template_name)
    
    # Create content for new template with inheritance directive
    content = f"""# Specialized template inheriting from {base_template_name}
# inherits: {base_template_name}

"""
    
    # Add metadata
    content += f'''"""TEMPLATE_METADATA
name: {specialized_name}
base: {base_template_name}
specialized: True
"""
'''
    
    # Add section overrides
    for section_name, section_content in overrides.items():
        if use_enhanced_sections:
            # Use enhanced section markers with OVERRIDE directive
            content += f"\n{MARKER_START}{section_name}\n{OVERRIDE_MARKER}\n{section_content}\n{MARKER_END}{section_name}\n"
        else:
            # Use old-style section markers for backward compatibility
            content += f"\n# SECTION: {section_name}\n{section_content}\n"
    
    # Create temporary file for the template
    import tempfile
    fd, temp_path = tempfile.mkstemp(suffix='.py')
    
    try:
        with os.fdopen(fd, 'w') as f:
            f.write(content)
        
        # Register the new template
        template = register_template(temp_path, specialized_name)
        
        # Validate the template
        valid, errors = template.validate()
        if not valid:
            logger.error(f"Failed to create specialized template: {errors}")
            raise TemplateValidationError(f"Failed to create specialized template: {errors}")
        
        return template
    except Exception as e:
        # Clean up temporary file on error
        os.unlink(temp_path)
        raise e

def create_multi_template(templates: List[str], 
                         output_name: str,
                         section_overrides: Optional[Dict[str, str]] = None,
                         metadata: Optional[Dict[str, Any]] = None) -> Template:
    """
    Create a template that inherits from multiple parent templates
    
    Args:
        templates: List of template names to inherit from
        output_name: Name for the new template
        section_overrides: Optional dict of section overrides
        metadata: Optional additional metadata
        
    Returns:
        New combined template
    """
    if not templates:
        raise ValueError("At least one template must be provided")
    
    # Start building template content
    parents_str = ", ".join(templates)
    content = f"""# Multi-template combining features from: {parents_str}
# inherits: {parents_str}

"""
    
    # Add metadata
    metadata_content = ""
    if metadata:
        for key, value in metadata.items():
            metadata_content += f"{key}: {value}\n"
    
    content += f'''"""TEMPLATE_METADATA
name: {output_name}
parents: {parents_str}
multi_template: True
{metadata_content}
"""
'''
    
    # Add section overrides if provided
    if section_overrides:
        for section_name, section_content in section_overrides.items():
            content += f"\n{MARKER_START}{section_name}\n{OVERRIDE_MARKER}\n{section_content}\n{MARKER_END}{section_name}\n"
    
    # Create temporary file for the template
    import tempfile
    fd, temp_path = tempfile.mkstemp(suffix='.py')
    
    try:
        with os.fdopen(fd, 'w') as f:
            f.write(content)
        
        # Register the new template
        template = register_template(temp_path, output_name)
        
        # Validate the template
        valid, errors = template.validate()
        if not valid:
            logger.error(f"Failed to create multi-template: {errors}")
            raise TemplateValidationError(f"Failed to create multi-template: {errors}")
        
        return template
    except Exception as e:
        # Clean up temporary file on error
        os.unlink(temp_path)
        raise e

def create_edge_case_template(base_template_name: str,
                             specialized_name: str,
                             edge_case_type: str,
                             additional_sections: Dict[str, str] = None) -> Template:
    """
    Create a specialized template for handling edge cases
    
    Args:
        base_template_name: Name of the base template
        specialized_name: Name for the specialized template
        edge_case_type: Type of edge case (e.g., 'low_memory', 'cpu_only', 'multi_gpu')
        additional_sections: Optional additional section overrides
        
    Returns:
        New edge case template
    """
    # Get the base template
    base_template = get_template(base_template_name)
    
    # Create content for new template with inheritance directive
    content = f"""# Edge case template for {edge_case_type} scenarios
# inherits: {base_template_name}

"""
    
    # Add metadata specific to edge cases
    content += f'''"""TEMPLATE_METADATA
name: {specialized_name}
base: {base_template_name}
edge_case: {edge_case_type}
specialized: True
"""
'''
    
    # Add standard edge case handlers based on type
    edge_case_sections = {}
    
    if edge_case_type == 'low_memory':
        edge_case_sections['init'] = """def __init__(self, model_name=None, device=None, **kwargs):
    # Initialize with low memory optimizations
    # Import required libraries
    import torch
    import os
    from resource_pool import get_global_resource_pool
    
    # Use resource pool
    self.pool = get_global_resource_pool()
    
    # Get device with low memory settings
    self.device = device or self.pool.get_best_device(low_memory=True)
    
    # Set model optimization flags
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    torch.backends.cuda.matmul.allow_tf32 = True
    
    # Load model with memory efficient settings
    self.model = self.pool.get_model(
        model_type=self.__class__.__name__.lower(),
        model_name=model_name,
        low_memory=True,
        device=self.device
    )
"""
    
    elif edge_case_type == 'cpu_only':
        edge_case_sections['init'] = """def __init__(self, model_name=None, **kwargs):
    # Initialize for CPU-only environments
    # Import required libraries
    import torch
    from resource_pool import get_global_resource_pool
    
    # Force CPU usage
    self.device = "cpu"
    
    # Use resource pool
    self.pool = get_global_resource_pool()
    
    # Configure for CPU optimization
    torch.set_num_threads(self.pool.get_optimal_thread_count())
    
    # Load model with CPU optimizations
    self.model = self.pool.get_model(
        model_type=self.__class__.__name__.lower(),
        model_name=model_name,
        device=self.device,
        optimize_for_cpu=True
    )
"""
    
    elif edge_case_type == 'multi_gpu':
        edge_case_sections['init'] = """def __init__(self, model_name=None, devices=None, **kwargs):
    # Initialize with multi-GPU support
    # Import required libraries
    import torch
    from resource_pool import get_global_resource_pool
    
    # Use resource pool
    self.pool = get_global_resource_pool()
    
    # Configure devices
    if devices is None:
        # Auto-detect available GPUs
        if torch.cuda.is_available():
            self.devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        else:
            self.devices = ["cpu"]
    else:
        self.devices = devices
    
    # Load model with DP/DDP support
    self.model = self.pool.get_model(
        model_type=self.__class__.__name__.lower(),
        model_name=model_name,
        devices=self.devices,
        multi_gpu=True
    )
"""
    
    # Add edge case sections
    for section_name, section_content in edge_case_sections.items():
        content += f"\n{MARKER_START}{section_name}\n{OVERRIDE_MARKER}\n{section_content}\n{MARKER_END}{section_name}\n"
    
    # Add any additional sections
    if additional_sections:
        for section_name, section_content in additional_sections.items():
            content += f"\n{MARKER_START}{section_name}\n{OVERRIDE_MARKER}\n{section_content}\n{MARKER_END}{section_name}\n"
    
    # Create temporary file for the template
    import tempfile
    fd, temp_path = tempfile.mkstemp(suffix='.py')
    
    try:
        with os.fdopen(fd, 'w') as f:
            f.write(content)
        
        # Register the new template
        template = register_template(temp_path, specialized_name)
        
        # Validate the template
        valid, errors = template.validate()
        if not valid:
            logger.error(f"Failed to create edge case template: {errors}")
            raise TemplateValidationError(f"Failed to create edge case template: {errors}")
        
        return template
    except Exception as e:
        # Clean up temporary file on error
        os.unlink(temp_path)
        raise e

def verify_template(template_name: str) -> Dict[str, Any]:
    """
    Perform comprehensive verification of a template
    
    Args:
        template_name: Name of the template to verify
        
    Returns:
        Dictionary with verification results
    """
    if template_name not in _TEMPLATE_REGISTRY:
        raise TemplateNotFoundError(f"Template '{template_name}' not found")
    
    template = _TEMPLATE_REGISTRY[template_name]
    
    # Basic validation
    valid, errors = template.validate()
    
    # Additional verification steps
    verification_results = {
        'valid': valid,
        'errors': errors,
        'warnings': [],
        'checks': {}
    }
    
    # Check for required and common sections
    common_sections = ['init', 'methods', 'imports', 'setup_class', 'teardown_class']
    missing_common = [section for section in common_sections if section not in template.sections]
    if missing_common:
        verification_results['warnings'].append(f"Missing common sections: {', '.join(missing_common)}")
    
    # Check template inheritance depth
    inheritance_depth = _get_inheritance_depth(template)
    verification_results['checks']['inheritance_depth'] = inheritance_depth
    if inheritance_depth > 3:
        verification_results['warnings'].append(f"Deep inheritance chain (depth {inheritance_depth}) may affect maintainability")
    
    # Check for ResourcePool usage
    resource_pool_used = False
    for section_content in template.sections.values():
        if 'resource_pool' in section_content or 'get_global_resource_pool' in section_content:
            resource_pool_used = True
            break
    
    verification_results['checks']['resource_pool_used'] = resource_pool_used
    if not resource_pool_used:
        verification_results['warnings'].append("Template does not appear to use ResourcePool")
    
    # Check for hardware awareness
    hardware_aware = False
    for section_content in template.sections.values():
        if 'device' in section_content.lower() and ('cuda' in section_content.lower() or 'cpu' in section_content.lower()):
            hardware_aware = True
            break
    
    verification_results['checks']['hardware_aware'] = hardware_aware
    if not hardware_aware:
        verification_results['warnings'].append("Template may not be hardware-aware")
    
    # Check for circular dependencies
    circular = template._has_circular_inheritance()
    verification_results['checks']['circular_inheritance'] = circular
    if circular:
        verification_results['errors'].append("Circular inheritance detected")
    
    # Return verification results
    return verification_results

def _get_inheritance_depth(template: Template, visited=None) -> int:
    """Get the maximum inheritance depth of a template"""
    if visited is None:
        visited = set()
    
    if template.name in visited:
        return 0  # Avoid cycles
    
    visited.add(template.name)
    
    if not template.parent_templates:
        return 1
    
    max_parent_depth = 0
    for parent_name in template.parent_templates:
        parent = _TEMPLATE_REGISTRY.get(parent_name)
        if parent:
            parent_depth = _get_inheritance_depth(parent, visited.copy())
            max_parent_depth = max(max_parent_depth, parent_depth)
    
    return max_parent_depth + 1

def verify_template_compatibility(template_a: str, template_b: str) -> Dict[str, Any]:
    """
    Check if two templates are compatible for composition
    
    Args:
        template_a: First template name
        template_b: Second template name
        
    Returns:
        Dictionary with compatibility results
    """
    if template_a not in _TEMPLATE_REGISTRY:
        raise TemplateNotFoundError(f"Template '{template_a}' not found")
    
    if template_b not in _TEMPLATE_REGISTRY:
        raise TemplateNotFoundError(f"Template '{template_b}' not found")
    
    template_a_obj = _TEMPLATE_REGISTRY[template_a]
    template_b_obj = _TEMPLATE_REGISTRY[template_b]
    
    # Check for compatibility
    results = {
        'compatible': True,
        'warnings': [],
        'issues': []
    }
    
    # Check for direct conflicts in sections
    common_sections = set(template_a_obj.sections.keys()) & set(template_b_obj.sections.keys())
    if common_sections:
        results['warnings'].append(f"Templates share common sections: {', '.join(common_sections)}")
    
    # Check for incompatible hardware requirements
    hw_a = set([hw for hw, comp in template_a_obj.compatibility.items() if comp])
    hw_b = set([hw for hw, comp in template_b_obj.compatibility.items() if comp])
    
    incompatible_hw_a = set([hw for hw, comp in template_a_obj.compatibility.items() if not comp])
    incompatible_hw_b = set([hw for hw, comp in template_b_obj.compatibility.items() if not comp])
    
    # If A requires hardware that B explicitly doesn't support
    conflicts = hw_a & incompatible_hw_b
    if conflicts:
        results['compatible'] = False
        results['issues'].append(f"Template {template_a} requires hardware that {template_b} does not support: {', '.join(conflicts)}")
    
    # If B requires hardware that A explicitly doesn't support
    conflicts = hw_b & incompatible_hw_a
    if conflicts:
        results['compatible'] = False
        results['issues'].append(f"Template {template_b} requires hardware that {template_a} does not support: {', '.join(conflicts)}")
    
    # Check for required dependencies
    requires_a = template_a_obj.requires
    requires_b = template_b_obj.requires
    
    provides_a = template_a_obj.provides
    provides_b = template_b_obj.provides
    
    # Check if A requires something B provides
    for req in requires_a:
        if req not in provides_b:
            results['warnings'].append(f"Template {template_a} requires '{req}' which {template_b} does not provide")
    
    # Check if B requires something A provides
    for req in requires_b:
        if req not in provides_a:
            results['warnings'].append(f"Template {template_b} requires '{req}' which {template_a} does not provide")
    
    return results

def validate_and_test_template(template_path: str) -> Dict[str, Any]:
    """
    Comprehensive validation and testing of a template file
    
    Args:
        template_path: Path to the template file
        
    Returns:
        Dictionary with validation and test results
    """
    # Register the template
    template = register_template(template_path)
    
    # Verify the template
    verification = verify_template(template.name)
    
    # Test against hardware compatibility
    hw_info = detect_available_hardware()
    compatibility = {}
    
    for hw_type, hw_available in hw_info.get('hardware', {}).items():
        if not hw_available:
            continue
            
        compatibility[hw_type] = True
        
        # Check if template is explicitly incompatible
        if hw_type in template.compatibility and not template.compatibility[hw_type]:
            compatibility[hw_type] = False
    
    # Test with renderer
    render_result = {"success": True, "errors": []}
    try:
        # Try to render the template
        rendered = template.render({"model_name": "test_model", "device": "cpu"})
        
        # Check if rendered template is valid Python
        try:
            ast.parse(rendered)
        except SyntaxError as e:
            render_result["success"] = False
            render_result["errors"].append(f"Syntax error in rendered template at line {e.lineno}: {e.msg}")
    except Exception as e:
        render_result["success"] = False
        render_result["errors"].append(f"Error rendering template: {str(e)}")
    
    # Combine results
    return {
        "template": template.name,
        "verification": verification,
        "hardware_compatibility": compatibility,
        "render_test": render_result,
        "overall_success": verification['valid'] and render_result["success"] and True in compatibility.values()
    }

if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Template Inheritance System")
    parser.add_argument("--template-dir", type=str, help="Directory containing templates")
    parser.add_argument("--validate", action="store_true", help="Validate all templates")
    parser.add_argument("--test-compatibility", action="store_true", help="Test hardware compatibility")
    parser.add_argument("--show-inheritance", action="store_true", help="Show template inheritance graph")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--test-template", type=str, help="Path to a specific template to test")
    parser.add_argument("--check-compatibility", type=str, nargs=2, help="Check compatibility between two templates")
    parser.add_argument("--create-edge-case", type=str, nargs=3, 
                      help="Create edge case template: <base_template> <output_name> <edge_case_type>")
    parser.add_argument("--clear-cache", action="store_true", help="Clear the template cache")
    
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    if args.clear_cache:
        _TEMPLATE_CACHE.clear()
        print("Template cache cleared.")
    
    if args.template_dir:
        templates = register_template_directory(args.template_dir)
        print(f"Registered {len(templates)} templates from {args.template_dir}")
    
    if args.test_template:
        results = validate_and_test_template(args.test_template)
        
        print(f"\nTest Results for {results['template']}:")
        print(f"Overall success: {'✅' if results['overall_success'] else '❌'}")
        
        print("\nVerification:")
        print(f"Valid: {'✅' if results['verification']['valid'] else '❌'}")
        
        if results['verification']['errors']:
            print("Errors:")
            for error in results['verification']['errors']:
                print(f"  - {error}")
        
        if results['verification']['warnings']:
            print("Warnings:")
            for warning in results['verification']['warnings']:
                print(f"  - {warning}")
        
        print("\nHardware Compatibility:")
        for hw_type, compatible in results['hardware_compatibility'].items():
            print(f"  - {hw_type}: {'✅' if compatible else '❌'}")
        
        print("\nRender Test:")
        print(f"Success: {'✅' if results['render_test']['success'] else '❌'}")
        if results['render_test']['errors']:
            print("Errors:")
            for error in results['render_test']['errors']:
                print(f"  - {error}")
    
    if args.check_compatibility and len(args.check_compatibility) == 2:
        template_a, template_b = args.check_compatibility
        try:
            results = verify_template_compatibility(template_a, template_b)
            
            print(f"\nCompatibility between {template_a} and {template_b}:")
            print(f"Compatible: {'✅' if results['compatible'] else '❌'}")
            
            if results['issues']:
                print("Issues:")
                for issue in results['issues']:
                    print(f"  - {issue}")
            
            if results['warnings']:
                print("Warnings:")
                for warning in results['warnings']:
                    print(f"  - {warning}")
                    
            if results['compatible'] and not results['warnings'] and not results['issues']:
                print("  Templates are fully compatible!")
        except TemplateNotFoundError as e:
            print(f"Error: {str(e)}")
    
    if args.create_edge_case and len(args.create_edge_case) == 3:
        base_template, output_name, edge_case_type = args.create_edge_case
        try:
            template = create_edge_case_template(base_template, output_name, edge_case_type)
            print(f"Created edge case template: {template.name} ({edge_case_type})")
            print(f"Template file: {template.path}")
        except Exception as e:
            print(f"Error creating edge case template: {str(e)}")
    
    if args.validate:
        results = validate_all_templates()
        valid_count = sum(1 for r in results.values() if r['valid'])
        invalid_count = len(results) - valid_count
        
        print(f"\nValidation Results: {valid_count} valid, {invalid_count} invalid")
        
        if invalid_count > 0:
            print("\nInvalid Templates:")
            for template_name, result in results.items():
                if not result['valid']:
                    print(f"  - {template_name}:")
                    for error in result['errors']:
                        print(f"    - {error}")
    
    if args.test_compatibility:
        results = test_template_compatibility()
        compatible_count = sum(1 for r in results.values() if r['compatible'])
        incompatible_count = len(results) - compatible_count
        
        print(f"\nCompatibility Results: {compatible_count} compatible, {incompatible_count} incompatible")
        
        if incompatible_count > 0:
            print("\nIncompatible Templates:")
            for template_name, result in results.items():
                if not result['compatible']:
                    print(f"  - {template_name}:")
                    for hw_type, details in result['details'].items():
                        if not details['compatible']:
                            print(f"    - Incompatible with {hw_type}")
    
    if args.show_inheritance:
        inheritance_graph = get_template_inheritance_graph()
        
        if inheritance_graph:
            print("\nTemplate Inheritance Graph:")
            for parent, children in inheritance_graph.items():
                print(f"  - {parent} is inherited by:")
                for child in children:
                    print(f"    - {child}")
        else:
            print("\nNo inheritance relationships found.")