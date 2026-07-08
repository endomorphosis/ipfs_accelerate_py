"""
Skill Registry - Automatic discovery and registration of HF skills
"""

import os
import glob
import importlib.util
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path


logger = logging.getLogger(__name__)


@dataclass
class SkillInfo:
    """Information about a discovered HF skill"""
    name: str
    path: str
    model_id: str
    architecture: str
    task_type: str
    supported_hardware: List[str] = field(default_factory=list)
    module: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class SkillRegistry:
    """
    Registry for HF skills with automatic discovery
    
    Features:
    - Scan directories for hf_*.py files
    - Load and validate skills
    - Track supported models and hardware
    - Provide search and lookup capabilities
    """
    
    def __init__(self, skill_directories: List[str], skill_pattern: str = "hf_*.py"):
        self.skill_directories = skill_directories
        self.skill_pattern = skill_pattern
        self.skills: Dict[str, SkillInfo] = {}
        self.model_to_skill: Dict[str, str] = {}
        
    async def discover_skills(self) -> int:
        """
        Discover skills in configured directories
        
        Returns:
            Number of skills discovered
        """
        logger.info(f"Discovering skills in {len(self.skill_directories)} directories")
        discovered_count = 0
        
        for directory in self.skill_directories:
            if not os.path.exists(directory):
                logger.warning(f"Directory not found: {directory}")
                continue
                
            # Find all hf_*.py files
            pattern = os.path.join(directory, "**", self.skill_pattern)
            skill_files = glob.glob(pattern, recursive=True)
            
            logger.info(f"Found {len(skill_files)} potential skills in {directory}")
            
            for skill_file in skill_files:
                try:
                    skill_info = await self._load_skill(skill_file)
                    if skill_info:
                        self.skills[skill_info.name] = skill_info
                        self.model_to_skill[skill_info.model_id] = skill_info.name
                        discovered_count += 1
                        logger.debug(f"Registered skill: {skill_info.name} for model {skill_info.model_id}")
                except Exception as e:
                    logger.error(f"Error loading skill {skill_file}: {e}")
        
        logger.info(f"Discovered {discovered_count} skills total")
        return discovered_count
    
    async def _load_skill(self, skill_path: str) -> Optional[SkillInfo]:
        """Load a single skill file and extract metadata"""
        try:
            # Extract skill name from filename
            skill_name = Path(skill_path).stem
            
            # Load the module
            spec = importlib.util.spec_from_file_location(skill_name, skill_path)
            if not spec or not spec.loader:
                logger.warning(f"Could not load spec for {skill_path}")
                return None
                
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Extract metadata from the skill
            # Look for common patterns in skill files
            model_id = self._extract_model_id(module, skill_name)
            architecture = self._extract_architecture(module)
            task_type = self._extract_task_type(module)
            supported_hardware = self._extract_supported_hardware(module)
            
            return SkillInfo(
                name=skill_name,
                path=skill_path,
                model_id=model_id,
                architecture=architecture,
                task_type=task_type,
                supported_hardware=supported_hardware,
                module=module,
                metadata={}
            )
        except Exception as e:
            logger.error(f"Error loading skill from {skill_path}: {e}")
            return None
    
    def _extract_model_id(self, module: Any, skill_name: str) -> str:
        """Extract model ID from skill module"""
        # Try common attribute names
        for attr in ["MODEL_ID", "model_id", "model_name", "MODEL_NAME"]:
            if hasattr(module, attr):
                return getattr(module, attr)
        
        # Fallback: derive from skill name (e.g., hf_bert -> bert)
        if skill_name.startswith("hf_"):
            return skill_name[3:]
        return skill_name
    
    def _extract_architecture(self, module: Any) -> str:
        """Extract architecture type from skill module"""
        for attr in ["ARCHITECTURE", "architecture", "model_type"]:
            if hasattr(module, attr):
                return getattr(module, attr)
        return "unknown"
    
    def _extract_task_type(self, module: Any) -> str:
        """Extract task type from skill module"""
        for attr in ["TASK_TYPE", "task_type", "task"]:
            if hasattr(module, attr):
                return getattr(module, attr)
        return "unknown"
    
    def _extract_supported_hardware(self, module: Any) -> List[str]:
        """Extract supported hardware from skill module"""
        # Check for init methods that indicate hardware support
        supported = []
        for method_name in dir(module):
            if method_name.startswith("init_"):
                hardware = method_name[5:]  # Remove 'init_' prefix
                if hardware in ["cpu", "cuda", "rocm", "mps", "openvino", "apple", "qualcomm"]:
                    supported.append(hardware)
        
        return supported if supported else ["cpu"]  # Default to CPU if none found
    
    def get_skill(self, skill_name: str) -> Optional[SkillInfo]:
        """Get skill by name"""
        return self.skills.get(skill_name)
    
    def get_skill_for_model(self, model_id: str) -> Optional[SkillInfo]:
        """Get skill that handles a specific model"""
        skill_name = self.model_to_skill.get(model_id)
        if skill_name:
            return self.skills.get(skill_name)
        return None
    
    def list_skills(self) -> List[SkillInfo]:
        """List all registered skills"""
        return list(self.skills.values())
    
    def list_models(self) -> List[str]:
        """List all supported model IDs"""
        return list(self.model_to_skill.keys())
    
    def search_skills(
        self, 
        architecture: Optional[str] = None,
        task_type: Optional[str] = None,
        hardware: Optional[str] = None
    ) -> List[SkillInfo]:
        """Search skills by criteria"""
        results = []
        for skill in self.skills.values():
            if architecture and skill.architecture != architecture:
                continue
            if task_type and skill.task_type != task_type:
                continue
            if hardware and hardware not in skill.supported_hardware:
                continue
            results.append(skill)
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics"""
        return {
            "total_skills": len(self.skills),
            "total_models": len(self.model_to_skill),
            "architectures": len(set(s.architecture for s in self.skills.values())),
            "task_types": len(set(s.task_type for s in self.skills.values())),
        }
