"""
Workflow Management Tools for IPFS Accelerate MCP Server

This module provides MCP tools for creating, managing, and executing workflows.
"""

import logging
import traceback
from typing import Dict, List, Any, Optional

logger = logging.getLogger("ipfs_accelerate_mcp.tools.workflows")

# Import workflow manager
try:
    from ipfs_accelerate_py.workflow_manager import WorkflowManager, Workflow, WorkflowTask
    HAVE_WORKFLOW_MANAGER = True
except ImportError as e:
    logger.warning(f"Workflow manager not available: {e}")
    HAVE_WORKFLOW_MANAGER = False
    WorkflowManager = None

# Global workflow manager instance
_workflow_manager = None


def get_workflow_manager():
    """Get or create the global workflow manager instance"""
    global _workflow_manager
    if _workflow_manager is None and HAVE_WORKFLOW_MANAGER:
        _workflow_manager = WorkflowManager()
    return _workflow_manager


def register_tools(mcp):
    """Register workflow management tools with the MCP server"""
    
    @mcp.tool()
    def create_workflow(
        name: str,
        description: str,
        tasks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Create a new workflow
        
        Args:
            name: Workflow name
            description: Workflow description
            tasks: List of task definitions. Each task should have:
                - name: Task name
                - type: Task type ('inference', 'processing', 'custom')
                - config: Task configuration dict
                - dependencies: Optional list of task indices that must complete first
        
        Returns:
            Dictionary with workflow details
        """
        try:
            manager = get_workflow_manager()
            if not manager:
                return {
                    'status': 'error',
                    'error': 'Workflow manager not available'
                }
            
            workflow = manager.create_workflow(name, description, tasks)
            
            return {
                'status': 'success',
                'workflow_id': workflow.workflow_id,
                'name': workflow.name,
                'description': workflow.description,
                'task_count': len(workflow.tasks),
                'created_at': workflow.created_at
            }
        
        except Exception as e:
            logger.error(f"Error creating workflow: {e}")
            logger.debug(traceback.format_exc())
            return {
                'status': 'error',
                'error': str(e)
            }
    
    @mcp.tool()
    def list_workflows(status: Optional[str] = None) -> Dict[str, Any]:
        """
        List all workflows, optionally filtered by status
        
        Args:
            status: Optional status filter ('pending', 'running', 'paused', 'completed', 'failed', 'stopped')
        
        Returns:
            Dictionary with list of workflows
        """
        try:
            manager = get_workflow_manager()
            if not manager:
                return {
                    'status': 'error',
                    'error': 'Workflow manager not available'
                }
            
            workflows = manager.list_workflows(status=status)
            
            workflow_list = []
            for wf in workflows:
                progress = wf.get_progress()
                workflow_list.append({
                    'workflow_id': wf.workflow_id,
                    'name': wf.name,
                    'description': wf.description,
                    'status': wf.status,
                    'created_at': wf.created_at,
                    'started_at': wf.started_at,
                    'completed_at': wf.completed_at,
                    'progress': progress,
                    'task_count': len(wf.tasks),
                    'error': wf.error
                })
            
            return {
                'status': 'success',
                'workflows': workflow_list,
                'total': len(workflow_list)
            }
        
        except Exception as e:
            logger.error(f"Error listing workflows: {e}")
            logger.debug(traceback.format_exc())
            return {
                'status': 'error',
                'error': str(e)
            }
    
    @mcp.tool()
    def get_workflow(workflow_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a workflow
        
        Args:
            workflow_id: The workflow ID
        
        Returns:
            Dictionary with workflow details including tasks
        """
        try:
            manager = get_workflow_manager()
            if not manager:
                return {
                    'status': 'error',
                    'error': 'Workflow manager not available'
                }
            
            workflow = manager.get_workflow(workflow_id)
            if not workflow:
                return {
                    'status': 'error',
                    'error': f'Workflow {workflow_id} not found'
                }
            
            progress = workflow.get_progress()
            
            tasks_data = []
            for task in workflow.tasks:
                tasks_data.append({
                    'task_id': task.task_id,
                    'name': task.name,
                    'type': task.type,
                    'status': task.status,
                    'config': task.config,
                    'result': task.result,
                    'error': task.error,
                    'started_at': task.started_at,
                    'completed_at': task.completed_at,
                    'dependencies': task.dependencies
                })
            
            return {
                'status': 'success',
                'workflow': {
                    'workflow_id': workflow.workflow_id,
                    'name': workflow.name,
                    'description': workflow.description,
                    'status': workflow.status,
                    'created_at': workflow.created_at,
                    'started_at': workflow.started_at,
                    'completed_at': workflow.completed_at,
                    'error': workflow.error,
                    'progress': progress,
                    'tasks': tasks_data,
                    'metadata': workflow.metadata
                }
            }
        
        except Exception as e:
            logger.error(f"Error getting workflow: {e}")
            logger.debug(traceback.format_exc())
            return {
                'status': 'error',
                'error': str(e)
            }
    
    @mcp.tool()
    def start_workflow(workflow_id: str) -> Dict[str, Any]:
        """
        Start executing a workflow
        
        Args:
            workflow_id: The workflow ID to start
        
        Returns:
            Dictionary with operation result
        """
        try:
            manager = get_workflow_manager()
            if not manager:
                return {
                    'status': 'error',
                    'error': 'Workflow manager not available'
                }
            
            manager.start_workflow(workflow_id)
            
            return {
                'status': 'success',
                'workflow_id': workflow_id,
                'message': 'Workflow started successfully'
            }
        
        except Exception as e:
            logger.error(f"Error starting workflow: {e}")
            logger.debug(traceback.format_exc())
            return {
                'status': 'error',
                'error': str(e)
            }
    
    @mcp.tool()
    def pause_workflow(workflow_id: str) -> Dict[str, Any]:
        """
        Pause a running workflow
        
        Args:
            workflow_id: The workflow ID to pause
        
        Returns:
            Dictionary with operation result
        """
        try:
            manager = get_workflow_manager()
            if not manager:
                return {
                    'status': 'error',
                    'error': 'Workflow manager not available'
                }
            
            manager.pause_workflow(workflow_id)
            
            return {
                'status': 'success',
                'workflow_id': workflow_id,
                'message': 'Workflow paused successfully'
            }
        
        except Exception as e:
            logger.error(f"Error pausing workflow: {e}")
            logger.debug(traceback.format_exc())
            return {
                'status': 'error',
                'error': str(e)
            }
    
    @mcp.tool()
    def stop_workflow(workflow_id: str) -> Dict[str, Any]:
        """
        Stop a workflow
        
        Args:
            workflow_id: The workflow ID to stop
        
        Returns:
            Dictionary with operation result
        """
        try:
            manager = get_workflow_manager()
            if not manager:
                return {
                    'status': 'error',
                    'error': 'Workflow manager not available'
                }
            
            manager.stop_workflow(workflow_id)
            
            return {
                'status': 'success',
                'workflow_id': workflow_id,
                'message': 'Workflow stopped successfully'
            }
        
        except Exception as e:
            logger.error(f"Error stopping workflow: {e}")
            logger.debug(traceback.format_exc())
            return {
                'status': 'error',
                'error': str(e)
            }
    
    @mcp.tool()
    def delete_workflow(workflow_id: str) -> Dict[str, Any]:
        """
        Delete a workflow
        
        Args:
            workflow_id: The workflow ID to delete
        
        Returns:
            Dictionary with operation result
        """
        try:
            manager = get_workflow_manager()
            if not manager:
                return {
                    'status': 'error',
                    'error': 'Workflow manager not available'
                }
            
            manager.delete_workflow(workflow_id)
            
            return {
                'status': 'success',
                'workflow_id': workflow_id,
                'message': 'Workflow deleted successfully'
            }
        
        except Exception as e:
            logger.error(f"Error deleting workflow: {e}")
            logger.debug(traceback.format_exc())
            return {
                'status': 'error',
                'error': str(e)
            }
    
    @mcp.tool()
    def get_workflow_templates() -> Dict[str, Any]:
        """
        Get pre-built workflow templates for common AI pipelines
        
        Returns:
            Dictionary with available templates
        """
        try:
            templates = {
                'image_generation': {
                    'name': 'Image Generation Pipeline',
                    'description': 'LLM prompt enhancement → image generation → upscaling',
                    'use_case': 'Create high-quality images with enhanced prompts',
                    'models': ['gpt-4', 'stable-diffusion-xl', 'real-esrgan']
                },
                'video_generation': {
                    'name': 'Text-to-Video Pipeline',
                    'description': 'Enhanced prompt → image → animated video',
                    'use_case': 'Generate videos from text descriptions',
                    'models': ['gpt-4', 'stable-diffusion-xl', 'animatediff']
                },
                'safe_image': {
                    'name': 'Safe Image Generation',
                    'description': 'NSFW filter → image generation → quality validation',
                    'use_case': 'Generate safe, high-quality images with content filtering',
                    'models': ['nsfw-text-classifier', 'stable-diffusion', 'image-quality-scorer']
                },
                'multimodal': {
                    'name': 'Multimodal Content Pipeline',
                    'description': 'Text → Image → Audio → Video generation',
                    'use_case': 'Create complete multimedia content from text',
                    'models': ['gpt-4', 'dalle-3', 'tts-1', 'video-composer']
                }
            }
            
            return {
                'status': 'success',
                'templates': templates,
                'total': len(templates)
            }
        
        except Exception as e:
            logger.error(f"Error getting templates: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    @mcp.tool()
    def create_workflow_from_template(
        template_name: str,
        custom_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a workflow from a pre-built template
        
        Args:
            template_name: Name of the template ('image_generation', 'video_generation', 'safe_image', 'multimodal')
            custom_config: Optional custom configuration to override template defaults
        
        Returns:
            Dictionary with created workflow details
        """
        try:
            manager = get_workflow_manager()
            if not manager:
                return {
                    'status': 'error',
                    'error': 'Workflow manager not available'
                }
            
            # Get template
            template_map = {
                'image_generation': WorkflowManager.create_image_generation_pipeline,
                'video_generation': WorkflowManager.create_video_generation_pipeline,
                'safe_image': WorkflowManager.create_safe_image_pipeline,
                'multimodal': WorkflowManager.create_multimodal_pipeline
            }
            
            if template_name not in template_map:
                return {
                    'status': 'error',
                    'error': f'Unknown template: {template_name}. Available: {list(template_map.keys())}'
                }
            
            template = template_map[template_name]()
            
            # Apply custom config if provided
            if custom_config:
                if 'name' in custom_config:
                    template['name'] = custom_config['name']
                if 'description' in custom_config:
                    template['description'] = custom_config['description']
                # Can extend to support task-level customization
            
            # Create workflow from template
            workflow = manager.create_workflow(
                name=template['name'],
                description=template['description'],
                tasks=template['tasks']
            )
            
            return {
                'status': 'success',
                'workflow_id': workflow.workflow_id,
                'name': workflow.name,
                'description': workflow.description,
                'task_count': len(workflow.tasks),
                'template_used': template_name
            }
        
        except Exception as e:
            logger.error(f"Error creating workflow from template: {e}")
            logger.debug(traceback.format_exc())
            return {
                'status': 'error',
                'error': str(e)
            }
    
    logger.info("Workflow management tools registered successfully (9 tools including templates)")

