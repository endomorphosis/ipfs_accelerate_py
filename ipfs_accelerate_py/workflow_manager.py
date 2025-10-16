"""
Workflow Management System for IPFS Accelerate MCP Server

This module provides workflow definition, execution, and management capabilities.
"""

import json
import sqlite3
import time
import uuid
import logging
import asyncio
from typing import Dict, List, Any, Optional
from enum import Enum
from dataclasses import dataclass, asdict
from pathlib import Path

logger = logging.getLogger(__name__)


class WorkflowStatus(Enum):
    """Workflow execution status"""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


class TaskStatus(Enum):
    """Individual task status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class WorkflowTask:
    """Represents a single task in a workflow - designed for AI model pipelines"""
    task_id: str
    name: str
    type: str  # 'text_model', 'image_model', 'video_model', 'audio_model', 'filter_model', 'processing', 'custom'
    config: Dict[str, Any]  # Model-specific config (model_name, parameters, etc.)
    status: str = TaskStatus.PENDING.value
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    dependencies: List[str] = None  # List of task_ids that must complete first
    input_mapping: Optional[Dict[str, str]] = None  # Maps output keys from dependencies to input keys
    output_keys: Optional[List[str]] = None  # Keys to extract from result for downstream tasks

    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.input_mapping is None:
            self.input_mapping = {}
        if self.output_keys is None:
            self.output_keys = []


@dataclass
class Workflow:
    """Represents a complete workflow"""
    workflow_id: str
    name: str
    description: str
    tasks: List[WorkflowTask]
    status: str = WorkflowStatus.PENDING.value
    created_at: float = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()
        if self.metadata is None:
            self.metadata = {}

    def get_progress(self) -> Dict[str, int]:
        """Calculate workflow progress"""
        total = len(self.tasks)
        completed = sum(1 for t in self.tasks if t.status == TaskStatus.COMPLETED.value)
        running = sum(1 for t in self.tasks if t.status == TaskStatus.RUNNING.value)
        failed = sum(1 for t in self.tasks if t.status == TaskStatus.FAILED.value)
        
        return {
            'total': total,
            'completed': completed,
            'running': running,
            'failed': failed,
            'pending': total - completed - running - failed
        }


class WorkflowStorage:
    """Handles workflow persistence using SQLite"""
    
    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = str(Path.home() / ".ipfs_accelerate" / "workflows.db")
        
        self.db_path = db_path
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS workflows (
                    workflow_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    status TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    started_at REAL,
                    completed_at REAL,
                    error TEXT,
                    metadata TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    workflow_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    type TEXT NOT NULL,
                    config TEXT NOT NULL,
                    status TEXT NOT NULL,
                    result TEXT,
                    error TEXT,
                    started_at REAL,
                    completed_at REAL,
                    dependencies TEXT,
                    input_mapping TEXT,
                    output_keys TEXT,
                    FOREIGN KEY (workflow_id) REFERENCES workflows (workflow_id)
                )
            """)
            
            conn.commit()
    
    def save_workflow(self, workflow: Workflow):
        """Save or update a workflow"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO workflows 
                (workflow_id, name, description, status, created_at, started_at, completed_at, error, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                workflow.workflow_id,
                workflow.name,
                workflow.description,
                workflow.status,
                workflow.created_at,
                workflow.started_at,
                workflow.completed_at,
                workflow.error,
                json.dumps(workflow.metadata)
            ))
            
            # Save tasks
            for task in workflow.tasks:
                conn.execute("""
                    INSERT OR REPLACE INTO tasks
                    (task_id, workflow_id, name, type, config, status, result, error, 
                     started_at, completed_at, dependencies, input_mapping, output_keys)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    task.task_id,
                    workflow.workflow_id,
                    task.name,
                    task.type,
                    json.dumps(task.config),
                    task.status,
                    json.dumps(task.result) if task.result else None,
                    task.error,
                    task.started_at,
                    task.completed_at,
                    json.dumps(task.dependencies),
                    json.dumps(task.input_mapping),
                    json.dumps(task.output_keys)
                ))
            
            conn.commit()
    
    def load_workflow(self, workflow_id: str) -> Optional[Workflow]:
        """Load a workflow by ID"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            # Load workflow
            row = conn.execute(
                "SELECT * FROM workflows WHERE workflow_id = ?", 
                (workflow_id,)
            ).fetchone()
            
            if not row:
                return None
            
            # Load tasks
            task_rows = conn.execute(
                "SELECT * FROM tasks WHERE workflow_id = ? ORDER BY task_id",
                (workflow_id,)
            ).fetchall()
            
            tasks = []
            for task_row in task_rows:
                tasks.append(WorkflowTask(
                    task_id=task_row['task_id'],
                    name=task_row['name'],
                    type=task_row['type'],
                    config=json.loads(task_row['config']),
                    status=task_row['status'],
                    result=json.loads(task_row['result']) if task_row['result'] else None,
                    error=task_row['error'],
                    started_at=task_row['started_at'],
                    completed_at=task_row['completed_at'],
                    dependencies=json.loads(task_row['dependencies']),
                    input_mapping=json.loads(task_row['input_mapping']) if task_row.get('input_mapping') else {},
                    output_keys=json.loads(task_row['output_keys']) if task_row.get('output_keys') else []
                ))
            
            return Workflow(
                workflow_id=row['workflow_id'],
                name=row['name'],
                description=row['description'],
                tasks=tasks,
                status=row['status'],
                created_at=row['created_at'],
                started_at=row['started_at'],
                completed_at=row['completed_at'],
                error=row['error'],
                metadata=json.loads(row['metadata']) if row['metadata'] else {}
            )
    
    def list_workflows(self, status: Optional[str] = None, limit: int = 100) -> List[Workflow]:
        """List all workflows, optionally filtered by status"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            if status:
                rows = conn.execute(
                    "SELECT workflow_id FROM workflows WHERE status = ? ORDER BY created_at DESC LIMIT ?",
                    (status, limit)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT workflow_id FROM workflows ORDER BY created_at DESC LIMIT ?",
                    (limit,)
                ).fetchall()
            
            workflows = []
            for row in rows:
                workflow = self.load_workflow(row['workflow_id'])
                if workflow:
                    workflows.append(workflow)
            
            return workflows
    
    def delete_workflow(self, workflow_id: str):
        """Delete a workflow and its tasks"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM tasks WHERE workflow_id = ?", (workflow_id,))
            conn.execute("DELETE FROM workflows WHERE workflow_id = ?", (workflow_id,))
            conn.commit()


class WorkflowEngine:
    """Executes workflows and manages their lifecycle"""
    
    def __init__(self, storage: WorkflowStorage, ipfs_accelerate_instance=None):
        self.storage = storage
        self.ipfs_instance = ipfs_accelerate_instance
        self._running_workflows: Dict[str, asyncio.Task] = {}
    
    async def execute_task(self, workflow: Workflow, task: WorkflowTask, task_results: Dict[str, Any]) -> bool:
        """
        Execute a single task with support for AI model pipelines
        
        Args:
            workflow: The workflow being executed
            task: The task to execute
            task_results: Results from previously completed tasks (for data passing)
        
        Returns:
            bool: True if successful, False otherwise
        """
        logger.info(f"Executing task {task.task_id}: {task.name} (type: {task.type})")
        
        task.status = TaskStatus.RUNNING.value
        task.started_at = time.time()
        self.storage.save_workflow(workflow)
        
        try:
            # Prepare inputs from dependencies if input_mapping is defined
            task_inputs = dict(task.config.get('inputs', {}))
            
            # Apply input mapping from dependency outputs
            for input_key, mapping in task.input_mapping.items():
                # mapping format: "task_id.output_key" or just "task_id" for full output
                if '.' in mapping:
                    dep_task_id, output_key = mapping.split('.', 1)
                    if dep_task_id in task_results and output_key in task_results[dep_task_id]:
                        task_inputs[input_key] = task_results[dep_task_id][output_key]
                else:
                    # Use full output from dependency
                    dep_task_id = mapping
                    if dep_task_id in task_results:
                        task_inputs[input_key] = task_results[dep_task_id]
            
            # Execute based on task type
            if task.type == "text_model":
                # Text generation models (LLMs, prompt enhancement, etc.)
                task.result = await self._execute_text_model(task, task_inputs)
                
            elif task.type == "image_model":
                # Image generation/manipulation models
                task.result = await self._execute_image_model(task, task_inputs)
                
            elif task.type == "video_model":
                # Video generation/manipulation models
                task.result = await self._execute_video_model(task, task_inputs)
                
            elif task.type == "audio_model":
                # Audio generation/manipulation models
                task.result = await self._execute_audio_model(task, task_inputs)
                
            elif task.type == "filter_model":
                # Filter models (NSFW, quality check, etc.)
                task.result = await self._execute_filter_model(task, task_inputs)
                
            elif task.type == "processing":
                # Custom processing task
                task.result = await self._execute_processing(task, task_inputs)
                
            elif task.type in ["inference", "custom"]:
                # Legacy/generic inference task
                task.result = await self._execute_generic_inference(task, task_inputs)
                
            else:
                raise ValueError(f"Unknown task type: {task.type}")
            
            task.status = TaskStatus.COMPLETED.value
            task.completed_at = time.time()
            logger.info(f"Task {task.task_id} completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Task {task.task_id} failed: {e}")
            task.status = TaskStatus.FAILED.value
            task.error = str(e)
            task.completed_at = time.time()
            return False
        
        finally:
            self.storage.save_workflow(workflow)
    
    async def _execute_text_model(self, task: WorkflowTask, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a text/LLM model task"""
        model = task.config.get('model', 'gpt2')
        prompt = inputs.get('prompt', inputs.get('text', ''))
        max_length = task.config.get('max_length', 100)
        temperature = task.config.get('temperature', 0.7)
        
        # Simulated text generation
        await asyncio.sleep(1)
        
        return {
            'model': model,
            'prompt': prompt,
            'text': f"Generated text based on: {prompt[:50]}...",
            'enhanced_prompt': f"Enhanced: {prompt}",  # For prompt enhancement use case
            'parameters': {'max_length': max_length, 'temperature': temperature}
        }
    
    async def _execute_image_model(self, task: WorkflowTask, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an image generation/manipulation model"""
        model = task.config.get('model', 'stable-diffusion')
        prompt = inputs.get('prompt', inputs.get('text', ''))
        image_input = inputs.get('image')  # For img2img or upscaling
        
        # Simulated image generation
        await asyncio.sleep(2)
        
        return {
            'model': model,
            'prompt': prompt,
            'image': f"generated_image_{task.task_id}.png",  # Placeholder
            'image_url': f"https://placeholder.com/generated/{task.task_id}",
            'width': task.config.get('width', 512),
            'height': task.config.get('height', 512)
        }
    
    async def _execute_video_model(self, task: WorkflowTask, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a video generation model"""
        model = task.config.get('model', 'animatediff')
        prompt = inputs.get('prompt', '')
        image_input = inputs.get('image')  # Base image for video
        
        # Simulated video generation
        await asyncio.sleep(3)
        
        return {
            'model': model,
            'prompt': prompt,
            'video': f"generated_video_{task.task_id}.mp4",  # Placeholder
            'video_url': f"https://placeholder.com/generated/{task.task_id}.mp4",
            'duration': task.config.get('duration', 4),
            'fps': task.config.get('fps', 24)
        }
    
    async def _execute_audio_model(self, task: WorkflowTask, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an audio generation/manipulation model"""
        model = task.config.get('model', 'whisper')
        text_input = inputs.get('text', '')
        audio_input = inputs.get('audio')
        
        # Simulated audio processing
        await asyncio.sleep(1)
        
        return {
            'model': model,
            'audio': f"generated_audio_{task.task_id}.wav",
            'audio_url': f"https://placeholder.com/generated/{task.task_id}.wav",
            'transcript': text_input if audio_input else None,
            'duration': task.config.get('duration', 10)
        }
    
    async def _execute_filter_model(self, task: WorkflowTask, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a filter/classification model (NSFW, quality, etc.)"""
        model = task.config.get('model', 'nsfw-filter')
        image = inputs.get('image')
        text = inputs.get('text', '')
        
        # Simulated filter check
        await asyncio.sleep(0.5)
        
        filter_type = task.config.get('filter_type', 'nsfw')
        
        return {
            'model': model,
            'filter_type': filter_type,
            'passed': True,  # Simulated - would be actual filter result
            'confidence': 0.95,
            'labels': ['safe', 'appropriate'],
            'input_preserved': image or text  # Pass through the input
        }
    
    async def _execute_processing(self, task: WorkflowTask, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a custom processing task"""
        processing_type = task.config.get('processing_type', 'generic')
        
        # Simulated processing
        await asyncio.sleep(0.5)
        
        return {
            'processing_type': processing_type,
            'processed': True,
            'output': inputs  # Pass through with processing flag
        }
    
    async def _execute_generic_inference(self, task: WorkflowTask, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a generic inference task (legacy compatibility)"""
        if self.ipfs_instance:
            model = task.config.get('model', 'gpt2')
            model_inputs = inputs.get('inputs', ['Hello world'])
            
            # Use the existing inference infrastructure
            result = await self._run_inference(model, model_inputs)
            return result
        else:
            # Simulated inference
            await asyncio.sleep(1)
            return {
                'output': 'Simulated output',
                'model': task.config.get('model'),
                'inputs': inputs
            }
    
    async def _run_inference(self, model: str, inputs: List[str]) -> Dict[str, Any]:
        """Run inference using the IPFS accelerate instance"""
        if not self.ipfs_instance:
            raise RuntimeError("IPFS accelerate instance not available")
        
        # This would integrate with the actual inference system
        # For now, return a simulated result
        return {
            'model': model,
            'inputs': inputs,
            'outputs': [f"Output for: {inp}" for inp in inputs],
            'timestamp': time.time()
        }
    
    async def execute_workflow(self, workflow_id: str):
        """Execute a complete workflow"""
        workflow = self.storage.load_workflow(workflow_id)
        if not workflow:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        if workflow.status not in [WorkflowStatus.PENDING.value, WorkflowStatus.PAUSED.value]:
            raise ValueError(f"Workflow {workflow_id} is not in a runnable state: {workflow.status}")
        
        logger.info(f"Starting workflow {workflow_id}: {workflow.name}")
        workflow.status = WorkflowStatus.RUNNING.value
        workflow.started_at = time.time()
        self.storage.save_workflow(workflow)
        
        try:
            # Track completed tasks for dependency resolution
            completed_tasks = set()
            # Store task results for data passing
            task_results = {}
            
            while True:
                # Find tasks that are ready to run (dependencies met)
                runnable_tasks = []
                for task in workflow.tasks:
                    if task.status == TaskStatus.PENDING.value:
                        deps_met = all(dep in completed_tasks for dep in task.dependencies)
                        if deps_met:
                            runnable_tasks.append(task)
                
                if not runnable_tasks:
                    # Check if all tasks are done
                    all_done = all(
                        t.status in [TaskStatus.COMPLETED.value, TaskStatus.FAILED.value, TaskStatus.SKIPPED.value]
                        for t in workflow.tasks
                    )
                    if all_done:
                        break
                    
                    # Check for paused state
                    workflow = self.storage.load_workflow(workflow_id)
                    if workflow.status == WorkflowStatus.PAUSED.value:
                        logger.info(f"Workflow {workflow_id} paused")
                        return
                    
                    # Wait a bit before checking again
                    await asyncio.sleep(0.5)
                    continue
                
                # Execute runnable tasks (could be parallelized)
                for task in runnable_tasks:
                    success = await self.execute_task(workflow, task, task_results)
                    if success:
                        completed_tasks.add(task.task_id)
                        # Store task results for downstream tasks
                        if task.result:
                            task_results[task.task_id] = task.result
                
                # Reload workflow to get latest state
                workflow = self.storage.load_workflow(workflow_id)
            
            # Determine final status
            if any(t.status == TaskStatus.FAILED.value for t in workflow.tasks):
                workflow.status = WorkflowStatus.FAILED.value
                workflow.error = "One or more tasks failed"
            else:
                workflow.status = WorkflowStatus.COMPLETED.value
            
            workflow.completed_at = time.time()
            logger.info(f"Workflow {workflow_id} finished with status: {workflow.status}")
            
        except Exception as e:
            logger.error(f"Workflow {workflow_id} error: {e}")
            workflow.status = WorkflowStatus.FAILED.value
            workflow.error = str(e)
            workflow.completed_at = time.time()
        
        finally:
            self.storage.save_workflow(workflow)
            if workflow_id in self._running_workflows:
                del self._running_workflows[workflow_id]
    
    def start_workflow(self, workflow_id: str):
        """Start a workflow in the background"""
        if workflow_id in self._running_workflows:
            raise ValueError(f"Workflow {workflow_id} is already running")
        
        task = asyncio.create_task(self.execute_workflow(workflow_id))
        self._running_workflows[workflow_id] = task
        return task
    
    def pause_workflow(self, workflow_id: str):
        """Pause a running workflow"""
        workflow = self.storage.load_workflow(workflow_id)
        if not workflow:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        if workflow.status != WorkflowStatus.RUNNING.value:
            raise ValueError(f"Workflow {workflow_id} is not running")
        
        workflow.status = WorkflowStatus.PAUSED.value
        self.storage.save_workflow(workflow)
        logger.info(f"Workflow {workflow_id} paused")
    
    def stop_workflow(self, workflow_id: str):
        """Stop a workflow"""
        workflow = self.storage.load_workflow(workflow_id)
        if not workflow:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        workflow.status = WorkflowStatus.STOPPED.value
        workflow.completed_at = time.time()
        self.storage.save_workflow(workflow)
        
        # Cancel the task if it's running
        if workflow_id in self._running_workflows:
            self._running_workflows[workflow_id].cancel()
            del self._running_workflows[workflow_id]
        
        logger.info(f"Workflow {workflow_id} stopped")


class WorkflowManager:
    """High-level workflow management interface"""
    
    def __init__(self, storage: WorkflowStorage = None, ipfs_accelerate_instance=None):
        if storage is None:
            storage = WorkflowStorage()
        
        self.storage = storage
        self.engine = WorkflowEngine(storage, ipfs_accelerate_instance)
    
    def create_workflow(self, name: str, description: str, tasks: List[Dict[str, Any]]) -> Workflow:
        """Create a new workflow with AI model pipeline support"""
        workflow_id = str(uuid.uuid4())
        
        # Convert task dicts to WorkflowTask objects
        workflow_tasks = []
        task_id_map = {}  # Map task indices to IDs for dependency resolution
        
        for i, task_dict in enumerate(tasks):
            task_id = f"{workflow_id}-task-{i}"
            task_id_map[i] = task_id
            
            # Handle dependencies (can be task indices or IDs)
            dependencies = []
            for dep in task_dict.get('dependencies', []):
                if isinstance(dep, int):
                    # Convert task index to task ID
                    dependencies.append(task_id_map.get(dep, dep))
                else:
                    dependencies.append(dep)
            
            workflow_tasks.append(WorkflowTask(
                task_id=task_id,
                name=task_dict['name'],
                type=task_dict['type'],
                config=task_dict.get('config', {}),
                dependencies=dependencies,
                input_mapping=task_dict.get('input_mapping', {}),
                output_keys=task_dict.get('output_keys', [])
            ))
        
        workflow = Workflow(
            workflow_id=workflow_id,
            name=name,
            description=description,
            tasks=workflow_tasks
        )
        
        self.storage.save_workflow(workflow)
        logger.info(f"Created workflow {workflow_id}: {name}")
        return workflow
    
    @staticmethod
    def create_image_generation_pipeline() -> Dict[str, Any]:
        """
        Template for: Prompt Enhancement -> Image Generation -> Upscaling
        """
        return {
            'name': 'Image Generation Pipeline',
            'description': 'LLM prompt enhancement → image generation → upscaling',
            'tasks': [
                {
                    'name': 'Enhance Prompt',
                    'type': 'text_model',
                    'config': {
                        'model': 'gpt-4',
                        'inputs': {'prompt': 'a beautiful landscape'},
                        'max_length': 200
                    },
                    'dependencies': [],
                    'output_keys': ['enhanced_prompt']
                },
                {
                    'name': 'Generate Image',
                    'type': 'image_model',
                    'config': {
                        'model': 'stable-diffusion-xl',
                        'width': 1024,
                        'height': 1024
                    },
                    'dependencies': [0],  # Depends on task 0 (prompt enhancement)
                    'input_mapping': {
                        'prompt': '0.enhanced_prompt'  # Use enhanced_prompt from task 0
                    },
                    'output_keys': ['image', 'image_url']
                },
                {
                    'name': 'Upscale Image',
                    'type': 'image_model',
                    'config': {
                        'model': 'real-esrgan',
                        'scale': 4
                    },
                    'dependencies': [1],  # Depends on task 1 (image generation)
                    'input_mapping': {
                        'image': '1.image'  # Use image from task 1
                    },
                    'output_keys': ['image', 'image_url']
                }
            ]
        }
    
    @staticmethod
    def create_video_generation_pipeline() -> Dict[str, Any]:
        """
        Template for: Prompt Enhancement -> Image Generation -> Video Generation
        """
        return {
            'name': 'Text-to-Video Pipeline',
            'description': 'Enhanced prompt → image → animated video',
            'tasks': [
                {
                    'name': 'Enhance Prompt',
                    'type': 'text_model',
                    'config': {
                        'model': 'gpt-4',
                        'inputs': {'prompt': 'a serene mountain scene'},
                        'max_length': 150
                    },
                    'dependencies': [],
                    'output_keys': ['enhanced_prompt']
                },
                {
                    'name': 'Generate Base Image',
                    'type': 'image_model',
                    'config': {
                        'model': 'stable-diffusion-xl',
                        'width': 768,
                        'height': 768
                    },
                    'dependencies': [0],
                    'input_mapping': {
                        'prompt': '0.enhanced_prompt'
                    },
                    'output_keys': ['image']
                },
                {
                    'name': 'Generate Video',
                    'type': 'video_model',
                    'config': {
                        'model': 'animatediff',
                        'duration': 4,
                        'fps': 24
                    },
                    'dependencies': [0, 1],  # Depends on both prompt and image
                    'input_mapping': {
                        'prompt': '0.enhanced_prompt',
                        'image': '1.image'
                    },
                    'output_keys': ['video', 'video_url']
                }
            ]
        }
    
    @staticmethod
    def create_safe_image_pipeline() -> Dict[str, Any]:
        """
        Template for: NSFW Filter -> Image Generation -> Quality Check
        """
        return {
            'name': 'Safe Image Generation',
            'description': 'NSFW filter → image generation → quality validation',
            'tasks': [
                {
                    'name': 'Check Prompt Safety',
                    'type': 'filter_model',
                    'config': {
                        'model': 'nsfw-text-classifier',
                        'filter_type': 'nsfw',
                        'inputs': {'text': 'a beautiful portrait'}
                    },
                    'dependencies': [],
                    'output_keys': ['passed', 'input_preserved']
                },
                {
                    'name': 'Generate Image',
                    'type': 'image_model',
                    'config': {
                        'model': 'stable-diffusion',
                        'width': 512,
                        'height': 512
                    },
                    'dependencies': [0],
                    'input_mapping': {
                        'prompt': '0.input_preserved'
                    },
                    'output_keys': ['image']
                },
                {
                    'name': 'Validate Image Quality',
                    'type': 'filter_model',
                    'config': {
                        'model': 'image-quality-scorer',
                        'filter_type': 'quality'
                    },
                    'dependencies': [1],
                    'input_mapping': {
                        'image': '1.image'
                    },
                    'output_keys': ['passed', 'confidence']
                }
            ]
        }
    
    @staticmethod
    def create_multimodal_pipeline() -> Dict[str, Any]:
        """
        Template for: Text → Image → Audio Description → Video
        Complex multi-modal AI pipeline
        """
        return {
            'name': 'Multimodal Content Pipeline',
            'description': 'Text → Image → Audio → Video generation',
            'tasks': [
                {
                    'name': 'Generate Story',
                    'type': 'text_model',
                    'config': {
                        'model': 'gpt-4',
                        'inputs': {'prompt': 'Write a short story'},
                        'max_length': 300
                    },
                    'dependencies': [],
                    'output_keys': ['text']
                },
                {
                    'name': 'Generate Scene Image',
                    'type': 'image_model',
                    'config': {
                        'model': 'dalle-3',
                        'width': 1024,
                        'height': 1024
                    },
                    'dependencies': [0],
                    'input_mapping': {
                        'prompt': '0.text'
                    },
                    'output_keys': ['image']
                },
                {
                    'name': 'Generate Narration',
                    'type': 'audio_model',
                    'config': {
                        'model': 'tts-1',
                        'voice': 'alloy'
                    },
                    'dependencies': [0],
                    'input_mapping': {
                        'text': '0.text'
                    },
                    'output_keys': ['audio']
                },
                {
                    'name': 'Create Video',
                    'type': 'video_model',
                    'config': {
                        'model': 'video-composer',
                        'duration': 10,
                        'fps': 30
                    },
                    'dependencies': [1, 2],  # Depends on image and audio
                    'input_mapping': {
                        'image': '1.image',
                        'audio': '2.audio'
                    },
                    'output_keys': ['video', 'video_url']
                }
            ]
        }
    
    def get_workflow(self, workflow_id: str) -> Optional[Workflow]:
        """Get a workflow by ID"""
        return self.storage.load_workflow(workflow_id)
    
    def list_workflows(self, status: Optional[str] = None) -> List[Workflow]:
        """List all workflows"""
        return self.storage.list_workflows(status=status)
    
    def delete_workflow(self, workflow_id: str):
        """Delete a workflow"""
        self.storage.delete_workflow(workflow_id)
        logger.info(f"Deleted workflow {workflow_id}")
    
    def start_workflow(self, workflow_id: str):
        """Start executing a workflow"""
        return self.engine.start_workflow(workflow_id)
    
    def pause_workflow(self, workflow_id: str):
        """Pause a workflow"""
        self.engine.pause_workflow(workflow_id)
    
    def stop_workflow(self, workflow_id: str):
        """Stop a workflow"""
        self.engine.stop_workflow(workflow_id)
