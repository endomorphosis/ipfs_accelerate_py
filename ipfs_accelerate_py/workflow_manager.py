"""
Workflow Management System for IPFS Accelerate MCP Server

This module provides workflow definition, execution, and management capabilities.
"""

import json
import time
import uuid
import logging
import anyio
from typing import Dict, List, Any, Optional
from enum import Enum
from dataclasses import dataclass, asdict
from pathlib import Path

# Import DuckDB for storage (with fallback for migration scenarios)
try:
    import duckdb
    HAVE_DUCKDB = True
except ImportError:
    HAVE_DUCKDB = False
    raise ImportError("DuckDB is required for workflow storage. Install with: pip install duckdb")

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


# HuggingFace Pipeline Tags - Official taxonomy from HF Hub
# This allows automatic model classification without human intervention
# Tasks use these tags so models can be automatically mapped based on their pipeline_tag
HF_PIPELINE_TAGS = {
    # Text tasks
    'text-generation', 'text2text-generation', 'text-classification',
    'token-classification', 'question-answering', 'fill-mask',
    'summarization', 'translation', 'sentence-similarity',
    'conversational', 'feature-extraction',
    # Audio tasks
    'text-to-speech', 'automatic-speech-recognition', 'audio-classification',
    'audio-to-audio', 'voice-activity-detection',
    # Image tasks
    'text-to-image', 'image-to-text', 'image-classification',
    'image-segmentation', 'object-detection', 'depth-estimation',
    'image-to-image', 'unconditional-image-generation',
    # Video tasks
    'video-classification', 'image-to-video', 'text-to-video',
    # Multimodal tasks
    'visual-question-answering', 'document-question-answering',
    'table-question-answering',
    # Zero-shot tasks
    'zero-shot-classification', 'zero-shot-image-classification',
    'zero-shot-object-detection',
    # Other
    'reinforcement-learning', 'robotics',
    # Legacy/custom (for backwards compatibility and special processing)
    'inference', 'processing', 'filter'
}


@dataclass
class WorkflowTask:
    """Represents a single task in a workflow - designed for AI model pipelines
    
    Tasks use HuggingFace pipeline_tag taxonomy for automatic model classification.
    This allows the scraper to automatically map models to task types based on their
    pipeline_tag without human intervention.
    
    Memory management fields help prevent OOM and control resource allocation:
    - vram_pinned: Keep model loaded in VRAM (prevents unloading)
    - preemptable: Allow task to be interrupted for higher priority work
    - max_memory_mb: Maximum memory allocation for this task (0 = unlimited)
    - batch_size: Inference batch size (affects memory vs throughput)
    - priority: Task priority for execution scheduling (1=lowest, 10=highest)
    """
    task_id: str
    name: str
    type: str  # HuggingFace pipeline_tag (e.g., 'text-generation', 'text-to-image', 'image-to-video', etc.)
    config: Dict[str, Any]  # Model-specific config (model_name, parameters, etc.)
    status: str = TaskStatus.PENDING.value
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    dependencies: List[str] = None  # List of task_ids that must complete first
    input_mapping: Optional[Dict[str, str]] = None  # Maps output keys from dependencies to input keys
    output_keys: Optional[List[str]] = None  # Keys to extract from result for downstream tasks
    # Memory management fields
    vram_pinned: bool = False  # Pin model in VRAM (prevents unloading)
    preemptable: bool = True  # Allow task to be interrupted
    max_memory_mb: int = 0  # Max memory in MB (0 = unlimited)
    batch_size: int = 1  # Inference batch size
    priority: int = 5  # Priority 1-10 (1=lowest, 10=highest)

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
    """Handles workflow persistence using DuckDB with Parquet storage
    
    Migrated from SQLite to DuckDB for better performance, analytics capabilities,
    and Parquet file format for efficient storage and data interchange.
    """
    
    def __init__(self, db_path: str = None):
        if not HAVE_DUCKDB:
            raise RuntimeError("DuckDB is required for workflow storage. Install with: pip install duckdb")
        
        if db_path is None:
            # Use .duckdb extension instead of .db
            db_path = str(Path.home() / ".ipfs_accelerate" / "workflows.duckdb")
        
        self.db_path = db_path
        self.parquet_dir = str(Path(self.db_path).parent / "workflow_data")
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.parquet_dir).mkdir(parents=True, exist_ok=True)
        
        # Migrate from SQLite if it exists
        self._migrate_from_sqlite_if_needed()
        
        self._init_db()
    
    def _migrate_from_sqlite_if_needed(self):
        """Migrate data from legacy SQLite database if it exists"""
        legacy_db = str(Path.home() / ".ipfs_accelerate" / "workflows.db")
        if not Path(legacy_db).exists():
            return
        
        logger.info(f"Found legacy SQLite database, migrating to DuckDB...")
        
        try:
            import sqlite3
            
            # Connect to both databases
            sqlite_conn = sqlite3.connect(legacy_db)
            sqlite_conn.row_factory = sqlite3.Row
            
            duck_conn = duckdb.connect(self.db_path)
            
            # Create tables in DuckDB first
            self._create_tables_in_connection(duck_conn)
            
            # Migrate workflows
            workflows = sqlite_conn.execute("SELECT * FROM workflows").fetchall()
            for row in workflows:
                duck_conn.execute("""
                    INSERT INTO workflows 
                    (workflow_id, name, description, status, created_at, started_at, completed_at, error, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    row['workflow_id'], row['name'], row['description'], row['status'],
                    row['created_at'], row['started_at'], row['completed_at'], 
                    row['error'], row['metadata']
                ))
            
            # Migrate tasks
            tasks = sqlite_conn.execute("SELECT * FROM tasks").fetchall()
            for row in tasks:
                # Handle optional columns that may not exist in older SQLite DBs
                vram_pinned = row['vram_pinned'] if 'vram_pinned' in row.keys() else 0
                preemptable = row['preemptable'] if 'preemptable' in row.keys() else 1
                max_memory_mb = row['max_memory_mb'] if 'max_memory_mb' in row.keys() else 0
                batch_size = row['batch_size'] if 'batch_size' in row.keys() else 1
                priority = row['priority'] if 'priority' in row.keys() else 5
                
                duck_conn.execute("""
                    INSERT INTO tasks
                    (task_id, workflow_id, name, type, config, status, result, error,
                     started_at, completed_at, dependencies, input_mapping, output_keys,
                     vram_pinned, preemptable, max_memory_mb, batch_size, priority)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    row['task_id'], row['workflow_id'], row['name'], row['type'],
                    row['config'], row['status'], row['result'], row['error'],
                    row['started_at'], row['completed_at'], row['dependencies'],
                    row['input_mapping'], row['output_keys'],
                    vram_pinned, preemptable, max_memory_mb, batch_size, priority
                ))
            
            duck_conn.commit()
            duck_conn.close()
            sqlite_conn.close()
            
            # Rename the old SQLite database as backup
            backup_path = legacy_db + ".migrated_backup"
            Path(legacy_db).rename(backup_path)
            logger.info(f"Migration complete! SQLite backup saved to: {backup_path}")
            
        except Exception as e:
            logger.error(f"Error during SQLite to DuckDB migration: {e}")
            logger.info("Continuing with fresh DuckDB database")
    
    def _create_tables_in_connection(self, conn):
        """Create tables in a DuckDB connection"""
        conn.execute("""
            CREATE TABLE IF NOT EXISTS workflows (
                workflow_id VARCHAR PRIMARY KEY,
                name VARCHAR NOT NULL,
                description VARCHAR,
                status VARCHAR NOT NULL,
                created_at DOUBLE NOT NULL,
                started_at DOUBLE,
                completed_at DOUBLE,
                error VARCHAR,
                metadata VARCHAR
            )
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                task_id VARCHAR PRIMARY KEY,
                workflow_id VARCHAR NOT NULL,
                name VARCHAR NOT NULL,
                type VARCHAR NOT NULL,
                config VARCHAR NOT NULL,
                status VARCHAR NOT NULL,
                result VARCHAR,
                error VARCHAR,
                started_at DOUBLE,
                completed_at DOUBLE,
                dependencies VARCHAR,
                input_mapping VARCHAR,
                output_keys VARCHAR,
                vram_pinned BOOLEAN DEFAULT false,
                preemptable BOOLEAN DEFAULT true,
                max_memory_mb INTEGER DEFAULT 0,
                batch_size INTEGER DEFAULT 1,
                priority INTEGER DEFAULT 5
            )
        """)
    
    def _init_db(self):
        """Initialize DuckDB database schema"""
        with duckdb.connect(self.db_path) as conn:
            self._create_tables_in_connection(conn)
            
            # Export to Parquet for efficient storage and analytics
            workflows_parquet = str(Path(self.parquet_dir) / "workflows.parquet")
            tasks_parquet = str(Path(self.parquet_dir) / "tasks.parquet")
            
            # Create Parquet files if they don't exist
            try:
                conn.execute(f"COPY (SELECT * FROM workflows) TO '{workflows_parquet}' (FORMAT PARQUET)")
            except:
                pass  # Table may be empty
            
            try:
                conn.execute(f"COPY (SELECT * FROM tasks) TO '{tasks_parquet}' (FORMAT PARQUET)")
            except:
                pass  # Table may be empty
    
    def save_workflow(self, workflow: Workflow):
        """Save or update a workflow"""
        with duckdb.connect(self.db_path) as conn:
            # Insert or replace workflow
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
                     started_at, completed_at, dependencies, input_mapping, output_keys,
                     vram_pinned, preemptable, max_memory_mb, batch_size, priority)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    json.dumps(task.output_keys),
                    task.vram_pinned,
                    task.preemptable,
                    task.max_memory_mb,
                    task.batch_size,
                    task.priority
                ))
            
            # Export to Parquet after each save for persistence
            workflows_parquet = str(Path(self.parquet_dir) / "workflows.parquet")
            tasks_parquet = str(Path(self.parquet_dir) / "tasks.parquet")
            
            conn.execute(f"COPY (SELECT * FROM workflows) TO '{workflows_parquet}' (FORMAT PARQUET, OVERWRITE_OR_IGNORE)")
            conn.execute(f"COPY (SELECT * FROM tasks) TO '{tasks_parquet}' (FORMAT PARQUET, OVERWRITE_OR_IGNORE)")
    
    def load_workflow(self, workflow_id: str) -> Optional[Workflow]:
        """Load a workflow by ID"""
        with duckdb.connect(self.db_path) as conn:
            # Load workflow
            result = conn.execute(
                "SELECT * FROM workflows WHERE workflow_id = ?", 
                (workflow_id,)
            ).fetchone()
            
            if not result:
                return None
            
            # DuckDB returns tuples, so we need to get column names
            columns = [desc[0] for desc in conn.execute("SELECT * FROM workflows WHERE workflow_id = ?", (workflow_id,)).description]
            row = dict(zip(columns, result))
            
            # Load tasks
            task_results = conn.execute(
                "SELECT * FROM tasks WHERE workflow_id = ? ORDER BY task_id",
                (workflow_id,)
            ).fetchall()
            
            task_columns = [desc[0] for desc in conn.execute("SELECT * FROM tasks LIMIT 0").description]
            
            tasks = []
            for task_result in task_results:
                task_row = dict(zip(task_columns, task_result))
                
                tasks.append(WorkflowTask(
                    task_id=task_row['task_id'],
                    name=task_row['name'],
                    type=task_row['type'],
                    config=json.loads(task_row['config']),
                    status=task_row['status'],
                    result=json.loads(task_row['result']) if task_row.get('result') else None,
                    error=task_row.get('error'),
                    started_at=task_row.get('started_at'),
                    completed_at=task_row.get('completed_at'),
                    dependencies=json.loads(task_row['dependencies']) if task_row.get('dependencies') else [],
                    input_mapping=json.loads(task_row['input_mapping']) if task_row.get('input_mapping') else {},
                    output_keys=json.loads(task_row['output_keys']) if task_row.get('output_keys') else [],
                    vram_pinned=task_row.get('vram_pinned', False) if isinstance(task_row.get('vram_pinned'), bool) else bool(task_row.get('vram_pinned', False)),
                    preemptable=task_row.get('preemptable', True) if isinstance(task_row.get('preemptable'), bool) else bool(task_row.get('preemptable', True)),
                    max_memory_mb=int(task_row.get('max_memory_mb', 0)),
                    batch_size=int(task_row.get('batch_size', 1)),
                    priority=int(task_row.get('priority', 5))
                ))
            
            return Workflow(
                workflow_id=row['workflow_id'],
                name=row['name'],
                description=row['description'],
                tasks=tasks,
                status=row['status'],
                created_at=row['created_at'],
                started_at=row.get('started_at'),
                completed_at=row.get('completed_at'),
                error=row.get('error'),
                metadata=json.loads(row['metadata']) if row.get('metadata') else {}
            )
    
    def list_workflows(self, status: Optional[str] = None, limit: int = 100) -> List[Workflow]:
        """List all workflows, optionally filtered by status"""
        with duckdb.connect(self.db_path) as conn:
            if status:
                results = conn.execute(
                    "SELECT workflow_id FROM workflows WHERE status = ? ORDER BY created_at DESC LIMIT ?",
                    (status, limit)
                ).fetchall()
            else:
                results = conn.execute(
                    "SELECT workflow_id FROM workflows ORDER BY created_at DESC LIMIT ?",
                    (limit,)
                ).fetchall()
            
            workflows = []
            for result in results:
                workflow = self.load_workflow(result[0])
                if workflow:
                    workflows.append(workflow)
            
            return workflows
    
    def delete_workflow(self, workflow_id: str):
        """Delete a workflow and its tasks"""
        with duckdb.connect(self.db_path) as conn:
            conn.execute("DELETE FROM tasks WHERE workflow_id = ?", (workflow_id,))
            conn.execute("DELETE FROM workflows WHERE workflow_id = ?", (workflow_id,))
            
            # Update Parquet files
            workflows_parquet = str(Path(self.parquet_dir) / "workflows.parquet")
            tasks_parquet = str(Path(self.parquet_dir) / "tasks.parquet")
            
            conn.execute(f"COPY (SELECT * FROM workflows) TO '{workflows_parquet}' (FORMAT PARQUET, OVERWRITE_OR_IGNORE)")
            conn.execute(f"COPY (SELECT * FROM tasks) TO '{tasks_parquet}' (FORMAT PARQUET, OVERWRITE_OR_IGNORE)")
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
            
            # Execute based on HuggingFace pipeline_tag
            # Map pipeline tags to execution methods
            task_result = await self._execute_hf_pipeline_task(task, task_inputs)
            task.result = task_result
            
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
    
    async def _execute_hf_pipeline_task(self, task: WorkflowTask, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a task based on HuggingFace pipeline_tag
        
        This method routes execution based on the official HuggingFace taxonomy,
        allowing automatic model classification without human intervention.
        The model scraper can automatically map models based on their pipeline_tag.
        """
        pipeline_tag = task.type
        
        # Map HuggingFace pipeline tags to execution methods
        # Text tasks
        if pipeline_tag in ['text-generation', 'text2text-generation', 'conversational', 
                           'summarization', 'translation', 'fill-mask']:
            return await self._execute_text_generation(task, inputs)
        
        elif pipeline_tag in ['text-classification', 'token-classification', 'question-answering',
                             'sentence-similarity', 'zero-shot-classification', 
                             'table-question-answering']:
            return await self._execute_text_analysis(task, inputs)
        
        # Image tasks
        elif pipeline_tag in ['text-to-image', 'unconditional-image-generation']:
            return await self._execute_image_generation(task, inputs)
        
        elif pipeline_tag in ['image-to-image', 'image-segmentation', 'object-detection',
                             'depth-estimation', 'image-classification',
                             'zero-shot-image-classification', 'zero-shot-object-detection']:
            return await self._execute_image_analysis(task, inputs)
        
        elif pipeline_tag == 'image-to-text':
            return await self._execute_image_to_text(task, inputs)
        
        # Video tasks
        elif pipeline_tag in ['image-to-video', 'text-to-video', 'video-classification']:
            return await self._execute_video_task(task, inputs)
        
        # Audio tasks
        elif pipeline_tag in ['text-to-speech', 'audio-to-audio']:
            return await self._execute_audio_generation(task, inputs)
        
        elif pipeline_tag in ['automatic-speech-recognition', 'audio-classification',
                             'voice-activity-detection']:
            return await self._execute_audio_analysis(task, inputs)
        
        # Multimodal tasks
        elif pipeline_tag in ['visual-question-answering', 'document-question-answering']:
            return await self._execute_multimodal(task, inputs)
        
        # Feature extraction
        elif pipeline_tag == 'feature-extraction':
            return await self._execute_feature_extraction(task, inputs)
        
        # Special/legacy types
        elif pipeline_tag == 'filter':
            return await self._execute_filter(task, inputs)
        
        elif pipeline_tag == 'processing':
            return await self._execute_processing(task, inputs)
        
        elif pipeline_tag == 'inference':
            return await self._execute_generic_inference(task, inputs)
        
        else:
            # Unknown pipeline tag - try generic inference
            logger.warning(f"Unknown pipeline_tag '{pipeline_tag}', using generic inference")
            return await self._execute_generic_inference(task, inputs)
    
    async def _execute_text_generation(self, task: WorkflowTask, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute text generation tasks (text-generation, summarization, translation, etc.)"""
        model = task.config.get('model', 'gpt2')
        prompt = inputs.get('prompt', inputs.get('text', ''))
        max_length = task.config.get('max_length', 100)
        temperature = task.config.get('temperature', 0.7)
        
        # Simulated text generation
        await anyio.sleep(1)
        logger.info(f"Executing {task.type} with model {model}")
        
        # Simulated output
        result = {
            'text': f"Generated text from {model} for prompt: {prompt[:50]}...",
            'enhanced_prompt': f"Enhanced: {prompt}",
            'model': model,
            'pipeline_tag': task.type,
            'parameters': {'max_length': max_length, 'temperature': temperature}
        }
        
        return result
    
    async def _execute_text_analysis(self, task: WorkflowTask, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute text analysis tasks (classification, QA, NER, etc.)"""
        model = task.config.get('model', 'bert-base')
        text = inputs.get('text', inputs.get('prompt', ''))
        
        await anyio.sleep(0.5)
        logger.info(f"Executing {task.type} with model {model}")
        
        result = {
            'label': 'positive',
            'score': 0.95,
            'analysis': f"Analysis result from {model}",
            'model': model,
            'pipeline_tag': task.type
        }
        
        return result
    
    async def _execute_image_generation(self, task: WorkflowTask, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute image generation tasks (text-to-image, unconditional generation)"""
        model = task.config.get('model', 'stable-diffusion-xl')
        prompt = inputs.get('prompt', inputs.get('text', ''))
        
        await anyio.sleep(2)
        logger.info(f"Executing {task.type} with model {model}")
        
        result = {
            'image': f"generated_image_{task.task_id}.png",
            'image_url': f"https://placeholder.com/generated/{task.task_id}",
            'prompt_used': prompt,
            'model': model,
            'pipeline_tag': task.type,
            'width': task.config.get('width', 1024),
            'height': task.config.get('height', 1024)
        }
        
        return result
    
    async def _execute_image_analysis(self, task: WorkflowTask, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute image analysis/manipulation tasks (segmentation, detection, classification, etc.)"""
        model = task.config.get('model', 'image-processor')
        image = inputs.get('image', '/path/to/input.png')
        
        await anyio.sleep(1)
        logger.info(f"Executing {task.type} with model {model}")
        
        result = {
            'processed_image': f"processed_{image}",
            'detections': [],
            'labels': ['object1', 'object2'],
            'scores': [0.95, 0.87],
            'model': model,
            'pipeline_tag': task.type
        }
        
        return result
    
    async def _execute_image_to_text(self, task: WorkflowTask, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute image-to-text tasks (captioning, OCR, etc.)"""
        model = task.config.get('model', 'blip-2')
        image = inputs.get('image', '/path/to/input.png')
        
        await anyio.sleep(1)
        logger.info(f"Executing {task.type} with model {model}")
        
        result = {
            'text': f"Description of image {image}",
            'captions': ['A detailed description of the image'],
            'model': model,
            'pipeline_tag': task.type
        }
        
        return result
    
    async def _execute_video_task(self, task: WorkflowTask, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute video tasks (generation, classification, etc.)"""
        model = task.config.get('model', 'animatediff')
        prompt = inputs.get('prompt', '')
        image_input = inputs.get('image')
        
        await anyio.sleep(3)
        logger.info(f"Executing {task.type} with model {model}")
        
        result = {
            'video': f"generated_video_{task.task_id}.mp4",
            'video_url': f"https://placeholder.com/generated/{task.task_id}.mp4",
            'frames': 30,
            'duration': task.config.get('duration', 4),
            'fps': task.config.get('fps', 24),
            'model': model,
            'pipeline_tag': task.type
        }
        
        return result
    
    async def _execute_audio_generation(self, task: WorkflowTask, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute audio generation tasks (TTS, audio-to-audio, etc.)"""
        model = task.config.get('model', 'elevenlabs-tts')
        text = inputs.get('text', inputs.get('prompt', ''))
        
        await anyio.sleep(1.5)
        logger.info(f"Executing {task.type} with model {model}")
        
        result = {
            'audio': f"generated_audio_{task.task_id}.wav",
            'audio_url': f"https://placeholder.com/generated/{task.task_id}.wav",
            'text_used': text,
            'duration': task.config.get('duration', 10),
            'model': model,
            'pipeline_tag': task.type
        }
        
        return result
    
    async def _execute_audio_analysis(self, task: WorkflowTask, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute audio analysis tasks (ASR, classification, etc.)"""
        model = task.config.get('model', 'whisper-large')
        audio = inputs.get('audio', '/path/to/input.wav')
        
        await anyio.sleep(1)
        logger.info(f"Executing {task.type} with model {model}")
        
        result = {
            'text': f"Transcription from {audio}",
            'language': 'en',
            'confidence': 0.96,
            'model': model,
            'pipeline_tag': task.type
        }
        
        return result
    
    async def _execute_multimodal(self, task: WorkflowTask, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute multimodal tasks (VQA, document QA, etc.)"""
        model = task.config.get('model', 'multimodal-model')
        
        await anyio.sleep(1.5)
        logger.info(f"Executing {task.type} with model {model}")
        
        result = {
            'answer': 'Response to multimodal query',
            'confidence': 0.92,
            'model': model,
            'pipeline_tag': task.type
        }
        
        return result
    
    async def _execute_feature_extraction(self, task: WorkflowTask, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute feature extraction tasks"""
        model = task.config.get('model', 'sentence-transformers')
        
        await anyio.sleep(0.5)
        logger.info(f"Executing {task.type} with model {model}")
        
        result = {
            'embeddings': [0.1, 0.2, 0.3],  # Simulated
            'dimensions': 768,
            'model': model,
            'pipeline_tag': task.type
        }
        
        return result
    
    async def _execute_filter(self, task: WorkflowTask, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute content filtering tasks (NSFW, quality checks, etc.)"""
        model = task.config.get('model', 'nsfw-filter')
        image = inputs.get('image')
        text = inputs.get('text', '')
        
        await anyio.sleep(0.3)
        logger.info(f"Executing filter with model {model}")
        
        result = {
            'passed': True,
            'score': 0.95,
            'labels': ['safe', 'appropriate'],
            'model': model,
            'pipeline_tag': task.type
        }
        
        # Preserve inputs for downstream tasks if filter passes
        result['text'] = text
        if image:
            result['image'] = image
        
        return result
    
    async def _execute_processing(self, task: WorkflowTask, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a custom processing task"""
        processing_type = task.config.get('processing_type', 'generic')
        
        # Simulated processing
        await anyio.sleep(0.5)
        
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
            await anyio.sleep(1)
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
                    await anyio.sleep(0.5)
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
        
        task = # TODO: Replace with task group - asyncio.create_task(self.execute_workflow(workflow_id))
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
                output_keys=task_dict.get('output_keys', []),
                vram_pinned=task_dict.get('vram_pinned', False),
                preemptable=task_dict.get('preemptable', True),
                max_memory_mb=task_dict.get('max_memory_mb', 0),
                batch_size=task_dict.get('batch_size', 1),
                priority=task_dict.get('priority', 5)
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
    
    def list_workflows(self, status: Optional[str] = None, limit: int = 100) -> List[Workflow]:
        """List workflows, optionally filtered by status"""
        return self.storage.list_workflows(status=status, limit=limit)
    
    def get_workflow(self, workflow_id: str) -> Optional[Workflow]:
        """Get a specific workflow by ID"""
        return self.storage.load_workflow(workflow_id)
    
    def update_workflow(
        self, 
        workflow_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        tasks: Optional[List[Dict[str, Any]]] = None
    ) -> Workflow:
        """
        Update an existing workflow
        
        Args:
            workflow_id: ID of workflow to update
            name: Optional new name
            description: Optional new description
            tasks: Optional new task list (replaces all tasks)
        
        Returns:
            Updated workflow
        
        Raises:
            ValueError: If workflow not found or if workflow is currently running
        """
        workflow = self.storage.load_workflow(workflow_id)
        if not workflow:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        # Don't allow updating running workflows
        if workflow.status == WorkflowStatus.RUNNING.value:
            raise ValueError(f"Cannot update running workflow {workflow_id}. Pause or stop it first.")
        
        # Update basic fields
        if name is not None:
            workflow.name = name
        if description is not None:
            workflow.description = description
        
        # Update tasks if provided
        if tasks is not None:
            # Convert task dicts to WorkflowTask objects
            workflow_tasks = []
            task_id_map = {}
            
            for i, task_dict in enumerate(tasks):
                task_id = f"{workflow_id}-task-{i}"
                task_id_map[i] = task_id
                
                # Handle dependencies
                dependencies = []
                for dep in task_dict.get('dependencies', []):
                    if isinstance(dep, int):
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
            
            workflow.tasks = workflow_tasks
            # Reset status to pending when tasks change
            workflow.status = WorkflowStatus.PENDING.value
            workflow.started_at = None
            workflow.completed_at = None
            workflow.error = None
        
        self.storage.save_workflow(workflow)
        logger.info(f"Updated workflow {workflow_id}")
        return workflow
    
    def delete_workflow(self, workflow_id: str):
        """Delete a workflow"""
        return self.storage.delete_workflow(workflow_id)
    
    def start_workflow(self, workflow_id: str):
        """Start workflow execution"""
        return self.engine.start_workflow(workflow_id)
    
    def pause_workflow(self, workflow_id: str):
        """Pause workflow execution"""
        return self.engine.pause_workflow(workflow_id)
    
    def stop_workflow(self, workflow_id: str):
        """Stop workflow execution"""
        return self.engine.stop_workflow(workflow_id)
    
    @staticmethod
    def create_image_generation_pipeline() -> Dict[str, Any]:
        """
        Template for: Prompt Enhancement -> Image Generation -> Upscaling
        Uses HuggingFace pipeline tags: text-generation, text-to-image, image-to-image
        """
        return {
            'name': 'Image Generation Pipeline',
            'description': 'LLM prompt enhancement → image generation → upscaling',
            'tasks': [
                {
                    'name': 'Enhance Prompt',
                    'type': 'text-generation',  # HuggingFace pipeline tag
                    'config': {
                        'model': 'gpt-4',
                        'inputs': {'prompt': 'a beautiful landscape'},
                        'max_length': 200
                    },
                    'dependencies': [],
                    'output_keys': ['enhanced_prompt', 'text']
                },
                {
                    'name': 'Generate Image',
                    'type': 'text-to-image',  # HuggingFace pipeline tag
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
                    'type': 'image-to-image',  # HuggingFace pipeline tag
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
        Uses HuggingFace pipeline tags: text-generation, text-to-image, image-to-video
        """
        return {
            'name': 'Text-to-Video Pipeline',
            'description': 'Enhanced prompt → image → animated video',
            'tasks': [
                {
                    'name': 'Enhance Prompt',
                    'type': 'text-generation',  # HuggingFace pipeline tag
                    'config': {
                        'model': 'gpt-4',
                        'inputs': {'prompt': 'a serene mountain scene'},
                        'max_length': 150
                    },
                    'dependencies': [],
                    'output_keys': ['enhanced_prompt', 'text']
                },
                {
                    'name': 'Generate Base Image',
                    'type': 'text-to-image',  # HuggingFace pipeline tag
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
                    'name': 'Animate to Video',
                    'type': 'image-to-video',  # HuggingFace pipeline tag
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
        Uses HuggingFace pipeline tags: filter (custom), text-to-image, image-classification
        """
        return {
            'name': 'Safe Image Generation',
            'description': 'NSFW filter → image generation → quality validation',
            'tasks': [
                {
                    'name': 'NSFW Filter',
                    'type': 'filter',  # Custom filter type
                    'config': {
                        'model': 'nsfw-classifier',
                        'filter_type': 'nsfw'
                    },
                    'dependencies': [],
                    'output_keys': ['passed', 'text']
                },
                {
                    'name': 'Generate Image',
                    'type': 'text-to-image',  # HuggingFace pipeline tag
                    'config': {
                        'model': 'stable-diffusion-xl',
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
                    'name': 'Quality Check',
                    'type': 'image-classification',  # HuggingFace pipeline tag
                    'config': {
                        'model': 'image-quality-classifier',
                        'threshold': 0.8
                    },
                    'dependencies': [1],
                    'input_mapping': {
                        'image': '1.image'
                    },
                    'output_keys': ['quality_score', 'approved']
                }
            ]
        }
    
    @staticmethod
    def create_multimodal_pipeline() -> Dict[str, Any]:
        """
        Template for: Text -> Image -> Audio -> Video (Complete Multimodal)
        Uses HuggingFace pipeline tags: text-generation, text-to-image, text-to-speech, image-to-video
        """
        return {
            'name': 'Multimodal Content Pipeline',
            'description': 'Text generation → image → audio narration → video composition',
            'tasks': [
                {
                    'name': 'Generate Script',
                    'type': 'text-generation',  # HuggingFace pipeline tag
                    'config': {
                        'model': 'gpt-4',
                        'inputs': {'prompt': 'Create a short story about nature'},
                        'max_length': 300
                    },
                    'dependencies': [],
                    'output_keys': ['text']
                },
                {
                    'name': 'Generate Scene Image',
                    'type': 'text-to-image',  # HuggingFace pipeline tag
                    'config': {
                        'model': 'stable-diffusion-xl',
                        'width': 1024,
                        'height': 576
                    },
                    'dependencies': [0],
                    'input_mapping': {
                        'prompt': '0.text'
                    },
                    'output_keys': ['image']
                },
                {
                    'name': 'Generate Narration',
                    'type': 'text-to-speech',  # HuggingFace pipeline tag
                    'config': {
                        'model': 'elevenlabs-tts',
                        'voice': 'narrator'
                    },
                    'dependencies': [0],
                    'input_mapping': {
                        'text': '0.text'
                    },
                    'output_keys': ['audio']
                },
                {
                    'name': 'Create Final Video',
                    'type': 'image-to-video',  # HuggingFace pipeline tag
                    'config': {
                        'model': 'animatediff',
                        'duration': 10,
                        'fps': 30
                    },
                    'dependencies': [1, 2],
                    'input_mapping': {
                        'image': '1.image',
                        'audio': '2.audio'
                    },
                    'output_keys': ['video', 'video_url']
                }
            ]
        }
