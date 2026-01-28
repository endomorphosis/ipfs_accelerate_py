"""
Datasets Manager - Main integration point for ipfs_datasets_py

This module provides a high-level interface for managing datasets, models,
and distributed operations using ipfs_datasets_py services.
"""

import os
from typing import Optional, Dict, Any, List
from pathlib import Path


class DatasetsManager:
    """
    High-level manager for ipfs_datasets_py integration.
    
    Provides unified interface for:
    - Dataset loading and management
    - Model storage and retrieval via IPFS
    - Event and provenance logging
    - P2P workflow coordination
    
    Attributes:
        enabled (bool): Whether datasets integration is active
        dataset_manager: ipfs_datasets_py DatasetManager (if available)
        audit_logger: AuditLogger instance (if available)
        provenance_tracker: ProvenanceTracker instance (if available)
        workflow_scheduler: P2PWorkflowScheduler instance (if available)
    
    Example:
        >>> manager = DatasetsManager()
        >>> if manager.enabled:
        ...     manager.log_event("model_loaded", {"model": "bert"})
        ...     manager.track_provenance("inference", {"input": "text"})
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the datasets manager.
        
        Args:
            config: Optional configuration dictionary
                - cache_dir: Directory for caching (default: ~/.cache/ipfs_accelerate)
                - enable_audit: Enable audit logging (default: True)
                - enable_provenance: Enable provenance tracking (default: True)
                - enable_p2p: Enable P2P workflow scheduling (default: False)
        """
        self.config = config or {}
        self.enabled = False
        self.dataset_manager = None
        self.audit_logger = None
        self.provenance_tracker = None
        self.workflow_scheduler = None
        
        # Set up cache directory
        cache_dir = self.config.get('cache_dir')
        if not cache_dir:
            cache_dir = os.path.expanduser('~/.cache/ipfs_accelerate')
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Try to initialize ipfs_datasets_py components
        self._initialize()
    
    def _initialize(self):
        """Initialize ipfs_datasets_py components if available."""
        try:
            from ipfs_datasets_py import DatasetManager
            from ipfs_datasets_py.audit import AuditLogger
            from ipfs_datasets_py.data_provenance import ProvenanceTracker
            
            # Initialize dataset manager
            self.dataset_manager = DatasetManager(
                cache_dir=str(self.cache_dir)
            )
            
            # Initialize audit logger if enabled
            if self.config.get('enable_audit', True):
                log_dir = self.cache_dir / 'audit_logs'
                log_dir.mkdir(exist_ok=True)
                self.audit_logger = AuditLogger(
                    log_dir=str(log_dir),
                    console_output=False
                )
            
            # Initialize provenance tracker if enabled
            if self.config.get('enable_provenance', True):
                provenance_dir = self.cache_dir / 'provenance'
                provenance_dir.mkdir(exist_ok=True)
                self.provenance_tracker = ProvenanceTracker(
                    storage_path=str(provenance_dir)
                )
            
            # Initialize P2P workflow scheduler if enabled
            if self.config.get('enable_p2p', False):
                try:
                    from ipfs_datasets_py.p2p_workflow_scheduler import get_scheduler
                    self.workflow_scheduler = get_scheduler()
                except ImportError:
                    # P2P features may not be available
                    pass
            
            self.enabled = True
            
        except ImportError:
            # ipfs_datasets_py not available - use fallbacks
            self.enabled = False
    
    def log_event(self, event_type: str, data: Dict[str, Any], 
                  level: str = 'INFO', category: str = 'GENERAL') -> bool:
        """
        Log an event using the audit logger.
        
        Args:
            event_type: Type of event (e.g., "model_loaded", "inference_started")
            data: Event data dictionary
            level: Log level (INFO, WARNING, ERROR, CRITICAL)
            category: Event category (GENERAL, SECURITY, PERFORMANCE, etc.)
        
        Returns:
            bool: True if logged successfully, False if logging unavailable
        
        Example:
            >>> manager.log_event("inference_completed", {
            ...     "model": "bert-base",
            ...     "duration_ms": 150,
            ...     "success": True
            ... })
        """
        if not self.enabled or not self.audit_logger:
            return False
        
        try:
            from ipfs_datasets_py.audit import AuditLevel, AuditCategory
            
            # Convert string level to enum
            level_enum = getattr(AuditLevel, level.upper(), AuditLevel.INFO)
            category_enum = getattr(AuditCategory, category.upper(), AuditCategory.GENERAL)
            
            self.audit_logger.log_event(
                event_type=event_type,
                data=data,
                level=level_enum,
                category=category_enum
            )
            return True
        except Exception:
            return False
    
    def track_provenance(self, operation: str, data: Dict[str, Any],
                        record_type: str = 'TRANSFORMATION') -> Optional[str]:
        """
        Track data provenance for an operation.
        
        Args:
            operation: Operation name (e.g., "inference", "preprocessing")
            data: Operation data and metadata
            record_type: Type of provenance record (SOURCE, TRANSFORMATION, etc.)
        
        Returns:
            Optional[str]: CID of provenance record, or None if unavailable
        
        Example:
            >>> cid = manager.track_provenance("model_inference", {
            ...     "model": "bert-base",
            ...     "input_cid": "Qm...",
            ...     "output_cid": "Qm..."
            ... })
        """
        if not self.enabled or not self.provenance_tracker:
            return None
        
        try:
            from ipfs_datasets_py.data_provenance import ProvenanceRecordType
            
            # Convert string type to enum
            type_enum = getattr(ProvenanceRecordType, record_type.upper(), 
                              ProvenanceRecordType.TRANSFORMATION)
            
            record = self.provenance_tracker.create_record(
                record_type=type_enum,
                operation=operation,
                metadata=data
            )
            return record.cid
        except Exception:
            return None
    
    def submit_workflow(self, task_id: str, task_type: str, 
                       data: Dict[str, Any], priority: int = 5) -> bool:
        """
        Submit a workflow task to the P2P scheduler.
        
        Args:
            task_id: Unique task identifier
            task_type: Type of task (e.g., "inference", "preprocessing")
            data: Task data and parameters
            priority: Task priority (1-10, higher = more important)
        
        Returns:
            bool: True if submitted successfully, False if P2P unavailable
        
        Example:
            >>> manager.submit_workflow(
            ...     task_id="infer-001",
            ...     task_type="inference",
            ...     data={"model": "bert", "input": "text"},
            ...     priority=7
            ... )
        """
        if not self.enabled or not self.workflow_scheduler:
            return False
        
        try:
            from ipfs_datasets_py.p2p_workflow_scheduler import WorkflowDefinition
            
            workflow = WorkflowDefinition(
                id=task_id,
                task_type=task_type,
                data=data,
                priority=priority
            )
            
            self.workflow_scheduler.submit_workflow(workflow)
            return True
        except Exception:
            return False
    
    def load_dataset(self, dataset_name: str, split: Optional[str] = None,
                    streaming: bool = False) -> Optional[Any]:
        """
        Load a dataset using the dataset manager.
        
        Args:
            dataset_name: Name of dataset (e.g., "squad", "glue")
            split: Dataset split (e.g., "train", "test")
            streaming: Whether to use streaming mode
        
        Returns:
            Dataset object or None if unavailable
        
        Example:
            >>> dataset = manager.load_dataset("squad", split="train")
        """
        if not self.enabled or not self.dataset_manager:
            return None
        
        try:
            return self.dataset_manager.load_dataset(
                dataset_name,
                split=split,
                streaming=streaming
            )
        except Exception:
            return None
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get status of all integrated services.
        
        Returns:
            Dict with status of each component
        
        Example:
            >>> status = manager.get_status()
            >>> print(f"Audit logging: {status['audit_logger']}")
        """
        return {
            'enabled': self.enabled,
            'dataset_manager': self.dataset_manager is not None,
            'audit_logger': self.audit_logger is not None,
            'provenance_tracker': self.provenance_tracker is not None,
            'workflow_scheduler': self.workflow_scheduler is not None,
            'cache_dir': str(self.cache_dir),
        }
