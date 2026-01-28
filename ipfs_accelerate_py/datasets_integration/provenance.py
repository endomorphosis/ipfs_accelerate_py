"""
Provenance Logger - Track data lineage and event history

This module provides provenance tracking for distributed operations,
maintaining immutable audit trails of data transformations and events.
"""

import json
import time
from typing import Optional, Dict, Any, List
from pathlib import Path
from datetime import datetime


class ProvenanceLogger:
    """
    Logger for tracking data provenance and operation history.
    
    Provides a unified interface for tracking:
    - Model inference operations
    - Data transformations
    - Worker execution logs
    - Pull request and GitHub Copilot interactions
    - Event chains and data lineage
    
    Uses ipfs_datasets_py's ProvenanceTracker when available, falling back
    to local JSON-based logging otherwise.
    
    Attributes:
        enabled (bool): Whether IPFS provenance tracking is active
        provenance_tracker: ProvenanceTracker instance (if available)
        log_dir (Path): Local log directory
    
    Example:
        >>> logger = ProvenanceLogger()
        >>> logger.log_inference("bert-base", {"input": "text", "output": "embeddings"})
        >>> logger.log_transformation("tokenize", {"text": "hello"})
    """
    
    def __init__(self, log_dir: Optional[str] = None):
        """
        Initialize the provenance logger.
        
        Args:
            log_dir: Directory for storing logs (default: ~/.cache/ipfs_accelerate/provenance)
        """
        self.enabled = False
        self.provenance_tracker = None
        
        # Set up log directory
        if log_dir:
            self.log_dir = Path(log_dir)
        else:
            self.log_dir = Path.home() / '.cache' / 'ipfs_accelerate' / 'provenance'
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize local log file
        self.local_log_file = self.log_dir / 'provenance.jsonl'
        
        # Try to initialize provenance tracker
        self._initialize()
    
    def _initialize(self):
        """Initialize ProvenanceTracker if ipfs_datasets_py is available."""
        try:
            from ipfs_datasets_py.data_provenance import ProvenanceTracker
            
            self.provenance_tracker = ProvenanceTracker(
                storage_path=str(self.log_dir / 'ipfs_provenance')
            )
            self.enabled = True
            
        except (ImportError, Exception):
            # IPFS provenance not available - will use local logging
            self.enabled = False
    
    def _write_local_log(self, record: Dict[str, Any]):
        """Write a record to local JSONL log file."""
        with open(self.local_log_file, 'a') as f:
            f.write(json.dumps(record) + '\n')
    
    def log_inference(self, model_name: str, data: Dict[str, Any],
                     metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        Log a model inference operation.
        
        Args:
            model_name: Name of the model used
            data: Inference data (input, output, metrics, etc.)
            metadata: Additional metadata (hardware, duration, etc.)
        
        Returns:
            Optional[str]: CID of provenance record, or None if unavailable
        
        Example:
            >>> logger.log_inference("bert-base-uncased", {
            ...     "input_text": "Hello world",
            ...     "output_embedding": [0.1, 0.2, ...],
            ...     "duration_ms": 150
            ... })
        """
        record = {
            'timestamp': datetime.utcnow().isoformat(),
            'type': 'inference',
            'model': model_name,
            'data': data,
            'metadata': metadata or {}
        }
        
        # Write to local log
        self._write_local_log(record)
        
        # Try to write to IPFS provenance tracker
        if self.enabled and self.provenance_tracker:
            try:
                from ipfs_datasets_py.data_provenance import ProvenanceRecordType
                
                prov_record = self.provenance_tracker.create_record(
                    record_type=ProvenanceRecordType.TRANSFORMATION,
                    operation='model_inference',
                    metadata={
                        'model': model_name,
                        **data,
                        **(metadata or {})
                    }
                )
                return prov_record.cid
            except Exception:
                pass
        
        return None
    
    def log_transformation(self, operation: str, data: Dict[str, Any],
                          input_cid: Optional[str] = None,
                          output_cid: Optional[str] = None) -> Optional[str]:
        """
        Log a data transformation operation.
        
        Args:
            operation: Name of transformation (e.g., "tokenize", "preprocess")
            data: Transformation data and parameters
            input_cid: CID of input data (if available)
            output_cid: CID of output data (if available)
        
        Returns:
            Optional[str]: CID of provenance record, or None if unavailable
        
        Example:
            >>> logger.log_transformation("tokenization", {
            ...     "tokenizer": "bert-tokenizer",
            ...     "max_length": 512
            ... }, input_cid="Qm...", output_cid="Qm...")
        """
        record = {
            'timestamp': datetime.utcnow().isoformat(),
            'type': 'transformation',
            'operation': operation,
            'data': data,
            'input_cid': input_cid,
            'output_cid': output_cid
        }
        
        # Write to local log
        self._write_local_log(record)
        
        # Try to write to IPFS provenance tracker
        if self.enabled and self.provenance_tracker:
            try:
                from ipfs_datasets_py.data_provenance import ProvenanceRecordType
                
                prov_record = self.provenance_tracker.create_record(
                    record_type=ProvenanceRecordType.TRANSFORMATION,
                    operation=operation,
                    metadata=data,
                    parent_cids=[input_cid] if input_cid else None
                )
                return prov_record.cid
            except Exception:
                pass
        
        return None
    
    def log_worker_execution(self, worker_id: str, task_id: str, 
                            data: Dict[str, Any]) -> Optional[str]:
        """
        Log worker execution for distributed operations.
        
        Args:
            worker_id: Unique identifier for the worker
            task_id: Unique identifier for the task
            data: Execution data (status, result, duration, etc.)
        
        Returns:
            Optional[str]: CID of provenance record, or None if unavailable
        
        Example:
            >>> logger.log_worker_execution("worker-001", "task-123", {
            ...     "status": "completed",
            ...     "duration_ms": 5000,
            ...     "result_cid": "Qm..."
            ... })
        """
        record = {
            'timestamp': datetime.utcnow().isoformat(),
            'type': 'worker_execution',
            'worker_id': worker_id,
            'task_id': task_id,
            'data': data
        }
        
        # Write to local log
        self._write_local_log(record)
        
        # Try to write to IPFS provenance tracker
        if self.enabled and self.provenance_tracker:
            try:
                from ipfs_datasets_py.data_provenance import ProvenanceRecordType
                
                prov_record = self.provenance_tracker.create_record(
                    record_type=ProvenanceRecordType.TRANSFORMATION,
                    operation='worker_execution',
                    metadata={
                        'worker_id': worker_id,
                        'task_id': task_id,
                        **data
                    }
                )
                return prov_record.cid
            except Exception:
                pass
        
        return None
    
    def log_pr_activity(self, pr_number: int, activity_type: str,
                       data: Dict[str, Any]) -> Optional[str]:
        """
        Log pull request related activity.
        
        Args:
            pr_number: Pull request number
            activity_type: Type of activity (e.g., "copilot_suggestion", "review")
            data: Activity data
        
        Returns:
            Optional[str]: CID of provenance record, or None if unavailable
        
        Example:
            >>> logger.log_pr_activity(123, "copilot_suggestion", {
            ...     "file": "model.py",
            ...     "suggestion": "Add error handling",
            ...     "accepted": True
            ... })
        """
        record = {
            'timestamp': datetime.utcnow().isoformat(),
            'type': 'pr_activity',
            'pr_number': pr_number,
            'activity_type': activity_type,
            'data': data
        }
        
        # Write to local log
        self._write_local_log(record)
        
        # Try to write to IPFS provenance tracker
        if self.enabled and self.provenance_tracker:
            try:
                from ipfs_datasets_py.data_provenance import ProvenanceRecordType
                
                prov_record = self.provenance_tracker.create_record(
                    record_type=ProvenanceRecordType.EXPORT,
                    operation='pr_activity',
                    metadata={
                        'pr_number': pr_number,
                        'activity_type': activity_type,
                        **data
                    }
                )
                return prov_record.cid
            except Exception:
                pass
        
        return None
    
    def get_lineage(self, cid: str) -> List[Dict[str, Any]]:
        """
        Get the provenance lineage for a CID.
        
        Args:
            cid: Content identifier to trace
        
        Returns:
            List of provenance records in lineage chain
        
        Example:
            >>> lineage = logger.get_lineage("Qm...")
            >>> for record in lineage:
            ...     print(f"{record['operation']}: {record['timestamp']}")
        """
        if self.enabled and self.provenance_tracker and cid:
            try:
                return self.provenance_tracker.get_lineage(cid)
            except Exception:
                pass
        
        return []
    
    def query_logs(self, filters: Optional[Dict[str, Any]] = None,
                  limit: int = 100) -> List[Dict[str, Any]]:
        """
        Query local logs with optional filters.
        
        Note: Returns logs in chronological order (oldest first).
        For most recent logs, use a larger limit and slice from the end.
        
        Args:
            filters: Dictionary of field:value filters
            limit: Maximum number of records to return
        
        Returns:
            List of matching log records in chronological order
        
        Example:
            >>> # Get most recent inference logs
            >>> logs = logger.query_logs({"type": "inference"}, limit=100)
            >>> recent_logs = logs[-10:]  # Get last 10
        """
        records = []
        
        if not self.local_log_file.exists():
            return records
        
        with open(self.local_log_file, 'r') as f:
            for line in f:
                try:
                    record = json.loads(line)
                    
                    # Apply filters
                    if filters:
                        match = all(
                            record.get(key) == value
                            for key, value in filters.items()
                        )
                        if not match:
                            continue
                    
                    records.append(record)
                    
                    if len(records) >= limit:
                        break
                        
                except json.JSONDecodeError:
                    continue
        
        return records
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get status of provenance logger.
        
        Returns:
            Dict with status information
        
        Example:
            >>> status = logger.get_status()
            >>> print(f"IPFS enabled: {status['ipfs_enabled']}")
        """
        log_count = 0
        if self.local_log_file.exists():
            with open(self.local_log_file, 'r') as f:
                log_count = sum(1 for _ in f)
        
        return {
            'ipfs_enabled': self.enabled,
            'log_dir': str(self.log_dir),
            'local_log_count': log_count,
            'provenance_tracker': self.provenance_tracker is not None,
        }
