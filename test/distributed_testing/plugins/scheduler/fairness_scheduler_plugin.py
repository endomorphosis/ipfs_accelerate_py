#!/usr/bin/env python3
"""
Fairness Scheduler Plugin for Distributed Testing Framework

This plugin implements a fairness-based scheduling algorithm that ensures
fair resource allocation across different users, projects, and priorities.
"""

import asyncio
import logging
import time
import random
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set, DefaultDict

from .base_scheduler_plugin import BaseSchedulerPlugin
from .scheduler_plugin_interface import SchedulingStrategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FairnessSchedulerPlugin(BaseSchedulerPlugin):
    """
    Fairness Scheduler Plugin for Distributed Testing Framework.
    
    This scheduler implements fair resource allocation across different users,
    projects, and priorities, ensuring that no single user or project monopolizes
    the testing resources.
    """
    
    def __init__(self):
        """Initialize the fairness scheduler plugin."""
        super().__init__(
            name="FairnessScheduler",
            version="1.0.0",
            description="Fair resource allocation scheduler for distributed testing",
            strategies=[
                SchedulingStrategy.FAIR_SHARE,
                SchedulingStrategy.PRIORITY_BASED,
                SchedulingStrategy.ROUND_ROBIN
            ]
        )
        
        # Track resource allocation by user and project
        self.user_allocation: DefaultDict[str, int] = defaultdict(int)
        self.project_allocation: DefaultDict[str, int] = defaultdict(int)
        
        # User and project quotas
        self.user_quotas: Dict[str, int] = {}
        self.project_quotas: Dict[str, int] = {}
        
        # User and project weights
        self.user_weights: Dict[str, float] = {}
        self.project_weights: Dict[str, float] = {}
        
        # Historical resource usage
        self.historical_usage: Dict[str, Dict[str, Any]] = {}
        
        # Fair share allocations
        self.fair_shares: Dict[str, float] = {}
        
        # Additional configuration
        self.config.update({
            "fairness_window_hours": 24,  # Window for historical usage calculation
            "enable_quotas": True,  # Enable quota enforcement
            "recalculate_interval": 60,  # Interval to recalculate fair shares (seconds)
            "default_user_weight": 1.0,  # Default user weight
            "default_project_weight": 1.0,  # Default project weight
            "max_consecutive_same_user": 3,  # Maximum consecutive tasks from the same user
            "enable_priority_boost": True,  # Enable priority-based boosts
            "priority_boost_factor": 1.5,  # Factor for priority boosts
        })
        
        # Track consecutive allocations
        self.consecutive_allocations: Dict[str, int] = {}
        
        # Last recalculation time
        self.last_recalculation = time.time()
        
        logger.info("FairnessSchedulerPlugin initialized")
    
    async def initialize(self, coordinator: Any, config: Dict[str, Any] = None) -> bool:
        """
        Initialize the scheduler plugin.
        
        Args:
            coordinator: Reference to the coordinator instance
            config: Configuration dictionary for the scheduler
            
        Returns:
            bool: True if initialization succeeded, False otherwise
        """
        # Call base class initialization
        result = await super().initialize(coordinator, config)
        
        if not result:
            return False
        
        # Load user and project weights and quotas if available
        if hasattr(coordinator, 'user_weights'):
            self.user_weights = coordinator.user_weights
        
        if hasattr(coordinator, 'project_weights'):
            self.project_weights = coordinator.project_weights
        
        if hasattr(coordinator, 'user_quotas'):
            self.user_quotas = coordinator.user_quotas
        
        if hasattr(coordinator, 'project_quotas'):
            self.project_quotas = coordinator.project_quotas
        
        # Calculate initial fair shares
        self._calculate_fair_shares()
        
        logger.info("FairnessSchedulerPlugin initialized with coordinator")
        return True
    
    def get_configuration_schema(self) -> Dict[str, Any]:
        """
        Get the configuration schema for this scheduler plugin.
        
        Returns:
            Dict[str, Any]: Configuration schema
        """
        # Get base schema
        schema = super().get_configuration_schema()
        
        # Add fairness-specific options
        schema.update({
            "fairness_window_hours": {
                "type": "integer",
                "default": 24,
                "description": "Time window for historical usage calculation in hours"
            },
            "enable_quotas": {
                "type": "boolean",
                "default": True,
                "description": "Enable quota enforcement for users and projects"
            },
            "recalculate_interval": {
                "type": "integer",
                "default": 60,
                "description": "Interval to recalculate fair shares in seconds"
            },
            "default_user_weight": {
                "type": "number",
                "default": 1.0,
                "description": "Default weight for users without explicit weight"
            },
            "default_project_weight": {
                "type": "number",
                "default": 1.0,
                "description": "Default weight for projects without explicit weight"
            },
            "max_consecutive_same_user": {
                "type": "integer",
                "default": 3,
                "description": "Maximum number of consecutive tasks from the same user"
            },
            "enable_priority_boost": {
                "type": "boolean",
                "default": True,
                "description": "Enable priority-based boosts for scheduling"
            },
            "priority_boost_factor": {
                "type": "number",
                "default": 1.5,
                "description": "Factor for priority-based boosts"
            }
        })
        
        return schema
    
    async def schedule_task(self, task_id: str, task_data: Dict[str, Any],
                           available_workers: Dict[str, Dict[str, Any]],
                           worker_load: Dict[str, int]) -> Optional[str]:
        """
        Schedule a task using fairness-based allocation.
        
        Args:
            task_id: ID of the task to schedule
            task_data: Task data including requirements and metadata
            available_workers: Dictionary of available worker IDs to worker data
            worker_load: Dictionary of worker IDs to current task counts
            
        Returns:
            Optional[str]: Selected worker ID or None if no suitable worker found
        """
        # Store task data
        self.tasks[task_id] = task_data
        self.task_status[task_id] = "pending"
        
        # Update worker load data
        self.worker_load.update(worker_load)
        
        # Update worker data
        for worker_id, worker_data in available_workers.items():
            self.workers[worker_id] = worker_data
            self.worker_status[worker_id] = "active"
        
        # If no available workers, return None
        if not available_workers:
            logger.debug(f"No available workers for task {task_id}")
            return None
        
        # Recalculate fair shares if needed
        current_time = time.time()
        if current_time - self.last_recalculation > self.config["recalculate_interval"]:
            self._calculate_fair_shares()
            self.last_recalculation = current_time
        
        # Get user and project from task data
        user_id = task_data.get("user_id", "unknown")
        project_id = task_data.get("project_id", "unknown")
        
        # Check quotas if enabled
        if self.config["enable_quotas"]:
            # Check user quota
            if user_id in self.user_quotas:
                user_usage = self.user_allocation[user_id]
                if user_usage >= self.user_quotas[user_id]:
                    logger.info(f"User {user_id} exceeded quota {self.user_quotas[user_id]}")
                    return None
            
            # Check project quota
            if project_id in self.project_quotas:
                project_usage = self.project_allocation[project_id]
                if project_usage >= self.project_quotas[project_id]:
                    logger.info(f"Project {project_id} exceeded quota {self.project_quotas[project_id]}")
                    return None
        
        # Check consecutive allocations limit
        if user_id in self.consecutive_allocations:
            if self.consecutive_allocations[user_id] >= self.config["max_consecutive_same_user"]:
                # Skip this user if they've had too many consecutive tasks
                if len(self.user_allocation) > 1:  # Only apply if there are other users
                    logger.info(f"User {user_id} has had {self.consecutive_allocations[user_id]} consecutive tasks, prioritizing other users")
                    # We don't return None here, just continue with scheduling and apply a penalty
        
        worker_id = None
        
        # Schedule based on active strategy
        if self._active_strategy == SchedulingStrategy.FAIR_SHARE:
            # Use fair share scheduling
            worker_id = self._fair_share_scheduling(task_id, task_data, available_workers)
        
        elif self._active_strategy == SchedulingStrategy.PRIORITY_BASED:
            # Use priority-based scheduling
            worker_id = self._priority_based_scheduling(task_id, task_data, available_workers)
        
        elif self._active_strategy == SchedulingStrategy.ROUND_ROBIN:
            # Use round-robin scheduling from base class
            worker_ids = list(available_workers.keys())
            if worker_ids:
                worker_id = worker_ids[hash(task_id) % len(worker_ids)]
        
        # Fallback to round-robin if no worker selected
        if worker_id is None:
            worker_ids = list(available_workers.keys())
            if worker_ids:
                worker_id = worker_ids[hash(task_id) % len(worker_ids)]
        
        # Update metrics and state if a worker was selected
        if worker_id:
            self.metrics["tasks_scheduled"] += 1
            self.metrics["strategy_usage"][self._active_strategy.value] += 1
            
            # Update user and project allocation
            self.user_allocation[user_id] += 1
            self.project_allocation[project_id] += 1
            
            # Update consecutive allocations
            for uid in list(self.consecutive_allocations.keys()):
                if uid != user_id:
                    self.consecutive_allocations[uid] = 0
            
            self.consecutive_allocations[user_id] = self.consecutive_allocations.get(user_id, 0) + 1
            
            # Update task-worker mapping
            self.task_worker[task_id] = worker_id
            
            # Update worker load
            self.worker_load[worker_id] = self.worker_load.get(worker_id, 0) + 1
            
            # Update task status
            self.task_status[task_id] = "assigned"
            
            # Update historical usage
            timestamp = datetime.now().isoformat()
            
            if user_id not in self.historical_usage:
                self.historical_usage[user_id] = {"tasks": []}
            
            self.historical_usage[user_id]["tasks"].append({
                "task_id": task_id,
                "timestamp": timestamp,
                "worker_id": worker_id
            })
            
            # Limit history size
            max_history = self.config["history_window_size"]
            if len(self.historical_usage[user_id]["tasks"]) > max_history:
                self.historical_usage[user_id]["tasks"] = self.historical_usage[user_id]["tasks"][-max_history:]
            
            if self.config["detailed_logging"]:
                logger.info(f"Scheduled task {task_id} to worker {worker_id} using {self._active_strategy.value} strategy")
                logger.info(f"User {user_id} allocation: {self.user_allocation[user_id]}, Project {project_id} allocation: {self.project_allocation[project_id]}")
        
        return worker_id
    
    async def update_task_status(self, task_id: str, status: str,
                              worker_id: Optional[str],
                              execution_time: Optional[float] = None,
                              result: Any = None) -> None:
        """
        Update task status with fairness-specific tracking.
        
        Args:
            task_id: ID of the task
            status: New status of the task
            worker_id: ID of the worker that processed the task
            execution_time: Execution time in seconds
            result: Task result or error information
        """
        # Call base class implementation
        await super().update_task_status(task_id, status, worker_id, execution_time, result)
        
        # Additional fairness-specific tracking
        if task_id in self.tasks:
            task_data = self.tasks[task_id]
            user_id = task_data.get("user_id", "unknown")
            project_id = task_data.get("project_id", "unknown")
            
            # Update historical usage with task completion status
            if user_id in self.historical_usage:
                for task in self.historical_usage[user_id]["tasks"]:
                    if task["task_id"] == task_id:
                        task["status"] = status
                        if execution_time is not None:
                            task["execution_time"] = execution_time
                        break
            
            # Handle task completion or failure
            if status == "completed" or status == "failed":
                # Decrement allocation counts when task is done
                if user_id in self.user_allocation:
                    self.user_allocation[user_id] = max(0, self.user_allocation[user_id] - 1)
                
                if project_id in self.project_allocation:
                    self.project_allocation[project_id] = max(0, self.project_allocation[project_id] - 1)
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get fairness-specific metrics.
        
        Returns:
            Dict[str, Any]: Dictionary of scheduler metrics
        """
        # Get base metrics
        metrics = super().get_metrics()
        
        # Add fairness-specific metrics
        fairness_metrics = {
            "user_allocation": dict(self.user_allocation),
            "project_allocation": dict(self.project_allocation),
            "fair_shares": dict(self.fair_shares),
            "active_users": len([u for u, count in self.user_allocation.items() if count > 0]),
            "active_projects": len([p for p, count in self.project_allocation.items() if count > 0])
        }
        
        # Add fair share deviation metrics
        if self.fair_shares:
            # Calculate fair share deviation for users
            user_deviations = {}
            for user_id, alloc in self.user_allocation.items():
                fair_share = self.fair_shares.get(f"user:{user_id}", 0)
                if fair_share > 0:
                    user_deviations[user_id] = (alloc - fair_share) / fair_share
            
            fairness_metrics["user_fair_share_deviation"] = user_deviations
        
        # Track scheduling fairness score
        if self.user_allocation and len(self.user_allocation) > 1:
            # Gini coefficient for allocation fairness
            values = sorted(self.user_allocation.values())
            n = len(values)
            if n > 0 and sum(values) > 0:
                gini = sum((2 * i - n - 1) * values[i] for i in range(n)) / (n * sum(values))
                fairness_metrics["gini_coefficient"] = abs(gini)
                fairness_metrics["fairness_score"] = 1.0 - abs(gini)
        
        metrics["fairness"] = fairness_metrics
        
        return metrics
    
    def _calculate_fair_shares(self) -> None:
        """
        Calculate fair shares for users and projects.
        
        This method recalculates the fair allocation of resources based on
        user and project weights and current system load.
        """
        total_users = len(self.user_weights) or 1
        total_weight = sum(self.user_weights.values()) or (total_users * self.config["default_user_weight"])
        total_workers = len(self.workers) or 1
        
        # Calculate worker capacity
        total_capacity = total_workers * self.config["max_tasks_per_worker"]
        
        # Calculate fair shares for users
        for user_id, weight in self.user_weights.items():
            # Calculate fair share based on weight proportion
            fair_share = (weight / total_weight) * total_capacity
            self.fair_shares[f"user:{user_id}"] = fair_share
        
        # Calculate fair shares for projects
        total_projects = len(self.project_weights) or 1
        project_weight_sum = sum(self.project_weights.values()) or (total_projects * self.config["default_project_weight"])
        
        for project_id, weight in self.project_weights.items():
            # Calculate fair share based on weight proportion
            fair_share = (weight / project_weight_sum) * total_capacity
            self.fair_shares[f"project:{project_id}"] = fair_share
        
        # Add default fair share for unknown users and projects
        self.fair_shares["user:unknown"] = total_capacity / (total_users + 1)
        self.fair_shares["project:unknown"] = total_capacity / (total_projects + 1)
        
        if self.config["detailed_logging"]:
            logger.info(f"Recalculated fair shares: {self.fair_shares}")
    
    def _fair_share_scheduling(self, task_id: str, task_data: Dict[str, Any],
                              available_workers: Dict[str, Dict[str, Any]]) -> Optional[str]:
        """
        Implement fair share scheduling algorithm.
        
        Args:
            task_id: ID of the task to schedule
            task_data: Task data including user and project
            available_workers: Dictionary of available workers
            
        Returns:
            Optional[str]: Selected worker ID or None if no suitable worker found
        """
        user_id = task_data.get("user_id", "unknown")
        project_id = task_data.get("project_id", "unknown")
        
        # Calculate user and project shares
        user_fair_share = self.fair_shares.get(f"user:{user_id}", 0)
        project_fair_share = self.fair_shares.get(f"project:{project_id}", 0)
        
        current_user_allocation = self.user_allocation.get(user_id, 0)
        current_project_allocation = self.project_allocation.get(project_id, 0)
        
        # Check if user or project is over their fair share
        user_share_ratio = current_user_allocation / user_fair_share if user_fair_share > 0 else float('inf')
        project_share_ratio = current_project_allocation / project_fair_share if project_fair_share > 0 else float('inf')
        
        # If either is over fair share, apply a scheduling penalty based on the excess
        is_over_share = False
        penalty_score = 0.0
        
        if user_share_ratio > 1.0:
            is_over_share = True
            penalty_score += (user_share_ratio - 1.0) * 10.0
        
        if project_share_ratio > 1.0:
            is_over_share = True
            penalty_score += (project_share_ratio - 1.0) * 5.0
        
        # Apply consecutive allocation penalty
        consecutive_count = self.consecutive_allocations.get(user_id, 0)
        if consecutive_count >= self.config["max_consecutive_same_user"]:
            is_over_share = True
            penalty_score += consecutive_count
        
        # If over fair share, decide whether to schedule anyway
        if is_over_share:
            # Get the number of other active users
            other_active_users = sum(1 for u, count in self.user_allocation.items() 
                                   if u != user_id and count > 0)
            
            # If there are other users with tasks, apply the penalty score
            if other_active_users > 0:
                # Probability decreases as penalty score increases
                probability_threshold = max(0.1, 1.0 / (1.0 + penalty_score))
                
                # Random chance to schedule anyway
                if random.random() > probability_threshold:
                    logger.debug(f"User {user_id} over fair share, delaying task {task_id}")
                    return None
        
        # Apply priority boost if enabled
        priority_score = 0.0
        if self.config["enable_priority_boost"]:
            priority = task_data.get("priority", 5)  # Default priority 5
            priority_boost = (priority / 5.0) * self.config["priority_boost_factor"]
            priority_score = priority_boost
        
        # Score workers based on fair share allocation
        worker_scores = []
        
        for worker_id, worker_data in available_workers.items():
            # Base score starts at 10
            score = 10.0
            
            # Subtract worker load
            load = self.worker_load.get(worker_id, 0)
            score -= load * 2.0
            
            # Add priority score
            score += priority_score
            
            # Check hardware match if applicable
            if "hardware_requirements" in task_data and self._match_hardware(task_data, {worker_id: worker_data}):
                score += 5.0
            
            # Track worker score
            worker_scores.append((worker_id, score))
        
        if not worker_scores:
            return None
        
        # Select worker with highest score
        worker_scores.sort(key=lambda x: x[1], reverse=True)
        
        if self.config["detailed_logging"]:
            logger.debug(f"Worker scores for task {task_id}: {worker_scores}")
        
        return worker_scores[0][0]
    
    def _priority_based_scheduling(self, task_id: str, task_data: Dict[str, Any],
                                 available_workers: Dict[str, Dict[str, Any]]) -> Optional[str]:
        """
        Implement priority-based scheduling algorithm.
        
        Args:
            task_id: ID of the task to schedule
            task_data: Task data including priority
            available_workers: Dictionary of available workers
            
        Returns:
            Optional[str]: Selected worker ID or None if no suitable worker found
        """
        # Get task priority
        priority = task_data.get("priority", 5)  # Default priority 5
        
        # Score workers based on priority and load
        worker_scores = []
        
        for worker_id, worker_data in available_workers.items():
            # Base score
            score = 10.0
            
            # Subtract worker load
            load = self.worker_load.get(worker_id, 0)
            score -= load * 2.0
            
            # Add priority score (higher priority = higher score)
            score += (priority - 5) * 2.0  # Priority 1-10, centered at 5
            
            # Check hardware match if applicable
            if "hardware_requirements" in task_data and self._match_hardware(task_data, {worker_id: worker_data}):
                score += 5.0
            
            # Track worker score
            worker_scores.append((worker_id, score))
        
        if not worker_scores:
            return None
        
        # Select worker with highest score
        worker_scores.sort(key=lambda x: x[1], reverse=True)
        
        return worker_scores[0][0]
    
    def _match_hardware(self, task_data: Dict[str, Any],
                       available_workers: Dict[str, Dict[str, Any]]) -> Optional[str]:
        """
        Match task to worker based on hardware requirements.
        
        Args:
            task_data: Task data including requirements
            available_workers: Dictionary of available workers
            
        Returns:
            Optional[str]: Selected worker ID or None if no match found
        """
        # Skip if no hardware requirements
        if "hardware_requirements" not in task_data:
            return None
            
        hardware_reqs = task_data["hardware_requirements"]
        
        # Match workers to requirements
        matching_workers = []
        
        for worker_id, worker_data in available_workers.items():
            capabilities = worker_data.get("capabilities", {})
            
            # Check if worker matches all requirements
            matches_all = True
            
            for req_name, req_value in hardware_reqs.items():
                if req_name not in capabilities:
                    matches_all = False
                    break
                
                capability_value = capabilities[req_name]
                
                # Check based on type
                if isinstance(req_value, bool):
                    if req_value != capability_value:
                        matches_all = False
                        break
                elif isinstance(req_value, (int, float)):
                    if capability_value < req_value:
                        matches_all = False
                        break
                elif isinstance(req_value, str):
                    if capability_value != req_value:
                        matches_all = False
                        break
                elif isinstance(req_value, list):
                    if capability_value not in req_value:
                        matches_all = False
                        break
            
            if matches_all:
                matching_workers.append(worker_id)
        
        if not matching_workers:
            return None
            
        # Return the first matching worker for basic implementation
        # A more sophisticated implementation could score each worker
        return matching_workers[0]